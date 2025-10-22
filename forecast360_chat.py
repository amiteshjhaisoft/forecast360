# forecast360_agent.py
# Forecast360 — Weaviate-backed AI Agent with Azure Blob → Weaviate pre-ingestion
# Author: Amitesh Jha | iSoft

from __future__ import annotations
import os, io, json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import urllib.parse as _urlparse
import streamlit as st
import pandas as pd

# ---------- Weaviate (client v4; Agents optional) ----------
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

# Agents are optional; guard import so lack of weaviate-agents won't crash the app
QueryAgentImportError = None
try:
    from weaviate.agents.query import QueryAgent
    from weaviate_agents.classes import QueryAgentCollectionConfig
except Exception as e:
    QueryAgent = None  # type: ignore
    QueryAgentCollectionConfig = None  # type: ignore
    QueryAgentImportError = e

# ---------- Azure Blob ----------
from azure.storage.blob import BlobServiceClient

# ---------- Optional extractors (best-effort) ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    import docx2txt
except Exception:
    docx2txt = None
try:
    from pptx import Presentation
except Exception:
    Presentation = None


# =========================
# Settings & helpers
# =========================

DEFAULT_COLLECTION = "Forecast360"
_CLOUD_HOST_MARKERS = ("weaviate.network", "weaviate.cloud", "semi.network", "gcp.weaviate.cloud")

@dataclass
class WeaviateSettings:
    url: str
    api_key: Optional[str]
    collection: str
    top_k: int
    alpha: float
    timeout_s: int

@dataclass
class AzureSettings:
    connection_string: Optional[str]
    account_url: Optional[str]
    account_key: Optional[str]
    container: str
    prefix: str


def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """Get value from st.secrets (nested) or env. Example: 'weaviate.url' or 'AZURE_CONNECTION_STRING'."""
    if "." in path:
        g, k = path.split(".", 1)
        try:
            return st.secrets[g][k]
        except Exception:
            pass
    try:
        return st.secrets[path]
    except Exception:
        pass
    return os.environ.get(path.replace(".", "_").upper(), default)


def resolve_weaviate_settings() -> WeaviateSettings:
    url = _get_secret("weaviate.url") or _get_secret("WEAVIATE_URL") or "http://localhost"
    api_key = _get_secret("weaviate.api_key") or _get_secret("WEAVIATE_API_KEY")
    collection = _get_secret("weaviate.collection", DEFAULT_COLLECTION) or DEFAULT_COLLECTION
    top_k = int(_get_secret("weaviate.top_k", "5"))
    alpha = float(_get_secret("weaviate.alpha", "0.5"))
    timeout_s = int(_get_secret("weaviate.timeout_s", "60"))
    return WeaviateSettings(url=url, api_key=api_key, collection=collection, top_k=top_k, alpha=alpha, timeout_s=timeout_s)


def resolve_azure_settings() -> AzureSettings:
    conn = _get_secret("azure.connection_string") or _get_secret("AZURE_CONNECTION_STRING")
    acct_url = _get_secret("azure.account_url") or _get_secret("AZURE_ACCOUNT_URL")
    acct_key = _get_secret("azure.account_key") or _get_secret("AZURE_ACCOUNT_KEY")
    container = _get_secret("azure.container", "knowledgebase") or "knowledgebase"
    prefix = _get_secret("azure.prefix", "KB/") or "KB/"
    return AzureSettings(
        connection_string=conn or None,
        account_url=acct_url or None,
        account_key=acct_key or None,
        container=container,
        prefix=prefix,
    )


# ---------- URL normalization & safe netloc split ----------

def _split_netloc_safe(netloc: str) -> Tuple[Optional[str], str, Optional[str]]:
    netloc = (netloc or "").strip()
    auth = None
    hostport = netloc
    if "@" in netloc:
        auth, hostport = netloc.rsplit("@", 1)
        auth = auth.strip(); hostport = hostport.strip()
    if hostport.startswith("["):
        end = hostport.find("]")
        if end != -1:
            host = hostport[: end + 1]
            remainder = hostport[end + 1 :].strip()
            m = re.search(r":(\d+)$", remainder)
            port_str = m.group(1) if m else None
            return auth, host, port_str
    m = re.search(r":(\d+)$", hostport)
    if m:
        host = hostport[: m.start()].strip()
        port_str = m.group(1)
    else:
        host = hostport.strip()
        port_str = None
    return auth, host, port_str


def _normalize_weaviate_url(raw: str) -> str:
    """Add scheme if missing; strip default ports (:443 https, :80 http); preserve non-default."""
    raw = (raw or "").strip()
    if not raw:
        return "http://localhost"
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        scheme = "https" if any(m in raw for m in _CLOUD_HOST_MARKERS) else "http"
        raw = f"{scheme}://{raw}"
    parsed = _urlparse.urlsplit(raw)
    scheme = parsed.scheme.lower()
    auth, host, port_str = _split_netloc_safe(parsed.netloc)
    if port_str and ((scheme == "https" and port_str == "443") or (scheme == "http" and port_str == "80")):
        port_str = None
    netloc = host if not port_str else f"{host}:{port_str}"
    if auth:
        netloc = f"{auth}@{netloc}"
    return _urlparse.urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


# ---------- Weaviate connector (strong verification; no diagnostics UI) ----------

@st.cache_resource(show_spinner=False)
def connect_weaviate(url: str, api_key: Optional[str], timeout_s: int = 60):
    """
    - Normalizes URL
    - Uses cloud helper for hosted clusters, local helper otherwise
    - No unsupported kwargs
    - Verifies with a data-plane call; raises helpful error
    """
    url = _normalize_weaviate_url(url)

    def _verify_or_raise(c):
        try:
            _ = c.collections.list_all()
        except Exception as e:
            kind = type(e).__name__
            msg = getattr(e, "message", None) or str(e)
            raise RuntimeError(
                f"Weaviate verification failed ({kind}): {msg}\n"
                f"Resolved URL: {url}\n"
                f"Auth provided: {'yes' if api_key else 'no'}"
            )

    if any(marker in url for marker in _CLOUD_HOST_MARKERS):
        c = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key) if api_key else None,
        )
        _verify_or_raise(c)
        return c

    parsed = _urlparse.urlsplit(url)
    host = (parsed.hostname or "localhost").strip()
    _, _, port_str = _split_netloc_safe(parsed.netloc)
    port = int(port_str) if port_str else 8080
    c = weaviate.connect_to_local(http_host=host, http_port=port)
    _verify_or_raise(c)
    return c


def ensure_collection(client, name: str) -> str:
    """
    Ensure the collection exists. Prefer server vectorizer; if unavailable,
    fall back to 'none' so BM25-only search still works.
    Returns the vectorizer mode: 'transformers' or 'none'.
    """
    # Does it already exist?
    try:
        exists = any(c.name == name for c in client.collections.list_all().collections)
    except Exception:
        try:
            client.collections.get(name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        # Try with server-side vectorizer first
        try:
            client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="block_id", data_type=DataType.TEXT),
                ],
            )
            return "transformers"
        except Exception:
            # Fall back: no vectorizer
            client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="block_id", data_type=DataType.TEXT),
                ],
            )
            return "none"
    else:
        # Try to infer vectorizer by probing hybrid; if it fails, assume 'none'
        try:
            _ = client.collections.get(name)
            # quick probe via bm25 (safe) — if works, return 'unknown'; we'll handle at query-time
            return "unknown"
        except Exception:
            return "unknown"


def _azure_blob_service(az: AzureSettings) -> BlobServiceClient:
    if az.connection_string:
        return BlobServiceClient.from_connection_string(az.connection_string)
    if az.account_url and az.account_key:
        return BlobServiceClient(account_url=az.account_url, credential=az.account_key)
    raise RuntimeError("Provide Azure Blob 'connection_string' or 'account_url' + 'account_key' via secrets/env.")


# =========================
# Extraction, chunking, upsert
# =========================

def extract_text_from_blob(name: str, content: bytes) -> List[Tuple[str, Dict[str, Optional[str]]]]:
    name_l = name.lower()
    meta_base: Dict[str, Optional[str]] = {"source": name, "path": name}

    def _as_utf8(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode("latin-1", "ignore")

    items: List[Tuple[str, Dict[str, Optional[str]]]] = []

    if name_l.endswith((".txt", ".md", ".log", ".csv")):
        items.append((_as_utf8(content), {**meta_base}))
    elif name_l.endswith(".json"):
        try:
            obj = json.loads(_as_utf8(content))
            items.append((json.dumps(obj, indent=2, ensure_ascii=False), {**meta_base}))
        except Exception:
            items.append((_as_utf8(content), {**meta_base}))
    elif name_l.endswith(".pdf") and PdfReader:
        try:
            reader = PdfReader(io.BytesIO(content))
            for i, page in enumerate(reader.pages, 1):
                txt = page.extract_text() or ""
                if txt.strip():
                    items.append((txt, {**meta_base, "page": str(i)}))
        except Exception:
            pass
    elif name_l.endswith(".docx") and docx2txt:
        try:
            txt = docx2txt.process(io.BytesIO(content))
            if txt and txt.strip():
                items.append((txt, {**meta_base}))
        except Exception:
            pass
    elif name_l.endswith(".pptx") and Presentation:
        try:
            prs = Presentation(io.BytesIO(content))
            page = 0
            for slide in prs.slides:
                page += 1
                texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        texts.append(shape.text)
                slide_txt = "\n".join(texts).strip()
                if slide_txt:
                    items.append((slide_txt, {**meta_base, "page": str(page)}))
        except Exception:
            pass
    elif name_l.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(content))
            items.append((df.to_csv(index=False), {**meta_base}))
        except Exception:
            pass
    else:
        try:
            items.append((_as_utf8(content), {**meta_base}))
        except Exception:
            pass

    return items


def chunk_text(text: str, max_len: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    parts: List[str] = []
    i = 0
    while i < len(text):
        parts.append(text[i:i + max_len])
        i += max_len - overlap
    return parts


def upsert_chunks(coll, records: List[Dict[str, Optional[str]]]):
    try:
        coll.data.insert_many(records)
    except Exception:
        for r in records:
            try:
                coll.data.insert(r)
            except Exception:
                pass


def ingest_azure_prefix_to_weaviate(client, az: AzureSettings, collection_name: str) -> Dict[str, int]:
    ensure_collection(client, collection_name)
    coll = client.collections.get(collection_name)

    svc = _azure_blob_service(az)
    container = svc.get_container_client(az.container)

    total_blobs = total_chunks = success_files = failed_files = 0
    prog = st.progress(0.0)

    for i, blob in enumerate(container.list_blobs(name_starts_with=az.prefix), 1):
        total_blobs += 1
        prog.progress(min(0.99, i / (i + 1)))

        try:
            data = container.download_blob(blob.name).readall()
            pairs = extract_text_from_blob(blob.name, data)
            to_insert: List[Dict[str, Optional[str]]] = []
            file_chunks = 0

            for txt, meta in pairs:
                for j, ctext in enumerate(chunk_text(txt, 1000, 200)):
                    to_insert.append({
                        "text": ctext,
                        "title": os.path.basename(blob.name),
                        "source": meta.get("source"),
                        "path": meta.get("path"),
                        "page": int(meta["page"]) if meta.get("page") else None,
                        "block_id": f"{blob.name}#{j}",
                    })
                    file_chunks += 1

            if to_insert:
                upsert_chunks(coll, to_insert)
                total_chunks += file_chunks

            success_files += 1
        except Exception as e:
            failed_files += 1
            st.write(f"⚠️ Failed: {blob.name} — {e}")

    prog.progress(1.0)
    return {"files_seen": total_blobs, "files_ok": success_files, "files_failed": failed_files, "chunks_upserted": total_chunks}


# =========================
# Agent & search (vector → hybrid → BM25)
# =========================

def build_agent(client, collection_name: str):
    """Return a QueryAgent if 'weaviate-agents' is installed; else None."""
    if QueryAgent is None or QueryAgentCollectionConfig is None:
        return None
    try:
        return QueryAgent(client=client, collections=[QueryAgentCollectionConfig(name=collection_name)])
    except Exception:
        return None  # if agent wiring requires vectorizer, gracefully skip


def query_snippets(client, collection_name: str, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Optional[str]]]:
    """
    Try hybrid; if the server/collection has no vectorizer, automatically fall back to BM25.
    """
    coll = client.collections.get(collection_name)

    # Attempt hybrid first
    try:
        res = coll.query.hybrid(
            query=query, limit=top_k, alpha=alpha,
            return_metadata=["score", "distance"],
            return_properties=["text", "title", "source", "path", "page", "block_id"],
        )
    except Exception as e:
        msg = str(e).lower()
        if "without vectorizer" in msg or "vectorfrominput" in msg or "no vectorizer" in msg:
            # Fall back to BM25-only
            res = coll.query.bm25(
                query=query, limit=top_k,
                return_properties=["text", "title", "source", "path", "page", "block_id"],
            )
            objects = getattr(res, "objects", []) or []
        else:
            raise

    objects = getattr(res, "objects", []) if 'res' in locals() else []
    items: List[Dict[str, Optional[str]]] = []
    for o in (objects or []):
        props = getattr(o, "properties", {}) or {}
        meta = getattr(o, "metadata", {}) or {}
        score = meta.get("score")
        items.append({
            "text": props.get("text") or "",
            "title": props.get("title") or "",
            "source": props.get("source") or props.get("path") or "",
            "page": str(props.get("page") or ""),
            "block_id": props.get("block_id") or "",
            "score": float(score) if isinstance(score, (int, float)) else score,
            "distance": meta.get("distance"),
        })
    return items


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Forecast360 Agent (Weaviate)", page_icon="🤖", layout="wide")
st.title("🤖 Forecast360 — Knowledge Agent")
st.caption("Ingest from Azure Blob → Weaviate ‘Forecast360’, then chat with your KB.")
st.divider()

# Read settings only from secrets/env
wset = resolve_weaviate_settings()
aset = resolve_azure_settings()

# Sidebar: just actions (no secrets/diagnostics shown)
with st.sidebar:
    st.subheader("Actions")
    connect_clicked = st.button("🔌 Connect / Refresh Weaviate", type="primary", use_container_width=True)
    ingest_clicked = st.button("⬆️ Ingest from Azure → Weaviate", use_container_width=True)

# Connect to Weaviate
if "client" not in st.session_state or connect_clicked:
    try:
        st.session_state.client = connect_weaviate(wset.url, wset.api_key, timeout_s=wset.timeout_s)
        st.success("Connected to Weaviate.")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.stop()

# Ensure collection exists (and capture vectorizer mode for info)
try:
    vectorizer_mode = ensure_collection(st.session_state.client, wset.collection)
except Exception as e:
    st.error(f"Failed to ensure collection '{wset.collection}': {e}")
    st.stop()

# Ingest pre-step
if ingest_clicked:
    with st.spinner("Ingesting from Azure Blob…"):
        try:
            summary = ingest_azure_prefix_to_weaviate(st.session_state.client, aset, wset.collection)
            st.success("Ingestion complete.")
            st.json(summary)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

# Prepare agent (may be None when vectorizer is off / agents missing)
if "agent" not in st.session_state or connect_clicked:
    st.session_state.agent = build_agent(st.session_state.client, wset.collection)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Click **Ingest** to load Azure KB, then ask about your Forecast360 knowledge base."}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input(f"Ask a question about the {wset.collection} KB…")

def _render_snippets(snips: List[Dict[str, Optional[str]]]):
    if not snips:
        return
    st.markdown("#### 🔎 Supporting snippets")
    for i, s in enumerate(snips, 1):
        score = s.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        with st.expander(f"Match {i} — score={score_str} | {s.get('source')}", expanded=(i == 1)):
            if s.get("title"):
                st.markdown(f"**{s['title']}**")
            if s.get("page"):
                st.markdown(f"_page_: {s['page']}")
            if s.get("block_id"):
                st.markdown(f"_block_: `{s['block_id']}`")
            st.write(s.get("text") or "")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer_text = None

                # Try QueryAgent first (if available); if it fails due to vectorizer, fall back
                if st.session_state.agent is not None:
                    try:
                        result = st.session_state.agent.run(prompt)
                        answer_text = getattr(result, "content", None) or str(result)
                    except Exception as e:
                        if any(s in str(e).lower() for s in ["without vectorizer", "vectorfrominput", "no vectorizer"]):
                            pass  # will fall back below
                        else:
                            raise

                if not answer_text:
                    # Hybrid → BM25 fallback path
                    hits = query_snippets(st.session_state.client, wset.collection, prompt, top_k=wset.top_k, alpha=wset.alpha)
                    if hits:
                        joined = "\n\n".join(s["text"] for s in hits[:3] if s.get("text"))
                        answer_text = f"Here are the most relevant excerpts I found:\n\n{joined}"
                    else:
                        answer_text = "I couldn't find relevant excerpts in the knowledge base."

                st.markdown(answer_text)

            # Show supporting matches from whatever worked
            _render_snippets(query_snippets(st.session_state.client, wset.collection, prompt, top_k=wset.top_k, alpha=wset.alpha))

        st.session_state.messages.append({"role": "assistant", "content": answer_text})

    except Exception as e:
        with st.chat_message("assistant"):
            if QueryAgent is None and QueryAgentImportError is not None:
                st.error(
                    "Weaviate Agents not available. Install with:\n\n"
                    "`pip install -U weaviate-client weaviate-agents`\n\n"
                    f"Details: {QueryAgentImportError}"
                )
            else:
                st.error(f"Agent error: {e}")
