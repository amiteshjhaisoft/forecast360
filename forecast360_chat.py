# forecast360_agent.py
# Forecast360 — Weaviate-backed AI Agent with Azure Blob → Weaviate pre-ingestion
# Author: Amitesh Jha | iSoft

from __future__ import annotations
import os, io, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple

import streamlit as st

# ---------- Weaviate (client v4 + optional Agents) ----------
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

# Weaviate Agents are optional; guard the import so the app won't crash if missing
QueryAgentImportError = None
try:
    from weaviate.agents.query import QueryAgent
    from weaviate_agents.classes import QueryAgentCollectionConfig
except Exception as e:  # WeaviateAgentsNotInstalledError or version mismatch
    QueryAgent = None  # type: ignore
    QueryAgentCollectionConfig = None  # type: ignore
    QueryAgentImportError = e

# ---------- Azure Blob ----------
from azure.storage.blob import BlobServiceClient

# ---------- Lightweight parsers (optional heavy ones are gated) ----------
import pandas as pd

try:
    from pypdf import PdfReader  # optional
except Exception:
    PdfReader = None

try:
    import docx2txt  # optional
except Exception:
    docx2txt = None

try:
    from pptx import Presentation  # optional
except Exception:
    Presentation = None


# =========================
# Settings / helpers
# =========================

DEFAULT_COLLECTION = "Forecast360"

@dataclass
class WeaviateSettings:
    url: str
    api_key: Optional[str] = None
    collection: str = DEFAULT_COLLECTION
    top_k: int = 5
    alpha: float = 0.5
    timeout_s: int = 60


@dataclass
class AzureSettings:
    connection_string: Optional[str] = None
    account_url: Optional[str] = None          # e.g. https://<acct>.blob.core.windows.net
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    container: str = "knowledgebase"
    prefix: str = "KB/"


def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch value from Streamlit secrets (nested) or environment.
    Examples: 'weaviate.url', 'azure.connection_string', 'WEAVIATE_URL' (env).
    """
    # try st.secrets deeply
    if "." in path:
        group, key = path.split(".", 1)
        try:
            return st.secrets[group][key]
        except Exception:
            pass
    else:
        try:
            return st.secrets[path]
        except Exception:
            pass
    # env fallback (turn dots into underscores)
    return os.environ.get(path.replace(".", "_").upper(), default)


def resolve_weaviate_settings() -> WeaviateSettings:
    url = (
        _get_secret("weaviate.url")
        or _get_secret("WEAVIATE_URL")
        or "http://localhost:8080"
    )
    api_key = _get_secret("weaviate.api_key") or _get_secret("WEAVIATE_API_KEY") or None
    collection = _get_secret("weaviate.collection", DEFAULT_COLLECTION) or DEFAULT_COLLECTION
    # UI-independent tuning knobs (can be kept fixed or read from secrets too)
    top_k = int(_get_secret("weaviate.top_k", "5"))
    alpha = float(_get_secret("weaviate.alpha", "0.5"))
    timeout_s = int(_get_secret("weaviate.timeout_s", "60"))
    return WeaviateSettings(url=url, api_key=api_key, collection=collection, top_k=top_k, alpha=alpha, timeout_s=timeout_s)


def resolve_azure_settings() -> AzureSettings:
    # Prefer connection string; otherwise allow account_url + account_key or account_name + account_key (unused here)
    conn = _get_secret("azure.connection_string") or _get_secret("AZURE_CONNECTION_STRING")
    acct_url = _get_secret("azure.account_url") or _get_secret("AZURE_ACCOUNT_URL")
    acct_name = _get_secret("azure.account_name") or _get_secret("AZURE_ACCOUNT_NAME")
    acct_key = _get_secret("azure.account_key") or _get_secret("AZURE_ACCOUNT_KEY")
    container = _get_secret("azure.container", "knowledgebase") or "knowledgebase"
    prefix = _get_secret("azure.prefix", "KB/") or "KB/"
    return AzureSettings(
        connection_string=conn or None,
        account_url=acct_url or None,
        account_name=acct_name or None,
        account_key=acct_key or None,
        container=container,
        prefix=prefix,
    )


@st.cache_resource(show_spinner=False)
def connect_weaviate(url: str, api_key: Optional[str], timeout_s: int = 60):
    """Connect to Weaviate (cloud or local). Cached across reruns."""
    if url.startswith("http"):
        if any(k in url for k in ("weaviate.network", "weaviate.cloud", "semi.network")):
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key) if api_key else None,
                timeout=timeout_s,
            )
        else:
            host = url.replace("http://", "").replace("https://", "").split("/")[0]
            parts = host.split(":")
            http_host = parts[0]
            http_port = int(parts[1]) if len(parts) > 1 else 8080
            return weaviate.connect_to_local(http_host=http_host, http_port=http_port, grpc_port=None, timeout=timeout_s)
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key) if api_key else None,
        timeout=timeout_s,
    )


def ensure_collection(client, name: str):
    """Ensure a collection exists with reasonable KB properties and server-side vectorizer."""
    try:
        exists = any(c.name == name for c in client.collections.list_all().collections)
    except Exception:
        try:
            client.collections.get(name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),  # Make sure your Weaviate has this module
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="block_id", data_type=DataType.TEXT),
            ],
        )


def _azure_blob_client(az: AzureSettings) -> BlobServiceClient:
    if az.connection_string:
        return BlobServiceClient.from_connection_string(az.connection_string)
    if az.account_url and az.account_key:
        return BlobServiceClient(account_url=az.account_url, credential=az.account_key)
    raise RuntimeError("Provide Azure Blob connection_string or account_url + account_key via secrets/env.")


# =========================
# Text extraction & chunker
# =========================

def extract_text_from_blob(name: str, content: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (text, meta) pairs extracted from a blob file."""
    name_l = name.lower()
    meta_base = {"source": name, "path": name}

    def _as_utf8(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode("latin-1", "ignore")

    items: List[Tuple[str, Dict[str, Any]]] = []

    if name_l.endswith((".txt", ".md", ".log")):
        items.append((_as_utf8(content), {**meta_base}))
    elif name_l.endswith(".csv"):
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
                    items.append((txt, {**meta_base, "page": i}))
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
                text_runs = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text_runs.append(shape.text)
                slide_txt = "\n".join(text_runs).strip()
                if slide_txt:
                    items.append((slide_txt, {**meta_base, "page": page}))
        except Exception:
            pass
    elif name_l.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(content))
            items.append((df.to_csv(index=False), {**meta_base}))
        except Exception:
            pass
    else:
        # Fallback: try decode as text
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


def upsert_chunks(coll, records: List[Dict[str, Any]]):
    try:
        coll.data.insert_many(records)
    except Exception:
        for r in records:
            try:
                coll.data.insert(r)
            except Exception:
                pass


def ingest_azure_prefix_to_weaviate(client, az: AzureSettings, collection_name: str) -> Dict[str, int]:
    """Stream blobs under container/prefix, extract text, chunk, and upsert into Weaviate."""
    ensure_collection(client, collection_name)
    coll = client.collections.get(collection_name)

    svc = _azure_blob_client(az)
    container = svc.get_container_client(az.container)

    total_blobs = 0
    total_chunks = 0
    success_files = 0
    failed_files = 0

    blob_iter = container.list_blobs(name_starts_with=az.prefix)
    prog = st.progress(0.0)
    for i, blob in enumerate(blob_iter, 1):
        total_blobs += 1
        prog.progress(min(0.99, i / max(total_blobs, i + 1)))

        try:
            data = container.download_blob(blob.name).readall()
            pairs = extract_text_from_blob(blob.name, data)
            to_insert: List[Dict[str, Any]] = []
            file_chunks = 0

            for txt, meta in pairs:
                for j, ctext in enumerate(chunk_text(txt, 1000, 200)):
                    rec = {
                        "text": ctext,
                        "title": os.path.basename(blob.name),
                        "source": meta.get("source"),
                        "path": meta.get("path"),
                        "page": int(meta.get("page", 0)) if meta.get("page") else None,
                        "block_id": f"{blob.name}#{j}",
                    }
                    to_insert.append(rec)
                    file_chunks += 1

            if to_insert:
                upsert_chunks(coll, to_insert)
                total_chunks += file_chunks

            success_files += 1

        except Exception as e:
            failed_files += 1
            st.write(f"⚠️ Failed: {blob.name} — {e}")

    prog.progress(1.0)
    return {
        "files_seen": total_blobs,
        "files_ok": success_files,
        "files_failed": failed_files,
        "chunks_upserted": total_chunks,
    }


def build_agent(client, collection_name: str):
    """Return a QueryAgent if available; otherwise None (we'll fallback)."""
    if QueryAgent is None or QueryAgentCollectionConfig is None:
        return None
    return QueryAgent(
        client=client,
        collections=[QueryAgentCollectionConfig(name=collection_name)],
    )


def hybrid_snippets(client, collection_name: str, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
    try:
        coll = client.collections.get(collection_name)
    except Exception as e:
        st.info(f"Could not access collection '{collection_name}': {e}")
        return []

    try:
        res = coll.query.hybrid(
            query=query,
            limit=top_k,
            alpha=alpha,
            return_metadata=["score", "distance"],
            return_properties=["text", "title", "source", "path", "page", "block_id"],
        )
        items = []
        for o in res.objects or []:
            props = getattr(o, "properties", {}) or {}
            meta = getattr(o, "metadata", {}) or {}
            items.append({
                "text": props.get("text") or "",
                "title": props.get("title") or "",
                "source": props.get("source") or props.get("path") or "",
                "page": props.get("page") or "",
                "block_id": props.get("block_id") or "",
                "score": meta.get("score"),
                "distance": meta.get("distance"),
            })
        return items
    except Exception as e:
        st.warning(f"Hybrid snippet fetch failed: {e}")
        return []


# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="Forecast360 Agent (Weaviate)", page_icon="🤖", layout="wide")
st.title("🤖 Forecast360 — Knowledge Agent")
st.caption("Ingest from Azure Blob → Weaviate ‘Forecast360’, then chat with your KB.")
st.divider()

# Read settings ONLY from secrets/env (no sidebar inputs)
wset = resolve_weaviate_settings()
aset = resolve_azure_settings()

# Minimal sidebar: action buttons only (no connection inputs)
with st.sidebar:
    st.subheader("Actions")
    connect_clicked = st.button("🔌 Connect / Refresh Weaviate", type="primary", use_container_width=True)
    ingest_clicked = st.button("⬆️ Ingest from Azure → Weaviate", use_container_width=True)

# Connect Weaviate
if "client" not in st.session_state or connect_clicked:
    try:
        st.session_state.client = connect_weaviate(wset.url, wset.api_key, timeout_s=wset.timeout_s)
        st.success("Connected to Weaviate.")
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.stop()

# Ensure collection exists
try:
    ensure_collection(st.session_state.client, wset.collection)
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

# Prepare agent (or fallback)
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

def render_snippets(snips: List[Dict[str, Any]]):
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
    # user turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant turn
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if st.session_state.agent is not None:
                    # Use QueryAgent (requires weaviate-agents)
                    result = st.session_state.agent.run(prompt)
                    answer_text = getattr(result, "content", None) or str(result)
                else:
                    # Fallback: hybrid search summary (works without agents package)
                    st.info("Using hybrid-search fallback (install 'weaviate-agents' for QueryAgent).")
                    snips = hybrid_snippets(st.session_state.client, wset.collection, prompt, top_k=wset.top_k, alpha=wset.alpha)
                    if snips:
                        # naive summary: join top snippets (you can replace with LLM if desired)
                        joined = "\n\n".join(s["text"] for s in snips[:3] if s.get("text"))
                        answer_text = f"Here are the most relevant excerpts I found:\n\n{joined}"
                    else:
                        answer_text = "I couldn't find relevant excerpts in the knowledge base."

                st.markdown(answer_text)

            # Always show supporting matches
            snips = hybrid_snippets(st.session_state.client, wset.collection, prompt, top_k=wset.top_k, alpha=wset.alpha)
            render_snippets(snips)

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
