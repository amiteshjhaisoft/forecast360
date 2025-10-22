# forecast360_agent.py
# Forecast360 — Weaviate-backed AI Agent with Azure Blob → Weaviate pre-ingestion
# Author: Amitesh Jha | iSoft

from __future__ import annotations
import os, io, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple

import streamlit as st

# ---------- Weaviate (client v4 + Agents) ----------
import weaviate
from weaviate.classes.init import Auth
from weaviate.agents.query import QueryAgent
from weaviate_agents.classes import QueryAgentCollectionConfig
from weaviate.classes.config import Property, DataType, Configure

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
    account_name: Optional[str] = None         # if you prefer account/key
    account_key: Optional[str] = None
    container: str = "knowledgebase"
    prefix: str = "KB/"


def _get_secret(path: str, default: Optional[str] = None) -> Optional[str]:
    """
    Fetch value from Streamlit secrets (nested) or environment.
    'path' can be 'weaviate.url' or 'azure.connection_string' or single-level 'WEAVIATE_URL'.
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
    # env fallback
    return os.environ.get(path.replace(".", "_").upper(), default)


def resolve_weaviate_settings() -> WeaviateSettings:
    url = (
        _get_secret("weaviate.url")
        or _get_secret("WEAVIATE_URL")
        or "http://localhost:8080"
    )
    api_key = _get_secret("weaviate.api_key") or _get_secret("WEAVIATE_API_KEY") or None
    collection = _get_secret("weaviate.collection", DEFAULT_COLLECTION) or DEFAULT_COLLECTION

    with st.sidebar:
        st.subheader("🔧 Weaviate Agent Settings")
        url = st.text_input("Weaviate URL", url, help="Cloud or local (e.g., http://localhost:8080)")
        api_key = st.text_input("Weaviate API Key (optional for local)", api_key or "", type="password")
        collection = st.text_input("Collection name", collection, help="Weaviate collection for KB")
        top_k = st.slider("Top-K snippets", 1, 20, 5)
        alpha = st.slider("Hybrid α (0=BM25, 1=Vector)", 0.0, 1.0, 0.5, 0.05)
    return WeaviateSettings(url=url, api_key=api_key or None, collection=collection, top_k=top_k, alpha=alpha)


def resolve_azure_settings() -> AzureSettings:
    conn = _get_secret("azure.connection_string") or _get_secret("AZURE_CONNECTION_STRING")
    acct_url = _get_secret("azure.account_url") or _get_secret("AZURE_ACCOUNT_URL")
    acct_name = _get_secret("azure.account_name") or _get_secret("AZURE_ACCOUNT_NAME")
    acct_key = _get_secret("azure.account_key") or _get_secret("AZURE_ACCOUNT_KEY")
    container = _get_secret("azure.container", "knowledgebase") or "knowledgebase"
    prefix = _get_secret("azure.prefix", "KB/") or "KB/"

    with st.sidebar:
        st.subheader("🗄️ Azure Blob Source")
        container = st.text_input("Container", container)
        prefix = st.text_input("Prefix (folder)", prefix)
        with st.expander("Azure auth (choose one)"):
            conn = st.text_input("Connection string", conn or "", type="password")
            acct_url = st.text_input("Account URL", acct_url or "", help="https://<account>.blob.core.windows.net")
            col1, col2 = st.columns(2)
            with col1:
                acct_name = st.text_input("Account name", acct_name or "")
            with col2:
                acct_key = st.text_input("Account key", acct_key or "", type="password")

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
    if url.startswith("http"):
        if any(k in url for k in ("weaviate.network", "weaviate.cloud", "semi.network")):
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key) if api_key else None,
                timeout=timeout_s,
            )
        else:
            # local
            host = url.replace("http://", "").replace("https://", "")
            host = host.split("/")[0]
            parts = host.split(":")
            http_host = parts[0]
            http_port = int(parts[1]) if len(parts) > 1 else 8080
            return weaviate.connect_to_local(http_host=http_host, http_port=http_port, grpc_port=None, timeout=timeout_s)
    # else: raw cluster identifier
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key) if api_key else None,
        timeout=timeout_s,
    )


def ensure_collection(client, name: str):
    """
    Ensure a collection exists with reasonable KB properties and server-side vectorizer.
    You can switch to Configure.Vectorizer.none() if you intend to push vectors yourself.
    """
    try:
        exists = any(c.name == name for c in client.collections.list_all().collections)
    except Exception:
        # older client fallbacks
        try:
            client.collections.get(name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),  # requires module on the Weaviate server
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT, description="Optional title"),
                Property(name="source", data_type=DataType.TEXT, description="Blob name or URI"),
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
    raise RuntimeError("Provide Azure Blob connection_string or account_url + account_key.")


# =========================
# Text extraction & chunker
# =========================

def extract_text_from_blob(name: str, content: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (text, meta) pairs extracted from a blob file.
    meta includes source, page, etc.
    """
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
        # Keep text; pandas reading is optional here
        items.append((_as_utf8(content), {**meta_base}))
    elif name_l.endswith(".json"):
        # pretty print JSON to text
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
        # Fallback: try decode
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
        if i < 0:  # safety
            break
    return parts


def upsert_chunks(coll, records: List[Dict[str, Any]]):
    # Try bulk insert if available
    try:
        coll.data.insert_many(records)
    except Exception:
        for r in records:
            try:
                coll.data.insert(r)
            except Exception:
                pass


def ingest_azure_prefix_to_weaviate(client, az: AzureSettings, collection_name: str) -> Dict[str, int]:
    """
    Streams blobs under container/prefix, extracts text, chunks, and upserts into Weaviate.
    Returns summary counts.
    """
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
    file_counter = 0

    for blob in blob_iter:
        total_blobs += 1
        file_counter += 1
        prog.progress(min(0.99, file_counter / max(total_blobs, file_counter + 1)))

        try:
            downloader = container.download_blob(blob.name)
            data = downloader.readall()
            pairs = extract_text_from_blob(blob.name, data)
            file_chunks = 0
            to_insert: List[Dict[str, Any]] = []

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


def build_agent(client, collection_name: str) -> QueryAgent:
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

wset = resolve_weaviate_settings()
aset = resolve_azure_settings()

with st.sidebar:
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
            st.success(f"Ingestion complete: {summary}")
            st.json(summary)
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

# Build agent
if "agent" not in st.session_state or connect_clicked:
    try:
        st.session_state.agent = build_agent(st.session_state.client, wset.collection)
    except Exception as e:
        st.error(f"Agent init failed: {e}")
        st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Click **Ingest** to load Azure KB, then ask about your Forecast360 knowledge base."}
    ]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about the Forecast360 KB…")

def render_snippets(snips: List[Dict[str, Any]]):
    if not snips:
        return
    st.markdown("#### 🔎 Supporting snippets")
    for i, s in enumerate(snips, 1):
        with st.expander(f"Match {i} — score={s.get('score'):.4f if s.get('score') is not None else 'n/a'} | {s.get('source')}", expanded=(i == 1)):
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

    # agent turn
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.agent.run(prompt)

                # Extract answer
                answer_text = None
                citations = None
                try:
                    answer_text = getattr(result, "content", None)
                    citations = getattr(result, "citations", None) or getattr(result, "evidence", None)
                except Exception:
                    pass
                if not answer_text:
                    answer_text = str(result)

                st.markdown(answer_text)

            # Show supporting matches
            snips = hybrid_snippets(st.session_state.client, wset.collection, prompt, top_k=wset.top_k, alpha=wset.alpha)
            render_snippets(snips)

        st.session_state.messages.append({"role": "assistant", "content": answer_text})

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Agent error: {e}")
