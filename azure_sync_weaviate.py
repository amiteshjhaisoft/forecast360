# kb_sync_azure.py
# Ingest/refresh Weaviate from Azure Blob Storage (uses Streamlit secrets)
# Author: Amitesh Jha | iSoft

from __future__ import annotations
import io, os, json, hashlib, re
from typing import Any, Dict, List, Optional, Iterable, Tuple

import numpy as np
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

import weaviate
import weaviate.classes as wvc

# Optional local embeddings (used when your collection vectorizer is set to "none")
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # graceful


# ----------------------- Secrets helpers -----------------------
def _sget(st_secrets, section: str, key: str, default: Any = None) -> Any:
    try:
        return st_secrets[section].get(key, default)  # type: ignore
    except Exception:
        return default


# ----------------------- Text extraction -----------------------
_TEXT_EXT = {".txt", ".md", ".json", ".csv", ".tsv", ".html", ".htm"}

def _ext(path: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", path)
    return m.group(1).lower() if m else ""

def _to_text(name: str, data: bytes) -> Optional[str]:
    ext = _ext(name)
    try:
        if ext in {".txt", ".md"}:
            return data.decode("utf-8", errors="ignore")
        if ext in {".json"}:
            return data.decode("utf-8", errors="ignore")
        if ext in {".csv", ".tsv"}:
            return data.decode("utf-8", errors="ignore")
        if ext in {".html", ".htm"}:
            # light scrub: remove tags crudely
            html = data.decode("utf-8", errors="ignore")
            text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
    except Exception:
        return None
    return None  # unsupported type (PDF/DOCX/etc are intentionally skipped to keep deps minimal)


# ----------------------- Chunking -----------------------
def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        # try to cut at sentence boundary
        cut = re.search(r"(?<=\.|\?|!)\s", text[i:end])
        if cut:
            end = i + cut.end()
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap if end - overlap > i else end
    return chunks


# ----------------------- Embedding (optional) -----------------------
def _maybe_embedder(model_name: Optional[str]) -> Optional["SentenceTransformer"]:
    if not model_name or SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


# ----------------------- Azure Blob helpers -----------------------
def _azure_blob_client(st_secrets) -> BlobServiceClient:
    conn_str = _sget(st_secrets, "azure", "connection_string") or _sget(st_secrets, "azure_blob", "connection_string")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    account_url = _sget(st_secrets, "azure", "account_url") or _sget(st_secrets, "azure_blob", "account_url")
    if not account_url:
        raise RuntimeError("Azure Blob config missing. Provide either connection_string or account_url in secrets.")
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return BlobServiceClient(account_url=account_url, credential=cred)


# ----------------------- Weaviate helpers -----------------------
def _weaviate_client(st_secrets) -> weaviate.WeaviateClient:
    url = _sget(st_secrets, "weaviate", "url", "")
    if not url:
        raise RuntimeError("Set [weaviate].url in secrets.")
    api_key = _sget(st_secrets, "weaviate", "api_key", "")
    auth = wvc.init.Auth.api_key(api_key) if api_key else None

    # timeouts via AdditionalConfig
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=auth,
        additional_config=wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=30, query=60, insert=120)
        ),
    )
    return client


def _collection_info(coll) -> Tuple[bool, Optional[int]]:
    """Return (needs_manual_vectors, dims_if_known) from collection config."""
    try:
        cfg = coll.config.get()
        # vectorizer: "none" means you must send vectors
        vcfg = getattr(cfg, "vectorizer", None)
        needs_vec = (str(vcfg).lower() == "none")
        dims = None
        try:
            dims = int(getattr(cfg, "vector_index_config", {}).get("vector_cache_max_objects", None))  # not reliable
        except Exception:
            pass
        return needs_vec, dims
    except Exception:
        return False, None


# ----------------------- Core sync -----------------------
def sync_from_azure(
    st_secrets,
    collection_name: str,
    container_key: str = "container",     # secrets['azure' or 'azure_blob'][container_key]
    prefix_key: str = "prefix",           # optional subfolder in the container
    embed_model_key: str = ("rag", "embed_model"),  # ('section','key') for local embedding model
    delete_before_upsert: bool = True,
    max_docs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Reads text-like files from an Azure Blob *folder* and upserts chunks into a Weaviate v4 collection.
    Properties stored: text, source
    If the collection has vectorizer=none, we compute vectors locally and send them.
    """
    container = _sget(st_secrets, "azure", container_key) or _sget(st_secrets, "azure_blob", container_key)
    if not container:
        raise RuntimeError("Set [azure].[container] (or [azure_blob]) in secrets.")
    prefix = _sget(st_secrets, "azure", prefix_key) or _sget(st_secrets, "azure_blob", prefix_key) or ""

    # Connect
    bs = _azure_blob_client(st_secrets)
    client = _weaviate_client(st_secrets)
    coll = client.collections.use(collection_name)

    needs_vec, _ = _collection_info(coll)
    embed_model_name = None
    if isinstance(embed_model_key, (tuple, list)) and len(embed_model_key) == 2:
        embed_model_name = _sget(st_secrets, embed_model_key[0], embed_model_key[1])
    embedder = _maybe_embedder(embed_model_name) if needs_vec else None

    container_client = bs.get_container_client(container)
    blob_iter = container_client.list_blobs(name_starts_with=prefix or None)

    inserted, skipped, cleared_sources = 0, 0, 0
    batch_props: List[Dict[str, Any]] = []
    batch_vecs: Optional[List[List[float]]] = [] if needs_vec else None
    batch_size = 64
    processed_docs = 0

    def _flush():
        nonlocal inserted, batch_props, batch_vecs
        if not batch_props:
            return
        if needs_vec:
            coll.data.insert_many(objects=batch_props, vectors=batch_vecs)  # vectors aligned to objects
        else:
            coll.data.insert_many(objects=batch_props)
        inserted += len(batch_props)
        batch_props = []
        if batch_vecs is not None:
            batch_vecs = []

    for blob in blob_iter:
        if max_docs and processed_docs >= max_docs:
            break
        name = blob.name
        if _ext(name) not in _TEXT_EXT:
            skipped += 1
            continue

        bclient = container_client.get_blob_client(name)
        data = bclient.download_blob().readall()
        text = _to_text(name, data)
        if not text:
            skipped += 1
            continue

        src = f"azure://{container}/{name}"

        # Clear old records for this source (idempotent refresh)
        if delete_before_upsert:
            try:
                coll.data.delete_many(
                    where=wvc.query.Filter.by_property("source").equal(src)
                )
                cleared_sources += 1
            except Exception:
                pass

        # Chunk
        chunks = _chunk_text(text, max_chars=1400, overlap=200)
        if not chunks:
            skipped += 1
            continue

        # Prepare objects (properties + optional vectors)
        for ch in chunks:
            obj = {"text": ch, "source": src}
            batch_props.append(obj)
            if needs_vec:
                if embedder is None:
                    raise RuntimeError(
                        "Collection is configured with vectorizer='none' but no local embedding model is available. "
                        "Set [rag].embed_model in secrets and install sentence-transformers."
                    )
                vec = embedder.encode([ch], normalize_embeddings=True)[0].astype("float32").tolist()
                batch_vecs.append(vec)  # type: ignore

            if len(batch_props) >= batch_size:
                _flush()

        processed_docs += 1

    _flush()
    client.close()
    return {
        "inserted_chunks": inserted,
        "skipped_files": skipped,
        "cleared_sources": cleared_sources,
        "processed_files": processed_docs,
        "vectorizer_none": needs_vec,
        "used_embed_model": embed_model_name if needs_vec else None,
    }
