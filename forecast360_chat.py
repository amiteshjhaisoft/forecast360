# Author: Amitesh Jha | iSoft | 2025-10-12
# deci_int.py — Streamlit RAG chat (Claude-only, no sidebar, hardcoded settings)
# Forecast360 customizations:
# - Azure Blob is the single source of truth (mirror prefix to ./KB even if meta/version.json is absent)
# - Secrets-first Azure config (Streamlit secrets -> env), with clear diagnostics
# - Auto-index with FAISS (HuggingFace all-MiniLM-L6-v2) and signature-based change detection
# - Inline status bar; compact controls (chat height, quick reindex, show KB stats)
# - Chat slash-commands: /read <name>, /open <name>, /show <name>, /set key=value ...
# - Full-document summarization via Claude when documents are long
# - Robust loaders; safe fallbacks for CSV/XLSX/TSV/HTML/PDF; placeholders for non-text assets
# - Clean, Forecast360-themed chat UI; no sidebar

from __future__ import annotations

import os, glob, time, base64, hashlib, json, shutil, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd

# ---- Runtime hygiene
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---- LangChain / Vector
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader, UnstructuredPowerPointLoader
)

# ---- Anthropic (Claude only)
try:
    from anthropic import Anthropic as _AnthropicClientNew
except Exception:
    _AnthropicClientNew = None
try:
    from anthropic import Client as _AnthropicClientOld
except Exception:
    _AnthropicClientOld = None

# ---- Azure SDK (used to pull KB down to ./KB)
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    try:
        from azure.identity import DefaultAzureCredential
    except Exception:
        DefaultAzureCredential = None  # type: ignore
    _AZURE_OK = True
except Exception:
    BlobServiceClient = ContainerClient = DefaultAzureCredential = None  # type: ignore
    _AZURE_OK = False

# ---------------- Constants (hardcoded; no sidebar)
DEFAULT_CLAUDE = "claude-sonnet-4-5"
_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_MODEL_KW = {"device": "cpu", "trust_remote_code": False}
_ENCODE_KW = {"normalize_embeddings": True}

TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".pptx", ".pptm", ".doc", ".odt"}
SPREADSHEET_EXTS = {".xlsx", ".xlsm", ".xltx"}
SUPPORTED_TEXT_DOCS = TEXT_EXTS | DOC_EXTS | SPREADSHEET_EXTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
SUPPORTED_EXTS = SUPPORTED_TEXT_DOCS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

GREETING_RE = re.compile(
    r"""^\s*(hi|hello|hey|hiya|yo|hola|namaste|namaskar|g'day|good\s+(morning|afternoon|evening))[\s!,.?]*$""",
    re.IGNORECASE,
)

VectorStoreType = FAISS

# ---------------- Minimal Claude wrapper (no tools, plain text)
class ClaudeDirect(BaseChatModel):
    model: str = DEFAULT_CLAUDE
    temperature: float = 0.2
    max_tokens: int = 800
    _client: object = None

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client)

    @property
    def _llm_type(self) -> str:
        return "claude_direct"

    def _convert_msgs(self, messages: list[BaseMessage]):
        out = []
        for m in messages:
            role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else "user")
            if isinstance(m.content, str):
                text = m.content
            else:
                parts = m.content or []
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parts)
            out.append({"role": role, "content": [{"type": "text", "text": text}]})
        return out

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        amsgs = self._convert_msgs(messages)
        resp = self._client.messages.create(
            model=self.model, messages=amsgs, temperature=self.temperature, max_tokens=self.max_tokens
        )
        text = ""
        content = getattr(resp, "content", []) or []
        for blk in content:
            if getattr(blk, "type", None) == "text":
                text += getattr(blk, "text", "") or ""
            elif isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "") or ""
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])

# ---------------- Theme / CSS (Forecast360 look)
try:
    st.set_page_config(page_title="Forecast360 • Chat", page_icon="💬", layout="wide")
except Exception:
    pass

USER_AVATAR_PATH: Optional[Path] = None
ASSIST_AVATAR_PATH: Optional[Path] = None


def _first_existing(paths: list[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def _resolve_avatar_paths() -> Tuple[Optional[Path], Optional[Path]]:
    user_env = os.getenv("USER_AVATAR_PATH")
    asst_env = os.getenv("ASSISTANT_AVATAR_PATH")
    user = _first_existing([
        Path(user_env).expanduser().resolve() if user_env else None,
        Path.cwd() / "assets" / "avatar.png",
        Path.cwd() / "assets" / "user.png",
        Path.cwd() / "assets" / "me.png",
    ])
    asst = _first_existing([
        Path(asst_env).expanduser().resolve() if asst_env else None,
        Path.cwd() / "assets" / "llm.png",
        Path.cwd() / "assets" / "assistant.png",
        Path.cwd() / "assets" / "bot.png",
        Path.cwd() / "assets" / "robot.png",
    ])
    return user, asst


USER_AVATAR_PATH, ASSIST_AVATAR_PATH = _resolve_avatar_paths()


def _img_to_data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = (path.suffix.lower().lstrip(".") or "png")
    mime = "image/png" if ext in ("png", "apng") else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml")
    return f"data:{mime};base64,{b64}"


USER_AVATAR_URI = _img_to_data_uri(USER_AVATAR_PATH)
ASSIST_AVATAR_URI = _img_to_data_uri(ASSIST_AVATAR_PATH)
user_bg = f"background-image:url('{USER_AVATAR_URI}');" if USER_AVATAR_URI else ""
asst_bg = f"background-image:url('{ASSIST_AVATAR_URI}');" if ASSIST_AVATAR_URI else ""

st.markdown(f"""
<style>
:root{{ --bg:#f7f8fb; --panel:#fff; --text:#0b1220; --border:#e7eaf2;
       --bubble-user:#eef4ff; --bubble-assist:#f6f7fb; --accent:#2563eb; }}
.chat-card{{ background:var(--panel); border:1px solid var(--border); border-radius:14px;
            box-shadow:0 6px 16px rgba(16,24,40,.05); overflow:hidden; }}
.msg{{ display:flex; align-items:flex-start; gap:.65rem; margin:.45rem 0; }}
.avatar{{ width:32px; height:32px; border-radius:50%; border:1px solid var(--border);
          background-size:cover; background-position:center; background-repeat:no-repeat; flex:0 0 32px; }}
.avatar.user {{ {user_bg} }}
.avatar.assistant {{ {asst_bg} }}
.bubble{{ border:1px solid var(--border); background:var(--bubble-assist);
         padding:.8rem .95rem; border-radius:12px; max-width:960px; white-space:pre-wrap; line-height:1.45; }}
.msg.user .bubble{{ background:var(--bubble-user); }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px;
                padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
.small-note{{opacity:.85;font-size:.85rem}}
</style>
""", unsafe_allow_html=True)

# ---------------- KB + Azure helpers (download-only mirror)

def _local_kb_dir() -> Path:
    p = Path.cwd() / "KB"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _kb_local_version() -> Optional[str]:
    vf = _local_kb_dir() / "meta" / "version.json"
    if vf.exists():
        try:
            return json.loads(vf.read_text(encoding="utf-8")).get("version")
        except Exception:
            return None
    return None


def _azure_cfg() -> Dict[str, Any]:
    # Prefer Streamlit secrets; then env
    try:
        az = st.secrets.get("azure", {})  # type: ignore
    except Exception:
        az = {}
    return {
        "account_url":       az.get("account_url")         or os.getenv("AZURE_ACCOUNT_URL"),
        "connection_string": az.get("connection_string")   or os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        "container":         az.get("container")           or os.getenv("AZURE_BLOB_CONTAINER", "forecast360-kb"),
        "prefix":            az.get("prefix", "KB"),
        "sas_url":           az.get("container_sas_url")   or os.getenv("AZURE_BLOB_CONTAINER_URL"),
    }


def _azure_diag(cfg: Dict[str, Any]) -> str:
    flags = []
    flags.append("sas_url" if cfg.get("sas_url") else "-")
    flags.append("conn_str" if cfg.get("connection_string") else "-")
    flags.append("acct_url" if cfg.get("account_url") else "-")
    return "/".join(flags) + f" · container='{cfg.get('container')}' · prefix='{cfg.get('prefix')}'"


def _azure_container_client() -> Optional["ContainerClient"]:
    if not _AZURE_OK:
        return None
    cfg = _azure_cfg()
    if cfg["sas_url"]:
        try:
            return ContainerClient.from_container_url(cfg["sas_url"])
        except Exception:
            return None
    if cfg["connection_string"]:
        try:
            svc = BlobServiceClient.from_connection_string(cfg["connection_string"])
            return svc.get_container_client(cfg["container"])
        except Exception:
            return None
    if cfg["account_url"] and DefaultAzureCredential is not None:
        try:
            cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            svc = BlobServiceClient(account_url=cfg["account_url"], credential=cred)
            return svc.get_container_client(cfg["container"])
        except Exception:
            return None
    return None


def _azure_kb_version() -> Optional[str]:
    if not _AZURE_OK:
        return None
    try:
        cli = _azure_container_client()
        if not cli:
            return None
        pref = _azure_cfg()["prefix"].rstrip("/") + "/"
        blob = pref + "meta/version.json"
        txt = cli.download_blob(blob).readall().decode("utf-8")
        return json.loads(txt).get("version")
    except Exception:
        return None


def _download_entire_prefix(cli: "ContainerClient", prefix: str, dest: Path) -> int:
    """Mirror ALL blobs under prefix into dest. Returns number of files downloaded."""
    count = 0
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    for blob in cli.list_blobs(name_starts_with=prefix or None):
        rel = blob.name[len(prefix):] if prefix else blob.name
        if not rel or rel.endswith("/"):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = cli.download_blob(blob.name).readall()
            with open(target, "wb") as f:
                f.write(data)
            count += 1
        except Exception:
            pass
    return count


def sync_kb_from_azure_if_needed(status_placeholder=None) -> Tuple[Path, str]:
    """
    If Azure has a different KB version, or local KB is empty/missing meta,
    mirror prefix -> ./KB (download all files). Returns (kb_path, status_label).
    """
    kb_root = _local_kb_dir()
    if not _AZURE_OK:
        return kb_root, "Azure SDK missing — using local KB only."

    cli = _azure_container_client()
    if not cli:
        return kb_root, "Azure not configured — using local KB only."

    cfg  = _azure_cfg()
    pref = cfg["prefix"].rstrip("/") if cfg["prefix"] else ""

    remote_ver = _azure_kb_version()
    local_ver  = _kb_local_version()

    local_empty = len(list(kb_root.rglob("*"))) == 0

    should_mirror = (remote_ver is not None and remote_ver != local_ver) or (remote_ver is None and local_empty)

    if not should_mirror:
        label = f"KB sync OK · remote={remote_ver or 'unknown'} · local={local_ver or 'unknown'} · {_azure_diag(cfg)}"
        return kb_root, label

    # wipe local KB to avoid stale files
    try:
        shutil.rmtree(kb_root, ignore_errors=True)
    except Exception:
        pass
    kb_root.mkdir(parents=True, exist_ok=True)

    downloaded = _download_entire_prefix(cli, pref, kb_root)
    label = f"Synced {downloaded} files from Azure · remote={remote_ver or 'unknown'} · {_azure_diag(cfg)}"
    return kb_root, label

# ---------------- Small utils

def human_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.2f} s"


def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def iter_files(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(list(set(paths)))


def compute_kb_signature(folder: str) -> Tuple[str, int]:
    files = iter_files(folder)
    lines = []
    base = os.path.abspath(folder)
    for p in files:
        try:
            stt = os.stat(p)
            rel = os.path.relpath(os.path.abspath(p), base)
            lines.append(f"{rel}|{stt.st_size}|{int(stt.st_mtime)}")
        except Exception:
            continue
    lines.sort()
    raw = "\n".join(lines) + str(SUPPORTED_TEXT_DOCS)
    return stable_hash(raw if raw else f"EMPTY-{time.time()}"), len(files)

# ---------------- Loading

def _fallback_read(path: str) -> str:
    try:
        if path.lower().endswith(tuple(SPREADSHEET_EXTS)):
            df = pd.read_excel(path).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"Spreadsheet content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        if path.lower().endswith((".csv", ".tsv")):
            sep = "\t" if path.lower().endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"CSV/TSV content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading file {Path(path).name}: {e}")
        return ""


def load_one(path: str) -> List[Document]:
    p = path.lower()
    if p.endswith(tuple(IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS)):
        doc_type = "Image" if p.endswith(tuple(IMAGE_EXTS)) else ("Audio" if p.endswith(tuple(AUDIO_EXTS)) else "Video")
        placeholder_content = (
            f"This document is a {doc_type} file. "
            f"Text content unavailable (requires OCR/transcription). "
            f"Metadata: {Path(path).name}."
        )
        return [Document(page_content=placeholder_content, metadata={"source": path, "type": doc_type, "status": "placeholder"})]

    try:
        if p.endswith(".pdf"):
            return PyPDFLoader(path).load()
        if p.endswith((".html", ".htm")):
            return BSHTMLLoader(path).load()
        if p.endswith(".docx"):
            return Docx2txtLoader(path).load()
        if p.endswith((".pptx", ".pptm")):
            return UnstructuredPowerPointLoader(path).load()
        if p.endswith(".csv"):
            return CSVLoader(path).load()
        if p.endswith(".tsv"):
            return CSVLoader(path, csv_args={"delimiter": "\t"}).load()
        if p.endswith(tuple(TEXT_EXTS | SPREADSHEET_EXTS | {".doc", ".odt"})):
            txt = _fallback_read(path)
            return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
        txt = _fallback_read(path)
        return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
    except Exception as e:
        st.warning(f"Failed to load/process {Path(path).name}. Error: {e}")
        return []


def load_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    files_to_load = [p for p in iter_files(folder) if Path(p).suffix.lower() in SUPPORTED_EXTS]
    for path in files_to_load:
        docs.extend(load_one(path))
    return docs

# ---------------- Indexing (FAISS)

@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200


@st.cache_resource(show_spinner=False)
def _cached_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=_EMB_MODEL,
        model_kwargs=_EMB_MODEL_KW,
        encode_kwargs=_ENCODE_KW,
    )


def _make_embeddings():
    return _cached_embeddings()


def _faiss_dir(persist_dir: str, collection_name: str) -> Path:
    return Path(persist_dir).expanduser().resolve() / collection_name


def index_folder_langchain(folder: str, persist_dir: str, collection_name: str,
                           emb_model: str, chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    raw_docs = load_documents(folder)
    faiss_dir = _faiss_dir(persist_dir, collection_name)

    if not raw_docs:
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir, ignore_errors=True)
        return (0, 0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size, chunk_overlap=chunk_cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    splat = splitter.split_documents(raw_docs)
    embeddings = _make_embeddings()

    faiss_db = FAISS.from_documents(documents=splat, embedding=embeddings)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_db.save_local(str(faiss_dir))
    return (len(raw_docs), len(splat))


def get_vectorstore(persist_dir: str, collection_name: str, emb_model: str) -> Optional[FAISS]:
    key = f"_vs::{persist_dir}::{collection_name}::{emb_model}"
    if key in st.session_state:
        return st.session_state[key]
    faiss_path = _faiss_dir(persist_dir, collection_name)
    if not faiss_path.exists():
        return None
    try:
        vs = FAISS.load_local(
            folder_path=str(faiss_path),
            embeddings=_make_embeddings(),
            allow_dangerous_deserialization=True,
        )
        st.session_state[key] = vs
        return vs
    except Exception as e:
        st.error(f"Failed to load FAISS index from disk. Error: {e}")
        return None

# ---------------- Claude init (hardcoded model) — lazy, cached

def _strip_proxy_env() -> None:
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(v, None)


def _get_secret_api_key() -> Optional[str]:
    try:
        s = st.secrets
    except Exception:
        s = None
    if s:
        for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for parent in ("anthropic","claude","secrets"):
            if parent in s and isinstance(s[parent], dict):
                ns = s[parent]
                for k in ("api_key","ANTHROPIC_API_KEY","key","token"):
                    v = ns.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


@st.cache_resource(show_spinner=False)
def _anthropic_client_from_secrets_cached():
    _strip_proxy_env()
    api_key = _get_secret_api_key()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY. Set it in .streamlit/secrets.toml under [anthropic] api_key=\"...\" or env.")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if _AnthropicClientNew is not None:
        return _AnthropicClientNew(api_key=api_key)
    if _AnthropicClientOld is not None:
        return _AnthropicClientOld(api_key=api_key)
    raise RuntimeError("Anthropic SDK not installed correctly.")


def make_llm(model_name: str, temperature: float):
    client = _anthropic_client_from_secrets_cached()
    return ClaudeDirect(client=client, model=model_name or DEFAULT_CLAUDE,
                        temperature=temperature, max_tokens=800)


def make_chain(vs: VectorStoreType, llm: BaseChatModel, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False
    )

# ---------------- Defaults + auto-index (hardcoded)

def settings_defaults() -> Dict[str, Any]:
    kb_dir = str(_local_kb_dir())
    return {
        "persist_dir": ".faiss_index",
        "collection_name": f"kb-{stable_hash(kb_dir)}",
        "base_folder": kb_dir,
        "emb_model": _EMB_MODEL,
        "chunk_cfg": ChunkingConfig(),
        "claude_model": DEFAULT_CLAUDE,
        "temperature": 0.2,
        "top_k": 5,
        "auto_index_min_interval_sec": 8,
        "chat_height": 560,
    }


def auto_index_if_needed(status_placeholder: Optional[object] = None) -> Optional[VectorStoreType]:
    folder = st.session_state.get("base_folder")
    persist = st.session_state.get("persist_dir")
    colname = st.session_state.get("collection_name")
    emb_model = st.session_state.get("emb_model")
    min_gap = int(st.session_state.get("auto_index_min_interval_sec", 8))

    sig_now, file_count = compute_kb_signature(folder)
    last_sig = st.session_state.get("_kb_last_sig")
    last_time = float(st.session_state.get("_kb_last_index_ts", 0.0))
    now = time.time()

    need_index = (last_sig != sig_now) or (last_sig is None)
    throttled = (now - last_time) < min_gap
    target = status_placeholder if status_placeholder is not None else st

    faiss_path = _faiss_dir(persist, colname)
    index_exists = faiss_path.is_dir() and any(faiss_path.iterdir())

    if need_index and not throttled:
        try:
            target.markdown('<div class="status-inline">Indexing…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(folder, persist, colname, emb_model,
                                                      st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            label = f"Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks"
        except Exception as e:
            label = f"Auto-index failed: <b>{e}</b>"
        target.markdown(f'<div class="status-inline">{label}</div>', unsafe_allow_html=True)
    elif not index_exists:
        try:
            target.markdown('<div class="status-inline">Index missing — building…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(folder, persist, colname, emb_model,
                                                      st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            target.markdown(f'<div class="status-inline">Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks</div>',
                            unsafe_allow_html=True)
        except Exception as e:
            target.markdown(f'<div class="status-inline">Auto-index failed: <b>{e}</b></div>',
                            unsafe_allow_html=True)
    else:
        ts = st.session_state.get("_kb_last_index_ts")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        target.markdown(
            f'<div class="status-inline">Auto-index is <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Index: <code>{colname}</code></div>',
            unsafe_allow_html=True
        )

    try:
        return get_vectorstore(persist, colname, emb_model)
    except Exception:
        return None

# ---------------- UI helpers
def _avatar_for_role(role: str) -> Optional[str]:
    # Prefer data URIs (work in Streamlit avatars); fall back to local path, then emoji
    if role == "user":
        return USER_AVATAR_URI or (str(USER_AVATAR_PATH) if USER_AVATAR_PATH else "👤")
    if role == "assistant":
        return ASSIST_AVATAR_URI or (str(ASSIST_AVATAR_PATH) if ASSIST_AVATAR_PATH else "🤖")
    return None

def render_chat_history():
    for message in st.session_state.get("messages", []):
        role = message["role"]
        with st.chat_message(role, avatar=_avatar_for_role(role)):
            st.markdown(message["content"])


def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    if not source_docs:
        return ""
    from collections import Counter
    names = []
    for d in source_docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "unknown")
        try:
            display = str(Path(src).resolve().relative_to(Path(kb_root).resolve())) if kb_root else Path(src).name
        except Exception:
            display = Path(src).name
        names.append(display)
    counts = Counter(names)
    lines = [f"- {name}" + (f" ×{n}" if n > 1 else "") for name, n in counts.items()]
    return "\n\n**Sources**\n" + "\n".join(lines)


def read_whole_file_from_disk(path: str) -> str:
    docs = load_one(path)
    return "".join(
        (f"\n\n--- [{i}] {Path((d.metadata or {}).get('source','')).name} ---\n") + (d.page_content or "")
        for i, d in enumerate(docs, 1)
    ).strip()


def read_whole_doc_by_name(name_or_stem: str, base_folder: str) -> Tuple[str, List[str]]:
    name_or_stem = name_or_stem.lower().strip()
    candidates = [p for p in iter_files(base_folder) if name_or_stem in os.path.basename(p).lower()]
    texts = []
    for p in candidates:
        try:
            texts.append(read_whole_file_from_disk(p))
        except Exception as e:
            texts.append(f"[Error reading {os.path.basename(p)}: {e}]")
    return ("\n\n".join(t for t in texts if t.strip()) or ""), candidates

# ---------------- LLM & Chain

def make_llm_and_chain(vs: VectorStoreType):
    llm = make_llm(st.session_state["claude_model"], float(st.session_state["temperature"]))
    chain = make_chain(vs, llm, int(st.session_state["top_k"]))
    return llm, chain


_SLASH_SET_RE = re.compile(r"/set\s+(.+)$", re.IGNORECASE)
_KV_RE = re.compile(r"(\w+)\s*=\s*([\w\.\-]+)")


def _apply_settings_kv(s: str) -> str:
    changed = []
    for k, v in _KV_RE.findall(s):
        try:
            if k in {"top_k", "chat_height"}:
                st.session_state[k] = int(v)
            elif k in {"temperature"}:
                st.session_state[k] = float(v)
            elif k in {"claude_model"}:
                st.session_state[k] = v
            elif k in {"chunk_size", "chunk_overlap"}:
                cfg = st.session_state.get("chunk_cfg", ChunkingConfig())
                if k == "chunk_size":
                    cfg.chunk_size = int(v)
                else:
                    cfg.chunk_overlap = int(v)
                st.session_state["chunk_cfg"] = cfg
            changed.append(f"{k}→{st.session_state[k]}")
        except Exception:
            pass
    return ", ".join(changed) if changed else "No changes applied."


def handle_user_input(query: str, vs: Optional[VectorStoreType]):
    st.session_state.setdefault("messages", [])

    # Slash: /set key=val ... (update settings)
    mset = _SLASH_SET_RE.search(query)
    if mset:
        applied = _apply_settings_kv(mset.group(1))
        st.session_state["messages"].append({"role": "assistant", "content": f"Applied settings: {applied}"})
        st.rerun(); return

    st.session_state["messages"].append({"role": "user", "content": query})

    # Full-document commands
    m = re.match(r"^\s*(read|open|show)\s+(.+)$", query, flags=re.IGNORECASE)
    if m:
        target = m.group(2).strip().strip('"').strip("'")
        full_text, files = read_whole_doc_by_name(target, st.session_state["base_folder"])
        if not files:
            st.session_state["messages"].append({"role": "assistant", "content": f"Couldn't find a file containing “{target}” in the Knowledge Base folder."})
            st.rerun(); return

        if len(full_text) > 8000:
            try:
                llm, _ = make_llm_and_chain(vs or FAISS.from_texts([""], _make_embeddings()))
                summary = llm.predict(f"Summarize the following document concisely, focusing on key facts and numbers:\n\n{full_text[:180000]}")
                reply = f"**Full-document summary for:** {', '.join(Path(p).name for p in files)}\n\n{summary}"
            except Exception as e:
                reply = f"Loaded the full document but failed to summarize: {e}\n\n--- RAW BEGIN ---\n{full_text[:20000]}\n--- RAW TRUNCATED ---"
        else:
            reply = f"**Full document content:**\n\n{full_text}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun(); return

    if GREETING_RE.match(query):
        st.session_state["messages"].append({"role": "assistant", "content": "Hello"})
        st.rerun(); return

    if vs is None:
        st.session_state["messages"].append({"role": "assistant", "content": "Vector store unavailable. Ensure the FAISS index exists."})
        st.rerun(); return

    t0 = time.time()
    try:
        _, chain = make_llm_and_chain(vs)
        with st.spinner("Querying Claude with RAG..."):
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "Not found in Knowledge Base."
            sources = result.get("source_documents", []) or []
        citation_block = build_citation_block(sources, kb_root=st.session_state.get("base_folder"))
        msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
    except Exception as e:
        msg = f"RAG error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.rerun()

# ---------------- Main (no sidebar; Azure KB → local; auto-index; chat)

def main():
    # 1) Hardcoded defaults
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

    # 2) Sync local KB with Azure (source of truth)
    kb_dir, sync_label = sync_kb_from_azure_if_needed()
    st.session_state["base_folder"] = str(kb_dir)
    st.session_state["collection_name"] = f"kb-{stable_hash(str(kb_dir))}"

    # 3) Top header + compact controls
    st.markdown("### 💬 Chat with Forecast360")
    st.markdown(f"<div class='status-inline'><b>KB Sync:</b> {sync_label}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1], vertical_alignment="center")
    with c1:
        hero_status = st.container()
        vs = auto_index_if_needed(status_placeholder=hero_status)
    with c2:
        if st.button("Reindex KB", use_container_width=True):
            # Force reindex by clearing signature and last-time
            st.session_state.pop("_kb_last_sig", None)
            st.session_state["_kb_last_index_ts"] = 0
    with c3:
        st.session_state["chat_height"] = st.number_input("Chat height (px)", 420, 1200, st.session_state.get("chat_height", 560), step=20)
        
    # with c4:
    #     counts = st.session_state.get("_kb_last_counts", {})
    #     st.markdown(
    #         f"<div class='small-note'>Files: <b>{counts.get('files','–')}</b> · Docs: <b>{counts.get('docs','–')}</b> · Chunks: <b>{counts.get('chunks','–')}</b></div>",
    #         unsafe_allow_html=True,
    #     )

    # 4) Auto-index
    # hero_status = st.container()
    # vs = auto_index_if_needed(status_placeholder=hero_status)

    # 5) Chat UI (no sidebar)
    st.session_state.setdefault("messages", [{"role": "assistant", "content": "Hi! Ask me anything about Forecast360."}])
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    chat_area = st.container(height=int(st.session_state.get("chat_height", 560)), border=False)
    with chat_area:
        render_chat_history()
    user_text = st.chat_input("Type your question… ")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)


if __name__ == "__main__":
    main()
