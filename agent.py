# Author: Amitesh Jha | iSoft
# Forecast360 AI Agent — Strict RAG (Weaviate collection only)
# Weaviate v4 client compatible

from __future__ import annotations
import os, re, itertools, random
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# --- Embeddings (for query vectors) ---
from sentence_transformers import SentenceTransformer

# --- Optional Cross-Encoder reranker (graceful if missing) ---
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# --- Weaviate v4 client (Collections API) ---
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth, AdditionalConfig, Timeout

from azure_sync_weaviate import sync_from_azure


# ============================ Configuration ============================

def _sget(section: str, key: str) -> str:
    """Return a trimmed secret or empty string if missing (we'll fail fast below)."""
    try:
        return str(st.secrets[section][key]).strip()  # type: ignore
    except Exception:
        return ""

WEAVIATE_URL     = _sget("weaviate", "url")
WEAVIATE_API_KEY = _sget("weaviate", "api_key")
COLLECTION_NAME  = _sget("weaviate", "collection")  # REQUIRED (no fallback)

EMB_MODEL_NAME   = _sget("rag", "embed_model") or "sentence-transformers/all-MiniLM-L6-v2"
try:
    TOP_K        = int(_sget("rag", "top_k") or 8)
except Exception:
    TOP_K        = 8

ANTHROPIC_MODEL  = _sget("anthropic", "model") or "claude-3-5-sonnet-20240620"
ANTHROPIC_KEY    = _sget("anthropic", "api_key")
if ANTHROPIC_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY
    os.environ.setdefault("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

ASSISTANT_ICON = _sget("ui", "assistant_icon") or "assets/forecast360.png"
USER_ICON      = _sget("ui", "user_icon") or "assets/avatar.png"
PAGE_ICON      = _sget("ui", "page_icon") or "assets/forecast360.png"
PAGE_TITLE     = _sget("ui", "page_title") or "Forecast360 AI Agent"

# Fail fast if critical secrets are missing
if not WEAVIATE_URL:
    raise RuntimeError("Missing [weaviate].url in Streamlit secrets.")
if not COLLECTION_NAME:
    raise RuntimeError("Missing [weaviate].collection in Streamlit secrets.")


# ============================ Prompts (Weaviate-only, synonym-aware) ============================

PROMPTS = {
    "system": (
        "I am the Forecast360 AI Agent — a professional decision-intelligence and time-series forecasting assistant "
        "developed by iSoft ANZ Pvt Ltd.\n\n"
        "Purpose:\n"
        "I answer strictly from the Forecast360 Weaviate knowledge base (documents, captions/alt text, OCR from images, "
        "ASR transcripts from audio/video, slide notes, file names, metadata). I do not use external sources or UI state.\n\n"
        "Retrieval & Matching:\n"
        "- Match by exact keywords and by synonyms, acronyms, abbreviations, plural/singular, and morphological variants.\n"
        "- Map generic terms to Forecast360 vocabulary where possible (e.g., error → RMSE/MAE/MAPE; model → ARIMA/SARIMA/"
        "Prophet/TBATS/XGBoost/LightGBM/TFT; pipeline → ingestion→prep→training→validation→forecast→reporting).\n\n"
        "Persona & Style:\n"
        "- I speak as “I/me/my”. I am analytical, precise, supportive, and concise.\n"
        "- I prefer structured bullets and short paragraphs and never invent facts.\n"
        "- If information is missing, I respond exactly: “Insufficient Context.”\n"
        "- I avoid external URLs or speculative content."
    ),
    "retrieval_template": (
        "Answer strictly using the Forecast360 Weaviate knowledge base content below.\n\n"
        "User Question:\n{question}\n\n"
        "Knowledge Base Context (Weaviate — text/captions/OCR/ASR/metadata):\n{kb}\n\n"
        "Instructions:\n"
        "- Map question terms to synonyms/acronyms/variants and align to Forecast360 terminology before answering.\n"
        "- Use only facts supported by the KB context. If insufficient, reply only: “Insufficient Context.”\n"
        "- Be concise and structured. When naming artifacts (models/metrics/files/pipeline steps), use the exact names from context."
    ),
    "query_rewrite": (
        "Rewrite the user’s question into a single precise Forecast360/time-series retrieval query optimized for both "
        "semantic and lexical search. Include the core concept plus common synonyms/abbreviations/acronyms and variants "
        "in a compact form (use OR or commas). Return only the rewritten query."
    ),
    "loading": [
        "Analyzing your query and related synonyms…",
        "Searching Forecast360 knowledge across documents, images, and transcripts…",
        "Ranking results and aligning terminology to Forecast360…",
        "Synthesizing a grounded answer from the retrieved context…",
        "Verifying details from the knowledge base…",
    ],
}

COMPANY_RULES = PROMPTS["system"]
SYNTHESIS_PROMPT_TEMPLATE = PROMPTS["retrieval_template"]


# ============================ Helpers ============================

def _connect_weaviate():
    auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )

    # Ensure the collection exists and is queryable
    try:
        coll = client.collections.use(COLLECTION_NAME)  # raises if missing
        # Light health check: tiny BM25 (succeeds even if empty)
        _ = coll.query.bm25(query="__healthcheck__", limit=1)
    except Exception as e:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found or not queryable: {e}")

    return client

@st.cache_resource(show_spinner=False)
def _load_embed_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def _load_reranker():
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None

RERANKER = _load_reranker()


# ============= Schema helpers (v4 Collections) =============

def _pick_text_and_source_fields(client: Any, class_name: str) -> Tuple[str, Optional[str]]:
    forced_text = _sget("weaviate", "text_property")
    forced_src  = _sget("weaviate", "source_property")
    if forced_text:
        return forced_text, (forced_src or None)

    text_field: Optional[str] = None
    source_field: Optional[str] = None
    try:
        coll = client.collections.get(class_name)
        cfg = coll.config.get()
        props = getattr(cfg, "properties", []) or []
        names = [getattr(p, "name", "") for p in props]

        # likely text properties
        for cand in ["text", "content", "body", "chunk", "passage", "document", "value"]:
            if cand in names:
                text_field = cand
                break

        if not text_field:
            for p in props:
                dts = [str(dt).lower() for dt in (getattr(p, "data_type", []) or [])]
                if any("text" in dt for dt in dts):
                    text_field = getattr(p, "name", None)
                    if text_field:
                        break

        # likely source/url properties
        for cand in ["source", "url", "page", "path", "file", "document", "uri", "source_path"]:
            if cand in names:
                source_field = cand
                break
    except Exception:
        pass

    return (text_field or "text", source_field)


# ============================ Retrieval (v4) ============================

def _sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _best_extract_sentences(question: str, texts: List[str], max_pick: int = 6) -> List[str]:
    q_terms = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sents = list(itertools.chain.from_iterable(_sent_split(t) for t in texts))
    scored = []
    for s in sents:
        base = sum(s.lower().count(t) for t in q_terms)
        scored.append((base, len(s), s))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    picked = [s for sc, ln, s in scored if sc > 0] or [s for sc, ln, s in scored]
    # clean + dedupe
    seen, out = set(), []
    for s in picked:
        ss = re.sub(r"\s+", " ", s).strip()
        k = ss.lower()
        if len(ss) >= 24 and k not in seen:
            seen.add(k); out.append(ss)
        if len(out) >= max_pick:
            break
    return out

def _apply_reranker(query: str, candidates: List[Dict[str,Any]], topk: int) -> List[Dict[str,Any]]:
    if not candidates or RERANKER is None:
        return candidates[:topk]
    scores = []
    B = 64
    try:
        for i in range(0, len(candidates), B):
            pairs = [(query, c["text"]) for c in candidates[i:i+B]]
            scores.extend(RERANKER.predict(pairs))
        for c, s in zip(candidates, scores):
            c["rerank"] = float(s)
        candidates.sort(key=lambda x: (x.get("rerank", 0.0), x.get("score", 0.0)), reverse=True)
    except Exception:
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates[:topk]

def _collect_from_objects(objs, text_field: str, source_field: Optional[str]) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    if not objs:
        return out
    for o in objs:
        props = getattr(o, "properties", {}) or {}
        text_val = str(props.get(text_field, "") or "")
        if not text_val.strip():
            continue
        src_val = str(props.get(source_field, "") or "") if source_field else ""
        md = getattr(o, "metadata", None)
        score_val = 0.0
        if md is not None:
            dist = getattr(md, "distance", None)
            if isinstance(dist, (int, float)):
                score_val = 1.0 - float(dist)
            sc = getattr(md, "score", None)
            if isinstance(sc, (int, float)) and sc > score_val:
                score_val = float(sc)
        out.append({"text": text_val, "source": src_val, "score": score_val})
    return out

def _search_weaviate(client: Any, class_name: str, text_field: str, source_field: Optional[str],
                     embedder: SentenceTransformer, query: str, k: int) -> List[Dict[str,Any]]:
    """
    v4 flow: near-vector → hybrid → bm25
    """
    want = max(k, 24)
    coll = client.collections.get(class_name)

    # Prepare query vector
    qv = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    qv_list = qv.tolist()

    # 1) near-vector
    try:
        res = coll.query.near_vector(
            near_vector=qv_list,
            limit=want,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception:
        pass

    # 2) hybrid (vector + keyword)
    try:
        res = coll.query.hybrid(
            query=query,
            vector=qv_list,
            alpha=0.6,
            limit=want,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception:
        pass

    # 3) bm25 fallback
    try:
        res = coll.query.bm25(
            query=query,
            limit=want,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception:
        pass

    return []

def retrieve(client: Any, class_name: str, query: str, k: int = TOP_K) -> List[Dict[str,Any]]:
    model = _load_embed_model(EMB_MODEL_NAME)
    text_field, source_field = _pick_text_and_source_fields(client, class_name)
    prelim = _search_weaviate(client, class_name, text_field, source_field, model, query, k)
    prelim = [x for x in prelim if x.get("text")]
    if not prelim:
        return []
    # simple diversity: cap near-duplicates
    uniq, seen = [], set()
    for c in prelim:
        key = re.sub(r"\W+", "", c["text"].lower())[:280]
        if key not in seen:
            seen.add(key); uniq.append(c)
    return _apply_reranker(query, uniq, k)


# ============================ LLM Synthesis & Query Rewrite ============================

def _anthropic_answer(question: str, kb_blocks: List[str]) -> Optional[str]:
    """Synthesize an answer strictly from KB blocks. If model absent/unavailable, return None."""
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
    try:
        import anthropic as _anthropic
    except Exception:
        return None

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        kb="\n".join(f"- {c}" for c in kb_blocks)
    )
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=PROMPTS["system"],
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        if not text:
            return None
        if text.lower().strip().replace(".", "") == "insufficient context":
            return None
        return text
    except Exception:
        return None

def _rewrite_query(orig_question: str) -> str:
    """Use Claude to rewrite the query into a precise Forecast360/TS retrieval query; fallback to original."""
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key or not orig_question.strip():
        return orig_question
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=PROMPTS["system"],
            max_tokens=120,
            temperature=0.0,
            messages=[{"role": "user", "content": PROMPTS["query_rewrite"] + "\n\nUser question:\n" + orig_question}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        return text or orig_question
    except Exception:
        return orig_question


# ============================ Agent ============================

class Forecast360Agent:
    def __init__(self, client: Any, class_name: str):
        self.client = client
        self.class_name = class_name
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def _answer(self, user_q: str) -> str:
        # 1) Query rewrite (domain-focused)
        q_precise = _rewrite_query(user_q)

        # 2) Retrieve strictly from Weaviate
        hits = retrieve(self.client, self.class_name, q_precise, TOP_K)
        if not hits:
            return "Insufficient Context."
        texts = [h["text"] for h in hits]
        # Pick the most relevant sentences for grounding (keeps synthesis concise)
        excerpts = _best_extract_sentences(q_precise, texts, max_pick=6) or texts[:6]

        # 3) LLM Synthesis (strictly grounded to KB only)
        llm_ans = _anthropic_answer(q_precise, excerpts)
        if llm_ans:
            return llm_ans

        # 4) Fallback extractive summary (still KB-only)
        parts = ["Here’s what I can confirm from the knowledge base:"]
        parts += [f"- {ex}" for ex in excerpts]
        return "\n".join(parts)

    def respond(self, user_q: str) -> str:
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! I’m the Forecast360 AI Agent. Ask me about models, pipelines, metrics, forecasts, "
                    "or KB content. I answer strictly from the Forecast360 knowledge base.")
        try:
            return self._answer(user_q)
        except Exception:
            return ("Sorry, something went wrong while processing your request. "
                    "Please try again in a moment.")


# ============================ Streamlit UI ============================

def _render_agent_core(set_config: bool = False):
    # If embedding inside a parent app (tabs), do not re-set page config.
    if set_config:
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

    st.markdown("""
    <style>
    .stButton>button { border-radius: 10px; border-color: #007bff; color: #007bff; }
    .stButton>button:hover { background-color: #007bff; color: white; }
    .bottom-actions .stButton>button {
        width: 42px; min-width: 42px; height: 42px; min-height: 42px;
        padding: 0; border-radius: 12px; font-size: 18px;
    }
    .bottom-actions { margin-top: 6px; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([1, 8], vertical_alignment="center")
    with c1: st.image(ASSISTANT_ICON, width=80)
    with c2:
        st.markdown("### Forecast360 AI Agent")
        st.caption(f"Knowledge Base Collection: **{COLLECTION_NAME}**")
        st.markdown('<span>Created by Amitesh Jha | iSoft</span>', unsafe_allow_html=True)

    # --- KB Refresh (Azure Blob -> Weaviate) ---
    with st.container():
        st.markdown("""
        <style>
        .kb-row { display: inline-flex; align-items: center; gap: 8px; margin-top: 6px; }
        .kb-caption { font-size: 0.9rem; color: var(--text-color-secondary,#6b6f76); white-space: nowrap; }
        .kb-refresh-btn button {
            width: 26px; height: 26px; min-width: 26px; min-height: 26px;
            padding: 0; border-radius: 6px; font-size: 14px; line-height: 1;
        }
        </style>
        <div class="kb-row">
          <span class="kb-caption">Weaviate ← Azure Blob (refresh to re-sync latest files)</span>
          <span class="kb-refresh-btn">""", unsafe_allow_html=True)

        if st.button("🔄", key="refresh_kb", help="Refresh KB", use_container_width=False):
            with st.spinner("Refreshing knowledge base…"):
                try:
                    stats = sync_from_azure(
                        st_secrets=st.secrets,
                        collection_name=COLLECTION_NAME,
                        container_key="container",
                        prefix_key="prefix",
                        embed_model_key=("rag", "embed_model"),
                        delete_before_upsert=True,
                        max_docs=None,
                    )
                    st.success(
                        f"Done. Files: {stats['processed_files']} | "
                        f"Chunks: {stats['inserted_chunks']} | "
                        f"Cleared: {stats['cleared_sources']} | "
                        f"Skipped: {stats['skipped_files']}"
                        + (f" | Local embeddings: {stats['used_embed_model']}" if stats.get("vectorizer_none") else "")
                    )
                    st.toast("Knowledge base refreshed.")
                except Exception as e:
                    st.error(f"KB refresh failed: {e}")

        st.markdown("</span></div>", unsafe_allow_html=True)

    # --- Weaviate Connection and Agent Setup ---
    if "f360_client" not in st.session_state:
        with st.spinner("Connecting to the Forecast360 knowledge base…"):
            try:
                st.session_state["f360_client"] = _connect_weaviate()
            except Exception as e:
                st.error(f"⚠️ Weaviate configuration error: {e}")
                st.stop()

    agent = Forecast360Agent(st.session_state["f360_client"], COLLECTION_NAME)

    # Initial assistant message (only once)
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Hello! I’m your Forecast360 AI Agent. I answer strictly from the Weaviate knowledge base. How can I help?"
        })

    # Render chat history
    for m in st.session_state["messages"]:
        avatar = ASSISTANT_ICON if m["role"] == "assistant" else (USER_ICON if os.path.exists(USER_ICON) else "👤")
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # If the parent app wants to queue a question (e.g., from another widget)
    if "pending_query" in st.session_state:
        pq = st.session_state.pop("pending_query")
        st.session_state["_suspend_gs"] = True  # keep GS tab quiet this run

        st.session_state["messages"].append({"role": "user", "content": pq})
        with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
            st.markdown(pq)
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(pq)
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        st.session_state["_suspend_gs"] = False  # clear suspension

    # Chat input — avoid extra reruns; suspend GS during this turn
    user_q = st.chat_input("Ask me about Forecast360…", key="chat_box")
    if user_q:
        st.session_state["_suspend_gs"] = True

        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
            st.markdown(user_q)
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(user_q)
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        st.session_state["_suspend_gs"] = False  # clear suspension


# Public API for embedding inside your tabbed app
def render_agent():
    _render_agent_core(set_config=False)

# Standalone runner (useful for `streamlit run agent.py`)
def run():
    _render_agent_core(set_config=True)

if __name__ == "__main__":
    run()
