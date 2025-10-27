# Author: Amitesh Jha | iSoft
# Forecast360 AI Agent — Strict RAG (Weaviate collection: Forecast360)
# Uses Weaviate v4 client + Sentence Transformers + Anthropic (Claude)
from __future__ import annotations

import os
import re
import json
import itertools
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# --- Embeddings (for query vectors) ---
from sentence_transformers import SentenceTransformer

# --- Optional Cross-Encoder reranker (graceful if missing) ---
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# --- Weaviate v4 client (Collections API) ---
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.client import WeaviateClient  # Explicitly import the type

# Note: 'azure_sync_weaviate' dependency is assumed to exist.
from azure_sync_weaviate import sync_from_azure  # type: ignore

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================ Configuration & Secrets ============================

def _sget(section: str, key: str, default: Any = None) -> Any:
    """Safely get a secret from st.secrets."""
    try:
        return st.secrets[section].get(key, default)  # type: ignore
    except (KeyError, AttributeError):
        return default

# Environment/Secrets Configuration
WEAVIATE_URL        = _sget("weaviate", "url", "")
WEAVIATE_API_KEY    = _sget("weaviate", "api_key", "")
COLLECTION_NAME     = _sget("weaviate", "collection", "Forecast360").strip()

EMB_MODEL_NAME      = _sget("rag", "embed_model", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K               = int(_sget("rag", "top_k", 8))

# Anthropic (Claude) Configuration
ANTHROPIC_MODEL     = _sget("anthropic", "model", "claude-sonnet-4-5")
ANTHROPIC_KEY       = _sget("anthropic", "api_key")
if ANTHROPIC_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY
    os.environ.setdefault("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

# UI Configuration
ASSISTANT_ICON      = _sget("ui", "assistant_icon", "assets/forecast360.png")
USER_ICON           = _sget("ui", "user_icon", "assets/avatar.png")
PAGE_ICON           = _sget("ui", "page_icon", "assets/forecast360.png")
PAGE_TITLE          = _sget("ui", "page_title", "Forecast360 AI Agent")

# ============================ Prompts (Domain-Focused) ============================

PROMPTS = {
    "system": (
        "You are the **Forecast360 AI Agent**, a professional decision-intelligence and time-series assistant "
        "developed by iSoft ANZ Pvt Ltd.\n\n"
        "### Knowledge Source Policy\n"
        "- Answer **STRICTLY** from the Forecast360 Weaviate Knowledge Base (collection: **{collection_name}**).\n"
        "- The KB may include **text reports, tables, images (OCR text), audio/video transcripts, charts, and metadata**.\n"
        "- Use only the retrieved KB content; do **not** use external sources, prior memory, or UI state.\n\n"
        "### Retrieval & Matching\n"
        "- Search across **all modalities** and ground every claim in retrieved chunks.\n"
        "- Match by exact keywords, synonyms, acronyms, abbreviations, plural/singular, and morphological variants.\n"
        "- Normalize generic terms to Forecast360 vocabulary "
        "(e.g., error → RMSE/MAE/MAPE; model → ARIMA/SARIMA/Prophet/LightGBM/TFT; "
        "pipeline → ingestion→prep→training→validation→forecast→reporting).\n\n"
        "### Style & Constraints\n"
        "- Speak as **'I/me/my'**. Be analytical, precise, supportive, and concise.\n"
        "- Use short paragraphs or **≤6 bullet points**. **Never invent facts.**\n"
        "- If the KB is insufficient, reply **EXACTLY**: 'Insufficient Context.'"
    ),
    "retrieval_template": (
        "Use **only** the Forecast360 knowledge base chunks below to answer.\n\n"
        "### User Question\n{question}\n\n"
        "### Retrieved KB Chunks (multimodal)\n{kb}\n\n"
        "Write a **concise, analytical** answer that directly addresses the question by synthesizing information "
        "from any relevant modality (text, tables, chart OCR, audio/video transcripts, logs, metadata). "
        "Maintain Forecast360 terminology and precision.\n\n"
        "If the KB is insufficient, reply **EXACTLY**: 'Insufficient Context.'"
    ),
    "query_rewrite": (
        "Rewrite the user's question into a single **multimodal retrieval query** optimized for the Forecast360 KB. "
        "Include core concepts plus Forecast360-specific synonyms, abbreviations, and related terms for text, tables, images (OCR), "
        "and transcripts.\n"
        "Return **ONLY** the rewritten query."
    ),
    "loading": [
        "Analyzing your multimodal query…",
        "Retrieving insights across text, tables, OCR, and transcripts…",
        "Evaluating models, metrics, and logs from the knowledge base…",
        "Synthesizing a grounded, data-driven answer…",
        "Connecting to Forecast360’s decision-intelligence context…",
    ],
}

COMPANY_RULES = PROMPTS["system"]
SYNTHESIS_PROMPT_TEMPLATE = PROMPTS["retrieval_template"]

# ============================ Weaviate & Model Initialization ============================

@st.cache_resource(show_spinner=False)
def get_weaviate_client() -> WeaviateClient:
    """Connects to Weaviate and verifies the collection exists."""
    if not WEAVIATE_URL:
        raise RuntimeError("Configuration Error: Set [weaviate].url in secrets.")

    auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )

    try:
        coll = client.collections.get(COLLECTION_NAME)
        coll.config.get()
    except Exception as e:
        client.close()
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found or client unreachable: {e}") from e

    logger.info(f"Successfully connected to Weaviate cluster at {WEAVIATE_URL}")
    return client

@st.cache_resource(show_spinner=False)
def load_embed_model(name: str) -> SentenceTransformer:
    logger.info(f"Loading embedding model: {name}")
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_reranker():
    if CrossEncoder is None:
        logger.warning("CrossEncoder not installed. Reranking disabled.")
        return None
    try:
        rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder reranker loaded.")
        return rr
    except Exception as e:
        logger.error(f"Error loading CrossEncoder reranker: {e}")
        return None

EMBEDDER = load_embed_model(EMB_MODEL_NAME)
RERANKER = load_reranker()

# ============================ Schema & Retrieval Helpers (v4) ============================

def _pick_text_and_source_fields(client: WeaviateClient, class_name: str) -> Tuple[str, Optional[str]]:
    forced_text = _sget("weaviate", "text_property", None)
    forced_src  = _sget("weaviate", "source_property", None)
    if forced_text:
        return forced_text, forced_src

    text_field: Optional[str] = None
    source_field: Optional[str] = None
    try:
        coll = client.collections.get(class_name)
        cfg = coll.config.get()
        props = getattr(cfg, "properties", []) or []
        names = [getattr(p, "name", "") for p in props]

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

        for cand in ["source", "url", "page", "path", "file", "document", "uri", "source_path"]:
            if cand in names:
                source_field = cand
                break
    except Exception as e:
        logger.warning(f"Schema introspection failed for '{class_name}': {e}. Using defaults.")

    return (text_field or "text", source_field)

def _sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _best_extract_sentences(question: str, texts: List[str], max_pick: int = 6) -> List[str]:
    q_terms = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sents = list(itertools.chain.from_iterable(_sent_split(t) for t in texts))
    scored = []
    for s in sents:
        base = sum(s.lower().count(t) for t in q_terms)
        scored.append((base, -len(s), s))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked = [s for sc, _, s in scored if sc > 0] or [s for sc, _, s in scored]
    seen, out = set(), []
    for s in picked:
        ss = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s)).strip()
        k = ss.lower()[:280]
        if len(ss) >= 24 and k not in seen:
            seen.add(k); out.append(ss)
            if len(out) >= max_pick:
                break
    return out

def _apply_reranker(query: str, candidates: List[Dict[str,Any]], topk: int) -> List[Dict[str,Any]]:
    if not candidates or RERANKER is None:
        return candidates[:topk]
    scores = []
    BATCH_SIZE = 64
    try:
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i+BATCH_SIZE]
            pairs = [(query, c["text"]) for c in batch]
            scores.extend(RERANKER.predict(pairs))  # type: ignore
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: (x.get("rerank_score", 0.0), x.get("score", 0.0)), reverse=True)
    except Exception as e:
        logger.error(f"Reranker failed, falling back to original score: {e}")
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return candidates[:topk]

def _collect_from_objects(objects: List[wvc.DataItem[Dict, wvc.Metadata]],
                          text_field: str, source_field: Optional[str]) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    if not objects:
        return out
    for o in objects:
        props = o.properties or {}
        text_val = str(props.get(text_field, "") or "")
        if not text_val.strip():
            continue
        src_val = str(props.get(source_field, "") or "") if source_field else ""
        md = o.metadata
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

def _search_weaviate(client: WeaviateClient, class_name: str, text_field: str,
                     source_field: Optional[str], query: str, k: int) -> List[Dict[str,Any]]:
    WANT = max(k, 24)
    coll = client.collections.get(class_name)

    qv = EMBEDDER.encode([query], normalize_embeddings=True)[0].astype("float32")
    qv_list = qv.tolist()

    # 1) Near-vector
    try:
        res = coll.query.near_vector(
            near_vector=qv_list,
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(distance=True, score=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception as e:
        logger.debug(f"NearVector search failed: {e}")

    # 2) Hybrid
    try:
        res = coll.query.hybrid(
            query=query,
            vector=qv_list,
            alpha=0.6,
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception as e:
        logger.debug(f"Hybrid search failed: {e}")

    # 3) BM25
    try:
        res = coll.query.bm25(
            query=query,
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits:
            return hits
    except Exception as e:
        logger.debug(f"BM25 search failed: {e}")

    return []

def retrieve(client: WeaviateClient, class_name: str, query: str, k: int = TOP_K) -> List[Dict[str,Any]]:
    text_field, source_field = _pick_text_and_source_fields(client, class_name)
    prelim = _search_weaviate(client, class_name, text_field, source_field, query, k)
    prelim = [x for x in prelim if x.get("text")]
    if not prelim:
        return []
    uniq, seen = [], set()
    for c in prelim:
        key = re.sub(r"\W+", "", c["text"].lower())[:280]
        if key not in seen:
            seen.add(key); uniq.append(c)
    return _apply_reranker(query, uniq, k)

# ============================ LLM Synthesis & Query Rewrite ============================

def _anthropic_call(system_prompt: str, user_prompt: str, max_tokens: int) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
    try:
        import anthropic as _anthropic
    except ImportError:
        logger.error("Anthropic library not installed. Cannot call Claude.")
        return None
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        if not text:
            return None
        if text.lower().strip().replace(".", "") == "insufficient context":
            return None
        return text
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        return None

def _anthropic_answer(question: str, context_blocks: List[str]) -> Optional[str]:
    context_str = "\n".join(f"- {c}" for c in context_blocks)
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, kb=context_str)
    return _anthropic_call(system_prompt=PROMPTS["system"], user_prompt=prompt, max_tokens=500)

def _rewrite_query(orig_question: str) -> str:
    if not orig_question.strip():
        return orig_question
    prompt = PROMPTS["query_rewrite"] + "\n\nUser question:\n" + orig_question
    rewritten_query = _anthropic_call(system_prompt=PROMPTS["system"], user_prompt=prompt, max_tokens=120)
    return rewritten_query or orig_question

# ============================ Agent Core ============================

class Forecast360Agent:
    def __init__(self, client: WeaviateClient, class_name: str):
        self.client = client
        self.class_name = class_name
        st.session_state.setdefault("messages", [])

    def _answer(self, user_q: str) -> str:
        q_precise = _rewrite_query(user_q)
        logger.info(f"Original Q: '{user_q}' | Rewritten Q: '{q_precise}'")
        hits = retrieve(self.client, self.class_name, q_precise, TOP_K)
        if not hits:
            return ("We couldn’t find that detail in the Forecast360 knowledge base. "
                    "Please rephrase or ask about a different topic.")
        texts = [h["text"] for h in hits]
        excerpts = _best_extract_sentences(q_precise, texts, max_pick=6) or texts[:6]
        llm_ans = _anthropic_answer(q_precise, excerpts)
        if llm_ans:
            return llm_ans
        parts = ["I cannot synthesize a complete answer, but here are relevant details from the knowledge base:"]
        parts += [f"- {ex}" for ex in excerpts]
        return "\n".join(parts)

    def respond(self, user_q: str) -> str:
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! I am Forecast360 AI Agent. Ask me about **forecasting models**, **pipelines**, "
                    "**accuracy metrics**, or our **Azure integrations** — I’ll answer from the knowledge base.")
        try:
            return self._answer(user_q)
        except Exception as e:
            logger.error(f"Agent response failed due to exception: {e}")
            return ("Sorry, something went wrong while processing your request. "
                    "Please try again in a moment. (Error details logged).")

# ============================ UI Helpers ============================

def _display_kb_refresh_area():
    st.markdown("""
    <style>
    .kb-row { display: inline-flex; align-items: center; gap: 8px; margin-top: 6px; }
    .kb-caption { font-size: 0.9rem; color: var(--text-color-secondary,#6b6f76); white-space: nowrap; }
    .kb-refresh-btn button { width: 26px; height: 26px; min-width: 26px; min-height: 26px; padding: 0; border-radius: 6px; font-size: 14px; line-height: 1; }
    </style>
    <div class="kb-row">
      <span class="kb-caption">Knowledge Base: Weaviate ← Azure Blob (refresh to re-sync latest files)</span>
      <span class="kb-refresh-btn">
    """, unsafe_allow_html=True)

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
                    f"Done. Files: {stats.get('processed_files', 0)} | "
                    f"Chunks: {stats.get('inserted_chunks', 0)} | "
                    f"Cleared: {stats.get('cleared_sources', 0)} | "
                    f"Skipped: {stats.get('skipped_files', 0)}"
                )
                st.toast("Knowledge base refreshed.")
            except Exception as e:
                st.error(f"KB refresh failed. Ensure 'azure_sync_weaviate' is configured: {e}")

    st.markdown("</span></div>", unsafe_allow_html=True)

# -------------------- 🔹 Fragment: only chat reruns on input 🔹 --------------------

@st.fragment
def _chat_fragment(agent: Forecast360Agent):
    # Initial assistant message once
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Hello! I’m your Forecast360 AI Agent. How can I help today?"
        })

    # Render history
    for m in st.session_state["messages"]:
        avatar = ASSISTANT_ICON if m["role"] == "assistant" else (USER_ICON if os.path.exists(USER_ICON) else "👤")
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # Chat input (fragment reruns on submit, not the whole page)
    user_q = st.chat_input("Ask me...", key="chat_box")

    # Either externally provided pending query or new user input
    query_to_process = st.session_state.pop("pending_query", user_q)

    if query_to_process:
        # Echo user
        st.session_state["messages"].append({"role": "user", "content": query_to_process})
        with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
            st.markdown(query_to_process)

        # Answer
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(query_to_process)
            st.markdown(reply)

        st.session_state["messages"].append({"role": "assistant", "content": reply})

        # IMPORTANT: no st.rerun() — the fragment re-render is enough

# ============================ Page Renderers ============================

def _render_agent_core(set_config: bool = False):
    if set_config:
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

    st.markdown("""
    <style>
    .stButton>button { border-radius: 10px; border-color: #007bff; color: #007bff; }
    .stButton>button:hover { background-color: #007bff; color: white; }
    .bottom-actions { margin-top: 6px; }
    </style>
    """, unsafe_allow_html=True)

    # Header (static; not in fragment)
    c1, c2 = st.columns([1, 8], vertical_alignment="center")
    with c1: st.image(ASSISTANT_ICON, width=80)
    with c2:
        st.markdown("### Forecast360 AI Agent")
        st.markdown('<span>Created by Amitesh Jha | iSoft</span>', unsafe_allow_html=True)

    # KB Refresh Area (static; not in fragment)
    _display_kb_refresh_area()

    # Connection (one-time; not in fragment)
    if "f360_client" not in st.session_state:
        with st.spinner("Connecting to the Forecast360 knowledge base…"):
            try:
                st.session_state["f360_client"] = get_weaviate_client()
            except RuntimeError as e:
                st.error(f"⚠️ Connection Error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"⚠️ An unexpected connection error occurred: {e}")
                st.stop()

    agent = Forecast360Agent(st.session_state["f360_client"], COLLECTION_NAME)

    # 🔹 Only this fragment reruns during chat
    _chat_fragment(agent)

def render_agent():
    """Public API for embedding the agent inside a parent app/tab."""
    _render_agent_core(set_config=False)

def run():
    """Standalone runner for `streamlit run agent.py`."""
    _render_agent_core(set_config=True)

if __name__ == "__main__":
    run()
