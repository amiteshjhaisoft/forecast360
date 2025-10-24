# Author: Amitesh Jha | iSoft
# Forecast360 AI Agent — Strict RAG (Weaviate collection: Forecast360) → (optional) Claude
# Weaviate v4 client compatible

from __future__ import annotations
import os, re, json, itertools, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

def _sget(section: str, key: str, default: Any = None) -> Any:
    try:
        return st.secrets[section].get(key, default)  # type: ignore
    except Exception:
        return default

WEAVIATE_URL     = _sget("weaviate", "url", "")
WEAVIATE_API_KEY = _sget("weaviate", "api_key", "")
COLLECTION_NAME  = _sget("weaviate", "collection", "Forecast360").strip()  # REQUIRED

EMB_MODEL_NAME   = _sget("rag", "embed_model", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K            = int(_sget("rag", "top_k", 8))

ANTHROPIC_MODEL  = _sget("anthropic", "model", "claude-sonnet-4-5")
ANTHROPIC_KEY    = _sget("anthropic", "api_key")
if ANTHROPIC_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY
    os.environ.setdefault("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

ASSISTANT_ICON = _sget("ui", "assistant_icon", "assets/forecast360.png")
USER_ICON      = _sget("ui", "user_icon", "assets/avatar.png")
PAGE_ICON      = _sget("ui", "page_icon", "assets/forecast360.png")
PAGE_TITLE     = _sget("ui", "page_title", "Forecast360 AI Agent")

# ============================ Prompts (Domain-Focused) ============================

PROMPTS = {
    "system": (
        "You are the Forecast360 AI Agent — a professional Decision-Intelligence and Time-Series Forecasting "
        "assistant developed by iSoft ANZ Pvt Ltd.\n\n"
        "Purpose:\n"
        "You help users understand, operate, and optimize Forecast360 — a platform for data ingestion, feature "
        "engineering, model training, validation, forecasting, and visualization of time-series data.\n\n"
        "Persona:\n"
        "- You speak as “We” or “Our team”.\n"
        "- You are analytical, precise, and supportive.\n"
        "- You explain concepts clearly — technical but approachable.\n"
        "- You never guess; if information is not found in context, say “Insufficient Context.”\n"
        "- Avoid URLs or speculative statements.\n"
        "- Maintain a professional, solution-oriented tone.\n\n"
        "Knowledge Scope:\n"
        "- Forecast360’s modules, architecture, connectors, pipelines, and dashboards.\n"
        "- Time-series forecasting algorithms (ARIMA, SARIMA, Prophet, TBATS, XGBoost, LightGBM, TFT, etc.).\n"
        "- Forecast accuracy metrics (RMSE, MAE, MAPE, R², etc.).\n"
        "- Data preparation, chunking, and embedding techniques.\n"
        "- Forecast360’s integration with Azure services (Blob, Databricks, Synapse, ADF).\n"
        "- Model versioning, retraining, and leaderboard evaluation.\n"
        "- Visualization and decision intelligence workflows.\n\n"
        "Tone:\n"
        "- Friendly yet authoritative.\n"
        "- Use bullet points and concise explanations for technical topics."
    ),
    "retrieval_template": (
        "Answer the question strictly using Forecast360’s verified knowledge base context.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{ctx}\n\n"
        "Rules:\n"
        "- Use the Forecast360 domain vocabulary (models, pipelines, dashboards, metrics, integrations).\n"
        "- Be concise, factual, and professional.\n"
        "- Use structured lists or short paragraphs for clarity.\n"
        "- If information is missing, respond only: “Insufficient Context.”"
    ),
    "query_rewrite": (
        "Reinterpret the user’s question in terms of Forecast360’s time-series forecasting context.\n\n"
        "Examples:\n"
        "- 'How does it work?' → 'How does Forecast360 perform time-series forecasting end-to-end?'\n"
        "- 'Which models do you use?' → 'Which forecasting algorithms are implemented in Forecast360?'\n"
        "- 'How accurate are forecasts?' → 'How does Forecast360 measure and display forecast accuracy?'\n\n"
        "Return only the rewritten, precise query."
    ),
    "loading": [
        "Analyzing your forecasting query…",
        "Retrieving the most relevant Forecast360 insights…",
        "Evaluating models and metrics from the knowledge base…",
        "Synthesizing a data-driven answer from Forecast360 context…",
        "Connecting to Forecast360’s model intelligence engine…",
    ],
}

# Backward-compatible minimal prompts (kept for fallback)
COMPANY_RULES = PROMPTS["system"]

SYNTHESIS_PROMPT_TEMPLATE = PROMPTS["retrieval_template"]

# ============================ Helpers ============================

def _connect_weaviate():
    if not WEAVIATE_URL:
        raise RuntimeError("Set [weaviate].url in secrets.")

    auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )

    # quick check: collection must exist (use() is the idiomatic v4 call)
    try:
        _ = client.collections.use(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found or client unreachable: {e}")

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

# ============= Introspective schema helpers (v4 Collections) =============

def _pick_text_and_source_fields(client: Any, class_name: str) -> Tuple[str, Optional[str]]:
    """
    Infer the main text field and an optional source/url-like field using v4 collection config.
    Prefers names: text/content/body/chunk/passsage/document/value for text;
    url/source/page/path/file/document/uri for source.
    """
    text_field = None
    source_field = None
    try:
        coll = client.collections.get(class_name)
        cfg = coll.config.get()
        props = getattr(cfg, "properties", []) or []
        names = [getattr(p, "name", "") for p in props]

        for cand in ["text","content","body","chunk","passage","document","value"]:
            if cand in names:
                text_field = cand
                break
        if not text_field:
            for p in props:
                dts = [str(dt).lower() for dt in (getattr(p, "data_type", []) or [])]
                if any("text" in dt for dt in dts):
                    text_field = getattr(p, "name", None)
                    if text_field: break

        for cand in ["url","source","page","path","file","document","uri"]:
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
        ss = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s)).strip()
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

def _anthropic_answer(question: str, context_blocks: List[str]) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
    try:
        import anthropic as _anthropic
    except Exception:
        return None
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        ctx="\n".join(f"- {c}" for c in context_blocks)
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
    """Use Claude to rewrite the query into a precise Forecast360/TS context; fallback to original."""
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

        # 2) Retrieve
        hits = retrieve(self.client, self.class_name, q_precise, TOP_K)
        if not hits:
            return ("We couldn’t find that detail in the Forecast360 knowledge base. "
                    "Please rephrase or ask about a different topic.")
        texts = [h["text"] for h in hits]
        excerpts = _best_extract_sentences(q_precise, texts, max_pick=6) or texts[:6]

        # 3) LLM Synthesis (strictly grounded)
        llm_ans = _anthropic_answer(q_precise, excerpts)
        if llm_ans:
            return llm_ans

        # 4) Fallback extractive summary
        parts = ["Here’s what we can confirm from our knowledge base:"]
        parts += [f"- {ex}" for ex in excerpts]
        return "\n".join(parts)

    def respond(self, user_q: str) -> str:
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! We’re the Forecast360 AI Agent. Ask us about forecasting models, pipelines, dashboards, "
                    "accuracy metrics, or Azure integrations — we’ll answer using our knowledge base.")
        try:
            return self._answer(user_q)
        except Exception:
            return ("Sorry, something went wrong while processing your request. "
                    "Please try again in a moment.")

# ============================ Streamlit UI (same look & feel) ============================

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
    st.markdown('<span>Created by Amitesh Jha | iSoft</span>', unsafe_allow_html=True)

# Connect to Weaviate
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
        "role":"assistant",
        "content":"Hello! I’m your Forecast360 AI Agent. How can I help today?"
    })

# Render chat history
for m in st.session_state["messages"]:
    avatar = ASSISTANT_ICON if m["role"]=="assistant" else (USER_ICON if os.path.exists(USER_ICON) else "👤")
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Process queued query (if any)
if "pending_query" in st.session_state:
    pq = st.session_state.pop("pending_query")
    st.session_state["messages"].append({"role":"user","content":pq})
    with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
        st.markdown(pq)
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        # keep UI unchanged; optionally rotate a richer loading phrase
        loading_msg = random.choice(PROMPTS["loading"])
        with st.spinner(loading_msg):
            reply = agent.respond(pq)
        st.markdown(reply)
    st.session_state["messages"].append({"role":"assistant","content":reply})
    st.rerun()

# Chat input
user_q = st.chat_input("Ask me...", key="chat_box")
if user_q:
    st.session_state["messages"].append({"role":"user","content":user_q})
    with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
        st.markdown(user_q)
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        loading_msg = random.choice(PROMPTS["loading"])
        with st.spinner(loading_msg):
            reply = agent.respond(user_q)
        st.markdown(reply)
    st.session_state["messages"].append({"role":"assistant","content":reply})
    st.rerun()

# --- KB Refresh (Azure Blob -> Weaviate) ---
with st.container():
    # scope styles to this block only
    st.markdown("""
    <style>
    #kb-refresh .stButton>button {
        width: 28px; height: 28px;
        min-width: 28px; min-height: 28px;
        padding: 0; border-radius: 6px;
        font-size: 14px; line-height: 1;
    }
    </style>
    <div id="kb-refresh"></div>
    """, unsafe_allow_html=True)

    # col_a, col_b = st.columns([1, 0.12])
    # with col_a:
        st.caption("Knowledge Base: Weaviate ← Azure Blob folder (refresh to re-sync latest files).")
    # with col_b:
        # icon-only, smallest possible; tooltip via help
        if st.button("🔄", key="refresh_kb", help="Refresh KB", use_container_width=False):
            with st.spinner("Refreshing…"):
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

