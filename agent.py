# agent.py  (Drop-in replacement)
# Author: Amitesh Jha | iSoft
# Forecast360 AI Agent — RAG + App-State Awareness (Weaviate v4)

from __future__ import annotations
import os, re, json, itertools, random, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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

WEAVIATE_URL       = _sget("weaviate", "url", "")
WEAVIATE_API_KEY   = _sget("weaviate", "api_key", "")
COLLECTION_NAME    = _sget("weaviate", "collection", "Forecast360").strip()  # REQUIRED
TEXT_PROP_OVERRIDE = _sget("weaviate", "text_property", None)
SRC_PROP_OVERRIDE  = _sget("weaviate", "source_property", None)

EMB_MODEL_NAME     = _sget("rag", "embed_model", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT      = int(_sget("rag", "top_k", 8))
MAX_SENTS_DEFAULT  = int(_sget("rag", "max_sents", 6))

ANTHROPIC_MODEL    = _sget("anthropic", "model", "claude-sonnet-4-5")
ANTHROPIC_KEY      = _sget("anthropic", "api_key")
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
        "You help users understand, operate, and optimize Forecast360 — for data ingestion, feature engineering, "
        "model training, validation, forecasting, and visualization of time-series data.\n\n"
        "Persona:\n"
        "- Speak as “I/me/my”. Be analytical, precise, supportive.\n"
        "- Explain clearly: technical but approachable.\n"
        "- Be grounded: never invent. If info is missing, respond exactly: “Insufficient Context.”\n"
        "- Avoid URLs or speculation.\n\n"
        "Scope:\n"
        "- Forecast360 modules, architecture, connectors, pipelines, dashboards.\n"
        "- Algorithms (ARIMA/SARIMA/Prophet/TBATS/XGBoost/LightGBM/TFT etc.).\n"
        "- Metrics (RMSE/MAE/MASE/MAPE/sMAPE/R² etc.).\n"
        "- Azure integrations (Blob/Databricks/Synapse/ADF), versioning, CV/leaderboards.\n"
        "- App state: uploaded data, chosen columns, resampling, transforms, forecasts.\n\n"
        "Tone:\n"
        "- Friendly, authoritative, action-oriented. Use bullets or short paragraphs.\n"
    ),
    "retrieval_template": (
        "Answer strictly using Forecast360’s verified context.\n\n"
        "Question:\n{question}\n\n"
        "Knowledge Base Context (Weaviate):\n{kb}\n\n"
        "Local Session Context (app state):\n{local}\n\n"
        "Rules:\n"
        "- Use Forecast360 vocabulary (modules/pipelines/models/metrics/integrations).\n"
        "- Be concise, factual, structured.\n"
        "- If info is missing, respond only: “Insufficient Context.”"
    ),
    "query_rewrite": (
        "Rewrite the question in precise Forecast360/time-series terms. Only return the rewritten query.\n\n"
        "Examples:\n"
        "- 'How does it work?' → 'How does Forecast360 perform end-to-end time-series forecasting?'\n"
        "- 'Which models do you use?' → 'Which forecasting algorithms are implemented in Forecast360?'\n"
        "- 'How accurate are forecasts?' → 'How does Forecast360 compute and present forecast accuracy?'\n"
    ),
    "loading": [
        "Analyzing your forecasting query…",
        "Retrieving the most relevant Forecast360 insights…",
        "Evaluating models and metrics from the knowledge base…",
        "Synthesizing a data-driven answer from Forecast360 context…",
        "Connecting to Forecast360’s model intelligence engine…",
    ],
}


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
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
    )
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

# ========= Local App-State Context (lets agent answer about “my data/forecast”) =========

def _fmt_int(n): 
    try: return f"{int(n):,}"
    except: return str(n)

def _brief_df(df: pd.DataFrame, name: str) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"{name}: none"
    rows, cols = df.shape
    head_cols = ", ".join([str(c) for c in list(df.columns)[:8]]) + ("…" if df.shape[1] > 8 else "")
    return f"{name}: shape=({_fmt_int(rows)}×{_fmt_int(cols)}); columns=[{head_cols}]"

def _local_context() -> Dict[str, Any]:
    s = st.session_state
    ctx = {}

    # Uploaded data & choices
    df = s.get("uploaded_df")
    ctx["uploaded_df"] = _brief_df(df, "uploaded_df")
    ctx["source_name"] = s.get("source_name")
    ctx["date_col"]    = s.get("date_col")
    ctx["target_col"]  = s.get("target_col")
    ctx["resample"]    = s.get("resample_freq")
    ctx["transform"]   = s.get("target_transform")
    ctx["seasonal_m"]  = s.get("seasonal_m")
    ctx["fill_method"] = s.get("fill_method")

    # Cleaning stats
    ctx["raw_rows"]    = s.get("raw_rows")
    ctx["raw_cols"]    = s.get("raw_cols")
    ctx["clean_rows"]  = s.get("clean_rows")
    ctx["clean_cols"]  = s.get("clean_cols")
    ctx["drop_breakdown"] = s.get("drop_breakdown")

    # CV / metrics
    ctx["cv_folds"]    = s.get("cv_folds")
    ctx["cv_horizon"]  = s.get("cv_horizon")
    ctx["cv_gap"]      = s.get("cv_gap")
    ctx["cv_metrics"]  = s.get("cv_metrics")

    # Model selections
    for k in ["sel_ARMA","sel_ARIMA","sel_ARIMAX","sel_SARIMA","sel_SARIMAX",
              "sel_AutoARIMA","sel_HWES","sel_Prophet","sel_TBATS","sel_XGB","sel_LGBM","sel_TFT"]:
        ctx[k] = s.get(k)

    # Optional results (if your pipeline stores them)
    # e.g., 'forecast_df', 'cv_leaderboard', 'best_model_name', 'residuals_df'
    for k in ["forecast_df", "cv_leaderboard", "best_model_name", "residuals_df", "resample_summary"]:
        if isinstance(s.get(k), pd.DataFrame):
            ctx[k] = _brief_df(s.get(k), k)
        else:
            ctx[k] = s.get(k)

    return ctx

def _summarize_local_context(ctx: Dict[str, Any], max_lines: int = 20) -> str:
    items = []
    add = lambda k, v: items.append(f"- {k}: {v}") if v not in [None, "", {}] else None
    for k in ["uploaded_df","source_name","date_col","target_col","resample",
              "transform","seasonal_m","fill_method","raw_rows","raw_cols",
              "clean_rows","clean_cols","drop_breakdown","cv_folds","cv_horizon",
              "cv_gap","cv_metrics","best_model_name","forecast_df","cv_leaderboard",
              "residuals_df","resample_summary"]:
        add(k, ctx.get(k))
    if len(items) > max_lines:
        items = items[:max_lines] + ["- …"]
    return "\n".join(items)

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

def _pick_text_and_source_fields(client: Any, class_name: str) -> Tuple[str, Optional[str]]:
    if TEXT_PROP_OVERRIDE:
        return TEXT_PROP_OVERRIDE, SRC_PROP_OVERRIDE
    try:
        coll = client.collections.get(class_name)
        cfg = coll.config.get()
        props = getattr(cfg, "properties", []) or []
        names = [getattr(p, "name", "") for p in props]
        text_field = None
        for cand in ["text","content","body","chunk","passage","document","value"]:
            if cand in names: text_field = cand; break
        if not text_field:
            for p in props:
                dts = [str(dt).lower() for dt in (getattr(p, "data_type", []) or [])]
                if any("text" in dt for dt in dts):
                    text_field = getattr(p, "name", None); break
        source_field = None
        for cand in ["source","url","page","path","file","document","uri","source_path"]:
            if cand in names: source_field = cand; break
        return (text_field or "text", source_field)
    except Exception:
        return ("text", None)

def _search_weaviate(client: Any, class_name: str, text_field: str, source_field: Optional[str],
                     embedder: SentenceTransformer, query: str, k: int) -> List[Dict[str,Any]]:
    want = max(k, 24)
    coll = client.collections.get(class_name)
    qv = embedder.encode([query], normalize_embeddings=True)[0].astype("float32").tolist()

    # 1) near-vector
    try:
        res = coll.query.near_vector(near_vector=qv, limit=want,
                                     return_metadata=wvc.query.MetadataQuery(distance=True))
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits: return hits
    except Exception: pass

    # 2) hybrid
    try:
        res = coll.query.hybrid(query=query, vector=qv, alpha=0.6, limit=want,
                                return_metadata=wvc.query.MetadataQuery(score=True))
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits: return hits
    except Exception: pass

    # 3) bm25
    try:
        res = coll.query.bm25(query=query, limit=want,
                              return_metadata=wvc.query.MetadataQuery(score=True))
        hits = _collect_from_objects(res.objects, text_field, source_field)
        if hits: return hits
    except Exception: pass

    return []

def retrieve(client: Any, class_name: str, query: str, k: int) -> List[Dict[str,Any]]:
    model = _load_embed_model(EMB_MODEL_NAME)
    text_field, source_field = _pick_text_and_source_fields(client, class_name)
    prelim = _search_weaviate(client, class_name, text_field, source_field, model, query, k)
    prelim = [x for x in prelim if x.get("text")]
    if not prelim:
        return []
    # diversity
    uniq, seen = [], set()
    for c in prelim:
        key = re.sub(r"\W+", "", c["text"].lower())[:300]
        if key not in seen:
            seen.add(key); uniq.append(c)
    return _apply_reranker(query, uniq, k)

# ============================ LLM Synthesis & Query Rewrite ============================

def _anthropic_answer(question: str, kb_blocks: List[str], local_block: str) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
    try:
        import anthropic as _anthropic
    except Exception:
        return None
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        kb="\n".join(f"- {c}" for c in kb_blocks) if kb_blocks else "(none)",
        local=local_block or "(none)",
    )
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=PROMPTS["system"],
            max_tokens=700,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        if not text or text.lower().strip().replace(".", "") == "insufficient context":
            return None
        return text
    except Exception:
        return None

def _rewrite_query(orig_question: str) -> str:
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

# ============================ Router ============================

_LOCAL_HINTS = [
    r"\b(my|uploaded|current)\s+(data|dataset|file|csv|excel|json|parquet|xml)\b",
    r"\b(my|current)\s+(forecast|predictions?|results?)\b",
    r"\b(columns?|date(_|\s*)col|target(_|\s*)col|resampl|transform|mape|rmse|mae|smape|mase|metrics?)\b",
    r"\b(model|algorithm)s?\b.*\b(selected|enabled|used)\b",
]

def _needs_local_context(q: str) -> bool:
    ql = q.lower()
    if any(re.search(pat, ql) for pat in _LOCAL_HINTS):
        return True
    # short heuristic: if user asks “what is my …”
    if re.search(r"\b(my|our)\b.*\b(data|forecast|metric|model)\b", ql):
        return True
    return False

# ============================ Agent Core ============================

def _format_answer(text: str, sources: List[str]) -> str:
    # sources pretty list
    srcs = [s for s in sources if s]
    if srcs:
        src_block = "\n".join(f"- {s}" for s in srcs[:8])
        return f"{text}\n\n**Sources**\n{src_block}"
    return text

class Forecast360Agent:
    def __init__(self, client: Any, class_name: str):
        self.client = client
        self.class_name = class_name
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def _answer(self, user_q: str, *, top_k: int, max_sents: int, use_local: bool) -> str:
        # 1) rewrite for domain
        q_precise = _rewrite_query(user_q)

        # 2) gather contexts
        kb_hits: List[Dict[str,Any]] = retrieve(self.client, self.class_name, q_precise, top_k)
        kb_texts = [h["text"] for h in kb_hits]
        kb_sources = [h.get("source","") for h in kb_hits]

        local_ctx = _local_context() if use_local or _needs_local_context(q_precise) else {}
        local_block = _summarize_local_context(local_ctx) if local_ctx else ""

        if not kb_texts and not local_ctx:
            return "Insufficient Context."

        # 3) sentence selection for answerability
        excerpts = _best_extract_sentences(q_precise, kb_texts, max_pick=max_sents) if kb_texts else []

        # 4) LLM grounded synthesis
        llm_ans = _anthropic_answer(q_precise, excerpts, local_block)
        if llm_ans:
            return _format_answer(llm_ans, kb_sources)

        # 5) Fallback extractive summary
        parts = []
        if excerpts:
            parts.append("**From the knowledge base:**")
            parts += [f"- {ex}" for ex in excerpts]
        if local_block:
            parts.append("\n**From your current session:**")
            for line in local_block.splitlines():
                parts.append(line)
        if not parts:
            return "Insufficient Context."
        return _format_answer("\n".join(parts), kb_sources)

    def respond(self, user_q: str, *, top_k: int, max_sents: int, use_local: bool) -> str:
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! I’m the Forecast360 AI Agent. Ask me about models, pipelines, metrics, forecasts, "
                    "or your currently uploaded data. I’ll answer using the knowledge base and your session context.")

        try:
            return self._answer(user_q, top_k=top_k, max_sents=max_sents, use_local=use_local)
        except Exception:
            return ("Sorry, something went wrong while processing your request. "
                    "Please try again in a moment.")

# ============================ Streamlit UI ============================

def _render_agent_core(set_config: bool = False):
    if set_config:
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

    # Admin knobs
    with st.sidebar:
        st.subheader("🔧 Agent Settings")
        top_k = st.number_input("Top-K (KB results)", min_value=3, max_value=30, value=TOP_K_DEFAULT, step=1)
        max_sents = st.number_input("Max sentences (KB extract)", min_value=3, max_value=15, value=MAX_SENTS_DEFAULT, step=1)
        use_local = st.checkbox("Use Local Session Context", value=True,
                                help="When on, the agent can answer about your uploaded data, chosen columns, forecasts, and metrics.")
        st.caption("These settings affect only the AI Agent responses.")

    # Style
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

    # KB Refresh
    with st.container():
        st.markdown("""
        <style>
        .kb-row { display: inline-flex; align-items: center; gap: 8px; margin-top: 6px; }
        .kb-caption { font-size: 0.9rem; color: var(--text-color-secondary,#6b6f76); white-space: nowrap; }
        .kb-refresh-btn button { width: 26px; height: 26px; min-width: 26px; min-height: 26px; padding: 0; border-radius: 6px; font-size: 14px; line-height: 1; }
        </style>
        <div class="kb-row">
          <span class="kb-caption">Knowledge Base: Weaviate ← Azure Blob (refresh to re-sync latest files)</span>
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

    # Connect Weaviate
    if "f360_client" not in st.session_state:
        with st.spinner("Connecting to the Forecast360 knowledge base…"):
            try:
                st.session_state["f360_client"] = _connect_weaviate()
            except Exception as e:
                st.error(f"⚠️ Weaviate configuration error: {e}")
                st.stop()

    agent = Forecast360Agent(st.session_state["f360_client"], COLLECTION_NAME)

    # Initial bot message
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role":"assistant",
            "content":"Hello! I'm your Forecast360 AI Agent. Ask about models, CV metrics, pipelines, or your currently uploaded data/forecast."
        })

    # History
    for m in st.session_state["messages"]:
        avatar = ASSISTANT_ICON if m["role"]=="assistant" else (USER_ICON if os.path.exists(USER_ICON) else "👤")
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # Pending queued query (optional)
    if "pending_query" in st.session_state:
        pq = st.session_state.pop("pending_query")
        st.session_state["messages"].append({"role":"user","content":pq})
        with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
            st.markdown(pq)
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(pq, top_k=top_k, max_sents=max_sents, use_local=use_local)
            st.markdown(reply)
        st.session_state["messages"].append({"role":"assistant","content":reply})
        st.rerun()

    # Chat input
    user_q = st.chat_input("Ask me anything about Forecast360 or your current session…", key="chat_box")
    if user_q:
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user", avatar=(USER_ICON if os.path.exists(USER_ICON) else "👤")):
            st.markdown(user_q)
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(user_q, top_k=top_k, max_sents=max_sents, use_local=use_local)
            st.markdown(reply)
        st.session_state["messages"].append({"role":"assistant","content":reply})
        st.rerun()

# Public API for embedding inside your tabbed app
def render_agent():
    _render_agent_core(set_config=False)

# Standalone runner
def run():
    _render_agent_core(set_config=True)

if __name__ == "__main__":
    run()
