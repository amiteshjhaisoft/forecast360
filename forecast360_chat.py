# Author: Amitesh Jha | iSoft
# Forecast360 AI Agent — Strict RAG (Weaviate collection: Forecast360) → (optional) Claude
# Weaviate v3 client compatible

from __future__ import annotations
import os, re, json, itertools
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

# --- Weaviate v3 client ---
try:
    import weaviate
except ImportError:
    weaviate = None  # we'll error nicely later


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


# ============================ Helpers ============================

def _normalize_weaviate_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw.rstrip("/")

def _connect_weaviate() -> Any:
    if not weaviate:
        raise RuntimeError("Missing dependency: install 'weaviate-client>=3.25,<4'.")
    url = _normalize_weaviate_url(WEAVIATE_URL)
    if not url:
        raise RuntimeError("Set [weaviate].url in .streamlit/secrets.toml")
    auth = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    client = weaviate.Client(
        url=url,
        auth_client_secret=auth,
        timeout_config=(15, 120),
        startup_period=45,
    )
    # Health check & collection presence
    try:
        schema = client.schema.get() or {}
        classes = [c.get("class") for c in schema.get("classes", [])]
        if COLLECTION_NAME not in classes:
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found in schema.")
    except Exception as e:
        raise RuntimeError(f"Weaviate not reachable/healthy: {e}")
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


# ============= Introspective schema helpers (v3 schema dict) =============

def _pick_text_and_source_fields(client: Any, class_name: str) -> Tuple[str, Optional[str]]:
    """
    Infer main text field and optional source/url-like field from v3 schema.
    Prefers: text/content/body/chunk (text) and url/source/page/path/file/document/uri (source).
    """
    try:
        sch = client.schema.get() or {}
        for c in sch.get("classes", []):
            if (c.get("class") or "").strip() == class_name:
                props = c.get("properties", []) or []
                names = [p.get("name","") for p in props]

                text_field = None
                for n in ["text","content","body","chunk","passage","document","value"]:
                    if n in names:
                        text_field = n; break
                if not text_field:
                    for p in props:
                        dts = [dt.lower() for dt in (p.get("dataType") or [])]
                        if "text" in dts:
                            text_field = p.get("name"); break

                source_field = None
                for n in ["url","source","page","path","file","document","uri"]:
                    if n in names:
                        source_field = n; break

                return (text_field or "text", source_field)
    except Exception:
        pass
    return "text", "url"


# ============================ Retrieval (v3) ============================

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

def _search_weaviate(client: Any, class_name: str, text_field: str, source_field: Optional[str],
                     embedder: SentenceTransformer, query: str, k: int) -> List[Dict[str,Any]]:
    """
    v3 query flow: near-vector → hybrid → bm25
    Note: we request _additional.distance and convert to a similarity ~ (1 - distance).
    """
    qv = embedder.encode([query], normalize_embeddings=True)[0].astype("float32").tolist()
    fields = [text_field]
    if source_field:
        fields.append(source_field)

    base = client.query.get(class_name, fields).with_additional(["distance"])

    def _run(build):
        try:
            res = build.do()
            return res["data"]["Get"].get(class_name, [])
        except Exception:
            return []

    want = max(k, 24)

    # 1) near-vector
    hits = _run(base.with_near_vector({"vector": qv}).with_limit(want))
    # 2) hybrid (vector + keyword)
    if not hits:
        hits = _run(base.with_hybrid(query=query, vector=qv, alpha=0.6).with_limit(want))
    # 3) bm25 fallback
    if not hits:
        hits = _run(base.with_bm25(query=query).with_limit(want))

    out = []
    for h in hits:
        add = h.get("_additional", {}) or {}
        dist = add.get("distance", None)
        # similarity rough mapping for cosine distance (sim ≈ 1 - dist)
        score = None
        if isinstance(dist, (int, float)):
            score = 1.0 - float(dist)
        text_val = h.get(text_field, "") or ""
        src_val  = h.get(source_field, "") if source_field else ""
        if not str(text_val).strip():
            continue
        out.append({"text": str(text_val), "source": str(src_val or ""), "score": score if score is not None else 0.0})
    return out

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
    # rerank (optional)
    return _apply_reranker(query, uniq, k)


# ============================ LLM Synthesis (optional) ============================

COMPANY_RULES = """
- Persona: You are the Forecast360 AI Agent, a courteous, precise representative speaking as “We”.
- Grounding: Use ONLY the provided context. If a fact is not present, say “Insufficient Context.”
- No speculation or extrapolation. No generic promises.
- Tone: Polite, concise, professional. No URLs in the answer.
"""

SYNTHESIS_PROMPT_TEMPLATE = """
Answer the user’s question STRICTLY from the context below.
If the context does not explicitly contain the answer, reply only with: “Insufficient Context.”

Question: {question}

Context:
---
{ctx}
---
"""

def _anthropic_answer(question: str, context_blocks: List[str]) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
    try:
        import anthropic as _anthropic
    except Exception:
        return None
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, ctx="\n".join(f"- {c}" for c in context_blocks))
    try:
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=COMPANY_RULES,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role":"user","content":prompt}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        if not text:
            return None
        if text.lower().strip().replace(".", "") == "insufficient context":
            return None
        return text
    except Exception:
        return None


# ============================ Agent ============================

class Forecast360Agent:
    def __init__(self, client: Any, class_name: str):
        self.client = client
        self.class_name = class_name
        if "messages" not in st.session_state: st.session_state["messages"] = []

    def _answer(self, user_q: str) -> str:
        hits = retrieve(self.client, self.class_name, user_q, TOP_K)
        if not hits:
            return ("We couldn’t find that detail in the Forecast360 knowledge base. "
                    "Please rephrase or ask about a different topic.")
        texts = [h["text"] for h in hits]
        excerpts = _best_extract_sentences(user_q, texts, max_pick=6) or texts[:6]

        llm_ans = _anthropic_answer(user_q, excerpts)
        if llm_ans:
            return llm_ans

        parts = ["Here’s what we can confirm from our knowledge base:"]
        parts += [f"- {ex}" for ex in excerpts]
        return "\n".join(parts)

    def respond(self, user_q: str) -> str:
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! I’m the Forecast360 AI Agent. Ask me anything about Forecast360—"
                    "features, setup, data flows, dashboards, or architecture—and I’ll answer using our knowledge base.")
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
        with st.spinner("Analyzing your query…"):
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
        with st.spinner("Analyzing your query…"):
            reply = agent.respond(user_q)
        st.markdown(reply)
    st.session_state["messages"].append({"role":"assistant","content":reply})
    st.rerun()

# Bottom-right compact actions removed (no crawler/cache to refresh/clear)
