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
from weaviate.client import WeaviateClient # Explicitly import the type

# Note: 'azure_sync_weaviate' dependency is assumed to exist.
from azure_sync_weaviate import sync_from_azure # type: ignore

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
        "I am the **Forecast360 AI Agent**—a professional decision-intelligence and time-series forecasting assistant "
        "developed by iSoft ANZ Pvt Ltd.\n\n"
        "**Knowledge Base & Source Rule:** I answer **STRICTLY** from the Forecast360 Weaviate knowledge base (documents, captions/alt text, OCR, ASR transcripts, slide notes, file names, metadata). I **MUST NOT** use external sources or UI state.\n\n"
        "**Retrieval & Matching:**\n"
        "- Match by exact keywords, synonyms, acronyms, abbreviations, plural/singular, and morphological variants.\n"
        "- **Map generic terms** to Forecast360 vocabulary (e.g., error → RMSE/MAE/MAPE; model → ARIMA/Prophet/TFT; pipeline → ingestion→prep→training→validation→forecast→reporting).\n\n"
        "**Style & Constraint:**\n"
        "- Speak as **'I/me/my'**. Be analytical, precise, supportive, and concise.\n"
        "- Use short **bullet points** (max 6 bullets total). Never invent facts.\n"
        "- If information is missing or the KB is truly insufficient, respond **EXACTLY** and **ONLY** with: 'Insufficient Context.'"
    ),
    "retrieval_template": (
        "Answer **STRICTLY** from the Forecast360 knowledge base chunks provided below.\n\n"
        "User Question:\n{question}\n\n"
        "KB Chunks:\n{kb}\n\n"
        "Write a **VERY CONCISE** answer using bullet points. The total number of bullets must be **MAX 5**, and each bullet must be **MAX 15 words**.\n"
        "**YOU MUST** use Forecast360 terminology. Each point must be directly grounded in the KB chunks.\n"
        "If the KB is insufficient to answer, reply **EXACTLY** and **ONLY** with: 'Insufficient Context.'\n"
        "End your valid answer with a line: **Sources: <comma-separated short source labels>**"
    ),
    "query_rewrite": (
        "Rewrite the user’s question into a single retrieval query optimized for both semantic and lexical search against the Forecast360 time-series DB.\n"
        "The rewritten query should include the core concept plus **Forecast360-specific** synonyms, abbreviations, and acronyms (use commas to separate terms).\n"
        "Example: 'what caused the jump in sales' -> 'anomaly detection, spike cause, demand surge, ingestion pipeline, data validation, forecast error'\n"
        "Return **ONLY** the rewritten query text."
    ),
    "anomaly_explanation_template": (
        "The user is asking for an explanation regarding an **anomaly, unexpected forecast, or significant error**.\n\n"
        "User Question:\n{question}\n\n"
        "KB Context (Anomalies/Metadata/Model Logs):\n{kb}\n\n"
        "Provide a concise, root-cause analysis using Forecast360 vocabulary. The response must identify the likely cause(s) found in the KB.\n"
        "* **Start with:** 'My analysis indicates the deviation is likely due to...'\n"
        "* List **1 to 3 primary causes** in separate bullet points (e.g., 'A holiday flag was missing in the ingestion pipeline.', 'The SARIMA model failed to capture the recent upward trend.', 'Data validation identified an outlier at T-3').\n"
        "If no definitive cause is mentioned in the KB, reply: 'Insufficient Context to determine the precise root cause.'"
    ),
    "model_comparison_template": (
        "The user is asking for a recommendation or comparison between different models/pipelines.\n\n"
        "User Question:\n{question}\n\n"
        "KB Context (Model Metadata/Validation Metrics):\n{kb}\n\n"
        "Recommend the **best-performing model/pipeline** based on the provided metrics and context.\n"
        "* **Model Recommendation:** State the recommended model/pipeline explicitly (e.g., 'I recommend the LightGBM pipeline.').\n"
        "* **Grounding:** List the **top two supporting metrics** (e.g., 'Lowest MAPE at 3.1%' and 'Highest R-squared on the validation set.').\n"
        "If the KB does not contain validation metrics for comparison, reply: 'Insufficient Context for comparative model recommendation.'"
    ),
    "loading": [
        "Analyzing your forecasting query…",
        "Retrieving the most relevant Forecast360 insights…",
        "Evaluating models and metrics from the knowledge base…",
        "Synthesizing a data-driven answer from Forecast360 context…",
        "Connecting to Forecast360’s model intelligence engine…",
    ],
}

# Aliases for internal use
COMPANY_RULES = PROMPTS["system"]
SYNTHESIS_PROMPT_TEMPLATE = PROMPTS["retrieval_template"]


# ============================ Weaviate & Model Initialization ============================

@st.cache_resource(show_spinner=False)
def get_weaviate_client() -> WeaviateClient:
    """Connects to Weaviate and verifies the collection exists."""
    if not WEAVIATE_URL:
        raise RuntimeError("Configuration Error: Set [weaviate].url in secrets.")

    auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

    # Connect to Weaviate (WeaviateClient is the new return type for v4)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )

    # Quick check: collection must exist
    try:
        # Use .get() and .config.get() to check existence in v4
        coll = client.collections.get(COLLECTION_NAME)
        coll.config.get()
    except Exception as e:
        client.close()
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found or client unreachable: {e}") from e

    logger.info(f"Successfully connected to Weaviate cluster at {WEAVIATE_URL}")
    return client


@st.cache_resource(show_spinner=False)
def load_embed_model(name: str) -> SentenceTransformer:
    """Load the Sentence Transformer model for embedding."""
    logger.info(f"Loading embedding model: {name}")
    return SentenceTransformer(name)


@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load the optional Cross-Encoder reranker."""
    if CrossEncoder is None:
        logger.warning("CrossEncoder not installed. Reranking disabled.")
        return None
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder reranker loaded.")
        return reranker
    except Exception as e:
        logger.error(f"Error loading CrossEncoder reranker: {e}")
        return None

# Global Model Instances
EMBEDDER = load_embed_model(EMB_MODEL_NAME)
RERANKER = load_reranker()


# ============================ Introspective Schema Helpers ============================

def _pick_text_and_source_fields(client: WeaviateClient, class_name: str) -> Tuple[str, Optional[str]]:
    """
    Picks the main text property and an optional source/url-like property.
    Prioritizes secrets override, then schema introspection.
    """
    # --- Secrets-based override (preferred if set) ---
    forced_text = _sget("weaviate", "text_property", None)
    forced_src  = _sget("weaviate", "source_property", None)
    if forced_text:
        return forced_text, forced_src

    # --- Heuristic fallback via schema inspection ---
    text_field: Optional[str] = None
    source_field: Optional[str] = None
    try:
        coll = client.collections.get(class_name)
        cfg = coll.config.get()
        # Ensure props is a list of property objects
        props = getattr(cfg, "properties", []) or []
        names = [getattr(p, "name", "") for p in props]

        # Prioritized content field names
        for cand in ["text", "content", "body", "chunk", "passage", "document", "value"]:
            if cand in names:
                text_field = cand
                break
        
        # Fallback to any 'text' datatype
        if not text_field:
            for p in props:
                dts = [str(dt).lower() for dt in (getattr(p, "data_type", []) or [])]
                if any("text" in dt for dt in dts):
                    text_field = getattr(p, "name", None)
                    if text_field:
                        break

        # Prioritized source/url field names
        for cand in ["source", "url", "page", "path", "file", "document", "uri", "source_path"]:
            if cand in names:
                source_field = cand
                break
    except Exception as e:
        logger.warning(f"Schema introspection failed for '{class_name}': {e}. Using defaults.")

    # Default to 'text' if no content field is found
    return (text_field or "text", source_field)


# ============================ Retrieval Helpers (v4) ============================

def _sent_split(text: str) -> List[str]:
    """Simple sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def _best_extract_sentences(question: str, texts: List[str], max_pick: int = 6) -> List[str]:
    """Extracts a few top sentences from chunks based on term overlap."""
    q_terms = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(w) > 3]
    sents = list(itertools.chain.from_iterable(_sent_split(t) for t in texts))
    scored = []
    
    for s in sents:
        # Score by term overlap
        base = sum(s.lower().count(t) for t in q_terms)
        # Prioritize high score, then shorter length (less noise)
        scored.append((base, -len(s), s))
        
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Filter to sentences with at least 1 match, or take all if no matches
    picked = [s for sc, _, s in scored if sc > 0] or [s for sc, _, s in scored]
    
    # Clean, dedupe, and enforce minimum length
    seen, out = set(), []
    for s in picked:
        # Remove URLs and extra whitespace
        ss = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s)).strip()
        # Dedupe key is the first 280 chars of the cleaned, lowercased sentence
        k = ss.lower()[:280]
        if len(ss) >= 24 and k not in seen:
            seen.add(k); out.append(ss)
            if len(out) >= max_pick:
                break
    return out

def _apply_reranker(query: str, candidates: List[Dict[str,Any]], topk: int) -> List[Dict[str,Any]]:
    """Applies the cross-encoder reranker if available."""
    if not candidates or RERANKER is None:
        return candidates[:topk]
    
    scores = []
    BATCH_SIZE = 64
    try:
        # Batch prediction for efficiency
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i+BATCH_SIZE]
            pairs = [(query, c["text"]) for c in batch]
            # Reranker returns a score for each pair
            scores.extend(RERANKER.predict(pairs)) # type: ignore
        
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s) # Add new score
        
        # Sort by rerank score (primary) and original score (secondary)
        candidates.sort(key=lambda x: (x.get("rerank_score", 0.0), x.get("score", 0.0)), reverse=True)
        
    except Exception as e:
        logger.error(f"Reranker failed, falling back to original score: {e}")
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
    return candidates[:topk]

def _collect_from_objects(objects: List[wvc.DataItem[Dict, wvc.Metadata]], 
                          text_field: str, source_field: Optional[str]) -> List[Dict[str,Any]]:
    """Extracts text, source, and score from Weaviate v4 DataItem objects."""
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
            # Weaviate v4 returns 'distance' for vector search, 'score' for hybrid/BM25
            dist = getattr(md, "distance", None)
            if isinstance(dist, (int, float)):
                # Convert distance (0=perfect match) to score (1=perfect match)
                score_val = 1.0 - float(dist)
            
            sc = getattr(md, "score", None)
            if isinstance(sc, (int, float)) and sc > score_val:
                score_val = float(sc)
                
        out.append({"text": text_val, "source": src_val, "score": score_val})
        
    return out

def _search_weaviate(client: WeaviateClient, class_name: str, text_field: str, 
                     source_field: Optional[str], query: str, k: int) -> List[Dict[str,Any]]:
    """
    Performs a cascading Weaviate search: NearVector -> Hybrid -> BM25.
    Returns the first successful result set.
    """
    WANT = max(k, 24) # Fetch more than K for better reranking/diversity
    coll = client.collections.get(class_name)

    # Prepare query vector
    qv = EMBEDDER.encode([query], normalize_embeddings=True)[0].astype("float32")
    qv_list = qv.tolist()
    
    search_hits: List[Dict[str, Any]] = []

    # 1) Near-vector (Pure semantic)
    try:
        res = coll.query.near_vector(
            near_vector=qv_list,
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(distance=True, score=True)
        )
        search_hits = _collect_from_objects(res.objects, text_field, source_field)
        if search_hits:
            logger.debug(f"Retrieved {len(search_hits)} via NearVector.")
            return search_hits
    except Exception as e:
        logger.debug(f"NearVector search failed: {e}")

    # 2) Hybrid (Vector + Keyword)
    try:
        res = coll.query.hybrid(
            query=query,
            vector=qv_list,
            alpha=0.6, # Prioritizes vector search slightly
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        search_hits = _collect_from_objects(res.objects, text_field, source_field)
        if search_hits:
            logger.debug(f"Retrieved {len(search_hits)} via Hybrid.")
            return search_hits
    except Exception as e:
        logger.debug(f"Hybrid search failed: {e}")

    # 3) BM25 fallback (Pure keyword)
    try:
        res = coll.query.bm25(
            query=query,
            limit=WANT,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        search_hits = _collect_from_objects(res.objects, text_field, source_field)
        if search_hits:
            logger.debug(f"Retrieved {len(search_hits)} via BM25.")
            return search_hits
    except Exception as e:
        logger.debug(f"BM25 search failed: {e}")

    logger.info("Weaviate search failed all methods or returned no results.")
    return []

def retrieve(client: WeaviateClient, class_name: str, query: str, k: int = TOP_K) -> List[Dict[str,Any]]:
    """Master retrieval function: Search -> Diversity Filter -> Rerank."""
    text_field, source_field = _pick_text_and_source_fields(client, class_name)
    
    prelim = _search_weaviate(client, class_name, text_field, source_field, query, k)
    prelim = [x for x in prelim if x.get("text")]
    
    if not prelim:
        return []

    # Simple diversity filter (remove near-duplicates based on first 280 chars)
    uniq, seen = [], set()
    for c in prelim:
        key = re.sub(r"\W+", "", c["text"].lower())[:280]
        if key not in seen:
            seen.add(key); uniq.append(c)

    # Rerank and select the final top K
    return _apply_reranker(query, uniq, k)


# ============================ LLM Synthesis & Query Rewrite ============================

def _anthropic_call(system_prompt: str, user_prompt: str, max_tokens: int) -> Optional[str]:
    """Generic function for calling Anthropic API."""
    key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_KEY
    if not key:
        return None
        
    try:
        import anthropic as _anthropic # Lazy import
    except ImportError:
        logger.error("Anthropic library not installed. Cannot call Claude.")
        return None
        
    try:
        client = _anthropic.Anthropic(api_key=key)
        
        # Claude 3 models use the 'messages' API
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        # Extract text content from the response
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        
        if not text:
            return None
            
        # Check for the explicit 'Insufficient Context' response
        if text.lower().strip().replace(".", "") == "insufficient context":
            return None
            
        return text
        
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        return None

def _anthropic_answer(question: str, context_blocks: List[str]) -> Optional[str]:
    """Generates the grounded answer using the retrieval template."""
    context_str = "\n".join(f"- {c}" for c in context_blocks)
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        kb=context_str
    )
    return _anthropic_call(
        system_prompt=PROMPTS["system"],
        user_prompt=prompt,
        max_tokens=500
    )

def _rewrite_query(orig_question: str) -> str:
    """Uses Claude to rewrite the query for better RAG retrieval; fallbacks to original."""
    if not orig_question.strip():
        return orig_question
        
    prompt = PROMPTS["query_rewrite"] + "\n\nUser question:\n" + orig_question
    
    rewritten_query = _anthropic_call(
        system_prompt=PROMPTS["system"],
        user_prompt=prompt,
        max_tokens=120 # Query rewrites should be short
    )
    
    return rewritten_query or orig_question


# ============================ Agent Class ============================

class Forecast360Agent:
    """Core RAG agent logic for handling user queries."""
    
    def __init__(self, client: WeaviateClient, class_name: str):
        self.client = client
        self.class_name = class_name
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def _answer(self, user_q: str) -> str:
        """The main RAG pipeline: Rewrite -> Retrieve -> Synthesize -> Fallback."""
        
        # 1) Query rewrite (domain-focused for better retrieval)
        q_precise = _rewrite_query(user_q)
        logger.info(f"Original Q: '{user_q}' | Rewritten Q: '{q_precise}'")
        
        # 2) Retrieve relevant chunks
        hits = retrieve(self.client, self.class_name, q_precise, TOP_K)
        
        if not hits:
            return ("We couldn’t find that detail in the Forecast360 knowledge base. "
                    "Please rephrase or ask about a different topic.")
                    
        texts = [h["text"] for h in hits]
        # Extract key sentences from the full chunks for cleaner context
        excerpts = _best_extract_sentences(q_precise, texts, max_pick=6) or texts[:6]

        # 3) LLM Synthesis (strictly grounded)
        llm_ans = _anthropic_answer(q_precise, excerpts)
        
        if llm_ans:
            return llm_ans

        # 4) Fallback: Extractive Summary (when LLM fails or explicitly returns 'Insufficient Context')
        parts = ["I cannot synthesize a complete answer, but here are the most relevant details found in the knowledge base:"]
        parts += [f"- {ex}" for ex in excerpts]
        return "\n".join(parts)

    def respond(self, user_q: str) -> str:
        """Handles chat input, including initial greetings."""
        ql = user_q.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|greetings|good (morning|afternoon|evening))[\W_]*", ql or ""):
            return ("Hello! I am Forecast360 AI Agent. Ask me about **forecasting models**, **pipelines**, **accuracy metrics**, "
                    "or our **Azure integrations** — I'll answer using our specialized knowledge base.")
        try:
            return self._answer(user_q)
        except Exception as e:
            logger.error(f"Agent response failed due to exception: {e}")
            return ("Sorry, something went wrong while processing your request. "
                    "Please try again in a moment. (Error details logged).")

# ============================ Streamlit UI (Refactored) ============================

def _display_kb_refresh_area():
    """Renders the KB refresh button linked to Azure sync."""
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
    
    # The button itself
    if st.button("🔄", key="refresh_kb", help="Refresh KB", use_container_width=False):
        with st.spinner("Refreshing knowledge base…"):
            try:
                # Call the external sync function
                stats = sync_from_azure(
                    st_secrets=st.secrets,
                    collection_name=COLLECTION_NAME,
                    container_key="container",
                    prefix_key="prefix",
                    embed_model_key=("rag", "embed_model"),
                    delete_before_upsert=True,
                    max_docs=None,
                )
                
                st.success(f"Done. Files: {stats.get('processed_files', 0)} | Chunks: {stats.get('inserted_chunks', 0)} | Cleared: {stats.get('cleared_sources', 0)} | Skipped: {stats.get('skipped_files', 0)}")
                st.toast("Knowledge base refreshed.")
            except Exception as e:
                st.error(f"KB refresh failed. Ensure 'azure_sync_weaviate' is configured: {e}")
    
    st.markdown("</span></div>", unsafe_allow_html=True)


def _render_agent_core(set_config: bool = False):
    """Renders the Streamlit UI and manages chat flow."""
    
    if set_config:
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout="centered"
        )
        
    st.markdown("""
    <style>
    .stButton>button { border-radius: 10px; border-color: #007bff; color: #007bff; }
    .stButton>button:hover { background-color: #007bff; color: white; }
    .bottom-actions { margin-top: 6px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    c1, c2 = st.columns([1, 8], vertical_alignment="center")
    with c1: st.image(ASSISTANT_ICON, width=80)
    with c2:
        st.markdown("### Forecast360 AI Agent")
        st.markdown('<span>Created by Amitesh Jha | iSoft</span>', unsafe_allow_html=True)
    
    # KB Refresh Area
    _display_kb_refresh_area()

    # --- Weaviate Connection and Agent Setup ---
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
    
    # Initialize Agent
    agent = Forecast360Agent(st.session_state["f360_client"], COLLECTION_NAME)
    
    # Initial assistant message (only once)
    if not st.session_state["messages"]:
        st.session_state["messages"].append({
            "role":"assistant",
            "content": "Hello! I’m your Forecast360 AI Agent. How can I help today?"
        })

    # Render chat history
    for m in st.session_state["messages"]:
        avatar = ASSISTANT_ICON if m["role"] == "assistant" else (USER_ICON if os.path.exists(USER_ICON) else "👤")
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # Handle user input
    user_q = st.chat_input("Ask me...", key="chat_box")
    
    # Check for pending query (useful for external triggering) OR new chat input
    query_to_process = st.session_state.pop("pending_query", user_q)
    
    if query_to_process:
        # Add user query to history
        st.session_state["messages"].append({"role":"user","content":query_to_process})
        
        # Display user message
        avatar_user = USER_ICON if os.path.exists(USER_ICON) else "👤"
        with st.chat_message("user", avatar=avatar_user):
            st.markdown(query_to_process)
            
        # Generate and display assistant response
        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            loading_msg = random.choice(PROMPTS["loading"])
            with st.spinner(loading_msg):
                reply = agent.respond(query_to_process)
            st.markdown(reply)
            
        # Add assistant response to history
        st.session_state["messages"].append({"role":"assistant","content":reply})
        st.rerun()


# ============================ Public API & Main Execution ============================

def render_agent():
    """Public API for embedding the agent inside a parent app/tab."""
    _render_agent_core(set_config=False)

def run():
    """Standalone runner for `streamlit run agent.py`."""
    _render_agent_core(set_config=True)

if __name__ == "__main__":
    run()
