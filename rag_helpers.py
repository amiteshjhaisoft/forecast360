# f360_rag_helpers.py
# Shared RAG utilities for Forecast360 (Weaviate v4)
# - vectorizer-aware search (near_text vs near_vector)
# - optional query rewrite via Anthropic, with guardrails
# - field override via secrets
# - light scoring penalties for surrogate/short chunks
# Author: Amitesh Jha | iSoft

from __future__ import annotations
import os, re, itertools
from typing import Any, Dict, List, Optional, Tuple

import weaviate
import weaviate.classes as wvc

# Optional local embeddings (used when collection.vectorizer == "none")
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional reranker
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


# ========= Config =========

MIN_TEXT_LEN = 60          # prefer real paragraphs over tiny/meta lines
DEFAULT_TOP_K = 8


# ========= Secrets helper (tiny shim) =========
def sget(st_secrets, section: str, key: str, default: Any = None) -> Any:
    try:
        return st_secrets[section].get(key, default)  # type: ignore
    except Exception:
        return default


# ========= Prompt bits (you can also import from your main app) =========
QUERY_REWRITE_INSTRUCTION = (
    "Reinterpret the user’s question in terms of Forecast360’s time-series forecasting context.\n\n"
    "Examples:\n"
    "- 'How does it work?' → 'How does Forecast360 perform time-series forecasting end-to-end?'\n"
    "- 'Which models do you use?' → 'Which forecasting algorithms are implemented in Forecast360?'\n"
    "- 'How accurate are forecasts?' → 'How does Forecast360 measure and display forecast accuracy?'\n\n"
    "Return only the rewritten, precise query."
)


# ========= Light utilities =========
def _sent_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def best_extract_sentences(question: str, texts: List[str], max_pick: int = 6) -> List[str]:
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


def _collection_vectorizer(coll) -> str:
    try:
        cfg = coll.config.get()
        v = getattr(cfg, "vectorizer", None)
        return (str(v) if v is not None else "").lower()
    except Exception:
        return ""


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


def _penalize_surrogates(c: Dict[str, Any]) -> float:
    """Small penalty for short/surrogate chunks to push them down."""
    t = (c.get("text") or "").strip()
    p = 0.0
    if len(t) < MIN_TEXT_LEN:
        p -= 0.05
    if t.startswith("[IMAGE]") or t.startswith("[AUDIO]") or t.startswith("[VIDEO]") or t.startswith("[FILE]") or t.startswith("[BINARY]"):
        p -= 0.10
    return p


# ========= Field picking (with secrets override) =========
def pick_text_and_source_fields(st_secrets, client: Any, class_name: str) -> Tuple[str, Optional[str]]:
    forced_text = sget(st_secrets, "weaviate", "text_property", None)
    forced_src  = sget(st_secrets, "weaviate", "source_property", None)
    if forced_text:
        return forced_text, forced_src

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


# ========= Embeddings / reranker loaders =========
def load_embedder(model_name: Optional[str]) -> Optional["SentenceTransformer"]:
    if not model_name or SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def load_reranker() -> Optional["CrossEncoder"]:
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


# ========= Vectorizer-aware search =========
def search_weaviate(
    st_secrets,
    client: Any,
    class_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    embedder: Optional["SentenceTransformer"] = None,
    text_field: Optional[str] = None,
    source_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    want = max(top_k, 24)
    coll = client.collections.get(class_name)
    vtype = _collection_vectorizer(coll)

    if not text_field or text_field is None:
        text_field, source_field = pick_text_and_source_fields(st_secrets, client, class_name)

    def _finish(objs):
        out = _collect_from_objects(objs, text_field, source_field)
        for c in out:
            c["score"] = c.get("score", 0.0) + _penalize_surrogates(c)
        return out

    hits: List[Dict[str, Any]] = []

    # Branch 1: server-side vectorizer present → near_text/hybrid(query only)
    if vtype and vtype != "none":
        try:
            res = coll.query.near_text(
                query=query,
                limit=want,
                return_metadata=wvc.query.MetadataQuery(score=True)
            )
            hits = _finish(res.objects)
            if hits:
                return hits
        except Exception:
            pass

        try:
            res = coll.query.hybrid(
                query=query,
                alpha=0.6,
                limit=want,
                return_metadata=wvc.query.MetadataQuery(score=True)
            )
            hits = _finish(res.objects)
            if hits:
                return hits
        except Exception:
            pass

    # Branch 2: vectorizer == 'none' → need client vectors
    else:
        if embedder is None:
            raise RuntimeError("Collection has vectorizer='none' but no embedder was provided.")
        qv = embedder.encode([query], normalize_embeddings=True)[0].astype("float32").tolist()
        try:
            res = coll.query.near_vector(
                near_vector=qv,
                limit=want,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            hits = _finish(res.objects)
            if hits:
                return hits
        except Exception:
            pass

        try:
            res = coll.query.hybrid(
                query=query,
                vector=qv,
                alpha=0.6,
                limit=want,
                return_metadata=wvc.query.MetadataQuery(score=True)
            )
            hits = _finish(res.objects)
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
        return _finish(res.objects)
    except Exception:
        return []


# ========= Query rewrite (safe) =========
def rewrite_query_safe(orig_question: str, system_prompt: Optional[str] = None, max_tokens: int = 120) -> str:
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_APIKEY") or ""
    if not key or not orig_question.strip():
        return orig_question
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            system=system_prompt or "",
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role":"user","content": QUERY_REWRITE_INSTRUCTION + "\n\nUser question:\n" + orig_question}],
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"]).strip()
        # Guardrails: fallback if weird/too short/different topic
        if not text or len(text) < 8:
            return orig_question
        share = len(set(re.findall(r"[A-Za-z]{4,}", text.lower())) &
                    set(re.findall(r"[A-Za-z]{4,}", orig_question.lower())))
        return text if share >= 1 else orig_question
    except Exception:
        return orig_question


# ========= Retrieve wrapper (vectorizer-aware) =========
def retrieve(
    st_secrets,
    client: Any,
    class_name: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    embed_model_name: Optional[str] = None,
    use_query_rewrite: bool = True,
    system_prompt_for_rewrite: Optional[str] = None,
    reranker: Optional["CrossEncoder"] = None
) -> List[Dict[str, Any]]:
    q = rewrite_query_safe(query, system_prompt_for_rewrite) if use_query_rewrite else query

    embedder = load_embedder(embed_model_name)
    hits = search_weaviate(
        st_secrets=st_secrets,
        client=client,
        class_name=class_name,
        query=q,
        top_k=top_k,
        embedder=embedder,
    )

    # Optional rerank (if provided)
    if not hits or reranker is None:
        return hits[:top_k]

    try:
        pairs = [(q, h["text"]) for h in hits]
        scores = reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank"] = float(s)
        hits.sort(key=lambda x: (x.get("rerank", 0.0), x.get("score", 0.0)), reverse=True)
        return hits[:top_k]
    except Exception:
        return hits[:top_k]
