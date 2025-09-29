# Author: Amitesh Jha | iSOFT
# decision_intelligence.py
from __future__ import annotations

import os, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import streamlit as st

# Optional: install as needed
# pip install anthropic requests gTTS
try:
    import requests  # Ollama REST (chat + embeddings)
except Exception:
    requests = None

# ------------------------- Configuration -------------------------
KB_ROOT_DEFAULT = "KnowledgeBase/llm_snapshots"
LATEST_JSON_NAME = "latest.json"
MANIFEST_NAME = "manifest.json"

# Broadened fallbacks
CONTENT_FILE_FALLBACKS = (
    "content.jsonl", "content.json", "content.ndjson",
    "chunks.jsonl", "snapshot.jsonl", "kb.jsonl", "kb.ndjson", "lines.jsonl"
)

AVATAR_PATH = "assets/Forecast360.png"  # optional avatar

# ------------------------- Snapshot discovery -------------------------

@dataclass
class SnapshotPointer:
    folder: Path
    manifest_path: Path
    content_path: Path

def _app_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd()).resolve()

def _kb_root() -> Path:
    app = _app_root()
    env_root = os.environ.get("F360_KB_ROOT", "").strip()
    if env_root:
        p = Path(env_root)
        if not p.is_absolute():
            p = app / p
        return p.resolve()
    return (app / KB_ROOT_DEFAULT).resolve()

def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _candidate_latest_json_paths(app: Path, kb_root: Path) -> List[Path]:
    kbase = kb_root.parent if kb_root.name == "llm_snapshots" else kb_root
    cands = [app / LATEST_JSON_NAME, kbase / LATEST_JSON_NAME, kb_root / LATEST_JSON_NAME]
    seen, out = set(), []
    for p in cands:
        if p in seen: 
            continue
        seen.add(p)
        if p.exists():
            out.append(p)
    out += [p for p in cands if p not in out]
    return out

def _parse_latest_pointer(j: dict, base: Path) -> Optional[Path]:
    cand = j.get("path") or j.get("folder") or j.get("dir") or j.get("snapshot_dir") or ""
    if not cand:
        return None
    p = Path(cand)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p if p.exists() and p.is_dir() else None

def _discover_latest_from_latest_json(kb_root: Path) -> Optional[Path]:
    app = _app_root()
    for latest_path in _candidate_latest_json_paths(app, kb_root):
        if not latest_path.exists():
            continue
        try:
            j = _read_json(latest_path)
            p = _parse_latest_pointer(j, latest_path.parent) or _parse_latest_pointer(j, app)
            if p:
                return p
        except Exception:
            continue
    return None

def _has_content_or_manifest(d: Path) -> bool:
    if (d / MANIFEST_NAME).exists():
        return True
    for fn in CONTENT_FILE_FALLBACKS:
        if (d / fn).exists():
            return True
    for pat in ("*.jsonl", "*.ndjson"):
        if any(d.glob(pat)):
            return True
    return False

def _find_candidate_snapshot_dirs(kb_root: Path, max_depth: int = 2) -> List[Path]:
    if not kb_root.exists():
        return []
    out: List[Path] = [d for d in kb_root.iterdir() if d.is_dir() and _has_content_or_manifest(d)]
    frontier = [d for d in kb_root.iterdir() if d.is_dir()]
    depth = 1
    while depth <= max_depth and frontier:
        new_frontier: List[Path] = []
        for parent in frontier:
            for d in parent.iterdir():
                if d.is_dir():
                    if _has_content_or_manifest(d):
                        out.append(d)
                    new_frontier.append(d)
        frontier = new_frontier
        depth += 1
    uniq, seen = [], set()
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq

def _pick_latest_dir(dirs: List[Path]) -> Optional[Path]:
    if not dirs:
        return None
    def _score(p: Path):
        digits = "".join(c for c in p.name if c.isdigit())
        if digits:
            try: return (0, int(digits))
            except Exception: pass
        return (1, int(p.stat().st_mtime))
    return sorted(dirs, key=_score, reverse=True)[0]

def _detect_content_file(snap_dir: Path) -> Optional[Path]:
    for fn in CONTENT_FILE_FALLBACKS:
        p = (snap_dir / fn).resolve()
        if p.exists():
            return p
    for pat in ("*.jsonl", "*.ndjson"):
        for p in snap_dir.glob(pat):
            if p.is_file():
                return p.resolve()
    for p in snap_dir.glob("*.json"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            first_line = txt.splitlines()[0] if "\n" in txt else txt
            obj = json.loads(first_line)
            if isinstance(obj, dict) and any(k in obj for k in ("text", "content", "chunk", "chunks", "rows", "items")):
                return p.resolve()
        except Exception:
            continue
    return None

def _resolve_latest_snapshot() -> Optional[SnapshotPointer]:
    kb_root = _kb_root()
    if not kb_root.exists():
        return None
    snap_dir = _discover_latest_from_latest_json(kb_root) or _pick_latest_dir(_find_candidate_snapshot_dirs(kb_root, 2))
    if snap_dir is None:
        return None
    manifest_path = snap_dir / MANIFEST_NAME
    content_path = None

    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            cf = manifest.get("content_file") or (manifest.get("files", {}) or {}).get("content") or ""
            if cf:
                trial = (snap_dir / cf).resolve()
                if trial.exists():
                    content_path = trial
        except Exception:
            pass

    if content_path is None or not content_path.exists():
        content_path = _detect_content_file(snap_dir)

    if not content_path or not content_path.exists():
        return None

    return SnapshotPointer(folder=snap_dir, manifest_path=manifest_path, content_path=content_path)

# ------------------------- Content loading -------------------------

@dataclass
class KBContent:
    raw_lines: List[str]
    texts: List[str]
    n_bytes: int = 0
    n_rows: int = 0

def _read_all_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # try latin-1 as last resort
        return p.read_text(encoding="latin-1")

def _coerce_to_lines_any(json_text: str) -> List[str]:
    """
    Accept JSONL, JSON array, or single JSON object; return a list of JSON strings (one per row).
    """
    s = json_text.strip()
    if not s:
        return []
    # JSON Lines? quick check: many lines and most look like objects
    lines = s.splitlines()
    if len(lines) > 1 and lines[0].strip().startswith("{"):
        return [ln for ln in lines if ln.strip()]
    # Try full JSON parse (array or object)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for item in obj:
                try:
                    out.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    pass
            return out
        elif isinstance(obj, dict):
            return [json.dumps(obj, ensure_ascii=False)]
    except Exception:
        # fallback: treat each non-empty line as a row
        return [ln for ln in lines if ln.strip()]
    return []

def _load_latest_kb_content(snap: Optional[SnapshotPointer] = None) -> Optional[KBContent]:
    snap = snap or _resolve_latest_snapshot()
    if not snap or not snap.content_path.exists():
        return None

    raw_text = _read_all_text(snap.content_path)
    rows = _coerce_to_lines_any(raw_text)

    raw_lines, texts = [], []
    for s in rows:
        s = s.strip()
        if not s:
            continue
        raw_lines.append(s)
        try:
            obj = json.loads(s)
            txt = obj.get("text") or obj.get("content") or obj.get("chunk") or ""
            if isinstance(txt, dict):
                txt = txt.get("text") or ""
            texts.append(str(txt) if txt else s)
        except Exception:
            texts.append(s)

    if not texts:
        return KBContent(raw_lines=[], texts=[], n_bytes=len(raw_text.encode("utf-8")), n_rows=0)
    return KBContent(raw_lines=raw_lines, texts=texts, n_bytes=len(raw_text.encode("utf-8")), n_rows=len(raw_lines))

# ------------------------- Quick-Facts helpers -------------------------

@dataclass
class DIQuickFacts:
    best_rmse: Optional[tuple[str, float]]
    best_mape: Optional[tuple[str, float]]
    best_mae: Optional[tuple[str, float]]
    table: List[Dict]

def _to_float(x):
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "").strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return None

def _maybe_row_from_obj(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    name = (obj.get("model") or obj.get("name") or obj.get("algo") or obj.get("algorithm")
            or obj.get("Model") or obj.get("MODEL"))
    keys = {k.lower(): k for k in obj.keys()}
    rmse = _to_float(obj.get(keys.get("rmse", ""), None))
    mape = _to_float(obj.get(keys.get("mape", ""), None))
    mae  = _to_float(obj.get(keys.get("mae", ""), None))
    if rmse is None:
        rmse = _to_float(obj.get("RMSE") or obj.get("root_mean_squared_error") or obj.get("root_mse"))
    if mape is None:
        mape = _to_float(obj.get("MAPE") or obj.get("mean_absolute_percentage_error"))
    if mae is None:
        mae = _to_float(obj.get("MAE")  or obj.get("mean_absolute_error"))
    if name and any(v is not None for v in (rmse, mape, mae)):
        return {"model": str(name), "rmse": rmse, "mape": mape, "mae": mae}
    return None

def _extract_quick_facts(kb: KBContent) -> DIQuickFacts:
    rows: List[dict] = []
    for line in kb.raw_lines:
        try:
            obj = json.loads(line)
            r = _maybe_row_from_obj(obj)
            if r:
                rows.append(r)
            for k in ("leaderboard", "models", "rows", "items", "table"):
                arr = obj.get(k)
                if isinstance(arr, list):
                    for it in arr:
                        rr = _maybe_row_from_obj(it)
                        if rr:
                            rows.append(rr)
        except Exception:
            m = re.findall(r"([A-Za-z0-9\-\_\/\.\+]+)\s*[,|]\s*rmse\s*[:=]\s*([0-9\.,%]+)", line, flags=re.I)
            for name, rmse_s in m:
                rows.append({"model": name, "rmse": _to_float(rmse_s), "mape": None, "mae": None})

    dedup: Dict[str, dict] = {}
    for r in rows:
        key = r["model"]
        old = dedup.get(key)
        if not old:
            dedup[key] = r
        else:
            new_rmse = r.get("rmse")
            old_rmse = old.get("rmse")
            if new_rmse is not None and (old_rmse is None or new_rmse < old_rmse):
                dedup[key] = r

    rows = list(dedup.values())

    def _best(metric):
        candidates = [(r["model"], r.get(metric)) for r in rows if r.get(metric) is not None]
        if not candidates:
            return None
        name, val = min(candidates, key=lambda x: x[1])
        return (name, float(val))

    return DIQuickFacts(_best("rmse"), _best("mape"), _best("mae"), rows)

def _facts_summary(f: DIQuickFacts) -> str:
    if not f or (not f.best_rmse and not f.best_mape and not f.best_mae):
        return "No leaderboard metrics found in the latest snapshot."
    parts = []
    if f.best_rmse: parts.append(f"Best RMSE: {f.best_rmse[0]} → {f.best_rmse[1]:.4f}")
    if f.best_mape: parts.append(f"Best MAPE: {f.best_mape[0]} → {f.best_mape[1]*100:.2f}%")
    if f.best_mae:  parts.append(f"Best MAE: {f.best_mae[0]}  → {f.best_mae[1]:.4f}")
    return " • ".join(parts)

# ------------------------- LLM backends -------------------------

def _anthropic_client():
    try:
        import anthropic
    except Exception as e:
        raise RuntimeError("Anthropic SDK not installed. Run: pip install anthropic") from e
    key = os.environ.get("ANTHROPIC_API_KEY") or (st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment or st.secrets.")
    return anthropic.Anthropic(api_key=key)

def _ollama_request(model: str, messages: List[dict], stream: bool = True):
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests")
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {"model": model, "messages": messages, "stream": stream}
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp

SYSTEM_TEMPLATE = (
    "You are Decision Intelligence for Forecast360. Use ONLY the content from the latest KnowledgeBase snapshot. "
    "Answer DIRECTLY and numerically when asked about metrics. If the requested metric or model is not present, say "
    "you cannot find it in the latest snapshot. Prefer the 'Quick facts' summary and leaderboard entries. "
    "When stating a metric, include the model name and the exact value (e.g., 'ARIMA: RMSE 12.34'). Keep answers concise."
)

def _batch_context(texts: List[str], max_chars: int = 12000) -> str:
    out, size = [], 0
    for tline in texts:
        if size + len(tline) + 1 > max_chars:
            break
        out.append(tline)
        size += len(tline) + 1
    return "\n".join(out)

# ------------------------- Retrieval (vectors) -------------------------

def _ollama_embed(texts: List[str], model: str = None) -> np.ndarray:
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests")
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embeddings")
    embed_model = model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    vecs = []
    for t in texts:
        r = requests.post(url, json={"model": embed_model, "prompt": t}, timeout=180)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return np.array(vecs, dtype=np.float32)

def _ensure_vector_index(snap_dir: Path, kb_texts: List[str]) -> tuple[np.ndarray, List[str]]:
    epath = snap_dir / "embeddings.npy"
    cpath = snap_dir / "chunks.jsonl"
    if epath.exists() and cpath.exists():
        try:
            vecs = np.load(epath)
            lines = []
            for x in cpath.read_text(encoding="utf-8").splitlines():
                try:
                    lines.append(json.loads(x)["text"])
                except Exception:
                    lines.append(x)
            return vecs, lines
        except Exception:
            pass
    vecs = _ollama_embed(kb_texts)
    np.save(epath, vecs)
    with cpath.open("w", encoding="utf-8") as f:
        for tline in kb_texts:
            f.write(json.dumps({"text": tline}, ensure_ascii=False) + "\n")
    return vecs, kb_texts

def _top_k(query: str, vecs: np.ndarray, chunks: List[str], k: int = 8) -> List[str]:
    qv = _ollama_embed([query])[0]
    denom = (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qv) + 1e-9) + 1e-9)
    sims = (vecs @ qv) / denom
    idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in idx.tolist()]

# ------------------------- Streamlit UI -------------------------

def _ensure_state():
    st.session_state.setdefault("di_messages", [])
    st.session_state.setdefault("di_provider", "Ollama")
    st.session_state.setdefault("di_model_ollama", "llama3.2:1b-instruct-q4_K_M")
    st.session_state.setdefault("di_model_claude", "claude-3-7-sonnet-latest")

def _snapshot_label(snap: Optional[SnapshotPointer]) -> str:
    return (snap.folder.name if snap else "none (no snapshot found)")

def _render_header(latest_ok: bool, snap_info: Optional[str], kb_meta: Optional[KBContent]):
    st.markdown("### 🧠 Decision Intelligence")
    st.caption(f"Context: **Latest KnowledgeBase snapshot** — {snap_info or 'not available'}")
    if kb_meta is not None:
        st.caption(f"KB file size: **{kb_meta.n_bytes} bytes**, rows detected: **{kb_meta.n_rows}**")
    if not latest_ok:
        st.info("I’ll still answer, but I don’t see usable snapshot content yet. "
                "Run **Getting Started → Run Analysis** to populate one.", icon="💡")

def _stream_ollama(model: str, sys_prompt: str, user_prompt: str, context: str):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Quick facts and context below.\n\n{context}\n\nQuestion:\n{user_prompt}"},
    ]
    resp = _ollama_request(model=model, messages=messages, stream=True)
    buf = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            part = json.loads(line.decode("utf-8"))
            token = part.get("message", {}).get("content", "")
            if token:
                buf.append(token)
                yield token
        except Exception:
            continue

def _call_claude(model: str, sys_prompt: str, user_prompt: str, context: str) -> str:
    client = _anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=sys_prompt,
        messages=[{"role": "user", "content": f"Quick facts and context below.\n\n{context}\n\nQuestion:\n{user_prompt}"}],
        temperature=0.2,
    )
    parts = []
    for blk in message.content:
        if getattr(blk, "type", None) == "text":
            parts.append(blk.text)
        elif isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(blk.get("text") or "")
    return "".join(parts).strip()

def _voice_reply_tts(text: str) -> None:
    try:
        from gtts import gTTS
        from io import BytesIO
        import base64
        tts = gTTS(text=text, lang="en")
        buf = BytesIO(); tts.write_to_fp(buf)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")
    except Exception:
        st.info("Install `gTTS` for voice playback (optional).")

def run_di_chat(popover_mode: bool = False):
    _ensure_state()

    st.button("🔄 Refresh latest snapshot", help="Re-scan KnowledgeBase now", use_container_width=False)

    snap = _resolve_latest_snapshot()
    kb = _load_latest_kb_content(snap) if snap else None
    latest_ok = kb is not None and len(kb.texts) > 0
    snap_info = snap.folder.name if snap else None

    _render_header(latest_ok, snap_info, kb)

    if snap and getattr(snap, "content_path", None):
        st.caption(f"Content file: `{snap.content_path.name}`")

    di_facts = _extract_quick_facts(kb) if latest_ok else None
    facts_text = _facts_summary(di_facts) if di_facts else "—"
    if latest_ok:
        with st.container():
            st.caption("Quick facts (from latest snapshot)")
            st.info(facts_text)

    with st.expander("Model Settings", expanded=not popover_mode):
        colp, colm = st.columns([0.45, 0.55])
        with colp:
            _ = st.selectbox(
                "Provider", ["Ollama", "Claude"],
                index=0 if st.session_state.get("di_provider", "Ollama") == "Ollama" else 1,
                key="di_provider", help="Choose your LLM provider",
            )
            st.caption(f"**Provider:** {st.session_state['di_provider']}")
        with colm:
            label = f"{st.session_state['di_provider']} model"
            if st.session_state["di_provider"] == "Ollama":
                st.text_input(label, key="di_model_ollama",
                              value=st.session_state.get("di_model_ollama", "llama3.2:1b-instruct-q4_K_M"),
                              help="Example: llama3.2:instruct, mistral. Set OLLAMA_EMBED_MODEL (default: nomic-embed-text).")
                st.caption("Runs against your local Ollama at http://localhost:11434")
            else:
                st.text_input(label, key="di_model_claude",
                              value=st.session_state.get("di_model_claude", "claude-3-7-sonnet-latest"),
                              help="Requires ANTHROPIC_API_KEY")

    t_chat, t_voice, t_avatar = st.tabs(["💬 Chat", "🎙️ Voice (optional)", "🧑‍💼 Avatar (optional)"])

    with t_chat:
        st.caption(f"🔎 Scanning snapshot: **{_snapshot_label(snap)}**")
        for m in st.session_state["di_messages"]:
            avatar = AVATAR_PATH if m["role"] == "assistant" and Path(AVATAR_PATH).exists() else None
            st.chat_message(m["role"], avatar=avatar).markdown(m["content"])

        user_msg = st.chat_input(f"Ask about snapshot: {_snapshot_label(snap)} …", disabled=False)

        if user_msg:
            st.session_state["di_messages"].append({"role": "user", "content": user_msg})

            if not latest_ok:
                with st.chat_message("assistant", avatar=(AVATAR_PATH if Path(AVATAR_PATH).exists() else None)):
                    msg = (
                        "I can see a snapshot folder but its content file has no usable rows yet. "
                        "Please re-run **Getting Started → Run Analysis** (it should write non-empty text rows "
                        "into the snapshot), then ask again."
                    )
                    st.markdown(msg)
                st.session_state["di_messages"].append({"role": "assistant", "content": msg})
                st.stop()

            vecs, chunks = _ensure_vector_index(snap.folder, kb.texts)
            retrieved = _top_k(user_msg, vecs, chunks, k=8)

            facts_block = f"Quick facts:\n{facts_text}\n\n"
            retrieval_block = "Top relevant chunks:\n" + "\n---\n".join(retrieved) + "\n\n"
            context = facts_block + retrieval_block + _batch_context(
                kb.texts, max_chars=12000 - len(facts_block) - len(retrieval_block)
            )
            system = SYSTEM_TEMPLATE

            if st.session_state["di_provider"] == "Ollama":
                model = (st.session_state.get("di_model_ollama") or "llama3.2:1b-instruct-q4_K_M").strip()
                with st.chat_message("assistant", avatar=(AVATAR_PATH if Path(AVATAR_PATH).exists() else None)):
                    placeholder = st.empty(); acc = []
                    try:
                        for tok in _stream_ollama(model, system, user_msg, context):
                            acc.append(tok); placeholder.markdown("".join(acc))
                    except Exception as e:
                        err = f"Ollama error: {e}"
                        st.error(err); st.session_state["di_messages"].append({"role": "assistant", "content": err})
                    else:
                        st.session_state["di_messages"].append({"role": "assistant", "content": "".join(acc)})
            else:
                model = (st.session_state.get("di_model_claude") or "claude-3-7-sonnet-latest").strip()
                with st.chat_message("assistant", avatar=(AVATAR_PATH if Path(AVATAR_PATH).exists() else None)):
                    try:
                        text = _call_claude(model, system, user_msg, context)
                        st.markdown(text)
                        st.session_state["di_messages"].append({"role": "assistant", "content": text})
                    except Exception as e:
                        err = f"Claude error: {e}"
                        st.error(err); st.session_state["di_messages"].append({"role": "assistant", "content": err)})

    with t_voice:
        st.write("Use **Chat** above and press **Speak reply** to hear TTS.")
        if st.button("🔊 Speak last assistant reply"):
            last = next((m for m in reversed(st.session_state["di_messages"]) if m["role"] == "assistant"), None)
            if not last: st.info("No assistant reply yet.")
            else: _voice_reply_tts(last["content"])

    with t_avatar:
        st.write("Lightweight placeholder: shows assistant text with a simple avatar.")
        st.markdown(
            """
            <div style="display:flex;gap:12px;align-items:flex-start;">
              <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#4f46e5,#06b6d4);"></div>
              <div style="flex:1;">
                <em>Avatar will read the latest assistant message (use Voice tab for audio).</em>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    st.set_page_config(page_title="Forecast360 – Decision Intelligence", page_icon="🧠", layout="wide")
    st.title("🧠 Decision Intelligence (Module Preview)")
    run_di_chat()