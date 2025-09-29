# Author: Amitesh Jha | iSOFT
# f360_llm_snapshot.py
from __future__ import annotations

import os, json, time, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None

KB_ROOT_DEFAULT = "KnowledgeBase/llm_snapshots"
LATEST_JSON_NAME = "latest.json"
MANIFEST_NAME = "manifest.json"
CONTENT_FILE_NAME = "content.jsonl"

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

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _chunk_text(txt: str, max_len: int = 1200, overlap: int = 120) -> List[str]:
    txt = str(txt or "").strip()
    if not txt:
        return []
    chunks, i, n = [], 0, len(txt)
    while i < n:
        j = min(i + max_len, n)
        chunks.append(txt[i:j])
        i = max(0, j - overlap)
    return chunks

def _hash_signature(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@dataclass
class LlmSnapshot:
    page_name: str
    run_tag: str = "manual"
    min_interval_sec: int = 20
    records: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_text(self, title: str, text: str):
        chunks = _chunk_text(text)
        if not chunks:
            return
        for ch in chunks:
            self.records.append({"type": "text", "title": title, "text": ch})

    def add_metric(self, name: str, value: Any):
        self.records.append({"type": "metric", "name": name, "value": value})

    def add_kv(self, title: str, kv: Dict[str, Any]):
        self.records.append({"type": "kv", "title": title, "values": kv})

    def add_table(self, title: str, df_or_list):
        if pd is not None and isinstance(df_or_list, pd.DataFrame):
            df = df_or_list.copy()
            sample = df.head(50).to_csv(index=False)
            summary = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": list(map(str, df.columns))}
            self.records.append({"type": "table", "title": title, "sample_csv": sample, "summary": summary})
        else:
            try:
                if isinstance(df_or_list, list) and df_or_list and isinstance(df_or_list[0], dict):
                    sample_rows = df_or_list[:50]
                else:
                    sample_rows = [{"value": str(df_or_list)}]
                self.records.append({"type": "table", "title": title, "rows": sample_rows})
            except Exception:
                self.records.append({"type": "table", "title": title, "rows": [{"value": "unavailable"}]})

    def _write_content(self, dir_: Path):
        with (dir_ / CONTENT_FILE_NAME).open("w", encoding="utf-8") as f:
            for r in self.records:
                if r["type"] == "text":
                    out = {"type": "text", "title": r.get("title"), "text": r["text"]}
                elif r["type"] == "metric":
                    out = {"type": "metric", "name": r["name"], "value": r["value"], "text": f"{r['name']} = {r['value']}"}
                elif r["type"] == "kv":
                    out = {"type": "kv", "title": r.get("title"), "values": r.get("values", {}), "text": json.dumps(r.get("values", {}))}
                elif r["type"] == "table":
                    if "sample_csv" in r:
                        out = {
                            "type": "table",
                            "title": r.get("title"),
                            "sample_csv": r["sample_csv"],
                            "summary": r.get("summary"),
                            "text": f"{r.get('title','table')} | cols={r.get('summary',{}).get('columns','?')} rows={r.get('summary',{}).get('rows','?')}",
                        }
                    else:
                        out = {"type": "table", "title": r.get("title"), "rows": r.get("rows", []), "text": f"{r.get('title','table')} rows={len(r.get('rows',[]))}"}
                else:
                    out = {"type": r.get("type", "unknown"), "payload": r}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    def write(self, name: str = "forecast360") -> Path:
        kb_root = _kb_root(); kb_root.mkdir(parents=True, exist_ok=True)
        stamp = _now_tag()
        snap_dir = kb_root / f"{stamp}_{name}"; snap_dir.mkdir(parents=True, exist_ok=True)

        # guarantee at least one line so DI never sees an empty file
        if not self.records:
            self.add_text("Info", f"Snapshot created at {stamp} for {name} (no explicit records were provided).")

        self._write_content(snap_dir)

        manifest = {
            "name": name, "page": self.page_name, "run_tag": self.run_tag,
            "timestamp": stamp, "items": len(self.records), "content_file": CONTENT_FILE_NAME,
        }
        (snap_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        latest_obj = {"path": str(snap_dir), "name": name, "ts": stamp}
        (_app_root() / LATEST_JSON_NAME).write_text(json.dumps(latest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        kb_base = kb_root.parent if kb_root.name == "llm_snapshots" else kb_root
        (kb_base / LATEST_JSON_NAME).write_text(json.dumps(latest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        (kb_root / LATEST_JSON_NAME).write_text(json.dumps(latest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return snap_dir

    def autosave_if_changed(self, signature: dict, name: str = "forecast360"):
        tag = _hash_signature(signature)
        last = _read_last_meta(name)
        now = time.time()
        if last and last.get("hash") == tag and (now - last.get("time", 0) < self.min_interval_sec):
            return None
        out_dir = self.write(name=name)
        _write_last_meta(name, {"hash": tag, "time": now, "dir": str(out_dir)})
        return out_dir

_LAST_META_FILE = ".f360_lastmeta.json"

def _read_last_meta(name: str) -> Optional[dict]:
    fp = _app_root() / _LAST_META_FILE
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8")).get(name)
    except Exception:
        return None

def _write_last_meta(name: str, payload: dict):
    fp = _app_root() / _LAST_META_FILE
    allm = {}
    if fp.exists():
        try:
            allm = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            allm = {}
    allm[name] = payload
    fp.write_text(json.dumps(allm, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------- Public helpers -------------------------

def capture_forecast360_results(
    page_name: str,
    models_summary_text: str | None = None,
    best_model: str | None = None,
    metrics: dict | None = None,
    input_table=None,
    output_table=None,
    run_tag: str = "auto",
    name: str = "forecast360",
    min_interval_sec: int = 30,
):
    """
    Call this after computing results/plots; writes a robust snapshot for DI.
    """
    snap = LlmSnapshot(page_name=page_name, run_tag=run_tag, min_interval_sec=min_interval_sec)

    if models_summary_text:
        snap.add_text("Model summary", models_summary_text)
    if best_model:
        snap.add_kv("Best model", {"name": best_model})
    if metrics:
        for k, v in metrics.items():
            snap.add_metric(k, v)

    if pd is not None:
        if input_table is not None and isinstance(input_table, pd.DataFrame):
            snap.add_table("Input sample", input_table)
        if output_table is not None and isinstance(output_table, pd.DataFrame):
            snap.add_table("Forecast output", output_table)

    sig = {"models_summary_text": models_summary_text, "best_model": best_model, "metrics": metrics}
    return snap.autosave_if_changed(signature=sig, name=name)

def gather_and_snapshot_forecast360(page_name: str = "One Pager", run_tag: str = "one-pager") -> Optional[str]:
    """
    Conservative session_state gatherer so you ALWAYS get some content.
    """
    s = LlmSnapshot(page_name=page_name, run_tag=run_tag, min_interval_sec=20)

    # Try common fields your app may set
    ss = getattr(__import__("streamlit"), "session_state")
    best_model = ss.get("best_model") or ss.get("gs_best_model")
    leaderboard_text = ss.get("leaderboard_text") or ss.get("gs_leaderboard_text")
    metrics = ss.get("metrics") or ss.get("gs_metrics") or {}
    input_df = ss.get("last_input_df")
    output_df = ss.get("last_forecast_df")

    if leaderboard_text:
        s.add_text("Leaderboard", leaderboard_text)
    if best_model:
        s.add_kv("Best model", {"name": best_model})
    if isinstance(metrics, dict) and metrics:
        for k, v in metrics.items():
            s.add_metric(str(k), v)

    # If there’s nothing yet, add a minimal “context” line so DI never sees empty content
    if not s.records:
        s.add_text("Context", "Forecast360 snapshot with minimal content (no models/metrics captured this run).")

    if pd is not None:
        if input_df is not None and isinstance(input_df, pd.DataFrame):
            s.add_table("Input sample", input_df)
        if output_df is not None and isinstance(output_df, pd.DataFrame):
            s.add_table("Forecast output", output_df)

    out = s.autosave_if_changed(signature={"best_model": best_model, "metrics": metrics}, name="forecast360")
    return str(out) if out else None