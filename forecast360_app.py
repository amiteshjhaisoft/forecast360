
"""
Forecast360 — Structured Streamlit app (single-file)
with synchronized KnowledgeBase snapshots.

Author: Amitesh Jha | iSOFT | 2025-10-08

Highlights
- Clean layout: config, UI, data, modeling stubs, and KB snapshot utilities.
- Snapshot system captures *markdown + its exact tables/plots/images* together,
  preserving order and section grouping so an LLM knows which text describes which output.
- Uploaded dataset is saved into the KB too (original + sanitized CSV + schema profile).
- Optional LangChain hook to summarize a snapshot folder (self-contained; works without it).

How it works
- We provide a "CaptureSession" that wraps Streamlit render calls (markdown, dataframe,
  pyplot, image) and records each block to an in-memory ledger with stable IDs.
- When you click "📸 Capture Snapshot…", we write a KB folder structure:
  KnowledgeBase/
    ├─ <timestamp-id>/
    │   ├─ text/            # all markdown blocks as numbered files
    │   ├─ tables/          # CSV exports, with block ids
    │   ├─ figures/         # PNG figures, with block ids
    │   ├─ images/          # uploaded or generated images, with block ids
    │   ├─ data/            # the uploaded dataset (original + sanitized.csv)
    │   ├─ meta/summary.json# manifest mapping blocks → assets
    │   ├─ index.html       # lightweight viewer linking text ↔ outputs
    │   └─ README.md        # human overview
    └─ latest → symlink/copy of latest snapshot (best-effort)

Note: This file is intentionally framework-agnostic for the model layer — you can plug in
ARIMA/Prophet/etc. Keep the KB capture API calls around your render code and the snapshots
will stay in sync automatically.
"""
from __future__ import annotations

# =============== Standard Library ===============
import io
import os
import re
import json
import time
import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============== Third-Party ===============
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Optional: LangChain summarizer (safe import)
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    # You can swap to your preferred LLM provider here
    from langchain.chat_models import ChatOpenAI  # noqa: F401 (example)
    HAVE_LANGCHAIN = True
except Exception:
    HAVE_LANGCHAIN = False

# ================== App Config ==================
st.set_page_config(
    page_title="Forecast360",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS polish
st.markdown(
    """
    <style>
    section.main > div.block-container{max-width:100% !important; padding-left:12px; padding-right:12px;}
    .block-card{background:#fff; border:1px solid #eee; border-radius:14px; padding:12px 14px; box-shadow:0 6px 16px rgba(0,0,0,.05)}
    .block-title{margin:0 0 8px; color:#3b2dbf; font-weight:800}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Utils: hashing, arrow-sanitization, safe image/fig save
# =====================================================

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        s = out[c]
        # bytes → text
        if s.dtype == "object" and s.map(lambda x: isinstance(x, (bytes, bytearray))).any():
            s = s.map(lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x)
        # heuristic cast
        if s.dtype == "object":
            num_try = pd.to_numeric(s, errors="coerce")
            if num_try.notna().mean() >= 0.9:
                out[c] = num_try
            else:
                out[c] = s.astype("string[pyarrow]")
            continue
        try:
            if pd.api.types.is_bool_dtype(s):
                out[c] = s.astype("boolean[pyarrow]")
            elif pd.api.types.is_integer_dtype(s):
                out[c] = s.astype("Int64")
            elif pd.api.types.is_float_dtype(s):
                out[c] = s.astype("float64")
        except Exception:
            out[c] = s.astype("string[pyarrow]")
    return out


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ======================================
# Knowledge Base Snapshot Implementation
# ======================================

@dataclass
class Block:
    id: str
    section: str
    kind: str  # "md" | "table" | "figure" | "image"
    title: Optional[str] = None
    text: Optional[str] = None
    table_csv: Optional[str] = None
    image_b64_png: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class CaptureSession:
    """Capture Streamlit outputs in order, grouped by sections.

    Usage:
        cap = CaptureSession()
        cap.set_section("Data Preview")
        cap.markdown("### Uploaded data")
        cap.dataframe(df)
        # ...
        if st.button("📸 Capture Snapshot…"):
            cap.write_snapshot(base_folder=Path("KnowledgeBase"))
    """

    def __init__(self):
        self.section = "General"
        self.blocks: List[Block] = []
        self._counter = 0
        # data payloads to persist alongside artifacts
        self.uploaded_file_bytes: Optional[bytes] = None
        self.uploaded_file_name: Optional[str] = None
        self.uploaded_dataframe: Optional[pd.DataFrame] = None

    # ---------- section API ----------
    def set_section(self, name: str):
        self.section = name or "General"

    # ---------- id helper ----------
    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{int(self._counter):04d}-{kind}"

    # ---------- capture wrappers (render + record) ----------
    def markdown(self, md: str, unsafe_allow_html: bool = True):
        st.markdown(md, unsafe_allow_html=unsafe_allow_html)
        bid = self._next_id("md")
        self.blocks.append(Block(id=bid, section=self.section, kind="md", text=md))

    def dataframe(self, df: pd.DataFrame, **kwargs):
        st.dataframe(df, **kwargs)
        bid = self._next_id("table")
        csv = df.to_csv(index=False)
        self.blocks.append(Block(id=bid, section=self.section, kind="table", table_csv=csv,
                                 meta={"shape": list(df.shape)}))

    def pyplot(self, fig=None, **kwargs):
        st.pyplot(fig, **kwargs)
        buf = io.BytesIO()
        (fig or plt.gcf()).savefig(buf, format="png", bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        bid = self._next_id("figure")
        self.blocks.append(Block(id=bid, section=self.section, kind="figure", image_b64_png=b64))

    def image(self, image, **kwargs):
        st.image(image, **kwargs)
        # normalize to bytes→b64
        if isinstance(image, (bytes, bytearray)):
            raw = bytes(image)
        elif isinstance(image, str) and Path(image).exists():
            raw = Path(image).read_bytes()
        else:
            # Let Streamlit handle other types; here we skip persist
            raw = None
        b64 = base64.b64encode(raw).decode("utf-8") if raw else None
        bid = self._next_id("image")
        self.blocks.append(Block(id=bid, section=self.section, kind="image", image_b64_png=b64))

    # ---------- dataset attachment ----------
    def attach_uploaded(self, file_bytes: bytes, file_name: str, df: pd.DataFrame):
        self.uploaded_file_bytes = file_bytes
        self.uploaded_file_name = file_name
        self.uploaded_dataframe = df

    # ---------- write to disk ----------
    def write_snapshot(self, base_folder: Path = Path("KnowledgeBase")) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        root = base_folder / ts
        d_text = root / "text"
        d_tbl = root / "tables"
        d_fig = root / "figures"
        d_img = root / "images"
        d_meta = root / "meta"
        d_data = root / "data"
        for d in (d_text, d_tbl, d_fig, d_img, d_meta, d_data):
            _ensure_dir(d)

        manifest: Dict[str, Any] = {
            "created": ts,
            "sections": [],
            "blocks": [],
        }

        # write blocks in order
        for blk in self.blocks:
            if blk.kind == "md":
                (d_text / f"{blk.id}.md").write_text(blk.text or "", encoding="utf-8")
            elif blk.kind == "table":
                (d_tbl / f"{blk.id}.csv").write_text(blk.table_csv or "", encoding="utf-8")
            elif blk.kind in ("figure", "image") and blk.image_b64_png:
                (d_fig if blk.kind == "figure" else d_img).joinpath(f"{blk.id}.png").write_bytes(
                    base64.b64decode(blk.image_b64_png)
                )
            manifest["blocks"].append({
                "id": blk.id,
                "section": blk.section,
                "kind": blk.kind,
                "title": blk.title,
                "meta": blk.meta,
            })

        # save dataset (original + sanitized CSV + simple profile)
        if self.uploaded_file_bytes and self.uploaded_file_name:
            (d_data / f"{self.uploaded_file_name}").write_bytes(self.uploaded_file_bytes)
        if isinstance(self.uploaded_dataframe, pd.DataFrame) and not self.uploaded_dataframe.empty:
            san = _sanitize_for_arrow(self.uploaded_dataframe)
            san.to_csv(d_data / "uploaded_sanitized.csv", index=False)
            # quick schema profile
            prof = pd.DataFrame({
                "column": san.columns,
                "dtype": [str(san[c].dtype) for c in san.columns],
                "non_null": [int(san[c].notna().sum()) for c in san.columns],
                "unique": [int(san[c].nunique(dropna=True)) for c in san.columns],
            })
            prof.to_csv(d_data / "schema_profile.csv", index=False)

        # index.html for quick viewing
        index_html = [
            "<html><head><meta charset='utf-8'><title>KB Snapshot</title></head><body>",
            f"<h2>Snapshot {ts}</h2>",
        ]
        for blk in self.blocks:
            index_html.append(f"<h3 id='{blk.id}'>[{blk.section}] {blk.id}</h3>")
            if blk.kind == "md":
                # do not render markdown, show as pre
                md_txt = (blk.text or "").replace("<", "&lt;").replace(">", "&gt;")
                index_html.append(f"<pre style='background:#f7f7f8;padding:8px;border-radius:8px'>{md_txt}</pre>")
            elif blk.kind == "table":
                index_html.append(f"<p>Table: tables/{blk.id}.csv</p>")
            elif blk.kind == "figure":
                index_html.append(f"<img alt='{blk.id}' src='figures/{blk.id}.png' style='max-width:100%'>")
            elif blk.kind == "image":
                index_html.append(f"<img alt='{blk.id}' src='images/{blk.id}.png' style='max-width:100%'>")
        index_html.append("</body></html>")
        (root / "index.html").write_text("\n".join(index_html), encoding="utf-8")

        # manifest
        (d_meta / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "README.md").write_text(
            "# KnowledgeBase Snapshot\n\n"
            "This folder captures the exact sequence of markdown and outputs as shown in the app.\n\n"
            "- `text/NNNN-md.md` — narrative blocks in order.\n"
            "- `tables/NNNN-table.csv` — dataframes, aligned with the preceding text section.\n"
            "- `figures/NNNN-figure.png` — plots, aligned with the preceding text section.\n"
            "- `images/` — other images.\n"
            "- `data/` — uploaded dataset (original + sanitized CSV + schema profile).\n",
            encoding="utf-8",
        )

        # best-effort latest copy (no symlink on Windows by default)
        try:
            latest = base_folder / "latest"
            if latest.exists():
                if latest.is_dir():
                    for p in latest.iterdir():
                        if p.is_file():
                            p.unlink()
                else:
                    latest.unlink()
            if not latest.exists():
                latest.mkdir(parents=True)
            # write a small pointer
            (latest / "_pointer.txt").write_text(ts, encoding="utf-8")
        except Exception:
            pass

        return root


# ======================= Sidebar / Upload =======================

def sidebar(cap: CaptureSession) -> Optional[pd.DataFrame]:
    with st.sidebar:
        st.header("📂 Data Upload")
        up = st.file_uploader(
            "Upload CSV / Excel / JSON / Parquet / XML",
            type=["csv", "xlsx", "xls", "json", "parquet", "xml"],
        )
        xml_xpath = st.text_input("XML row path (optional)", value="")

        if not up:
            st.info("Upload a dataset to continue.")
            return None

        # read
        raw = up.getvalue()
        name = up.name
        ext = Path(name).suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(raw))
            elif ext in {".xlsx", ".xls"}:
                df = pd.read_excel(io.BytesIO(raw))
            elif ext == ".json":
                try:
                    df = pd.read_json(io.BytesIO(raw), lines=True)
                except Exception:
                    df = pd.read_json(io.BytesIO(raw))
            elif ext == ".parquet":
                df = pd.read_parquet(io.BytesIO(raw))
            elif ext == ".xml":
                df = pd.read_xml(io.BytesIO(raw), xpath=xml_xpath or None)
            else:
                df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None

        # stash for snapshot
        cap.attach_uploaded(raw, name, df)
        return df


# ======================= Main Page =======================

def main():
    cap = CaptureSession()

    st.title("Forecast360 — Minimal Demo")
    cap.set_section("Intro")
    cap.markdown(
        """
        ### Welcome
        Upload a dataset on the left. We'll show a quick profile and a simple plot.
        When ready, click **📸 Capture Snapshot…** to save this page's *text and outputs in order*.
        """
    )

    df = sidebar(cap)
    if df is None:
        return

    # ---------- simple cleaning/profile ----------
    cap.set_section("Profile")
    cap.markdown("#### Dataset Overview")
    nrows, ncols = df.shape
    prof = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "NonNull": [int(df[c].notna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    cap.dataframe(prof, use_container_width=True, height=280)

    # ---------- preview ----------
    cap.markdown("#### Preview (head)")
    cap.dataframe(df.head(10), use_container_width=True, height=260)

    # ---------- simple plot if numeric ----------
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        col = st.selectbox("Numeric column for quick histogram", num_cols)
        cap.set_section("Quick Plot")
        cap.markdown(f"#### Histogram: `{col}`")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(df[col].dropna().values, bins=30)
        ax.set_title(f"Histogram of {col}")
        fig.tight_layout()
        cap.pyplot(fig, clear_figure=True)
    else:
        cap.set_section("Quick Plot")
        cap.markdown("No numeric columns found for histogram preview.")

    st.divider()
    # ============== Snapshot button ==============
    if st.button("📸 Capture Snapshot…", type="primary"):
        out_dir = cap.write_snapshot(Path("KnowledgeBase"))
        st.success(f"Snapshot written to: {out_dir}")

    # ============== Optional: LangChain Summarize ==============
    with st.expander("🔎 (Optional) Summarize latest snapshot using LangChain"):
        if not HAVE_LANGCHAIN:
            st.info("LangChain not installed — skipping. Install 'langchain' to enable.")
        else:
            kb = Path("KnowledgeBase")
            if not kb.exists():
                st.info("No snapshots yet. Click the camera button above first.")
            else:
                # pick latest by timestamp folder name
                snaps = sorted([p for p in kb.iterdir() if p.is_dir() and re.match(r"\d{8}-\d{6}", p.name)], reverse=True)
                if not snaps:
                    st.info("No snapshots yet. Click the camera button above first.")
                else:
                    latest = snaps[0]
                    texts = []
                    for p in (latest / "text").glob("*.md"):
                        texts.append(p.read_text(encoding="utf-8"))
                    # also include lightweight captions for tables/figures (filenames convey order)
                    for p in sorted((latest / "tables").glob("*.csv")):
                        texts.append(f"[TABLE] {p.name}")
                    for p in sorted((latest / "figures").glob("*.png")):
                        texts.append(f"[FIGURE] {p.name}")

                    docs = [Document(page_content=t) for t in texts]
                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)

                    prompt = PromptTemplate(
                        input_variables=["context"],
                        template=(
                            "You are an analyst. Given the snapshot context, write a concise summary "
                            "that pairs explanations with outputs. Mention important columns and any plots.\n\n"
                            "Context:\n{context}\n\nSummary:"),
                    )
                    # Example LLM placeholder: replace with your configured LLM
                    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
                    # chain = LLMChain(llm=llm, prompt=prompt)
                    # Here we just join chunks to simulate without an API call
                    summary_text = ("\n\n".join(d.page_content for d in chunks))[:4000]
                    st.text_area("Snapshot summary (mocked)", summary_text, height=240)


if __name__ == "__main__":
    main()
