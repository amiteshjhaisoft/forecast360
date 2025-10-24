# Author: Amitesh Jha | iSOFT

# from __future__ import annotations

# # ---- Standard Library
# import os
# import sys
# import io
# import re
# import json
# import time
# import base64
# import hashlib
# import logging
# import shutil
# import tempfile
# from pathlib import Path
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Literal
# from datetime import datetime
# import pytz

# # ---- Environment / Config
# from dotenv import load_dotenv

# # ---- Web / IO utilities
# import requests
# import httpx

# # ---- Streamlit UI
# import streamlit as st

# # ---- Core Data
# import numpy as np
# import pandas as pd
# import pyarrow as pa

# # ---- Visualization
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from PIL import Image

# # ---- File Parsing / Ingestion
# from pypdf import PdfReader
# import openpyxl          # .xlsx reader
# import xlrd              # .xls reader
# # import pyxlsb          # .xlsb (enable if you actually use it)
# import yaml
# import csv
# import zipfile

# # ---- Classic ML / Stats
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import statsmodels.api as sm

# # ===========================
# # LangChain / LangGraph stack
# # ===========================

# # ---- LangChain Core
# from langchain_core.documents import Document
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import (
#     Runnable,
#     RunnableMap,
#     RunnableLambda,
#     RunnableParallel,
# )
# from langchain_core.callbacks import BaseCallbackHandler

# # ---- Text Splitters & Loaders (only those that match installed deps)
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import (
#     PyPDFLoader,
#     TextLoader,
#     CSVLoader,
#     JSONLoader,
#     # UnstructuredExcelLoader,          # requires 'unstructured' (not installed)
#     # UnstructuredHTMLLoader,           # requires 'unstructured' (not installed)
# )

# # ---- Vector Store (FAISS) & Embeddings
# from langchain_community.vectorstores import FAISS

# # Guarded: HuggingFaceEmbeddings typically needs sentence-transformers/torch.
# # Keep optional so import doesn't crash if those aren't installed.
# try:
#     from langchain_community.embeddings import HuggingFaceEmbeddings  # optional
# except Exception:  # pragma: no cover
#     HuggingFaceEmbeddings = None  # type: ignore

# # ---- Claude (only) LLM
# from langchain_anthropic import ChatAnthropic

# # ---- Tools / Agents (optional)
# # --- compatibility import for LangChain Tool ---
# try:
#     # Newer LangChain (>= 0.2.x)
#     from langchain_core.tools import Tool
# except Exception:
#     try:
#         # Mid-era
#         from langchain.tools import Tool
#     except Exception:
#         # Older LangChain
#         from langchain.agents import Tool

# # from langchain.agents import AgentExecutor, create_react_agent

# # ---- Memory / History
# from langchain.memory import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # ---- LangGraph (state graph for agents / workflows)
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

# # ---- Tracing / Observability (optional)
# try:
#     from langsmith import Client as LangSmithClient
# except Exception:
#     LangSmithClient = None  # type: ignore

# ---- Standard Library
import os
import io
import re
import json
import base64
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# ---- Streamlit UI
import streamlit as st

# ---- Core Data
import numpy as np
import pandas as pd

# ---- Visualization
import matplotlib.pyplot as plt
from PIL import Image




from kb_capture import KBCapture, kb_set_block
from kb_sync_azure import sync_folder_to_blob

# AI Agent
from forecast360_chat import run as run_ai_agent

# Patch once so ALL renders/uploads are captured
if "kb" not in st.session_state:
    st.session_state.kb = KBCapture(folder_name="KB").patch()  # <- no keep_last
kb = st.session_state.kb

# ======================= Forecast360 — Unified Tooltips (widgets + content) =======================
# Paste once after st.set_page_config(...) or right after `import streamlit as st`

import html
import streamlit as st
from typing import Any, Callable, Dict, Optional

# ------------------------------- 1) WIDGET TOOLTIP PATCH ---------------------------------

# Exact label -> tooltip (aligned to labels used in this app)
_TOOLTIP_MAP: Dict[str, str] = {
    # Upload & parsing
    "Upload CSV / Excel / JSON / Parquet / XML":
        "Upload a single data file. Supported: CSV, Excel, JSON, Parquet, XML.",
    "XML row path (optional XPath)":
        "XPath pointing at each repeated record node. Examples: .//row, .//record, or .//item.",
    "Date/Time column (auto-detected)":
        "The timestamp column used to order/index your time series.",
    "Target column (numeric, auto-detected)":
        "Numeric measure to analyse and forecast (e.g., Sales, Demand).",

    # Resampling & missing data
    "Resample frequency":
        "Resample by SUM to D/W/M/Q; choose 'raw' to keep original granularity.",
    "Missing values":
        "How to handle gaps after resampling or in raw data.",
    "Constant value (for missing)":
        "Used only when Missing values = constant. Fills all gaps with this number.",

    # Exogenous features
    "🧩 Exogenous features (for ARIMAX/SARIMAX/Auto_ARIMA/Tree/TFT)":
        "Optional driver variables (calendar features, chosen columns, and lags).",
    "Use calendar features (dow, month, month-start/end, weekend)":
        "Adds calendar features aligned to the series index.",
    "Exog columns by name (comma or JSON list)":
        "Driver columns from your dataset (e.g., price, promo). Use comma or JSON list.",
    "Scale exog":
        "Standardize exogenous features (mean 0, std 1) for stability.",
    "Exog lags":
        "Create lagged versions of exogenous features (0 keeps contemporaneous).",
    "Additional numeric exogenous columns (optional)":
        "Pick extra numeric columns to include as drivers.",

    # Pattern, transforms, outliers
    "Additive vs Multiplicative pattern":
        "Multiplicative if variability rises with the level; otherwise additive.",
    "Seasonal period (m)":
        "Seasonal cycle length: 'auto' or explicit (e.g., 7, 12, 24, 52, 365).",
    "Target transform":
        "Variance-stabilizing transform (log1p/Box–Cox); reversed for outputs.",
    "Winsorize outliers":
        "Clip extremes to reduce outlier impact.",
    "Outlier z-threshold (z)":
        "Higher z keeps more extremes (e.g., 3.5 = fairly tolerant).",

    # CV / holdout
    "Folds":
        "Number of rolling-origin folds for backtesting.",
    "Horizon (H)":
        "Forecast steps per fold (evaluation window).",
    "Gap":
        "Gap between train end and test start to reduce leakage.",
    "Metrics":
        "Primary/secondary error metrics to evaluate models.",

    # Models (selectors)
    "ARMA": "Autoregressive Moving Average (non-differenced).",
    "ARIMA": "ARIMA(p,d,q) with differencing (univariate).",
    "ARIMAX": "ARIMA with exogenous regressors.",
    "SARIMA": "Seasonal ARIMA with seasonal orders (univariate).",
    "SARIMAX": "Seasonal ARIMA with exogenous regressors.",
    "Auto_ARIMA": "Auto-selects ARIMA/SARIMA orders.",
    "HWES": "Exponential smoothing (trend/seasonal).",
    "Prophet": "Additive model with trend/seasonality/holidays.",
    "TBATS": "Multiple/long seasonalities.",
    "XGBoost": "Gradient-boosted trees with lag/covariates.",
    "LightGBM": "LightGBM regressor with lag/covariates.",
    "TFT": "Temporal Fusion Transformer (via Darts).",

    # Per-model knobs
    "p (ARMA)": "AR order (autoregressive terms).",
    "q (ARMA)": "MA order (moving-average terms).",
    "trend (ARMA)": "Deterministic trend: n=None, c=const, t=trend, ct=both.",
    "p (ARIMA)": "AR order.",
    "d (ARIMA)": "Differencing order.",
    "q (ARIMA)": "MA order.",
    "trend (ARIMA)": "Deterministic trend for ARIMA.",
    "p": "Non-seasonal AR order.",
    "d": "Non-seasonal differencing.",
    "q": "Non-seasonal MA order.",
    "P": "Seasonal AR order.",
    "D": "Seasonal differencing.",
    "Q": "Seasonal MA order.",
    "m (SARIMA)": "Seasonal period (m). 'auto' uses detected/chosen m.",
    "trend": "Deterministic trend component.",
    "m (SARIMAX)": "Seasonal period (m).",
    "seasonal_periods (JSON list)":
        "TBATS seasonal periods, e.g., [7, 365.25].",

    # XGB / LGBM knobs
    "n_estimators": "Number of boosting trees.",
    "max_depth": "Maximum tree depth (complexity).",
    "learning_rate": "Shrinkage rate; lower is safer but needs more trees.",
    "subsample": "Row sampling fraction per tree.",
    "colsample_bytree": "Feature sampling fraction per tree.",
    "reg_alpha": "L1 regularization (sparsity).",
    "reg_lambda": "L2 regularization (stability).",
    "num_leaves": "Max leaves per LightGBM tree.",
    "feature_fraction": "Feature sampling per LightGBM tree.",
    "bagging_fraction": "Row sampling per LightGBM iteration.",
    "bagging_freq": "How often to bag (0=off).",
    "lambda_l1": "L1 regularization (LightGBM).",
    "lambda_l2": "L2 regularization (LightGBM).",
    "min_data_in_leaf": "Minimum samples per leaf.",

    # Profile / viz
    "Preview rows": "How many rows to preview (max 25).",
    "Numeric": "Numeric column for the boxplot.",
    "Category": "Categorical column used to group values.",
    "Palette": "Matplotlib palette for boxplot categories.",
    "Method": "Correlation method: Pearson, Spearman, or Kendall.",
    "Show heatmap": "Toggle the correlation heatmap.",
    "Columns (optional subset)": "Limit correlation to selected numeric columns.",
}

def _infer_tooltip(label: Any) -> str:
    if not label:
        return "Hover for guidance on how this input affects the analysis."
    lab = str(label).strip()
    if lab in _TOOLTIP_MAP:
        return _TOOLTIP_MAP[lab]
    l = lab.lower()
    if "xpath" in l:
        return "XPath to each repeated record node (e.g., .//row, .//record, .//item)."
    if any(k in l for k in ("date", "time", "timestamp")):
        return "Select the timestamp column used as the index."
    if any(k in l for k in ("target", "value", "measure", "y ")):
        return "Numeric measure to model and forecast."
    if "freq" in l or "resample" in l:
        return "Sampling frequency (D/W/M/Q). Choose 'raw' to keep original."
    if "horizon" in l or "steps" in l:
        return "Number of periods to forecast ahead."
    if "season" in l and ("(m)" in l or l.strip() in {"m", "seasonal period", "seasonal_periods"}):
        return "Seasonal cycle length (e.g., 7 weekly, 12 monthly)."
    if any(k in l for k in ("upload", "file", "data")):
        return "Upload a data file (CSV/Excel/JSON/Parquet/XML)."
    if any(k in l for k in ("exog", "feature", "driver")):
        return "Optional driver variables aligned with the series; can be lagged and scaled."
    if "missing" in l:
        return "How gaps are handled (forward/backfill, interpolation, constant, etc.)."
    if "metric" in l:
        return "Error metrics used to compare models."
    if "palette" in l:
        return "Pick a matplotlib palette for category colors."
    return "Hover for guidance on how this input affects the analysis."

def _with_default_help(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(label: Any, *args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("help"):
            kwargs["help"] = _infer_tooltip(label)
        return fn(label, *args, **kwargs)
    return wrapped

def install_unified_tooltips() -> None:
    """Patch common Streamlit inputs with default help= tooltips (idempotent)."""
    if getattr(st.session_state, "_tooltips_patched", False):
        return
    for _name in (
        "text_input", "number_input", "selectbox", "multiselect", "checkbox",
        "slider", "date_input", "file_uploader", "radio", "text_area",
        "time_input", "select_slider"
    ):
        if hasattr(st, _name):
            setattr(st, _name, _with_default_help(getattr(st, _name)))
    st.session_state["_tooltips_patched"] = True

# ------------------------------- 2) CONTENT TOOLTIP PATCH ---------------------------------

# Map for Getting Started / content sections
_CONTENT_TOOLTIP_MAP: Dict[str, str] = {
    "Getting Started": "Quick steps to ingest data, profile it, and run forecasts.",
    "Upload Data": "Load CSV/Excel/JSON/Parquet/XML to begin analysis.",
    "Data Preview": "A quick look at your dataset to confirm parsing and columns.",
    "Profiling": "Basic stats, missingness, and distributions.",
    "Visualization": "Exploratory plots to spot trends, seasonality, and outliers.",
    "Exogenous Features": "Optional drivers (calendar/features) to enrich models.",
    "Backtesting": "Rolling-origin evaluation to estimate real forecast accuracy.",
    "Models": "Choose and configure models to compare performance.",
    "Forecasts": "Generate predictions with confidence intervals.",
    "Knowledge Base": "Captures artifacts (tables/figs/html) into the KB folder.",
}

def _infer_content_tooltip(text: Any) -> str:
    if not text:
        return "Section description and usage tips."
    s = str(text).strip()
    if s in _CONTENT_TOOLTIP_MAP:
        return _CONTENT_TOOLTIP_MAP[s]
    l = s.lower()
    if any(k in l for k in ("upload", "ingest", "load")):
        return "Provide a supported file to begin (CSV/Excel/JSON/Parquet/XML)."
    if any(k in l for k in ("preview", "sample", "head")):
        return "Preview your data to verify parsing and column types."
    if any(k in l for k in ("profile", "profiling", "summary", "describe")):
        return "Data profile: types, missingness, basic stats, and distributions."
    if any(k in l for k in ("visual", "chart", "plot", "graph", "heatmap")):
        return "Exploratory visualizations to spot trends and outliers."
    if any(k in l for k in ("exog", "feature", "driver", "covariate")):
        return "Optional driver variables aligned to your time index; can be lagged."
    if any(k in l for k in ("backtest", "cv", "fold", "validation", "holdout")):
        return "Rolling-origin evaluation to estimate out-of-sample accuracy."
    if any(k in l for k in ("model", "arima", "prophet", "xgboost", "lightgbm", "tft", "sarima")):
        return "Model selection/tuning for your time series forecasting."
    if any(k in l for k in ("forecast", "prediction", "horizon", "interval")):
        return "Generate future values with uncertainty bounds."
    if any(k in l for k in ("knowledge base", "kb", "snapshot")):
        return "Persist artifacts into the Knowledge Base for RAG/DI."
    return "Section description and usage tips."

def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)

def _patch_heading_api() -> None:
    """Monkey-patch title/header/subheader to render with native browser tooltip (idempotent)."""
    if getattr(st.session_state, "_content_tooltips_headings_patched", False):
        return

    def _mk_heading_wrapper(tag: str) -> Callable[..., Any]:
        def _wrapped(body: Any, *args, **kwargs) -> None:
            tooltip: Optional[str] = kwargs.pop("tooltip", None)
            tip = tooltip or _infer_content_tooltip(body)
            st.markdown(
                f'<{tag} class="f360-heading" title="{_esc(tip)}">{_esc(body)}</{tag}>',
                unsafe_allow_html=True,
            )
        return _wrapped

    # Patch headings
    st.title     = _mk_heading_wrapper("h1")   # type: ignore[assignment]
    st.header    = _mk_heading_wrapper("h2")   # type: ignore[assignment]
    st.subheader = _mk_heading_wrapper("h3")   # type: ignore[assignment]

    # Subtle CSS hint for hover
    st.markdown(
        """
        <style>
          .f360-heading[title] { cursor: help; }
          .f360-md-tip { display:inline-block; margin-left:.35rem; cursor:help; user-select:none; }
          .f360-md-tip:hover { opacity:.85; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_content_tooltips_headings_patched"] = True

def md(markdown_text: str, tooltip: Optional[str] = None, **st_markdown_kwargs: Any) -> None:
    """
    Render markdown plus an inline info glyph `ⓘ` that shows a tooltip on hover.
    Usage: md("**Step 1:** Upload Data", tooltip="Supported: CSV/Excel/JSON/Parquet/XML.")
    """
    st.markdown(markdown_text, **st_markdown_kwargs)
    tip = tooltip or _infer_content_tooltip(markdown_text)
    st.markdown(f'<span class="f360-md-tip" title="{_esc(tip)}">ⓘ</span>', unsafe_allow_html=True)

def install_content_tooltips() -> None:
    _patch_heading_api()

# ------------------------------- 3) INSTALL BOTH PATCHES ---------------------------------
install_unified_tooltips()     # widget tooltips
install_content_tooltips()     # title/header/subheader + md() helper
# ======================= /end Forecast360 — Unified Tooltips =======================



def render_kb_footer():
    """
    Hardcoded KB footer (no button):
    - Enforces folder=KB, keep_last=3, ai_summary=True
    - Auto-flushes once per render *only if new items were captured*
    """
    kb = st.session_state.get("kb")
    if kb is None:
        st.warning("KB capture is not initialized.")
        return

    # Enforce constants (hard lock)
    kb.folder_name = "KB"
    kb.keep_last   = 30 # <- fixed

    st.markdown("### 📦 Knowledge Base Snapshot")
    st.caption("Auto-saving this page’s captured uploads, tables, and plots…")

    # only flush if something new appeared since last flush
    items = getattr(kb, "_items", [])
    if not items:
        st.info("Nothing to save yet.")
        return

    last_ts = items[-1].ts if hasattr(items[-1], "ts") else ""
    sig = f"{len(items)}::{last_ts}"

    if st.session_state.get("_kb_last_sig") == sig:
        st.caption("No changes since last snapshot.")
        return

    out_dir = kb.flush(ai_summary=True)
    st.session_state["_kb_last_sig"] = sig
    st.success(f"Snapshot saved to: {out_dir}")



def page_home():
    # -- Cache image -> base64 so reruns are fast
    @st.cache_data(show_spinner=False)
    def _img_to_base64_first(paths):
        for p in paths:
            fp = Path(p)
            if fp.is_file():
                try:
                    return base64.b64encode(fp.read_bytes()).decode("utf-8")
                except Exception:
                    continue
        return None

    img_b64 = _img_to_base64_first(["assets/forecast360.png"])

    # -- Scoped CSS (home only)
    # -- Styles
    st.markdown(
        """
        <style>
        .home-wrap{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(0,183,255,.10), transparent 60%),
                        radial-gradient(1200px 600px at 110% 110%, rgba(255,79,160,.08), transparent 60%),
                        linear-gradient(135deg, rgba(255,136,0,.06), rgba(0,183,255,.06) 50%, rgba(255,79,160,.06));
            border: 1px solid #eaeaea; border-radius: 22px; padding: 22px 22px 14px; margin-bottom: 14px;
            box-shadow: 0 10px 24px rgba(0,0,0,.04);
        }
        .home-cols{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 26px; align-items: center; }
        .home-left h1{ margin: 0 0 8px; font-weight: 800; letter-spacing: .2px; }
        .home-left h5{ margin: 0 0 10px; font-weight: 600; color:#0f172a; opacity:.85; }
        .home-left p{ margin: 0 0 10px; color: #334155; line-height:1.5; }

        .home-right{ display:flex; align-items:center; justify-content:center; }
        .logo-wrap{
            width: min(360px, 90%);
            aspect-ratio: 1 / 1;
            display:flex; align-items:center; justify-content:center;
            background: radial-gradient(60% 60% at 50% 45%, rgba(255,255,255,.25), transparent 70%);
            border-radius: 50%;
            box-shadow: 0 20px 40px rgba(2, 6, 23, 0.12), inset 0 1px 0 rgba(255,255,255,.3);
        }
        .logo-wrap img{
            width: 100%; height: auto; display:block;
            filter: drop-shadow(0 10px 24px rgba(2, 6, 23, 0.16));
            border-radius: 50%;
        }

        .kpis{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; }
        .kcard{
            background:#fff; border:1px solid #eee; border-radius:16px; padding:14px 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,.05);
        }
        .kcard h4{ margin:0 0 6px; font-size:18px; }
        .kcard p{ margin:0; color:#475569; font-size:13px; }

        /* Small, subtle footer inside the hero */
        .home-footer{
            margin-top: 8px;
            text-align: center;
            font-size: 12px;
            line-height: 1.4;
            opacity: .75;
        }
        @media (max-width: 1100px){
            .home-cols{ grid-template-columns: 1fr; gap: 18px; }
            .logo-wrap{ width:min(300px, 70%); margin: 6px auto 0; }
            .kpis{ grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 680px){
            .kpis{ grid-template-columns: 1fr; }
            .home-footer{ font-size: 11px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -- Hero
    st.markdown(
        f"""
        <div class="home-wrap">
        <div class="home-cols">
            <div class="home-left">
            <h1 class="hero-title">Forecast360</h1>
            <h5>AI Powered Forecasting. No Code. Decisions in Minutes.</h5>
            <p>Upload any time series, auto-profile, compare models, forecast with confidence intervals,
                and turn it into <b>actionable decisions</b> with an AI analyst, speech and a talking avatar.</p>
            <p>Bring your own local LLM via <b>Ollama</b> or connect <b>Claude.ai</b>.
                Data stays local; artifacts live in your local <b>Knowledge Base</b>.</p>
            </div>
            <div class="home-right">
            <div class="logo-wrap">
                {("<img src='data:image/png;base64," + img_b64 + "' alt='Forecast360 logo'/>") if img_b64 else ""}
            </div>
            </div>
        </div>

        <div class="home-footer">© {datetime.now().year} iSOFT ANZ Pvt Ltd. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    icon_path = Path("assets/forecast360.png")
    page_icon = Image.open(icon_path) if icon_path.is_file() else "📈"

    st.set_page_config(
        page_title="Forecast360",
        page_icon=page_icon,
        layout="wide",
    )
    page_home()  # make sure page_home() is imported/defined

if __name__ == "__main__":
    main()


# st.divider()

# --- Render the CTA button ONLY if sidebar not shown yet ---
if not st.session_state.get("show_sidebar", False):

    # Inject styles once
    if not st.session_state.get("_home_btn_css"):
        st.session_state["_home_btn_css"] = True
        st.markdown("""
        <style>
          /* shared center-row so this aligns with copyright block */
          .hero-center-row{display:flex;justify-content:center;margin:18px 0 6px;}

          /* pretty pill button */
          .hero-center-row .stButton>button{
            padding:10px 28px;font-weight:700;font-size:16px;letter-spacing:.2px;border:0;border-radius:999px;color:#fff;
            background:linear-gradient(135deg,#0b1f3a 0%,#12497b 55%,#2a7cc4 100%);
            box-shadow:0 10px 22px rgba(18,73,123,.28),0 2px 6px rgba(0,0,0,.06);
            transition:transform .12s ease, box-shadow .12s ease, filter .12s ease; cursor:pointer;
          }
          .hero-center-row .stButton>button:hover{transform:translateY(-1px);box-shadow:0 14px 30px rgba(18,73,123,.32),0 4px 10px rgba(0,0,0,.08);filter:brightness(1.06) saturate(1.05);}
          .hero-center-row .stButton>button:active{transform:translateY(0);box-shadow:0 8px 18px rgba(18,73,123,.22),0 2px 6px rgba(0,0,0,.06);}
          .hero-center-row .stButton>button:focus{outline:none;}
          .hero-center-row .stButton>button:focus-visible{box-shadow:0 0 0 4px rgba(34,197,94,.35),0 10px 22px rgba(18,73,123,.28);}
        </style>
        """, unsafe_allow_html=True)

    # Centered exactly in the middle column so it lines up with copyright block
    btn_l, btn_c, btn_r = st.columns([1, 1, 1])
    with btn_c:
        st.markdown('<div class="hero-center-row">', unsafe_allow_html=True)
        if st.button("Let's Start Time Series Data Forecasting", key="start_btn"):
            st.session_state["show_sidebar"] = True
        st.markdown("</div>", unsafe_allow_html=True)


# ------ Sidebar Code -----

ACCEPTED_EXTS = ["csv", "xlsx", "xls", "json", "parquet", "xml"]

def sidebar_getting_started():
    """Sidebar content for Getting Started page (ONLY place with file upload)."""
    with st.sidebar:
        # ---- Branding --------------------------------------------------------
        if Path("assets/logo.png").exists():
            st.image("assets/logo.png", caption="iSOFT ANZ Pvt Ltd", use_container_width=True)

        st.subheader("🚀 Getting Started")

        # # ---- Data Upload -----------------------------------------------------
        # st.header("📂 Data Upload")
        # up = st.file_uploader(
        #     "Upload CSV / Excel / JSON / Parquet / XML",
        #     type=ACCEPTED_EXTS,
        #     accept_multiple_files=False,
        #     key="gs_file",
        # )
        # xml_xpath = st.text_input(
        #     "XML row path (optional XPath)",
        #     value=st.session_state.get("xml_xpath", ""),
        #     help="e.g., .//row  or  .//record  or  .//item",
        #     key="xml_xpath",
        # )
        # ---- Data Upload -----------------------------------------------------
        st.header("📂 Data Upload")
        up = st.file_uploader(
            "Upload CSV / Excel / JSON / Parquet / XML",
            type=ACCEPTED_EXTS,
            accept_multiple_files=False,
            key="gs_file",
        )
        
        # Normalize: some wrappers (or Streamlit internals) can return a list even when
        # accept_multiple_files=False. Ensure we work with a single UploadedFile or None.
        if isinstance(up, list):
            up = up[0] if up else None
        
        xml_xpath = st.text_input(
            "XML row path (optional XPath)",
            value=st.session_state.get("xml_xpath", ""),
            help="e.g., .//row  or  .//record  or  .//item",
            key="xml_xpath",
        )

        _df_in_state = st.session_state.get("uploaded_df")
        if up is None and not (isinstance(_df_in_state, pd.DataFrame) and not _df_in_state.empty):
            st.caption("Upload a file to unlock the rest of the sidebar settings.")
            return

        st.divider()
        st.markdown("**🔎 Column detection — Automatic**")

        # # ---- Read uploaded file ---------------------------------------------
        # _data, source_name = None, None
        # if up is not None:
        #     try:
        #         if "cached_read" in globals() and callable(globals().get("cached_read")):
        #             _data = cached_read(up.getvalue(), up.name, xml_xpath=xml_xpath)
        #         else:
        #             ext = Path(up.name).suffix.lower()
        #             raw = up.getvalue()
        #             if ext == ".csv":
        #                 _data = pd.read_csv(io.StringIO(raw.decode("utf-8", "ignore")))
        #             elif ext in {".xlsx", ".xls"}:
        #                 _data = pd.read_excel(up)
        #             elif ext == ".json":
        #                 _data = pd.read_json(io.BytesIO(raw))
        #             elif ext == ".parquet":
        #                 _data = pd.read_parquet(io.BytesIO(raw))
        #             elif ext == ".xml":
        #                 try:
        #                     _data = pd.read_xml(io.BytesIO(raw), xpath=xml_xpath or ".//row")
        #                 except Exception:
        #                     _data = pd.read_xml(io.BytesIO(raw))
        #             else:
        #                 st.warning(f"Unsupported extension: {ext}")
        #                 _data = None
        #         source_name = up.name
        #         st.session_state["raw_rows"], st.session_state["raw_cols"] = int(_data.shape[0]), int(_data.shape[1])
        #     except Exception as e:
        #         st.error(f"Failed to read file: {e}")
        #         _data, source_name = None, None

        # ---- Read uploaded file ---------------------------------------------
        _data, source_name = None, None
        
        # Normalize (defensive): sometimes a list sneaks through even with accept_multiple_files=False
        if isinstance(up, list):
            up = up[0] if up else None
        
        if up is not None:
            try:
                # Pull name + raw bytes safely
                source_name = getattr(up, "name", None)
                raw = (
                    up.getvalue() if hasattr(up, "getvalue")
                    else (up.read() if hasattr(up, "read") else None)
                )
                if raw is None:
                    raise ValueError("Uploaded object missing readable content")
        
                # Cached fast path (if you defined cached_read(bytes, name, xml_xpath=...))
                if "cached_read" in globals() and callable(globals().get("cached_read")):
                    _data = cached_read(raw, source_name or "uploaded", xml_xpath=xml_xpath)
                else:
                    ext = Path(source_name or "").suffix.lower()
        
                    if ext == ".csv":
                        # Try utf-8, then fallback to latin-1
                        try:
                            _data = pd.read_csv(io.StringIO(raw.decode("utf-8", "ignore")))
                        except Exception:
                            _data = pd.read_csv(io.StringIO(raw.decode("latin-1", "ignore")))
        
                    elif ext in {".xlsx", ".xls"}:
                        # Pandas can read from file-like buffer
                        _data = pd.read_excel(io.BytesIO(raw))
        
                    elif ext == ".json":
                        # Try standard JSON; if it fails, try lines
                        try:
                            _data = pd.read_json(io.BytesIO(raw))
                        except ValueError:
                            _data = pd.read_json(io.BytesIO(raw), lines=True)
        
                    elif ext == ".parquet":
                        _data = pd.read_parquet(io.BytesIO(raw))
        
                    elif ext == ".xml":
                        # Prefer provided XPath; fallback to pandas' default
                        try:
                            _data = pd.read_xml(io.BytesIO(raw), xpath=(xml_xpath or ".//row"))
                        except Exception:
                            _data = pd.read_xml(io.BytesIO(raw))
        
                    else:
                        st.warning(f"Unsupported extension: {ext or 'unknown'}")
                        _data = None
        
                # Record shape if we loaded a DataFrame successfully
                if _data is not None and hasattr(_data, "shape"):
                    st.session_state["raw_rows"] = int(_data.shape[0])
                    st.session_state["raw_cols"] = int(_data.shape[1])
        
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                _data, source_name = None, None

        
        # ---- Helpers ---------------------------------------------------------
        def _infer_target(df: pd.DataFrame) -> str:
            if df is None or df.empty or not len(df.columns):
                return ""
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return df.columns[-1]
            pri = ["value","amount","sales","target","y","qty","quantity","revenue","demand","passengers","count","units","volume"]
            for p in pri:
                for c in num_cols:
                    if p in c.lower():
                        return c
            return num_cols[0]

        def _infer_time_col_fallback(df: pd.DataFrame) -> str | None:
            if df is None or df.empty:
                return None
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    return c
            for c in df.columns:
                s = df[c]
                if s.dtype == "object":
                    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        return c
            return None

        if _data is None:
            _data = st.session_state.get("uploaded_df")
            source_name = st.session_state.get("source_name")

        if _data is None or not isinstance(_data, pd.DataFrame) or _data.empty:
            st.info("No data available yet. Upload a file above to detect columns.")
            return

        # ---------- Auto-detect Date/Time & Target ----------
        try:
            guess_date = (
                infer_time_col(_data)
                if "infer_time_col" in globals() and callable(globals().get("infer_time_col"))
                else _infer_time_col_fallback(_data)
            )
        except Exception:
            guess_date = _infer_time_col_fallback(_data)

        if guess_date is None or guess_date not in _data.columns:
            guess_date = _data.columns[0]

        guess_target = _infer_target(_data)

        date_col = st.selectbox(
            "Date/Time column (auto-detected)",
            list(_data.columns),
            index=list(_data.columns).index(guess_date) if guess_date in _data.columns else 0,
            key="date_col",
            help="Choose the timestamp column used to order the time series. If text dates, we’ll parse them for you.",
        )

        # --- Strict numeric target only ---
        numeric_cols = [c for c in _data.columns if pd.api.types.is_numeric_dtype(_data[c])]
        if not numeric_cols:
            st.error("No numeric columns found for the Target. Please ensure your dataset has at least one numeric column.")
            st.stop()

        target_options = numeric_cols
        target_index = target_options.index(guess_target) if guess_target in target_options else 0
        target_col = st.selectbox(
            "Target column (numeric, auto-detected)",
            options=target_options,
            index=target_index,
            key="target_col",
            help="This is the series you want to model/forecast (e.g., Sales). Only numeric columns are allowed.",
        )

        # --- Coerce & clean, track drops ---
        n0 = int(len(_data))
        _data[date_col] = pd.to_datetime(_data[date_col], errors="coerce", infer_datetime_format=True)
        bad_date = int(_data[date_col].isna().sum()); _data = _data.dropna(subset=[date_col])

        _data[target_col] = pd.to_numeric(_data[target_col], errors="coerce")
        bad_target = int(_data[target_col].isna().sum()); _data = _data.dropna(subset=[target_col])

        before_dedup = int(len(_data))
        exact_dupes = int(_data.duplicated(keep="last").sum())
        _data = _data.drop_duplicates(keep="last")
        dupes_dropped = exact_dupes

        st.session_state["drop_breakdown"] = {
            "Unparseable dates": bad_date,
            "Non-numeric target": bad_target,
            "Exact-duplicate rows removed": dupes_dropped,
        }
        st.session_state["clean_rows"], st.session_state["clean_cols"] = int(_data.shape[0]), int(_data.shape[1])
        st.session_state["uploaded_df"] = _data
        st.session_state["source_name"] = source_name
        st.session_state["__numeric_candidates"] = numeric_cols

        st.divider()

        # ---- Resampling & gaps ----------------------------------------------
        st.markdown("**⏳ Resampling & gaps**")
        freq_choices = ["raw", "D", "W", "M", "Q"]
        _default_freq = st.session_state.get("resample_freq", "W"); _default_freq = _default_freq if _default_freq in freq_choices else "W"
        freq_opt = st.selectbox("Resample frequency", freq_choices, index=freq_choices.index(_default_freq), help="Resample by sum. Set to 'raw' to keep original granularity.", key="resample_freq")

        mv_options = ["ffill","bfill","zero","mean","median","mode","interpolate_linear","interpolate_time","constant","drop","none"]
        _default_mv = st.session_state.get("missing_values", "median"); _default_mv = _default_mv if _default_mv in mv_options else "median"
        missing_values = st.selectbox("Missing values", mv_options, index=mv_options.index(_default_mv), key="missing_values", help="How to handle gaps after resampling or in the raw data.")
        st.session_state["fill_method"] = missing_values
        const_val = st.number_input("Constant value (for missing)", value=float(st.session_state.get("const_missing_value", 0.0)), format="%.4f", key="const_missing_value") if missing_values=="constant" else None

        st.divider()

        # ---- Exogenous features ---------------------------------------------
        st.header("🧩 Exogenous features (for ARIMAX/SARIMAX/Auto_ARIMA/Tree/TFT)")
        st.checkbox("Use calendar features (dow, month, month-start/end, weekend)", value=st.session_state.get("use_calendar_exog", True), key="use_calendar_exog")
        st.text_input("Exog columns by name (comma or JSON list)", value=st.session_state.get("exog_cols_text", ""), key="exog_cols_text")
        st.checkbox("Scale exog", value=st.session_state.get("scale_exog", True), key="scale_exog")
        st.multiselect("Exog lags", [0,1,7], default=st.session_state.get("exog_lags", [0,1,7]), key="exog_lags")
        if not st.session_state.get("__numeric_candidates"):
            st.caption("Upload data to choose additional exogenous columns.")
            st.session_state.setdefault("exog_additional_cols", [])
        else:
            st.multiselect("Additional numeric exogenous columns (optional)", options=st.session_state["__numeric_candidates"], default=st.session_state.get("exog_additional_cols", []), key="exog_additional_cols", help="These are inferred from the uploaded file’s numeric columns.")

        st.divider()

        # ---- Pattern & seasonality ------------------------------------------
        st.header("🌊 Pattern & seasonality")
        st.radio("Additive vs Multiplicative pattern", ["Auto-detect","Additive","Multiplicative"], index=["Auto-detect","Additive","Multiplicative"].index(st.session_state.get("pattern_type","Auto-detect")), key="pattern_type", help="Auto-detect uses rolling std/mean correlation; multiplicative if variability increases with level.")
        st.caption("Auto-detect uses rolling std/mean scaling; multiplicative suggested if variability grows with the level.")
        st.divider()

        # ---- Common knobs ----------------------------------------------------
        st.header("⚙️ Common knobs")
        st.text_input("Seasonal period (m)", value=st.session_state.get("seasonal_m","auto"), key="seasonal_m", help="auto or explicit like 7/12/24/52/365")
        st.selectbox("Target transform", ["none","log1p","boxcox"], index=["none","log1p","boxcox"].index(st.session_state.get("target_transform","none")), key="target_transform")
        st.checkbox("Winsorize outliers", value=st.session_state.get("winsorize", True), key="winsorize")
        st.number_input("Outlier z-threshold (z)", value=st.session_state.get("outlier_z", 3.5), step=0.1, key="outlier_z")

        st.subheader("Holdout / CV")
        st.number_input("Folds", value=st.session_state.get("cv_folds", 3), step=1, min_value=2, key="cv_folds")
        st.number_input("Horizon (H)", value=st.session_state.get("cv_horizon", 12), step=1, min_value=1, key="cv_horizon")
        st.number_input("Gap", value=st.session_state.get("cv_gap", 0), step=1, min_value=0, key="cv_gap")
        st.multiselect("Metrics", ["RMSE","MAE","MASE","MAPE","sMAPE"], default=st.session_state.get("cv_metrics",["RMSE","MAE","MASE","MAPE","sMAPE"]), key="cv_metrics")

        st.divider()

        # ---- Per-model settings ---------------------------------------------
        st.header("🧠 Per-model settings")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("ARIMA Family")
            sel_ARMA      = st.checkbox("ARMA",       value=st.session_state.get("sel_ARMA", True),    key="sel_ARMA")
            sel_ARIMA     = st.checkbox("ARIMA",      value=st.session_state.get("sel_ARIMA", True),   key="sel_ARIMA")
            sel_ARIMAX    = st.checkbox("ARIMAX",     value=st.session_state.get("sel_ARIMAX", True), key="sel_ARIMAX")
            sel_SARIMA    = st.checkbox("SARIMA",     value=st.session_state.get("sel_SARIMA", True), key="sel_SARIMA")
            sel_SARIMAX   = st.checkbox("SARIMAX",    value=st.session_state.get("sel_SARIMAX", True),key="sel_SARIMAX")
            sel_AutoARIMA = st.checkbox("Auto_ARIMA", value=st.session_state.get("sel_AutoARIMA", False), key="sel_AutoARIMA")

        with col_right:
            st.subheader("Other Models")
            sel_HWES    = st.checkbox("HWES",     value=st.session_state.get("sel_HWES", False),   key="sel_HWES")
            sel_Prophet = st.checkbox("Prophet",  value=st.session_state.get("sel_Prophet", False),key="sel_Prophet")
            sel_TBATS   = st.checkbox("TBATS",    value=st.session_state.get("sel_TBATS", False),  key="sel_TBATS")
            sel_XGB     = st.checkbox("XGBoost",  value=st.session_state.get("sel_XGB", False),    key="sel_XGB")
            sel_LGBM    = st.checkbox("LightGBM", value=st.session_state.get("sel_LGBM", False),   key="sel_LGBM")
            sel_TFT     = st.checkbox("TFT",      value=st.session_state.get("sel_TFT", False),    key="sel_TFT")

        if sel_ARMA:
            with st.expander("ARMA — Core", expanded=True):
                st.slider("p (ARMA)", 0, 5, st.session_state.get("arma_p", 1), key="arma_p")
                st.slider("q (ARMA)", 0, 5, st.session_state.get("arma_q", 1), key="arma_q")
                st.selectbox("trend (ARMA)", ["n","c","t","ct"], index=1, key="arma_trend")

        if sel_ARIMA:
            with st.expander("ARIMA — Core", expanded=True):
                st.slider("p (ARIMA)", 0, 5, st.session_state.get("arima_p", 1), key="arima_p")
                st.slider("d (ARIMA)", 0, 2, st.session_state.get("arima_d", 1), key="arima_d")
                st.slider("q (ARIMA)", 0, 5, st.session_state.get("arima_q", 1), key="arima_q")
                st.selectbox("trend (ARIMA)", ["n","c","t","ct"], index=1, key="arima_trend")

        if sel_SARIMA:
            with st.expander("SARIMA — Core", expanded=True):
                st.slider("p", 0, 3, st.session_state.get("sarima_p", 1), key="sarima_p")
                st.slider("d", 0, 1, st.session_state.get("sarima_d", 0), key="sarima_d")
                st.slider("q", 0, 3, st.session_state.get("sarima_q", 1), key="sarima_q")
                st.slider("P", 0, 3, st.session_state.get("sarima_P", 1), key="sarima_P")
                st.slider("D", 0, 1, st.session_state.get("sarima_D", 0), key="sarima_D")
                st.slider("Q", 0, 3, st.session_state.get("sarima_Q", 1), key="sarima_Q")
                st.text_input("m (SARIMA)", value=st.session_state.get("sarima_m", "auto"), key="sarima_m")
                st.selectbox("trend", ["n","c","t","ct"], index=1, key="sarima_trend")

        if sel_ARIMAX:
            with st.expander("ARIMAX — Core", expanded=True):
                st.slider("p", 0, 5, st.session_state.get("arimax_p", 1), key="arimax_p")
                st.slider("d", 0, 2, st.session_state.get("arimax_d", 1), key="arimax_d")
                st.slider("q", 0, 5, st.session_state.get("arimax_q", 1), key="arimax_q")
                st.selectbox("trend", ["n","c","t","ct"], index=1, key="arimax_trend")

        if sel_SARIMAX:
            with st.expander("SARIMAX — Core", expanded=True):
                st.slider("p", 0, 3, st.session_state.get("sarimax_p", 1), key="sarimax_p")
                st.slider("d", 0, 1, st.session_state.get("sarimax_d", 0), key="sarimax_d")
                st.slider("q", 0, 3, st.session_state.get("sarimax_q", 1), key="sarimax_q")
                st.slider("P", 0, 3, st.session_state.get("sarimax_P", 1), key="sarimax_P")
                st.slider("D", 0, 1, st.session_state.get("sarimax_D", 0), key="sarimax_D")
                st.slider("Q", 0, 3, st.session_state.get("sarimax_Q", 1), key="sarimax_Q")
                st.text_input("m (SARIMAX)", value=st.session_state.get("sarimax_m", "auto"), key="sarimax_m")

        if sel_HWES:
            with st.expander("HWES — Core", expanded=True):
                st.selectbox("trend", [None,"add","mul"], index=0, key="hwes_trend")
                st.selectbox("seasonal", [None,"add","mul"], index=0, key="hwes_seasonal")
                st.text_input("seasonal_periods", value=st.session_state.get("hwes_sp", "m"), key="hwes_sp")
                st.checkbox("damped_trend", value=st.session_state.get("hwes_damped", False), key="hwes_damped")

        if sel_Prophet:
            st.expander("Prophet — standard settings", expanded=False)

        if sel_TBATS:
            with st.expander("TBATS — Core", expanded=True):
                st.text_input("seasonal_periods (JSON list)", value=st.session_state.get("tbats_sp", "[7, 365.25]"), key="tbats_sp")

        if sel_XGB:
            with st.expander("XGBoost — Core", expanded=True):
                st.slider("n_estimators", 200, 1000, st.session_state.get("xgb_estimators", 500), key="xgb_estimators")
                st.slider("max_depth", 3, 8, st.session_state.get("xgb_depth", 5), key="xgb_depth")
                st.slider("learning_rate", 0.01, 0.1, st.session_state.get("xgb_lr", 0.05), key="xgb_lr")
                st.slider("subsample", 0.6, 1.0, st.session_state.get("xgb_subsample", 0.8), key="xgb_subsample")
                st.slider("colsample_bytree", 0.6, 1.0, st.session_state.get("xgb_colsample", 0.8), key="xgb_colsample")
                st.slider("reg_alpha", 0.0, 1.0, st.session_state.get("xgb_alpha", 0.1), key="xgb_alpha")
                st.slider("reg_lambda", 0.0, 3.0, st.session_state.get("xgb_lambda", 1.0), key="xgb_lambda")

        if sel_LGBM:
            with st.expander("LightGBM — Core", expanded=True):
                st.slider("num_leaves", 31, 255, st.session_state.get("lgbm_leaves", 64), key="lgbm_leaves")
                st.slider("n_estimators", 300, 1500, st.session_state.get("lgbm_estimators", 500), key="lgbm_estimators")
                st.slider("learning_rate", 0.01, 0.1, st.session_state.get("lgbm_lr", 0.05), key="lgbm_lr")
                st.slider("feature_fraction", 0.6, 1.0, st.session_state.get("lgbm_ff", 0.8), key="lgbm_ff")
                st.slider("bagging_fraction", 0.6, 1.0, st.session_state.get("lgbm_bf", 0.8), key="lgbm_bf")
                st.slider("bagging_freq", 1, 5, st.session_state.get("lgbm_bfreq", 1), key="lgbm_bfreq")
                st.slider("lambda_l1", 0.0, 1.0, st.session_state.get("lgbm_l1", 0.0), key="lgbm_l1")
                st.slider("lambda_l2", 0.0, 3.0, st.session_state.get("lgbm_l2", 0.0), key="lgbm_l2")
                st.slider("min_data_in_leaf", 20, 200, st.session_state.get("lgbm_minleaf", 50), key="lgbm_minleaf")

        st.divider()

# ------ Sidebar Code End -----  


def page_getting_started():
    """
    Full Getting Started page:
      - Summary, Preview, Profile, Boxplot-by-Category, Correlation
      - Preparation & Validation (line, histogram, rolling KPIs)
      - STL decomposition + diagnostics + ACF bars
      - Rolling CV Leaderboard (models autodetect)
      - Train best model on full series, Forecast & 80% intervals
      - Residual diagnostics: ACF, PACF, residual stats & plot
    Visuals use a single professional theme for consistency.
    """
    import json, io, math, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---------------------- Globals / Feature Flags ----------------------
    # Optional libs (graceful fallback if absent)
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.seasonal import STL
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf, adfuller
        from statsmodels.stats.diagnostic import acorr_ljungbox as _ljung
        HAVE_STATSM = True
    except Exception:
        HAVE_STATSM = False

    try:
        import pmdarima as pmd
        HAVE_PM = True
    except Exception:
        HAVE_PM = False

    try:
        from prophet import Prophet
        HAVE_PROPHET = True
    except Exception:
        HAVE_PROPHET = False

    try:
        from tbats import TBATS
        HAVE_TBATS = True
    except Exception:
        HAVE_TBATS = False

    try:
        import xgboost as xgb
        HAVE_XGB = True
    except Exception:
        HAVE_XGB = False

    try:
        import lightgbm as lgb
        HAVE_LGBM = True
    except Exception:
        HAVE_LGBM = False

    try:
        from darts import TimeSeries
        from darts.models import TFTModel
        HAVE_TFT = True
    except Exception:
        HAVE_TFT = False

    # ---------------------- Unified Professional Theme ----------------------
    # Fonts, colors and lines applied everywhere for visual consistency
    matplotlib.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor":   "#ffffff",
        "axes.edgecolor":   "#e5e7eb",
        "axes.grid":        True,
        "grid.color":       "#e5e7eb",
        "grid.alpha":       0.6,
        "axes.titleweight": "bold",
        "axes.titlecolor":  "#1f2a44",
        "axes.labelcolor":  "#374151",
        "xtick.color":      "#4b5563",
        "ytick.color":      "#4b5563",
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.prop_cycle":  plt.cycler(color=[
            "#4f46e5", "#06b6d4", "#ef4444", "#22c55e", "#f59e0b", "#0ea5e9", "#a855f7"
        ])
    })

    st.markdown("""
    <style>
      .block-card {
        background: linear-gradient(135deg, #fafbfd, #f6fbff);
        border: 1px solid #dce7f7;
        padding: 12px 14px; border-radius: 14px;
        box-shadow: 0 6px 18px rgba(31,42,68,.06);
        margin: 8px 0 14px 0;
      }
      .block-card h4 {margin: 0 0 6px 0; color:#1f2a44;}
      .small-note {font-size: 0.9rem; color: #6b7280;}
      .dataframe {font-size: 0.92rem;}
    </style>
    """, unsafe_allow_html=True)

    # ---------------------- Safe Helpers ----------------------
    def _st_df(df, **k):
        """Stable table renderer (Styler or DataFrame)."""
        try:
            if hasattr(df, "to_excel") and isinstance(df, pd.DataFrame):
                return st.dataframe(df, **k)
            else:
                return st.dataframe(df, **k)
        except TypeError:
            k.pop("hide_index", None); k.pop("use_container_width", None)
            return st.dataframe(df, **k)

    def _sanitize_for_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
        # Best-effort sanitization if caller had an Arrow helper
        try:
            if "_sanitize_for_arrow" in globals() and callable(globals()["_sanitize_for_arrow"]):
                return globals()["_sanitize_for_arrow"](df)
        except Exception:
            pass
        return df

    def _dedupe_index(df: pd.DataFrame, keep="last"):
        """Drop duplicate index values safely."""
        if not isinstance(df.index, pd.DatetimeIndex) and not isinstance(df.index, pd.Index):
            return df
        try:
            return df[~df.index.duplicated(keep=keep)]
        except Exception:
            return df

    def _safe_align_series_to_index(s: pd.Series, idx: pd.Index) -> pd.Series:
        try:
            s = s.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, errors="coerce")
            return s.reindex(idx)
        except Exception:
            return pd.Series(index=idx, dtype=float)

    def _profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        rows, n = [], int(len(df))
        for c in df.columns:
            s = df[c]
            is_num = pd.api.types.is_numeric_dtype(s)
            miss   = int(s.isna().sum())
            uniq   = int(s.nunique(dropna=True))
            pct_m  = round((miss / n) * 100, 2) if n else 0.0

            mean = median = stddev = min_v = max_v = mode_val = None
            if is_num and s.notna().any():
                s2 = pd.to_numeric(s, errors="coerce")
                good = s2.dropna()
                if len(good):
                    mean   = float(good.mean())
                    median = float(good.median())
                    stddev = float(good.std(ddof=1)) if len(good) > 1 else None
                    min_v  = float(good.min())
                    max_v  = float(good.max())
                    try:
                        m = good.mode()
                        mv = m.iloc[0] if len(m) else None
                        mode_val = float(mv) if mv is not None else None
                    except Exception:
                        mode_val = None

            rows.append({
                "Column": c,
                "Dtype":  str(s.dtype),
                "Count":  n,
                "Missing": miss,
                "Pct_missing": pct_m,
                "Unique": uniq,
                "Mean": mean, "Median": median, "Mode": mode_val,
                "Std_dev": stddev, "Min": min_v, "Max": max_v,
            })
        return pd.DataFrame(rows)

    def _align(a, b):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        n = min(len(a), len(b))
        return a[:n], b[:n]

    def _align_xy(x, y):
        n = min(len(x), len(y))
        return x[:n], np.asarray(y).reshape(-1)[:n]

    # Metrics
    def rmse(a, b):
        a, b = _align(a, b)
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")
    def mae(a, b):
        a, b = _align(a, b)
        return float(np.mean(np.abs(a - b))) if len(a) else float("nan")
    def mase(a, b, season=1):
        a, b = _align(a, b)
        season = max(1, int(season))
        if len(a) <= season:
            return np.nan
        naives = np.abs(a[season:] - a[:-season])
        denom = naives.mean() if len(naives) else np.nan
        return float(mae(a, b) / denom) if denom and denom > 0 else np.nan
    def mape(a, b):
        a, b = _align(a, b)
        denom = np.clip(np.abs(a), 1e-9, None)
        return float(np.mean(np.abs(a - b) / denom) * 100.0) if len(a) else float("nan")
    def smape(a, b):
        a, b = _align(a, b)
        denom = np.clip((np.abs(a) + np.abs(b)) / 2.0, 1e-9, None)
        return float(np.mean(np.abs(a - b) / denom) * 100.0) if len(a) else float("nan")

    # ---------------------- Read Uploaded Data from State ----------------------
    st.markdown("---")
    df          = st.session_state.get("uploaded_df")
    source_name = st.session_state.get("source_name") or ""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("👈 Upload a file from the **Sidebar** to see outputs here.")
        st.divider()
        return

    dfd = _sanitize_for_arrow_safe(df.copy())
    n_rows, n_cols = dfd.shape
    ext = Path(source_name).suffix.replace(".", "").upper() if source_name else "-"

    # numeric candidates stored
    st.session_state["__numeric_candidates"] = [
        c for c in dfd.columns if pd.api.types.is_numeric_dtype(dfd[c])
    ]
    prof = _profile_dataframe(dfd)

    # ========================= Row 1: Summary | Preview =========================
    # st.header("Outputs")
    st.markdown('<div class="block-card"><h2>Outputs</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.45, 0.55], gap="large")
    with c1:
        # st.subheader("📁 Uploaded File Summary")

        raw_rows   = int(st.session_state.get("raw_rows",  n_rows))
        raw_cols   = int(st.session_state.get("raw_cols",  n_cols))
        clean_rows = int(st.session_state.get("clean_rows", n_rows))
        clean_cols = int(st.session_state.get("clean_cols", n_cols))
        bd         = st.session_state.get("drop_breakdown") or {}
        dropped    = max(raw_rows - clean_rows, 0)

        def _fmt(n):
            try: return f"{int(n):,}"
            except Exception: return str(n)

        rows = [
            ("🗂️ File Name",                   source_name or "-"),
            ("🧾 Type",                        ext or "-"),
            ("🔢 Columns (original)",          _fmt(raw_cols)),
            ("📊 Rows (original)",             _fmt(raw_rows)),
            ("🔢 Columns (after cleaning)",    _fmt(clean_cols)),
            ("📊 Rows (after cleaning)",       _fmt(clean_rows)),
            ("➖ Rows removed (invalid/dupes)", f"{_fmt(dropped)}" + (f" ({dropped/raw_rows:.1%})" if raw_rows else "")),
        ]
        if isinstance(bd, dict) and bd:
            rows.append(("🔎 Row-drop breakdown", ""))
            for reason, count in bd.items():
                count = int(count)
                pct_of_drop = f"{count/dropped:.1%}" if dropped else "—"
                pct_of_raw  = f"{count/raw_rows:.1%}" if raw_rows else "—"
                rows.append((f"• {reason}", f"{_fmt(count)}  ({pct_of_drop}; {pct_of_raw} of original)"))

        summary_df = pd.DataFrame(rows, columns=["Item", "Value"])
        st.markdown('<div class="block-card"><h4>📁 Uploaded File Summary</h4>', unsafe_allow_html=True)
        _st_df(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # st.subheader("👀 Data Preview")
        st.markdown('<div class="block-card"><h4>👀 Data Preview</h4>', unsafe_allow_html=True)
        max_allowed = 25
        max_preview = min(max_allowed, int(n_rows or 0))
        if max_preview <= 0:
            st.info("No rows to preview.")
        else:
            choices_all = [1, 3, 5, 8, 10, 15, 20, 25]
            choices = [c for c in choices_all if c <= max_preview]
            current = st.session_state.get("gs_preview_rows", 10)
            try:
                current = int(current)
            except Exception:
                current = 10
            if current not in choices:
                current = min(choices, key=lambda x: (abs(x - current), x))
                st.session_state["gs_preview_rows"] = current

            # stable height
            ROW_PX, HEADER_PX, EXTRA_PX = 36, 38, 24
            table_h = st.session_state.setdefault("gs_preview_height_px", HEADER_PX + 5*ROW_PX + EXTRA_PX)
            slot = st.empty()

            # st.markdown('<div class="block-card"><h4>Preview</h4>', unsafe_allow_html=True)
            slot.dataframe(dfd.head(current), use_container_width=True, height=table_h)

            _l, _r = st.columns([0.25, 0.75])
            with _l: st.markdown("**Rows**")
            with _r:
                try:
                    new_val = st.select_slider(
                        label="Rows", options=choices, value=current,
                        key="gs_preview_rows", label_visibility="collapsed",
                        help="Choose how many rows to preview (max 25).",
                    )
                except TypeError:
                    new_val = st.select_slider("Rows", options=choices, value=current, key="gs_preview_rows")

            if new_val != current:
                slot.dataframe(dfd.head(new_val), use_container_width=True, height=table_h)
            st.markdown("</div>", unsafe_allow_html=True)

    # ========================= Row 2: Profile | Boxplot =========================
    st.divider()
    c3, c4 = st.columns([0.40, 0.60], gap="large")
    with c3:
        st.markdown('<div class="block-card"><h4>📊 Data Profile — Column Statistics</h4>', unsafe_allow_html=True)
        with st.expander("Show profile", expanded=True): 
            #  st.markdown('<div class="block-card"><h4>Column statistics</h4>', unsafe_allow_html=True) 
             _st_df(prof, hide_index=True, use_container_width=True) 
             st.markdown("</div>", unsafe_allow_html=True)


    with c4:
        # st.subheader("📈 Profile Plots (Boxplot)")
        st.markdown('<div class="block-card"><h4>📈 Profile Plots (Boxplot)</h4>', unsafe_allow_html=True)
        num_cols = list(dfd.select_dtypes(include=["number"]).columns)
        cat_cols = list(dfd.select_dtypes(exclude=["number", "datetime", "datetimetz", "timedelta"]).columns)
        if not num_cols or not cat_cols:
            st.info("Need at least one numeric and one categorical column.")
        else:
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1: num_col = st.selectbox("Numeric",  num_cols, key="gs_prof_box_num")
            with cc2: cat_col = st.selectbox("Category", cat_cols, key="gs_prof_box_cat")
            with cc3:
                palette_name = st.selectbox(
                    "Palette",
                    ["tab20", "Set3", "Pastel1", "Dark2", "Accent", "viridis", "plasma", "coolwarm"],
                    index=0, key="gs_prof_box_palette",
                )
            with cc4:
                sort_by = st.selectbox("Sort", ["frequency", "median", "mean", "label"], index=0, key="gs_prof_box_sort")

            tmp = dfd[[cat_col, num_col]].copy()
            tmp["__cat__"] = tmp[cat_col].astype("object")
            tmp = tmp.dropna(subset=["__cat__", num_col])

            if tmp.empty:
                st.info("Not enough data to draw the boxplot.")
            else:
                freq = tmp["__cat__"].value_counts()
                top_keys = freq.head(10).index
                top_df = tmp[tmp["__cat__"].isin(top_keys)]
                stats = (top_df.groupby("__cat__")[num_col]
                         .agg(count="size", mean="mean", median="median")
                         .reset_index())

                if sort_by == "frequency":
                    stats = stats.sort_values("count", ascending=False)
                elif sort_by == "median":
                    stats = stats.sort_values("median", ascending=False)
                elif sort_by == "mean":
                    stats = stats.sort_values("mean", ascending=False)
                else:
                    stats["__label_str__"] = stats["__cat__"].astype(str)
                    stats = stats.sort_values("__label_str__").drop(columns="__label_str__")

                labels = stats["__cat__"].tolist()
                grouped = {k: v[num_col].to_numpy() for k, v in top_df.groupby("__cat__")}
                data = [grouped[k] for k in labels if k in grouped and len(grouped[k]) > 0]

                if not data:
                    st.info("Not enough data to draw the boxplot.")
                else:
                    cmap = plt.get_cmap(palette_name)
                    n = len(data)
                    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
                    fig_w = max(8.2, 6.8 + 0.25 * n)

                    fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=100)
                    bp = ax.boxplot(
                        data, patch_artist=True, labels=[str(x) for x in labels],
                        showmeans=True, meanline=False, widths=0.6, whis=1.5,
                    )
                    import matplotlib.colors as mcolors
                    for box, col in zip(bp["boxes"], colors):
                        box.set_facecolor(col); box.set_alpha(0.6)
                        edge = tuple(np.clip(np.array(mcolors.to_rgb(col)) * 0.55, 0, 1))
                        box.set_edgecolor(edge); box.set_linewidth(1.4)
                    for whisk in bp["whiskers"]: whisk.set_color("#666"); whisk.set_linewidth(1.0)
                    for cap   in bp["caps"]:     cap.set_color("#666"); cap.set_linewidth(1.0)
                    for med   in bp["medians"]:  med.set_color("#1f2a44"); med.set_linewidth(1.6)
                    for mean  in bp["means"]:
                        mean.set_marker("o"); mean.set_markerfacecolor("white")
                        mean.set_markeredgecolor("#1f2a44"); mean.set_markersize(5)

                    # jitter raw points
                    rng = np.random.default_rng(7)
                    for i, vals in enumerate(data, start=1):
                        if len(vals) == 0: continue
                        x = rng.normal(i, 0.06, size=len(vals))
                        ax.scatter(x, vals, s=12, c=[colors[i - 1]], alpha=0.35,
                                   edgecolors="white", linewidths=0.3, zorder=2)

                    ax.set_title(f"{num_col} by {cat_col} (top {len(labels)})", pad=10)
                    ax.set_ylabel(num_col)
                    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(15); tick.set_ha("right")
                    fig.tight_layout()
                    # st.markdown('<div class="block-card"><h4>Boxplot</h4>', unsafe_allow_html=True)
                    st.pyplot(fig, clear_figure=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # ========================= Correlation (table + heatmap) =========================
    st.divider()
    # st.subheader("🧩 Correlation")
    st.markdown('<div class="block-card"><h4>🧩 Correlation</h4>', unsafe_allow_html=True)
    num_df = dfd.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.info("Need at least two numeric columns to compute correlations.")
        return

    cc1, cc2, cc3 = st.columns([0.28, 0.30, 0.42])
    with cc1:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0, key="corr_method")
    with cc2:
        show_heat = st.checkbox("Show heatmap", value=True, key="corr_show_heatmap")
    with cc3:
        cols = st.multiselect("Columns (optional subset)",
                              list(num_df.columns),
                              default=list(num_df.columns),
                              key="corr_columns")

    if len(cols) < 2:
        st.warning("Please select at least two columns.")
        return

    corr = num_df[cols].corr(method=method)
    lhs, rhs = st.columns([0.48, 0.52], gap="large")
    with lhs:
        st.markdown('<div class="block-card"><h4>Correlation Matrix</h4>', unsafe_allow_html=True)
        _st_df(corr.round(3), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with rhs:
        st.markdown('<div class="block-card"><h4>Correlation Heatmap</h4>', unsafe_allow_html=True)
        if show_heat:
            fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=120)
            im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
            ax.set_title(f"Heatmap ({method})")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.ax.set_ylabel("corr", rotation=90, va="center")
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================= Preparation & Validation =========================
    st.divider()
    # st.markdown("## ✅ Preparation & Validation")
    st.markdown('<div class="block-card"><h4>✅ Preparation & Validation</h4>', unsafe_allow_html=True)

    # Choose date/target columns
    date_col = st.session_state.get("date_col")
    target_col = st.session_state.get("target_col")
    if (date_col is None) or (date_col not in dfd.columns):
        dt_like = [c for c in dfd.columns if pd.api.types.is_datetime64_any_dtype(dfd[c])]
        if dt_like:
            date_col = dt_like[0]
        else:
            name_hits = [c for c in dfd.columns if str(c).lower() in ("date","datetime","timestamp","time")]
            date_col = name_hits[0] if name_hits else dfd.columns[0]
        st.session_state["date_col"] = date_col

    if (target_col is None) or (target_col not in dfd.columns):
        num_cols = [c for c in dfd.columns if pd.api.types.is_numeric_dtype(dfd[c])]
        target_col = num_cols[0] if num_cols else dfd.columns[-1]
        st.session_state["target_col"] = target_col

    # Build time series
    ts = dfd[[date_col, target_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    ts = ts.set_index(date_col)
    ts = _dedupe_index(ts, 'last')

    inferred = pd.infer_freq(ts.index)
    eff_freq = st.session_state.get("resample_freq", "W")
    eff_freq = None if eff_freq == "raw" else eff_freq
    if eff_freq is None and inferred:
        eff_freq = inferred
    if eff_freq:
        ts = ts.resample(eff_freq).sum(numeric_only=True)

    y = ts[target_col].astype(float).copy()

    # Missing value policy
    mv_policy = (st.session_state.get("missing_values", "median") or "median").lower()
    if mv_policy == "ffill": y = y.ffill()
    elif mv_policy == "bfill": y = y.bfill()
    elif mv_policy == "zero": y = y.fillna(0.0)
    elif mv_policy == "mean": y = y.fillna(y.mean())
    elif mv_policy == "median": y = y.fillna(y.median())
    elif mv_policy == "mode":
        try: y = y.fillna(y.mode().iloc[0])
        except Exception: y = y.fillna(y.median())
    elif mv_policy == "interpolate_linear": y = y.interpolate(method="linear")
    elif mv_policy == "interpolate_time":
        try: y = y.interpolate(method="time")
        except Exception: y = y.interpolate(method="linear")
    elif mv_policy == "constant":
        y = y.fillna(float(st.session_state.get("const_missing_value", 0.0)))
    elif mv_policy == "drop":
        y = y.dropna()

    # Transform
    transform = (st.session_state.get("target_transform", "none") or "none").lower()
    transformer = ("none",)
    if transform == "log1p":
        y_tr = np.log1p(np.clip(y, a_min=0, a_max=None))
        transformer = ("log1p",)
    elif transform == "boxcox":
        try:
            from scipy.stats import boxcox
            shift = max(1e-6, -(y.min()) + 1e-6)
            y_bc, lam = boxcox((y + shift).values)
            y_tr = pd.Series(y_bc, index=y.index)
            transformer = ("boxcox", lam, shift)
        except Exception:
            y_tr = np.log1p(np.clip(y, a_min=0, a_max=None))
            transformer = ("log1p",)
    else:
        y_tr = y.copy()

    # Build exogenous X with calendar + user columns + lags + scaling
    def _guess_m(freq_code: str | None) -> int:
        if not freq_code: return 7
        code = str(freq_code).upper()[0]
        return {"D":7, "W":52, "M":12, "Q":4, "H":24}.get(code, 7)
    m = _guess_m(eff_freq or inferred)

    def _detect_pattern(yser: pd.Series) -> str:
        if yser.dropna().shape[0] < 10: return "additive"
        roll = yser.rolling(window=max(5, len(yser)//20))
        corr = roll.std().corr(yser)
        return "multiplicative" if (corr is not None and corr > 0.3) else "additive"

    pattern_ui = st.session_state.get("pattern_type", "Auto-detect")
    pattern = pattern_ui if pattern_ui != "Auto-detect" else _detect_pattern(y)

    # Parse exog
    def _parse_exog(txt: str):
        txt = (txt or "").strip()
        if not txt: return []
        try:
            if txt.startswith("["): return [s.strip() for s in json.loads(txt)]
        except Exception:
            pass
        return [t.strip() for t in txt.split(",") if t.strip()]

    exog_cols = list(dict.fromkeys(
        _parse_exog(st.session_state.get("exog_cols_text", "")) +
        list(st.session_state.get("exog_additional_cols", []))
    ))

    # Base matrix
    X = pd.DataFrame(index=y_tr.index)
    if st.session_state.get("use_calendar_exog", True):
        idx = y_tr.index
        X["dow"] = idx.dayofweek
        X["month"] = idx.month
        X["is_month_start"] = idx.is_month_start.astype(int)
        X["is_month_end"] = idx.is_month_end.astype(int)
        X["is_weekend"] = idx.dayofweek.isin([5,6]).astype(int)

    for c in exog_cols:
        if c in dfd.columns:
            s = dfd.set_index(pd.to_datetime(dfd[date_col], errors="coerce"))[c]
            X[c] = pd.to_numeric(_safe_align_series_to_index(s, y_tr.index), errors="coerce")

    # Lags
    lag_list = st.session_state.get("exog_lags", [0,1,7])
    try:
        lag_list = sorted(set(int(l) for l in lag_list if int(l) >= 0))
    except Exception:
        lag_list = [0,1,7]
    lagged = {}
    for c in list(X.columns):
        for L in lag_list:
            if L == 0: continue
            lagged[f"{c}_lag{L}"] = X[c].shift(L)
    if lagged:
        X = pd.concat([X, pd.DataFrame(lagged)], axis=1)

    if st.session_state.get("scale_exog", True) and not X.empty:
        X = (X - X.mean()) / X.std(ddof=0)
    X = _dedupe_index(X, 'last').reindex(y_tr.index)

    # --------- Prep tables/plots ---------
    PLOT_W, PLOT_H = 8.8, 3.0
    L1, R1 = st.columns([0.5, 0.5], gap="large")
    with L1:
        # === L1: Preparation summary ===================================================
        st.markdown('<div class="block-card"><h4>Preparation Summary</h4>', unsafe_allow_html=True)
        
        # Duplicate timestamps: prefer 'ts' if present; fall back to 'y'
        try:
            dup_ts = int(ts.index.duplicated().sum())  # if 'ts' exists
        except Exception:
            dup_ts = int(y.index.duplicated().sum())

        prep_df = pd.DataFrame(
            [
                ("Inferred frequency",                 inferred or "—"),
                ("Effective frequency",                eff_freq or "raw"),
                ("Index monotonic ↑",                  bool(y.index.is_monotonic_increasing)),
                ("Duplicate timestamps",               dup_ts),
                ("Pattern",                            str(pattern)),
                ("Seasonal period guess (m)",          (int(m) if m is not None else "—")),
                ("Missing values (prepared series)",   int(y.isna().sum())),
            ],
            columns=["Metric", "Value"],
        )

        _renderer = globals().get("_st_df")
        if callable(_renderer):
            _renderer(prep_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(prep_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # === L1: Descriptive stats =====================================================
        st.markdown(f'<div class="block-card"><h4>Descriptive Statistics · {target_col}</h4>', unsafe_allow_html=True,)

        stats = (
            y.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
            .to_frame("value")
            .reset_index(names="stat")
        )

        if callable(_renderer):
            _renderer(stats, use_container_width=True, hide_index=True)
        else:
            st.dataframe(stats, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)


    with R1:
        # === EDA section wrapper (L1) ============================================
        # st.markdown('<div class="eda-section"><h1>Exploratory Data Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<div class="block-card"><h4>Exploratory Data Analysis</h4>', unsafe_allow_html=True,)

        # 1) Line — target over time
        st.markdown(f'<div class="block-card"><h4>{target_col} Over Time</h4>', unsafe_allow_html=True,)
        fig1, ax1 = plt.subplots(figsize=(PLOT_W, PLOT_H))
        ax1.plot(y.index, y.values, lw=1.8, marker="o", markersize=2.5, alpha=0.95)
        ax1.set_title(f"{target_col} Over Time")  # optional; we already show an H4 above
        st.pyplot(fig1, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2) Histogram — distribution
        st.markdown(
            f'<div class="block-card"><h4>Distribution of {target_col}</h4>',
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(PLOT_W, 2.6))
        ax2.hist(y.dropna().values, bins=30, edgecolor="white", linewidth=0.6)
        ax2.set_ylabel("Count")
        ax2.set_title(f"Distribution of {target_col}")  # optional; H4 is above
        st.pyplot(fig2, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3) Rolling KPIs — mean & volatility
        st.markdown('<div class="block-card"><h4>Rolling mean &amp; volatility</h4>', unsafe_allow_html=True,)

        # Safe window sizes from seasonal period guess `m` (fallbacks if m is None/0)
        try:
            m_safe = int(m) if m is not None else 12
        except Exception:
            m_safe = 12
        r_short = max(3, (m_safe // 4) or 5)
        r_long  = max(7, (m_safe // 2) or 15)

        roll_mean_s = y.rolling(r_short).mean()
        roll_mean_l = y.rolling(r_long).mean()
        roll_std    = y.rolling(r_long).std()

        fig3, ax3 = plt.subplots(figsize=(PLOT_W, 2.6))
        ax3.plot(roll_mean_s.index, roll_mean_s.values, lw=1.4, label=f"Mean ({r_short})")
        ax3.plot(roll_mean_l.index, roll_mean_l.values, lw=1.6, label=f"Mean ({r_long})")
        ax3.fill_between(
            roll_std.index,
            (roll_mean_l - roll_std).values,
            (roll_mean_l + roll_std).values,
            alpha=0.15,
            label=f"±1σ ({r_long})",
        )
        ax3.legend(loc="upper left", frameon=False)
        ax3.set_title("Rolling mean & volatility")  # optional; H4 is above
        st.pyplot(fig3, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Close EDA section wrapper
        st.markdown("</div>", unsafe_allow_html=True)


    # ========================= Seasonality & Decomposition =========================
    st.divider()
    # st.markdown("## 🌊 Seasonality & Decomposition")
    st.markdown('<div class="block-card"><h4>🌊 Seasonality & Decomposition</h4>', unsafe_allow_html=True,)
    L2, R2 = st.columns([0.5, 0.5], gap="large")

    if HAVE_STATSM and y.dropna().shape[0] >= max(2*m, 20):
        stl = STL(y.dropna(), period=max(m,1)).fit()

        with L2:
            # st.markdown("#### Component summaries")
            st.markdown('<div class="block-card"><h4>Component Summaries</h4>', unsafe_allow_html=True,)
            comp_df = pd.DataFrame(
                {"mean":[stl.observed.mean(), stl.trend.mean(), stl.seasonal.mean(), stl.resid.mean()],
                 "std": [stl.observed.std(),  stl.trend.std(),  stl.seasonal.std(),  stl.resid.std()]},
                index=["Observed","Trend","Seasonal","Resid"]
            ).reset_index(names="component")
            # st.markdown('<div class="block-card">', unsafe_allow_html=True)
            _st_df(comp_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # st.markdown("#### Seasonality diagnostics")
            st.markdown('<div class="block-card"><h4>Seasonality diagnostics</h4>', unsafe_allow_html=True,)
            var = np.nanvar
            tr, se, re = stl.trend.dropna().values, stl.seasonal.dropna().values, stl.resid.dropna().values
            n = min(len(tr), len(se), len(re)); tr, se, re = tr[-n:], se[-n:], re[-n:]
            Ft = max(0.0, 1 - var(re)/max(var(tr + re), 1e-12))
            Fs = max(0.0, 1 - var(re)/max(var(se + re), 1e-12))
            vs, vt, vr = var(se), var(tr), var(re); denom = (vs + vt + vr) or np.nan
            p = lambda x: "–" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{100*x:,.1f}%"
            diag_df = pd.DataFrame({
                "metric": ["Seasonal strength (Fs)", "Trend strength (Ft)",
                           "Variance share · Seasonal", "Variance share · Trend", "Variance share · Residual"],
                "value":  [p(Fs), p(Ft), p(vs/denom if denom else np.nan),
                           p(vt/denom if denom else np.nan), p(vr/denom if denom else np.nan)]
            })
            # st.markdown('<div class="block-card">', unsafe_allow_html=True)
            _st_df(diag_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # st.markdown("#### Autocorrelation & tests")
            # st.markdown('<div class="block-card"><h4>Autocorrelation</h4>', unsafe_allow_html=True,)
            y_clean = pd.Series(y).dropna().values
            nlags = int(min(len(y_clean)//2, max(60, 2*m)))
            ac = _acf(y_clean, nlags=nlags, fft=True)
            peaks = [(lag, val) for lag, val in enumerate(ac) if lag > 0]
            peaks.sort(key=lambda t: t[1], reverse=True)
            top_peaks = peaks[:3]
            colA, colB = st.columns([0.55, 0.45])

            with colA:
                # Build the small table of top ACF lags safely
                if top_peaks:
                    df_top = pd.DataFrame({
                        "top lag": [p[0] for p in top_peaks],
                        "ACF":     [round(p[1], 3) for p in top_peaks],
                    })
                else:
                    df_top = pd.DataFrame({"top lag": ["–"], "ACF": ["–"]})

                st.markdown('<div class="block-card"><h4>Autocorrelation</h4>', unsafe_allow_html=True,)
                _st_df(df_top, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with colB:
                try:
                    adf_p = adfuller(y_clean, autolag="AIC")[1]
                except Exception:
                    adf_p = np.nan

                lags = [m] if m else [10]
                try:
                    lb_p = float(_ljung(y_clean, lags=lags, return_df=True)["lb_pvalue"].iloc[0])
                except Exception:
                    lb_p = np.nan

                st.markdown('<div class="block-card"><h4>Tests</h4>', unsafe_allow_html=True,)
                _st_df(
                    pd.DataFrame(
                        [
                            ("ADF stationarity p", round(adf_p, 4) if not np.isnan(adf_p) else "—"),
                            (f"Ljung–Box p @ m={m or 10}", round(lb_p, 4) if not np.isnan(lb_p) else "—"),
                        ],
                        columns=["test", "p-value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)


        with R2:
            # Seasonal–Trend decomposition
            st.markdown('<div class="block-card"><h4>Seasonal–Trend Decomposition</h4>', unsafe_allow_html=True)
            # --- Plot STL components (Observed, Trend, Seasonal, Resid) -------------------
            fig, axes = plt.subplots(4, 1, figsize=(8.8, 5.8), sharex=True)
            cols = [0.20, 0.40, 0.60, 0.80]
            for ax, s, t, c in zip(
                axes,
                [stl.observed, stl.trend, stl.seasonal, stl.resid],
                ["Observed", "Trend", "Seasonal", "Resid"],
                cols,
            ):
                ax.plot(s, lw=1.8, color=plt.get_cmap("viridis")(c))
                ax.set_title(t)
                ax.grid(alpha=0.5)
            fig.tight_layout()

            st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)  # close block-card

            # --- ACF bars -----------------------------------------------------------------
            st.markdown('<div class="block-card"><h4>Autocorrelation (bars)</h4>', unsafe_allow_html=True)
            max_show = min(nlags, 48)
            figc, axc = plt.subplots(figsize=(8.8, 2.6))
            axc.bar(range(1, max_show + 1), ac[1 : max_show + 1], edgecolor="white", linewidth=0.4)
            axc.set_title(f"ACF (first {max_show} lags)")
            st.pyplot(figc, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)


    else:
        st.info("Not enough points or library unavailable for STL decomposition — need at least ~2 seasonal cycles.")

    # ========================= Rolling CV & Leaderboard =========================
    st.divider()
    # st.markdown("## 🏆 Leaderboard (Rolling Cross Validation)")
    st.markdown('<div class="block-card"><h4>🏆 Leaderboard (Rolling Cross Validation) - Leaderboard</h4>', unsafe_allow_html=True,)

    cfg = {
        "metrics": st.session_state.get("cv_metrics", ["RMSE","MAE","MASE","MAPE","sMAPE"]),
        "folds": int(st.session_state.get("cv_folds", 3)),
        "h":     int(st.session_state.get("cv_horizon", 12)),
        "gap":   int(st.session_state.get("cv_gap", 0)),
        # Model on/off toggles
        "models": {
            "ARMA": st.session_state.get("sel_ARMA", True),
            "ARIMA": st.session_state.get("sel_ARIMA", True),
            "ARIMAX": st.session_state.get("sel_ARIMAX", False),
            "SARIMA": st.session_state.get("sel_SARIMA", False),
            "SARIMAX": st.session_state.get("sel_SARIMAX", False),
            "Auto_ARIMA": st.session_state.get("sel_AutoARIMA", False),
            "HWES": st.session_state.get("sel_HWES", False),
            "Prophet": st.session_state.get("sel_Prophet", False),
            "TBATS": st.session_state.get("sel_TBATS", False),
            "XGBoost": st.session_state.get("sel_XGB", False),
            "LightGBM": st.session_state.get("sel_LGBM", False),
            "TFT": st.session_state.get("sel_TFT", False),
        },
        # Params (read but also have safety defaults inside)
        "arma": dict(p=st.session_state.get("arma_p",1), q=st.session_state.get("arma_q",1), trend=st.session_state.get("arma_trend","c")),
        "arima": dict(p=st.session_state.get("arima_p",1), d=st.session_state.get("arima_d",1), q=st.session_state.get("arima_q",1), trend=st.session_state.get("arima_trend","c")),
        "sarima": dict(p=st.session_state.get("sarima_p",1), d=st.session_state.get("sarima_d",0), q=st.session_state.get("sarima_q",1),
                       P=st.session_state.get("sarima_P",1), D=st.session_state.get("sarima_D",0), Q=st.session_state.get("sarima_Q",1),
                       m=st.session_state.get("sarima_m","auto"), trend=st.session_state.get("sarima_trend","c")),
        "arimax": dict(p=st.session_state.get("arimax_p",1), d=st.session_state.get("arimax_d",1), q=st.session_state.get("arimax_q",1), trend=st.session_state.get("arimax_trend","c")),
        "sarimax": dict(p=st.session_state.get("sarimax_p",1), d=st.session_state.get("sarimax_d",0), q=st.session_state.get("sarimax_q",1),
                        P=st.session_state.get("sarimax_P",1), D=st.session_state.get("sarimax_D",0), Q=st.session_state.get("sarimax_Q",1),
                        m=st.session_state.get("sarimax_m","auto"), trend=st.session_state.get("sarimax_trend","c")),
        "auto_arima": dict(
            seasonal=st.session_state.get("auto_seasonal", True),
            m=st.session_state.get("auto_m", "auto"),
            stepwise=st.session_state.get("auto_stepwise", True),
            suppress_warnings=st.session_state.get("auto_suppress_warnings", True),
            max_p=st.session_state.get("auto_max_p", 5),
            max_q=st.session_state.get("auto_max_q", 5),
            max_P=st.session_state.get("auto_max_P", 2),
            max_Q=st.session_state.get("auto_max_Q", 2),
            max_d=st.session_state.get("auto_max_d", 2),
            max_D=st.session_state.get("auto_max_D", 1),
        ),
        "hwes": dict(
            trend=st.session_state.get("hwes_trend", None),
            seasonal=st.session_state.get("hwes_seasonal", None),
            seasonal_periods=st.session_state.get("hwes_sp", "auto"),
            damped_trend=st.session_state.get("hwes_damped", False),
            use_boxcox=st.session_state.get("hwes_boxcox", False),
        ),
        "prophet": dict(
            growth=st.session_state.get("prophet_growth", "linear"),
            changepoint_prior_scale=st.session_state.get("prophet_cps", 0.05),
            seasonality_mode=st.session_state.get("prophet_seasonality_mode", "additive"),
            weekly_seasonality=st.session_state.get("prophet_weekly", True),
            yearly_seasonality=st.session_state.get("prophet_yearly", True),
            daily_seasonality=st.session_state.get("prophet_daily", False),
        ),
        "tbats": dict(
            seasonal_periods=st.session_state.get("tbats_sp", "[7, 365.25]"),
            use_arma_errors=st.session_state.get("tbats_use_arma", True),
            use_box_cox=st.session_state.get("tbats_boxcox", False),
        ),
        "xgboost": dict(
            n_estimators=st.session_state.get("xgb_estimators", 500),
            max_depth=st.session_state.get("xgb_depth", 5),
            learning_rate=st.session_state.get("xgb_lr", 0.05),
            subsample=st.session_state.get("xgb_subsample", 0.8),
            colsample_bytree=st.session_state.get("xgb_colsample", 0.8),
            reg_alpha=st.session_state.get("xgb_alpha", 0.1),
            reg_lambda=st.session_state.get("xgb_lambda", 1.0),
        ),
        "lightgbm": dict(
            n_estimators=st.session_state.get("lgbm_estimators", 400),
            learning_rate=st.session_state.get("lgbm_lr", 0.05),
            subsample=st.session_state.get("lgbm_subsample", 0.8),
            colsample_bytree=st.session_state.get("lgbm_colsample", 0.8),
            random_state=st.session_state.get("lgbm_random_state", 42),
        ),
        "tft": dict(
            input_chunk_length=st.session_state.get("tft_in_len", None),
            output_chunk_length=st.session_state.get("tft_out_len", None),
            hidden_size=st.session_state.get("tft_hidden", 16),
            n_epochs=st.session_state.get("tft_epochs", 50),
            batch_size=st.session_state.get("tft_batch", 32),
            random_state=st.session_state.get("tft_seed", 42),
        ),
    }

    METRIC_FUNS = {"RMSE": rmse, "MAE": mae, "MASE": lambda a,b: mase(a,b, season=(m if m>1 else 1)), "MAPE": mape, "sMAPE": smape}
    primary_metric = cfg["metrics"][0] if cfg["metrics"] else "RMSE"

    # CV split prep
    y_cv = y_tr.dropna()
    X_cv = X.loc[y_cv.index] if not X.empty else pd.DataFrame(index=y_cv.index)
    n, H, G = len(y_cv), max(1, cfg["h"]), max(0, cfg["gap"])
    folds = max(2, cfg["folds"])
    min_train = max(2*m + 10, 20)
    max_folds = max(1, (n - (H + G) - min_train) // H + 1)
    folds = min(folds, max_folds)
    if folds < 2:
        st.warning("Not enough data for the requested CV setup; reducing to a single evaluation.")
        folds = 1

    def _is_nonempty_2d(X):
        return isinstance(X, pd.DataFrame) and (not X.empty) and X.shape[0] > 0 and X.shape[1] > 0
    def _clean_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X

    # Statsmodels exog cleaning
    def _sanitize_Xy_for_statsmodels(y_in: pd.Series, X_in: pd.DataFrame | None):
        if X_in is None or getattr(X_in, "empty", True):
            yy = pd.to_numeric(y_in, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            return yy, None
        Xc = X_in.copy()
        Xc = _dedupe_index(Xc, 'last').reindex(y_in.index)
        for c in Xc.columns:
            if not pd.api.types.is_numeric_dtype(Xc[c]):
                Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
        Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
        policy = mv_policy
        if policy in ("ffill", "forward_fill"): Xc = Xc.ffill()
        elif policy in ("bfill", "backfill"):   Xc = Xc.bfill()
        elif policy == "mean":   Xc = Xc.apply(lambda s: s.fillna(s.mean()))
        elif policy == "median": Xc = Xc.apply(lambda s: s.fillna(s.median()))
        elif policy in ("const","constant"): Xc = Xc.fillna(float(st.session_state.get("const_missing_value", 0.0)))
        if not Xc.empty:
            Xc = Xc.loc[:, Xc.notna().any(axis=0)]
            nunique = Xc.nunique(dropna=True) if not Xc.empty else pd.Series(dtype=int)
            keep = [c for c in Xc.columns if int(nunique.get(c, 2)) > 1]
            Xc = Xc[keep] if keep else pd.DataFrame(index=Xc.index)
        both = pd.concat([pd.to_numeric(y_in, errors="coerce"), Xc], axis=1)
        both.replace([np.inf, -np.inf], np.nan, inplace=True)
        both = both.dropna()
        y_out = both.iloc[:, 0]
        X_out = both.iloc[:, 1:]
        if X_out.empty:
            X_out = None
        return y_out, X_out

    def _sanitize_future_exog_for_statsmodels(Xf: pd.DataFrame | None, train_cols: list[str] | None = None):
        if Xf is None or getattr(Xf, "empty", True):
            return None
        Xf = Xf.copy()
        for c in Xf.columns:
            if not pd.api.types.is_numeric_dtype(Xf[c]):
                Xf[c] = pd.to_numeric(Xf[c], errors="coerce")
        Xf.replace([np.inf, -np.inf], np.nan, inplace=True)
        policy = mv_policy
        if policy in ("ffill", "forward_fill"): Xf = Xf.ffill()
        elif policy in ("bfill", "backfill"):   Xf = Xf.bfill()
        elif policy == "mean":   Xf = Xf.apply(lambda s: s.fillna(s.mean()))
        elif policy == "median": Xf = Xf.apply(lambda s: s.fillna(s.median()))
        elif policy in ("const","constant"): Xf = Xf.fillna(float(st.session_state.get("const_missing_value", 0.0)))
        if train_cols is not None:
            Xf = Xf.reindex(columns=list(train_cols), fill_value=np.nan)
            if policy in ("ffill", "forward_fill"): Xf = Xf.ffill().bfill()
            elif policy in ("bfill", "backfill"):   Xf = Xf.bfill().ffill()
            elif policy == "mean":   Xf = Xf.apply(lambda s: s.fillna(s.mean()))
            elif policy == "median": Xf = Xf.apply(lambda s: s.fillna(s.median()))
            elif policy in ("const","constant"): Xf = Xf.fillna(float(st.session_state.get("const_missing_value", 0.0)))
            Xf = Xf.fillna(0.0)
        return Xf

    # Fit/Predict for each model
    def _params_for(name: str) -> dict:
        return cfg.get(name.lower().replace("-", "_"), {})

    def _make_tree_features(y_trn: pd.Series, X_trn: pd.DataFrame | None):
        base = pd.DataFrame({"y": y_trn.values}, index=y_trn.index)
        base["lag1"] = base["y"].shift(1)
        base["lag7"] = base["y"].shift(min(7, max(1, m)))
        base["lag_m"] = base["y"].shift(max(1, m))
        feat = base[["lag1","lag7","lag_m"]]
        if X_trn is not None and not X_trn.empty:
            feat = feat.join(X_trn, how="left")
        feat = feat.dropna()
        yy = base.loc[feat.index, "y"]
        return feat, yy

    def _recursive_forecast_tree(model, last_y: np.ndarray, X_hist: pd.DataFrame | None,
                                 X_fut: pd.DataFrame | None, steps: int) -> np.ndarray:
        out = []
        y_hist = pd.Series(last_y, index=y_tr.index[-len(last_y):])
        for t in range(steps):
            row = {
                "lag1": (out[-1] if out else y_hist.iloc[-1]),
                "lag7": (out[-min(7, len(out))] if len(out) >= 7 else y_hist.iloc[-min(7, len(y_hist))]),
                "lag_m": (out[-min(m, len(out))] if len(out) >= m else y_hist.iloc[-min(m, len(y_hist))]),
            }
            ex = (X_fut.iloc[[t]].to_dict("records")[0] if (X_fut is not None and len(X_fut) > t) else {})
            row.update(ex)
            Xrow = pd.DataFrame([row])
            out.append(float(model.predict(Xrow)[0]))
        return np.array(out, dtype=float)

    def _fit_predict(model_name: str,
                     y_train: pd.Series,
                     X_train: pd.DataFrame | None,
                     steps: int,
                     X_future: pd.DataFrame | None):
        # Statsmodels family
        if model_name == "ARMA":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            p, q = cfg["arma"]["p"], cfg["arma"]["q"]; trend = cfg["arma"]["trend"]
            mod = SARIMAX(y_train, order=(p, 0, q), trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.forecast(steps=steps)
            return np.asarray(fc), res

        if model_name == "ARIMA":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            p, d, q = cfg["arima"]["p"], cfg["arima"]["d"], cfg["arima"]["q"]; trend = cfg["arima"]["trend"]
            mod = SARIMAX(y_train, order=(p, d, q), trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.forecast(steps=steps)
            return np.asarray(fc), res

        if model_name == "ARIMAX":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            p, d, q = cfg["arimax"]["p"], cfg["arimax"]["d"], cfg["arimax"]["q"]; trend = cfg["arimax"]["trend"]
            y_trn, X_trn = _sanitize_Xy_for_statsmodels(y_train, X_train)
            train_cols = list(X_trn.columns) if X_trn is not None else None
            Xf = _sanitize_future_exog_for_statsmodels(X_future, train_cols=train_cols)
            if X_trn is None or Xf is None:
                mod = SARIMAX(y_trn, order=(p, d, q), trend=trend, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False); fc = res.forecast(steps=steps)
            else:
                mod = SARIMAX(y_trn, exog=X_trn, order=(p, d, q), trend=trend, enforce_stationality=False, enforce_invertibility=False)
                res = mod.fit(disp=False); fc = res.forecast(steps=steps, exog=Xf)
            return np.asarray(fc), res

        if model_name == "SARIMA":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            P, D, Q = cfg["sarima"]["P"], cfg["sarima"]["D"], cfg["sarima"]["Q"]
            p, d, q = cfg["sarima"]["p"], cfg["sarima"]["d"], cfg["sarima"]["q"]
            mm = m if str(cfg["sarima"]["m"]).lower() == "auto" else int(float(cfg["sarima"]["m"]))
            trend = cfg["sarima"]["trend"]
            mod = SARIMAX(y_train, order=(p, d, q),
                          seasonal_order=(P, D, Q, max(1, mm)), trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.forecast(steps=steps)
            return np.asarray(fc), res

        if model_name == "SARIMAX":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            P, D, Q = cfg["sarimax"]["P"], cfg["sarimax"]["D"], cfg["sarimax"]["Q"]
            p, d, q = cfg["sarimax"]["p"], cfg["sarimax"]["d"], cfg["sarimax"]["q"]
            mm = m if str(cfg["sarimax"]["m"]).lower() == "auto" else int(float(cfg["sarimax"]["m"]))
            y_trn, X_trn = _sanitize_Xy_for_statsmodels(y_train, X_train)
            train_cols = list(X_trn.columns) if X_trn is not None else None
            Xf = _sanitize_future_exog_for_statsmodels(X_future, train_cols=train_cols)
            if X_trn is None or Xf is None:
                mod = SARIMAX(y_trn, order=(p, d, q), seasonal_order=(P, D, Q, max(1, mm)),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False); fc = res.forecast(steps=steps)
            else:
                mod = SARIMAX(y_trn, exog=X_trn, order=(p, d, q), seasonal_order=(P, D, Q, max(1, mm)),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False); fc = res.forecast(steps=steps, exog=Xf)
            return np.asarray(fc), res

        if model_name == "Auto_ARIMA":
            if not HAVE_PM:
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["auto_arima"]
            seasonal = bool(pars.get("seasonal", m > 1))
            mm = m if str(pars.get("m", "auto")).lower() == "auto" else int(float(pars["m"]))
            y_trn, X_trn = _sanitize_Xy_for_statsmodels(y_train, X_train)
            train_cols = list(X_trn.columns) if X_trn is not None else None
            Xf = _sanitize_future_exog_for_statsmodels(X_future, train_cols=train_cols)
            if X_trn is None or Xf is None:
                ar = pmd.auto_arima(y_trn, seasonal=seasonal, m=max(1, mm),
                                    stepwise=pars.get("stepwise", True),
                                    suppress_warnings=pars.get("suppress_warnings", True),
                                    max_p=pars.get("max_p", 5), max_q=pars.get("max_q", 5),
                                    max_P=pars.get("max_P", 2), max_Q=pars.get("max_Q", 2),
                                    max_d=pars.get("max_d", 2), max_D=pars.get("max_D", 1))
                fc = ar.predict(n_periods=steps)
            else:
                ar = pmd.auto_arima(y_trn, X=X_trn, seasonal=seasonal, m=max(1, mm),
                                    stepwise=pars.get("stepwise", True),
                                    suppress_warnings=pars.get("suppress_warnings", True),
                                    max_p=pars.get("max_p", 5), max_q=pars.get("max_q", 5),
                                    max_P=pars.get("max_P", 2), max_Q=pars.get("max_Q", 2),
                                    max_d=pars.get("max_d", 2), max_D=pars.get("max_D", 1))
                fc = ar.predict(n_periods=steps, X=Xf)
            return np.asarray(fc), ar

        if model_name == "HWES":
            if not HAVE_STATSM:
                return np.repeat(y_train.iloc[-1], steps), None
            # pattern controls add/mul
            tkw = {"add": "add", "mul": "mul"}["add" if pattern == "additive" else "mul"]
            sp = st.session_state.get("hwes_sp", "auto")
            if isinstance(sp, str) and sp.strip().lower() in ("auto", "m"):
                sp = m
            try:
                sp = int(float(sp))
            except Exception:
                sp = m
            mod = ExponentialSmoothing(y_train, trend=tkw,
                                       seasonal=(tkw if sp > 1 else None),
                                       seasonal_periods=max(1, sp))
            res = mod.fit()
            fc = res.forecast(steps)
            return np.asarray(fc), res

        if model_name == "Prophet":
            if not (HAVE_PROPHET and cfg["models"].get("Prophet", False)):
                return np.repeat(y_train.iloc[-1], steps), None
            dfp = pd.DataFrame({"ds": pd.to_datetime(y_train.index), "y": y_train.values}).dropna()
            if len(dfp) < 2:
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["prophet"]
            mprop = Prophet(
                growth=pars.get("growth","linear"),
                changepoint_prior_scale=pars.get("changepoint_prior_scale",0.05),
                seasonality_mode=pars.get("seasonality_mode","additive"),
                weekly_seasonality=pars.get("weekly_seasonality",True),
                yearly_seasonality=pars.get("yearly_seasonality",True),
                daily_seasonality=pars.get("daily_seasonality",False),
            )
            mprop.fit(dfp)
            freq_str = (y_train.index.freqstr
                        if getattr(y_train.index, "freqstr", None)
                        else (pd.infer_freq(y_train.index) or (eff_freq or "W")))
            future = pd.DataFrame({"ds": pd.date_range(dfp["ds"].iloc[-1], periods=steps, freq=freq_str, inclusive="right")})
            fc = mprop.predict(future)["yhat"].values
            return np.asarray(fc), mprop

        if model_name == "TBATS":
            if not (HAVE_TBATS and cfg["models"].get("TBATS", False)):
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["tbats"]
            try:
                sp = json.loads(pars.get("seasonal_periods", "[7, 365.25]"))
            except Exception:
                sp = [max(1, m)]
            estimator = TBATS(
                seasonal_periods=sp,
                use_arma_errors=pars.get("use_arma_errors", True),
                use_box_cox=pars.get("use_box_cox", False),
            )
            res = estimator.fit(y_train.values)
            fc = res.forecast(steps=steps)
            return np.asarray(fc), res

        if model_name == "XGBoost":
            if not (HAVE_XGB and cfg["models"].get("XGBoost", False)):
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["xgboost"]
            Xtr, yy = _make_tree_features(y_train, X_train)
            if len(yy) == 0 or Xtr.empty:
                return np.repeat(y_train.iloc[-1], steps), None
            model = xgb.XGBRegressor(
                n_estimators=pars["n_estimators"], max_depth=pars["max_depth"],
                learning_rate=pars["learning_rate"], subsample=pars["subsample"],
                colsample_bytree=pars["colsample_bytree"], reg_alpha=pars["reg_alpha"],
                reg_lambda=pars["reg_lambda"], tree_method="hist"
            )
            model.fit(Xtr, yy)
            Xf = X_future.copy() if (X_future is not None and not X_future.empty) else pd.DataFrame(index=range(steps))
            # Align to training cols
            Xf = _clean_numeric_df(Xf.reindex(columns=list(Xtr.columns), fill_value=np.nan)).fillna(0.0)
            return _recursive_forecast_tree(model, y_train.values, Xtr, Xf, steps), model

        if model_name == "LightGBM":
            if not (HAVE_LGBM and cfg["models"].get("LightGBM", False)):
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["lightgbm"]
            Xtr, yy = _make_tree_features(y_train, X_train)
            if len(yy) == 0 or Xtr.empty:
                return np.repeat(y_train.iloc[-1], steps), None
            Xtr = _clean_numeric_df(Xtr).dropna()
            yy  = pd.to_numeric(yy, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            both = Xtr.join(yy.rename("y"), how="inner")
            if both.empty or both.shape[1] == 1:
                return np.repeat(y_train.iloc[-1], steps), None
            yy  = both.pop("y"); Xtr = both
            if not _is_nonempty_2d(Xtr) or len(yy) == 0:
                return np.repeat(y_train.iloc[-1], steps), None
            model = lgb.LGBMRegressor(
                n_estimators=pars["n_estimators"], learning_rate=pars["learning_rate"],
                subsample=pars["subsample"], colsample_bytree=pars["colsample_bytree"],
                random_state=pars["random_state"],
            )
            model.fit(Xtr, yy)
            Xf = X_future.copy() if (X_future is not None and not X_future.empty) else pd.DataFrame(index=range(steps))
            Xf = _clean_numeric_df(Xf.reindex(columns=list(Xtr.columns), fill_value=np.nan)).fillna(0.0)
            return _recursive_forecast_tree(model, y_train.values, Xtr, Xf, steps), model

        if model_name == "TFT":
            if not (HAVE_TFT and cfg["models"].get("TFT", False)):
                return np.repeat(y_train.iloc[-1], steps), None
            pars = cfg["tft"] or {}
            try:
                m_safe = int(m);  m_safe = m_safe if m_safe > 0 else 1
            except Exception:
                m_safe = 1
            try:
                steps_safe = int(steps); steps_safe = steps_safe if steps_safe > 0 else 1
            except Exception:
                steps_safe = 1
            in_len = pars.get("input_chunk_length")
            if not isinstance(in_len, int) or in_len <= 0:
                in_len = max(2 * m_safe, 24)
            out_len = pars.get("output_chunk_length")
            if not isinstance(out_len, int) or out_len <= 0:
                out_len = max(steps_safe, 1)

            y_vals = pd.to_numeric(y_train.values, errors="coerce")
            mask   = ~np.isnan(y_vals)
            if mask.sum() < (in_len + out_len):
                return np.repeat(y_train.iloc[-1], steps_safe), None
            series = TimeSeries.from_times_and_values(
                times=pd.to_datetime(y_train.index)[mask],
                values=y_vals[mask],
            )
            tft_kwargs = dict(
                input_chunk_length=in_len, output_chunk_length=out_len,
                hidden_size=pars.get("hidden_size",16), n_epochs=pars.get("n_epochs",50),
                batch_size=pars.get("batch_size",32), random_state=pars.get("random_state",42),
            )
            try:
                model = TFTModel(add_relative_index=True, **tft_kwargs)
            except TypeError:
                model = TFTModel(add_encoders={"relative_index": {"future": True}}, **tft_kwargs)
            model.fit(series, verbose=False)
            fc = model.predict(steps_safe).values().ravel()
            return np.asarray(fc), model

        # default naive
        return np.repeat(y_train.iloc[-1], steps), None

    # Model selection available
    model_names = [
        mname for mname, enabled in cfg["models"].items() if enabled and (
            (mname in {"Prophet"}   and HAVE_PROPHET) or
            (mname in {"TBATS"}     and HAVE_TBATS)   or
            (mname in {"XGBoost"}   and HAVE_XGB)     or
            (mname in {"LightGBM"}  and HAVE_LGBM)    or
            (mname in {"TFT"}       and HAVE_TFT)     or
            (mname not in {"Prophet","TBATS","XGBoost","LightGBM","TFT"})
        )
    ]
    if not model_names:
        st.warning("No models selected or required libraries are unavailable.")
        return

    # Rolling CV
    rows = []
    for fold in range(folds):
        train_end = len(y_cv) - (folds - fold)*H - G
        if train_end < min_train: 
            continue
        test_start = train_end + G
        test_end = min(len(y_cv), test_start + H)
        y_tr_fold = y_cv.iloc[:train_end]
        y_te_fold = y_cv.iloc[test_start:test_end]
        X_tr_fold = X_cv.iloc[:train_end] if not X_cv.empty else pd.DataFrame(index=y_tr_fold.index)
        X_te_fold = X_cv.iloc[test_start:test_end] if not X_cv.empty else pd.DataFrame(index=y_te_fold.index)

        for model_name in model_names:
            yhat, _ = _fit_predict(model_name, y_tr_fold,
                                   X_tr_fold if not X_tr_fold.empty else None,
                                   steps=len(y_te_fold),
                                   X_future=X_te_fold if not X_te_fold.empty else None)
            # inverse transform
            if transformer[0] == "log1p":
                yhat_inv = np.expm1(yhat)
                y_true_inv = y.iloc[test_start:test_end].values
            elif transformer[0] == "boxcox":
                lam, shift = transformer[1], transformer[2]
                if lam == 0: yhat_inv = np.exp(yhat) - shift
                else:        yhat_inv = np.power(yhat * lam + 1, 1/lam) - shift
                y_true_inv = y.iloc[test_start:test_end].values
            else:
                yhat_inv = yhat; y_true_inv = y.iloc[test_start:test_end].values

            metrics = {
                "RMSE": rmse(y_true_inv, yhat_inv),
                "MAE": mae(y_true_inv, yhat_inv),
                "MASE": mase(y_true_inv, yhat_inv, season=(m if m>1 else 1)),
                "MAPE": mape(y_true_inv, yhat_inv),
                "sMAPE": smape(y_true_inv, yhat_inv),
            }
            rows.append({"model": model_name, "fold": fold+1, **metrics})

    lb = pd.DataFrame(rows)
    if lb.empty:
        st.warning("Cross-validation could not run with the given settings (insufficient data).")
        return

    agg = lb.groupby("model").mean(numeric_only=True).reset_index()
    agg = agg.sort_values(primary_metric, ascending=True).reset_index(drop=True)
    # st.markdown('<div class="block-card"><h4>Leaderboard</h4>', unsafe_allow_html=True)
    _st_df(agg.assign(folds=int(folds))[["model","folds","MAE","RMSE","MAPE","MASE","sMAPE"]],
           use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    best_model_name = str(agg.iloc[0]["model"])
    st.success(f"Best-fit model by **{primary_metric}**: **{best_model_name}**")

    # ========================= Train Best & Forecast =========================
    st.divider()
    # st.markdown("## 🧪 Forecast & Diagnostics")
    st.markdown('<div class="block-card"><h4>🧪 Forecast & Diagnostics</h4>', unsafe_allow_html=True,)

    # Build future index
    if eff_freq:
        future_idx = pd.date_range(y_tr.index[-1], periods=cfg["h"]+1, freq=eff_freq, inclusive="right")
    else:
        future_idx = pd.date_range(y_tr.index[-1], periods=cfg["h"]+1, freq="W", inclusive="right")

    X_full = _dedupe_index(X, 'last').reindex(y_tr.index) if not X.empty else pd.DataFrame(index=y_tr.index)
    X_future = pd.DataFrame(index=future_idx)
    if st.session_state.get("use_calendar_exog", True):
        idx = future_idx
        X_future["dow"] = idx.dayofweek
        X_future["month"] = idx.month
        X_future["is_month_start"] = idx.is_month_start.astype(int)
        X_future["is_month_end"] = idx.is_month_end.astype(int)
        X_future["is_weekend"] = idx.dayofweek.isin([5,6]).astype(int)
    for c in exog_cols:
        if c in dfd.columns:
            last = _safe_align_series_to_index(
                dfd.set_index(pd.to_datetime(dfd[date_col], errors="coerce"))[c], y_tr.index
            ).ffill().iloc[-1]
            X_future[c] = last
    if not X_full.empty or not X_future.empty:
        allX = pd.concat([X_full, X_future])
        for col in list(allX.columns):
            for L in lag_list:
                if L == 0: continue
                allX[f"{col}_lag{L}"] = allX[col].shift(L)
        if st.session_state.get("scale_exog", True):
            mu, sd = allX.mean(), allX.std(ddof=0)
            allX = (allX - mu) / sd
        X_full = allX.loc[y_tr.index]
        X_future = allX.loc[future_idx]

    # Fit best
    yhat_tr, fitted = _fit_predict(best_model_name, y_tr, X_full if not X_full.empty else None,
                                   steps=len(future_idx),
                                   X_future=X_future if not X_future.empty else None)

    # inverse transform forecast
    if transformer[0] == "log1p":
        y_fc = np.expm1(yhat_tr)
    elif transformer[0] == "boxcox":
        lam, shift = transformer[1], transformer[2]
        y_fc = np.exp(yhat_tr) - shift if lam == 0 else np.power(yhat_tr * lam + 1, 1/lam) - shift
    else:
        y_fc = yhat_tr

    # Intervals
    if HAVE_STATSM and hasattr(fitted, "get_forecast"):
        try:
            res_fc = fitted.get_forecast(
                steps=len(future_idx),
                exog=(X_future if best_model_name in {"ARIMAX","SARIMAX"} and not X_future.empty else None)
            )
            ci = res_fc.conf_int(alpha=0.2)  # 80%
            lower, upper = ci.iloc[:,0].values, ci.iloc[:,1].values
        except Exception:
            s = np.std(y.values[-m:]) if m > 1 else np.std(y.values)
            lower = y_fc - 1.28*s; upper = y_fc + 1.28*s
    else:
        s = np.std(y.values[-m:]) if m > 1 else np.std(y.values)
        lower = y_fc - 1.28*s; upper = y_fc + 1.28*s

    # Derived frames
    _fx, _yhat = _align_xy(future_idx, y_fc)
    _,  _lo = _align_xy(_fx, lower)
    _,  _hi = _align_xy(_fx, upper)
    forecast_df = pd.DataFrame({"date": pd.to_datetime(_fx), "yhat": _yhat, "lo80": _lo, "hi80": _hi}).set_index("date")

    # History tables
    hist_tail = pd.DataFrame({"y": y}).tail(200)
    hist_stats = pd.DataFrame({
        "Metric": ["Start", "End", "Observations", "Freq (effective)", "Last value"],
        "Value": [
            str(y.index.min()) if len(y) else "—",
            str(y.index.max()) if len(y) else "—",
            f"{len(y):,}", eff_freq or "raw",
            f"{(y.iloc[-1] if len(y) else np.nan):,.4f}",
        ],
    })

    # 1) HISTORY (tables + plot)
    l, r = st.columns([0.48, 0.52], gap="large")
    with l:
        st.markdown('<div class="block-card"><h4>Series Overview</h4>', unsafe_allow_html=True)
        _st_df(hist_tail, use_container_width=True, height=260)
        st.markdown('<div class="small-note">Showing last 200 observations.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="block-card"><h4>Series Summary</h4>', unsafe_allow_html=True)
        _st_df(hist_stats, use_container_width=True, height=190, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r:
        st.markdown('<div class="block-card"><h4>History</h4>', unsafe_allow_html=True,)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y.index, y.values, linewidth=2.0, marker="o", markersize=3, alpha=0.95)
        ax.set_title("History")
        ax.grid(alpha=0.35)
        fig.tight_layout()
        # st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 1.1) Data quality & distribution
    _y_series = pd.Series(y).dropna()
    n_obs = len(_y_series)
    idx_num = np.arange(n_obs, dtype=float)

    def _pct_change(series, step):
        try:
            return float((series.iloc[-1] / series.iloc[-1-step] - 1) * 100.0) if len(series) > step else np.nan
        except Exception:
            return np.nan

    _season_map = {"D":7, "W":52, "MS":12, "M":12, "Q":4}
    _season_lag = next((v for k, v in _season_map.items() if str(eff_freq).upper().startswith(k)), 1)

    wow = _pct_change(_y_series, 1)
    mom = _pct_change(_y_series, 4 if _season_lag in (52, 1) else 1)
    yoy = _pct_change(_y_series, _season_lag) if _season_lag > 1 else np.nan

    try:
        slope = float(np.cov(idx_num, _y_series.values, ddof=0)[0, 1] / (np.var(idx_num) + 1e-12))
    except Exception:
        slope = np.nan

    try:
        adf_p = float(adfuller(_y_series.dropna(), autolag="AIC")[1]) if HAVE_STATSM else np.nan
    except Exception:
        adf_p = np.nan

    _more_stats = pd.DataFrame({
        "Metric": [
            "Missing values", "Duplicate index", "Std Dev", "CV (%)",
            "Trend slope", "Δ last vs prev (%)",
            "MoM (≈4-step) (%)" if _season_lag in (52,1) else "Δ (1-step) (%)",
            ("YoY (%)" if _season_lag > 1 else "YoY (%) (n/a)"),
            "ADF p-value (stationarity)"
        ],
        "Value": [
            int(len(y) - len(_y_series)),
            int(_y_series.index.duplicated().sum()) if hasattr(_y_series.index, "duplicated") else 0,
            np.nanstd(_y_series),
            (np.nanstd(_y_series) / np.nanmean(_y_series) * 100.0) if np.nanmean(_y_series) not in (0, np.nan) else np.nan,
            slope, wow, mom, yoy, adf_p
        ]
    })

    l2, r2 = st.columns([0.48, 0.52], gap="large")
    with l2:
        st.markdown('<div class="block-card"><h4>Distribution of values</h4>', unsafe_allow_html=True)
        _st_df(_more_stats, use_container_width=True, height=260, hide_index=True)
        st.markdown('<div class="small-note">Slope > 0 ⇒ upward trend. ADF p < 0.05 ⇒ likely stationary.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="block-card"><h4>🧪 Forecast & Diagnostics</h4>', unsafe_allow_html=True,)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        if n_obs:
            counts, bins, patches = ax.hist(_y_series.values, bins=max(10, int(np.sqrt(max(n_obs, 1)))), alpha=0.50,
                                            edgecolor="white", linewidth=0.6)
            ax.set_title("Distribution of values")
            ax.grid(alpha=0.35)
            if len(counts) > 3:
                c = pd.Series(counts).rolling(3, center=True).mean().values
                xb = 0.5 * (bins[:-1] + bins[1:])
                ax.plot(xb, c, linewidth=2.0)
        fig.tight_layout()
        # st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2) Forecast table + plot
    _forecast_disp = forecast_df.tail(cfg["h"]).copy()
    l, r = st.columns([0.48, 0.52], gap="large")
    with l:
        st.markdown('<div class="block-card"><h4>Forecast Table</h4>', unsafe_allow_html=True)
        _st_df(_forecast_disp, use_container_width=True, height=300)
        st.markdown(f'<div class="small-note">Horizon: {cfg["h"]} • Model: <b>{best_model_name}</b></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r:
        st.markdown('<div class="block-card"><h4>Forecast</h4>', unsafe_allow_html=True,)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(forecast_df.index, forecast_df["yhat"].values, linewidth=2.0, marker="o", markersize=3, alpha=0.95)
        ax.set_title("Forecast")
        ax.grid(alpha=0.35)
        fig.tight_layout()
        # st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3) Intervals table + band plot
    rng = forecast_df[["lo80","hi80"]].tail(cfg["h"]).agg(["min","max"]).T
    l, r = st.columns([0.48, 0.52], gap="large")
    with l:
        st.markdown('<div class="block-card"><h4>Intervals (80%)</h4>', unsafe_allow_html=True)
        _st_df(forecast_df[["lo80","yhat","hi80"]].tail(cfg["h"]),
               use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="block-card"><h4>Interval Range (min/max)</h4>', unsafe_allow_html=True)
        _st_df(rng, use_container_width=True, height=140)
        st.markdown("</div>", unsafe_allow_html=True)
    with r:
        st.markdown('<div class="block-card"><h4>Forecast Interval (approx)</h4>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(forecast_df.index, forecast_df["yhat"].values, linewidth=2.0, marker="o", markersize=3, alpha=0.95, label="yhat")
        ax.fill_between(forecast_df.index, forecast_df["lo80"].values, forecast_df["hi80"].values,
                        alpha=0.25, label="80% band")
        ax.legend(loc="upper left", frameon=False)
        ax.set_title("Forecast Interval (approx)")
        ax.grid(alpha=0.35)
        fig.tight_layout()
        # st.markdown('<div class="Forecast interval (approx)">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4) Residuals & correlations
    st.divider()
    # st.markdown("#### Diagnostics")
    st.markdown('<div class="block-card"><h4>Diagnostics</h4>', unsafe_allow_html=True)
    if HAVE_STATSM and fitted is not None:
        try:
            resid = pd.Series(getattr(fitted, "resid", np.array([])), index=y_tr.index[:len(getattr(fitted, "resid", []))]).dropna()
        except Exception:
            resid = pd.Series(dtype=float)

        if not resid.empty:
            # ACF table + plot
            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Autocorrelation (top lags)</h4>', unsafe_allow_html=True)
                try:
                    L = min(25, max(2, len(resid)//2))
                    acf_vals = _acf(resid, nlags=L, fft=True)
                    acf_df = pd.DataFrame({"lag": list(range(len(acf_vals))), "acf": acf_vals}).head(L+1)
                    _st_df(acf_df, use_container_width=True, height=260, hide_index=True)
                except Exception:
                    st.info("ACF values not available.")
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="block-card"><h4>Autocorrelation</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3.2))
                try:
                    plot_acf(resid, ax=ax, lags=min(25, len(resid)//2))
                except Exception:
                    ax.plot(resid.index, resid.values, lw=1.8)
                ax.set_title("Autocorrelation")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                # st.markdown('<div class="block-card">', unsafe_allow_html=True)
                st.pyplot(fig, clear_figure=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # PACF table + plot
            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Partial Autocorrelation (top lags)</h4>', unsafe_allow_html=True)
                try:
                    L = min(25, max(2, len(resid)//2))
                    pacf_vals = _pacf(resid, nlags=L, method="ywm")
                    pacf_df = pd.DataFrame({"lag": list(range(len(pacf_vals))), "pacf": pacf_vals}).head(L+1)
                    _st_df(pacf_df, use_container_width=True, height=260, hide_index=True)
                except Exception:
                    st.info("PACF values not available.")
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="block-card"><h4>Partial Autocorrelation</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3.2))
                try:
                    plot_pacf(resid, ax=ax, lags=min(25, len(resid)//2), method="ywm")
                except Exception:
                    ax.plot(resid.index, resid.values, lw=1.8)
                ax.set_title("Partial Autocorrelation")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                # st.markdown('<div class="block-card">', unsafe_allow_html=True)
                st.pyplot(fig, clear_figure=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Residuals summary + plot
            _res_stats = pd.DataFrame({
                "Metric": ["Mean", "Std", "Skew", "Kurtosis", "Ljung–Box p-value"],
                "Value": [
                    np.nanmean(resid), np.nanstd(resid), pd.Series(resid).skew(), pd.Series(resid).kurt(),
                    (lambda: float(_ljung(resid, lags=[min(10, max(1, len(resid)//4))], return_df=True)["lb_pvalue"].iloc[0])
                     if True else np.nan)()
                ],
            })

            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Residuals Summary</h4>', unsafe_allow_html=True)
                _st_df(_res_stats, use_container_width=True, height=210, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with r:
                st.markdown('<div class="block-card"><h4>Residuals</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.plot(resid.index, resid.values, linewidth=2.0)
                ax.set_title(f"Residuals ({best_model_name})")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                # st.markdown('<div class="block-card">', unsafe_allow_html=True)
                st.pyplot(fig, clear_figure=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Residual diagnostics not available for the selected model.")
    else:
        st.info("Residual diagnostics not available (statsmodels missing or no fitted model).")

    st.divider()
    
    # 1) capture snapshot locally
    out_dir = kb.flush()  # -> ./KB/{tables,figs,images,html,uploads,text,meta}
    
    # 2) read secrets / env
    az = st.secrets.get("azure", {})
    account_url       = az.get("account_url")         or os.getenv("AZURE_ACCOUNT_URL")
    connection_string = az.get("connection_string")   or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_sas_url = az.get("container_sas_url")   or os.getenv("AZURE_BLOB_CONTAINER_URL")
    container         = az.get("container")           or os.getenv("AZURE_BLOB_CONTAINER", "forecast360-kb")
    prefix            = az.get("prefix", "KB")        # optional virtual folder in the container
    
    # 3) upload to Azure Blob
    try:
        sync_folder_to_blob(
            local_folder=out_dir,
            container=container,
            prefix=prefix,
            account_url=account_url,
            connection_string=connection_string,
            container_sas_url=container_sas_url,
            delete_extraneous=False,  # True => strict mirror
            verbose=False,
        )
        st.success(f"✅ Snapshot saved to the **Knowledge Base:** Azure container {container!r} (prefix {prefix!r}).")
        # Optional: stop capturing further UI
        # kb.unpatch()
    except Exception as e:
        st.error(f"Azure upload failed: {e}")
    
    # 4) friendly footer (Sydney local time)
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        tz = ZoneInfo("Australia/Sydney")
    except Exception:
        from dateutil import tz as _tz  # fallback (pip install tzdata if needed)
        tz = _tz.gettz("Australia/Sydney")
    
    local_time = datetime.now(tz)
    formatted_time = local_time.strftime("%A, %d %B %Y %I:%M:%S %p %Z")
    st.info(f"🕒 Local Date & Time: **{formatted_time}**")
     
#     # render_chat_popover()
# # --- app render ---
# if st.session_state.get("show_sidebar"):
#     # Sidebar
#     try:
#         sidebar_getting_started()
#     except Exception as e:
#         st.sidebar.error(f"Sidebar failed: {e}")

#     # Main page
#     try:
#         page_getting_started()
#     except Exception as e:
#         st.error(f"Error rendering Getting Started: {e}")


# --- app render with BUTTONS ---
if "view" not in st.session_state:
    st.session_state["view"] = "Home"  # default landing on first load

if st.session_state.get("show_sidebar", True):
    try:
        sidebar_getting_started()
    except Exception as e:
        st.sidebar.error(f"Sidebar failed: {e}")

# Top nav buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🏠 Home", type="primary" if st.session_state["view"] == "Home" else "secondary"):
        st.session_state["view"] = "Home"
with col2:
    if st.button("🤖 AI Agent", type="primary" if st.session_state["view"] == "AI Agent" else "secondary"):
        st.session_state["view"] = "AI Agent"

st.divider()

# Main view
try:
    if st.session_state["view"] == "Home":
        page_getting_started()
    else:
        # from forecast360_chat import run as run_ai_agent
        run_ai_agent()
except Exception as e:
    st.error(f"Error rendering {st.session_state['view']}: {e}")
