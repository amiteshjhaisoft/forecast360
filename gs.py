# Author: Amitesh Jha | iSOFT

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

# Patch once so ALL renders/uploads are captured
if "kb" not in st.session_state:
    st.session_state.kb = KBCapture(folder_name="KB").patch()  # <- no keep_last
kb = st.session_state.kb

# Init the sidebar-flow flag once per session (used by CTA + sidebar gating)
st.session_state.setdefault("show_sidebar", False)

# ======================= Forecast360 — Unified Tooltips (widgets + content) =======================
import html
from typing import Any, Callable, Dict, Optional

def _render_getting_started_cta():
    """Centered CTA shown only on the Getting Started tab before the sidebar is enabled."""
    # Inject CSS once (separate key from any Home CSS)
    if not st.session_state.get("_gs_btn_css"):
        st.session_state["_gs_btn_css"] = True
        st.markdown("""
        <style>
          .hero-center-row{display:flex;justify-content:center;margin:18px 0 6px;}
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

    _, btn_c, _ = st.columns([1, 1, 1])
    with btn_c:
        st.markdown('<div class="hero-center-row">', unsafe_allow_html=True)
        if st.button("Let's Start Time Series Data Forecasting", key="gs_start_btn"):
            st.session_state["show_sidebar"] = True
        st.markdown("</div>", unsafe_allow_html=True)

# ------ Sidebar Code -----

ACCEPTED_EXTS = ["csv", "xlsx", "xls", "json", "parquet", "xml"]

def sidebar_getting_started():
    """Sidebar content for Getting Started page (ONLY place with file upload)."""
    # 🔒 Only show the sidebar after user explicitly starts Getting Started
    if not st.session_state.get("show_sidebar", False):
        return

    with st.sidebar:
        # ---- Branding --------------------------------------------------------
        if Path("assets/logo.png").exists():
            st.image("assets/logo.png", caption="iSOFT ANZ Pvt Ltd", use_container_width=True)

        st.subheader("🚀 Getting Started")

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
        try:
            if hasattr(df, "to_excel") and isinstance(df, pd.DataFrame):
                return st.dataframe(df, **k)
            else:
                return st.dataframe(df, **k)
        except TypeError:
            k.pop("hide_index", None); k.pop("use_container_width", None)
            return st.dataframe(df, **k)

    def _sanitize_for_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "_sanitize_for_arrow" in globals() and callable(globals()["_sanitize_for_arrow"]):
                return globals()["_sanitize_for_arrow"](df)
        except Exception:
            pass
        return df

    def _dedupe_index(df: pd.DataFrame, keep="last"):
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
    st.markdown('<div class="block-card"><h2>Outputs</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.45, 0.55], gap="large")
    with c1:
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

            ROW_PX, HEADER_PX, EXTRA_PX = 36, 38, 24
            table_h = st.session_state.setdefault("gs_preview_height_px", HEADER_PX + 5*ROW_PX + EXTRA_PX)
            slot = st.empty()
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

# --------- ⬇️ ADD THIS EXPORTED WRAPPER (call it from app.py) ---------
def getting_started_tab():
    """
    Use inside app.py tab:
        from gs import getting_started_tab
        with tab_gs:
            getting_started_tab()
    """
    # If user hasn't started yet, show CTA and stop further rendering for this tab
    if not st.session_state.get("show_sidebar", False):
        _render_getting_started_cta()
        return

    # Once started, render sidebar + page body
    sidebar_getting_started()
    page_getting_started()
# ----------------------------------------------------------------------
