# Author: Amitesh Jha | iSOFT

# ===============================================
# Imports and Shared Utilities (from top section)
# ===============================================
import os
from typing import Optional, Dict, List
import sys
import io
import json
import base64
import shutil
import subprocess
import warnings
import requests
import hashlib
import pathlib
import zipfile
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# (no-snapshot) capture_forecast360_results disabled, gather_and_snapshot_forecast360

# === Performance & Safety Utilities: baseline imports ===
import warnings
import logging
import matplotlib.pyplot as plt

# ---------- Safe alignment helper (handles duplicate indexes) ----------
def _dedupe_index(obj, how: str = "last"):
    """
    Collapse duplicate index labels:
      how ∈ {"first","last","sum","mean"}.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)) and obj.index.has_duplicates:
        if how in ("sum", "mean"):
            agg = {"sum": "sum", "mean": "mean"}[how]
            return obj.groupby(level=0).agg(agg)
        keep = "first" if how == "first" else "last"
        return obj[~obj.index.duplicated(keep=keep)]
    return obj

def _safe_align_series_to_index(obj, index: pd.Index, dedupe: str = "last") -> pd.Series:
    """
    Return a Series aligned to `index` without crashing on label/shape mismatches.
    - Accepts Series/DataFrame/list/ndarray/scalar.
    - Deduplicates the source index before label-based reindex.
    - If lengths match, align by *position* to avoid reindex errors.
    """
    if obj is None:
        return pd.Series(index=index, dtype="float64")

    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        s = obj.squeeze().copy()
        if not isinstance(s, pd.Series):
            s = s.iloc[:, 0]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        s = pd.Series(obj)
    else:
        return pd.Series([obj] * len(index), index=index)

    s = _dedupe_index(s, how=dedupe)

    if len(s) == len(index):
        return pd.Series(s.to_numpy(), index=index, name=getattr(s, "name", None))

    return s.reindex(index)

# Back-compat aliases
safe_align_series_to_index = _safe_align_series_to_index
safealignseriestoindex = _safe_align_series_to_index
safe_align_to_index = _safe_align_series_to_index


# ---------- Arrow/Parquet sanitiser ----------
def _sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a pandas DataFrame Arrow/Parquet-safe:
    - bytes/bytearray  -> UTF-8 strings
    - mixed-type object columns:
        * all-bool        -> nullable boolean ("boolean[pyarrow]")
        * mostly numeric  -> numeric (coerce errors to NaN)
        * otherwise       -> string ("string[pyarrow]")
    - native numeric/bool -> Arrow-friendly dtypes
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    for c in out.columns:
        s = out[c]

        # bytes -> text
        if s.dtype == "object" and s.map(lambda x: isinstance(x, (bytes, bytearray))).any():
            s = s.map(lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x)

        # If column already cleanly bools → nullable bool
        try:
            import numpy as _np
            if s.dropna().map(type).isin({bool, _np.bool_}).all():
                out[c] = s.astype("boolean[pyarrow]")
                continue
        except Exception:
            pass

        # If object and mixed → decide numeric vs text
        if s.dtype == "object":
            # Try numeric; if majority parse, keep numeric
            num_try = pd.to_numeric(s, errors="coerce")
            if num_try.notna().mean() >= 0.9:
                out[c] = num_try
            else:
                out[c] = s.astype("string[pyarrow]")
            continue

        # Native dtypes: normalise to Arrow-friendly
        try:
            if pd.api.types.is_bool_dtype(s):
                out[c] = s.astype("boolean[pyarrow]")
            elif pd.api.types.is_integer_dtype(s):
                # nullable integer so NaNs are supported
                out[c] = s.astype("Int64")
            elif pd.api.types.is_float_dtype(s):
                out[c] = s.astype("float64")
            # datetimes are fine as-is
        except Exception:
            # last resort: stringify
            out[c] = s.astype("string[pyarrow]")

    return out

def st_df(obj, *args, **kwargs):
    """Streamlit dataframe with automatic Arrow-safe sanitization."""
    try:
        import pandas as _pd
        from pandas.io.formats.style import Styler as _Styler
        if isinstance(obj, _Styler):
            return st.dataframe(obj, *args, **kwargs)
        if isinstance(obj, _pd.DataFrame):
            obj = _sanitize_for_arrow(obj)
    except Exception:
        pass
    return st.dataframe(obj, *args, **kwargs)

# (no-snapshot) capture_forecast360_results disabled

def _fallback_summary(best_model: Optional[str], leaderboard: Optional[Dict | List[Dict]]) -> Optional[str]:
    """Build a tiny summary if the app didn't produce one."""
    parts: list[str] = []
    if best_model:
        parts.append(f"Best model: {best_model}")

    if leaderboard:
        # Accept either list of rows or a flat dict of headline metrics
        if isinstance(leaderboard, list):
            lines = []
            for r in leaderboard:
                if isinstance(r, dict):
                    model = r.get("model") or r.get("name")
                    metrics_str = ", ".join(
                        f"{k.upper()}={v}" for k, v in r.items() if k.lower() in ("rmse", "mape", "mae")
                    )
                    if model or metrics_str:
                        lines.append(f"- {model}: {metrics_str}".strip(": "))
            if lines:
                parts.append("Leaderboard:\n" + "\n".join(lines))
        elif isinstance(leaderboard, dict):
            parts.append("Key metrics:\n" + "\n".join(f"- {k}: {v}" for k, v in leaderboard.items()))

    return "\n\n".join(parts) if parts else None

st.set_page_config(
    page_title='Forecast360',
    page_icon='assets/Forecast360.png',  # Path to your icon file
    layout='wide',
    initial_sidebar_state='expanded'
)

# ========= Unified, meaningful tooltips for ALL Streamlit inputs =========
# Drop this block once right after st.set_page_config(...)

# 1) Exact text → tooltip (covers labels used across Forecast360)
_TOOLTIP_MAP = {
    # Upload & parsing
    "Upload CSV / Excel / JSON / Parquet / XML":
        "Upload a single data file. Supported: CSV, Excel, JSON, Parquet, XML.",
    "XML row path (optional XPath)":
        "XPath to each repeated record node, e.g. .//row or .//record or .//item.",
    "Date/Time column (auto-detected)":
        "The timestamp column used to index and sort your time series.",
    "Target column (numeric, auto-detected)":
        "The numeric measure to analyze and forecast (e.g., sales, demand).",
    # Resampling & missing data
    "Resample frequency":
        "Resample to D/W/M/Q by SUM; choose 'raw' to keep original granularity.",
    "Missing values":
        "How to handle gaps (after resampling or in raw data).",
    "Constant value (for missing)":
        "Only used when Missing values = constant. Fills gaps with this number.",
    # Exogenous features
    "🧩 Exogenous features (for ARIMAX/SARIMAX/Auto_ARIMA/Tree/TFT)":
        "Optional driver variables (calendar features, custom columns, and lags).",
    "Use calendar features (dow, month, month-start/end, weekend)":
        "Adds standard calendar features aligned to your series index.",
    "Exog columns by name (comma or JSON list)":
        "Extra driver columns from your dataset (e.g., price, promo). Use comma or JSON list.",
    "Scale exog":
        "Standardize exogenous features (mean 0, std 1) for model stability.",
    "Exog lags":
        "Create lagged versions of exogenous features (0 keeps contemporaneous).",
    "Additional numeric exogenous columns (optional)":
        "Pick numeric columns to include as drivers in models that support exog.",
    # Pattern, transforms, outliers
    "Additive vs Multiplicative pattern":
        "Multiplicative if variability rises with the level; otherwise additive.",
    "Seasonal period (m)":
        "Seasonal cycle length: auto or explicit (e.g., 7, 12, 24, 52, 365).",
    "Target transform":
        "Variance-stabilizing transform (log1p/Box–Cox); reversed for outputs.",
    "Winsorize outliers":
        "Clip extreme values to reduce the impact of outliers.",
    "Outlier z-threshold (z)":
        "Higher z keeps more extremes (e.g., 3.5 = fairly tolerant).",
    # CV / holdout
    "Folds":
        "Number of rolling origin folds for backtesting.",
    "Horizon (H)":
        "Forecast steps per fold (evaluation window size).",
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
    "Auto_ARIMA": "Automatically selects ARIMA/SARIMA orders.",
    "HWES": "Exponential smoothing (trend/seasonal).",
    "Prophet": "Additive model with trend/seasonality/holidays.",
    "TBATS": "TBATS for multiple/long seasonalities.",
    "XGBoost": "Gradient-boosted trees with lag/covariates.",
    "LightGBM": "LightGBM regressor with lag/covariates.",
    "TFT": "Temporal Fusion Transformer (via Darts).",
    # Per-model knobs (most appear inside expanders)
    "p (ARMA)": "AR order (number of autoregressive terms).",
    "q (ARMA)": "MA order (number of moving-average terms).",
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
    "m (SARIMA)": "Seasonal period (m). 'auto' uses detected / chosen m.",
    "trend": "Deterministic trend component.",
    "m (SARIMAX)": "Seasonal period (m).",
    "seasonal_periods (JSON list)":
        "TBATS seasonal periods, e.g., [7, 365.25].",
    # XGB / LGBM knobs
    "n_estimators": "Number of boosting trees.",
    "max_depth": "Maximum tree depth (controls complexity).",
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
    "Preview rows": "Change how many rows are shown in the preview (max 25).",
    "Numeric": "Numeric column to analyze in the boxplot.",
    "Category": "Categorical column to group numeric values by.",
    "Palette": "Color palette for boxplot categories.",
    "Method": "Correlation method: Pearson, Spearman, or Kendall.",
    "Show heatmap": "Toggle the correlation heatmap.",
    "Columns (optional subset)": "Limit correlation to a subset of numeric columns.",
}

# 2) Heuristics so custom labels still get a good tooltip
def _infer_tooltip(label: str) -> str:
    if not label:
        return "Hover for guidance on how this input affects the analysis."
    lab = str(label).strip()
    if lab in _TOOLTIP_MAP:
        return _TOOLTIP_MAP[lab]
    l = lab.lower()
    if "xpath" in l:
        return "XPath to each repeated record node (e.g., .//row or .//record or .//item)."
    if "date" in l or "time" in l:
        return "Select the timestamp column used as the index."
    if "target" in l or "value" in l:
        return "Numeric measure to model and forecast."
    if "freq" in l:
        return "Sampling frequency (D/W/M/Q). Choose 'raw' to keep original."
    if "horizon" in l or "steps" in l:
        return "Number of periods to forecast ahead."
    if "season" in l and "(m)" in l or l == "m":
        return "Seasonal cycle length (e.g., 7 weekly, 12 monthly)."
    if "upload" in l or "file" in l:
        return "Upload a data file (CSV/Excel/JSON/Parquet/XML)."
    if "exog" in l or "feature" in l or "driver" in l:
        return "Optional driver variables aligned with the series; can be lagged and scaled."
    if "missing" in l:
        return "How gaps are handled (forward/backfill, interpolation, constant, etc.)."
    if "metric" in l:
        return "Error metrics used to compare models."
    return "Hover for guidance on how this input affects the analysis."

# 3) Wrapper: injects a helpful `help=` when missing on any widget
def _with_default_help(fn):
    def wrapped(label, *args, **kwargs):
        if "help" not in kwargs or not kwargs.get("help"):
            kwargs["help"] = _infer_tooltip(label)
        return fn(label, *args, **kwargs)
    return wrapped

# 4) Patch common Streamlit inputs (single pass)
for _name in (
    "text_input", "number_input", "selectbox", "multiselect", "checkbox",
    "slider", "date_input", "file_uploader", "radio", "text_area",
    "time_input", "select_slider"
):
    if hasattr(st, _name):
        setattr(st, _name, _with_default_help(getattr(st, _name)))
# ======================== end unified tooltips ============================

# ===== Global visual theme (fonts, colors, cards) =====
st.markdown("""
<style>
:root{
  --brand:#3b2dbf; --ink:#111827; --muted:#6b7280; --muted2:#4b5563;
  --bg1:#faf6ff; --bg2:#f0fbff; --border:#cfc9ff;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  color: var(--ink);
}

h1,h2,h3,h4 { color: var(--brand); font-weight: 800; letter-spacing: .2px; }
h1 { font-size: 2.0rem; } h2 { font-size: 1.5rem; } h3 { font-size: 1.2rem; } h4 { font-size: 1.05rem; }

.block-card {
  background: linear-gradient(135deg, var(--bg1), var(--bg2));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 6px 18px rgba(75,0,130,.07);
}
.block-card h4 { margin: 0 0 6px 0; color: var(--brand); }
.small-note { font-size: .92rem; color: var(--muted); }
.small-note b { color: var(--muted2); }

/* Dataframe polish */
div[data-testid="stDataFrame"] thead tr th { background: #ffffff; color:#374151; font-weight:700; }
div[data-testid="stDataFrame"] tbody tr:hover td { background: rgba(123,92,255,.06); }

/* Input polish */
[data-baseweb="select"] > div, .stTextInput>div>div>input { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ===== Matplotlib theme (coherent palette) =====
# import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e5e7eb",
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.alpha": 0.6,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#3b2dbf",
    "axes.labelcolor": "#374151",
    "xtick.color": "#4b5563",
    "ytick.color": "#4b5563",
    "axes.prop_cycle": plt.cycler(color=[
        "#7b5cff","#36cfc9","#ef476f","#ffd166","#06d6a0","#118ab2","#a78bfa"
    ])
})

st.markdown("<style>section.main > div.block-container{max-width:100% !important;padding-left:12px;padding-right:12px;}.main .block-container{max-width:100% !important;}[data-testid=stAppViewContainer] .block-container{max-width:100% !important;}[data-testid=stToolbar]{right:8px;}[data-testid=stSidebar] .block-container{padding:0.5rem 0.5rem;}</style>", unsafe_allow_html=True)

# import numpy as np
# import pandas as pd

def _safe_series(name, arr, cols):
    try:
        s = pd.Series(arr, index=list(cols), name=name, dtype="float64")
        return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:
        return pd.Series([], dtype="float64")

def _fi_from_tree_model(model, exog_cols):
    """Tree/GBM models that expose .feature_importances_"""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    s = _safe_series("importance", np.asarray(imp, dtype="float64"), exog_cols)
    if s.empty:
        return None
    return (
        s / (s.sum() or 1.0)
    ).sort_values(ascending=False).reset_index(names=["feature"]).rename(columns={"importance":"importance"})

def _fi_from_sarimax(fit_res, exog_cols):
    """
    Statsmodels SARIMAX/ARIMAX:
    Use absolute t-values of exogenous coefficients as a proxy for importance.
    """
    try:
        # Pull tvalues for params that correspond to exog columns.
        # Statsmodels names can be like 'x1', 'x2' or 'exog.x1' depending on how it was built.
        tvals = getattr(fit_res, "tvalues", None)
        if tvals is None:
            return None
        tv = pd.Series(tvals)
        # Try direct match then suffix/prefix heuristics
        out = {}
        for c in exog_cols:
            if c in tv.index:
                out[c] = abs(float(tv[c]))
            elif f"exog.{c}" in tv.index:
                out[c] = abs(float(tv[f"exog.{c}"]))
            elif f"{c}" in tv.index:
                out[c] = abs(float(tv[f"{c}"]))
            else:
                # last resort: find any key that endswith the col name
                hits = [k for k in tv.index if str(k).endswith(c)]
                out[c] = abs(float(tv[hits[0]])) if hits else 0.0

        s = pd.Series(out, dtype="float64")
        if s.sum() == 0:
            return None
        s = s / s.sum()
        return s.sort_values(ascending=False).reset_index(names=["feature"]).rename(columns={0:"importance"})
    except Exception:
        return None

def _permute_ts_importance(predict_fn, y_true, exog_df, window=200, metric="rmse", random_state=42):
    """
    Time-series-safe permutation importance:
    - Keeps index/order; only permutes values within the chosen tail window.
    - Calls predict_fn(exog_df_slice) to get yhat on that same window.
    Required:
      predict_fn: callable that accepts a DataFrame exog slice and returns yhat aligned to it
      y_true: pd.Series aligned to exog_df.index
      exog_df: DataFrame of exogenous features aligned to y_true
    """
    rng = np.random.default_rng(random_state)
    if exog_df is None or exog_df.empty:
        return None
    if not isinstance(y_true, (pd.Series, pd.DataFrame)) or y_true.empty:
        return None

    # choose evaluation slice from the tail
    n = len(exog_df)
    w = min(int(window), n)
    X = exog_df.iloc[-w:].copy()
    yt = y_true.iloc[-w:].copy()
    try:
        yhat_base = predict_fn(X)
        if isinstance(yhat_base, pd.DataFrame):
            # pick first column if DF
            yhat_base = yhat_base.iloc[:, 0]
        yhat_base = pd.Series(yhat_base, index=yt.index)  # align
    except Exception:
        return None

    def _rmse(a, b):
        diff = np.asarray(a) - np.asarray(b)
        return float(np.sqrt(np.nanmean(diff**2))) if len(diff) else np.nan

    def _mae(a, b):
        diff = np.asarray(a) - np.asarray(b)
        return float(np.nanmean(np.abs(diff))) if len(diff) else np.nan

    base_score = _rmse(yt, yhat_base) if metric.lower()=="rmse" else _mae(yt, yhat_base)
    if not np.isfinite(base_score):
        return None

    deltas = {}
    for col in X.columns:
        Xp = X.copy()
        # circular shift keeps marginal distribution & avoids leakage across train/test boundaries
        shift = int(rng.integers(1, max(2, w//3)))
        Xp[col] = Xp[col].shift(shift).fillna(method="bfill")
        try:
            yhat_p = predict_fn(Xp)
            if isinstance(yhat_p, pd.DataFrame):
                yhat_p = yhat_p.iloc[:, 0]
            yhat_p = pd.Series(yhat_p, index=yt.index)
            score = _rmse(yt, yhat_p) if metric.lower()=="rmse" else _mae(yt, yhat_p)
        except Exception:
            score = base_score

        deltas[col] = max(0.0, float(score - base_score))  # importance = increase in error

    s = pd.Series(deltas, dtype="float64")
    if s.sum() == 0:
        return None
    s = s / s.sum()
    return s.sort_values(ascending=False).reset_index(names=["feature"]).rename(columns={0:"importance"})

def build_feature_importance_df(model=None,
                                fit_res=None,
                                exog_cols=None,
                                y_series=None,
                                full_exog_df=None,
                                predict_fn=None) -> pd.DataFrame | None:
    """
    Try multiple strategies to compute feature importance and return a tidy DF.
    Strategy order:
      1) Tree/GBM .feature_importances_
      2) SARIMAX/ARIMAX t-values
      3) Permutation importance on tail window (needs predict_fn, y_series, full_exog_df)
    """
    exog_cols = list(exog_cols or [])
    if not exog_cols:
        return None  # univariate models → no FI

    # 1) Trees/GBMs
    df = _fi_from_tree_model(model, exog_cols)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    # 2) SARIMAX/ARIMAX t-values
    df = _fi_from_sarimax(fit_res, exog_cols)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    # 3) Permutation (requires predict_fn + data)
    if callable(predict_fn) and isinstance(full_exog_df, pd.DataFrame) and isinstance(y_series, pd.Series):
        df = _permute_ts_importance(predict_fn, y_series, full_exog_df)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df

    return None

# ==== Optional deps & feature flags (safe imports) ====
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    np = None
    HAVE_NUMPY = False

# statsmodels (SARIMAX, HWES, STL, ACF/PACF, Ljung-Box)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import STL
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAVE_STATSM = True
except Exception:
    SARIMAX = ExponentialSmoothing = STL = None
    def plot_acf(*args, **kwargs): pass
    def plot_pacf(*args, **kwargs): pass
    def acorr_ljungbox(*args, **kwargs): return None
    HAVE_STATSM = False

# pmdarima (Auto_ARIMA)
try:
    import pmdarima as pmd
    HAVE_PM = True
except Exception:
    pmd = None
    HAVE_PM = False

# Prophet
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    Prophet = None
    HAVE_PROPHET = False

# TBATS
try:
    from tbats import TBATS
    HAVE_TBATS = True
except Exception:
    TBATS = None
    HAVE_TBATS = False

# XGBoost
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    xgb = None
    HAVE_XGB = False

# LightGBM
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except Exception:
    lgb = None
    HAVE_LGBM = False

# TFT (via Darts)
try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TFTModel
    HAVE_TFT = True
except Exception:
    HAVE_TFT = False

def _snap_get(snap: dict, key: str):
    """Return (kind, payload) or (None, None)."""
    try:
        return snap.get(key, (None, None))
    except Exception:
        return (None, None)

def _snap_df(snap: dict, key: str) -> pd.DataFrame | None:
    """Read a df that was stored as ('df_json', json_str)."""
    try:
        kind, payload = _snap_get(snap, key)
        if kind == "df_json" and payload:
            return pd.read_json(io.StringIO(payload), orient="split")
    except Exception:
        pass
    return None

# ===== Snapshot persistence (removed) =====
def _gs_store() -> dict:
    return {}

def gs_clear_snapshot():
    return None

def gs_save_snapshot(**artifacts):
    return None

def _snap_get(snap: dict, key: str):
    return (None, None)

def _snap_df(snap: dict, key: str):
    return None

# -----------------------------
# File types (single source of truth)
# -----------------------------
SUPPORTED_FILE_TYPES = [".csv", ".xlsx", ".xls", ".parquet", ".json", ".xml"]
ACCEPTED_EXTS = [ext.lstrip(".") for ext in SUPPORTED_FILE_TYPES]  # for st.file_uploader

# Backward-compat aliases (if older code used these names)
SUPPORTED_TYPES = ACCEPTED_EXTS
ACCEPTED_FILE_TYPES = ACCEPTED_EXTS

# [Insert all shared utility functions from the original file here:
#  fill_missing(), knowledge base functions, data loaders, cached_read(),
#  model fitters, metric calculations, etc.]

# Safe fallback for infer_time_col if not provided elsewhere
try:
    infer_time_col
except NameError:
    def infer_time_col(df: pd.DataFrame):
        """Heuristic to infer a time column if your real helper isn't loaded yet."""
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            return dt_cols[0]
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ("date", "time", "timestamp", "datetime", "period")):
                try:
                    _ = pd.to_datetime(df[c], errors="raise")
                    return c
                except Exception:
                    continue
        return df.columns[0] if len(df.columns) else None

# ========================= File reader (CSV/XLSX/Parquet/JSON/XML) =========================
def read_any(file_bytes: bytes, name: str, xml_xpath: str = "") -> pd.DataFrame:
    low = (name or "").lower()
    try:
        if low.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        if low.endswith(".xlsx") or low.endswith(".xls"):
            return pd.read_excel(io.BytesIO(file_bytes))
        if low.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(file_bytes))
        if low.endswith(".json"):
            try:
                return pd.read_json(io.BytesIO(file_bytes), lines=True)
            except Exception:
                return pd.read_json(io.BytesIO(file_bytes))
        if low.endswith(".xml"):
            try:
                if xml_xpath.strip():
                    return pd.read_xml(io.BytesIO(file_bytes), xpath=xml_xpath.strip())
                return pd.read_xml(io.BytesIO(file_bytes))
            except Exception:
                try:
                    import xmltodict
                    obj = xmltodict.parse(io.BytesIO(file_bytes).read())
                    return pd.json_normalize(obj)
                except Exception:
                    return pd.DataFrame()
        # default: try CSV
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def cached_read(file_bytes: bytes, name: str, xml_xpath: str = "") -> pd.DataFrame:
    return read_any(file_bytes, name, xml_xpath=xml_xpath)

# ========================= LLM & KB Helpers (DI) =========================
import json as _json

def _build_kb_context_from_snapshot(snap: dict, top_k: int = 6, max_chars: int = 4000) -> str:
    """
    Produce a compact, human-readable context string from the snapshot.
    - For DataFrames: include head + basic shape.
    - For meta/settings: include JSON compact.
    - Truncate to max_chars.
    """
    if not isinstance(snap, dict) or not snap:
        return "No snapshot context is available."

    parts = []
    # Prefer well-known keys first
    priority_keys = [
        "meta_info", "settings_info", "best_fit_summary", "feature_importance_df",
        "leaderboard_df", "forecast_df", "y_df", "exog_df"
    ]
    seen = set()

    def _add(key: str, payload):
        try:
            if key in seen: 
                return
            seen.add(key)
            if isinstance(payload, tuple) and len(payload) == 2:
                kind, val = payload
                if kind == "df_json":
                    df = pd.read_json(io.StringIO(val), orient="split")
                    head_txt = df.head(10).to_csv(index=False)
                    parts.append(f"[{key}] DataFrame shape={df.shape}\n{head_txt}")
                else:
                    parts.append(f"[{key}] {_json.dumps(val, ensure_ascii=False) if isinstance(val, (dict,list)) else str(val)[:1000]}")
            else:
                parts.append(f"[{key}] {str(payload)[:1000]}")
        except Exception:
            pass

    for k in priority_keys:
        if k in snap:
            _add(k, snap.get(k))

    for k, v in snap.items():
        if k not in seen:
            _add(k, v)
        if sum(len(p) for p in parts) > max_chars * 1.1:
            break

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text if text.strip() else "Snapshot exists but contains no readable artifacts."

def sidebar_getting_started():
    """Sidebar content for Getting Started page (ONLY place with file upload)."""
    with st.sidebar:
        # ---- Branding --------------------------------------------------------
        if Path("assets/isoft_logo.png").exists():
            st.image(
                "assets/isoft_logo.png",
                caption="iSOFT ANZ Pvt Ltd",
                use_column_width=True,
                # use_container_width=True,
            )

        st.subheader("🚀 Getting Started")

        # ---- Data Upload -----------------------------------------------------
        st.header("📂 Data Upload")
        up = st.file_uploader(
            "Upload CSV / Excel / JSON / Parquet / XML",
            type=ACCEPTED_EXTS,              # <- extensions w/o dot
            accept_multiple_files=False,
            key="gs_file",                   # <- also stored in st.session_state["gs_file"]
        )
        xml_xpath = st.text_input(
            "XML row path (optional XPath)",
            value=st.session_state.get("xml_xpath", ""),
            help="e.g., .//row  or  .//record  or  .//item",
            key="xml_xpath",
        )
        # ---- Stop here until a file is uploaded -----------------------------
        _df_in_state = st.session_state.get("uploaded_df")
        if up is None and not (isinstance(_df_in_state, pd.DataFrame) and not _df_in_state.empty):
            st.caption("Upload a file to unlock the rest of the sidebar settings.")
            return
        st.divider()
        st.markdown("**🔎 Column detection — Automatic**")

        # ---- Read uploaded file ---------------------------------------------
        _data = None
        source_name = None

        if up is not None:
            try:
                # Prefer project helper if available
                if "cached_read" in globals() and callable(globals().get("cached_read")):
                    _data = cached_read(up.getvalue(), up.name, xml_xpath=xml_xpath)
                else:
                    # Lightweight fallback readers
                    ext = Path(up.name).suffix.lower()
                    raw = up.getvalue()

                    if ext == ".csv":
                        _data = pd.read_csv(io.StringIO(raw.decode("utf-8", "ignore")))
                    elif ext in {".xlsx", ".xls"}:
                        _data = pd.read_excel(up)
                    elif ext == ".json":
                        _data = pd.read_json(io.BytesIO(raw))
                    elif ext == ".parquet":
                        _data = pd.read_parquet(io.BytesIO(raw))
                    elif ext == ".xml":
                        try:
                            _data = pd.read_xml(io.BytesIO(raw), xpath=xml_xpath or ".//row")
                        except Exception:
                            _data = pd.read_xml(io.BytesIO(raw))
                    else:
                        st.warning(f"Unsupported extension: {ext}")
                        _data = None

                source_name = up.name
                # Store original (raw) shape before any cleaning
                st.session_state["raw_rows"] = int(_data.shape[0])
                st.session_state["raw_cols"] = int(_data.shape[1])
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                _data = None
                source_name = None

                # ---- Helpers ---------------------------------------------------------
        def _infer_target(df: pd.DataFrame) -> str:
            """Prefer numeric columns; use priority keywords when available."""
            if df is None or df.empty or not len(df.columns):
                return ""
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return df.columns[-1]  # fallback: last column if nothing numeric
            pri = ["value","amount","sales","target","y","qty","quantity","revenue","demand","passengers","count","units","volume"]
            for p in pri:
                for c in num_cols:
                    if p in c.lower():
                        return c
            return num_cols[0]

        def _infer_time_col_fallback(df: pd.DataFrame) -> str | None:
            """Heuristic datetime detection if `infer_time_col` isn't available."""
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

        # ---------- Decide which DataFrame to use for detection ----------
        if _data is None:
            # re-use previously uploaded df if user already uploaded in this session
            _data = st.session_state.get("uploaded_df")
            source_name = st.session_state.get("source_name")

        if _data is None or not isinstance(_data, pd.DataFrame) or _data.empty:
            st.info("No data available yet. Upload a file above to detect columns.")
            return

        # ---------- Auto-detect Date/Time & Target (now actually runs) ----------
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

        # ---------- Selectors with helpful tooltips ----------
        date_col = st.selectbox(
            "Date/Time column (auto-detected)",
            list(_data.columns),
            index=list(_data.columns).index(guess_date) if guess_date in _data.columns else 0,
            key="date_col",
            help="Choose the timestamp column used to order the time series. If text dates, we’ll parse them for you.",
        )

        # numeric_cols = [c for c in _data.columns if pd.api.types.is_numeric_dtype(_data[c])]
        # target_options = numeric_cols or list(_data.columns)
        # target_index = target_options.index(guess_target) if guess_target in target_options else 0

        # target_col = st.selectbox(
        #     "Target column (numeric, auto-detected)",
        #     options=target_options,
        #     index=target_index,
        #     key="target_col",
        #     help="This is the series you want to model/forecast (e.g., Sales). Numeric is recommended.",
        # )

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

        # # --- Coerce selected columns to correct dtypes for time-series ---
        # # 1) Date → datetime64 (drop invalid)
        # _data[date_col] = pd.to_datetime(_data[date_col], errors="coerce", infer_datetime_format=True)
        
        # _data = _data.dropna(subset=[date_col])

        # # 2) Target → numeric (coerce) and drop rows where it’s NaN
        # _data[target_col] = pd.to_numeric(_data[target_col], errors="coerce")
        # _data = _data.dropna(subset=[target_col])

        # # 3) De-duplicate on date, keep latest
        # _data = _data.sort_values(by=[date_col]).drop_duplicates(subset=[date_col], keep="last")

        # --- Coerce selected columns & record why rows are dropped ---
        # 0) Baseline
        n0 = int(len(_data))

        # 1) Date → datetime64
        _data[date_col] = pd.to_datetime(_data[date_col], errors="coerce", infer_datetime_format=True)
        bad_date = int(_data[date_col].isna().sum())
        _data = _data.dropna(subset=[date_col])

        # 2) Target → numeric
        _data[target_col] = pd.to_numeric(_data[target_col], errors="coerce")
        bad_target = int(_data[target_col].isna().sum())
        _data = _data.dropna(subset=[target_col])

        # 3) Duplicates — drop only exact full-row duplicates (all columns match)
        before_dedup = int(len(_data))
        exact_dupes = int(_data.duplicated(keep="last").sum())  # count first (for breakdown)
        _data = _data.drop_duplicates(keep="last")              # no subset ⇒ all columns
        dupes_dropped = exact_dupes

        # Save the breakdown for the dashboard
        st.session_state["drop_breakdown"] = {
            "Unparseable dates": bad_date,
            "Non-numeric target": bad_target,
            "Exact-duplicate rows removed": dupes_dropped,
        }

        # Store cleaned shape after coercions/drops/dedup
        st.session_state["clean_rows"] = int(_data.shape[0])
        st.session_state["clean_cols"] = int(_data.shape[1])
        # ---------- Stash for downstream use ----------
        st.session_state["uploaded_df"] = _data
        st.session_state["source_name"] = source_name
        st.session_state["__numeric_candidates"] = numeric_cols

        st.divider()

        # ---- Resampling & gaps ----------------------------------------------
        st.markdown("**⏳ Resampling & gaps**")

        # Resample frequency
        freq_choices = ["raw", "D", "W", "M", "Q"]
        _default_freq = st.session_state.get("resample_freq", "W")
        if _default_freq not in freq_choices:
            _default_freq = "W"

        freq_opt = st.selectbox(
            "Resample frequency",
            freq_choices,
            index=freq_choices.index(_default_freq),
            help="Resample by sum. Set to 'raw' to keep original granularity.",
            key="resample_freq",
        )

        # Missing value handling
        mv_options = [
            "ffill", "bfill", "zero", "mean", "median", "mode",
            "interpolate_linear", "interpolate_time",
            "constant", "drop", "none",
        ]
        _default_mv = st.session_state.get("missing_values", "median")
        if _default_mv not in mv_options:
            _default_mv = "median"

        missing_values = st.selectbox(
            "Missing values",
            mv_options,
            index=mv_options.index(_default_mv),
            key="missing_values",
            help="How to handle gaps after resampling or in the raw data.",
        )

        # Back-compat alias if other code expects 'fill_method'
        fill_method = missing_values  # local alias
        st.session_state["fill_method"] = missing_values  # optional: session alias

        # Only show constant input if 'constant' is selected
        const_val = None
        if missing_values == "constant":
            const_val = st.number_input(
                "Constant value (for missing)",
                value=float(st.session_state.get("const_missing_value", 0.0)),
                format="%.4f",
                key="const_missing_value",
            )

        st.divider()

        # ---- Exogenous features ---------------------------------------------
        st.header("🧩 Exogenous features (for ARIMAX/SARIMAX/Auto_ARIMA/Tree/TFT)")
        st.checkbox(
            "Use calendar features (dow, month, month-start/end, weekend)",
            value=st.session_state.get("use_calendar_exog", True),
            key="use_calendar_exog",
        )
        st.text_input(
            "Exog columns by name (comma or JSON list)",
            value=st.session_state.get("exog_cols_text", ""),
            key="exog_cols_text",
        )
        st.checkbox("Scale exog", value=st.session_state.get("scale_exog", True), key="scale_exog")
        st.multiselect("Exog lags", [0, 1, 7], default=st.session_state.get("exog_lags", [0, 1, 7]), key="exog_lags")

        # Additional numeric exog — shown after data is loaded
        if not st.session_state.get("__numeric_candidates"):
            st.caption("Upload data to choose additional exogenous columns.")
            st.session_state.setdefault("exog_additional_cols", [])
        else:
            st.multiselect(
                "Additional numeric exogenous columns (optional)",
                options=st.session_state["__numeric_candidates"],
                default=st.session_state.get("exog_additional_cols", []),
                key="exog_additional_cols",
                help="These are inferred from the uploaded file’s numeric columns.",
            )

        st.divider()

        # ---- Pattern & seasonality ------------------------------------------
        st.header("🌊 Pattern & seasonality")
        st.radio(
            "Additive vs Multiplicative pattern",
            ["Auto-detect", "Additive", "Multiplicative"],
            index=["Auto-detect", "Additive", "Multiplicative"].index(st.session_state.get("pattern_type", "Auto-detect")),
            key="pattern_type",
            help="Auto-detect uses rolling std/mean correlation; multiplicative if variability increases with level.",
        )
        st.caption("Auto-detect uses rolling std/mean scaling; multiplicative suggested if variability grows with the level.")

        st.divider()

        # ---- Common knobs ----------------------------------------------------
        st.header("⚙️ Common knobs")
        st.text_input(
            "Seasonal period (m)",
            value=st.session_state.get("seasonal_m", "auto"),
            key="seasonal_m",
            help="auto or explicit like 7/12/24/52/365",
        )
        st.selectbox(
            "Target transform",
            ["none", "log1p", "boxcox"],
            index=["none", "log1p", "boxcox"].index(st.session_state.get("target_transform", "none")),
            key="target_transform",
        )
        st.checkbox("Winsorize outliers", value=st.session_state.get("winsorize", True), key="winsorize")
        st.number_input("Outlier z-threshold (z)", value=st.session_state.get("outlier_z", 3.5), step=0.1, key="outlier_z")

        st.subheader("Holdout / CV")
        st.number_input("Folds", value=st.session_state.get("cv_folds", 3), step=1, min_value=2, key="cv_folds")
        st.number_input("Horizon (H)", value=st.session_state.get("cv_horizon", 12), step=1, min_value=1, key="cv_horizon")
        st.number_input("Gap", value=st.session_state.get("cv_gap", 0), step=1, min_value=0, key="cv_gap")
        st.multiselect(
            "Metrics",
            ["RMSE", "MAE", "MASE", "MAPE", "sMAPE"],
            default=st.session_state.get("cv_metrics", ["RMSE", "MAE", "MASE", "MAPE", "sMAPE"]),
            key="cv_metrics",
        )

        st.divider()

        # ---- Per-model settings ---------------------------------------------
        st.header("🧠 Per-model settings")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("ARIMA Family")
            sel_ARMA      = st.checkbox("ARMA",       value=st.session_state.get("sel_ARMA", True),    key="sel_ARMA")
            sel_ARIMA     = st.checkbox("ARIMA",      value=st.session_state.get("sel_ARIMA", True),   key="sel_ARIMA")
            sel_ARIMAX    = st.checkbox("ARIMAX",     value=st.session_state.get("sel_ARIMAX", False), key="sel_ARIMAX")
            sel_SARIMA    = st.checkbox("SARIMA",     value=st.session_state.get("sel_SARIMA", False), key="sel_SARIMA")
            sel_SARIMAX   = st.checkbox("SARIMAX",    value=st.session_state.get("sel_SARIMAX", False),key="sel_SARIMAX")
            sel_AutoARIMA = st.checkbox("Auto_ARIMA", value=st.session_state.get("sel_AutoARIMA", False), key="sel_AutoARIMA")

        with col_right:
            st.subheader("Other Models")
            sel_HWES    = st.checkbox("HWES",     value=st.session_state.get("sel_HWES", False),   key="sel_HWES")
            sel_Prophet = st.checkbox("Prophet",  value=st.session_state.get("sel_Prophet", False),key="sel_Prophet")
            sel_TBATS   = st.checkbox("TBATS",    value=st.session_state.get("sel_TBATS", False),  key="sel_TBATS")
            sel_XGB     = st.checkbox("XGBoost",  value=st.session_state.get("sel_XGB", False),    key="sel_XGB")
            sel_LGBM    = st.checkbox("LightGBM", value=st.session_state.get("sel_LGBM", False),   key="sel_LGBM")
            sel_TFT     = st.checkbox("TFT",      value=st.session_state.get("sel_TFT", False),    key="sel_TFT")

        # Expanders (shown only when the model is selected)
        if sel_ARMA:
            with st.expander("ARMA — Core", expanded=True):
                st.slider("p (ARMA)", 0, 5, st.session_state.get("arma_p", 1), key="arma_p")
                st.slider("q (ARMA)", 0, 5, st.session_state.get("arma_q", 1), key="arma_q")
                st.selectbox("trend (ARMA)", ["n", "c", "t", "ct"], index=1, key="arma_trend")

        if sel_ARIMA:
            with st.expander("ARIMA — Core", expanded=True):
                st.slider("p (ARIMA)", 0, 5, st.session_state.get("arima_p", 1), key="arima_p")
                st.slider("d (ARIMA)", 0, 2, st.session_state.get("arima_d", 1), key="arima_d")
                st.slider("q (ARIMA)", 0, 5, st.session_state.get("arima_q", 1), key="arima_q")
                st.selectbox("trend (ARIMA)", ["n", "c", "t", "ct"], index=1, key="arima_trend")

        if sel_SARIMA:
            with st.expander("SARIMA — Core", expanded=True):
                st.slider("p", 0, 3, st.session_state.get("sarima_p", 1), key="sarima_p")
                st.slider("d", 0, 1, st.session_state.get("sarima_d", 0), key="sarima_d")
                st.slider("q", 0, 3, st.session_state.get("sarima_q", 1), key="sarima_q")
                st.slider("P", 0, 3, st.session_state.get("sarima_P", 1), key="sarima_P")
                st.slider("D", 0, 1, st.session_state.get("sarima_D", 0), key="sarima_D")
                st.slider("Q", 0, 3, st.session_state.get("sarima_Q", 1), key="sarima_Q")
                st.text_input("m (SARIMA)", value=st.session_state.get("sarima_m", "auto"), key="sarima_m")
                st.selectbox("trend", ["n", "c", "t", "ct"], index=1, key="sarima_trend")

        if sel_ARIMAX:
            with st.expander("ARIMAX — Core", expanded=True):
                st.slider("p", 0, 5, st.session_state.get("arimax_p", 1), key="arimax_p")
                st.slider("d", 0, 2, st.session_state.get("arimax_d", 1), key="arimax_d")
                st.slider("q", 0, 5, st.session_state.get("arimax_q", 1), key="arimax_q")
                st.selectbox("trend", ["n", "c", "t", "ct"], index=1, key="arimax_trend")

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
                st.selectbox("trend", [None, "add", "mul"], index=0, key="hwes_trend")
                st.selectbox("seasonal", [None, "add", "mul"], index=0, key="hwes_seasonal")
                st.text_input("seasonal_periods", value=st.session_state.get("hwes_sp", "m"), key="hwes_sp")
                st.checkbox("damped_trend", value=st.session_state.get("hwes_damped", False), key="hwes_damped")

        if sel_Prophet:
            st.expander("Prophet — standard settings", expanded=False)

        if sel_TBATS:
            with st.expander("TBATS — Core", expanded=True):
                st.text_input(
                    "seasonal_periods (JSON list)",
                    value=st.session_state.get("tbats_sp", "[7, 365.25]"),
                    key="tbats_sp",
                )

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

        
def page_home():
    import base64
    from pathlib import Path
    import streamlit as st

    # --- Load hero image (base64) ---
    def _img_to_base64(paths: list[str]) -> str | None:
        for p in paths:
            fp = Path(p)
            if fp.exists():
                try:
                    return base64.b64encode(fp.read_bytes()).decode("utf-8")
                except Exception:
                    pass
        return None

    img_b64 = _img_to_base64(["assets/Forecast360.png"])

    # --- Local CSS for Home layout ---
    st.markdown("""
    <style>
      .home-wrap{
        background: radial-gradient(1200px 600px at 10% -10%, rgba(0,183,255,.10), transparent 60%),
                    radial-gradient(1200px 600px at 110% 110%, rgba(255,79,160,.08), transparent 60%),
                    linear-gradient(135deg, rgba(255,136,0,.06), rgba(0,183,255,.06) 50%, rgba(255,79,160,.06));
        border: 1px solid #eaeaea; border-radius: 22px; padding: 22px 22px 18px; margin-bottom: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,.04);
      }
      .home-cols{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 26px; align-items: center; }
      .home-left h1{ margin: 0 0 8px; font-weight: 800; letter-spacing: .2px; }
      .home-left h5{ margin: 0 0 10px; font-weight: 600; color:#0f172a; opacity:.85; }
      .home-left p{ margin: 0 0 10px; color: #334155; line-height:1.5; }

      /* RHS image block */
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

      /* KPI cards */
      .kpis{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; }
      .kcard{
        background:#fff; border:1px solid #eee; border-radius:16px; padding:14px 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,.05);
      }
      .kcard h4{ margin:0 0 6px; font-size:18px; }
      .kcard p{ margin:0; color:#475569; font-size:13px; }

      /* Responsive */
      @media (max-width: 1100px){
        .home-cols{ grid-template-columns: 1fr; gap: 18px; }
        .logo-wrap{ width:min(300px, 70%); margin: 6px auto 0; }
        .kpis{ grid-template-columns: repeat(2, 1fr); }
      }
      @media (max-width: 680px){
        .kpis{ grid-template-columns: 1fr; }
      }
    </style>
    """, unsafe_allow_html=True)

    # --- Hero ---
    st.markdown(f"""
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
            {"<img src='data:image/png;base64," + img_b64 + "' alt='Forecast360 logo'/>" if img_b64 else ""}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    
def page_getting_started():
    """Main content for Getting Started page (dashboard layout)."""
    from io import BytesIO, StringIO
    from pathlib import Path

    st.markdown("---")

    # ------------------------ Helpers ------------------------
    def _read_general_upload(up_file, xml_xpath: str = "") -> pd.DataFrame:
        ext = Path(up_file.name).suffix.lower()
        raw = up_file.getvalue()
        if ext == ".csv":
            return pd.read_csv(StringIO(raw.decode("utf-8", "ignore")))
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(up_file)
        if ext == ".json":
            return pd.read_json(BytesIO(raw))
        if ext == ".parquet":
            return pd.read_parquet(BytesIO(raw))
        if ext == ".xml":
            try:
                return pd.read_xml(BytesIO(raw), xpath=xml_xpath or ".//row")
            except Exception:
                return pd.read_xml(BytesIO(raw))
        raise ValueError(f"Unsupported extension: {ext}")

   
    def _arrow_compatible_df(df: pd.DataFrame) -> pd.DataFrame:
        try:
            return _sanitize_for_arrow(df)
        except Exception:
            # even on failure return the original so the app doesn't crash
            return df

    def _profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        rows = []
        n = int(len(df))

        def _native(v):
            if v is None:
                return None
            try:
                import numpy as _np
                if isinstance(v, (_np.generic,)):
                    v = v.item()
            except Exception:
                pass
            try:
                if isinstance(v, float) and np.isnan(v):
                    return None
            except Exception:
                pass
            return v

        for c in df.columns:
            s = df[c]
            is_num = pd.api.types.is_numeric_dtype(s)

            missing = int(s.isna().sum())
            unique = int(s.nunique(dropna=True))
            pct_missing = round((missing / n) * 100, 2) if n else 0.0

            if is_num and s.notna().any():
                mean   = _native(s.mean())
                median = _native(s.median())
                stddev = _native(s.std(ddof=1)) if s.notna().sum() > 1 else None
                min_v  = _native(s.min())
                max_v  = _native(s.max())
                try:
                    m = s.dropna().mode()
                    mode_val = _native(m.iloc[0]) if len(m) else None
                except Exception:
                    mode_val = None
            else:
                mean = median = stddev = min_v = max_v = None
                mode_val = None  # Mode only for numeric

            rows.append(
                {
                    "Column": c,
                    "Dtype": str(s.dtype),
                    "Count": n,
                    "Missing": missing,
                    "Pct_missing": pct_missing,
                    "Unique": unique,
                    "Mean": mean,
                    "Median": median,
                    "Mode": mode_val,
                    "Std_dev": stddev,
                    "Min": min_v,
                    "Max": max_v,
                }
            )
        return pd.DataFrame(rows)

    # ----------------- Retrieve uploaded data -----------------
    up = st.session_state.get("gs_file")
    xml_xpath = st.session_state.get("xml_xpath", "")
    df = st.session_state.get("uploaded_df")
    source_name = st.session_state.get("source_name")

    if df is None and up is not None:
        try:
            if "cached_read" in globals() and callable(globals().get("cached_read")):
                df = cached_read(up.getvalue(), up.name, xml_xpath=xml_xpath)
            else:
                df = _read_general_upload(up, xml_xpath=xml_xpath)
            source_name = up.name
            st.session_state["uploaded_df"] = df
            st.session_state["source_name"] = source_name
        except Exception as e:
            st.error(f"❌ Could not read the uploaded file: {e}")
            return

    if df is None:
        st.info("👈 Upload a file from the **Sidebar** to see outputs here.")
        st.divider()
        return

    # --------------------- Basics & Profile --------------------
    dfd = _arrow_compatible_df(df)
    n_rows, n_cols = dfd.shape
    ext = (Path(source_name).suffix.replace(".", "").upper() if source_name else "-") or "-"
    st.session_state["__numeric_candidates"] = [c for c in dfd.columns if pd.api.types.is_numeric_dtype(dfd[c])]
    prof = _profile_dataframe(dfd)

    # =================== DASHBOARD: Outputs ====================
    st.header("Outputs")

    # ---------- Row 1: Summary (LHS) | Data Preview (RHS) ----------
    row1_lhs, row1_rhs = st.columns([0.45, 0.55], gap="large")

    with row1_lhs:
        
        st.subheader("📁 Uploaded File Summary")

        raw_rows   = int(st.session_state.get("raw_rows",  n_rows))
        raw_cols   = int(st.session_state.get("raw_cols",  n_cols))
        clean_rows = int(st.session_state.get("clean_rows", n_rows))
        clean_cols = int(st.session_state.get("clean_cols", n_cols))
        bd         = st.session_state.get("drop_breakdown") or {}

        dropped = max(raw_rows - clean_rows, 0)

        def _fmt(n): 
            try: 
                return f"{int(n):,}"
            except Exception:
                return str(n)

        rows = [
            ("🗂️ File Name",                      source_name or "-"),
            ("🧾 Type",                           ext),
            ("🔢 Columns (original)",             _fmt(raw_cols)),
            ("📊 Rows (original)",                _fmt(raw_rows)),
            ("🔢 Columns (after cleaning)",       _fmt(clean_cols)),
            ("📊 Rows (after cleaning)",          _fmt(clean_rows)),
            ("➖ Rows removed (invalid/dupes)",    f"{_fmt(dropped)}" + (f"  ({dropped/raw_rows:.1%} of original)" if raw_rows else "")),
        ]

        # Inline breakdown as additional rows
        if isinstance(bd, dict) and bd:
            rows.append(("🔎 Row-drop breakdown", ""))  # section header row
            for reason, count in bd.items():
                count = int(count)
                pct_of_drop = f"{count/dropped:.1%}" if dropped else "—"
                pct_of_raw  = f"{count/raw_rows:.1%}" if raw_rows else "—"
                rows.append((
                    f"• {reason}",
                    f"{_fmt(count)}  ({pct_of_drop} of dropped; {pct_of_raw} of original)"
                ))

        summary_merged = pd.DataFrame(rows, columns=["Item", "Value"])
        st_df(summary_merged, hide_index=True, use_container_width=True)

 

    with row1_rhs:
        
        st.subheader("👀 Data Preview")

        # Fixed height equal to 3-row preview (keeps layout stable)
        def _fixed_height_for_rows(rows: int = 3) -> int:
            ROW_PX = int(st.session_state.get("ui_row_px", 36))
            HEADER_PX = int(st.session_state.get("ui_header_px", 38))
            EXTRA_PX = int(st.session_state.get("ui_table_pad_px", 24))
            return HEADER_PX + rows * ROW_PX + EXTRA_PX

        max_allowed = 25
        max_preview = min(max_allowed, int(n_rows or 0))
        if max_preview <= 0:
            st.info("No rows to preview.")
        else:
            choices_all = [1, 3, 5, 8, 10, 15, 20, 25]
            choices = [c for c in choices_all if c <= max_preview]

            current = st.session_state.get("gs_preview_rows", 5)
            try:
                current = int(current)
            except Exception:
                current = 3
            if current not in choices:
                current = min(choices, key=lambda x: (abs(x - current), x))
                st.session_state["gs_preview_rows"] = current

            TABLE_HEIGHT_PX = st.session_state.setdefault("gs_preview_height_px", _fixed_height_for_rows(5))

            table_slot = st.empty()
            table_slot.dataframe(dfd.head(current), use_container_width=True, height=TABLE_HEIGHT_PX)

            c1, c2 = st.columns([0.18, 0.82])
            with c1:
                st.markdown("**Preview rows**")
            with c2:
                try:
                    new_val = st.select_slider(
                        label="Preview rows",
                        options=choices,
                        value=current,
                        key="gs_preview_rows",
                        label_visibility="collapsed",
                        help="Choose how many rows to preview (max 25).",
                    )
                except TypeError:
                    new_val = st.select_slider(
                        label="",
                        options=choices,
                        value=current,
                        key="gs_preview_rows",
                        help="Choose how many rows to preview (max 25).",
                    )

            if new_val != current:
                table_slot.dataframe(dfd.head(new_val), use_container_width=True, height=TABLE_HEIGHT_PX)

    # ---------- Row 2: Data Profile Plots (RHS) ----------
    st.divider()
    row2_lhs, row2_rhs = st.columns([0.40, 0.30], gap="large")
    with row2_lhs:
        st.subheader("📊 Data Profile")
        with st.expander("Show profile", expanded=True):
            st.dataframe(prof, hide_index=True, use_container_width=True)

    with row2_rhs:
        st.subheader("📈 Data Profile Plots")
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
       

        FIG_W, FIG_H, DPI = 8.2, 4.6, 100  # stable sizing

        num_cols = list(dfd.select_dtypes(include=["number"]).columns)

        # exclude numeric, datetime (tz and non-tz), and timedeltas just in case
        cat_cols = list(
            dfd.select_dtypes(exclude=["number", "datetime", "datetimetz", "timedelta"]).columns
        )

        plot_slot = st.empty()

        if not num_cols or not cat_cols:
            st.info("Need at least one numeric and one categorical column.")
        else:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            with c1:
                num_col = st.selectbox("Numeric", num_cols, key="gs_prof_box_num")
            with c2:
                cat_col = st.selectbox("Category", cat_cols, key="gs_prof_box_cat")
            with c3:
                palette_name = st.selectbox(
                    "Palette",
                    ["tab20", "Set3", "Pastel1", "Dark2", "Accent", "viridis", "plasma", "coolwarm"],
                    index=0,
                    key="gs_prof_box_palette",
                )
            with c4:
                sort_by = st.selectbox("Sort", ["frequency", "median", "mean", "label"], index=0, key="gs_prof_box_sort")

            # --- Prepare once (avoid repeated filtering/conversion) ---
            tmp = dfd[[cat_col, num_col]].copy()
            tmp["__cat__"] = tmp[cat_col].astype("object")
            tmp = tmp.dropna(subset=["__cat__", num_col])

            if tmp.empty:
                st.info("Not enough data to draw the boxplot.")
            else:
                # top-10 categories by frequency
                freq = tmp["__cat__"].value_counts()
                top_keys = freq.head(10).index

                top_df = tmp[tmp["__cat__"].isin(top_keys)]

                # compute stats for sorting
                stats = (
                    top_df.groupby("__cat__")[num_col]
                    .agg(count="size", mean="mean", median="median")
                    .reset_index()
                )

                if sort_by == "frequency":
                    stats = stats.sort_values("count", ascending=False)
                elif sort_by == "median":
                    stats = stats.sort_values("median", ascending=False)
                elif sort_by == "mean":
                    stats = stats.sort_values("mean", ascending=False)
                else:  # label
                    # safe string sort for mixed types
                    stats["__label_str__"] = stats["__cat__"].astype(str)
                    stats = stats.sort_values("__label_str__").drop(columns="__label_str__")

                labels = stats["__cat__"].tolist()

                # build data arrays in sorted order
                grouped = {k: v[num_col].to_numpy() for k, v in top_df.groupby("__cat__")}
                data = [grouped[k] for k in labels if k in grouped and len(grouped[k]) > 0]

                if not data:
                    st.info("Not enough data to draw the boxplot.")
                else:
                    cmap = plt.get_cmap(palette_name)
                    n = len(data)
                    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

                    # widen a bit if many categories
                    fig_w = max(FIG_W, 6.8 + 0.25 * n)

                    fig, ax = plt.subplots(figsize=(fig_w, FIG_H), dpi=DPI)

                    bp = ax.boxplot(
                        data,
                        patch_artist=True,
                        labels=[str(x) for x in labels],
                        showmeans=True,
                        meanline=False,
                        widths=0.6,
                        whis=1.5,
                    )

                    # Style boxes
                    for i, (box, col) in enumerate(zip(bp["boxes"], colors)):
                        box.set_facecolor(col)
                        box.set_alpha(0.6)
                        edge = tuple(np.clip(np.array(mcolors.to_rgb(col)) * 0.55, 0, 1))
                        box.set_edgecolor(edge)
                        box.set_linewidth(1.4)

                    for whisk in bp["whiskers"]:
                        whisk.set_color("#666"); whisk.set_linewidth(1.0)
                    for cap in bp["caps"]:
                        cap.set_color("#666"); cap.set_linewidth(1.0)
                    for med in bp["medians"]:
                        med.set_color("#1f1f1f"); med.set_linewidth(1.6)
                    for mean in bp["means"]:
                        mean.set_marker("o")
                        mean.set_markerfacecolor("white")
                        mean.set_markeredgecolor("#1f1f1f")
                        mean.set_markersize(5)

                    # jittered raw points
                    rng = np.random.default_rng(7)
                    for i, vals in enumerate(data, start=1):
                        if len(vals) == 0:
                            continue
                        x = rng.normal(i, 0.06, size=len(vals))
                        ax.scatter(
                            x, vals, s=12, c=[colors[i - 1]],
                            alpha=0.35, edgecolors="white", linewidths=0.3, zorder=2,
                        )

                    ax.set_title(f"{num_col} by {cat_col} (top {len(labels)})", pad=10)
                    ax.set_ylabel(num_col)
                    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(15); tick.set_ha("right")

                    fig.tight_layout()
                    plot_slot.pyplot(fig, clear_figure=True)

        # ========================== CONFIG PULL ==========================
        cfg = {
            "freq": st.session_state.get("resample_freq", "W"),
            "mv": st.session_state.get("missing_values", "median"),
            "const_val": float(st.session_state.get("const_missing_value", 0.0)),
            "use_calendar": st.session_state.get("use_calendar_exog", True),
            "exog_text": st.session_state.get("exog_cols_text", ""),
            "scale_exog": st.session_state.get("scale_exog", True),
            "exog_lags": st.session_state.get("exog_lags", [0,1,7]),
            "exog_additional": st.session_state.get("exog_additional_cols", []),
            "pattern": st.session_state.get("pattern_type", "Auto-detect"),
            "seasonal_m": st.session_state.get("seasonal_m", "auto"),
            "transform": st.session_state.get("target_transform", "none"),
            "winsorize": st.session_state.get("winsorize", True),
            "outlier_z": float(st.session_state.get("outlier_z", 3.5)),
            "folds": int(st.session_state.get("cv_folds", 3)),
            "h": int(st.session_state.get("cv_horizon", 12)),
            "gap": int(st.session_state.get("cv_gap", 0)),
            "metrics": st.session_state.get("cv_metrics", ["RMSE","MAE","MASE","MAPE","sMAPE"]),

            # model selections (labels match Sidebar; internal normalization happens later)
            "models": {
                "ARMA": st.session_state.get("sel_ARMA", True),
                "ARIMA": st.session_state.get("sel_ARIMA", True),
                "ARIMAX": st.session_state.get("sel_ARIMAX", False),
                "SARIMA": st.session_state.get("sel_SARIMA", False),
                "SARIMAX": st.session_state.get("sel_SARIMAX", False),
                "Auto_ARIMA": st.session_state.get("sel_AutoARIMA", False),  # underscore everywhere
                "HWES": st.session_state.get("sel_HWES", False),
                # Model Zoo:
                "Prophet": st.session_state.get("sel_Prophet", False),
                "TBATS": st.session_state.get("sel_TBATS", False),
                "XGBoost": st.session_state.get("sel_XGB", False),
                "LightGBM": st.session_state.get("sel_LGBM", False),
                "TFT": st.session_state.get("sel_TFT", False),
            },

            # per-model params
            "arma": dict(p=st.session_state.get("arma_p",1), q=st.session_state.get("arma_q",1),
                        trend=st.session_state.get("arma_trend","c")),
            "arima": dict(p=st.session_state.get("arima_p",1), d=st.session_state.get("arima_d",1),
                        q=st.session_state.get("arima_q",1), trend=st.session_state.get("arima_trend","c")),
            "sarima": dict(p=st.session_state.get("sarima_p",1), d=st.session_state.get("sarima_d",0),
                        q=st.session_state.get("sarima_q",1), P=st.session_state.get("sarima_P",1),
                        D=st.session_state.get("sarima_D",0), Q=st.session_state.get("sarima_Q",1),
                        m=st.session_state.get("sarima_m","auto"), trend=st.session_state.get("sarima_trend","c")),
            "arimax": dict(p=st.session_state.get("arimax_p",1), d=st.session_state.get("arimax_d",1),
                        q=st.session_state.get("arimax_q",1), trend=st.session_state.get("arimax_trend","c")),
            "sarimax": dict(p=st.session_state.get("sarimax_p",1), d=st.session_state.get("sarimax_d",0),
                            q=st.session_state.get("sarimax_q",1), P=st.session_state.get("sarimax_P",1),
                            D=st.session_state.get("sarimax_D",0), Q=st.session_state.get("sarimax_Q",1),
                            m=st.session_state.get("sarimax_m","auto"),
                            trend=st.session_state.get("sarimax_trend","c")),  # only used if you exposed it

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
                trend=st.session_state.get("hwes_trend", None),             # "add", "mul", or None
                seasonal=st.session_state.get("hwes_seasonal", None),       # "add", "mul", or None
                seasonal_periods=st.session_state.get("hwes_sp", "auto"),   # <-- read the Sidebar's key
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

            # NOTE: cfg key must be lowercase 'lightgbm' so _params_for('LightGBM') finds it
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

    # ---------- Row 3: Correlation (LHS table | RHS heatmap) ----------
    st.divider()
    st.subheader("🧩 Correlation")

    num_df = dfd.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.info("Need at least two numeric columns to compute correlations.")
    else:
        cc1, cc2, cc3 = st.columns([0.28, 0.30, 0.42])
        with cc1:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0, key="corr_method")
        with cc2:
            show_heat = st.checkbox("Show heatmap", value=True, key="corr_show_heatmap")
        with cc3:
            cols = st.multiselect(
                "Columns (optional subset)",
                options=list(num_df.columns),
                default=list(num_df.columns),
                key="corr_columns",
            )

        if len(cols) >= 2:
            corr = num_df[cols].corr(method=method)

            lhs_corr, rhs_corr = st.columns([0.48, 0.52], gap="large")
            with lhs_corr:
                st.markdown("**Correlation matrix**")
                st.dataframe(corr.round(3), use_container_width=True)

            with rhs_corr:
                st.markdown("**Correlation heatmap**")
                if show_heat:
                    import matplotlib.pyplot as plt
                    FIG_W, FIG_H, DPI = 7.6, 6.0, 120
                    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
                    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="equal")
                    ax.set_xticks(range(len(cols)))
                    ax.set_xticklabels(cols, rotation=45, ha="right")
                    ax.set_yticks(range(len(cols)))
                    ax.set_yticklabels(cols)
                    ax.set_title(f"Correlation heatmap ({method})")
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.set_ylabel("corr", rotation=90, va="center")
                    for i in range(len(cols)):
                        for j in range(len(cols)):
                            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Please select at least two columns.")
        st.divider()
        # ------------------------- Center Run Button -------------------------
       
        # c1, c2, c3 = st.columns([1,1,1])
        # with c2:
        #     run = st.button("▶️ Run Analysis", use_container_width=True)
        #     st.caption("Uses **all** sidebar settings and selected models.")

        # # if not run:
        # #     return
       

        # ----- FEATURE IMPORTANCE (if exogenous features exist) -----
    fi_df = None
    try:
        # quick adapter so permutation path can call back into your model
        def _predict_fn(Xp: pd.DataFrame):
            # Return a pd.Series or 1-D array aligned to Xp.index
            if hasattr(best_model, "predict"):
                yhat = best_model.predict(Xp)
            elif hasattr(fit_result, "get_prediction"):  # statsmodels
                yhat = fit_result.get_prediction(exog=Xp).predicted_mean
            else:
                raise RuntimeError("No predict path available.")
            return pd.Series(np.asarray(yhat).ravel(), index=Xp.index)

        fi_df = build_feature_importance_df(
            model=best_model,
            fit_res=fit_result,
            exog_cols=list(exog_df.columns) if isinstance(exog_df, pd.DataFrame) else [],
            y_series=y if isinstance(y, pd.Series) else None,
            full_exog_df=exog_df if isinstance(exog_df, pd.DataFrame) else None,
            predict_fn=_predict_fn
        )
    except Exception:
        fi_df = None

    if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
        # normalize & keep top 20 for readability
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df["importance"] = fi_df["importance"] / (fi_df["importance"].sum() or 1.0)
        fi_df = fi_df.head(20).reset_index(drop=True)

        # save to snapshot so Visualization → Model Details shows it
        gs_save_snapshot(feature_importance_df=fi_df)
    else:
        # optional: explicitly clear previous FI if the new run had none
        gs_save_snapshot(feature_importance_df=pd.DataFrame(columns=["feature","importance"]))

        # ========================== PREP STEP ==========================
        # --- ensure date_col / target_col exist before we touch them ---
        date_col = st.session_state.get("date_col")
        target_col = st.session_state.get("target_col")

        # If date_col missing/invalid, pick a sensible default
        if (date_col is None) or (date_col not in dfd.columns):
            dt_like = [c for c in dfd.columns if pd.api.types.is_datetime64_any_dtype(dfd[c])]
            if dt_like:
                date_col = dt_like[0]
            else:
                name_hits = [c for c in dfd.columns if str(c).lower() in ("date","datetime","timestamp","time")]
                date_col = name_hits[0] if name_hits else dfd.columns[0]
            st.session_state["date_col"] = date_col

        # If target_col missing/invalid, prefer a numeric column
        if (target_col is None) or (target_col not in dfd.columns):
            num_cols = [c for c in dfd.columns if pd.api.types.is_numeric_dtype(dfd[c])]
            target_col = num_cols[0] if num_cols else dfd.columns[-1]
            st.session_state["target_col"] = target_col

        # make datetime index + sort + dedupe
        ts = dfd[[date_col, target_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        ts = ts.set_index(date_col)
        ts = _dedupe_index(ts, 'last')

        # infer/choose frequency
        inferred = pd.infer_freq(ts.index)
        eff_freq = None if cfg["freq"] == "raw" else cfg["freq"]
        if eff_freq is None and inferred:
            eff_freq = inferred
        if eff_freq:
            ts = ts.resample(eff_freq).sum(numeric_only=True)

        y = ts[target_col].astype(float)
        y_tr = y.copy()
        # >>> INSERT YOUR TRANSFORM BLOCK HERE <<<
        transformer = ("none",)
        if cfg["transform"] == "log1p":
            y_tr = np.log1p(np.clip(y, a_min=0, a_max=None))
            transformer = ("log1p",)
        elif cfg["transform"] == "boxcox":
            try:
                from scipy.stats import boxcox
                shift = max(1e-6, -(y.min()) + 1e-6)
                y_bc, lam = boxcox((y + shift).values)
                y_tr = pd.Series(y_bc, index=y.index)
                transformer = ("boxcox", lam, shift)
            except Exception:
                y_tr = np.log1p(np.clip(y, a_min=0, a_max=None))
                transformer = ("log1p",)
        # <<< END TRANSFORM BLOCK >>>

        # now it's safe to use y_tr
        X = pd.DataFrame(index=y_tr.index)
        # ... build lags/exog, CV folds, call _fit_predict, etc.

        # fill missing values
        mv = cfg["mv"]
        if mv == "ffill": y = y.ffill()
        elif mv == "bfill": y = y.bfill()
        elif mv == "zero": y = y.fillna(0.0)
        elif mv == "mean": y = y.fillna(y.mean())
        elif mv == "median": y = y.fillna(y.median())
        elif mv == "mode":
            try: y = y.fillna(y.mode().iloc[0])
            except Exception: y = y.fillna(y.median())
        elif mv == "interpolate_linear": y = y.interpolate(method="linear")
        elif mv == "interpolate_time":
            try: y = y.interpolate(method="time")
            except Exception: y = y.interpolate(method="linear")
        elif mv == "constant": y = y.fillna(cfg["const_val"])
        elif mv == "drop": y = y.dropna()
        # else: none

        def _sanitize_Xy_for_statsmodels(y: pd.Series,
                                        X: pd.DataFrame | None,
                                        cfg: dict) -> tuple[pd.Series, pd.DataFrame | None]:
            """Align y & X; coerce numeric; handle inf/nans per mv; drop all-NaN/constant exog; joint drop."""
            if X is None or getattr(X, "empty", True):
                yy = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                return yy, None

            Xc = X.copy()
            # align to y
            Xc = _dedupe_index(Xc, 'last')
            Xc = Xc.reindex(y.index)
            # coerce numeric
            for c in Xc.columns:
                if not pd.api.types.is_numeric_dtype(Xc[c]):
                    Xc[c] = pd.to_numeric(Xc[c], errors="coerce")

            # replace inf → NaN, then impute per Sidebar
            Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
            mv = (cfg.get("mv") or "median").lower()
            if mv in ("ffill", "forward_fill"):
                Xc = Xc.ffill()
            elif mv in ("bfill", "backfill"):
                Xc = Xc.bfill()
            elif mv == "mean":
                Xc = Xc.apply(lambda s: s.fillna(s.mean()))
            elif mv == "median":
                Xc = Xc.apply(lambda s: s.fillna(s.median()))
            elif mv in ("const", "constant"):
                Xc = Xc.fillna(cfg.get("const_val", 0.0))
            # else: leave as-is; we’ll drop leftovers jointly with y

            # drop all-NaN columns (rare but possible) and constant columns (avoid singularities)
            if not Xc.empty:
                Xc = Xc.loc[:, Xc.notna().any(axis=0)]
                nunique = Xc.nunique(dropna=True) if not Xc.empty else pd.Series(dtype=int)
                keep = [c for c in Xc.columns if int(nunique.get(c, 2)) > 1]
                Xc = Xc[keep] if keep else pd.DataFrame(index=Xc.index)

            # final joint drop across y & X
            both = pd.concat([pd.to_numeric(y, errors="coerce"), Xc], axis=1)
            both.replace([np.inf, -np.inf], np.nan, inplace=True)
            both = both.dropna()

            y_out = both.iloc[:, 0]
            X_out = both.iloc[:, 1:]
            if X_out.empty:
                X_out = None
            return y_out, X_out

        def _sanitize_future_exog_for_statsmodels(
            Xf: pd.DataFrame | None,
            cfg: dict,
            train_cols: list[str] | None = None,
        ) -> pd.DataFrame | None:
            """
            Clean and ALIGN future exog to the columns used during training.
            - Coerces numeric, fixes inf/NaN per Sidebar MV policy
            - Drops extra columns, adds missing ones, preserves column order
            - Ensures shape matches training exog exactly
            """
            if Xf is None or getattr(Xf, "empty", True):
                return None

            Xf = Xf.copy()

            # 1) coerce numeric
            for c in Xf.columns:
                if not pd.api.types.is_numeric_dtype(Xf[c]):
                    Xf[c] = pd.to_numeric(Xf[c], errors="coerce")

            # 2) replace inf -> NaN
            Xf.replace([np.inf, -np.inf], np.nan, inplace=True)

            # 3) impute per Sidebar MV policy (first pass)
            mv = (cfg.get("mv") or "median").lower()
            if mv in ("ffill", "forward_fill"):
                Xf = Xf.ffill()
            elif mv in ("bfill", "backfill"):
                Xf = Xf.bfill()
            elif mv == "mean":
                Xf = Xf.apply(lambda s: s.fillna(s.mean()))
            elif mv == "median":
                Xf = Xf.apply(lambda s: s.fillna(s.median()))
            elif mv in ("const", "constant"):
                Xf = Xf.fillna(cfg.get("const_val", 0.0))

            # ---- ALIGN COLUMNS TO TRAINING ----
            if train_cols is not None:
                # Reindex drops extras and adds missing (as NaN)
                Xf = Xf.reindex(columns=list(train_cols), fill_value=np.nan)

                # Second pass impute to ensure newly created missing columns are filled
                if mv in ("ffill", "forward_fill"):
                    Xf = Xf.ffill().bfill()
                elif mv in ("bfill", "backfill"):
                    Xf = Xf.bfill().ffill()
                elif mv == "mean":
                    Xf = Xf.apply(lambda s: s.fillna(s.mean()))
                elif mv == "median":
                    Xf = Xf.apply(lambda s: s.fillna(s.median()))
                elif mv in ("const", "constant"):
                    Xf = Xf.fillna(cfg.get("const_val", 0.0))

                # As a final safety backstop (should be no-ops in practice)
                Xf = Xf.fillna(0.0)

            return Xf

        # m guess
        def _guess_m(freq_code: str | None) -> int:
            if not freq_code: return 7
            code = freq_code.upper()[0]
            return {"D":7, "W":52, "M":12, "Q":4, "H":24}.get(code, 7)
        m_auto = _guess_m(eff_freq or inferred)
        m_ui = cfg["seasonal_m"]
        m = m_auto if str(m_ui).strip().lower() == "auto" else int(float(m_ui))

        # pattern detect
        def _detect_pattern(yser: pd.Series) -> str:
            if yser.dropna().shape[0] < 10: return "additive"
            roll = yser.rolling(window=max(5, len(yser)//20))
            corr = roll.std().corr(yser)
            return "multiplicative" if (corr is not None and corr > 0.3) else "additive"
        pattern = cfg["pattern"] if cfg["pattern"] != "Auto-detect" else _detect_pattern(y)

        # exogenous features
        def _parse_exog(txt: str):
            txt = (txt or "").strip()
            if not txt: return []
            try:
                if txt.startswith("["): return [s.strip() for s in json.loads(txt)]
            except Exception:
                pass
            return [t.strip() for t in txt.split(",") if t.strip()]

        exog_cols = list(dict.fromkeys(_parse_exog(cfg["exog_text"]) + list(cfg["exog_additional"])))
        X = pd.DataFrame(index=y_tr.index)

        if cfg["use_calendar"]:
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

        # lags
        lag_list = sorted(set(int(l) for l in (cfg["exog_lags"] or [0]) if int(l) >= 0))
        lagged = {}
        for c in list(X.columns):
            for L in lag_list:
                if L == 0: continue
                lagged[f"{c}_lag{L}"] = X[c].shift(L)
        if lagged:
            X = pd.concat([X, pd.DataFrame(lagged)], axis=1)

        if cfg["scale_exog"] and not X.empty:
            X = (X - X.mean()) / X.std(ddof=0)
        X = _dedupe_index(X, 'last').reindex(y_tr.index)
        # ================== Preparation & Validation ==================
        import numpy as np
        import pandas as _pd
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from statsmodels.tsa.stattools import acf, adfuller
        from statsmodels.stats.diagnostic import acorr_ljungbox

        PRIMARY_CMAP, GRID_ALPHA, LINE_W = "viridis", 0.25, 1.8
        PLOT_W,  PLOT_H  = 8.8, 3.0
        PLOT_H2 = 2.6     # compact height to fit an extra plot

        def _theme_axes(ax, title=None):
            ax.grid(alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if title: ax.set_title(title, pad=8, weight="bold")

        st.markdown("## ✅ Preparation & Validation")
        L1, R1 = st.columns([0.5, 0.5], gap="large")

        with L1:
            st.markdown("#### Preparation summary")
            prep_df = _pd.DataFrame(
                [("Inferred frequency", inferred or "—"),
                ("Effective frequency",  eff_freq or "raw"),
                ("Index monotonic ↑",   bool(y.index.is_monotonic_increasing)),
                ("Duplicate timestamps", int(ts.index.duplicated().sum())),
                ("Pattern",              str(pattern)),
                ("Seasonal period guess (m)", int(m)),
                ("Missing values (prepared series)", int(y.isna().sum()))],
                columns=["Metric","Value"]
            )
            st.dataframe(prep_df, use_container_width=True, hide_index=True)

            st.markdown(f"#### Descriptive stats · {target_col}")
            stats = y.describe(percentiles=[.10,.25,.50,.75,.90]).to_frame("value").reset_index(names="stat")
            st.dataframe(stats.style.background_gradient(axis=None, cmap="Greens").format(precision=3),
                        use_container_width=True, hide_index=True)

        with R1:
            st.markdown("#### Exploratory Data Analysis")

            # 1) Line
            fig1, ax1 = plt.subplots(figsize=(PLOT_W, PLOT_H))
            ax1.plot(y.index, y.values, lw=LINE_W, color=get_cmap(PRIMARY_CMAP)(0.25))
            _theme_axes(ax1, f"{target_col} over time")
            st.pyplot(fig1, clear_figure=True)

            # 2) Histogram
            fig2, ax2 = plt.subplots(figsize=(PLOT_W, PLOT_H2))
            ax2.hist(y.dropna().values, bins=30, color=get_cmap(PRIMARY_CMAP)(0.65),
                    edgecolor="white", linewidth=0.6)
            ax2.set_ylabel("Count"); _theme_axes(ax2, f"Distribution of {target_col}")
            st.pyplot(fig2, clear_figure=True)

            # 3) NEW: Rolling KPIs (fills bottom space)
            r_short, r_long = max(3, int(m//4) or 5), max(7, int(m//2) or 15)
            roll_mean_s = y.rolling(r_short).mean()
            roll_mean_l = y.rolling(r_long).mean()
            roll_std    = y.rolling(r_long).std()

            fig3, ax3 = plt.subplots(figsize=(PLOT_W, PLOT_H2))
            ax3.plot(roll_mean_s.index, roll_mean_s.values, lw=1.4, label=f"Mean ({r_short})",
                    color=get_cmap(PRIMARY_CMAP)(0.45))
            ax3.plot(roll_mean_l.index, roll_mean_l.values, lw=1.6, label=f"Mean ({r_long})",
                    color=get_cmap(PRIMARY_CMAP)(0.75))
            ax3.fill_between(roll_std.index,
                            (roll_mean_l - roll_std).values,
                            (roll_mean_l + roll_std).values,
                            alpha=0.15, color=get_cmap(PRIMARY_CMAP)(0.65),
                            label=f"±1σ ({r_long})")
            ax3.legend(loc="upper left", frameon=False)
            _theme_axes(ax3, "Rolling mean & volatility")
            st.pyplot(fig3, clear_figure=True)

        st.divider()
        # ============================================================
        # 2) SEASONALITY & DECOMPOSITION  (same layout/spacing)
        # ============================================================
        st.markdown("## 🌊 Seasonality & Decomposition")
        L2, R2 = st.columns([0.5, 0.5], gap="large")

        if HAVE_STATSM and y.dropna().shape[0] >= max(2*m, 20):
            stl = STL(y.dropna(), period=max(m,1)).fit()

            # ---------- LEFT: tables ----------
            with L2:
                st.markdown("#### Component summaries")
                comp_df = _pd.DataFrame(
                    {"mean":[stl.observed.mean(), stl.trend.mean(), stl.seasonal.mean(), stl.resid.mean()],
                    "std": [stl.observed.std(),  stl.trend.std(),  stl.seasonal.std(),  stl.resid.std()]},
                    index=["Observed","Trend","Seasonal","Resid"]
                ).reset_index(names="component")
                st.dataframe(comp_df.style.background_gradient(axis=None, cmap="Greens").format(precision=3),
                            use_container_width=True, hide_index=True)

                # Diagnostics to balance height
                st.markdown("#### Seasonality diagnostics")
                var = np.nanvar
                tr, se, re = stl.trend.dropna().values, stl.seasonal.dropna().values, stl.resid.dropna().values
                n = min(len(tr), len(se), len(re)); tr, se, re = tr[-n:], se[-n:], re[-n:]
                Ft = max(0.0, 1 - var(re)/max(var(tr + re), 1e-12))
                Fs = max(0.0, 1 - var(re)/max(var(se + re), 1e-12))
                vs, vt, vr = var(se), var(tr), var(re); denom = (vs + vt + vr) or np.nan
                p = lambda x: "–" if (x is None or np.isnan(x)) else f"{100*x:,.1f}%"
                diag_df = _pd.DataFrame({"metric": ["Seasonal strength (Fs)", "Trend strength (Ft)",
                                                    "Variance share · Seasonal", "Variance share · Trend", "Variance share · Residual"],
                                        "value":  [p(Fs), p(Ft), p(vs/denom if denom else np.nan),
                                                    p(vt/denom if denom else np.nan), p(vr/denom if denom else np.nan)]})
                st.dataframe(diag_df, use_container_width=True, hide_index=True)

                st.markdown("#### Autocorrelation & tests")
                y_clean = _pd.Series(y).dropna().values
                nlags = int(min(len(y_clean)//2, max(60, 2*m)))
                ac = acf(y_clean, nlags=nlags, fft=True)
                peaks = [(lag, val) for lag, val in enumerate(ac) if lag > 0]
                peaks.sort(key=lambda t: t[1], reverse=True)
                top_peaks = peaks[:3]
                colA, colB = st.columns([0.55, 0.45])
                with colA:
                    st.dataframe(_pd.DataFrame({"top lag":[p[0] for p in top_peaks] or ["–"],
                                                "ACF":[round(p[1],3) for p in top_peaks] or ["–"]}),
                                use_container_width=True, hide_index=True)
                with colB:
                    adf_p = adfuller(y_clean, autolag="AIC")[1]
                    lags = [m] if m else [10]
                    lb   = acorr_ljungbox(y_clean, lags=lags, return_df=True)
                    lb_p = float(lb["lb_pvalue"].iloc[0])
                    st.dataframe(_pd.DataFrame([("ADF stationarity p", round(adf_p,4)),
                                                (f"Ljung–Box p @ m={m or 10}", round(lb_p,4))],
                                            columns=["test","p-value"]),
                                use_container_width=True, hide_index=True)

            # ---------- RIGHT: plots (stack + 2 bottom fillers) ----------
            with R2:
                # st.markdown("#### STL components")
                # fig, axes = plt.subplots(4, 1, figsize=(PLOT_W, 5.8), sharex=True)
                # cols = [get_cmap(PRIMARY_CMAP)(i) for i in (0.20, 0.40, 0.60, 0.80)]
                # for ax, s, t, c in zip(
                #     axes, [stl.observed, stl.trend, stl.seasonal, stl.resid],
                #     ["Observed","Trend","Seasonal","Resid"], cols
                # ):
                #     ax.plot(s, lw=LINE_W, color=c); _theme_axes(ax, t)
                # fig.tight_layout(); st.pyplot(fig, clear_figure=True)

                # --- Header with hover tooltip ---
                tooltip_html = (
                    "<strong>STL = Seasonal and Trend decomposition using Loess</strong><br>"
                    "<u>Components</u>:<br>"
                    "• <em>Trend (Tₜ)</em> – long-term movement<br>"
                    "• <em>Seasonal (Sₜ)</em> – repeating within-period pattern<br>"
                    "• <em>Remainder/Residual (Rₜ)</em> – leftover noise/anomalies<br>"
                    "<u>Model forms</u>:<br>"
                    "Additive: yₜ = Tₜ + Sₜ + Rₜ<br>"
                    "Multiplicative: yₜ = Tₜ × Sₜ × Rₜ"
                )

                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
                    <h4 style="margin:0;">Seasonal and Trend decomposition using Loess components</h4>
                    <span title="{tooltip_html}" style="cursor:help;font-size:18px;">❓</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --- Optional fallback for users who prefer clicking instead of hover ---
                with st.popover("More about STL"):
                    st.markdown(tooltip_html, unsafe_allow_html=True)

                # --- Your existing plot ---
                fig, axes = plt.subplots(4, 1, figsize=(PLOT_W, 5.8), sharex=True)
                cols = [get_cmap(PRIMARY_CMAP)(i) for i in (0.20, 0.40, 0.60, 0.80)]
                for ax, s, t, c in zip(
                    axes,
                    [stl.observed, stl.trend, stl.seasonal, stl.resid],
                    ["Observed","Trend","Seasonal","Resid"],
                    cols,
                ):
                    ax.plot(s, lw=LINE_W, color=c); _theme_axes(ax, t)
                fig.tight_layout(); st.pyplot(fig, clear_figure=True)

                
                # NEW: ACF bars (visual complement)
                st.markdown("#### Autocorrelation (bars)")
                max_show = min(nlags, 48)
                figc, axc = plt.subplots(figsize=(PLOT_W, 2.6))
                axc.bar(range(1, max_show+1), ac[1:max_show+1], color=get_cmap(PRIMARY_CMAP)(0.55), edgecolor="white", linewidth=0.4)
                _theme_axes(axc, f"ACF (first {max_show} lags)")
                st.pyplot(figc, clear_figure=True)

        else:
            st.info("Not enough points for decomposition — need at least ~2 seasonal cycles.")

        st.divider()
        # ================= Rolling CV & Leaderboard =================
        st.markdown("## 🏆 Leaderboard (Rolling Cross Validation)")
        # metrics 
        def _align_three(x, y1, y2):
            """
            Trim x, y1, y2 to same min length and drop rows where y1/y2 are NaN.
            Works with datetime x.
            """
            x  = np.asarray(x)
            y1 = np.asarray(y1, dtype="float64")
            y2 = np.asarray(y2, dtype="float64")
            n = min(len(x), len(y1), len(y2))
            x, y1, y2 = x[:n], y1[:n], y2[:n]
            mask = (~np.isnan(y1)) & (~np.isnan(y2))
            return x[mask], y1[mask], y2[mask]

        def _align(a, b):
            a = np.asarray(a, dtype=float).ravel()
            b = np.asarray(b, dtype=float).ravel()
            n = min(len(a), len(b))
            return a[:n], b[:n]

        def rmse(a, b):
            a, b = _align(a, b)
            return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")

        def mae(a, b):
            a, b = _align(a, b)
            return float(np.mean(np.abs(a - b))) if len(a) else float("nan")

        def mase(a, b, season=m if m>1 else 1):
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

        # ⬇️ INSERT THIS HELPER RIGHT HERE (before METRIC_FUNS)
        def _align_xy(x, y):
            """Trim x and y to the same length for plotting."""
            n = min(len(x), len(y))
            x_aligned = x[:n]
            y_aligned = np.asarray(y).reshape(-1)[:n]
            return x_aligned, y_aligned

        METRIC_FUNS = {"RMSE": rmse, "MAE": mae, "MASE": mase, "MAPE": mape, "sMAPE": smape}
        primary_metric = cfg["metrics"][0] if cfg["metrics"] else "RMSE"

        # split indices for rolling CV
        y_cv = y_tr.dropna()
        X_cv = X.loc[y_cv.index] if not X.empty else pd.DataFrame(index=y_cv.index)
        n = len(y_cv); H = max(1, cfg["h"]); G = max(0, cfg["gap"])
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

        def _fit_predict(model_name: str,
                        y_train: pd.Series,
                        X_train: pd.DataFrame | None,
                        steps: int,
                        X_future: pd.DataFrame | None):
            """
            Fit model on y_train (+ optional exog) and predict 'steps' ahead.
            Returns (forecast_array, fitted_object).
            Relies on outer-scope vars/flags: cfg, m, pattern, HAVE_STATSM/HAVE_PM/HAVE_TBATS/HAVE_XGB/HAVE_LGBM/HAVE_TFT
            """
            # --------------------------- helpers ---------------------------
            def _params_for(name: str) -> dict:
                key = name.lower().replace("-", "_")
                return cfg.get(key, {})

            def _make_tree_features(y: pd.Series, X: pd.DataFrame | None, horizon_steps: int) -> tuple[pd.DataFrame, pd.Series]:
                base = pd.DataFrame({"y": y.values}, index=y.index)
                base["lag1"] = base["y"].shift(1)
                base["lag7"] = base["y"].shift(min(7, max(1, m)))
                base["lag_m"] = base["y"].shift(max(1, m))
                feat = base[["lag1","lag7","lag_m"]]
                if X is not None and not X.empty:
                    feat = feat.join(X, how="left")
                feat = feat.dropna()
                yy = base.loc[feat.index, "y"]
                return feat, yy

            def _recursive_forecast_tree(model, last_y: np.ndarray, X_hist: pd.DataFrame | None,
                                        X_fut: pd.DataFrame | None, steps: int) -> np.ndarray:
                out = []
                y_hist = pd.Series(last_y, index=y_train.index[-len(last_y):])
                x_hist = X_hist.copy() if X_hist is not None else pd.DataFrame(index=y_hist.index)
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

            # --------------------------- statsmodels family ---------------------------
            if model_name == "ARMA":
                p, q = cfg["arma"]["p"], cfg["arma"]["q"]
                trend = cfg["arma"]["trend"]
                mod = SARIMAX(y_train, order=(p, 0, q), trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                fc = res.forecast(steps=steps)
                return np.asarray(fc), res

            if model_name == "ARIMA":
                p, d, q = cfg["arima"]["p"], cfg["arima"]["d"], cfg["arima"]["q"]
                trend = cfg["arima"]["trend"]
                mod = SARIMAX(y_train, order=(p, d, q), trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                fc = res.forecast(steps=steps)
                return np.asarray(fc), res

            if model_name == "ARIMAX":
                p, d, q = cfg["arimax"]["p"], cfg["arimax"]["d"], cfg["arimax"]["q"]
                trend = cfg["arimax"]["trend"]
                y_tr, X_tr = _sanitize_Xy_for_statsmodels(y_train, X_train, cfg)
                train_cols = list(X_tr.columns) if X_tr is not None else None
                Xf = _sanitize_future_exog_for_statsmodels(X_future, cfg, train_cols=train_cols)

                if X_tr is None or Xf is None:
                    mod = SARIMAX(y_tr, order=(p, d, q), trend=trend,
                                enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False); fc = res.forecast(steps=steps)
                else:
                    mod = SARIMAX(y_tr, exog=X_tr, order=(p, d, q), trend=trend,
                                enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False); fc = res.forecast(steps=steps, exog=Xf)
                return np.asarray(fc), res

            if model_name == "SARIMA":
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
                P, D, Q = cfg["sarimax"]["P"], cfg["sarimax"]["D"], cfg["sarimax"]["Q"]
                p, d, q = cfg["sarimax"]["p"], cfg["sarimax"]["d"], cfg["sarimax"]["q"]
                mm = m if str(cfg["sarimax"]["m"]).lower() == "auto" else int(float(cfg["sarimax"]["m"]))
                y_tr, X_tr = _sanitize_Xy_for_statsmodels(y_train, X_train, cfg)
                train_cols = list(X_tr.columns) if X_tr is not None else None
                Xf = _sanitize_future_exog_for_statsmodels(X_future, cfg, train_cols=train_cols)

                if X_tr is None or Xf is None:
                    mod = SARIMAX(y_tr, order=(p, d, q),
                                seasonal_order=(P, D, Q, max(1, mm)),
                                enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False); fc = res.forecast(steps=steps)
                else:
                    mod = SARIMAX(y_tr, exog=X_tr, order=(p, d, q),
                                seasonal_order=(P, D, Q, max(1, mm)),
                                enforce_stationarity=False, enforce_invertibility=False)
                    res = mod.fit(disp=False); fc = res.forecast(steps=steps, exog=Xf)
                return np.asarray(fc), res

            if model_name == "Auto_ARIMA":
                if not HAVE_PM:
                    mod = SARIMAX(y_train, order=(1, 1, 1))
                    res = mod.fit(disp=False); fc = res.forecast(steps=steps)
                    return np.asarray(fc), res

                pars = cfg["auto_arima"]
                seasonal = bool(pars.get("seasonal", m > 1))
                mm = m if str(pars.get("m", "auto")).lower() == "auto" else int(float(pars["m"]))

                y_tr, X_tr = _sanitize_Xy_for_statsmodels(y_train, X_train, cfg)
                train_cols = list(X_tr.columns) if X_tr is not None else None
                Xf = _sanitize_future_exog_for_statsmodels(X_future, cfg, train_cols=train_cols)

                if X_tr is None or Xf is None:
                    ar = pmd.auto_arima(y_tr, seasonal=seasonal, m=max(1, mm),
                                        stepwise=pars.get("stepwise", True),
                                        suppress_warnings=pars.get("suppress_warnings", True),
                                        max_p=pars.get("max_p", 5), max_q=pars.get("max_q", 5),
                                        max_P=pars.get("max_P", 2), max_Q=pars.get("max_Q", 2),
                                        max_d=pars.get("max_d", 2), max_D=pars.get("max_D", 1))
                    fc = ar.predict(n_periods=steps)
                else:
                    ar = pmd.auto_arima(y_tr, X=X_tr, seasonal=seasonal, m=max(1, mm),
                                        stepwise=pars.get("stepwise", True),
                                        suppress_warnings=pars.get("suppress_warnings", True),
                                        max_p=pars.get("max_p", 5), max_q=pars.get("max_q", 5),
                                        max_P=pars.get("max_P", 2), max_Q=pars.get("max_Q", 2),
                                        max_d=pars.get("max_d", 2), max_D=pars.get("max_D", 1))
                    fc = ar.predict(n_periods=steps, X=Xf)
                return np.asarray(fc), ar

            if model_name == "HWES":
                # pattern from outer scope
                trend_kw = {"add": "add", "mul": "mul"}.get("add" if pattern == "additive" else "mul")
                seasonal_kw = trend_kw
                sp = st.session_state.get("hwes_sp", "m")
                if isinstance(sp, str) and sp.strip().lower() == "m":
                    sp = m
                try:
                    sp = int(float(sp))
                except Exception:
                    sp = m
                mod = ExponentialSmoothing(y_train, trend=trend_kw,
                                        seasonal=(seasonal_kw if sp > 1 else None),
                                        seasonal_periods=max(1, sp))
                res = mod.fit()
                fc = res.forecast(steps)
                return np.asarray(fc), res

            # --------------------------- Prophet ---------------------------
            if model_name == "Prophet":
                if not cfg["models"].get("Prophet", False) or not HAVE_PROPHET:
                    return np.repeat(y_train.iloc[-1], steps), None

                dfp = pd.DataFrame({"ds": pd.to_datetime(y_train.index), "y": y_train.values}).dropna()
                if len(dfp) < 2:
                    return np.repeat(y_train.iloc[-1], steps), None

                from prophet import Prophet
                mprop = Prophet(
                    growth=cfg["prophet"]["growth"],
                    changepoint_prior_scale=cfg["prophet"]["changepoint_prior_scale"],
                    seasonality_mode=cfg["prophet"]["seasonality_mode"],
                    weekly_seasonality=cfg["prophet"]["weekly_seasonality"],
                    yearly_seasonality=cfg["prophet"]["yearly_seasonality"],
                    daily_seasonality=cfg["prophet"]["daily_seasonality"],
                )
                mprop.fit(dfp)

                # Robust freq detection
                freq_str = (y_train.index.freqstr
                            if getattr(y_train.index, "freqstr", None)
                            else (pd.infer_freq(y_train.index) or (eff_freq or "W")))
                future = pd.DataFrame({"ds": pd.date_range(dfp["ds"].iloc[-1], periods=steps, freq=freq_str, inclusive="right")})
                fc = mprop.predict(future)["yhat"].values
                return np.asarray(fc), mprop

            # --------------------------- TBATS -----------------------------
            if model_name == "TBATS":
                if not HAVE_TBATS:
                    return np.repeat(y_train.iloc[-1], steps), None
                from tbats import TBATS
                pars = _params_for("tbats")
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

            # --------------------------- XGBoost ---------------------------
            if model_name == "XGBoost":
                if not HAVE_XGB:
                    return np.repeat(y_train.iloc[-1], steps), None
                import xgboost as xgb
                pars = _params_for("xgboost")
                Xtr, yy = _make_tree_features(y_train, X_train, steps)
                model = xgb.XGBRegressor(
                    n_estimators=pars["n_estimators"], max_depth=pars["max_depth"],
                    learning_rate=pars["learning_rate"], subsample=pars["subsample"],
                    colsample_bytree=pars["colsample_bytree"], reg_alpha=pars["reg_alpha"],
                    reg_lambda=pars["reg_lambda"], tree_method="hist"
                )
                model.fit(Xtr, yy)
                return _recursive_forecast_tree(model, y_train.values, Xtr, X_future, steps), model

            # --------------------------- LightGBM --------------------------
            if model_name == "LightGBM":
                if not cfg["models"].get("LightGBM", False):
                    return np.repeat(y_train.iloc[-1], steps), None
                if not HAVE_LGBM:
                    return np.repeat(y_train.iloc[-1], steps), None

                pars = _params_for("lightgbm")   # NOTE: lowercase 'lightgbm' in cfg
                Xtr, yy = _make_tree_features(y_train, X_train, steps)

                # Clean / validate
                if Xtr is None or len(yy) == 0:
                    return np.repeat(y_train.iloc[-1], steps), None
                Xtr = _clean_numeric_df(Xtr).dropna()
                yy  = pd.to_numeric(yy, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

                # Align rows after NA drops on either side
                both = Xtr.join(yy.rename("y"), how="inner")
                if both.empty or both.shape[1] == 1:  # only 'y' left or nothing
                    return np.repeat(y_train.iloc[-1], steps), None

                yy  = both.pop("y")
                Xtr = both
                if not _is_nonempty_2d(Xtr) or len(yy) == 0:
                    return np.repeat(y_train.iloc[-1], steps), None

                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=pars["n_estimators"],
                    learning_rate=pars["learning_rate"],
                    subsample=pars["subsample"],
                    colsample_bytree=pars["colsample_bytree"],
                    random_state=pars["random_state"],
                )
                model.fit(Xtr, yy)

                # Predict recursively; X_future may be None or have extra/missing cols → align
                Xf = X_future.copy() if (X_future is not None and not X_future.empty) else pd.DataFrame(index=range(steps))
                # Keep exactly the training feature set and order:
                Xf = _clean_numeric_df(Xf.reindex(columns=list(Xtr.columns), fill_value=np.nan))
                Xf = Xf.fillna(0.0)

                return _recursive_forecast_tree(model, y_train.values, Xtr, Xf, steps), model

            # --------------------------- TFT (Darts) -----------------------
            if model_name == "TFT":
                if not cfg["models"].get("TFT", False) or not HAVE_TFT:
                    return np.repeat(y_train.iloc[-1], steps), None

                from darts import TimeSeries
                from darts.models import TFTModel

                pars = _params_for("tft") or {}

                # Safe defaults even if config keys exist but are None/invalid
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

                hidden_size = pars.get("hidden_size") or 16
                n_epochs    = pars.get("n_epochs") or 50
                batch_size  = pars.get("batch_size") or 32
                random_state= pars.get("random_state") or 42

                # Build target series (clean NaNs BEFORE constructing TimeSeries; don't call .dropna() on TimeSeries)
                y_vals = pd.to_numeric(y_train.values, errors="coerce")
                mask   = ~np.isnan(y_vals)
                if mask.sum() < (in_len + out_len):
                    return np.repeat(y_train.iloc[-1], steps_safe), None

                series = TimeSeries.from_times_and_values(
                    times=pd.to_datetime(y_train.index)[mask],
                    values=y_vals[mask],
                )

                # Optional explicit covariates (numeric, fill NaNs)
                def _prep_cov(df):
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        return None
                    df = df.copy()
                    for c in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    return df if df.shape[1] > 0 else None

                Xtr_num = _prep_cov(X_train)
                Xf_num  = _prep_cov(X_future)
                have_covs = (
                    Xtr_num is not None and
                    Xf_num  is not None and
                    list(Xtr_num.columns) == list(Xf_num.columns)
                )

                # Always init with encoders so TFT has future covariates even if we don't pass any explicitly
                tft_kwargs = dict(
                    input_chunk_length=in_len,
                    output_chunk_length=out_len,
                    hidden_size=hidden_size,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    random_state=random_state,
                )
                try:
                    model = TFTModel(add_relative_index=True, **tft_kwargs)
                except TypeError:
                    model = TFTModel(add_encoders={"relative_index": {"future": True}}, **tft_kwargs)

                # ---- Train WITHOUT explicit future_covariates to avoid coverage errors ----
                model.fit(series, verbose=False)

                # ---- Predict: try explicit covariates if we can build a well-covered series; else fallback ----
                if have_covs:
                    tr = Xtr_num.copy(); tr.index = pd.to_datetime(tr.index)
                    fu = Xf_num.copy();  fu.index = pd.to_datetime(fu.index)

                    # For prediction, covariates must cover at least: last `in_len` steps of history up to the end of horizon
                    req_start = series.time_index[-in_len] if len(series) >= in_len else series.start_time()
                    cov_df = pd.concat([tr[tr.index >= req_start], fu], axis=0)
                    cov_df = cov_df.loc[~cov_df.index.duplicated(keep="last")].sort_index()

                    # Build covariate TimeSeries only if it starts early enough
                    if len(cov_df) and cov_df.index.min() <= req_start:
                        cov_full = TimeSeries.from_times_and_values(cov_df.index, cov_df.values)
                        try:
                            fc = model.predict(steps_safe, future_covariates=cov_full).values().ravel()
                            return np.asarray(fc), model
                        except Exception:
                            # Coverage still not matching what Darts wants → fallback to encoder-only
                            pass

                # Encoder-only fallback (robust)
                fc = model.predict(steps_safe).values().ravel()
                return np.asarray(fc), model

# --------------------------- default naive ---------------------
            return np.repeat(y_train.iloc[-1], steps), None

        # perform CV
        rows = []
        model_names = [
            m for m, v in cfg["models"].items() if v and (
                (m == "Prophet"  and HAVE_PROPHET) or
                (m == "LightGBM" and HAVE_LGBM)    or
                (m == "TFT"      and HAVE_TFT)     or
                (m not in {"Prophet", "LightGBM", "TFT"})
            )
        ]

        if not model_names:
            st.warning("No models selected or required libraries are unavailable.")
            return   # this works because you're inside page_getting_started()

        for fold in range(folds):
            train_end = len(y_cv) - (folds - fold)*H - G
            if train_end < min_train: continue
            test_start = train_end + G
            test_end = min(len(y_cv), test_start + H)

            y_tr_fold = y_cv.iloc[:train_end]
            y_te_fold = y_cv.iloc[test_start:test_end]
            X_tr_fold = X_cv.iloc[:train_end] if not X_cv.empty else pd.DataFrame(index=y_tr_fold.index)
            X_te_fold = X_cv.iloc[test_start:test_end] if not X_cv.empty else pd.DataFrame(index=y_te_fold.index)

            for model_name in model_names:
                yhat, _ = _fit_predict(model_name, y_tr_fold, X_tr_fold if not X_tr_fold.empty else None,
                                    steps=len(y_te_fold),
                                    X_future=X_te_fold if not X_te_fold.empty else None)

                # inverse-transform if needed
                if transformer[0] == "log1p":
                    yhat_inv = np.expm1(yhat)
                    y_true_inv = y.iloc[test_start:test_end].values
                elif transformer[0] == "boxcox":
                    lam, shift = transformer[1], transformer[2]
                    # inverse boxcox
                    if lam == 0:
                        yhat_inv = np.exp(yhat) - shift
                    else:
                        yhat_inv = np.power(yhat * lam + 1, 1/lam) - shift
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

        # aggregate by model
        agg = lb.groupby("model").mean(numeric_only=True).reset_index()
        # order by primary metric
        agg = agg.sort_values(primary_metric, ascending=True).reset_index(drop=True)
        st.dataframe(agg.assign(folds=int(folds))[["model","folds","MAE","RMSE","MAPE","MASE","sMAPE"]], use_container_width=True)

        best_model = agg.iloc[0]["model"]
        st.success(f"Best-fit model by **{primary_metric}**: **{best_model}**")

        st.divider()

        # ================== Train best on full series & Forecast ==================
        st.markdown("## 🧪 Forecast & Diagnostics")

        # --- tiny CSS polish for consistent look (now colorful) ---
        st.markdown("""
        <style>
        :root{
        --card-bg1:#faf6ff; --card-bg2:#f0fbff; --card-border:#cfc9ff;
        --note:#6b7280; --note-strong:#4b5563;
        }
        .block-card {
        background: linear-gradient(135deg, var(--card-bg1), var(--card-bg2));
        border: 1px solid var(--card-border);
        padding: 12px 14px; border-radius: 14px;
        box-shadow: 0 6px 18px rgba(75,0,130,.07);
        }
        .block-card h4 {margin: 0 0 6px 0; color:#3b2dbf;}
        .small-note {font-size: 0.9rem; color: var(--note);}
        .small-note b {color: var(--note-strong);}
        .dataframe tbody tr th:only-of-type {vertical-align: middle;}
        </style>
        """, unsafe_allow_html=True)

        # --- Matplotlib theme (adds color, grid, nicer fonts) ---
        plt.rcParams.update({
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e5e7eb",
            "axes.grid": True,
            "grid.color": "#e5e7eb",
            "grid.alpha": 0.6,
            "axes.titleweight": "bold",
            "axes.titlecolor": "#3b2dbf",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "axes.prop_cycle": plt.cycler(color=[
                "#7b5cff","#36cfc9","#ef476f","#ffd166","#06d6a0","#118ab2","#a78bfa"
            ])
        })

        # --- build future index/exog for forecast horizon (unchanged) ---
        if eff_freq:
            future_idx = pd.date_range(y_tr.index[-1], periods=cfg["h"]+1, freq=eff_freq, inclusive="right")
        else:
            future_idx = pd.date_range(y_tr.index[-1], periods=cfg["h"]+1, freq="W", inclusive="right")

        X_full = _dedupe_index(X, 'last').reindex(y_tr.index) if not X.empty else pd.DataFrame(index=y_tr.index)
        X_future = pd.DataFrame(index=future_idx)

        if cfg.get("use_calendar"):
            idx = future_idx
            X_future["dow"] = idx.dayofweek
            X_future["month"] = idx.month
            X_future["is_month_start"] = idx.is_month_start.astype(int)
            X_future["is_month_end"] = idx.is_month_end.astype(int)
            X_future["is_weekend"] = idx.dayofweek.isin([5,6]).astype(int)

        for c in exog_cols:
            if c in dfd.columns:
                # naive persist last value onto the future index
                last = _safe_align_series_to_index(
                    dfd.set_index(pd.to_datetime(dfd[date_col], errors="coerce"))[c], y_tr.index
                ).ffill().iloc[-1]
                X_future[c] = last

        # add lags on FULL matrix, then split future slice
        if not X_full.empty or not X_future.empty:
            allX = pd.concat([X_full, X_future])
            for col in list(allX.columns):
                for L in lag_list:
                    if L == 0:
                        continue
                    allX[f"{col}_lag{L}"] = allX[col].shift(L)
            if cfg.get("scale_exog"):
                mu, sd = allX.mean(), allX.std(ddof=0)
                allX = (allX - mu) / sd
            X_full = allX.loc[y_tr.index]
            X_future = allX.loc[future_idx]

        # fit & predict
        yhat_tr, fitted = _fit_predict(
            best_model, y_tr, X_full if not X_full.empty else None,
            steps=len(future_idx),
            X_future=X_future if not X_future.empty else None
        )

        # inverse transform forecast
        if transformer[0] == "log1p":
            y_fc = np.expm1(yhat_tr)
        elif transformer[0] == "boxcox":
            lam, shift = transformer[1], transformer[2]
            y_fc = np.exp(yhat_tr) - shift if lam == 0 else np.power(yhat_tr * lam + 1, 1/lam) - shift
        else:
            y_fc = yhat_tr

        # compute intervals
        if HAVE_STATSM and hasattr(fitted, "get_forecast"):
            try:
                if best_model in {"Auto_ARIMA"} and HAVE_PM:
                    s = np.std(getattr(fitted, "resid", y.values - np.mean(y.values)))
                    lower = y_fc - 1.28*s; upper = y_fc + 1.28*s
                else:
                    res_fc = fitted.get_forecast(
                        steps=len(future_idx),
                        exog=(X_future if best_model in {"ARIMAX","SARIMAX"} and not X_future.empty else None)
                    )
                    ci = res_fc.conf_int(alpha=0.2)  # 80%
                    lower, upper = ci.iloc[:,0].values, ci.iloc[:,1].values
            except Exception:
                s = np.std(y.values[-m:]) if m > 1 else np.std(y.values)
                lower = y_fc - 1.28*s; upper = y_fc + 1.28*s
        else:
            s = np.std(y.values[-m:]) if m > 1 else np.std(y.values)
            lower = y_fc - 1.28*s; upper = y_fc + 1.28*s

        # -------- derived DataFrames for the UI --------
        # forecast table
        _fx, _y = _align_xy(future_idx, y_fc)
        _,  _lo = _align_xy(_fx, lower)
        _,  _hi = _align_xy(_fx, upper)
        forecast_df = pd.DataFrame({"date": pd.to_datetime(_fx), "yhat": _y, "lo80": _lo, "hi80": _hi}).set_index("date")

        # history table (tail) with quick stats (styled)
        hist_tail = pd.DataFrame({"y": y}).tail(200)
        hist_stats = pd.DataFrame({
            "Metric": ["Start", "End", "Observations", "Freq (effective)", "Last value"],
            "Value": [
                str(y.index.min()) if len(y) else "—",
                str(y.index.max()) if len(y) else "—",
                f"{len(y):,}",
                eff_freq or "raw",
                f"{(y.iloc[-1] if len(y) else np.nan):,.4f}",
            ],
        })

        # make dataframes colorful via Styler (same structure)
        _hist_tail_disp = hist_tail.style.background_gradient(cmap="PuBuGn").format({"y": "{:,.4f}"})
        _hist_stats_disp = hist_stats.style.hide(axis="index").background_gradient(cmap="BuPu")

        # ================== 1) HISTORY ==================
        l, r = st.columns([0.48, 0.52], gap="large")
        with l:
            st.markdown('<div class="block-card"><h4>Series overview</h4>', unsafe_allow_html=True)
            st.dataframe(_hist_tail_disp, use_container_width=True, height=260)
            st.markdown('<div class="small-note">Showing last 200 observations.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            # st.markdown('<div class="block-card">', unsafe_allow_html=True)
            # st.dataframe(_hist_stats_disp, use_container_width=True, height=190)
            # st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="block-card"><h4>Series summary</h4>'
                '<div class="small-note">Start/End, Observations, Effective Freq, and Last value.</div>',
                unsafe_allow_html=True
            )
            st.dataframe(_hist_stats_disp, use_container_width=True, height=190)
            st.markdown("</div>", unsafe_allow_html=True)

        with r:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(y.index, y.values, linewidth=2.2, marker="o", markersize=3, alpha=0.95)
            ax.set_title("History")
            ax.grid(alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # ================== 1.1) DATA QUALITY & DISTRIBUTION (fills the empty space) ==================
        # (keeps same libs & style)
        _y_series = pd.Series(y).dropna()
        n_obs = len(_y_series)
        idx_num = np.arange(n_obs, dtype=float)

        def _pct_change(series, step):
            try:
                return float((series.iloc[-1] / series.iloc[-1-step] - 1) * 100.0) if len(series) > step else np.nan
            except Exception:
                return np.nan

        # choose seasonal lag from effective frequency for YoY-like delta
        _season_map = {"D":7, "W":52, "MS":12, "M":12, "Q":4}
        _season_lag = next((v for k, v in _season_map.items() if str(eff_freq).upper().startswith(k)), 1)

        wow = _pct_change(_y_series, 1)                                            # last vs prev step
        mom = _pct_change(_y_series, 4 if _season_lag in (52, 1) else 1)          # rough 4-step or 1-step
        yoy = _pct_change(_y_series, _season_lag) if _season_lag > 1 else np.nan  # YoY if seasonal

        # simple trend slope via OLS on index order
        try:
            slope = float(np.cov(idx_num, _y_series.values, ddof=0)[0, 1] / (np.var(idx_num) + 1e-12))
        except Exception:
            slope = np.nan

        # stationarity (ADF) if available
        try:
            if HAVE_STATSM:
                from statsmodels.tsa.stattools import adfuller
                adf_p = float(adfuller(_y_series.dropna(), autolag="AIC")[1])
            else:
                adf_p = np.nan
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
        _more_stats_disp = (_more_stats
                            .style.hide(axis="index")
                            .background_gradient(cmap="BuGn")
                            .format({"Value": "{:,.4f}"}))

        l2, r2 = st.columns([0.48, 0.52], gap="large")
        with l2:
            st.markdown('<div class="block-card"><h4>Data quality & trend</h4>', unsafe_allow_html=True)
            st.dataframe(_more_stats_disp, use_container_width=True, height=260)
            st.markdown(
                '<div class="small-note">Slope > 0 ⇒ upward trend. ADF p < 0.05 ⇒ likely stationary.</div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            # histogram
            if n_obs:
                counts, bins, patches = ax.hist(_y_series.values, bins=max(10, int(np.sqrt(max(n_obs, 1)))), alpha=0.50)
                ax.set_title("Distribution of values")
                ax.grid(alpha=0.35)
                # simple smooth line on top of bars
                if len(counts) > 3:
                    c = pd.Series(counts).rolling(3, center=True).mean().values
                    xb = 0.5 * (bins[:-1] + bins[1:])
                    ax.plot(xb, c, linewidth=2.0, color="#7b5cff")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # ================== 2) FORECAST (LEVELS) ==================
        # style forecast table too
        _forecast_disp = (forecast_df.tail(cfg["h"])
                        .style.background_gradient(cmap="YlGnBu", subset=["yhat"])
                        .background_gradient(cmap="OrRd", subset=["hi80"])
                        .background_gradient(cmap="PuRd", subset=["lo80"])
                        .format("{:,.4f}"))

        l, r = st.columns([0.48, 0.52], gap="large")
        with l:
            st.markdown('<div class="block-card"><h4>Forecast table</h4>', unsafe_allow_html=True)
            st.dataframe(_forecast_disp, use_container_width=True, height=300)
            st.markdown(
                '<div class="small-note">Horizon: '
                f'{cfg["h"]} • Model: <b>{best_model}</b></div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with r:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(forecast_df.index, forecast_df["yhat"].values, linewidth=2.2, marker="o", markersize=3, alpha=0.95)
            ax.set_title("Forecast")
            ax.grid(alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # ================== 3) FORECAST INTERVALS ==================
        # styled ranges
        rng = forecast_df[["lo80","hi80"]].tail(cfg["h"]).agg(["min","max"]).T
        _rng_disp = rng.style.background_gradient(cmap="GnBu").format("{:,.4f}")
        _intervals_disp = (forecast_df[["lo80","yhat","hi80"]].tail(cfg["h"])
                        .style.background_gradient(cmap="PuBu", subset=["yhat"])
                        .background_gradient(cmap="YlOrRd", subset=["hi80"])
                        .background_gradient(cmap="Purples", subset=["lo80"])
                        .format("{:,.4f}"))

        l, r = st.columns([0.48, 0.52], gap="large")
        with l:
            st.markdown('<div class="block-card"><h4>Intervals (80%)</h4>', unsafe_allow_html=True)
            st.dataframe(_intervals_disp, use_container_width=True, height=300)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="block-card"><h4>Interval range (min/max)</h4>', unsafe_allow_html=True)
            st.dataframe(_rng_disp, use_container_width=True, height=140)
            st.markdown("</div>", unsafe_allow_html=True)

        with r:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(forecast_df.index, forecast_df["yhat"].values, linewidth=2.2, marker="o", markersize=3, alpha=0.95, label="yhat")
            ax.fill_between(
                forecast_df.index,
                forecast_df["lo80"].values,
                forecast_df["hi80"].values,
                alpha=0.25, color="#7b5cff", label="80% band"
            )
            ax.legend(loc="upper left", frameon=False)
            ax.set_title("Forecast interval (approx)")
            ax.grid(alpha=0.35)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        st.divider()

        # ================== 4) RESIDUALS & CORRELATIONS ==================
        st.markdown("#### Diagnostics")
        if HAVE_STATSM and hasattr(fitted, "resid"):
            resid = pd.Series(fitted.resid, index=y_tr.index[:len(fitted.resid)]).dropna()

            # ACF: table + plot
            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Autocorrelation (top lags)</h4>', unsafe_allow_html=True)
                try:
                    from statsmodels.tsa.stattools import acf as _acf
                    L = min(25, max(2, len(resid)//2))
                    acf_vals = _acf(resid, nlags=L, fft=True)
                    acf_df = pd.DataFrame({"lag": list(range(len(acf_vals))), "acf": acf_vals}).head(L+1)
                    _acf_disp = acf_df.style.hide(axis="index").background_gradient(cmap="Greens").format({"acf": "{:,.3f}"})
                    st.dataframe(_acf_disp, use_container_width=True, height=260)
                except Exception:
                    st.info("ACF values not available.")
                st.markdown("</div>", unsafe_allow_html=True)

            with r:
                fig, ax = plt.subplots(figsize=(8, 3.2))
                plot_acf(resid, ax=ax, lags=min(25, len(resid)//2))
                ax.set_title("Autocorrelation")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

            # PACF: table + plot
            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Partial Autocorrelation (top lags)</h4>', unsafe_allow_html=True)
                try:
                    from statsmodels.tsa.stattools import pacf as _pacf
                    L = min(25, max(2, len(resid)//2))
                    pacf_vals = _pacf(resid, nlags=L, method="ywm")
                    pacf_df = pd.DataFrame({"lag": list(range(len(pacf_vals))), "pacf": pacf_vals}).head(L+1)
                    _pacf_disp = pacf_df.style.hide(axis="index").background_gradient(cmap="Purples").format({"pacf":"{:,.3f}"})
                    st.dataframe(_pacf_disp, use_container_width=True, height=260)
                except Exception:
                    st.info("PACF values not available.")
                st.markdown("</div>", unsafe_allow_html=True)

            with r:
                fig, ax = plt.subplots(figsize=(8, 3.2))
                plot_pacf(resid, ax=ax, lags=min(25, len(resid)//2), method="ywm")
                ax.set_title("Partial Autocorrelation")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

            # Residuals: table + plot (after ACF & PACF)
            _res_stats = pd.DataFrame({
                "Metric": ["Mean", "Std", "Skew", "Kurtosis", "Ljung–Box p-value"],
                "Value": [
                    np.nanmean(resid), np.nanstd(resid), pd.Series(resid).skew(), pd.Series(resid).kurt(),
                    (lambda: __import__("statsmodels.stats.diagnostic", fromlist=["acorr_ljungbox"])
                            .acorr_ljungbox(resid, lags=[min(10, max(1, len(resid)//4))], return_df=True)["lb_pvalue"].iloc[0]
                    )() if True else np.nan
                ],
            })
            _res_stats_disp = _res_stats.style.hide(axis="index").background_gradient(cmap="Blues").format({"Value": "{:,.4f}"})

            l, r = st.columns([0.48, 0.52], gap="large")
            with l:
                st.markdown('<div class="block-card"><h4>Residuals summary</h4>', unsafe_allow_html=True)
                st.dataframe(_res_stats_disp, use_container_width=True, height=210)
                st.markdown("</div>", unsafe_allow_html=True)

            with r:
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.plot(resid.index, resid.values, linewidth=2.0, color="#ef476f")
                ax.set_title(f"Residuals ({best_model})")
                ax.grid(alpha=0.35)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

        else:
            st.info("Residual diagnostics not available for the selected model.")

        st.divider()    

# ============================== Runner ==============================
def _render_app():
    # import streamlit as st

    # # Page meta (safe to call once)
    # try:
    #     st.set_page_config(page_title="Forecast360", page_icon="📈", layout="wide")
    # except Exception:
    #     pass  # Streamlit may rerun; ignore duplicate set_page_config

    # Sidebar & main sections
    try:
        sidebar_getting_started()
    except Exception as e:
        st.sidebar.error(f"Sidebar failed: {e}")

    try:
        page_home()
    except Exception:
        st.title("Forecast360")

    try:
        page_getting_started()
    except Exception as e:
        st.error(f"Error rendering Getting Started: {e}")

    # Full-width CSS (idempotent)
    try:
        if not st.session_state.get("_css_full_width_injected"):
            st.markdown(
                """
                <style>
                section.main > div.block-container{
                    max-width:100% !important;
                    padding-left:12px; padding-right:12px;
                }
                .main .block-container{max-width:100% !important;}
                [data-testid=stAppViewContainer] .block-container{max-width:100% !important;}
                [data-testid=stToolbar]{right:8px;}
                [data-testid=stSidebar] .block-container{padding:0.5rem 0.5rem;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.session_state["_css_full_width_injected"] = True
    except Exception:
        pass



# Execute immediately when script runs under `streamlit run`
_render_app()

# ========================== SNAPSHOT FOR KB (ALL CONTENT, DEDUPED, SUMMARIZED) ==========================
# Drop this WHOLE block at the VERY END of your file. It:
# - Patches Streamlit renderers to capture exactly-what-was-shown (tables, plots, images, text, metrics).
# - On button click, OVERWRITES the KB folder you configured in the sidebar (session_state["base_folder"]).
# - De-duplicates by content hash, adds captions/sections, writes README.md + index.html,
#   and creates meta/summary.json + text/overview.md for LLM-friendly grounding (facts only).
#
# HOW TO LABEL SECTIONS IN YOUR PAGE (optional):
#   kb_set_section("Leaderboard")
#   ... render plots/tables ...
#
# A "📸 Capture Snapshot…" button will appear in the sidebar and inline; click it anytime.

# ---- safe import ----
try:
    import streamlit as _st
except Exception:
    _st = None

# ---- public helper: set current logical section to group outputs in the report ----
def kb_set_section(name: str):
    try:
        _st.session_state["_cap_section"] = str(name).strip()[:200] or "Page"
    except Exception:
        pass

# ---- internal helpers to extract figure captions (best-effort, real objects only) ----
def _kb_caption_mpl(fig):
    try:
        st = getattr(fig, "_suptitle", None)
        if st is not None and getattr(st, "get_text", None):
            t = st.get_text()
            if t: return t
        for ax in fig.axes:
            t = getattr(ax, "get_title", lambda: "")()
            if t: return t
    except Exception:
        pass
    return "Matplotlib figure"

def _kb_caption_plotly(fig):
    try:
        t = fig.layout.title.text
        if t: return t
    except Exception:
        pass
    return "Plotly figure"

def _kb_caption_altair(ch):
    try:
        t = ch.title
        if isinstance(t, str) and t.strip():
            return t
        if hasattr(t, "to_dict"):
            d = t.to_dict(); tt = d.get("text") or d.get("name")
            if tt: return str(tt)
    except Exception:
        pass
    return "Altair chart"

def _kb_now_section():
    try:
        return _st.session_state.get("_cap_section") or "Page"
    except Exception:
        return "Page"

# ---- registry for captured artifacts ----
def _kb_reg():
    if _st is None: return {}
    ss = _st.session_state
    ss.setdefault("_cap_tables", [])          # [(name, DataFrame, section)]
    ss.setdefault("_cap_mpl_png", [])         # [(name, png_bytes, section, caption)]
    ss.setdefault("_cap_plotly", [])          # [(name, go.Figure, section, caption)]
    ss.setdefault("_cap_altair", [])          # [(name, alt.Chart, section, caption)]
    ss.setdefault("_cap_images", [])          # [(name, "pil"/"ndarray"/"url", payload, section)]
    ss.setdefault("_cap_texts", [])           # [(name, text, section)]
    ss.setdefault("_cap_metrics", [])         # [(label, value, delta)]
    return ss

# ---- patch Streamlit renderers to capture WHAT WAS SHOWN ----
def _kb_patch_renderers():
    if _st is None: return
    ss = _kb_reg()
    if ss.get("_cap_patched"): return

    # ---------- text helper ----------
    def _push_texts(*args):
        for x in args:
            if isinstance(x, str) and x.strip():
                _st.session_state["_cap_texts"].append(("text", x, _kb_now_section()))

    # ---------- table / dataframe ----------
    if hasattr(_st, "table"):
        _orig_table = _st.table
        def _wrap_table(data, *a, **k):
            try:
                import pandas as pd
                df = data if getattr(data, "__class__", None).__name__ == "DataFrame" else pd.DataFrame(data)
                _st.session_state["_cap_tables"].append(("table", df, _kb_now_section()))
            except Exception: pass
            return _orig_table(data, *a, **k)
        _st.table = _wrap_table

    if hasattr(_st, "dataframe"):
        _orig_dataframe = _st.dataframe
        def _wrap_dataframe(data, *a, **k):
            try:
                import pandas as pd
                df = data if getattr(data, "__class__", None).__name__ == "DataFrame" else pd.DataFrame(data)
                _st.session_state["_cap_tables"].append(("dataframe", df, _kb_now_section()))
            except Exception: pass
            return _orig_dataframe(data, *a, **k)
        _st.dataframe = _wrap_dataframe

    # ---------- matplotlib (grab PNG BYTES BEFORE Streamlit clears figure) ----------
    if hasattr(_st, "pyplot"):
        _orig_pyplot = _st.pyplot
        def _mpl_has_content(fig) -> bool:
            try:
                if not getattr(fig, "axes", None): return False
                for ax in fig.axes:
                    if getattr(ax, "has_data", lambda: False)(): return True
                    if getattr(ax, "images", None) and ax.images: return True
                    if getattr(ax, "lines", None) and ax.lines: return True
                    if getattr(ax, "collections", None) and ax.collections: return True
                    if getattr(ax, "patches", None) and sum(p.get_visible() for p in ax.patches) > 1: return True
                return False
            except Exception:
                return True
        def _wrap_pyplot(fig=None, *a, **k):
            import io
            try:
                import matplotlib, matplotlib.pyplot as plt
                try: matplotlib.use("Agg", force=False)
                except Exception: pass
                if fig is None:
                    fig = plt.gcf()
                if fig is not None and _mpl_has_content(fig):
                    caption = _kb_caption_mpl(fig)
                    fig.canvas.draw()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                    buf.seek(0)
                    _st.session_state["_cap_mpl_png"].append(("pyplot", buf.read(), _kb_now_section(), caption))
            except Exception: pass
            return _orig_pyplot(fig, *a, **k)
        _st.pyplot = _wrap_pyplot

    # ---------- plotly ----------
    if hasattr(_st, "plotly_chart"):
        _orig_plotly = _st.plotly_chart
        def _wrap_plotly(fig, *a, **k):
            try:
                import plotly.graph_objects as go
                if isinstance(fig, go.Figure):
                    _st.session_state["_cap_plotly"].append(("plotly_chart", fig, _kb_now_section(), _kb_caption_plotly(fig)))
            except Exception: pass
            return _orig_plotly(fig, *a, **k)
        _st.plotly_chart = _wrap_plotly

    # ---------- altair ----------
    if hasattr(_st, "altair_chart"):
        _orig_altair = _st.altair_chart
        def _wrap_altair(chart, *a, **k):
            try:
                import altair as alt
                if isinstance(chart, alt.Chart):
                    _st.session_state["_cap_altair"].append(("altair_chart", chart, _kb_now_section(), _kb_caption_altair(chart)))
            except Exception: pass
            return _orig_altair(chart, *a, **k)
        _st.altair_chart = _wrap_altair

    # ---------- images ----------
    if hasattr(_st, "image"):
        _orig_image = _st.image
        def _wrap_image(image, *a, **k):
            try:
                import numpy as np
                from PIL import Image as PILImage
                if isinstance(image, PILImage):
                    _st.session_state["_cap_images"].append(("image", "pil", image.copy(), _kb_now_section()))
                elif isinstance(image, np.ndarray):
                    _st.session_state["_cap_images"].append(("image", "ndarray", image.copy(), _kb_now_section()))
                elif isinstance(image, str):
                    _st.session_state["_cap_images"].append(("image", "url", image, _kb_now_section()))
            except Exception: pass
            return _orig_image(image, *a, **k)
        _st.image = _wrap_image

    # ---------- text / markdown / titles ----------
    for _name in ("write","markdown","caption","text","subheader","header","title","info","warning","error","success"):
        if hasattr(_st, _name):
            _orig = getattr(_st, _name)
            def _factory(orig):
                def _wrapped(*a, **k):
                    try: _push_texts(*a)
                    except Exception: pass
                    return orig(*a, **k)
                return _wrapped
            setattr(_st, _name, _factory(_orig))

    # ---------- metrics ----------
    if hasattr(_st, "metric"):
        _orig_metric = _st.metric
        def _wrap_metric(label, value, delta=None, *a, **k):
            try:
                _st.session_state["_cap_metrics"].append((str(label), str(value), None if delta is None else str(delta)))
            except Exception: pass
            return _orig_metric(label, value, delta, *a, **k)
        _st.metric = _wrap_metric

    ss["_cap_patched"] = True
    ss["_cap_just_patched"] = True

# ---- run patch once & force a single rerun so hooks apply to the next render ----
if _st is not None:
    _kb_patch_renderers()
    if _st.session_state.get("_cap_just_patched"):
        _st.session_state["_cap_just_patched"] = False
        try: _st.rerun()
        except Exception: pass

# ----------------------------- SNAPSHOT WRITER -----------------------------------
def _kb_safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in str(s))[:120].strip("_")

def _kb_root():
    # Use the app's configured KB folder from the sidebar
    base = _st.session_state.get("base_folder") if _st is not None else None
    from pathlib import Path; import shutil, os
    root = Path(base).resolve() if base else Path.cwd() / "KB"
    try:
        if root.exists(): shutil.rmtree(root)
    except Exception: pass
    for p in ("tables","plots/mpl","plots/plotly","plots/altair","images","text","meta"):
        (root/p).mkdir(parents=True, exist_ok=True)
    return root

# ---- derivative factual summaries (for LLMs; no hallucination) ----
def _kb__write_derivative_summaries(root, items):
    import json, html as _html
    from pathlib import Path
    ss = _st.session_state if _st is not None else {}

    # Totals
    counts = {
        "tables": sum(1 for it in items if it["type"]=="table_csv"),
        "mpl":    sum(1 for it in items if it.get("lib")=="matplotlib"),
        "plotly": sum(1 for it in items if it.get("lib")=="plotly"),
        "altair": sum(1 for it in items if it.get("lib")=="altair" or it["type"]=="plot_spec"),
        "images": sum(1 for it in items if it["type"].startswith("image")),
        "text":   sum(1 for it in items if it["type"].startswith("text")),
    }

    # Try to find a "leaderboard-like" table for best model/metrics (purely heuristic, factual)
    best = {}; metrics = {}; forecast = {}; resid = {}
    try:
        import pandas as pd
        cap_tables = list(ss.get("_cap_tables", []))
        for t in cap_tables:
            name, df = t[0], t[1]
            cols = [str(c).lower() for c in getattr(df, "columns", [])]
            if not cols or len(df)==0: 
                continue
            # leaderboard?
            if any("model" in c or "algo" in c for c in cols) and any(m in cols for m in ("rmse","mae","mape","r2","smape","mase")):
                row0 = df.iloc[0].to_dict()
                model_col = next((c for c in df.columns if str(c).lower() in ("model","algo","algorithm","estimator","name")), None)
                if model_col:
                    best = {"model": str(row0.get(model_col)), "from_table": name}
                for k, v in row0.items():
                    lk = str(k).lower()
                    if lk in ("rmse","mae","mape","r2","smape","mase"):
                        metrics[lk] = v
                if best or metrics: 
                    break
            # forecast-like?
            if ("yhat" in cols or "forecast" in cols or "prediction" in cols) and ("ds" in cols or "date" in cols or "timestamp" in cols):
                date_col = next((c for c in df.columns if str(c).lower() in ("ds","date","timestamp")), None)
                if date_col is not None:
                    ds = pd.to_datetime(df[date_col], errors="coerce").dropna()
                    if len(ds)>0:
                        forecast = {"rows": int(len(ds)), "start_date": str(ds.min().date()), "end_date": str(ds.max().date()), "from_table": name}
            # residuals summary?
            if set(("metric","value")).issubset(set(cols)):
                tmp = df.copy(); tmp.columns = [str(c).lower() for c in tmp.columns]
                mm = {}
                for _, r in tmp.iterrows():
                    key = str(r.get("metric","")).lower()
                    val = r.get("value")
                    if key in ("rmse","mae","mape","r2","smape","mase","std","mean"):
                        mm[key] = val
                if mm: 
                    resid = {"metrics": mm, "from_table": name}
    except Exception:
        pass

    summary = {
        "totals": counts,
        "best_model": best,
        "metrics": metrics,
        "forecast": forecast,
        "residuals_summary": resid,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # overview.md (human-readable, factual)
    lines = ["# Snapshot Overview (factual)", ""]
    lines.append(f"- Tables: **{counts['tables']}**  |  Matplotlib: **{counts['mpl']}**  |  Plotly: **{counts['plotly']}**  |  Altair: **{counts['altair']}**  |  Images: **{counts['images']}**  |  Text: **{counts['text']}**")
    if best:
        src = best.get("from_table", "")
        lines.append(f"- Best model: **{best.get('model','-')}**" + (f"  _(source: {src})_" if src else ""))
    if metrics:
        lines.append("- Key metrics:")
        for k, v in metrics.items():
            lines.append(f"  - **{k.upper()}**: {v}")
    if forecast:
        lines.append(f"- Forecast horizon: **{forecast.get('rows','?')}** rows")
        if forecast.get("start_date") and forecast.get("end_date"):
            lines.append(f"  - Range: {forecast['start_date']} → {forecast['end_date']}")
        if forecast.get("from_table"):
            lines.append(f"  - Source table: `{forecast['from_table']}`")
    if resid.get("metrics"):
        lines.append("- Residuals summary (selected):")
        for k, v in resid["metrics"].items():
            lines.append(f"  - **{k}**: {v}")
        if resid.get("from_table"):
            lines.append(f"  - Source table: `{resid['from_table']}`")
    (root / "text" / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return [
        {"type":"json","name":"summary","path":str((root / "meta" / "summary.json").as_posix())},
        {"type":"text_md","name":"overview","path":str((root / "text" / "overview.md").as_posix())},
    ]


# ---- BACKWARD-COMPAT: normalize old bucket tuples to the new shape ----
def _kb__normalize_buckets(ss):
    """Ensure all capture buckets have (name, payload, section[, caption]) tuples."""
    # tables: (name, df) -> (name, df, section)
    if "_cap_tables" in ss:
        out = []
        for t in list(ss["_cap_tables"]):
            if len(t) >= 3:
                out.append((t[0], t[1], t[2]))
            elif len(t) == 2:
                out.append((t[0], t[1], _kb_now_section()))
        ss["_cap_tables"] = out

    # mpl: (name, png) -> (name, png, section, caption)
    if "_cap_mpl_png" in ss:
        out = []
        for t in list(ss["_cap_mpl_png"]):
            if len(t) >= 4:
                out.append((t[0], t[1], t[2], t[3]))
            elif len(t) == 3:
                out.append((t[0], t[1], t[2], "Matplotlib figure"))
            elif len(t) == 2:
                out.append((t[0], t[1], _kb_now_section(), "Matplotlib figure"))
        ss["_cap_mpl_png"] = out

    # plotly: (name, fig) -> (name, fig, section, caption)
    if "_cap_plotly" in ss:
        out = []
        for t in list(ss["_cap_plotly"]):
            if len(t) >= 4:
                out.append((t[0], t[1], t[2], t[3]))
            elif len(t) == 3:
                out.append((t[0], t[1], t[2], _kb_caption_plotly(t[1])))
            elif len(t) == 2:
                out.append((t[0], t[1], _kb_now_section(), _kb_caption_plotly(t[1])))
        ss["_cap_plotly"] = out

    # altair: (name, chart) -> (name, chart, section, caption)
    if "_cap_altair" in ss:
        out = []
        for t in list(ss["_cap_altair"]):
            if len(t) >= 4:
                out.append((t[0], t[1], t[2], t[3]))
            elif len(t) == 3:
                out.append((t[0], t[1], t[2], _kb_caption_altair(t[1])))
            elif len(t) == 2:
                out.append((t[0], t[1], _kb_now_section(), _kb_caption_altair(t[1])))
        ss["_cap_altair"] = out

    # images: (name, kind, payload) -> (name, kind, payload, section)
    if "_cap_images" in ss:
        out = []
        for t in list(ss["_cap_images"]):
            if len(t) >= 4:
                out.append((t[0], t[1], t[2], t[3]))
            elif len(t) == 3:
                out.append((t[0], t[1], t[2], _kb_now_section()))
        ss["_cap_images"] = out

    # texts: (name, text) -> (name, text, section)
    if "_cap_texts" in ss:
        out = []
        for t in list(ss["_cap_texts"]):
            if len(t) >= 3:
                out.append((t[0], t[1], t[2]))
            elif len(t) == 2:
                out.append((t[0], t[1], _kb_now_section()))
        ss["_cap_texts"] = out


# ---- main capture (DEDUPED + RESET) ----
def _kb_capture_now():
    if _st is None: return
    import json, time, io, hashlib, html
    from pathlib import Path

    # normalize old-format tuples to the new shape
    _kb__normalize_buckets(_st.session_state)
    def h_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
    def h_text(s: str) -> str:    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

    root = _kb_root()
    items = []
    ss = _st.session_state

    # -------- Tables (dedupe by CSV content) --------
    tbl_map = {}
    for name, df, section in list(ss.get("_cap_tables", [])):
        try:
            buf = io.StringIO(); df.to_csv(buf, index=False)
            d = h_text(buf.getvalue())
            if d not in tbl_map: tbl_map[d] = (name, df, section)
        except Exception: pass
    for i, (d, (name, df, section)) in enumerate(tbl_map.items(), 1):
        try:
            p = root / "tables" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.csv"
            df.to_csv(p, index=False)
            items.append({"type":"table_csv","path":str(p),"hash":d,"name":name,"section":section,
                          "columns":[str(c) for c in df.columns], "rows": int(getattr(df, "shape", [0,0])[0])})
        except Exception: pass

    # -------- Matplotlib (bytes captured before clear) --------
    mpl_map = {}
    for name, png, section, caption in list(ss.get("_cap_mpl_png", [])):
        try:
            d = h_bytes(png)
            if d not in mpl_map: mpl_map[d] = (name, png, section, caption)
        except Exception: pass
    for i, (d, (name, png, section, caption)) in enumerate(mpl_map.items(), 1):
        try:
            p = root / "plots" / "mpl" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.png"
            with open(p, "wb") as f: f.write(png)
            items.append({"type":"plot_png","lib":"matplotlib","path":str(p),"hash":d,"name":name,"section":section,"caption":caption})
        except Exception: pass

    # -------- Plotly (hash via PNG if kaleido, else JSON spec) --------
    has_kaleido = False
    try:
        import kaleido  # noqa: F401
        has_kaleido = True
    except Exception:
        pass
    pl_map = {}
    for name, fig, section, caption in list(ss.get("_cap_plotly", [])):
        try:
            if has_kaleido:
                pngb = fig.to_image(format="png")
                d = h_bytes(pngb)
                if d not in pl_map: pl_map[d] = (name, ("png", pngb, fig), section, caption)
            else:
                d = h_text(fig.to_json())
                if d not in pl_map: pl_map[d] = (name, ("html", None, fig), section, caption)
        except Exception: pass
    for i, (d, (name, payload, section, caption)) in enumerate(pl_map.items(), 1):
        kind, pngb, fig = payload
        try:
            if kind == "png":
                p = root / "plots" / "plotly" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.png"
                with open(p, "wb") as f: f.write(pngb)
                items.append({"type":"plot_png","lib":"plotly","path":str(p),"hash":d,"name":name,"section":section,"caption":caption})
            else:
                p = root / "plots" / "plotly" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.html"
                fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
                items.append({"type":"plot_html","lib":"plotly","path":str(p),"hash":d,"name":name,"section":section,"caption":caption})
        except Exception: pass

    # -------- Altair (hash by JSON spec) --------
    alt_map = {}
    for name, ch, section, caption in list(ss.get("_cap_altair", [])):
        try:
            spec = ch.to_json(); d = h_text(spec)
            if d not in alt_map: alt_map[d] = (name, spec, ch, section, caption)
        except Exception: pass
    saver_ok = False
    try:
        import altair_saver  # noqa: F401
        saver_ok = True
    except Exception:
        pass
    for i, (d, (name, spec, ch, section, caption)) in enumerate(alt_map.items(), 1):
        try:
            if saver_ok:
                from altair_saver import save
                p = root / "plots" / "altair" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.html"
                save(ch, str(p)); items.append({"type":"plot_html","lib":"altair","path":str(p),"hash":d,"name":name,"section":section,"caption":caption})
            else:
                p = root / "plots" / "altair" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.json"
                p.write_text(spec, encoding="utf-8"); items.append({"type":"plot_spec","lib":"altair","path":str(p),"hash":d,"name":name,"section":section,"caption":caption})
        except Exception: pass

    # -------- Images (hash PNG bytes or URL text) --------
    img_map = {}
    try:
        from PIL import Image
        import numpy as np, io as _io
        for name, kind, payload, section in list(ss.get("_cap_images", [])):
            try:
                if kind == "pil":
                    b = _io.BytesIO(); payload.save(b, format="PNG"); data = b.getvalue()
                    d = h_bytes(data); 
                    if d not in img_map: img_map[d] = (name, "png", data, section)
                elif kind == "ndarray":
                    im = Image.fromarray(payload); b = _io.BytesIO(); im.save(b, format="PNG")
                    data = b.getvalue(); d = h_bytes(data)
                    if d not in img_map: img_map[d] = (name, "png", data, section)
                elif kind == "url":
                    d = h_text(str(payload))
                    if d not in img_map: img_map[d] = (name, "url", str(payload), section)
            except Exception: pass
    except Exception:
        pass
    for i, (d, (name, kind, data, section)) in enumerate(img_map.items(), 1):
        try:
            if kind == "png":
                p = root / "images" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.png"
                with open(p, "wb") as f: f.write(data)
                items.append({"type":"image","path":str(p),"hash":d,"name":name,"section":section})
            else:
                p = root / "images" / f"{_kb_safe(name)}_{i:02d}_{d[:8]}.url.txt"
                p.write_text(data, encoding="utf-8")
                items.append({"type":"image_url","path":str(p),"hash":d,"name":name,"section":section})
        except Exception: pass

    # -------- Text + metrics (hash by content) --------
    txt_map = {}
    for name, txt, section in list(ss.get("_cap_texts", [])):
        try:
            d = h_text(txt)
            if d not in txt_map: txt_map[d] = (name, txt, section)
        except Exception: pass
    for i, (d, (name, txt, section)) in enumerate(txt_map.items(), 1):
        try:
            p = root / "text" / f"{i:03d}_{_kb_safe(name)}_{d[:8]}.md"
            p.write_text(f"# {name}\n\n{txt}", encoding="utf-8")
            items.append({"type":"text_md","path":str(p),"hash":d,"name":name,"section":section})
        except Exception: pass
    if ss.get("_cap_metrics"):
        lines = ["# Metrics"]
        for lbl, val, dlt in ss["_cap_metrics"]:
            lines.append(f"- **{lbl}**: {val}" + (f"  (Δ {dlt})" if dlt else ""))
        p = root / "text" / "metrics.md"
        p.write_text("\n".join(lines), encoding="utf-8"); items.append({"type":"text_md","path":str(p),"name":"metrics","section":"Page"})

    # -------- README.md + index.html (friendly, thumbnails/previews) --------
    import html as _html
    sections = {}
    for it in items:
        sec = it.get("section") or "Page"
        sections.setdefault(sec, []).append(it)

    # README.md (simple)
    counts = {
        "tables": sum(1 for it in items if it["type"]=="table_csv"),
        "mpl":    sum(1 for it in items if it.get("lib")=="matplotlib"),
        "plotly": sum(1 for it in items if it.get("lib")=="plotly"),
        "altair": sum(1 for it in items if it.get("lib")=="altair" or it["type"]=="plot_spec"),
        "images": sum(1 for it in items if it["type"].startswith("image")),
        "text":   sum(1 for it in items if it["type"].startswith("text")),
    }
    readme = [f"# KB Snapshot ({time.strftime('%Y-%m-%d %H:%M:%S')})", "", "## Overview"]
    for k, v in counts.items():
        readme.append(f"- **{k.capitalize()}**: {v}")
    from pathlib import Path as _Path
    for sec, group in sections.items():
        readme.append(f"\n## {sec}")
        for t in [x for x in group if x["type"]=="table_csv"]:
            cols = ", ".join(t.get("columns", [])[:10])
            readme.append(f"- Table: `{_Path(t['path']).name}`  — rows: {t.get('rows','?')}, cols: {len(t.get('columns', []))}; columns: {cols}")
        for pth in [x for x in group if x["type"].startswith("plot") or x["type"].startswith("image")]:
            cap = pth.get("caption") or pth.get("name") or _Path(pth["path"]).name
            readme.append(f"- Figure: `{_Path(pth['path']).name}` — {cap}")
        for tx in [x for x in group if x["type"].startswith("text")]:
            readme.append(f"- Text: `{_Path(tx['path']).name}`")
    (root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    # index.html
    css = """
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:24px;}
      .sec{margin-top:28px;}
      .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;}
      .card{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff;}
      .card img{max-width:100%;height:auto;border-radius:8px;}
      .muted{color:#6b7280;font-size:12px;}
      table.preview{border-collapse:collapse;width:100%;font-size:12px}
      table.preview th,table.preview td{border:1px solid #e5e7eb;padding:4px 6px;}
      .type{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;}
      h1{margin-top:0} a{color:#2563eb;text-decoration:none;} a:hover{text-decoration:underline;}
    </style>"""
    html_parts = [f"<html><head><meta charset='utf-8'><title>KB Snapshot</title>{css}</head><body>"]
    html_parts.append(f"<h1>KB Snapshot</h1><div class='muted'>Created {time.strftime('%Y-%m-%d %H:%M:%S')}</div>")
    html_parts.append("<h3>Overview</h3><ul>" + "".join(
        f"<li><b>{_html.escape(k.capitalize())}</b>: {v}</li>" for k, v in counts.items()) + "</ul>")
    for sec, group in sections.items():
        html_parts.append(f"<div class='sec'><h2>{_html.escape(sec)}</h2>")
        figs = [x for x in group if x['type'].startswith('plot') or x['type'].startswith('image')]
        if figs:
            html_parts.append("<div class='grid'>")
            for f in figs:
                cap = _html.escape((f.get("caption") or f.get("name") or _Path(f["path"]).name))
                rel = _Path(f["path"]).as_posix()
                html_parts.append(
                    f"<div class='card'><div class='type'>{_html.escape(f['type'])}</div>"
                    f"<a href='{rel}' target='_blank'><img src='{rel}' alt='{cap}'/></a>"
                    f"<div class='muted'>{cap}</div>"
                    f"<div class='muted'><code>{_html.escape(_Path(rel).name)}</code></div></div>"
                )
            html_parts.append("</div>")
        tbls = [x for x in group if x['type']=='table_csv']
        if tbls:
            html_parts.append("<h3>Tables</h3>")
            for t in tbls:
                rel = _Path(t["path"]).as_posix()
                try:
                    with open(rel, "r", encoding="utf-8") as fh:
                        lines = [next(fh).rstrip("\n") for _ in range(8)]
                    cells = [ln.split(",") for ln in lines]
                    head = cells[0] if cells else []
                    rows = cells[1:] if len(cells) > 1 else []
                    table_html = "<table class='preview'><thead><tr>" + "".join(
                        f"<th>{_html.escape(h)}</th>" for h in head) + "</tr></thead><tbody>"
                    table_html += "".join("<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
                    table_html += "</tbody></table>"
                except Exception:
                    table_html = "<div class='muted'>Preview unavailable</div>"
                html_parts.append(
                    f"<div class='card'><div class='type'>table</div>"
                    f"<div><b>{_html.escape(_Path(rel).name)}</b> — rows: {t.get('rows','?')}, cols: {len(t.get('columns',[]))}</div>"
                    f"{table_html}<div><a href='{rel}' target='_blank'>Open CSV</a></div></div>"
                )
        texts = [x for x in group if x['type'].startswith('text')]
        if texts:
            html_parts.append("<h3>Text</h3><div class='grid'>")
            for tx in texts:
                rel = _Path(tx["path"]).as_posix()
                html_parts.append(
                    f"<div class='card'><div class='type'>text</div>"
                    f"<div><a href='{rel}' target='_blank'>{_html.escape(_Path(rel).name)}</a></div></div>"
                )
            html_parts.append("</div>")
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    (root / "index.html").write_text("\n".join(html_parts), encoding="utf-8")

    # ---- add derivative factual summaries for LLMs ----
    items.extend(_kb__write_derivative_summaries(root, items))

    # ---- manifest + summary.txt ----
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "tables": len(tbl_map), "mpl": len(mpl_map), "plotly": len(pl_map),
            "altair": len(alt_map), "images": len(img_map), "text": len(txt_map),
        },
        "items": items,
    }
    (root / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "summary.txt").write_text(
        "Snapshot saved (deduped).\n" + "\n".join(f"{k}: {v}" for k, v in manifest["counts"].items()),
        encoding="utf-8",
    )

    # ---- reset buckets so next click starts fresh ----
    for key in ("_cap_tables","_cap_mpl_png","_cap_plotly","_cap_altair","_cap_images","_cap_texts","_cap_metrics"):
        try:
            if key in ss: ss[key].clear()
        except Exception: pass

    # trigger re-index on next render (your app's auto-index watches base_folder)
    try:
        _st.session_state["_kb_last_sig"] = None
    except Exception:
        pass

    try: _st.toast(f"Snapshot saved → {root.as_posix()}", icon="💾")
    except Exception: pass

# ---- UI: always show buttons once patch finished ----
if _st is not None and _st.session_state.get("_cap_patched", False):
    _st.sidebar.markdown("### 📸 Snapshot")
    if _st.sidebar.button("Capture Snapshot (tables • plots • images • text)", use_container_width=True, key="kb_snap_sidebar"):
        _kb_capture_now()
        _st.rerun()
    if _st.button("📸 Capture Snapshot of This Page", key="kb_snap_inline"):
        _kb_capture_now()
        _st.rerun()
# =========================================================================================
