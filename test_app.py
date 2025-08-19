# test_app.py
# Streamlit Data Preprocessing Studio — Final Hardened Version

import io
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import altair as alt

# =========================
# Global App Config
# =========================
st.set_page_config(
    page_title="Data Preprocessing Studio",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Constants & Defaults
# =========================
DEFAULT_RANDOM_STATE = 42
DEFAULT_PREVIEW_ROWS = int(os.getenv("DPS_PREVIEW_ROWS", "500"))
DEFAULT_MAX_UPLOAD_MB = int(os.getenv("DPS_MAX_UPLOAD_MB", "200"))

# =========================
# Session State
# =========================
def init_session():
    ss = st.session_state
    ss.setdefault("dps_raw_df", None)
    ss.setdefault("dps_df", None)
    ss.setdefault("dps_history", [])
    ss.setdefault("dps_pipeline", [])
    ss.setdefault("dps_changelog", [])
    ss.setdefault("dps_last_preview", None)
    ss.setdefault("dps_preview_df", None)
    ss.setdefault("dps_settings", {
        "preview_rows": DEFAULT_PREVIEW_ROWS,
        "max_upload_mb": DEFAULT_MAX_UPLOAD_MB,
        "random_state": DEFAULT_RANDOM_STATE,
        "atomic_apply": True,
        "low_memory_mode": True,
    })

init_session()

def S(key: str):
    return st.session_state[key]

def setS(key: str, value):
    st.session_state[key] = value

# =========================
# Utility Helpers
# =========================
def update_preview_sample():
    df = S("dps_df")
    if df is None or df.empty:
        setS("dps_preview_df", df)
        return
    n = S("dps_settings")["preview_rows"]
    rs = S("dps_settings")["random_state"]
    setS("dps_preview_df", df.sample(n=min(len(df), n), random_state=rs).copy())

def push_history(label: str):
    if S("dps_df") is not None:
        hist = S("dps_history")
        hist.append((label, S("dps_df").copy()))
        # Fix: avoid memory blowup, keep only last 10 snapshots
        if len(hist) > 10:
            hist.pop(0)

def undo_last():
    if S("dps_history"):
        label, df_prev = S("dps_history").pop()
        setS("dps_df", df_prev)
        update_preview_sample()
        S("dps_changelog").append(f"↩️ Undo: {label}")
        st.success(f"Undid: {label}")
    else:
        st.info("History is empty. Nothing to undo.")

def reset_all():
    setS("dps_raw_df", None)
    setS("dps_df", None)
    setS("dps_history", [])
    setS("dps_pipeline", [])
    setS("dps_changelog", [])
    setS("dps_last_preview", None)
    setS("dps_preview_df", None)

def dtype_split(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"shape": (0, 0), "columns": [], "dtypes": {}, "missing_total": 0,
                "missing_by_col": {}, "numeric_cols": [], "categorical_cols": [], "describe_numeric": {}}
    num_cols, cat_cols = dtype_split(df)
    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().sort_values(ascending=False).to_dict(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "describe_numeric": df[num_cols].describe().to_dict() if num_cols else {},
    }
    return stats

def compare_stats(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "shape_before": before.get("shape"),
        "shape_after": after.get("shape"),
        "missing_total_before": before.get("missing_total"),
        "missing_total_after": after.get("missing_total"),
        "n_columns_before": len(before.get("columns", [])),
        "n_columns_after": len(after.get("columns", [])),
    }

def safe_alt_histogram(df: pd.DataFrame, column: str, title: str):
    if df is None or df.empty or column not in df.columns:
        st.info("No data available for histogram.")
        return
    if not pd.api.types.is_numeric_dtype(df[column]):
        st.info(f"Column '{column}' is not numeric.")
        return
    chart = (
        alt.Chart(df.dropna(subset=[column]))
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=40), title=column),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip(f"{column}:Q", title=column), alt.Tooltip("count()", title="Count")],
        )
        .properties(height=250, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Preprocessing Step Helpers (robust)
# =========================
# 1) Missing Data
def impute_missing(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if columns is None or len(columns) == 0:
        columns = df.columns.tolist()

    # sanitize constant_value (best-effort cast)
    if strategy == "constant":
        cv = constant_value
        if isinstance(cv, str):
            s = cv.strip()
            try:
                cv = float(s) if "." in s else int(s)
            except Exception:
                cv = s  # keep as string
        constant_value = cv

    num_cols, _ = dtype_split(df)
    applied_cols = []
    if strategy in ("mean", "median"):
        target_cols = [c for c in columns if c in num_cols]
        for c in target_cols:
            val = df[c].mean() if strategy == "mean" else df[c].median()
            df[c] = df[c].fillna(val)
        applied_cols = target_cols
    elif strategy == "mode":
        target_cols = [c for c in columns if c in df.columns]
        for c in target_cols:
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
        applied_cols = target_cols
    elif strategy == "constant":
        target_cols = [c for c in columns if c in df.columns]
        df[target_cols] = df[target_cols].fillna(constant_value)
        applied_cols = target_cols
    msg = f"Imputation strategy '{strategy}' applied to columns: {applied_cols}."
    return df, msg

def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if df.empty:
        return df, "Dataset is empty; nothing to drop."
    if axis == "rows":
        initial = len(df)
        if columns:
            cols = [c for c in columns if c in df.columns]
            df = df.dropna(subset=cols)
            msg = f"Dropped rows missing any of columns {cols}. Removed {initial - len(df)} rows."
        elif threshold is not None:
            row_missing_ratio = df.isna().mean(axis=1)
            df = df.loc[row_missing_ratio < threshold]
            msg = f"Dropped rows with missing ratio ≥ {threshold:.2f}. Removed {initial - len(df)} rows."
        else:
            df = df.dropna()
            msg = f"Dropped rows with any missing values. Removed {initial - len(df)} rows."
    else:
        initial = len(df.columns)
        if threshold is not None:
            col_missing_ratio = df.isna().mean(axis=0)
            to_drop = col_missing_ratio[col_missing_ratio >= threshold].index.tolist()
            df = df.drop(columns=to_drop)
            msg = f"Dropped columns with missing ratio ≥ {threshold:.2f}: {to_drop}"
        else:
            to_drop = [c for c in df.columns if df[c].isna().any()]
            df = df.drop(columns=to_drop)
            msg = f"Dropped columns containing any missing values: {to_drop}"
    if df.empty:
        msg += " Resulting dataset is empty."
    return df, msg

# 2) Data Inconsistency
def normalize_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    trim: bool = True,
    collapse_spaces: bool = True,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    applied = []
    skipped = []
    for c in columns:
        if c not in df.columns:
            skipped.append(c)
            continue
        # operate only on object/string-like cols; keep NA as <NA> (not "None")
        if not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
            skipped.append(c)
            continue
        s = df[c].astype("string")
        if trim:
            s = s.str.strip()
        if lowercase:
            s = s.str.lower()
        if collapse_spaces:
            s = s.str.replace(r"\s+", " ", regex=True)
        df[c] = s
        applied.append(c)
    msg = f"Normalized text for columns: {applied}. Skipped: {skipped}."
    return df, msg

def standardize_dates(df: pd.DataFrame, columns: List[str], output_format: str = "%Y-%m-%d"):
    df = df.copy()
    report = []
    for c in columns:
        if c not in df.columns:
            report.append(f"{c}: not found")
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")  # Fix: removed deprecated arg
        n, failed = len(df), int(parsed.isna().sum())
        df[c] = parsed.dt.strftime(output_format)
        report.append(f"{c}: parsed {n - failed}/{n}, failed {failed}")
    return df, "Standardized dates. " + "; ".join(report)

def unit_convert(df: pd.DataFrame, column: Optional[str] = None, factor: float = 1.0, new_name: Optional[str] = None):
    df = df.copy()
    if not column or column not in df.columns:
        return df, f"Column '{column}' not found for unit conversion."
    numeric = pd.to_numeric(df[column], errors="coerce")
    finite_mask = np.isfinite(numeric)
    out = df[column].copy()
    out.loc[finite_mask] = (numeric.loc[finite_mask] * factor).astype(float)  # Fix: force float
    if new_name:
        df[new_name] = out
        msg = f"Created '{new_name}' from '{column}' (factor={factor}). Non-finite left: {(~finite_mask).sum()}."
    else:
        df[column] = out
        msg = f"Converted '{column}' in place (factor={factor}). Non-finite left: {(~finite_mask).sum()}."
    return df, msg

# 3 & 7) Outliers / Noisy Data
def detect_outliers_mask(
    df: pd.DataFrame, columns: List[str], method: str = "IQR", z_thresh: float = 3.0, iqr_k: float = 1.5
) -> pd.Series:
    if df is None or df.empty or not columns:
        return pd.Series(False, index=df.index if df is not None else pd.Index([]))
    mask = pd.Series(False, index=df.index)
    for c in columns:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = df[c]
        if method == "IQR":
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            mask_c = (x < lower) | (x > upper)
        else:  # Z-score
            mu = x.mean()
            sigma = x.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                mask_c = pd.Series(False, index=x.index)
            else:
                z = (x - mu) / sigma
                mask_c = z.abs() > z_thresh
        mask = mask | mask_c
    return mask

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str,  # 'remove' | 'cap' | 'log1p'
    detect_method: str = "IQR",
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if not columns:
        return df, "No columns selected for outlier handling."
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return df, "No numeric columns available for outlier handling."

    if method == "remove":
        mask = detect_outliers_mask(df, valid_cols, detect_method, z_thresh, iqr_k)
        removed = int(mask.sum())
        df = df.loc[~mask]
        msg = f"Removed {removed} outlier rows using {detect_method} on {valid_cols}."
    elif method == "cap":
        for c in valid_cols:
            if detect_method == "IQR":
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_k * iqr
                upper = q3 + iqr_k * iqr
            else:
                mu = df[c].mean()
                sigma = df[c].std(ddof=0)
                if sigma == 0 or np.isnan(sigma):
                    continue
                lower = mu - z_thresh * sigma
                upper = mu + z_thresh * sigma
            df[c] = df[c].clip(lower=lower, upper=upper)
        msg = f"Capped outliers using {detect_method} thresholds in columns: {valid_cols}."
    else:  # 'log1p'
        for c in valid_cols:
            min_val = df[c].min()
            shift = 1 - min_val if min_val <= 0 else 0
            df[c] = np.log1p(df[c] + shift)
        msg = f"Applied log1p transform to columns: {valid_cols}."
    if df.empty:
        msg += " Resulting dataset is empty."
    return df, msg

# 4) Duplicates
def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    initial = len(df)
    if subset:
        subset = [c for c in subset if c in df.columns]
    df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
    removed = initial - len(df)
    msg = f"Removed {removed} duplicate rows (keep={keep}) using columns: {subset if subset else 'ALL'}."
    if df.empty:
        msg += " Resulting dataset is empty."
    return df, msg

# 5) Categorical Encoding
def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = "onehot") -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df, "No valid columns selected for encoding."

    if method == "onehot":
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=cols, drop_first=False, dummy_na=False)
        added = list(set(df.columns) - before_cols)
        msg = f"One-hot encoded {len(cols)} columns. Added {len(added)} new columns."
    else:  # 'label'
        applied = []
        for c in cols:
            # ✅ Pandas 2.x compatible
            codes, uniques = pd.factorize(df[c], sort=True, use_na_sentinel=True)
            s = pd.Series(codes, index=df.index)
            # -1 means NaN when use_na_sentinel=True
            s = s.mask(s == -1)  
            df[c] = s.astype("Int64")  # nullable integer dtype
            applied.append(c)
        msg = f"Label-encoded columns (nullable Int64): {applied}."
    return df, msg

# 6) Scaling / Normalization
def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard"):
    if df.empty:
        return df.copy(), "Dataset is empty; nothing scaled."  # Fix: safe empty handling
    df = df.copy()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df, "No numeric columns selected for scaling."
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])  # Fix: no .values, keeps col alignment
    return df, f"Applied {scaler.__class__.__name__} to: {cols}."

# 8) Imbalanced Data (basic random over/under sampling)
def rebalance_dataset(
    df: pd.DataFrame,
    target: str,
    method: str = "oversample",
    ratio: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Basic oversampling / undersampling for imbalanced classification datasets.

    Parameters:
        df : DataFrame
            The dataset.
        target : str
            The target column for classification.
        method : str
            "oversample" or "undersample".
        ratio : float
            Oversample to ratio × majority count, or undersample to ratio × minority count.
        random_state : int, optional
            Random seed for reproducibility.

    Returns:
        (DataFrame, str)
            Rebalanced DataFrame and a descriptive message.
    """
    if target not in df.columns:
        return df, f"Target column '{target}' not found."
    if df.empty:
        return df.copy(), "Dataset is empty; skipping rebalancing."

    df = df.copy()
    rs = S("dps_settings")["random_state"] if random_state is None else random_state
    counts = df[target].value_counts(dropna=False)

    if counts.empty or len(counts) <= 1:
        return df, "Target column has only one class or is empty; skipping rebalancing."

    if method == "oversample":
        majority_count = counts.max()
        desired = max(1, int(round(majority_count * ratio)))
        dfs = []
        for cls, cnt in counts.items():
            subset = df[df[target] == cls]
            if cnt < desired:
                to_add = desired - cnt
                add = subset.sample(n=to_add, replace=True, random_state=rs)
                dfs.append(pd.concat([subset, add], axis=0))
            else:
                dfs.append(subset)
        df_bal = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
        msg = f"Oversampled minority classes to ~{desired} rows each (ratio={ratio})."

    else:  # undersample
        minority_count = counts.min()
        desired = max(1, int(round(minority_count * ratio)))
        dfs = []
        for cls, cnt in counts.items():
            subset = df[df[target] == cls]
            if cnt > desired:
                dfs.append(subset.sample(n=desired, replace=False, random_state=rs))
            else:
                dfs.append(subset)
        df_bal = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=rs).reset_index(drop=True)
        msg = f"Undersampled majority classes to ~{desired} rows each (ratio={ratio})."

    if df_bal.empty:
        msg += " Resulting dataset is empty."
    return df_bal, msg

# =========================
# Pipeline Runner
# =========================
def apply_step(df: pd.DataFrame, step: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Dispatch a single step with defensive parameter handling."""
    kind = step.get("kind")
    params = step.get("params", {}) or {}

    # small naming harmonization: accept 'column' or 'columns' for unit_convert
    if kind == "unit_convert" and "columns" in params and "column" not in params:
        cols = params.get("columns") or []
        params["column"] = cols[0] if cols else None

    if kind == "impute":
        return impute_missing(df, **params)
    if kind == "drop_missing":
        return drop_missing(df, **params)
    if kind == "normalize_text":
        return normalize_text(df, **params)
    if kind == "standardize_dates":
        return standardize_dates(df, **params)
    if kind == "unit_convert":
        return unit_convert(df, **params)
    if kind == "outliers":
        return handle_outliers(df, **params)
    if kind == "duplicates":
        return remove_duplicates(df, **params)
    if kind == "encode":
        return encode_categorical(df, **params)
    if kind == "scale":
        return scale_features(df, **params)
    if kind == "rebalance":
        return rebalance_dataset(df, **params)
    return df, f"Unknown step kind: {kind}"

def run_pipeline(df, pipeline, atomic=True):
    msgs, working = [], df.copy()
    for idx, step in enumerate(pipeline, start=1):
        try:
            t0 = time.perf_counter()
            working, msg = apply_step(working, step)
            msgs.append(f"{idx}. {msg} (took {time.perf_counter() - t0:.3f}s)")
        except Exception as e:
            err = f"Step {idx} '{step.get('kind')}' failed: {e}"
            if atomic:
                return df, msgs, err
            msgs.append(f"❌ {err} — continuing.")
    return working, msgs, None

# =========================
# UI: Sidebar - Navigation & Settings
# =========================
def sidebar_navigation_and_settings():
    st.sidebar.title("🧭 Navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "Upload",
            "Missing Data",
            "Data Inconsistency",
            "Outliers / Noisy Data",
            "Duplicates",
            "Categorical Encoding",
            "Scaling / Normalization",
            "Imbalanced Data",
            "Pipeline & Preview",
            "Dashboard & Download",
        ],
    )
    st.sidebar.markdown("---")
    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.sidebar.button("↩️ Undo Last"):
            undo_last()
    with colB:
        if st.sidebar.button("🔄 Reset All"):
            reset_all()
            st.rerun()

    with st.sidebar.expander("⚙️ Settings", expanded=False):
        settings = S("dps_settings")
        settings["preview_rows"] = st.number_input("Preview sample rows", min_value=50, max_value=5000, value=settings["preview_rows"], step=50)
        settings["max_upload_mb"] = st.number_input("Max upload size (MB)", min_value=10, max_value=2000, value=settings["max_upload_mb"], step=10)
        settings["random_state"] = st.number_input("Random seed", min_value=0, max_value=10_000, value=settings["random_state"], step=1)
        settings["atomic_apply"] = st.checkbox("Atomic pipeline (rollback on error)", value=settings["atomic_apply"])
        settings["low_memory_mode"] = st.checkbox("Low-memory previews (use sampling)", value=settings["low_memory_mode"])
        setS("dps_settings", settings)

    st.sidebar.caption("Tip: Add steps to the pipeline from each section, then apply them in batch.")
    return section

# =========================
# UI Sections
# =========================
def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV, chain preprocessing steps, preview changes, and download the cleaned dataset.")

    file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV only. For large files, previews are sampled.")
    if not file:
        st.info("Please upload a CSV file to get started.")
        return

    # Size validation
    size_mb = getattr(file, "size", None)
    if size_mb is not None:
        size_mb = size_mb / (1024 * 1024)
        if size_mb > S("dps_settings")["max_upload_mb"]:
            st.error(f"File is {size_mb:.1f} MB, which exceeds limit ({S('dps_settings')['max_upload_mb']} MB). Increase the limit in Settings or upload a smaller file.")
            return

    # Controls for sample/full load
    load_mode = st.radio("Load mode", ["Full file", "Sample first N rows"], horizontal=True)
    sample_n = st.number_input("N rows to sample (for initial load only)", min_value=100, max_value=500000, value=10000, step=5000, help="Useful for very large files.")
    try:
        file.seek(0)
        if load_mode == "Sample first N rows":
            df = pd.read_csv(file, nrows=sample_n, low_memory=False)
        else:
            df = pd.read_csv(file, low_memory=False)
        setS("dps_raw_df", df.copy())
        setS("dps_df", df.copy())
        setS("dps_history", [])
        setS("dps_pipeline", [])
        setS("dps_changelog", ["📥 Loaded dataset."])
        update_preview_sample()
        st.success(f"Loaded dataset with shape {df.shape}.")
        with st.expander("Peek at data", expanded=True):
            st.dataframe(S("dps_preview_df"))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

def section_missing_data():
    st.header("1) Handling Missing Data")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return

    cols = st.multiselect(
        "Columns to target (optional; default = all)",
        df.columns.tolist(),
        help="Choose specific columns, or leave empty to apply to all columns.",
    )

    with st.expander("Impute options"):
        strategy = st.selectbox("Imputation strategy", ["mean", "median", "mode", "constant"])
        const_val = None
        if strategy == "constant":
            const_val = st.text_input("Constant value", "0", help="Will try to parse as number; if not, keeps as string.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Imputation"):
                preview_df, msg = impute_missing(S("dps_preview_df"), cols, strategy, const_val)
                setS("dps_last_preview", (preview_df, msg))
                st.info(msg)
                st.dataframe(preview_df)
        with c2:
            if st.button("📦 Add to Pipeline (Impute)"):
                step = {"kind": "impute", "params": {"columns": cols, "strategy": strategy, "constant_value": const_val}}
                S("dps_pipeline").append(step)
                st.success("Added to pipeline.")

    with st.expander("Drop options"):
        axis = st.radio("Drop axis", ["rows", "columns"], horizontal=True)
        threshold = st.slider("Missingness threshold (ratio)", 0.0, 1.0, 0.5, 0.05, help="Drop rows/cols with missing ratio ≥ threshold.")
        subset = []
        help_txt = "Rows mode: drop rows with missing ratio ≥ threshold OR drop rows missing any of the selected columns."
        if axis == "rows":
            subset = st.multiselect("Drop rows if these columns are missing (optional)", df.columns.tolist(), help=help_txt)
        c3, c4 = st.columns(2)
        with c3:
            if st.button("🔍 Preview Drop"):
                preview_df, msg = drop_missing(S("dps_preview_df"), axis=axis, threshold=threshold, columns=subset if subset else None)
                setS("dps_last_preview", (preview_df, msg))
                st.info(msg)
                st.dataframe(preview_df)
        with c4:
            if st.button("📦 Add to Pipeline (Drop Missing)"):
                step = {"kind": "drop_missing", "params": {"axis": axis, "threshold": threshold, "columns": subset if subset else None}}
                S("dps_pipeline").append(step)
                st.success("Added to pipeline.")

def section_inconsistency():
    st.header("2) Data Inconsistency")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    num_cols, cat_cols = dtype_split(df)

    st.subheader("Text Normalization")
    text_cols = st.multiselect("Text columns", [c for c in df.columns if c in cat_cols])
    c1, c2, c3 = st.columns(3)
    with c1:
        lower = st.checkbox("lowercase", True)
    with c2:
        trim = st.checkbox("trim spaces", True)
    with c3:
        collapse = st.checkbox("collapse multiple spaces", True)
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("🔍 Preview Text Normalization"):
            preview_df, msg = normalize_text(S("dps_preview_df"), text_cols, lower, trim, collapse)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with cc2:
        if st.button("📦 Add to Pipeline (Normalize Text)"):
            step = {"kind": "normalize_text", "params": {"columns": text_cols, "lowercase": lower, "trim": trim, "collapse_spaces": collapse}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

    st.subheader("Date Standardization")
    date_cols = st.multiselect("Date-like columns", df.columns.tolist(), help="Columns that contain date strings.")
    fmt = st.text_input("Output date format", "%Y-%m-%d", help="Python datetime format string, e.g., %d/%m/%Y")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("🔍 Preview Date Standardization"):
            preview_df, msg = standardize_dates(S("dps_preview_df"), date_cols, fmt)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with dc2:
        if st.button("📦 Add to Pipeline (Standardize Dates)"):
            step = {"kind": "standardize_dates", "params": {"columns": date_cols, "output_format": fmt}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

    st.subheader("Unit Conversion")
    uc_col = st.selectbox("Column to convert", ["(none)"] + df.columns.tolist())
    colA, colB, colC = st.columns(3)
    with colA:
        factor = st.number_input("Multiply by factor", value=1.0, step=0.1, help="e.g., inches→cm = 2.54")
    with colB:
        new_name = st.text_input("New column name (optional)", "", help="Leave blank to overwrite same column")
    with colC:
        st.caption("Tip: This creates normalized units while preserving NA.")
    uc1, uc2 = st.columns(2)
    with uc1:
        if st.button("🔍 Preview Unit Conversion"):
            if uc_col != "(none)":
                preview_df, msg = unit_convert(S("dps_preview_df"), uc_col, factor, new_name or None)
            else:
                preview_df, msg = S("dps_preview_df"), "No column selected."
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with uc2:
        if st.button("📦 Add to Pipeline (Unit Convert)"):
            if uc_col == "(none)":
                st.warning("Please select a column.")
            else:
                step = {"kind": "unit_convert", "params": {"column": uc_col, "factor": factor, "new_name": new_name or None}}
                S("dps_pipeline").append(step)
                st.success("Added to pipeline.")

def section_outliers():
    st.header("3 & 7) Outliers / Noisy Data")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    num_cols, _ = dtype_split(df)
    cols = st.multiselect("Numeric columns to check", num_cols)
    col1, col2, col3 = st.columns(3)
    with col1:
        detect_method = st.selectbox("Detection method", ["IQR", "Z-score"])
    with col2:
        zt = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)
    with col3:
        ik = st.slider("IQR k (fence multiplier)", 0.5, 5.0, 1.5, 0.1)

    act = st.selectbox("Action on outliers", ["remove", "cap", "log1p"], help="Cap uses detection thresholds; log1p is a transform.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Outlier Handling"):
            preview_df, msg = handle_outliers(S("dps_preview_df"), cols, act, detect_method, z_thresh=zt, iqr_k=ik)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Outliers)"):
            step = {"kind": "outliers", "params": {"columns": cols, "method": act, "detect_method": detect_method, "z_thresh": zt, "iqr_k": ik}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

def section_duplicates():
    st.header("4) Data Duplication")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    subset = st.multiselect("Columns to consider (leave empty for all columns)", df.columns.tolist(),
                            help="Duplicates are determined on the selected subset. Empty = all columns.")
    keep_choice = st.radio("Duplicate handling", ["Keep first", "Keep last", "Drop all duplicates"], horizontal=True)
    keep = {"Keep first": "first", "Keep last": "last", "Drop all duplicates": False}[keep_choice]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Duplicate Removal"):
            preview_df, msg = remove_duplicates(S("dps_preview_df"), subset if subset else None, keep=keep)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Duplicates)"):
            step = {"kind": "duplicates", "params": {"subset": subset if subset else None, "keep": keep}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

def section_encoding():
    st.header("5) Categorical Data Handling")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    _, cat_cols = dtype_split(df)
    cols = st.multiselect("Categorical columns", cat_cols)
    method = st.radio("Encoding method", ["onehot", "label"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Encoding"):
            preview_df, msg = encode_categorical(S("dps_preview_df"), cols, method)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Encoding)"):
            step = {"kind": "encode", "params": {"columns": cols, "method": method}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

def section_scaling():
    st.header("6) Feature Scaling & Normalization")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    num_cols, _ = dtype_split(df)
    cols = st.multiselect("Numeric columns to scale", num_cols)
    method = st.radio("Scaler", ["standard", "minmax"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Scaling"):
            preview_df, msg = scale_features(S("dps_preview_df"), cols, method)
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Scaling)"):
            step = {"kind": "scale", "params": {"columns": cols, "method": method}}
            S("dps_pipeline").append(step)
            st.success("Added to pipeline.")

def section_imbalanced():
    st.header("8) Imbalanced Data (Classification)")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    target = st.selectbox("Target column (classification)", ["(none)"] + df.columns.tolist())
    method = st.radio("Method", ["oversample", "undersample"], horizontal=True)
    ratio = st.slider("Ratio", 0.2, 3.0, 1.0, 0.1, help="Oversample to ratio×majority; Undersample to ratio×minority.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Rebalancing"):
            if target != "(none)":
                preview_df, msg = rebalance_dataset(S("dps_preview_df"), target, method, ratio)
            else:
                preview_df, msg = S("dps_preview_df"), "No target chosen."
            setS("dps_last_preview", (preview_df, msg))
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Rebalance)"):
            if target == "(none)":
                st.warning("Please select a target column.")
            else:
                step = {"kind": "rebalance", "params": {"target": target, "method": method, "ratio": ratio}}
                S("dps_pipeline").append(step)
                st.success("Added to pipeline.")

def section_pipeline_preview():
    st.header("9) Pipeline & Preview")
    df = S("dps_df")
    raw = S("dps_raw_df")
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    st.subheader("Queued Steps")
    if not S("dps_pipeline"):
        st.info("Pipeline is empty. Add steps from the sections on the left.")
    else:
        for i, step in enumerate(S("dps_pipeline"), start=1):
            st.write(f"{i}. **{step['kind']}** — {step.get('params', {})}")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🧪 Preview Full Pipeline (on sample)"):
            # Always preview on a sample from the RAW dataset
            rs = S("dps_settings")["random_state"]
            n = S("dps_settings")["preview_rows"]
            if len(raw) > n:
                sample_df = raw.sample(n=n, random_state=rs).copy()
            else:
                sample_df = raw.copy()

            preview_df, msgs, err = run_pipeline(
                sample_df,
                S("dps_pipeline"),
                atomic=S("dps_settings")["atomic_apply"],
            )
            setS("dps_last_preview", (preview_df, "\n".join(msgs) + (f"\nERROR: {err}" if err else "")))
            if err:
                st.error(err)
            else:
                st.success("Pipeline preview complete.")
    with col2:
        if st.button("🚮 Clear Pipeline"):
            setS("dps_pipeline", [])
            st.info("Cleared pipeline.")
    with col3:
        progress_placeholder = st.empty()
        if st.button("✅ Apply Pipeline to Data"):
            if not S("dps_pipeline"):
                st.warning("Pipeline is empty.")
            else:
                push_history("Before pipeline")
                steps = S("dps_pipeline").copy()
                total = len(steps)
                msgs = []
                atomic = S("dps_settings")["atomic_apply"]
                t_all0 = time.perf_counter()
                # ✅ Always apply from the RAW dataset for determinism
                tmp_df = raw.copy()
                error_msg = None
                for i, step in enumerate(steps, start=1):
                    progress_placeholder.progress(i / total, text=f"Applying step {i}/{total}: {step['kind']}")
                    try:
                        t0 = time.perf_counter()
                        tmp_df, msg = apply_step(tmp_df, step)
                        dt = time.perf_counter() - t0
                        msgs.append(f"{i}. {msg} (took {dt:.3f}s)")
                        if tmp_df is None:
                            raise RuntimeError("Step returned None dataframe.")
                    except Exception as e:
                        error_msg = f"Step {i} '{step.get('kind')}' failed: {e}"
                        msgs.append(f"❌ {error_msg}")
                        if atomic:
                            tmp_df = df.copy()  # rollback to last stable dataset
                            break
                        # else continue
                progress_placeholder.empty()
                dt_all = time.perf_counter() - t_all0
                setS("dps_df", tmp_df)
                update_preview_sample()
                for m in msgs:
                    S("dps_changelog").append(f"✅ {m}" if not m.startswith("❌") else m)
                if error_msg:
                    st.error(error_msg + (" — Rolled back changes." if atomic else " — Applied partial changes."))
                else:
                    st.success(f"Applied pipeline to full dataset in {dt_all:.3f}s.")
                setS("dps_pipeline", [])  # always clear after apply

    st.markdown("---")
    if S("dps_last_preview") is not None:
        prev_df, msg = S("dps_last_preview")
        st.subheader("Latest Preview Result")
        with st.expander("Preview Summary", expanded=True):
            st.code(msg or "", language="text")
        st.dataframe(prev_df)

        num_cols, _ = dtype_split(prev_df if prev_df is not None else pd.DataFrame())
        if num_cols:
            column = st.selectbox("Preview histogram for column", num_cols)
            left, right = st.columns(2)
            with left:
                safe_alt_histogram(S("dps_df"), column, "Current Data")
            with right:
                safe_alt_histogram(prev_df, column, "Preview Data")

def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    df = S("dps_df")
    raw = S("dps_raw_df")
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    before_stats = compute_basic_stats(raw)
    after_stats = compute_basic_stats(df)
    comp = compare_stats(before_stats, after_stats)

    t1, t2, t3 = st.tabs(["Summary", "Distributions", "Change Log"])
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows (Before → After)", f"{before_stats['shape'][0]} → {after_stats['shape'][0]}")
        with c2:
            st.metric("Columns (Before → After)", f"{comp['n_columns_before']} → {comp['n_columns_after']}")
        with c3:
            st.metric("Missing Total (Before → After)", f"{comp['missing_total_before']} → {comp['missing_total_after']}")

        st.subheader("Missing by Column (After)")
        miss_after = pd.Series(after_stats["missing_by_col"])
        miss_after = miss_after[miss_after > 0] if not miss_after.empty else miss_after
        if miss_after.empty:
            st.info("No missing values in the current dataset.")
        else:
            st.dataframe(miss_after.rename("missing_count"))

        with st.expander("Dtypes (After)"):
            st.json(after_stats["dtypes"])

        with st.expander("Numeric Describe (After)"):
            if after_stats["numeric_cols"]:
                st.dataframe(pd.DataFrame(after_stats["describe_numeric"]))
            else:
                st.info("No numeric columns present.")

    with t2:
        num_cols, _ = dtype_split(df)
        if not num_cols:
            st.info("No numeric columns to visualize.")
        else:
            col = st.selectbox("Select numeric column", num_cols)
            a, b = st.columns(2)
            with a:
                st.subheader("Before")
                safe_alt_histogram(raw, col, f"Before: {col}")
            with b:
                st.subheader("After")
                safe_alt_histogram(df, col, f"After: {col}")

    with t3:
        if not S("dps_changelog"):
            st.info("No changes yet.")
        else:
            for i, msg in enumerate(S("dps_changelog"), start=1):
                st.write(f"{i}. {msg}")

    st.markdown("---")
    st.subheader("Download Processed Data")
    if df is None or df.empty:
        st.warning("Current dataset is empty; nothing to download.")
    else:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=buf.getvalue(),
            file_name="preprocessed_data.csv",
            mime="text/csv",
            help="Download the final processed dataset as a CSV file.",
        )
    st.caption("All processing happens within your session. Large-file previews use sampling for stability.")

# =========================
# Main App Entrypoint
# =========================
def main():
    section = sidebar_navigation_and_settings()

    if section == "Upload":
        section_upload()
    elif section == "Missing Data":
        section_missing_data()
    elif section == "Data Inconsistency":
        section_inconsistency()
    elif section == "Outliers / Noisy Data":
        section_outliers()
    elif section == "Duplicates":
        section_duplicates()
    elif section == "Categorical Encoding":
        section_encoding()
    elif section == "Scaling / Normalization":
        section_scaling()
    elif section == "Imbalanced Data":
        section_imbalanced()
    elif section == "Pipeline & Preview":
        section_pipeline_preview()
    elif section == "Dashboard & Download":
        section_dashboard_download()

    # Footer
    st.markdown("---")
    st.caption(
        "Pro tip: Use the ⚙️ Settings in the sidebar to control preview size, upload limits, randomness and atomic apply."
    )

if __name__ == "__main__":
    main()
