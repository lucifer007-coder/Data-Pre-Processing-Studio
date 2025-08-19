# Streamlit Data Preprocessing Studio

# Description:
# A complete Streamlit application to upload CSVs and perform common data preprocessing tasks
# with an intuitive, pipeline-based UI and rich previews/dashboards.

# =========================
# Imports
# =========================
import io
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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

# Seed for deterministic sampling
RANDOM_STATE = 42
PREVIEW_ROWS = 500  # sample rows for previews to keep UI snappy


# =========================
# Session State Bootstrap
# =========================
def init_session():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None  # the first loaded DataFrame
    if "df" not in st.session_state:
        st.session_state.df = None  # the working DataFrame
    if "history" not in st.session_state:
        st.session_state.history = []  # stack of (label, df_snapshot)
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []  # list of step dicts
    if "changelog" not in st.session_state:
        st.session_state.changelog = []  # user-readable messages
    if "last_preview" not in st.session_state:
        st.session_state.last_preview = None  # cache preview results (df, summary)


init_session()


# =========================
# Utility Helpers
# =========================
def sample_for_preview(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=RANDOM_STATE).copy()


def push_history(label: str):
    """Save a snapshot for undo, with a helpful label."""
    if st.session_state.df is not None:
        st.session_state.history.append((label, st.session_state.df.copy()))


def undo_last():
    """Undo the last applied step by restoring the previous snapshot."""
    if st.session_state.history:
        label, df_prev = st.session_state.history.pop()
        st.session_state.df = df_prev
        st.session_state.changelog.append(f"↩️ Undo: {label}")
        st.success(f"Undid: {label}")
    else:
        st.info("History is empty. Nothing to undo.")


def reset_all():
    """Clear everything and start fresh."""
    st.session_state.raw_df = None
    st.session_state.df = None
    st.session_state.history = []
    st.session_state.pipeline = []
    st.session_state.changelog = []
    st.session_state.last_preview = None


def dtype_split(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None:
        return {}
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


def alt_histogram(df: pd.DataFrame, column: str, title: str):
    # Build an Altair histogram with automatic binning
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=40), title=column),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip(f"{column}:Q", title=column), alt.Tooltip("count()", title="Count")],
        )
        .properties(height=250, title=title)
        .interactive()
    )
    return chart


# =========================
# Preprocessing Step Helpers
# =========================
# 1) Missing Data
def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if not columns:
        columns = df.columns.tolist()

    num_cols, cat_cols = dtype_split(df)

    if strategy in ("mean", "median"):
        target_cols = [c for c in columns if c in num_cols]
        for c in target_cols:
            val = df[c].mean() if strategy == "mean" else df[c].median()
            df[c] = df[c].fillna(val)
        msg = f"Imputed {len(target_cols)} numeric columns using {strategy}."
    elif strategy == "mode":
        target_cols = columns
        for c in target_cols:
            try:
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
            except Exception:
                pass
        msg = f"Imputed {len(target_cols)} columns using mode."
    elif strategy == "constant":
        val = constant_value if constant_value is not None else 0
        df[columns] = df[columns].fillna(val)
        msg = f"Imputed {len(columns)} columns with constant value {val}."
    else:
        msg = "No imputation performed (unknown strategy)."
    return df, msg


def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    axis: 'rows' or 'columns'
    threshold: if provided for rows => drop rows with missing ratio >= threshold (0-1)
               if provided for columns => drop columns with missing ratio >= threshold
    columns: if provided with rows, drop rows that are NA in ANY of these columns.
    """
    df = df.copy()
    if axis == "rows":
        initial = len(df)
        if columns:
            df = df.dropna(subset=columns)
            msg = f"Dropped rows with missing in selected columns {columns}. Removed {initial - len(df)} rows."
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
            # Drop columns that contain ANY missing values
            to_drop = [c for c in df.columns if df[c].isna().any()]
            df = df.drop(columns=to_drop)
            msg = f"Dropped columns containing any missing values: {to_drop}"
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
    for c in columns:
        if c in df.columns:
            df[c] = df[c].astype(str)
            if trim:
                df[c] = df[c].str.strip()
            if lowercase:
                df[c] = df[c].str.lower()
            if collapse_spaces:
                df[c] = df[c].str.replace(r"\s+", " ", regex=True)
    return df, f"Normalized text for columns: {columns}"


def standardize_dates(
    df: pd.DataFrame, columns: List[str], output_format: str = "%Y-%m-%d"
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    for c in columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = dt.dt.strftime(output_format)
        except Exception:
            pass
    return df, f"Standardized date format to {output_format} for columns: {columns}"


def unit_convert(
    df: pd.DataFrame, column: str, factor: float, new_name: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if column not in df.columns:
        return df, f"Column {column} not found for unit conversion."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, f"Column {column} is not numeric; skipped unit conversion."
    target = new_name if new_name else column
    df[target] = df[column] * factor
    if new_name:
        msg = f"Created '{new_name}' by converting '{column}' with factor {factor}."
    else:
        msg = f"Converted '{column}' in place with factor {factor}."
    return df, msg


# 3 & 7) Outliers / Noisy Data
def detect_outliers_mask(
    df: pd.DataFrame, columns: List[str], method: str = "IQR", z_thresh: float = 3.0, iqr_k: float = 1.5
) -> pd.Series:
    """Return boolean mask where True indicates outlier row (across any selected column)."""
    mask = pd.Series(False, index=df.index)
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
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

    if method == "remove":
        mask = detect_outliers_mask(df, columns, detect_method, z_thresh, iqr_k)
        removed = int(mask.sum())
        df = df.loc[~mask]
        msg = f"Removed {removed} outlier rows using {detect_method} on {columns}."
    elif method == "cap":
        for c in columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
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
        msg = f"Capped outliers using {detect_method} thresholds in columns: {columns}."
    else:  # 'log1p'
        for c in columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                # shift if negative values present
                min_val = df[c].min()
                shift = 1 - min_val if min_val <= 0 else 0
                df[c] = np.log1p(df[c] + shift)
        msg = f"Applied log1p transform to columns: {columns}."
    return df, msg


# 4) Duplicates
def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep: str = "first") -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    initial = len(df)
    df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
    removed = initial - len(df)
    msg = f"Removed {removed} duplicate rows (keep={keep}) using columns: {subset if subset else 'ALL'}."
    return df, msg


# 5) Categorical Encoding
def encode_categorical(
    df: pd.DataFrame, columns: List[str], method: str = "onehot"
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if method == "onehot":
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=columns, drop_first=False, dummy_na=False)
        added = list(set(df.columns) - before_cols)
        msg = f"One-hot encoded {len(columns)} columns. New columns: {len(added)}."
    else:  # label
        le_info = []
        for c in columns:
            if c in df.columns:
                le = LabelEncoder()
                df[c] = df[c].astype(str).fillna("NaN")
                df[c] = le.fit_transform(df[c])
                le_info.append(c)
        msg = f"Label encoded columns: {le_info}"
    return df, msg


# 6) Scaling / Normalization
def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df, "No numeric columns selected for scaling."
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    msg = f"Applied {'StandardScaler' if method == 'standard' else 'MinMaxScaler'} to columns: {cols}."
    return df, msg


# 8) Imbalanced Data (basic random over/under sampling)
def rebalance_dataset(
    df: pd.DataFrame, target: str, method: str = "oversample", ratio: float = 1.0, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, str]:
    """
    method:
      - 'oversample': upsample minority classes to match (ratio * majority count)
      - 'undersample': downsample majority classes to match (ratio * minority count)
    ratio:
      - scaling against the opposite side; e.g., oversample minority to (ratio * max_class_count)
    """
    if target not in df.columns:
        return df, f"Target column '{target}' not found."

    df = df.copy()
    counts = df[target].value_counts(dropna=False)
    if counts.empty or len(counts) <= 1:
        return df, "Target column has only one class or is empty; skipping rebalancing."

    if method == "oversample":
        majority_count = counts.max()
        desired = int(round(majority_count * ratio))
        dfs = []
        for cls, cnt in counts.items():
            subset = df[df[target] == cls]
            if cnt < desired:
                to_add = desired - cnt
                add = subset.sample(n=to_add, replace=True, random_state=random_state)
                dfs.append(pd.concat([subset, add], axis=0))
            else:
                dfs.append(subset)
        df_bal = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        msg = f"Oversampled minority classes to ~{desired} rows each (ratio={ratio})."
    else:  # undersample
        minority_count = counts.min()
        desired = int(round(minority_count * ratio))
        dfs = []
        for cls, cnt in counts.items():
            subset = df[df[target] == cls]
            if cnt > desired:
                dfs.append(subset.sample(n=desired, replace=False, random_state=random_state))
            else:
                dfs.append(subset)
        df_bal = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop_index=True)
        msg = f"Undersampled majority classes to ~{desired} rows each (ratio={ratio})."

    return df_bal, msg


# =========================
# Pipeline Runner
# =========================
def apply_step(df: pd.DataFrame, step: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    kind = step.get("kind")
    params = step.get("params", {})
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


def run_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    msgs = []
    for idx, step in enumerate(pipeline, start=1):
        df, msg = apply_step(df, step)
        msgs.append(f"{idx}. {msg}")
    return df, msgs


# =========================
# UI: Sidebar - Navigation & Controls
# =========================
def sidebar_navigation():
    st.sidebar.title("🧭 Navigation")
    section = st.sidebar.radio(
        "Go to section",
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
        help="Choose what you want to work on.",
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All", help="Clear dataset, pipeline, and history."):
        reset_all()
        st.experimental_rerun()
    if st.sidebar.button("↩️ Undo Last", help="Undo the last applied step."):
        undo_last()
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Use 'Add to Pipeline' on each section, then run them together.")
    return section


# =========================
# UI Sections
# =========================
def section_upload():
    st.title("🧹 Data Preprocessing Studio")
    st.caption("Upload a CSV, chain preprocessing steps, preview changes, and download the cleaned dataset.")
    file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV only. For large files, previews are sampled.")

    if file:
        try:
            df = pd.read_csv(file)
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = []
            st.session_state.pipeline = []
            st.session_state.changelog = ["📥 Loaded dataset."]
            st.success(f"Loaded dataset with shape {df.shape}.")
            with st.expander("Peek at data", expanded=True):
                st.dataframe(sample_for_preview(df))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.info("Please upload a CSV file to get started.")


def section_missing_data():
    st.header("1) Handling Missing Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return
    num_cols, cat_cols = dtype_split(df)
    cols = st.multiselect(
        "Columns to target (optional; default = all)",
        df.columns.tolist(),
        help="Choose specific columns, or leave empty to apply to all.",
    )
    with st.expander("Impute options"):
        strategy = st.selectbox("Imputation strategy", ["mean", "median", "mode", "constant"])
        const_val = None
        if strategy == "constant":
            const_val = st.text_input("Constant value (interpreted as string if not numeric)", "0")
            # Try to parse numeric
            try:
                if "." in const_val:
                    const_val = float(const_val)
                else:
                    const_val = int(const_val)
            except Exception:
                pass
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Preview Imputation"):
                preview_df, msg = impute_missing(sample_for_preview(df), cols, strategy, const_val)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(preview_df)
        with col2:
            if st.button("📦 Add to Pipeline (Impute)"):
                step = {"kind": "impute", "params": {"columns": cols, "strategy": strategy, "constant_value": const_val}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    with st.expander("Drop options"):
        axis = st.radio("Drop axis", ["rows", "columns"], horizontal=True)
        threshold = st.slider("Missingness threshold (ratio)", 0.0, 1.0, 0.5, 0.05, help="Drop rows/cols with missing ratio ≥ threshold.")
        subset = []
        if axis == "rows":
            subset = st.multiselect("For rows: require non-missing in selected columns (optional)", df.columns.tolist(), help="If selected, rows missing any of these will be dropped.")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("🔍 Preview Drop"):
                prev = sample_for_preview(df)
                preview_df, msg = drop_missing(prev, axis=axis, threshold=threshold, columns=subset if subset else None)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(preview_df)
        with col4:
            if st.button("📦 Add to Pipeline (Drop Missing)"):
                step = {
                    "kind": "drop_missing",
                    "params": {"axis": axis, "threshold": threshold, "columns": subset if subset else None},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")


def section_inconsistency():
    st.header("2) Data Inconsistency")
    df = st.session_state.df
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
            prev = sample_for_preview(df)
            preview_df, msg = normalize_text(prev, text_cols, lower, trim, collapse)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with cc2:
        if st.button("📦 Add to Pipeline (Normalize Text)"):
            step = {
                "kind": "normalize_text",
                "params": {"columns": text_cols, "lowercase": lower, "trim": trim, "collapse_spaces": collapse},
            }
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")

    st.subheader("Date Standardization")
    date_cols = st.multiselect("Date-like columns", df.columns.tolist(), help="Columns that contain date strings.")
    fmt = st.text_input("Output date format", "%Y-%m-%d", help="Python datetime format string, e.g., %d/%m/%Y")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("🔍 Preview Date Standardization"):
            prev = sample_for_preview(df)
            preview_df, msg = standardize_dates(prev, date_cols, fmt)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with dc2:
        if st.button("📦 Add to Pipeline (Standardize Dates)"):
            step = {"kind": "standardize_dates", "params": {"columns": date_cols, "output_format": fmt}}
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")

    st.subheader("Unit Conversion")
    uc_col = st.selectbox("Column to convert", ["(none)"] + df.columns.tolist())
    colA, colB, colC = st.columns(3)
    with colA:
        factor = st.number_input("Multiply by factor", value=1.0, step=0.1, help="e.g., inches→cm = 2.54")
    with colB:
        new_name = st.text_input("New column name (optional)", "", help="Leave blank to overwrite same column")
    with colC:
        st.caption("Tip: Use this to create normalized units.")
    uc1, uc2 = st.columns(2)
    with uc1:
        if st.button("🔍 Preview Unit Conversion"):
            prev = sample_for_preview(df)
            if uc_col != "(none)":
                preview_df, msg = unit_convert(prev, uc_col, factor, new_name or None)
            else:
                preview_df, msg = prev, "No column selected."
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with uc2:
        if st.button("📦 Add to Pipeline (Unit Convert)"):
            if uc_col == "(none)":
                st.warning("Please select a column.")
            else:
                step = {
                    "kind": "unit_convert",
                    "params": {"column": uc_col, "factor": factor, "new_name": new_name or None},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")


def section_outliers():
    st.header("3 & 7) Outliers / Noisy Data")
    df = st.session_state.df
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
            prev = sample_for_preview(df)
            preview_df, msg = handle_outliers(prev, cols, act, detect_method, z_thresh=zt, iqr_k=ik)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Outliers)"):
            step = {
                "kind": "outliers",
                "params": {"columns": cols, "method": act, "detect_method": detect_method, "z_thresh": zt, "iqr_k": ik},
            }
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")


def section_duplicates():
    st.header("4) Data Duplication")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return
    subset = st.multiselect("Columns to consider duplicates on (leave empty for all columns)", df.columns.tolist())
    keep = st.selectbox("Keep", ["first", "last", "False"], help="If 'False', drop all duplicates.")
    karg = False if keep == "False" else keep
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Duplicate Removal"):
            prev = sample_for_preview(df)
            preview_df, msg = remove_duplicates(prev, subset if subset else None, keep=karg)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Duplicates)"):
            step = {"kind": "duplicates", "params": {"subset": subset if subset else None, "keep": karg}}
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")


def section_encoding():
    st.header("5) Categorical Data Handling")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return
    _, cat_cols = dtype_split(df)
    cols = st.multiselect("Categorical columns", cat_cols)
    method = st.radio("Encoding method", ["onehot", "label"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Encoding"):
            prev = sample_for_preview(df)
            preview_df, msg = encode_categorical(prev, cols, method)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Encoding)"):
            step = {"kind": "encode", "params": {"columns": cols, "method": method}}
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")


def section_scaling():
    st.header("6) Feature Scaling & Normalization")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return
    num_cols, _ = dtype_split(df)
    cols = st.multiselect("Numeric columns to scale", num_cols)
    method = st.radio("Scaler", ["standard", "minmax"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Scaling"):
            prev = sample_for_preview(df)
            preview_df, msg = scale_features(prev, cols, method)
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Scaling)"):
            step = {"kind": "scale", "params": {"columns": cols, "method": method}}
            st.session_state.pipeline.append(step)
            st.success("Added to pipeline.")


def section_imbalanced():
    st.header("8) Imbalanced Data (Classification)")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return
    target = st.selectbox("Target column (classification)", ["(none)"] + st.session_state.df.columns.tolist())
    method = st.radio("Method", ["oversample", "undersample"], horizontal=True)
    ratio = st.slider("Ratio", 0.2, 3.0, 1.0, 0.1, help="Oversample to ratio×majority; Undersample to ratio×minority.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍 Preview Rebalancing"):
            prev = sample_for_preview(df)
            if target != "(none)":
                preview_df, msg = rebalance_dataset(prev, target, method, ratio)
            else:
                preview_df, msg = prev, "No target chosen."
            st.session_state.last_preview = (preview_df, msg)
            st.info(msg)
            st.dataframe(preview_df)
    with c2:
        if st.button("📦 Add to Pipeline (Rebalance)"):
            if target == "(none)":
                st.warning("Please select a target column.")
            else:
                step = {
                    "kind": "rebalance",
                    "params": {"target": target, "method": method, "ratio": ratio},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")


def section_pipeline_preview():
    st.header("9) Pipeline & Preview")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    st.subheader("Queued Steps")
    if not st.session_state.pipeline:
        st.info("Pipeline is empty. Add steps from the sections on the left.")
    else:
        for i, step in enumerate(st.session_state.pipeline, start=1):
            st.write(f"{i}. **{step['kind']}** — {step.get('params', {})}")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🧪 Preview Full Pipeline (on sample)"):
            prev = sample_for_preview(df)
            preview_df, msgs = run_pipeline(prev, st.session_state.pipeline)
            st.session_state.last_preview = (preview_df, "\n".join(msgs))
            st.success("Pipeline preview complete.")
    with col2:
        if st.button("🚮 Clear Pipeline"):
            st.session_state.pipeline = []
            st.info("Cleared pipeline.")
    with col3:
        progress_placeholder = st.empty()
        if st.button("✅ Apply Pipeline to Data"):
            if not st.session_state.pipeline:
                st.warning("Pipeline is empty.")
            else:
                push_history("Before pipeline")
                msgs = []
                tmp_df = st.session_state.df.copy()
                steps = st.session_state.pipeline.copy()
                total = len(steps)
                for i, step in enumerate(steps, start=1):
                    progress_placeholder.progress(i / total, text=f"Applying step {i}/{total}: {step['kind']}")
                    tmp_df, msg = apply_step(tmp_df, step)
                    msgs.append(msg)
                    time.sleep(0.05)
                progress_placeholder.empty()
                st.session_state.df = tmp_df
                st.session_state.changelog.extend([f"✅ {m}" for m in msgs])
                st.success("Applied pipeline to full dataset.")
                st.session_state.pipeline = []  # clear

    st.markdown("---")
    if st.session_state.last_preview is not None:
        prev_df, msg = st.session_state.last_preview
        st.subheader("Latest Preview Result")
        with st.expander("Preview Summary", expanded=True):
            st.code(msg or "", language="text")
        st.dataframe(prev_df)

        # Optional distribution comparison for a selected numeric column
        num_cols, _ = dtype_split(prev_df)
        if num_cols:
            column = st.selectbox("Preview histogram for column", num_cols)
            left, right = st.columns(2)
            with left:
                st.altair_chart(alt_histogram(sample_for_preview(st.session_state.df), column, "Current Data"), use_container_width=True)
            with right:
                st.altair_chart(alt_histogram(prev_df, column, "Preview Data"), use_container_width=True)


def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    df = st.session_state.df
    raw = st.session_state.raw_df
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
        st.dataframe(miss_after[miss_after > 0].rename("missing_count"))

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
                st.altair_chart(alt_histogram(sample_for_preview(raw), col, f"Before: {col}"), use_container_width=True)
            with b:
                st.subheader("After")
                st.altair_chart(alt_histogram(sample_for_preview(df), col, f"After: {col}"), use_container_width=True)

    with t3:
        if not st.session_state.changelog:
            st.info("No changes yet.")
        else:
            for i, msg in enumerate(st.session_state.changelog, start=1):
                st.write(f"{i}. {msg}")

    st.markdown("---")
    st.subheader("Download Processed Data")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "💾 Download CSV",
        data=buf.getvalue(),
        file_name="preprocessed_data.csv",
        mime="text/csv",
        help="Download the final processed dataset as a CSV file.",
    )
    st.caption("All processing happens in your browser session. Nothing is uploaded to external servers by this app code.")


# =========================
# Main App Entrypoint
# =========================
def main():
    section = sidebar_navigation()

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
        "Pro tip: Add multiple steps to the pipeline and apply them in one go. "
        "Use the Dashboard to understand how your dataset changed."
    )


if __name__ == "__main__":
    main()
