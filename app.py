# =========================
# Imports
# =========================
import io
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import altair as alt

# -------------------------
# Arrow-compatibility fix
# -------------------------
# Streamlit uses Apache Arrow under the hood.  When a column contains
# mixed types, Arrow refuses to serialize the DataFrame and the app
# crashes with an ArrowInvalid exception.  The following helper
# aggressively coerces every column to a type that Arrow understands.
def _arrowize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a *shallow* copy of `df` whose column dtypes are guaranteed to
    serialize to Arrow without raising ArrowInvalid.
    """
    if df is None or df.empty:
        return df

    df_out = df.copy()
    for col in df_out.columns:
        s = df_out[col]
        inferred = pd.api.types.infer_dtype(s, skipna=True)

        # CASE 1: numeric column that slipped into 'object' dtype
        # (usually because of strings such as 'NA', 'NULL', etc.)
        if inferred in ("mixed", "string", "mixed-integer"):
            # Try to parse as float; unparseable values become NaN
            parsed = pd.to_numeric(s, errors="coerce")
            # If at least one value survived, cast the column
            if parsed.notna().any():
                df_out[col] = parsed.astype("float64")
            else:
                # Fallback: plain string
                df_out[col] = s.astype(str)

        # CASE 2: Boolean columns might be "True"/"False" strings
        elif inferred == "boolean":
            df_out[col] = s.astype(bool)

        # CASE 3: Everything else is kept as-is
        else:
            pass

    return df_out


# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Safely sample DataFrame for preview purposes."""
    try:
        if df is None or df.empty:
            return df
        if len(df) <= n:
            return df.copy()
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
    except Exception as e:
        logger.error(f"Error in sample_for_preview: {e}")
        return df if df is not None else pd.DataFrame()

def push_history(label: str):
    """Save a snapshot for undo, with a helpful label."""
    try:
        if st.session_state.df is not None:
            st.session_state.history.append((label, st.session_state.df.copy()))
    except Exception as e:
        logger.error(f"Error pushing history: {e}")
        st.error(f"Failed to save history snapshot: {e}")

def undo_last():
    """Undo the last applied step by restoring the previous snapshot."""
    try:
        if st.session_state.history:
            label, df_prev = st.session_state.history.pop()
            st.session_state.df = df_prev
            st.session_state.changelog.append(f"↩️ Undo: {label}")
            st.success(f"Undid: {label}")
        else:
            st.info("History is empty. Nothing to undo.")
    except Exception as e:
        logger.error(f"Error in undo_last: {e}")
        st.error(f"Failed to undo: {e}")

def reset_all():
    """Clear everything and start fresh."""
    try:
        st.session_state.raw_df = None
        st.session_state.df = None
        st.session_state.history = []
        st.session_state.pipeline = []
        st.session_state.changelog = []
        st.session_state.last_preview = None
        st.success("Reset all data and pipeline.")
    except Exception as e:
        logger.error(f"Error in reset_all: {e}")
        st.error(f"Failed to reset: {e}")

def dtype_split(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_columns, categorical_columns)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols

@st.cache_data
def compute_basic_stats(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Compute comprehensive basic statistics for a DataFrame."""
    if df is None:
        logger.warning("DataFrame is None, returning empty statistics")
        return {
            "shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "missing_total": 0,
            "missing_by_col": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "describe_numeric": {},
            "memory_usage_mb": 0.0,
            "duplicate_rows": 0
        }

    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected pandas DataFrame or None, got {type(df)}")
        st.error(f"Invalid data type: expected DataFrame, got {type(df)}")
        return {}

    try:
        num_cols, cat_cols = dtype_split(df)
        missing_series = df.isna().sum()
        missing_total = int(missing_series.sum())
        missing_by_col = {k: int(v) for k, v in missing_series.sort_values(ascending=False).to_dict().items()}

        describe_numeric = {}
        if num_cols:
            numeric_df = df[num_cols]
            if len(numeric_df) > 1_000_000:
                numeric_df = numeric_df.sample(n=min(100_000, len(numeric_df)), random_state=42)
            describe_numeric = numeric_df.describe().to_dict()
            for col in describe_numeric:
                for stat in describe_numeric[col]:
                    val = describe_numeric[col][stat]
                    if pd.isna(val):
                        describe_numeric[col][stat] = None
                    elif isinstance(val, (np.integer, np.floating)):
                        describe_numeric[col][stat] = float(val) if np.isfinite(val) else None

        memory_usage_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
        duplicate_rows = int(df.duplicated().sum())

        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_total": missing_total,
            "missing_by_col": missing_by_col,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "describe_numeric": describe_numeric,
            "memory_usage_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows
        }
    except Exception as e:
        logger.error(f"Unexpected error in compute_basic_stats: {e}")
        st.error(f"Error computing statistics: {e}")
        return {
            "shape": (0, 0),
            "columns": [],
            "dtypes": {},
            "missing_total": 0,
            "missing_by_col": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "describe_numeric": {},
            "memory_usage_mb": 0.0,
            "duplicate_rows": 0,
            "error": str(e)
        }

def compare_stats(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare statistics between two DataFrames (before and after processing)."""
    if before is None:
        before = {}
    if after is None:
        after = {}
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}

    try:
        shape_before = before.get("shape", (0, 0))
        shape_after = after.get("shape", (0, 0))
        missing_before = before.get("missing_total", 0)
        missing_after = after.get("missing_total", 0)
        cols_before = before.get("columns", [])
        cols_after = after.get("columns", [])

        rows_change = shape_after[0] - shape_before[0]
        cols_change = shape_after[1] - shape_before[1]
        missing_change = missing_after - missing_before

        rows_pct_change = (rows_change / shape_before[0] * 100) if shape_before[0] > 0 else 0.0
        missing_pct_change = (missing_change / missing_before * 100) if missing_before > 0 else 0.0

        set_before = set(cols_before)
        set_after = set(cols_after)
        added_columns = list(set_after - set_before)
        removed_columns = list(set_before - set_after)

        return {
            "shape_before": shape_before,
            "shape_after": shape_after,
            "rows_change": rows_change,
            "rows_pct_change": round(rows_pct_change, 2),
            "columns_change": cols_change,
            "missing_total_before": int(missing_before),
            "missing_total_after": int(missing_after),
            "missing_change": int(missing_change),
            "missing_pct_change": round(missing_pct_change, 2),
            "n_columns_before": len(cols_before),
            "n_columns_after": len(cols_after),
            "added_columns": added_columns,
            "removed_columns": removed_columns,
        }
    except Exception as e:
        logger.error(f"Error in compare_stats: {e}")
        st.error(f"Error comparing statistics: {e}")
        return {
            "shape_before": (0, 0),
            "shape_after": (0, 0),
            "rows_change": 0,
            "rows_pct_change": 0.0,
            "columns_change": 0,
            "missing_total_before": 0,
            "missing_total_after": 0,
            "missing_change": 0,
            "missing_pct_change": 0.0,
            "n_columns_before": 0,
            "n_columns_after": 0,
            "added_columns": [],
            "removed_columns": [],
            "error": str(e)
        }

def alt_histogram(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    max_bins: int = 40,
    height: int = 250,
    width: int = 400
) -> Optional[alt.Chart]:
    """Create a robust Altair histogram with comprehensive error handling."""
    try:
        if not isinstance(df, pd.DataFrame):
            logger.error("df must be a pandas DataFrame")
            st.error("Invalid data type for histogram")
            return None
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            st.error(f"Column '{column}' not found in data")
            return None
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.error(f"Column '{column}' is not numeric")
            st.error(f"Column '{column}' is not numeric")
            return None

        clean_df = df[[column]].dropna()
        if clean_df.empty:
            st.warning(f"Column '{column}' contains only missing values")
            return None

        if clean_df[column].nunique() == 1:
            value = clean_df[column].iloc[0]
            single_df = pd.DataFrame({column: [value], 'count': [len(clean_df)]})
            chart = (
                alt.Chart(single_df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{column}:Q", title=column),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=[alt.Tooltip(f"{column}:Q", title=column), alt.Tooltip("count:Q", title="Count")]
                )
                .properties(height=height, width=width, title=title or f"Distribution of {column} (Single Value)")
            )
            return chart

        q1, q99 = clean_df[column].quantile([0.01, 0.99])
        filtered_df = clean_df[(clean_df[column] >= q1) & (clean_df[column] <= q99)].copy()
        if filtered_df.empty:
            filtered_df = clean_df.copy()

        n_unique = filtered_df[column].nunique()
        actual_bins = min(max_bins, max(10, min(n_unique, int(np.sqrt(len(filtered_df))))))
        title = title or f"Distribution of {column}"

        chart = (
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.7, stroke='white', strokeWidth=0.5)
            .encode(
                x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=actual_bins), title=column),
                y=alt.Y("count()", title="Count"),
                tooltip=[
                    alt.Tooltip(f"{column}:Q", title=column, format='.3f'),
                    alt.Tooltip("count()", title="Count")
                ],
                color=alt.value('steelblue')
            )
            .properties(height=height, width=width, title=alt.TitleParams(text=title, fontSize=14, anchor='start'))
            .interactive()
        )
        return chart
    except Exception as e:
        logger.error(f"Error creating histogram for column '{column}': {e}")
        st.error(f"Failed to create histogram: {e}")
        return None

# =========================
# Pre-processing Step Helpers
# =========================
# 1) Missing Data
def impute_missing(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    constant_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    try:
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
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
            msg = f"Imputed {len(target_cols)} columns using mode."
        elif strategy == "constant":
            val = constant_value if constant_value is not None else 0
            df[columns] = df[columns].fillna(val)
            msg = f"Imputed {len(columns)} columns with constant value {val}."
        else:
            msg = "No imputation performed (unknown strategy)."
        return df, msg
    except Exception as e:
        logger.error(f"Error in impute_missing: {e}")
        return df, f"Error in imputation: {e}"

def drop_missing(
    df: pd.DataFrame,
    axis: str = "rows",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    try:
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
                to_drop = [c for c in df.columns if df[c].isna().any()]
                df = df.drop(columns=to_drop)
                msg = f"Dropped columns containing any missing values: {to_drop}"
        return df, msg
    except Exception as e:
        logger.error(f"Error in drop_missing: {e}")
        return df, f"Error dropping missing values: {e}"

# 2) Data Inconsistency
def normalize_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    trim: bool = True,
    collapse_spaces: bool = True,
) -> Tuple[pd.DataFrame, str]:
    try:
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
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        return df, f"Error normalizing text: {e}"

def standardize_dates(
    df: pd.DataFrame, columns: List[str], output_format: str = "%Y-%m-%d"
) -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        for c in columns:
            dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = dt.dt.strftime(output_format)
        return df, f"Standardized date format to {output_format} for columns: {columns}"
    except Exception as e:
        logger.error(f"Error in standardize_dates: {e}")
        return df, f"Error standardizing dates: {e}"

def unit_convert(
    df: pd.DataFrame, column: str, factor: float, new_name: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    try:
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
    except Exception as e:
        logger.error(f"Error in unit_convert: {e}")
        return df, f"Error in unit conversion: {e}"

# 3 & 7) Outliers / Noisy Data
def detect_outliers_mask(
    df: pd.DataFrame, columns: List[str], method: str = "IQR", z_thresh: float = 3.0, iqr_k: float = 1.5
) -> pd.Series:
    try:
        mask = pd.Series(False, index=df.index)
        for c in columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x = df[c]
            if method == "IQR":
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                mask_c = (x < lower) | (x > upper)
            else:  # Z-score
                mu, sigma = x.mean(), x.std(ddof=0)
                if sigma == 0 or np.isnan(sigma):
                    mask_c = pd.Series(False, index=x.index)
                else:
                    z = (x - mu) / sigma
                    mask_c = z.abs() > z_thresh
            mask = mask | mask_c
        return mask
    except Exception as e:
        logger.error(f"Error in detect_outliers_mask: {e}")
        return pd.Series(False, index=df.index)

def handle_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str,  # 'remove' | 'cap' | 'log1p'
    detect_method: str = "IQR",
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
) -> Tuple[pd.DataFrame, str]:
    try:
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
                    q1, q3 = df[c].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                else:
                    mu, sigma = df[c].mean(), df[c].std(ddof=0)
                    if sigma == 0 or np.isnan(sigma):
                        continue
                    lower, upper = mu - z_thresh * sigma, mu + z_thresh * sigma
                df[c] = df[c].clip(lower=lower, upper=upper)
            msg = f"Capped outliers using {detect_method} thresholds in columns: {columns}."
        else:  # 'log1p'
            for c in columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    min_val = df[c].min()
                    shift = 1 - min_val if min_val <= 0 else 0
                    df[c] = np.log1p(df[c] + shift)
            msg = f"Applied log1p transform to columns: {columns}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in handle_outliers: {e}")
        return df, f"Error handling outliers: {e}"

# 4) Duplicates
def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]], keep: str = "first") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        initial = len(df)
        df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
        removed = initial - len(df)
        msg = f"Removed {removed} duplicate rows (keep={keep}) using columns: {subset if subset else 'ALL'}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df, f"Error removing duplicates: {e}"

# 5) Categorical Encoding
def encode_categorical(
    df: pd.DataFrame, columns: List[str], method: str = "onehot"
) -> Tuple[pd.DataFrame, str]:
    try:
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
    except Exception as e:
        logger.error(f"Error in encode_categorical: {e}")
        return df, f"Error encoding categorical data: {e}"

# 6) Scaling / Normalization
def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, str]:
    try:
        df = df.copy()
        cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return df, "No numeric columns selected for scaling."
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols].values)
        msg = f"Applied {'StandardScaler' if method == 'standard' else 'MinMaxScaler'} to columns: {cols}."
        return df, msg
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        return df, f"Error scaling features: {e}"

# 8) Imbalanced Data (basic random over/under sampling)
def rebalance_dataset(
    df: pd.DataFrame, target: str, method: str = "oversample", ratio: float = 1.0, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, str]:
    try:
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
                    add = subset.sample(n=desired - cnt, replace=True, random_state=random_state)
                    dfs.append(pd.concat([subset, add]))
                else:
                    dfs.append(subset)
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
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
            df_bal = pd.concat(dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            msg = f"Undersampled majority classes to ~{desired} rows each (ratio={ratio})."
        return df_bal, msg
    except Exception as e:
        logger.error(f"Error in rebalance_dataset: {e}")
        return df, f"Error rebalancing dataset: {e}"

# =========================
# Pipeline Runner
# =========================
def apply_step(df: pd.DataFrame, step: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    try:
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
    except Exception as e:
        logger.error(f"Error applying step {step.get('kind', 'unknown')}: {e}")
        return df, f"Error in step {step.get('kind', 'unknown')}: {e}"

def run_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    try:
        msgs = []
        for idx, step in enumerate(pipeline, start=1):
            df, msg = apply_step(df, step)
            msgs.append(f"{idx}. {msg}")
        return df, msgs
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return df, [f"Pipeline error: {e}"]

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
        st.rerun()
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

    file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV only. For large files, previews are sampled.",
    )

    if file:
        try:
            # 1. Load the file
            df = pd.read_csv(file)

            # 2. Reset all session state
            st.session_state.raw_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.history = []
            st.session_state.pipeline = []
            st.session_state.changelog = ["📥 Loaded dataset."]
            st.session_state.last_preview = None

            # 3. Notify the user
            st.success(f"Loaded dataset with shape {df.shape}.")

            # 4. Display a preview
            with st.expander("Peek at data", expanded=True):
                st.dataframe(_arrowize(sample_for_preview(df)))

        except pd.errors.EmptyDataError:
            logger.error("Empty CSV file.")
            st.error("The uploaded file is empty. Please choose a non-empty CSV.")
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing failed: {e}")
            st.error(f"Could not parse CSV file. Check for formatting issues.\n\n{e}")
        except Exception as e:
            logger.error(f"Unexpected error while reading CSV: {e}")
            st.error(f"Unexpected error: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

# =========================
# UI Sections (continued)
# =========================

def section_missing_data():
    st.header("1) Handling Missing Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                    st.dataframe(_arrowize(preview_df))
            with col2:
                if st.button("📦 Add to Pipeline (Impute)"):
                    step = {"kind": "impute", "params": {"columns": cols, "strategy": strategy, "constant_value": const_val}}
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")

        with st.expander("Drop options"):
            axis = st.radio("Drop axis", ["rows", "columns"], horizontal=True)
            threshold = st.slider("Missingness threshold (ratio)", 0.0, 1.0, 0.5, 0.05)
            subset = []
            if axis == "rows":
                subset = st.multiselect("For rows: require non-missing in selected columns (optional)", df.columns.tolist())
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔍 Preview Drop"):
                    prev = sample_for_preview(df)
                    preview_df, msg = drop_missing(prev, axis=axis, threshold=threshold, columns=subset or None)
                    st.session_state.last_preview = (preview_df, msg)
                    st.info(msg)
                    st.dataframe(_arrowize(preview_df))
            with col4:
                if st.button("📦 Add to Pipeline (Drop Missing)"):
                    step = {
                        "kind": "drop_missing",
                        "params": {"axis": axis, "threshold": threshold, "columns": subset or None},
                    }
                    st.session_state.pipeline.append(step)
                    st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_missing_data: {e}")
        st.error(f"Error in missing data section: {e}")

def section_inconsistency():
    st.header("2) Data Inconsistency")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                st.dataframe(_arrowize(preview_df))
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
        fmt = st.text_input("Output date format", "%Y-%m-%d", help="Python datetime format string.")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("🔍 Preview Date Standardization"):
                prev = sample_for_preview(df)
                preview_df, msg = standardize_dates(prev, date_cols, fmt)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with dc2:
            if st.button("📦 Add to Pipeline (Standardize Dates)"):
                step = {"kind": "standardize_dates", "params": {"columns": date_cols, "output_format": fmt}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")

        st.subheader("Unit Conversion")
        uc_col = st.selectbox("Column to convert", ["(none)"] + df.columns.tolist())
        colA, colB, colC = st.columns(3)
        with colA:
            factor = st.number_input("Multiply by factor", value=1.0, step=0.1)
        with colB:
            new_name = st.text_input("New column name (optional)", "")
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
                st.dataframe(_arrowize(preview_df))
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
    except Exception as e:
        logger.error(f"Error in section_inconsistency: {e}")
        st.error(f"Error in data inconsistency section: {e}")

def section_outliers():
    st.header("3 & 7) Outliers / Noisy Data")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        num_cols, _ = dtype_split(df)
        cols = st.multiselect("Numeric columns to check", num_cols)
        col1, col2, col3 = st.columns(3)
        with col1:
            detect_method = st.selectbox("Detection method", ["IQR", "Z-score"])
        with col2:
            zt = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)
        with col3:
            ik = st.slider("IQR k (fence multiplier)", 0.5, 5.0, 1.5, 0.1)

        act = st.selectbox("Action on outliers", ["remove", "cap", "log1p"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Outlier Handling"):
                prev = sample_for_preview(df)
                preview_df, msg = handle_outliers(prev, cols, act, detect_method, z_thresh=zt, iqr_k=ik)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Outliers)"):
                step = {
                    "kind": "outliers",
                    "params": {"columns": cols, "method": act, "detect_method": detect_method, "z_thresh": zt, "iqr_k": ik},
                }
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_outliers: {e}")
        st.error(f"Error in outliers section: {e}")

def section_duplicates():
    st.header("4) Data Duplication")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
        subset = st.multiselect("Columns to consider duplicates on (leave empty for all columns)", df.columns.tolist())
        keep = st.selectbox("Keep", ["first", "last", "False"], help="If 'False', drop all duplicates.")
        karg = False if keep == "False" else keep
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Preview Duplicate Removal"):
                prev = sample_for_preview(df)
                preview_df, msg = remove_duplicates(prev, subset or None, keep=karg)
                st.session_state.last_preview = (preview_df, msg)
                st.info(msg)
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Duplicates)"):
                step = {"kind": "duplicates", "params": {"subset": subset or None, "keep": karg}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_duplicates: {e}")
        st.error(f"Error in duplicates section: {e}")

def section_encoding():
    st.header("5) Categorical Data Handling")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Encoding)"):
                step = {"kind": "encode", "params": {"columns": cols, "method": method}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_encoding: {e}")
        st.error(f"Error in encoding section: {e}")

def section_scaling():
    st.header("6) Feature Scaling & Normalization")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                st.dataframe(_arrowize(preview_df))
        with c2:
            if st.button("📦 Add to Pipeline (Scaling)"):
                step = {"kind": "scale", "params": {"columns": cols, "method": method}}
                st.session_state.pipeline.append(step)
                st.success("Added to pipeline.")
    except Exception as e:
        logger.error(f"Error in section_scaling: {e}")
        st.error(f"Error in scaling section: {e}")

def section_imbalanced():
    st.header("8) Imbalanced Data (Classification)")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                st.dataframe(_arrowize(preview_df))
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
    except Exception as e:
        logger.error(f"Error in section_imbalanced: {e}")
        st.error(f"Error in imbalanced data section: {e}")

def section_pipeline_preview():
    st.header("9) Pipeline & Preview")
    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    try:
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
                    st.session_state.pipeline = []

        st.markdown("---")
        if st.session_state.last_preview is not None:
            prev_df, msg = st.session_state.last_preview
            st.subheader("Latest Preview Result")
            with st.expander("Preview Summary", expanded=True):
                st.code(msg or "", language="text")
            st.dataframe(_arrowize(prev_df))

            num_cols, _ = dtype_split(prev_df)
            if num_cols:
                column = st.selectbox("Preview histogram for column", num_cols)
                left, right = st.columns(2)
                with left:
                    chart1 = alt_histogram(sample_for_preview(st.session_state.df), column, "Current Data")
                    if chart1:
                        st.altair_chart(chart1, use_container_width=True)
                with right:
                    chart2 = alt_histogram(prev_df, column, "Preview Data")
                    if chart2:
                        st.altair_chart(chart2, use_container_width=True)
    except Exception as e:
        logger.error(f"Error in section_pipeline_preview: {e}")
        st.error(f"Error in pipeline preview section: {e}")

def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    df = st.session_state.df
    raw = st.session_state.raw_df
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        before_stats = compute_basic_stats(raw)
        after_stats = compute_basic_stats(df)
        comp = compare_stats(before_stats, after_stats)

        t1, t2, t3 = st.tabs(["Summary", "Distributions", "Change Log"])
        with t1:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows", f"{after_stats['shape'][0]}", f"{comp['rows_change']}")
            with c2:
                st.metric("Columns", f"{after_stats['shape'][1]}", f"{comp['columns_change']}")
            with c3:
                st.metric("Missing Values", f"{comp['missing_total_after']}", f"{comp['missing_change']}")
            with c4:
                st.metric("Memory (MB)", f"{after_stats.get('memory_usage_mb', 0):.2f}")

            if comp.get('added_columns'):
                st.success(f"Added columns: {', '.join(comp['added_columns'])}")
            if comp.get('removed_columns'):
                st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")

            st.subheader("Missing by Column (After)")
            miss_after = pd.Series(after_stats["missing_by_col"])
            if miss_after.sum() > 0:
                st.dataframe(_arrowize(miss_after[miss_after > 0].rename("missing_count")))
            else:
                st.info("No missing values remaining!")

            with st.expander("Dtypes (After)"):
                st.json(after_stats["dtypes"])

            with st.expander("Numeric Describe (After)"):
                if after_stats["numeric_cols"]:
                    st.dataframe(_arrowize(pd.DataFrame(after_stats["describe_numeric"])))
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
                    chart1 = alt_histogram(sample_for_preview(raw), col, f"Before: {col}")
                    if chart1:
                        st.altair_chart(chart1, use_container_width=True)
                with b:
                    st.subheader("After")
                    chart2 = alt_histogram(sample_for_preview(df), col, f"After: {col}")
                    if chart2:
                        st.altair_chart(chart2, use_container_width=True)

        with t3:
            if not st.session_state.changelog:
                st.info("No changes yet.")
            else:
                for i, msg in enumerate(st.session_state.changelog, start=1):
                    st.write(f"{i}. {msg}")

        st.markdown("---")
        st.subheader("Download Processed Data")
        buf = io.StringIO()
        _arrowize(df).to_csv(buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=buf.getvalue(),
            file_name="preprocessed_data.csv",
            mime="text/csv",
            help="Download the final processed dataset as a CSV file.",
        )
        st.caption("All processing happens in your browser session. Nothing is uploaded to external servers by this app code.")
    except Exception as e:
        logger.error(f"Error in section_dashboard_download: {e}")
        st.error(f"Error in dashboard section: {e}")

# =========================
# Main App Entrypoint
# =========================
def main():
    try:
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

        st.markdown("---")
        st.caption(
            "Pro tip: Add multiple steps to the pipeline and apply them in one go. "
            "Use the Dashboard to understand how your dataset changed."
        )
    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
