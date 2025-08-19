import io, json, warnings, time, pickle
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import streamlit as st

# Optional imports (graceful)
try:
    from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except Exception:
    KNNImputer = IterativeImputer = SimpleImputer = None
    StandardScaler = MinMaxScaler = PowerTransformer = PolynomialFeatures = None
    SklearnPipeline = IsolationForest = LocalOutlierFactor = resample = None
    SKLEARN_AVAILABLE = False

try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except Exception:
    dd = None
    DASK_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except Exception:
    pl = None
    POLARS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------
# Session state helpers & init
# ---------------------------
def S(k, default=None):
    return st.session_state.get(k, default)

def setS(k, v):
    st.session_state[k] = v

def init_session():
    defaults = {
        "dps_raw_df": None,
        "dps_df": None,
        "dps_preview_df": None,
        "dps_pipeline": [],
        "dps_history": [],
        "dps_future": [],
        "dps_changelog": [],
        "dps_settings": {
            "preview_rows": 200,
            "random_state": 42,
            "atomic_apply": True,
            "max_upload_mb": 500,
            "use_dask": False
        }
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ---------------------------
# I/O helpers
# ---------------------------
def safe_read_csv(file_like, nrows=None) -> pd.DataFrame:
    try:
        file_like.seek(0)
    except Exception:
        pass
    return pd.read_csv(file_like, nrows=nrows, low_memory=False)

# ---------------------------
# Utility helpers
# ---------------------------
def push_history(label: str):
    df = S("dps_df")
    if df is None:
        return
    hist = S("dps_history")
    hist.append((label, df.copy()))
    if len(hist) > 30:
        hist.pop(0)
    setS("dps_history", hist)

def undo_last():
    hist = S("dps_history")
    if not hist:
        st.info("No history to undo.")
        return
    label, df_prev = hist.pop()
    setS("dps_df", df_prev)
    setS("dps_history", hist)
    S("dps_changelog").append(f"Undid: {label}")
    st.success(f"Undid: {label}")

def redo_last():
    fut = S("dps_future")
    if not fut:
        st.info("Nothing to redo.")
        return
    item = fut.pop()
    setS("dps_df", item["df"])
    setS("dps_pipeline", item.get("pipeline", S("dps_pipeline")))
    push_history("redo")
    st.success("Redone last change.")

def update_preview_sample():
    df = S("dps_df")
    if df is None:
        setS("dps_preview_df", None)
        return
    n = int(S("dps_settings")["preview_rows"])
    rs = int(S("dps_settings")["random_state"])
    if len(df) <= n:
        sample = df.copy()
    else:
        sample = df.sample(n=n, random_state=rs).copy()
    setS("dps_preview_df", sample)

def dtype_split(df: pd.DataFrame):
    if df is None:
        return [], []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in df.columns if c not in num_cols]
    return num_cols, other

def compute_basic_stats(df: pd.DataFrame):
    if df is None:
        return {}
    return {
        "shape": df.shape,
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().sort_values(ascending=False).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

# ---------------------------
# Auto detect & type fixes (no infer_datetime_format)
# ---------------------------
def auto_detect_types(df: pd.DataFrame, apply: bool = False, date_format: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suggestions: Dict[str, Any] = {}
    if df is None:
        return df, suggestions
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_bool_dtype(ser):
            continue
        non_null = ser.dropna().astype(str).str.strip()
        if non_null.empty:
            continue
        n_total = len(non_null)
        n_numeric = non_null.str.match(r"^[+-]?\d+(\.\d+)?$").sum()
        if n_numeric / n_total >= 0.9:
            suggestions[col] = {"action": "to_numeric", "confidence": float(n_numeric / n_total)}
            if apply:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        # parse with optional explicit format to reduce ambiguity
        try:
            parsed = pd.to_datetime(ser, errors="coerce", format=date_format) if date_format else pd.to_datetime(ser, errors="coerce")
        except Exception:
            parsed = pd.to_datetime(ser, errors="coerce")
        n_dates = parsed.notna().sum()
        if n_dates / n_total >= 0.6:
            suggestions[col] = {"action": "to_datetime", "confidence": float(n_dates / n_total)}
            if apply:
                df[col] = parsed
            continue
        uniques = non_null.unique()
        lowered = [str(x).lower() for x in uniques]
        truthy = {"true", "t", "1", "yes", "y"}
        falsy = {"false", "f", "0", "no", "n"}
        if set(lowered).issubset(truthy.union(falsy)) and len(uniques) <= 3:
            suggestions[col] = {"action": "to_bool", "values": list(uniques)}
            if apply:
                df[col] = ser.astype(str).str.lower().map(lambda x: True if x in truthy else (False if x in falsy else pd.NA)).astype("boolean")
    return df, suggestions

# ---------------------------
# Missing data / imputation
# ---------------------------
def impute_missing(df: pd.DataFrame, columns: Optional[List[str]] = None, strategy: str = "mean", constant_value: Any = None, **kwargs):
    if df is None:
        return None, "No data"
    df = df.copy()
    cols = columns if columns else df.columns.tolist()
    if strategy in ("mean", "median"):
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                val = df[c].mean() if strategy == "mean" else df[c].median()
                df[c] = df[c].fillna(val)
        return df, f"Imputed numeric columns with {strategy}"
    if strategy == "mode":
        for c in cols:
            if c in df.columns:
                m = df[c].mode(dropna=True)
                if not m.empty:
                    df[c] = df[c].fillna(m.iloc[0])
        return df, "Imputed with mode"
    if strategy == "constant":
        df[cols] = df[cols].fillna(constant_value)
        return df, f"Imputed with constant {constant_value}"
    if strategy == "knn":
        if not SKLEARN_AVAILABLE:
            return df, "KNN imputer requires scikit-learn"
        numeric = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return df, "No numeric columns for KNN imputer"
        imputer = KNNImputer(**{k: v for k, v in kwargs.items() if k in ("n_neighbors", "weights")})
        arr = imputer.fit_transform(df[numeric])
        df[numeric] = arr
        return df, "KNN imputation applied"
    if strategy == "iterative":
        if not SKLEARN_AVAILABLE:
            return df, "Iterative imputer requires scikit-learn"
        numeric = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return df, "No numeric columns for IterativeImputer"
        imputer = IterativeImputer(random_state=int(S("dps_settings")["random_state"]))
        arr = imputer.fit_transform(df[numeric])
        df[numeric] = arr
        return df, "Iterative imputation applied"
    return df, f"Unknown imputation strategy {strategy}"

# ---------------------------
# Outlier detection & handling (mask dtype explicit)
# ---------------------------
def detect_outliers_mask(df: pd.DataFrame, columns: List[str], method: str = "IQR", **kwargs) -> pd.Series:
    if df is None:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=df.index, dtype=bool)
    for c in columns:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        if method == "IQR":
            k = float(kwargs.get("k", 1.5))
            q1 = s.quantile(0.25); q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr; upper = q3 + k * iqr
            mask = mask | ((df[c] < lower) | (df[c] > upper))
        elif method == "zscore":
            thresh = float(kwargs.get("thresh", 3.0))
            mu = s.mean(); sigma = s.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                continue
            mask = mask | ((df[c] - mu).abs() > thresh * sigma)
        elif method == "isolation_forest" and SKLEARN_AVAILABLE:
            valid = df[c].dropna()
            if len(valid) < 5:
                continue
            iso = IsolationForest(random_state=int(S("dps_settings")["random_state"]), contamination=kwargs.get("contamination", "auto"))
            preds = iso.fit_predict(valid.values.reshape(-1, 1))
            out_idx = valid.index[preds == -1]
            mask.loc[out_idx] = True
        elif method == "lof" and SKLEARN_AVAILABLE:
            valid = df[c].dropna()
            if len(valid) < 5:
                continue
            lof = LocalOutlierFactor(n_neighbors=int(kwargs.get("n_neighbors", 20)), contamination=kwargs.get("contamination", 0.05))
            preds = lof.fit_predict(valid.values.reshape(-1, 1))
            out_idx = valid.index[preds == -1]
            mask.loc[out_idx] = True
    return mask

def handle_outliers(df: pd.DataFrame, columns: List[str], detect_method: str = "IQR", action: str = "remove", **kwargs):
    if df is None:
        return None, "No data"
    df = df.copy()
    mask = detect_outliers_mask(df, columns, method=detect_method, **kwargs)
    n = int(mask.sum())
    if n == 0:
        return df, "No outliers detected"
    if action == "remove":
        df2 = df.loc[~mask].reset_index(drop=True)
        return df2, f"Removed {n} outlier rows"
    if action == "cap":
        for c in columns:
            if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            s = df[c].dropna()
            if s.empty:
                continue
            if detect_method == "IQR":
                k = float(kwargs.get("k", 1.5))
                q1 = s.quantile(0.25); q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - k * iqr; upper = q3 + k * iqr
            else:
                thresh = float(kwargs.get("thresh", 3.0))
                mu = s.mean(); sigma = s.std(ddof=0)
                lower = mu - thresh * sigma; upper = mu + thresh * sigma
            df[c] = df[c].clip(lower=lower, upper=upper)
        return df, f"Capped outliers in columns {columns}"
    if action == "mark":
        df["_is_outlier"] = mask
        return df, f"Marked {n} outliers in _is_outlier column"
    return df, f"Unknown outlier action {action}"

# ---------------------------
# Transformations
# ---------------------------
def apply_transform(df: pd.DataFrame, columns: List[str], transform: str = "log", **kwargs):
    if df is None:
        return None, "No data"
    df = df.copy()
    cols = [c for c in columns if c in df.columns]
    if transform == "log":
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.dropna().empty:
                continue
            minv = s.min(skipna=True)
            shift = 1 - minv if minv <= 0 else 0
            df[c] = np.log1p(s + shift)
        return df, f"Applied log1p to {cols}"
    if transform == "boxcox":
        if not SKLEARN_AVAILABLE:
            return df, "Box-Cox requires scikit-learn PowerTransformer"
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return df, "No numeric columns for Box-Cox"
        arr = df[numeric].copy()
        for c in numeric:
            minv = arr[c].min(skipna=True)
            if minv <= 0:
                arr[c] = arr[c] + (1 - minv)
        arr = arr.fillna(arr.mean())
        pt = PowerTransformer(method="box-cox")
        arr_t = pt.fit_transform(arr)
        df[numeric] = arr_t
        return df, f"Applied Box-Cox to {numeric}"
    if transform == "yeo-johnson":
        if not SKLEARN_AVAILABLE:
            return df, "Yeo-Johnson requires scikit-learn PowerTransformer"
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return df, "No numeric columns for Yeo-Johnson"
        arr = df[numeric].fillna(df[numeric].mean())
        pt = PowerTransformer(method="yeo-johnson")
        df[numeric] = pt.fit_transform(arr)
        return df, f"Applied Yeo-Johnson to {numeric}"
    if transform == "polynomial":
        if not SKLEARN_AVAILABLE:
            return df, "Polynomial features require scikit-learn"
        degree = int(kwargs.get("degree", 2))
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return df, "No numeric columns for polynomial"
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        arr = df[numeric].fillna(0).values
        transformed = poly.fit_transform(arr)
        new_cols = poly.get_feature_names_out(numeric)
        df_poly = pd.DataFrame(transformed, columns=new_cols, index=df.index)
        df = pd.concat([df.drop(columns=numeric), df_poly], axis=1)
        return df, f"Added polynomial features degree={degree} for {numeric}"
    if transform == "binning":
        bins = int(kwargs.get("bins", 5))
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[f"{c}_binned"] = pd.cut(df[c], bins=bins, labels=False)
        return df, f"Binned {cols} into {bins} bins"
    return df, f"Unknown transform {transform}"

# ---------------------------
# Encoding & scaling
# ---------------------------
def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = "onehot"):
    if df is None:
        return None, "No data"
    df = df.copy()
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df, "No columns selected"
    if method == "onehot":
        df = pd.get_dummies(df, columns=cols, drop_first=False, dummy_na=False)
        return df, f"One-hot encoded {cols}"
    if method == "label":
        for c in cols:
            cat = pd.Categorical(df[c])
            codes = pd.Series(cat.codes, index=df.index).where(lambda x: x != -1, other=pd.NA).astype("Int64")
            df[c] = codes
        return df, f"Label-encoded (nullable Int64) {cols}"
    return df, f"Unknown encoding {method}"

def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard"):
    if df is None:
        return None, "No data"
    df = df.copy()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return df, "No numeric columns selected"
    if method == "standard":
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            arr = scaler.fit_transform(df[cols].fillna(0))
            df[cols] = arr
            return df, f"Standard scaled {cols}"
        else:
            for c in cols:
                s = pd.to_numeric(df[c], errors="coerce")
                df[c] = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
            return df, f"Standard scaled {cols} (manual fallback)"
    else:
        if SKLEARN_AVAILABLE:
            scaler = MinMaxScaler()
            arr = scaler.fit_transform(df[cols].fillna(0))
            df[cols] = arr
            return df, f"MinMax scaled {cols}"
        else:
            for c in cols:
                s = pd.to_numeric(df[c], errors="coerce")
                mn, mx = s.min(skipna=True), s.max(skipna=True)
                df[c] = (s - mn) / (mx - mn) if mx != mn else 0.0
            return df, f"MinMax scaled {cols} (manual fallback)"

# ---------------------------
# Rebalancing
# ---------------------------
def rebalance_dataset(df: pd.DataFrame, target: str, method: str = "oversample", ratio: float = 1.0):
    if df is None:
        return None, "No data"
    if target not in df.columns:
        return df, f"Target {target} not found"
    counts = df[target].value_counts(dropna=False)
    if counts.empty or len(counts) <= 1:
        return df, "Not enough classes to rebalance"
    def get_subset(cls):
        if pd.isna(cls):
            return df[df[target].isna()]
        return df[df[target] == cls]
    if method == "oversample":
        majority = counts.max()
        desired = max(1, int(round(majority * ratio)))
        parts = []
        for cls in counts.index:
            subset = get_subset(cls)
            if subset.empty:
                continue
            if len(subset) < desired:
                add = subset.sample(n=desired - len(subset), replace=True, random_state=int(S("dps_settings")["random_state"]))
                parts.append(pd.concat([subset, add], axis=0))
            else:
                parts.append(subset)
        out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=int(S("dps_settings")["random_state"])).reset_index(drop=True)
        return out, f"Oversampled classes to ~{desired} each"
    else:
        minority = counts.min()
        desired = max(1, int(round(minority * ratio)))
        parts = []
        for cls in counts.index:
            subset = get_subset(cls)
            if len(subset) > desired:
                parts.append(subset.sample(n=desired, random_state=int(S("dps_settings")["random_state"])))
            else:
                parts.append(subset)
        out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=int(S("dps_settings")["random_state"])).reset_index(drop=True)
        return out, f"Undersampled classes to ~{desired} each"

# ---------------------------
# Simple wrappers
# ---------------------------
def drop_missing_wrapper(df: pd.DataFrame, axis: str="rows", threshold: Optional[float]=None, columns: Optional[List[str]]=None):
    if df is None:
        return None, "No data"
    df = df.copy()
    if axis == "rows":
        if columns:
            cols = [c for c in columns if c in df.columns]
            initial = len(df)
            df = df.dropna(subset=cols)
            return df, f"Dropped rows missing columns {cols} ({initial - len(df)} removed)"
        if threshold is not None:
            row_ratio = df.isna().mean(axis=1)
            initial = len(df)
            df = df.loc[row_ratio < threshold].reset_index(drop=True)
            return df, f"Dropped rows with missing ratio >= {threshold} ({initial - len(df)} removed)"
        initial = len(df)
        df = df.dropna().reset_index(drop=True)
        return df, f"Dropped rows with any missing values ({initial - len(df)} removed)"
    else:
        if threshold is not None:
            col_missing = df.isna().mean(axis=0)
            to_drop = col_missing[col_missing >= threshold].index.tolist()
            df = df.drop(columns=to_drop)
            return df, f"Dropped columns with missing ratio >= {threshold}: {to_drop}"
        to_drop = [c for c in df.columns if df[c].isna().any()]
        df = df.drop(columns=to_drop)
        return df, f"Dropped columns with any missing values: {to_drop}"

def normalize_text_wrapper(df: pd.DataFrame, columns: List[str], lowercase: bool=True, trim: bool=True, collapse_spaces: bool=True):
    if df is None:
        return None, "No data"
    df = df.copy()
    applied = []
    for c in columns:
        if c not in df.columns:
            continue
        if not (pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])):
            continue
        s = df[c].astype("string")
        if trim: s = s.str.strip()
        if lowercase: s = s.str.lower()
        if collapse_spaces: s = s.str.replace(r"\s+", " ", regex=True)
        df[c] = s
        applied.append(c)
    return df, f"Normalized text for columns: {applied}"

def standardize_dates_wrapper(df: pd.DataFrame, columns: List[str], output_format: str="%Y-%m-%d"):
    if df is None:
        return None, "No data"
    df = df.copy()
    applied = []
    for c in columns:
        if c not in df.columns:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        formatted = parsed.dt.strftime(output_format).where(parsed.notna(), other=pd.NA)
        df[c] = formatted.astype("string")
        applied.append(c)
    return df, f"Standardized dates for: {applied}"

def unit_convert_wrapper(df: pd.DataFrame, column: Optional[str]=None, factor: float=1.0, new_name: Optional[str]=None):
    if df is None:
        return None, "No data"
    df = df.copy()
    if not column or column not in df.columns:
        return df, f"Column '{column}' not found"
    numeric = pd.to_numeric(df[column], errors="coerce")
    finite_mask = numeric.notna()
    out = df[column].copy()
    out = out.astype("object")
    out.loc[finite_mask] = (numeric.loc[finite_mask] * factor).astype(float)
    if new_name:
        df[new_name] = out
        return df, f"Created {new_name} = {column} * {factor}"
    df[column] = out
    return df, f"Converted {column} in place by factor {factor}"

def remove_duplicates_wrapper(df: pd.DataFrame, subset: Optional[List[str]]=None, keep='first'):
    if df is None:
        return None, "No data"
    df = df.copy()
    initial = len(df)
    if subset:
        subset = [c for c in subset if c in df.columns]
    if keep == 'none':
        df = df.drop_duplicates(subset=subset if subset else None, keep=False)
    else:
        df = df.drop_duplicates(subset=subset if subset else None, keep=keep)
    removed = initial - len(df)
    return df, f"Removed {removed} duplicates (keep={keep})"

# ---------------------------
# Pipeline apply/run
# ---------------------------
def apply_step(df: pd.DataFrame, step: Dict[str, Any]):
    kind = step.get("kind")
    params = step.get("params", {}) or {}
    if kind == "impute":
        return impute_missing(df, **params)
    if kind == "drop_missing" or kind == "dropna":
        return drop_missing_wrapper(df, **params)
    if kind == "normalize_text":
        return normalize_text_wrapper(df, **params)
    if kind == "standardize_dates":
        return standardize_dates_wrapper(df, **params)
    if kind == "unit_convert":
        return unit_convert_wrapper(df, **params)
    if kind == "outliers":
        return handle_outliers(df, **params)
    if kind == "duplicates" or kind == "dedup":
        return remove_duplicates_wrapper(df, **params)
    if kind == "encode":
        return encode_categorical(df, **params)
    if kind == "scale":
        return scale_features(df, **params)
    if kind == "rebalance":
        return rebalance_dataset(df, **params)
    if kind == "transform":
        return apply_transform(df, **params)
    if kind == "auto_types":
        df_new, suggestions = auto_detect_types(df, apply=True)
        return df_new, f"Auto-type fixes applied to {list(suggestions.keys())}"
    return df, f"Unknown step kind: {kind}"

def run_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]], atomic: bool = True):
    if df is None:
        return None, ["No data"], "No data"
    msgs = []
    working = df.copy()
    for i, step in enumerate(pipeline, start=1):
        try:
            t0 = time.perf_counter()
            working, msg = apply_step(working, step)
            dt = time.perf_counter() - t0
            msgs.append(f"{i}. {msg} (t={dt:.3f}s)")
            if working is None:
                raise RuntimeError("Step returned None")
        except Exception as e:
            err = f"Step {i} {step.get('kind')} failed: {e}"
            if atomic:
                return df, msgs, err
            msgs.append(err)
    return working, msgs, None

# ---------------------------
# sklearn export (best-effort)
# ---------------------------
def build_sklearn_pipeline(pipeline: List[Dict[str, Any]]):
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn not available; cannot export pipeline")
        return None
    steps = []
    skipped = []
    for i, step in enumerate(pipeline):
        kind = step.get("kind"); params = step.get("params", {}) or {}
        if kind == "impute":
            strat = params.get("strategy", "mean")
            if strat in ("mean","median","most_frequent"):
                steps.append((f"impute_{i}", SimpleImputer(strategy=strat)))
            elif strat == "constant":
                steps.append((f"impute_{i}", SimpleImputer(strategy="constant", fill_value=params.get("constant_value", 0))))
            elif strat == "knn":
                steps.append((f"impute_{i}", KNNImputer()))
            elif strat == "iterative":
                steps.append((f"impute_{i}", IterativeImputer(random_state=int(S("dps_settings")["random_state"]))))
            else:
                skipped.append((i, kind))
        elif kind == "scale":
            method = params.get("method","standard")
            if method == "standard":
                steps.append((f"scale_{i}", StandardScaler()))
            else:
                steps.append((f"scale_{i}", MinMaxScaler()))
        elif kind == "transform" and params.get("transform") == "polynomial":
            degree = int(params.get("degree", 2))
            steps.append((f"poly_{i}", PolynomialFeatures(degree=degree, include_bias=False)))
        else:
            skipped.append((i, kind))
    if skipped:
        st.warning(f"Skipped {len(skipped)} pipeline steps when building sklearn pipeline: {skipped}")
    if not steps:
        st.info("No sklearn-compatible steps found")
        return None
    skpipe = SklearnPipeline(steps)
    return skpipe

# ---------------------------
# UI: sidebar navigation + settings
# ---------------------------
def sidebar_navigation_and_settings():
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Section", [
        "Upload", "Builder", "Missing Data", "Data Inconsistency", "Outliers",
        "Duplicates", "Categorical Encoding", "Scaling", "Imbalanced Data",
        "Transforms", "Pipeline & Preview", "Dashboard & Download"
    ])
    st.sidebar.markdown("---")
    if st.sidebar.button("Undo Last"):
        undo_last()
    if st.sidebar.button("Reset All"):
        for k in ["dps_raw_df","dps_df","dps_preview_df","dps_pipeline","dps_history","dps_changelog"]:
            setS(k, None if "df" in k or k=="dps_preview_df" else [])
        st.experimental_rerun()
    with st.sidebar.expander("Settings"):
        s = S("dps_settings")
        s["preview_rows"] = int(st.number_input("Preview rows", min_value=10, max_value=50000, value=int(s.get("preview_rows",200))))
        s["random_state"] = int(st.number_input("Random seed", min_value=0, max_value=100000, value=int(s.get("random_state",42))))
        s["atomic_apply"] = st.checkbox("Atomic apply (rollback on error)", value=bool(s.get("atomic_apply", True)))
        s["max_upload_mb"] = int(st.number_input("Max upload MB", min_value=1, max_value=10000, value=int(s.get("max_upload_mb",500))))
        if DASK_AVAILABLE or POLARS_AVAILABLE:
            s["use_dask"] = st.checkbox("Enable big-data mode (Dask/Polars)", value=s.get("use_dask", False))
        setS("dps_settings", s)
    return section

# ---------------------------
# UI sections
# ---------------------------

def section_upload():
    st.header("Upload / Import Data")
    col1, col2 = st.columns([3,1])
    with col1:
        files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
        if files:
            if len(files) == 1:
                f = files[0]
                size_mb = getattr(f, "size", 0) / (1024*1024)
                if size_mb > S("dps_settings")["max_upload_mb"]:
                    st.error(f"File too large: {size_mb:.1f} MB (limit {S('dps_settings')['max_upload_mb']} MB).")
                else:
                    try:
                        df = safe_read_csv(f)
                        setS("dps_raw_df", df); setS("dps_df", df.copy()); setS("dps_pipeline", []); setS("dps_history", []); setS("dps_changelog", ["Loaded dataset."])
                        update_preview_sample()
                        st.success(f"Loaded {f.name} with shape {df.shape}")
                        with st.expander("Peek at data", expanded=True):
                            st.dataframe(S("dps_preview_df") if S("dps_preview_df") is not None else S("dps_df").head(100))
                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")
            else:
                st.info(f"{len(files)} files uploaded. Use the Builder to combine them or upload one at a time.")
    with col2:
        st.subheader("Quick Tools")
        date_format = st.text_input("Optional date format (e.g. %Y-%m-%d)", key="upload_date_format")
        # preview for date_format
        if date_format and S("dps_df") is not None:
            try:
                df = S("dps_df")
                # Try to parse only candidate columns: those that are object and not numeric
                candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                preview_df = df[candidates].head(10).copy()
                for c in preview_df.columns:
                    preview_df[c] = pd.to_datetime(preview_df[c], errors="coerce", format=date_format)
                st.write("Preview parsing using provided format:")
                st.dataframe(preview_df)
            except Exception as e:
                st.warning(f"Date format preview failed: {e}")
        if st.button("Auto-detect column types (suggest)"):
            df = S("dps_df")
            if df is None:
                st.warning("Load data first")
            else:
                df0, suggestions = auto_detect_types(df, apply=False, date_format=(date_format or None))
                st.json(suggestions)
        if st.button("Auto-detect and fix types (apply)"):
            df = S("dps_df")
            if df is None:
                st.warning("Load data first")
            else:
                df_new, suggestions = auto_detect_types(df, apply=True, date_format=(date_format or None))
                setS("dps_df", df_new)
                push_history("Auto-type fixes")
                S("dps_changelog").append(f"Auto type fixes: {list(suggestions.keys())}")
                update_preview_sample()
                st.success("Applied auto-type fixes")

def section_builder():
    st.header("Builder — Add pipeline step (with previews)")
    df = S("dps_df")
    if df is None:
        st.info("Upload data first.")
        return
    cols = df.columns.tolist()
    kind = st.selectbox("Step type", [
        "drop_missing", "impute", "normalize_text", "standardize_dates",
        "unit_convert", "duplicates", "outliers", "transform",
        "encode", "scale", "rebalance", "auto_types"
    ])
    params = {}
    if kind in ("drop_missing","impute","normalize_text","standardize_dates","outliers","transform","encode","scale"):
        params["columns"] = st.multiselect("Columns", cols)
    if kind == "drop_missing":
        params["axis"] = st.selectbox("Axis", ["rows","columns"])
        if params["axis"] == "rows":
            params["threshold"] = None
        else:
            params["threshold"] = st.number_input("Drop column if missing ratio >= ", min_value=0.0, max_value=1.0, value=0.5)
    elif kind == "impute":
        params["strategy"] = st.selectbox("Strategy", ["mean","median","mode","constant","knn","iterative"])
        if params["strategy"] == "constant":
            params["constant_value"] = st.text_input("Constant value", "0")
        if st.button("Preview imputation"):
            sample = S("dps_preview_df") or S("dps_df").head(50)
            res, msg = impute_missing(sample.copy(), columns=params.get("columns"), strategy=params["strategy"], constant_value=params.get("constant_value"))
            st.info(msg); st.dataframe(res.head())
    elif kind == "normalize_text":
        params["lowercase"] = st.checkbox("Lowercase", True)
        params["trim"] = st.checkbox("Trim", True)
        params["collapse_spaces"] = st.checkbox("Collapse spaces", True)
        if st.button("Preview normalization"):
            sample = S("dps_preview_df") or S("dps_df").head(50)
            res, msg = normalize_text_wrapper(sample.copy(), params.get("columns", []), params["lowercase"], params["trim"], params["collapse_spaces"])
            st.info(msg); st.dataframe(res.head())
    elif kind == "standardize_dates":
        params["fmt"] = st.text_input("Date format for standardization", "%Y-%m-%d")
        if params.get("columns"):
            if st.button("Preview date standardization"):
                sample = S("dps_preview_df") or S("dps_df").head(50)
                try:
                    res, msg = standardize_dates_wrapper(sample.copy(), params["columns"], output_format=params["fmt"])
                    st.info(msg); st.dataframe(res[params["columns"]].head())
                except Exception as e:
                    st.warning(f"Preview failed: {e}")
    elif kind == "unit_convert":
        params["column"] = st.selectbox("Column", ["(none)"] + cols)
        params["factor"] = st.number_input("Factor", value=1.0)
        params["new_name"] = st.text_input("New column name (optional)")
        if st.button("Preview unit conversion"):
            if params["column"] and params["column"] != "(none)":
                sample = S("dps_preview_df") or S("dps_df").head(50)
                res, msg = unit_convert_wrapper(sample.copy(), params["column"], params["factor"], params["new_name"] or None)
                st.info(msg); st.dataframe(res.head())
    elif kind == "duplicates":
        params["subset"] = st.multiselect("Subset columns (empty = all)", cols)
        params["keep"] = st.selectbox("Keep option", ["first","last","none"])
        if st.button("Preview dedupe"):
            sample = S("dps_preview_df") or S("dps_df").head(200)
            res, msg = remove_duplicates_wrapper(sample.copy(), subset=params.get("subset") or None, keep=params["keep"])
            st.info(msg); st.dataframe(res.head())
    elif kind == "outliers":
        params["method"] = st.selectbox("Method", ["IQR","zscore","isolation_forest","lof"])
        params["action"] = st.selectbox("Action", ["remove","cap","mark"])
        params["threshold"] = st.number_input("Threshold / factor", value=1.5 if params.get("method","IQR")=="IQR" else 3.0)
        if st.button("Preview outliers"):
            sample = S("dps_preview_df") or S("dps_df").head(200)
            mask = detect_outliers_mask(sample.copy(), params.get("columns", []), method=params["method"].lower() if params["method"] else "IQR", k=params["threshold"], thresh=params["threshold"])
            st.write("Outlier mask (True = outlier):")
            st.dataframe(mask.head())
    elif kind == "transform":
        params["transform"] = st.selectbox("Transform", ["log","boxcox","yeojohnson","polynomial","binning"])
        if params["transform"] == "polynomial":
            params["degree"] = st.number_input("Degree", min_value=2, value=2)
        if params["transform"] == "binning":
            params["bins"] = st.number_input("Bins", min_value=2, value=5)
        if st.button("Preview transform"):
            sample = S("dps_preview_df") or S("dps_df").head(50)
            try:
                res, msg = apply_transform(sample.copy(), params.get("columns", []), transform=params["transform"], degree=params.get("degree",2), bins=params.get("bins",5))
                st.info(msg); st.dataframe(res.head())
            except Exception as e:
                st.warning(f"Preview failed: {e}")
    elif kind == "encode":
        params["method"] = st.selectbox("Method", ["onehot","label"])
        if st.button("Preview encoding"):
            sample = S("dps_preview_df") or S("dps_df").head(50)
            try:
                res, msg = encode_categorical(sample.copy(), params.get("columns", []), method=params["method"])
                st.info(msg); st.dataframe(res.head())
            except Exception as e:
                st.warning(f"Preview failed: {e}")
    elif kind == "scale":
        params["method"] = st.selectbox("Method", ["standard","minmax"])
        if st.button("Preview scaling"):
            sample = S("dps_preview_df") or S("dps_df").head(50)
            try:
                res, msg = scale_features(sample.copy(), params.get("columns", []), method=params["method"])
                st.info(msg); st.dataframe(res[params.get("columns", [])].head())
            except Exception as e:
                st.warning(f"Preview failed: {e}")
    elif kind == "rebalance":
        params["target"] = st.selectbox("Target column", cols)
        params["method"] = st.selectbox("Method", ["oversample","undersample"])
        params["ratio"] = st.number_input("Ratio", min_value=0.0, value=1.0)
        if st.button("Preview rebalance"):
            sample = S("dps_preview_df") or S("dps_df").head(200)
            try:
                res, msg = rebalance_dataset(sample.copy(), params["target"], method=params["method"], ratio=params["ratio"])
                st.info(msg); st.dataframe(res[params["target"]].value_counts())
            except Exception as e:
                st.warning(f"Preview failed: {e}")
    elif kind == "auto_types":
        if st.button("Preview auto type suggestions"):
            sample = S("dps_preview_df") or S("dps_df").head(200)
            _, suggestions = auto_detect_types(sample.copy(), apply=False)
            st.json(suggestions)

    if st.button("Add step to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind": kind, "params": params})
        setS("dps_pipeline", pipe)
        push_history(f"add {kind}")
        st.success(f"Added: {kind}")

def section_missing_data():
    st.header("Missing Data — Quick Actions")
    df = S("dps_df")
    if df is None:
        st.warning("Upload a dataset first.")
        return
    cols = st.multiselect("Columns (leave empty = all)", df.columns.tolist())
    strat = st.selectbox("Strategy", ["mean","median","mode","constant","knn","iterative"])
    const = None
    if strat == "constant":
        const = st.text_input("Constant value", "0")
    if st.button("Preview imputation on sample"):
        sample = S("dps_preview_df") or S("dps_df").head(200)
        res, msg = impute_missing(sample.copy(), columns=cols if cols else None, strategy=strat, constant_value=const)
        st.info(msg); st.dataframe(res.head())
    if st.button("Add imputation to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind":"impute","params":{"columns": cols if cols else None, "strategy": strat, "constant_value": const}})
        setS("dps_pipeline", pipe)
        push_history("add impute")
        st.success("Imputation step added")

def section_inconsistency():
    st.header("Data Inconsistency")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    st.subheader("Auto-detect types")
    date_format = st.text_input("If you know the date format, enter it here (Builder and Upload also accept).", key="incons_date_fmt")
    if st.button("Suggest conversions"):
        _, suggestions = auto_detect_types(df, apply=False, date_format=(date_format or None))
        st.json(suggestions)
    if st.button("Apply suggested type fixes"):
        df_new, suggestions = auto_detect_types(df, apply=True, date_format=(date_format or None))
        setS("dps_df", df_new); push_history("Auto-type fixes"); S("dps_changelog").append(f"Auto type fixes: {list(suggestions.keys())}")
        update_preview_sample(); st.success("Applied type conversions")

def section_outliers():
    st.header("Outliers")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    num_cols, _ = dtype_split(df)
    cols = st.multiselect("Numeric columns", num_cols)
    method = st.selectbox("Method", ["IQR","zscore","isolation_forest","lof"])
    action = st.selectbox("Action", ["remove","cap","mark"])
    threshold = st.number_input("Threshold / factor", value=1.5 if method=="IQR" else 3.0)
    if st.button("Preview outlier mask (sample)"):
        sample = S("dps_preview_df") or S("dps_df").head(200)
        mask = detect_outliers_mask(sample.copy(), cols, method=method.lower(), k=threshold, thresh=threshold)
        st.write("Mask (True = outlier):")
        st.dataframe(mask.head())
    if st.button("Add outlier handling to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind": "outliers", "params": {"columns": cols, "detect_method": method.lower(), "action": action, "k": threshold, "thresh": threshold}})
        setS("dps_pipeline", pipe)
        push_history("add outliers")
        st.success("Added outlier step")

def section_duplicates():
    st.header("Duplicates")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    subset = st.multiselect("Subset columns (leave empty = all columns)", df.columns.tolist())
    keep = st.selectbox("Keep", ["first","last","none"])
    if st.button("Preview remove duplicates"):
        sample = S("dps_preview_df") or S("dps_df").head(200)
        res, msg = remove_duplicates_wrapper(sample.copy(), subset=subset if subset else None, keep=keep)
        st.info(msg); st.dataframe(res.head())
    if st.button("Add dedupe to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind":"duplicates","params":{"subset": subset if subset else None, "keep": keep}})
        setS("dps_pipeline", pipe)
        push_history("add dedupe")
        st.success("Dedup step added")

def section_encoding():
    st.header("Categorical Encoding")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    cols = st.multiselect("Columns to encode", df.columns.tolist())
    method = st.selectbox("Method", ["onehot","label"])
    if st.button("Preview encode"):
        sample = S("dps_preview_df") or S("dps_df").head(50)
        res, msg = encode_categorical(sample.copy(), cols, method=method)
        st.info(msg); st.dataframe(res.head())
    if st.button("Add encode to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind":"encode","params":{"columns":cols,"method":method}})
        setS("dps_pipeline", pipe)
        push_history("add encode")
        st.success("Added encode step")

def section_scaling():
    st.header("Scaling")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    num_cols, _ = dtype_split(df)
    cols = st.multiselect("Numeric columns", num_cols)
    method = st.selectbox("Method", ["standard","minmax"])
    if st.button("Preview scale"):
        sample = S("dps_preview_df") or S("dps_df").head(50)
        res, msg = scale_features(sample.copy(), cols, method=method)
        st.info(msg); st.dataframe(res.head())
    if st.button("Add scale to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind":"scale","params":{"columns":cols,"method":method}})
        setS("dps_pipeline", pipe)
        push_history("add scale")
        st.success("Added scaling step")

def section_rebalance():
    st.header("Imbalanced Data (Rebalance)")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    target = st.selectbox("Target column", df.columns.tolist())
    method = st.selectbox("Method", ["oversample","undersample"])
    ratio = st.number_input("Ratio multiplier", min_value=0.0, value=1.0)
    if st.button("Preview rebalance (sample)"):
        sample = S("dps_preview_df") or S("dps_df").head(200)
        res, msg = rebalance_dataset(sample.copy(), target=target, method=method, ratio=ratio)
        st.info(msg); st.dataframe(res[target].value_counts())
    if st.button("Add rebalance to pipeline"):
        pipe = S("dps_pipeline")
        pipe.append({"kind":"rebalance","params":{"target":target,"method":method,"ratio":ratio}})
        setS("dps_pipeline", pipe)
        push_history("add rebalance")
        st.success("Added rebalance step")

def section_transforms():
    st.header("Transforms")
    df = S("dps_df")
    if df is None:
        st.warning("Upload data first.")
        return
    cols = st.multiselect("Columns", df.columns.tolist())
    transform = st.selectbox("Transform", ["log","boxcox","yeo-johnson","polynomial","binning"])
    kwargs = {}
    if transform == "polynomial":
        kwargs["degree"] = st.number_input("Degree", min_value=2, value=2)
    if transform == "binning":
        kwargs["bins"] = st.number_input("Bins", min_value=2, value=5)
    if st.button("Preview transform"):
        sample = S("dps_preview_df") or S("dps_df").head(50)
        res, msg = apply_transform(sample.copy(), columns=cols, transform=transform, **kwargs)
        st.info(msg); st.dataframe(res.head())
    if st.button("Add transform to pipeline"):
        pipe = S("dps_pipeline")
        params = {"columns": cols, "transform": transform}
        params.update(kwargs)
        pipe.append({"kind":"transform","params":params})
        setS("dps_pipeline", pipe)
        push_history("add transform")
        st.success("Transform step added")

def section_pipeline_preview():
    st.header("Pipeline & Preview")
    pl = S("dps_pipeline")
    if not pl:
        st.info("Pipeline is empty")
    else:
        for i, step in enumerate(pl, 1):
            st.write(f"{i}. {step.get('kind')} — {step.get('params')}")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run pipeline (apply)"):
            df = S("dps_df")
            if df is None:
                st.warning("Load data first")
            else:
                push_history("Before pipeline apply")
                out, msgs, err = run_pipeline(df, S("dps_pipeline"), atomic=S("dps_settings")["atomic_apply"])
                if err:
                    st.error(err)
                else:
                    setS("dps_df", out)
                    S("dps_changelog").extend(msgs)
                    update_preview_sample()
                    st.success("Pipeline applied")
    with col2:
        if st.button("Preview pipeline (on sample)"):
            sample = S("dps_preview_df") or S("dps_df")
            out, msgs, err = run_pipeline(sample, S("dps_pipeline"), atomic=True)
            if err:
                st.error(err)
            else:
                setS("dps_preview_df", out)
                st.info("Preview complete")
                st.dataframe(out.head())
    with col3:
        if st.button("Clear pipeline"):
            setS("dps_pipeline", [])
            st.success("Cleared pipeline")
    st.markdown("---")
    st.subheader("Save / Load pipeline")
    save_path = st.text_input("Save pipeline to path", value="pipeline.json")
    if st.button("Save pipeline to JSON"):
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(S("dps_pipeline"), f, default=str, indent=2)
            st.success(f"Saved pipeline to {save_path}")
        except Exception as e:
            st.error(f"Failed to save: {e}")
    load_file = st.file_uploader("Load pipeline JSON", type=["json"])
    if load_file:
        if st.button("Load pipeline from uploaded file"):
            try:
                pl_loaded = json.load(load_file)
                setS("dps_pipeline", pl_loaded)
                st.success("Loaded pipeline")
            except Exception as e:
                st.error(f"Failed to load pipeline: {e}")

    st.subheader("Export sklearn pipeline (best-effort)")
    if st.button("Build sklearn pipeline"):
        sk = build_sklearn_pipeline(S("dps_pipeline"))
        if sk is None:
            st.info("No sklearn pipeline built")
        else:
            st.success("Built sklearn pipeline object")
            if st.button("Pickle sklearn pipeline to sklearn_pipeline.pkl"):
                with open("sklearn_pipeline.pkl", "wb") as f:
                    pickle.dump(sk, f)
                st.success("Saved sklearn_pipeline.pkl")

def section_dashboard_download():
    st.header("Dashboard & Download")
    df = S("dps_df")
    if df is None:
        st.info("No data loaded")
        return
    st.subheader("Basic stats")
    st.json(compute_basic_stats(df))
    st.subheader("Preview sample")
    update_preview_sample()
    st.dataframe(S("dps_preview_df").head(200) if S("dps_preview_df") is not None else df.head(200))
    st.subheader("Download full dataset")
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", data=buf, file_name="processed_dataset.csv", mime="text/csv")
    if ALT_AVAILABLE:
        num_cols, _ = dtype_split(df)
        if num_cols:
            col = st.selectbox("Quick chart: choose numeric column", num_cols)
            chart = alt.Chart(df.sample(min(len(df), 10000), random_state=int(S("dps_settings")["random_state"]))).mark_bar().encode(
                alt.X(col, bin=True),
                y='count()'
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Main
# ---------------------------
def main():
    st.title("🧹 Data Preprocessing Studio — Hybrid (Fixed & Complete)")
    section = sidebar_navigation_and_settings()
    # Big-data warning if toggled
    if S("dps_settings").get("use_dask"):
        st.sidebar.warning("Big-data mode enabled — operations currently fall back to pandas for most steps.")
    if section == "Upload":
        section_upload()
    elif section == "Builder":
        section_builder()
    elif section == "Missing Data":
        section_missing_data()
    elif section == "Data Inconsistency":
        section_inconsistency()
    elif section == "Outliers":
        section_outliers()
    elif section == "Duplicates":
        section_duplicates()
    elif section == "Categorical Encoding":
        section_encoding()
    elif section == "Scaling":
        section_scaling()
    elif section == "Imbalanced Data":
        section_rebalance()
    elif section == "Transforms":
        section_transforms()
    elif section == "Pipeline & Preview":
        section_pipeline_preview()
    elif section == "Dashboard & Download":
        section_dashboard_download()
    else:
        st.info("Select a section from the sidebar.")

if __name__ == "__main__":
    main()
