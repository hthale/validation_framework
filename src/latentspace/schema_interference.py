import pandas as pd
import numpy as np
import re

def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}

    hex_pattern = re.compile(r"^[0-9A-Fa-f]+$")
    
    for col in df.columns:
        series = df[col].dropna()

        # 1. Empty column
        if series.empty:
            schema[col] = "unknown"
            continue

        # 2. Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            schema[col] = "numeric"
            continue

        # Try to parse numeric
        try:
            _ = pd.to_numeric(series, errors="raise")
            schema[col] = "numeric"
            continue
        except:
            pass

        # 3. Boolean detection
        lowered = series.astype(str).str.lower().unique()
        if set(lowered).issubset({"true", "false", "0", "1"}):
            schema[col] = "boolean"
            continue

        # 4. Geometry detection (WKB hex)
        if series.dtype == object and series.apply(lambda x: isinstance(x, str) and bool(hex_pattern.match(x))).mean() > 0.9:
            schema[col] = "geometry_wkb"
            continue

        # 5. List/dict/array
        if series.apply(lambda x: isinstance(x, (list, dict, np.ndarray))).mean() > 0.3:
            schema[col] = "json_struct"
            continue

        # 6. Categorical
        unique_ratio = series.nunique() / len(series)
        avg_len = series.astype(str).str.len().mean()

        if unique_ratio < 0.2 and avg_len < 30:
            schema[col] = "categorical"
            continue

        # 7. Text
        if avg_len > 30:
            schema[col] = "text"
            continue

        # Default
        schema[col] = "string"

    return schema
