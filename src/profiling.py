from collections import Counter
import re
import pandas as pd


def infer_string_pattern(values: pd.Series) -> dict:
    """
    Erkennt heuristisch ein dominantes String-Muster.
    Die Funktion ist bewusst einfach gehalten und erzeugt
    keine perfekte Regex-Generalisierung, sondern ein robustes
    Näherungsmuster für Profiling-Zwecke.
    """
    vals = values.dropna().astype(str)
    if vals.empty:
        return {"pattern_type": "empty", "pattern": None}

    patterns = []

    for v in vals.head(500):
        if re.fullmatch(r"\d+", v):
            patterns.append(r"^\d+$")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            patterns.append(r"^\d{4}-\d{2}-\d{2}$")
        elif v.endswith(".xml"):
            patterns.append(r"^.+\.xml$")
        elif v.endswith(".tif") or v.endswith(".tiff"):
            patterns.append(r"^.+\.tif(f)?$")
        else:
            gen = []
            for ch in v:
                if ch.isdigit():
                    gen.append(r"\d")
                elif ch.isalpha():
                    gen.append(r"[A-Za-zÄÖÜäöü]")
                else:
                    gen.append(re.escape(ch))
            patterns.append("^" + "".join(gen) + "$")

    most_common_pattern, count = Counter(patterns).most_common(1)[0]
    return {
        "pattern_type": "heuristic",
        "pattern": most_common_pattern,
        "support": int(count),
        "sample_size": int(len(patterns)),
        "support_ratio": float(count / len(patterns)) if patterns else 0.0,
    }


def profile_numeric(series: pd.Series) -> dict:
    """
    Berechnet numerische Profilkennzahlen.
    Boolean-Spalten werden vorher in 0/1 umgewandelt.
    """
    s = series.dropna()

    if s.empty:
        return {}

    if pd.api.types.is_bool_dtype(s):
        s = s.astype(int)

    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return {}

    min_val = s.min()
    max_val = s.max()
    mean_val = s.mean()
    std_val = s.std(ddof=1) if len(s) > 1 else 0.0
    val_range = max_val - min_val

    if pd.isna(std_val):
        std_val = 0.0

    stability_ratio = float(std_val / val_range) if val_range != 0 else 0.0
    distinct_values = sorted(s.dropna().unique().tolist())

    return {
        "min": float(min_val) if pd.notna(min_val) else None,
        "max": float(max_val) if pd.notna(max_val) else None,
        "mean": float(mean_val) if pd.notna(mean_val) else None,
        "stddev": float(std_val) if pd.notna(std_val) else None,
        "range": float(val_range) if pd.notna(val_range) else None,
        "distinct_count": int(s.nunique()),
        "distinct_values_if_small": distinct_values if len(distinct_values) <= 10 else None,
        "stability_ratio_stddev_div_range": stability_ratio,
    }


def profile_categorical(series: pd.Series, high_cardinality_threshold: int) -> dict:
    """
    Ermittelt Häufigkeiten und mögliche kleine Wertemengen
    für kategoriale Spalten.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return {}

    value_counts = s.value_counts(dropna=True)
    distinct_values = value_counts.index.tolist()

    return {
        "distinct_count": int(s.nunique()),
        "top_values": value_counts.head(20).to_dict(),
        "all_values_if_small": distinct_values if len(distinct_values) <= high_cardinality_threshold else None,
    }


def profile_string(series: pd.Series) -> dict:
    """
    Berechnet String-bezogene Kennzahlen:
    - min/max Länge
    - Mustererkennung
    - Beispielwerte
    """
    s = series.dropna().astype(str)
    if s.empty:
        return {}

    lengths = s.str.len()
    pattern_info = infer_string_pattern(s)

    return {
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "pattern_info": pattern_info,
        "examples": s.head(5).tolist(),
    }

def profile_date(series: pd.Series) -> dict:
    parsed = pd.to_datetime(series.astype("string"), errors="coerce", format="mixed")
    invalid = series.notna() & parsed.isna()
    valid = parsed.dropna()
    return {
        "min": str(valid.min().date()) if len(valid) else None,
        "max": str(valid.max().date()) if len(valid) else None,
        "missing_count": int(series.isna().sum()),
        "parseable_count": int(parsed.notna().sum()),
        "invalid_count": int(invalid.sum()),
        "invalid_examples": series[invalid].astype(str).head(5).tolist(),
        "format_conformance_checked": False,
    }


def classify_column(
    df: pd.DataFrame,
    col: str,
    numeric_columns: list[str],
    date_columns: list[str],
    high_cardinality_threshold: int,
) -> str:
    """
    Typisierung einer Spalte in numeric/date/categorical/string/unknown.
    """
    if col in date_columns:
        return "date"
    if col in numeric_columns:
        return "numeric"

    non_null = df[col].dropna()
    if non_null.empty:
        return "unknown"

    nunique = non_null.nunique()
    if nunique <= high_cardinality_threshold:
        return "categorical"

    return "string"


def profile_dataframe(
    df: pd.DataFrame,
    numeric_columns: list[str],
    date_columns: list[str],
    high_cardinality_threshold: int,
    excluded_columns: list[str] | None = None,
) -> dict:
    """
    Führt das Spaltenprofiling für alle relevanten Spalten durch.
    Interne Hilfsspalten können ausgeschlossen werden.
    """
    excluded_columns = set(excluded_columns or [])

    result = {
        "row_count": int(len(df)),
        "column_profiles": {},
    }

    for col in df.columns:
        if col in excluded_columns:
            continue

        dtype_group = classify_column(
            df,
            col,
            numeric_columns=numeric_columns,
            date_columns=date_columns,
            high_cardinality_threshold=high_cardinality_threshold,
        )

        series = df[col]
        null_rate = float(series.isna().mean())

        col_result = {
            "dtype_group": dtype_group,
            "null_rate": null_rate,
        }

        if dtype_group == "numeric":
            profile = profile_numeric(series)
        elif dtype_group == "categorical":
            profile = profile_categorical(series, high_cardinality_threshold)
        elif dtype_group == "string":
            profile = profile_string(series)
        elif dtype_group == "date":
            profile = profile_date(series)
        else:
            profile = {}

        col_result["profile"] = profile
        result["column_profiles"][col] = col_result

    return result