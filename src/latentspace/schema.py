import pandas as pd

def normalize_known_types(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    for col in cfg.known_numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in cfg.known_date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_null = df[col].dropna()

        if non_null.empty:
            continue

        # Binär-/Array-/Strukturspalten überspringen
        if non_null.apply(lambda x: isinstance(x, (bytes, bytearray, memoryview, list, dict))).mean() > 0.1:
            continue

        # Nur echte Strings strippen, nichts erzwingen
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return df