import pandas as pd


# ----------------------------------------------------------
# 1) Stratifizierung (generisch für ALKIS + DOP + andere)
# ----------------------------------------------------------
def add_stratification_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatische Stratifizierung.
    - Für DOP-Daten wird kachelname und bodenpixelgroesse genutzt.
    - Für ALKIS-Daten wird auf Gemarkung und Flurnummer zurückgegriffen.
    - Für sonstige Daten: generischer Fallback ("default").
    """
    df = df.copy()

    # -------------------------
    # DOP-Produkte (alte Logik)
    # -------------------------
    if "kachelname" in df.columns:
        kachelname_lower = df["kachelname"].astype("string").str.lower()
        df["stratum_product_type"] = kachelname_lower.str.extract(r"^(dop\d+)", expand=False)
    else:
        df["stratum_product_type"] = None

    if "bodenpixelgroesse" in df.columns:
        df["stratum_pixel_size"] = df["bodenpixelgroesse"].astype("Int64").astype("string")
    else:
        df["stratum_pixel_size"] = None

    if "aktualitaet" in df.columns:
        df["stratum_year"] = (
            pd.to_datetime(df["aktualitaet"], errors="coerce")
            .dt.year
            .astype("Int64")
            .astype("string")
        )
    else:
        df["stratum_year"] = None

    # -------------------------
    # ALKIS-Produkte (NEU)
    # -------------------------
    if df["stratum_product_type"].isna().all():
        # Gemarkungsnummer als Produkt-Ersatz
        if "gemarkungsnummer" in df.columns:
            df["stratum_product_type"] = df["gemarkungsnummer"].astype("string")
        else:
            df["stratum_product_type"] = None

    if df["stratum_pixel_size"].isna().all():
        # Flurnummer als "pixel size"-Ersatz
        if "flurnummer" in df.columns:
            df["stratum_pixel_size"] = df["flurnummer"].astype("Int64").astype("string")
        else:
            df["stratum_pixel_size"] = None

    if df["stratum_year"].isna().all():
        # entstehungsjahr verwenden, falls vorhanden
        for possible in ["beginnt", "zeitpunktderentstehung"]:
            if possible in df.columns:
                df["stratum_year"] = (
                    pd.to_datetime(df[possible], errors="coerce")
                    .dt.year
                    .astype("Int64")
                    .astype("string")
                )
                break

    # -------------------------
    # generischer Fallback für alles andere
    # -------------------------
    for col in ["stratum_product_type", "stratum_pixel_size", "stratum_year"]:
        df[col] = df[col].fillna("unknown")

    df["stratum_key"] = (
        df["stratum_product_type"].astype(str)
        + "__"
        + df["stratum_pixel_size"].astype(str)
        + "__"
        + df["stratum_year"].astype(str)
    )

    return df


# ----------------------------------------------------------
# 2) Stratifiziertes Sampling
# ----------------------------------------------------------
def stratified_sample_dataframe(
    df: pd.DataFrame,
    fraction: float,
    random_state: int,
    min_per_stratum: int = 1,
) -> pd.DataFrame:
    """
    Gleichmäßiges Sampling je Stratum.
    Sicherstellt, dass seltene Strata immer vorkommen (mind. min_per_stratum).
    """
    if len(df) == 0:
        return df.copy()

    if "stratum_key" not in df.columns:
        raise ValueError("Spalte 'stratum_key' fehlt. Bitte zuerst add_stratification_columns() aufrufen.")

    sampled_parts = []

    for _, group in df.groupby("stratum_key", dropna=False):
        n = max(min_per_stratum, int(round(len(group) * fraction)))
        n = min(n, len(group))  # nicht mehr ziehen als vorhanden
        sampled_group = group.sample(n=n, random_state=random_state)
        sampled_parts.append(sampled_group)

    if not sampled_parts:
        return df.iloc[0:0].copy()

    sampled_df = pd.concat(sampled_parts, axis=0)
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled_df


# ----------------------------------------------------------
# 3) Übersicht über Strata
# ----------------------------------------------------------
def get_stratum_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gibt eine kompakte Übersicht aller Strata.
    Nutzt ALKIS- und DOP-Felder, falls verfügbar.
    """
    if "stratum_key" not in df.columns:
        raise ValueError("Spalte 'stratum_key' fehlt. Bitte zuerst add_stratification_columns() aufrufen.")

    summary = (
        df.groupby(
            ["stratum_product_type", "stratum_pixel_size", "stratum_year", "stratum_key"],
            dropna=False
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return summary
