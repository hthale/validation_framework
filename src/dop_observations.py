"""Beobachtungen zu DOP-Bezeichnern, getrennt von bestätigten Fachregeln."""
from __future__ import annotations

import pandas as pd


def dop_name_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Abweichende Bezeichner protokollieren, ohne die fachliche Ursache zu behaupten."""
    required = {"xml_name", "tif_name", "kachelname"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["record_key", "observation", "review_status"])

    def stem(series: pd.Series) -> pd.Series:
        return series.astype("string").str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1].str.replace(r"\.[^.]+$", "", regex=True)

    xml = stem(df["xml_name"])
    tif = stem(df["tif_name"])
    kachel = df["kachelname"].astype("string")
    applicable = xml.notna() & tif.notna() & kachel.notna()
    differs = applicable & ((xml != tif) | (xml != kachel))
    columns = [c for c in ["record_key", "xml_name", "tif_name", "kachelname", "spektralkanaele", "tif_band_count"] if c in df]
    result = df.loc[differs.fillna(False), columns].copy()
    result["observation"] = "Dateibasename und/oder Kachelname unterscheiden sich"
    result["review_status"] = "fachlich_zu_klaeren"
    return result

