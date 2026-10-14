"""Stichprobenstrata anhand Cluster und LOF-Prüfpriorität."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from latentspace.anomaly_scores import rank_by_lof


def add_phase0_stratification_columns(df: pd.DataFrame, review_fraction: float = 0.05) -> pd.DataFrame:
    """Die oberen 5 % sind ein Prüfbudget, keine geschätzte Fehlerquote.

    ceil(N * Anteil) priorisierte Fälle; bei gleichen Scores bestimmt record_key
    die Reihenfolge. Weder 'other' noch ein Clusterlabel bestätigt Fehlerfreiheit.
    """
    if not np.isfinite(review_fraction) or not 0 <= review_fraction <= 1:
        raise ValueError("review_fraction muss zwischen 0 und 1 liegen.")
    ranked = rank_by_lof(df)
    count = math.ceil(len(ranked) * review_fraction)
    priority_keys = set(ranked.head(count)["record_key"])
    result = df.copy()
    if "cluster_label" in result:
        result["stratum_latent_cluster"] = result["cluster_label"].astype("string").fillna("unknown")
    else:
        result["stratum_latent_cluster"] = "unknown"
    result["stratum_outlier"] = np.where(result["record_key"].astype("string").isin(priority_keys), "review_priority", "other")
    result["stratum_key"] = result["stratum_latent_cluster"] + "__" + result["stratum_outlier"]
    return result
