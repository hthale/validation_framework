"""Geprüfte 1:1-Verknüpfung mit den Ergebnissen desselben Eingabebestands."""
from __future__ import annotations

import pandas as pd

from record_identity import validate_record_keys, with_record_key
from latentspace.anomaly_scores import rank_by_lof


def merge_phase0_frames(df: pd.DataFrame, phase0: pd.DataFrame) -> pd.DataFrame:
    left = with_record_key(df)
    right = rank_by_lof(phase0)
    required = ["record_key", "cluster_label", "lof_raw", "lof_z", "outlier_rank", "primary_score_name", "score_schema_version"]
    missing_columns = [c for c in required if c not in right.columns]
    if missing_columns:
        raise ValueError(f"Phase-0-Ergebnis: erforderliche Spalten fehlen: {missing_columns}")
    right["record_key"] = validate_record_keys(right["record_key"], context="Phase 0")

    missing_keys = ~left["record_key"].isin(right["record_key"])
    extra_keys = ~right["record_key"].isin(left["record_key"])
    if missing_keys.any() or extra_keys.any():
        raise ValueError(
            f"Phase-0-Ergebnis passt nicht zum Eingabebestand: "
            f"{int(missing_keys.sum())} Schlüssel fehlen, {int(extra_keys.sum())} sind zusätzlich. "
            "Eingabebestand und Zuordnung zum gespeicherten Phase-0-Lauf prüfen."
        )

    optional = [
        "reconstruction_error", "reconstruction_z", "string_reconstruction_z",
        "hdbscan_outlier_raw", "hdbscan_outlier_z", "centroid_distance_raw",
        "centroid_distance_z", "centroid_distance_is_fallback", "cluster_probability",
        "string_reconstruction_error", "dominant_modality", "tabular_modality_share",
        "string_modality_share", "umap_x", "umap_y",
    ]
    columns = required + [c for c in optional if c in right.columns]
    collisions = [c for c in columns if c != "record_key" and c in left.columns]
    if collisions:
        raise ValueError(f"Phase-0-Spalten bereits im Eingabebestand vorhanden: {collisions}")

    result = left.merge(right[columns], on="record_key", how="left", sort=False, validate="one_to_one")
    # Auch nicht fortlaufende und wiederholte DataFrame-Indizes bleiben erhalten.
    result.index = df.index
    return result
