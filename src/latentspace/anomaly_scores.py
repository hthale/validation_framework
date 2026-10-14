"""Gemeinsame Score-Definition für Training, Nachsortierung und Regelpipeline.

LOF-Rohwert: höher = lokal auffälliger. Kein Fehlerlabel und keine
Fehlerwahrscheinlichkeit. Die übrigen Kennzahlen bleiben Diagnostik.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from record_identity import validate_record_keys


SCORE_SCHEMA_VERSION = "lof_primary_v1"
PRIMARY_SCORE = "lof_raw"
LOF_JITTER_STD = 1e-6

# Diese abgeleiteten Ergebnisse sind weder Rohmerkmale noch Abhängigkeitsziele.
SCORE_COLUMNS = frozenset({
    "lof_raw", "lof_z", "outlier_rank", "primary_score_name", "score_schema_version",
    "reconstruction_error", "string_reconstruction_error", "reconstruction_z",
    "string_reconstruction_z", "hdbscan_outlier_raw", "hdbscan_outlier_z",
    "centroid_distance_raw", "centroid_distance_z", "centroid_distance_is_fallback",
    "cluster_label", "cluster_probability", "tabular_modality_share",
    "string_modality_share", "dominant_modality", "umap_x", "umap_y",
    "combined_outlier_score", "legacy_combined_outlier_score", "legacy_outlier_rank",
    "legacy_lof_z", "lof_raw_recomputed", "lof_rank", "rank_gain_lof",
})


def _finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name}: eindimensionaler Vektor mit endlichen Zahlen erforderlich.")
    return vector


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Bestehende Median/MAD-Transformation; rein bestandsbezogene Diagnostik."""
    values = _finite_vector(values, "z-Transformation")
    if len(values) == 0:
        return values.copy()
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std
    return 0.6745 * (values - med) / mad


def compute_lof_raw(Z: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    """LOF im vollständigen Z, mit demselben Float32-Jitter wie im bisherigen V2.

    Die Zeilenreihenfolge muss der ursprünglichen Modellreihenfolge entsprechen.
    Bei der Nachsortierung wird sie anhand von row_id rekonstruiert. Ungültige
    Werte werden nicht länger stillschweigend durch einen Score von 0 ersetzt.
    """
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 1:
        raise ValueError("LOF benötigt mindestens zwei Zeilen und eine latente Dimension.")
    if not np.isfinite(Z).all():
        raise ValueError("Der Latent Space enthält fehlende oder unendliche Werte.")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors muss eine positive ganze Zahl sein.")
    rng = np.random.default_rng(seed)
    Z_safe = Z + rng.normal(0, LOF_JITTER_STD, size=Z.shape).astype(np.float32)
    lof = LocalOutlierFactor(
        n_neighbors=min(int(n_neighbors), len(Z) - 1), metric="euclidean", novelty=False,
    )
    lof.fit_predict(Z_safe)
    raw = -lof.negative_outlier_factor_
    _finite_vector(raw, "LOF-Ergebnis")
    if (raw <= 0).any():
        raise ValueError("LOF hat nichtpositive Rohwerte erzeugt.")
    return raw


def build_outlier_scores(recon_error, string_recon_error, lof_raw, hdbscan_raw, centroid_raw):
    """Exportiert Einzelwerte; bildet keinen gewichteten Gesamtscore."""
    vectors = {
        "reconstruction_error": recon_error,
        "string_reconstruction_error": string_recon_error,
        "lof_raw": lof_raw,
        "hdbscan_outlier_raw": hdbscan_raw,
        "centroid_distance_raw": centroid_raw,
    }
    vectors = {name: _finite_vector(value, name) for name, value in vectors.items()}
    if len({len(value) for value in vectors.values()}) != 1:
        raise ValueError("Alle Score-Vektoren müssen dieselbe Länge haben.")
    result = pd.DataFrame(vectors)
    for raw, standard in [
        ("reconstruction_error", "reconstruction_z"),
        ("string_reconstruction_error", "string_reconstruction_z"),
        ("lof_raw", "lof_z"),
        ("hdbscan_outlier_raw", "hdbscan_outlier_z"),
        ("centroid_distance_raw", "centroid_distance_z"),
    ]:
        result[standard] = robust_zscore(result[raw].to_numpy())
    return result


def rank_by_lof(results: pd.DataFrame) -> pd.DataFrame:
    """Sortiert absteigend nach LOF, bei Gleichstand nach record_key.

    outlier_rank ist eine eindeutige Position (1..N), kein Fehlerlabel.
    Gleichstände werden nur zur reproduzierbaren Ausgabe aufgelöst.
    """
    if not results.columns.is_unique:
        raise ValueError("Ergebnistabelle enthält doppelte Spaltennamen.")
    if "lof_raw" not in results:
        raise ValueError("lof_raw fehlt. Alten V2-Lauf zuerst mit latentspace.rerank_lof umstellen.")
    if "record_key" not in results:
        raise ValueError("record_key fehlt; Ergebnissen werden keine neuen IDs zugewiesen.")
    if "combined_outlier_score" in results:
        raise ValueError("Alter Mischscore vorhanden. Explizite Umstellung mit latentspace.rerank_lof erforderlich.")
    for name, expected in [("score_schema_version", SCORE_SCHEMA_VERSION), ("primary_score_name", PRIMARY_SCORE)]:
        if name in results and (results[name].isna().any() or not results[name].eq(expected).all()):
            raise ValueError(f"Inkompatible Score-Metadaten in {name}.")
    result = results.copy()
    result["record_key"] = validate_record_keys(result["record_key"], context="LOF-Rangliste")
    result["lof_raw"] = _finite_vector(result["lof_raw"], "lof_raw")
    if (result["lof_raw"] <= 0).any():
        raise ValueError("lof_raw muss positiv sein.")
    result["lof_z"] = robust_zscore(result["lof_raw"].to_numpy())
    result = result.sort_values(["lof_raw", "record_key"], ascending=[False, True], kind="stable").reset_index(drop=True)
    result["outlier_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    result["primary_score_name"] = PRIMARY_SCORE
    result["score_schema_version"] = SCORE_SCHEMA_VERSION
    return result


def lof_metadata(n_rows: int, n_neighbors: int, seed: int) -> dict:
    return {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "primary_score_name": PRIMARY_SCORE,
        "higher_is_more_unusual": True,
        "score_is_error_probability": False,
        "ranking_tie_breaker": "record_key ascending",
        "lof": {
            "n_neighbors_requested": int(n_neighbors),
            "n_neighbors_effective": min(int(n_neighbors), n_rows - 1),
            "metric": "euclidean", "novelty": False, "dtype": "float32",
            "jitter_std": LOF_JITTER_STD, "seed": int(seed),
            "row_order": "row_id ascending (original model order)",
        },
        "interpretation": "Rangliste zur fachlichen Prüfung; keine bestätigten Fehler.",
        "modality_share_interpretation": "Anteile normierter Rekonstruktionsfehler; keine LOF-Ursachenerklärung.",
    }
