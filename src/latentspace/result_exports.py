"""Gemeinsame LOF-Ausgaben ohne Import des Trainingsmodells."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def latent_columns(results: pd.DataFrame) -> list[str]:
    columns = [c for c in results if re.fullmatch(r"latent_\d+", c)]
    columns.sort(key=lambda c: int(c.split("_")[1]))
    dimensions = [int(c.split("_")[1]) for c in columns]
    if not columns or dimensions != list(range(len(columns))):
        raise ValueError("Latente Dimensionen müssen lückenlos bei latent_00 beginnen.")
    return columns


def get_hover_columns(results: pd.DataFrame) -> list[str]:
    names = [
        "record_key", "row_id", "outlier_rank", "lof_raw", "lof_z", "cluster_label",
        "reconstruction_error", "string_reconstruction_error",
        "hdbscan_outlier_z", "centroid_distance_z", "centroid_distance_is_fallback",
        "kachelname", "xml_name", "tif_name", "aktualitaet", "bildflugname", "kamera_sensor",
        "gml_id", "flurstueckskennzeichen", "amtlicheflaeche", "gemarkungsnummer",
    ]
    return [name for name in names if name in results]


def save_umap_png(results: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(results["umap_x"], results["umap_y"], c=results["lof_raw"], s=18, alpha=0.85)
    fig.colorbar(scatter, ax=ax, label="LOF-Rohwert (höher = lokal auffälliger)")
    ax.set(xlabel="UMAP-1", ylabel="UMAP-2", title="Latent Space – LOF im vollständigen Merkmalsraum")
    fig.text(0.5, 0.015, "UMAP dient nur der Darstellung. Hohe LOF-Werte sind keine bestätigten Fehler.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_umap_html(results: pd.DataFrame, out_path: Path) -> bool:
    try:
        import plotly.express as px
    except ImportError:
        return False
    fig = px.scatter(
        results, x="umap_x", y="umap_y", color="lof_raw",
        hover_data=get_hover_columns(results),
        title="LOF-Rangliste – UMAP ausschließlich zur Darstellung",
        labels={"lof_raw": "LOF-Rohwert"},
    )
    fig.write_html(str(out_path), include_plotlyjs=True)
    return True


def build_neighbor_preview(results: pd.DataFrame, top_k: int = 100, k: int = 5) -> dict:
    """Nachbarn der LOF-Spitzenplätze; Schlüssel, Distanz und Rohdaten zur Prüfung."""
    if len(results) < 2:
        raise ValueError("Nachbarvorschau benötigt mindestens zwei Datensätze.")
    if top_k < 1 or k < 1:
        raise ValueError("top_k und k müssen positiv sein.")
    ordered = results.sort_values("outlier_rank").reset_index(drop=True)
    Z = ordered[latent_columns(ordered)].to_numpy(dtype=np.float32)
    if not np.isfinite(Z).all():
        raise ValueError("Ungültige latente Werte in der Nachbarvorschau.")
    count = min(k, len(ordered) - 1)
    model = NearestNeighbors(n_neighbors=count + 1, metric="euclidean").fit(Z)
    distances, indices = model.kneighbors(Z[:min(top_k, len(Z))])
    preview = [c for c in ("kachelname", "xml_name", "tif_name", "aktualitaet", "bildflugname", "kamera_sensor", "gml_id") if c in ordered]
    entries = []
    for pos, (dists, neighbors) in enumerate(zip(distances, indices)):
        choices = [(float(d), int(i)) for d, i in zip(dists, neighbors) if int(i) != pos]
        choices.sort(key=lambda item: (item[0], str(ordered.iloc[item[1]]["record_key"])))
        items = []
        for distance, index in choices[:count]:
            row = ordered.iloc[index]
            item = {"record_key": str(row["record_key"]), "distance": distance, "lof_raw": float(row["lof_raw"])}
            if "row_id" in row:
                item["row_id"] = int(row["row_id"])
            for col in preview:
                value = row[col]
                item[col] = None if value is None or (np.isscalar(value) and pd.isna(value)) else str(value)
            items.append(item)
        query = ordered.iloc[pos]
        entries.append({"record_key": str(query["record_key"]), "outlier_rank": int(query["outlier_rank"]), "lof_raw": float(query["lof_raw"]), "neighbors": items})
    return {"space": "full latent space, euclidean, without LOF jitter", "neighbor_k": count, "queries": entries}
