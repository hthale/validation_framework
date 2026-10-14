"""Stellt einen gespeicherten V2-Lauf auf LOF um, ohne ein Modell zu trainieren.

Aufruf aus dem Projekt mit PYTHONPATH=src:
python -m latentspace.rerank_lof --input PFAD --n-neighbors 20 --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd
import sklearn

from record_identity import validate_record_keys
from latentspace.anomaly_scores import compute_lof_raw, lof_metadata, rank_by_lof, robust_zscore
from latentspace.result_exports import (
    build_neighbor_preview, latent_columns, save_umap_html, save_umap_png,
)


def rerank_existing_results(
    results: pd.DataFrame, n_neighbors: int, seed: int, *, allow_lof_change: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if not results.columns.is_unique:
        raise ValueError("Doppelte Spaltennamen im gespeicherten Ergebnis.")
    if "score_schema_version" in results or "lof_raw" in results:
        raise ValueError("Die Datei enthält bereits einen neuen LOF-Export. Bitte den ursprünglichen V2-Lauf wählen.")
    required = {"row_id", "record_key", "lof_z", "cluster_label"}
    if required - set(results):
        raise ValueError(f"Unvollständiges V2-Ergebnis: {sorted(required - set(results))} fehlen.")
    work = results.copy()
    work["record_key"] = validate_record_keys(work["record_key"], context="Gespeicherter V2-Lauf")
    ids = pd.to_numeric(work["row_id"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or not np.array_equal(np.sort(ids), np.arange(len(work))):
        raise ValueError("row_id muss jede ursprüngliche Position von 0 bis N-1 genau einmal enthalten.")
    work["row_id"] = ids.astype(np.int64)
    work = work.sort_values("row_id", kind="stable").reset_index(drop=True)
    dimensions = latent_columns(work)
    raw = compute_lof_raw(work[dimensions].to_numpy(dtype=np.float32), n_neighbors, seed)
    old_z = work["lof_z"].to_numpy(dtype=float)
    new_z = robust_zscore(raw)
    if not np.isfinite(old_z).all():
        raise ValueError("Gespeicherte lof_z-Werte sind nicht vollständig endlich.")
    max_difference = float(np.max(np.abs(old_z - new_z)))
    matches = bool(np.allclose(old_z, new_z, rtol=1e-5, atol=1e-5))
    if not matches and not allow_lof_change:
        raise ValueError(
            f"LOF-Reproduktion weicht ab (max. z-Differenz {max_difference:.8g}). "
            "Parameter, vollständige Ergebnisdatei und scikit-learn-Version prüfen. "
            "--allow-lof-change nur für eine bewusst dokumentierte Neuberechnung verwenden."
        )
    renames = {
        "combined_outlier_score": "legacy_combined_outlier_score",
        "outlier_rank": "legacy_outlier_rank", "lof_z": "legacy_lof_z",
    }
    if any(name in work for name in renames.values()):
        raise ValueError("Legacy-Spalten bereits vorhanden; bitte die ursprüngliche Datei verwenden.")
    work = work.rename(columns=renames)
    work["lof_raw"] = raw
    # Der alte Zentroidwert für Noise ist ein pauschaler Ersatzwert, keine Distanz.
    if "centroid_distance_z" in work:
        work["centroid_distance_is_fallback"] = work["cluster_label"].eq(-1)
    ranked = rank_by_lof(work)
    overlap = None
    if "legacy_outlier_rank" in ranked:
        count = min(10, len(ranked))
        old_top = set(ranked.nsmallest(count, "legacy_outlier_rank")["record_key"])
        overlap = len(old_top & set(ranked.head(count)["record_key"]))
    info = {
        **lof_metadata(len(ranked), n_neighbors, seed),
        "row_count": len(ranked), "unique_record_keys": int(ranked["record_key"].nunique()),
        "latent_dim": len(dimensions), "training_performed": False,
        "evaluation_kind": "transductive_reranking_existing_latents",
        "source_lof_z_matches": matches,
        "source_lof_z_max_abs_difference": max_difference,
        "source_lof_z_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "allow_lof_change": bool(allow_lof_change),
        "top10_overlap_with_legacy": overlap,
        "centroid_noise_values": "Unveränderte historische Ersatzwerte, per Flag gekennzeichnet.",
        "validation_note": "Keine unabhängige Messung der Fehlererkennungsqualität.",
    }
    return ranked, info


def convert_file(
    input_path: Path, output_dir: Path, n_neighbors: int, seed: int,
    *, allow_lof_change: bool = False,
) -> tuple[pd.DataFrame, dict]:
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Eingabedatei fehlt: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Zielordner existiert bereits: {output_dir}. Mit --output einen neuen Ordner wählen.")
    with input_path.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    ranked, info = rerank_existing_results(
        pd.read_parquet(input_path), n_neighbors, seed, allow_lof_change=allow_lof_change,
    )
    info.update({
        "source_results": str(input_path), "source_sha256": source_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="lof_build_", dir=output_dir.parent))
    try:
        ranked.to_parquet(temporary / "latent_outlier_results_lof.parquet", index=False)
        ranked.head(250).to_csv(temporary / "top_outliers_lof.csv", index=False, encoding="utf-8-sig")
        columns = [c for c in ("record_key", "row_id", "outlier_rank", "lof_raw", "lof_z", "legacy_outlier_rank", "legacy_combined_outlier_score", "cluster_label", "kachelname") if c in ranked]
        ranked[columns].to_csv(temporary / "rank_comparison.csv", index=False, encoding="utf-8-sig")
        neighbors = build_neighbor_preview(ranked)
        (temporary / "top_outlier_neighbors_lof.json").write_text(json.dumps(neighbors, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        info["umap_html_created"] = False
        if {"umap_x", "umap_y"}.issubset(ranked.columns):
            xy = ranked[["umap_x", "umap_y"]].to_numpy(dtype=float)
            if not np.isfinite(xy).all():
                raise ValueError("Ungültige gespeicherte UMAP-Koordinaten.")
            ranked[["record_key", "row_id", "outlier_rank", "lof_raw", "lof_z", "umap_x", "umap_y", "cluster_label"]].to_csv(temporary / "umap_points_lof.csv", index=False, encoding="utf-8-sig")
            save_umap_png(ranked, temporary / "umap_plot_lof.png")
            info["umap_html_created"] = save_umap_html(ranked, temporary / "umap_plot_lof.html")
        (temporary / "rerank_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Zielordner wurde zwischenzeitlich angelegt: {output_dir}")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return ranked, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Vollständige latent_outlier_results_v2.parquet")
    parser.add_argument("--output", type=Path, help="Neuer Ordner; Standard: lof_primary neben der Eingabedatei")
    parser.add_argument("--n-neighbors", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-lof-change", action="store_true", help="Bewusste Neuberechnung trotz abweichender historischer LOF-z-Werte")
    args = parser.parse_args()
    output = args.output or args.input.parent / "lof_primary"
    try:
        ranked, info = convert_file(args.input, output, args.n_neighbors, args.seed, allow_lof_change=args.allow_lof_change)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Umstellung abgebrochen: {exc}\n")
    fields = ["row_count", "unique_record_keys", "primary_score_name", "training_performed", "source_lof_z_matches", "source_lof_z_max_abs_difference", "top10_overlap_with_legacy"]
    print(json.dumps({k: info[k] for k in fields}, ensure_ascii=False, indent=2))
    print(f"\nErgebnisordner: {output.resolve()}")
    print("\nTop 10 zur fachlichen Prüfung:")
    cols = [c for c in ("outlier_rank", "row_id", "lof_raw", "lof_z", "legacy_outlier_rank", "kachelname") if c in ranked]
    print(ranked[cols].head(10).to_string(index=False))
    if not info["umap_html_created"]:
        print("Interaktive HTML-Grafik nicht erstellt (Plotly oder UMAP-Koordinaten fehlen).")


if __name__ == "__main__":
    main()
