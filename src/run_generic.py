"""Eine unbekannte flache Parquet-Tabelle automatisch analysieren.

Aufruf: python src/run_generic.py --input data/meine_tabelle.parquet
Kein Produktprofil erforderlich. Kandidaten bleiben fachlich unbestätigt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from generic_schema import attach_identity, choose_groups, infer_columns, normalize_columns, tabular_preparation
from generic_rules import learn_profiles, learn_dependencies, describe_clusters
from generic_report import build_cases, render_report, original_value
from phase2_settings import Phase2Settings
from review_common import file_sha256, write_json, utc_now, atomic_text
from review_output import csv_text


SCOPE_NOTE = (
    "Automatische Bestandsanalyse ab Parquet. Autoencoder, LOF, Profiling und Clusterbeschreibungen "
    "verwenden den Gesamtbestand. Nur das Lernen von Attributabhängigkeiten besitzt eine getrennte "
    "Trainings-/Validierungsaufteilung und einen dabei unbenutzten Testanteil. "
    "Die Ergebnisse sind statistische Kandidaten und Prüfhinweise, keine bestätigten Produktregeln oder Fehler."
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "output_generic")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-string-fields", type=int, default=12)
    p.add_argument("--id-columns", nargs="+", default=[])
    p.add_argument("--group-columns", nargs="+", default=[])
    p.add_argument("--exclude-columns", nargs="+", default=[])
    return p


def analyze(args):
    if min(args.epochs, args.threads, args.n_neighbors, args.max_string_fields) < 1:
        raise ValueError("Epochen, Threads, Nachbarzahl und maximale Textspaltenzahl müssen positiv sein.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed muss zwischen 0 und 2**32-1 liegen.")
    source = args.input.resolve(strict=True)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Dieser Einstieg erwartet eine flache Parquet-Merkmalstabelle.")
    source_hash = file_sha256(source)
    raw_source = pd.read_parquet(source)
    if len(raw_source) < 12:
        raise ValueError("Der gemeinsame Analysemodus benötigt mindestens 12 Zeilen. Für Regelgüte sind deutlich mehr unabhängige Fälle nötig.")
    missing = set(args.id_columns + args.group_columns + args.exclude_columns) - set(raw_source)
    if missing:
        raise ValueError(f"Angegebene Spalten fehlen: {sorted(missing)}")
    raw, identity = attach_identity(raw_source, args.id_columns)
    grouped, grouping = choose_groups(raw, args.group_columns)
    excluded = sorted(set(args.id_columns + args.exclude_columns + grouping["columns"]))
    contract = infer_columns(raw, excluded=excluded)
    normalized, normalization_issues = normalize_columns(raw, contract)
    model_frame, prepared, feature_metadata = tabular_preparation(normalized, contract, args.max_string_fields)
    contract.update(identity=identity, grouping=grouping, latent_features=feature_metadata,
                    explicit_exclusions=args.exclude_columns)

    # Vorhandene Encoder verwenden zwei historische absolute Modulimporte.
    latent_dir = str(Path(__file__).resolve().parent / "latentspace")
    if latent_dir not in sys.path:
        sys.path.insert(0, latent_dir)
    os.environ.setdefault("NUMBA_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import joblib
    import torch
    from latentspace.config_latent import LatentV2Config
    from latentspace.main_v2 import train_model, compute_hdbscan_raw, compute_centroid_distance_raw, project_umap
    from latentspace.anomaly_scores import compute_lof_raw, build_outlier_scores, rank_by_lof, lof_metadata
    from latentspace.result_exports import save_umap_png, save_umap_html, build_neighbor_preview
    torch.set_num_threads(args.threads)
    cfg = LatentV2Config(
        parquet_path=str(source), output_root_dir=str(args.output.resolve()),
        epochs=args.epochs, random_state=args.seed, lof_n_neighbors=args.n_neighbors,
        string_columns=tuple(feature_metadata["string_columns"]),
        known_numeric_columns=(), known_date_columns=(), ignore_columns=(),
        enable_delivery_memory=False, umap_n_neighbors=min(15, len(raw)-1),
    )
    settings = Phase2Settings(group_columns=("__generic_group",), random_state=args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = args.output.resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    state = {"run_id": run_id, "status": "running", "started_at": utc_now(), "source_input": str(source),
             "input_sha256": source_hash, "scope_note": SCOPE_NOTE}
    def stage(name):
        state.update(stage=name, updated_at=utc_now())
        write_json(out / "run_status.json", state)
        print(f"[{name}]", flush=True)
    try:
        stage("01_merkmalsauswahl")
        raw.to_parquet(out / "input_with_ids.parquet", index=False)
        write_json(out / "feature_contract.json", contract)
        write_json(out / "normalization_issues.json", normalization_issues)
        print(f"{len(raw)} Zeilen; tabellarisch {prepared['X_tab'].shape[1]} Dimensionen; "
              f"{len(cfg.string_columns)} Zeichenkettenfelder; Gruppierung: {grouping['mode']}", flush=True)
        stage("02_latenter_raum")
        result = train_model(model_frame, cfg, prepared_tabular=prepared)
        Z = result["latent"]
        history = result["training_history"]
        if Z.shape != (len(raw), cfg.latent_dim) or not np.isfinite(Z).all():
            raise ValueError("Das Modell erzeugt ungültige latente Werte.")
        if len(history) != args.epochs or not all(np.isfinite(list(item.values())).all() for item in history):
            raise ValueError("Unvollständiger oder nicht endlicher Trainingsverlauf.")
        lof = compute_lof_raw(Z, cfg.lof_n_neighbors, cfg.random_state)
        labels, hdbscan, probabilities = compute_hdbscan_raw(Z, cfg.hdbscan_min_cluster_size, cfg.hdbscan_min_samples)
        centroid = compute_centroid_distance_raw(Z, labels)
        scores = build_outlier_scores(result["reconstruction_error"], result["string_reconstruction_error"], lof, hdbscan, centroid)
        projection = project_umap(Z, cfg)
        if not np.isfinite(projection).all():
            raise ValueError("Die UMAP-Darstellung enthält nicht endliche Werte.")
        coordinates = pd.DataFrame(Z, columns=[f"latent_{i:02d}" for i in range(Z.shape[1])])
        latent = pd.concat([raw, scores, result["modality_attribution"], coordinates], axis=1)
        latent["cluster_label"] = labels
        latent["cluster_probability"] = probabilities
        latent["centroid_distance_is_fallback"] = labels == -1
        latent["umap_x"], latent["umap_y"] = projection[:, 0], projection[:, 1]
        latent = rank_by_lof(latent)
        latent.to_parquet(out / "latent_outlier_results_lof.parquet", index=False)
        scalar_preview = [c for c, d in contract["columns"].items() if d["role"] not in {"unsupported", "technical"}]
        preview_columns = ["outlier_rank", "record_key", "lof_raw", "cluster_label", *scalar_preview]
        atomic_text(out / "top_outliers.csv", csv_text(latent[preview_columns].head(100).to_dict("records"), preview_columns))
        save_umap_png(latent, out / "umap.png")
        html_plot = save_umap_html(latent, out / "umap.html")
        # record_key ist die verbindliche Zuordnung; fremde row_id können Text sein.
        neighbors = build_neighbor_preview(latent.drop(columns="row_id", errors="ignore"), top_k=25, k=5)
        raw_lookup = raw.set_index("record_key")
        for query in neighbors["queries"]:
            for neighbor in query["neighbors"]:
                neighbor["original_values"] = original_value(raw_lookup.loc[neighbor["record_key"]].to_dict())
        write_json(out / "neighbors.json", neighbors)
        write_json(out / "training_history.json", history)
        torch.save(result["model"].state_dict(), out / "autoencoder.pt")
        joblib.dump(result["preprocessor"], out / "preprocessor.pkl")
        joblib.dump(result["tokenizer"], out / "tokenizer.pkl")
        write_json(out / "model_config.json", {"configuration": asdict(cfg), "feature_preparation": feature_metadata,
                                                "vocab_size": result["tokenizer"].vocab_size,
                                                "scoring": lof_metadata(len(raw), args.n_neighbors, args.seed),
                                                "fit_scope": "descriptive_full_population"})
        stage("03_regelkandidaten")
        profiles, profile_report = learn_profiles(normalized, contract, latent, args.seed)
        dependencies = learn_dependencies(grouped, settings, grouping, excluded)
        clusters, cluster_reports = describe_clusters(normalized, contract, latent, settings)
        catalog = {"schema_version": 1, "scope_note": SCOPE_NOTE, "profiling": profiles,
                   "dependency_candidates": dependencies["candidates"], "eligible_dependencies": dependencies["rules"],
                   "cluster_descriptions": clusters, "domain_rules": [], "approved_rules": []}
        ids = [r["rule_id"] for r in profiles + dependencies["candidates"] + clusters]
        if len(set(ids)) != len(ids):
            raise ValueError("Regelkatalog enthält mehrdeutige Regelkennungen.")
        write_json(out / "rule_catalog.json", catalog)
        write_json(out / "profiling_report.json", profile_report)
        write_json(out / "dependency_reports.json", dependencies["reports"])
        write_json(out / "dependency_feature_contract.json", dependencies["contract"])
        write_json(out / "dependency_split_summary.json", dependencies["summary"])
        split = dependencies["split"]
        if split is None:
            split = pd.DataFrame(columns=["record_key", "group_id", "partition"])
        split.to_csv(out / "dependency_split.csv", index=False, encoding="utf-8")
        write_json(out / "cluster_reports.json", cluster_reports)
        stage("04_bericht")
        cases, assessments, issues = build_cases(raw, normalized, latent, profiles, dependencies)
        write_json(out / "cases.json", cases)
        write_json(out / "rule_assessments.json", assessments)
        write_json(out / "rule_evaluation_issues.json", issues)
        if issues:
            raise ValueError("Technisch ungültige Regeln gefunden; rule_evaluation_issues.json prüfen.")
        if file_sha256(source) != source_hash:
            raise ValueError("Die Quelldatei wurde während des Laufs geändert; Ergebnisse nicht mit ihr vermischen.")
        summary = {"run_id": run_id, "completed_at": utc_now(), "row_count": len(raw),
                   "unique_record_keys": int(raw["record_key"].nunique()), "primary_score_name": "lof_raw",
                   "autoencoder_training_performed": True, "epochs": args.epochs,
                   "latent_dimensions": int(Z.shape[1]), "ml_candidate_count": len(dependencies["candidates"]),
                   "eligible_ml_rule_count": len(dependencies["rules"]), "profiling_rule_count": len(profiles),
                   "cluster_description_count": len(clusters), "cluster_count": len(set(labels)-{-1}),
                   "noise_count": int((labels == -1).sum()), "grouping": grouping,
                   "dependency_status": dependencies["summary"]["status"], "dependency_test_evaluated": False,
                   "invalid_rule_count": 0, "approved_rule_count": 0, "confirmed_error_count": 0,
                   "source_input_unchanged": True, "delivery_memory_written": False,
                   "scope_note": SCOPE_NOTE, "umap_html_created": html_plot}
        write_json(out / "summary.json", summary)
        render_report(out / "bericht.html", summary, contract, dependencies["candidates"], profiles, clusters, cases, cluster_reports)
        versions = {}
        for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "hdbscan", "umap-learn", "plotly"):
            try: versions[name] = version(name)
            except PackageNotFoundError: versions[name] = None
        src = Path(__file__).resolve().parent
        write_json(out / "run_manifest.json", {
            "source_input": str(source), "input_sha256": source_hash,
            "saved_input_sha256": file_sha256(out / "input_with_ids.parquet"),
            "python": platform.python_version(), "versions": versions,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256": {p.relative_to(src).as_posix(): file_sha256(p) for p in sorted(src.rglob("*.py"))},
            "output_sha256": {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "run_status.json"},
        })
        state.update(status="completed", completed_at=utc_now())
        write_json(out / "run_status.json", state)
        print(json.dumps({k: summary[k] for k in ("row_count", "ml_candidate_count", "eligible_ml_rule_count",
                                                "profiling_rule_count", "cluster_description_count", "cluster_count")}, ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}\nEingabekopie mit IDs: {out / 'input_with_ids.parquet'}", flush=True)
        return out
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=utc_now())
        write_json(out / "run_status.json", state)
        print(f"Unvollständiger Lauf: {out}; run_status.json enthält die Fehlerursache.", file=sys.stderr)
        raise


def main():
    p = parser()
    try:
        analyze(p.parse_args())
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, f"Analyse abgebrochen: {exc}\n")


if __name__ == "__main__":
    main()
