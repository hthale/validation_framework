"""Gemeinsame Prüfliste aus dem vorhandenen Phase-0- und Phase-1/2-Lauf erzeugen."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

import config as cfg
from phase2_settings import Phase2Settings
from phase2_split import split_summary as summarize_split, validate_split
from record_identity import with_record_key
from result_integration import integrate_results
from review_common import file_sha256, read_json, write_json
from review_decisions import initial_history
from review_output import write_views

REQUIRED_FILES = ["run_info.json", "schema.json", "phase2_split_manifest.csv", "phase2_split_summary.json",
                  "phase2_ml_training_report.json", "domain_rules.json", "stable_profile_rules.json", "phase2_all_dependency_candidates.json",
                  "phase2_ml_rules.json", "ssl_cluster_rules.json", "naming_observations.csv"]
RULE_FILES = {"domain": "domain_rules.json", "profiling": "stable_profile_rules.json",
              "dependency_candidates": "phase2_all_dependency_candidates.json",
              "dependency_eligible": "phase2_ml_rules.json", "cluster": "ssl_cluster_rules.json"}


def choose_phase12_run(base):
    base = Path(base)
    if all((base / name).is_file() for name in REQUIRED_FILES):
        return base
    candidates = [path for path in base.iterdir() if path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)] if base.is_dir() else []
    if not candidates:
        raise ValueError("Kein abgeschlossener Paket-03-Lauf gefunden. main_phase1_phase_2.py ausführen oder --phase12-dir angeben.")
    return max(candidates, key=lambda path: path.name)


def verify_phase2_provenance(info, summary, split, reports, rules, settings):
    if summary != summarize_split(split, settings) or info.get("split") != summary:
        raise ValueError("Gespeicherte Aufteilungsberichte und Manifest widersprechen sich.")
    expected_hashes = {}
    for part in ("train", "validation"):
        keys = sorted(str(k) for k in split.loc[split["partition"].eq(part), "record_key"])
        expected_hashes[part] = hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode("utf-8")).hexdigest()
    by_target = {}
    for report in reports:
        if report.get("source") != "ml_phase2":
            continue
        if report.get("evaluation_scope") != "validation_partition" or report.get("test_evaluated") is not False:
            raise ValueError("ML-Bericht stammt nicht aus der getrennten Validierung.")
        for part in ("train", "validation"):
            if report.get(part + "_record_keys_sha256") != expected_hashes[part]:
                raise ValueError("ML-Bericht und Aufteilungsmanifest verwenden unterschiedliche Datensätze.")
        if report.get("target_col") in by_target:
            raise ValueError("Mehrdeutiger ML-Trainingsbericht.")
        by_target[report["target_col"]] = report
    for rule in rules["dependency_eligible"]:
        report = by_target.get(rule.get("metadata", {}).get("target_col"))
        if report is None or report.get("model_gate_passed") is not True:
            raise ValueError("ML-Regel besitzt keinen zugehörigen bestandenen Modellbericht.")
    for target, report in by_target.items():
        count = sum(r.get("metadata", {}).get("target_col") == target for r in rules["dependency_eligible"])
        if count != report.get("rule_count", 0):
            raise ValueError("Anzahl der ML-Prüfkandidaten widerspricht dem Trainingsbericht.")


def build(input_path, phase0_path, phase12_dir, output_base, review_fraction=.05):
    input_path, phase0_path, phase12_dir = Path(input_path), Path(phase0_path), Path(phase12_dir)
    missing = [name for name in REQUIRED_FILES if not (phase12_dir / name).is_file()]
    if missing:
        raise ValueError(f"Unvollständiger Phase-1/2-Lauf: {missing}")
    info = read_json(phase12_dir / "run_info.json")
    input_hash, phase0_hash = file_sha256(input_path), file_sha256(phase0_path)
    if info.get("input_sha256") != input_hash or info.get("phase0_sha256") != phase0_hash:
        raise ValueError("Eingabe oder Phase-0-Datei stimmt nicht mit dem Phase-1/2-Lauf überein. Dateien desselben Laufs verwenden.")
    split_summary = read_json(phase12_dir / "phase2_split_summary.json")
    options = info.get("phase2_settings", {})
    if not options:
        protocol = split_summary["protocol"]
        options = {key: protocol[key] for key in ("group_columns", "validation_fraction", "test_fraction", "random_state")}
    options = {key: value for key, value in options.items() if key in {f.name for f in fields(Phase2Settings)}}
    for key in ("group_columns", "max_depths", "min_samples_leaves"):
        if key in options:
            options[key] = tuple(options[key])
    settings = Phase2Settings(**options)
    raw = with_record_key(pd.read_parquet(input_path)).drop(columns=["geometry", "__index_level_0__"], errors="ignore")
    phase0 = pd.read_parquet(phase0_path)
    split = pd.read_csv(phase12_dir / "phase2_split_manifest.csv", dtype="string")
    observations = pd.read_csv(phase12_dir / "naming_observations.csv", dtype={"record_key": "string"})
    rules = {key: read_json(phase12_dir / name) for key, name in RULE_FILES.items()}
    split = validate_split(raw, split, settings)
    verify_phase2_provenance(info, split_summary, split, read_json(phase12_dir / "phase2_ml_training_report.json"), rules, settings)
    provenance = {"integration_version": "korrekturpaket_04_v1", "input_path": str(input_path.resolve()), "input_sha256": input_hash,
                  "phase0_path": str(phase0_path.resolve()), "phase0_sha256": phase0_hash,
                  "phase12_dir": str(phase12_dir.resolve()),
                  "phase12_file_sha256": {name: file_sha256(phase12_dir / name) for name in REQUIRED_FILES}}
    bundle = integrate_results(raw, phase0, split, settings, read_json(phase12_dir / "schema.json"), rules, observations, provenance, review_fraction)
    # Erst nach vollständiger Prüfung der Eingaben entsteht ein neuer Ausgabeordner.
    directory = Path(output_base) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    history = initial_history(bundle)
    write_json(directory / "review_bundle.json", bundle)
    write_json(directory / "review_history.json", history)
    write_json(directory / "rule_catalog.json", bundle["rule_catalog"])
    write_json(directory / "rule_evaluation_issues.json", bundle["rule_evaluation_issues"])
    write_views(directory, bundle, history)
    print(f"Phase-1/2-Quelle: {phase12_dir.resolve()}")
    print(f"Prüfliste: {len(bundle['cases'])} Datensätze; LOF-Prüfauswahl: {bundle['summary']['lof_review_budget']}")
    print(f"Noch offen: {len(bundle['cases'])}; fachlich bestätigte Fehler: 0")
    print(f"Technisch ungültige Regeln: {bundle['summary']['rule_evaluation_issue_count']}")
    print(f"Ergebnisordner: {directory.resolve()}")
    print(f"Im Browser öffnen: {(directory / 'review.html').resolve()}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=cfg.PARQUET_PATH)
    parser.add_argument("--phase0", type=Path, default=cfg.PHASE0_RESULTS_PARQUET)
    parser.add_argument("--phase12-dir", type=Path)
    parser.add_argument("--output", type=Path, default=cfg.FRAMEWORK_ROOT / "data" / "output_review_dop_check_04")
    parser.add_argument("--review-fraction", type=float, default=.05)
    args = parser.parse_args()
    try:
        source = args.phase12_dir if args.phase12_dir is not None else choose_phase12_run(cfg.OUTPUT_DIR)
        build(args.input, args.phase0, source, args.output, args.review_fraction)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Prüfliste nicht erstellt: {exc}\n")


if __name__ == "__main__":
    main()
