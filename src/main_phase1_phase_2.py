"""Bestandsprofiling und getrennt validierte Phase-2-Regelkandidaten."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd
import sklearn

import config as cfg
from check_foundation import json_compatible
from classification import classify_and_prioritize_rules, summarize_rule_classes
from date_roles import apply_date_roles, date_diagnostics
from dop_observations import dop_name_observations
from export_utils import save_csv
from io_utils import load_data
from latentspace.anomaly_scores import rank_by_lof
from latentspace.phase0_integration import add_phase0_stratification_columns
from latentspace.schema_interference import infer_schema
from phase0_join import merge_phase0_frames
from phase2_ml import build_phase2_ml_rules
from phase2_split import load_or_create_split, select_partition, split_summary
from preprocessing import normalize_by_schema
from profiling import profile_dataframe
from project_profile import EXPECTED_SPECTRAL_CHANNELS
from record_identity import IDENTITY_COLUMNS, with_record_key
from rule_engine import (attach_single_column_rule_proposals, build_domain_specific_rules,
                         merge_stable_proposed_rules, validate_rules_on_full_dataframe)
from sampling import add_stratification_columns, stratified_sample_dataframe


def save_json(path, value):
    path.write_text(json.dumps(json_compatible(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_profiles(profiles):
    summary = {"sample_count": len(profiles), "sample_row_counts": [p["row_count"] for p in profiles], "columns": {}}
    if profiles:
        for col, detail in profiles[0]["column_profiles"].items():
            rates = [p["column_profiles"][col]["null_rate"] for p in profiles]
            summary["columns"][col] = {"dtype_group": detail["dtype_group"], "null_rate_min": min(rates), "null_rate_max": max(rates)}
    return summary


def export_assessments(output_dir, stem, results):
    classified = classify_and_prioritize_rules(results)
    save_json(output_dir / f"classified_{stem}_rules.json", classified)
    save_csv(output_dir / f"classified_{stem}_rules.csv", classified)
    save_json(output_dir / f"classification_summary_{stem}.json", summarize_rule_classes(classified))


def run(input_path, phase0_path, output_base, split_path):
    if cfg.USE_LLM_SUGGESTIONS:
        raise ValueError("LLM-Vorschläge aus Gesamtprofilen sind in dieser getrennten Auswertung deaktiviert.")
    print("Lade Daten ...")
    raw = with_record_key(load_data(input_path)).drop(columns=["geometry", "__index_level_0__"], errors="ignore")
    split = load_or_create_split(raw, split_path, cfg.PHASE2_SETTINGS)
    summary = split_summary(split, cfg.PHASE2_SETTINGS)
    print("Phase-2-Aufteilung nach räumlicher Kachel:")
    for part, counts in summary["counts"].items():
        print(f"  {part}: {counts['rows']} Datensätze, {counts['groups']} Kacheln")

    schema = apply_date_roles(infer_schema(raw.drop(columns=list(IDENTITY_COLUMNS), errors="ignore")), raw.columns)
    numeric_columns = [c for c, role in schema.items() if role == "numeric"]
    date_columns = [c for c, role in schema.items() if role == "date"]
    date_report = date_diagnostics(raw, date_columns)
    print(f"Datumsspalten: {date_columns}")
    for detail in date_report:
        print(f"  {detail['column']}: {detail['parseable_count']} lesbar, {detail['invalid_count']} ungültig, {detail['missing_count']} fehlend")

    if cfg.USE_PHASE0:
        # Rohwerte für Phase 2 erhalten; globale Schemainferenz beeinflusst nur das Bestandsprofiling.
        phase2_df = merge_phase0_frames(raw, pd.read_parquet(phase0_path))
        print(f"Phase-0-Merge: {len(phase2_df)} / {len(raw)} Datensätze eindeutig zugeordnet.")
        df = add_phase0_stratification_columns(normalize_by_schema(phase2_df, schema))
        stratification_type = "phase0_latent_cluster_stratification"
    else:
        phase2_df = raw.copy()
        df = add_stratification_columns(normalize_by_schema(raw, schema))
        stratification_type = "proxy_metadata_stratification"

    output_dir = Path(output_base) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    split.to_csv(output_dir / "phase2_split_manifest.csv", index=False, encoding="utf-8")
    save_json(output_dir / "phase2_split_summary.json", summary)
    save_json(output_dir / "schema.json", schema)
    save_json(output_dir / "date_diagnostics.json", date_report)
    observations = dop_name_observations(raw)
    observations.to_csv(output_dir / "naming_observations.csv", index=False, encoding="utf-8-sig")
    print(f"Benennungsabweichungen zur fachlichen Klärung: {len(observations)}")
    df.groupby("stratum_key", dropna=False).size().rename("count").reset_index().to_csv(output_dir / "stratum_distribution.csv", index=False)

    profiles, rule_sets, sample_metadata, all_hints = [], [], [], []
    print("Starte Phase 1: stratifiziertes Bestandsprofiling ...")
    for index in range(cfg.SAMPLE_RUNS):
        seed = cfg.BASE_RANDOM_STATE + index
        sample = stratified_sample_dataframe(df, fraction=cfg.SAMPLE_FRACTION, random_state=seed, min_per_stratum=1)
        profile = profile_dataframe(sample, numeric_columns, date_columns, cfg.HIGH_CARDINALITY_THRESHOLD, excluded_columns=cfg.INTERNAL_COLUMNS)
        profile = attach_single_column_rule_proposals(
            profile, const_stddev_range_threshold=cfg.CONST_STDDEV_RANGE_THRESHOLD,
            max_numeric_distinct_for_in_rule=cfg.MAX_NUMERIC_DISTINCT_FOR_IN_RULE,
            enable_string_rules=cfg.ENABLE_STRING_RULES, min_regex_support_ratio=cfg.MIN_REGEX_SUPPORT_RATIO,
            enable_string_length_rules=cfg.ENABLE_STRING_LENGTH_RULES)
        profiles.append(profile)
        rule_sets.append(profile["proposed_rules"])
        sample_metadata.append({"run": index + 1, "random_state": seed, "sample_fraction": cfg.SAMPLE_FRACTION,
                                "sample_row_count": len(sample), "stratification_type": stratification_type,
                                "evaluation_scope": "descriptive_full_population"})
        all_hints.append({"run": index + 1, "profiling_hints": profile["profiling_hints"]})
        save_json(output_dir / f"sample_profile_run_{index + 1}.json", profile)
        save_csv(output_dir / f"proposed_rules_run_{index + 1}.csv", profile["proposed_rules"])
        save_json(output_dir / f"profiling_hints_run_{index + 1}.json", profile["profiling_hints"])
        print(f"  Durchlauf {index + 1}: {len(sample)} Datensätze, {len(profile['proposed_rules'])} Vorschläge")

    stable = merge_stable_proposed_rules(rule_sets, cfg.MIN_RULE_OCCURRENCES)
    domain = build_domain_specific_rules(df, expected_spectral_channels=EXPECTED_SPECTRAL_CHANNELS)
    profile_results, profile_examples = validate_rules_on_full_dataframe(df, stable, cfg.MAX_EXAMPLE_VIOLATIONS)
    domain_results, domain_examples = validate_rules_on_full_dataframe(df, domain, cfg.MAX_EXAMPLE_VIOLATIONS)
    for item in profile_results + domain_results:
        item["evaluation_scope"] = "descriptive_full_population"
    save_json(output_dir / "sampling_metadata.json", sample_metadata)
    save_json(output_dir / "sampling_profile_summary.json", summarize_profiles(profiles))
    save_json(output_dir / "profiling_hints_all_runs.json", all_hints)
    for stem, rules in (("stable_profile_rules", stable), ("domain_rules", domain)):
        save_json(output_dir / f"{stem}.json", rules)
        save_csv(output_dir / f"{stem}.csv", rules)
    for stem, results, examples in (("profiling", profile_results, profile_examples), ("domain", domain_results, domain_examples),
                                   ("all", profile_results + domain_results, {**profile_examples, **domain_examples})):
        save_csv(output_dir / f"{stem}_rule_validation_results.csv", results)
        save_json(output_dir / f"{stem}_rule_violation_examples.json", examples)
        export_assessments(output_dir, stem, results)
    if cfg.USE_PHASE0:
        rank_by_lof(df).head(250).to_csv(output_dir / "phase0_outlier_candidates.csv", index=False)
    print(f"Phase 1: {len(stable)} wiederholt vorgeschlagene Profiling-Regeln, {len(domain)} Domänenregeln.")

    print("Starte Phase 2: Gruppen-CV im Training und separate Validierung ...")
    all_rules, reports = build_phase2_ml_rules(
        phase2_df, accuracy_threshold=cfg.PHASE2_ACCURACY_THRESHOLD, f1_threshold=cfg.PHASE2_F1_THRESHOLD,
        precision_threshold=cfg.PHASE2_PRECISION_THRESHOLD, excluded_columns=cfg.INTERNAL_COLUMNS,
        backend=cfg.PHASE2_BACKEND, settings=cfg.PHASE2_SETTINGS, split=split)
    dependencies = [r for r in all_rules if r["source"] != "ssl_cluster"]
    clusters = [r for r in all_rules if r["source"] == "ssl_cluster"]
    validation = select_partition(phase2_df, split, "validation")
    results, examples = validate_rules_on_full_dataframe(validation, dependencies, cfg.MAX_EXAMPLE_VIOLATIONS)
    for result in results:
        result["evaluation_scope"] = "validation_partition"
    candidates = [r for report in reports for r in report.get("candidate_rules", [])]
    save_json(output_dir / "phase2_all_dependency_candidates.json", candidates)
    save_json(output_dir / "phase2_ml_training_report.json", [{k: v for k, v in report.items() if k != "candidate_rules"} for report in reports])
    save_json(output_dir / "phase2_ml_violation_examples.json", examples)
    save_csv(output_dir / "phase2_ml_validation_results.csv", results)
    save_json(output_dir / "phase2_ml_validation_results.json", results)
    export_assessments(output_dir, "ml", results)
    for stem, rules in (("phase2_ml_rules", dependencies), ("ssl_cluster_rules", clusters)):
        save_json(output_dir / f"{stem}.json", rules)
        save_csv(output_dir / f"{stem}.csv", rules)
    save_json(output_dir / "run_info.json", {
        "input_path": str(Path(input_path).resolve()), "input_sha256": file_sha256(input_path),
        "phase0_path": str(Path(phase0_path).resolve()) if cfg.USE_PHASE0 else None,
        "phase0_sha256": file_sha256(phase0_path) if cfg.USE_PHASE0 else None,
        "split_path": str(Path(split_path).resolve()), "split": summary,
        "phase2_settings": asdict(cfg.PHASE2_SETTINGS),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "dependency_candidates": len(candidates), "dependency_review_candidates": len(dependencies),
        "cluster_descriptions": len(clusters), "autoencoder_training_performed": False,
        "note": "Testanteil nur für Phase-2-Abhängigkeiten reserviert. Bestandsprofiling und bisheriger Autoencoder verwendeten den Gesamtbestand; kein unberührter externer Testdatensatz.",
    })
    print(f"Phase 2: {len(dependencies)} Abhängigkeitskandidaten zur fachlichen Prüfung, {len(clusters)} Clusterbeschreibungen.")
    print("Keine Auswertung der Phase-2-Abhängigkeiten auf dem reservierten Testanteil.")
    print(f"Ergebnisordner: {output_dir.resolve()}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=cfg.PARQUET_PATH)
    parser.add_argument("--phase0", type=Path, default=cfg.PHASE0_RESULTS_PARQUET)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_DIR, help="Basisordner; jeder Lauf erhält einen Zeitstempel-Unterordner.")
    parser.add_argument("--split-file", type=Path, default=cfg.PHASE2_SPLIT_PATH)
    args = parser.parse_args()
    run(args.input, args.phase0, args.output, args.split_file)


if __name__ == "__main__":
    main()
