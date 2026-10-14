"""Bestehenden Vergleichslauf aus Paket 11 ohne erneutes Training auswerten."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def bootstrap(project):
    package11 = Path(__file__).resolve().parent.parent / "clusterregeln"
    if not (package11 / "run_rule_comparison.py").is_file():
        raise ValueError("Paket 12 neben den vorhandenen Ordner vergleichspaket_11_clusterregeln entpacken.")
    sys.path.insert(0, str(package11))
    from run_rule_comparison import bootstrap as existing_bootstrap
    source, adapter = existing_bootstrap(project)
    return source, adapter, package11


def latest_completed(root):
    import json
    if root.is_dir():
        for path in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
            status = path / "run_status.json"
            if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("status") == "completed":
                return path.resolve()
    raise ValueError(f"Kein abgeschlossener Paket-11-Lauf gefunden: {root}. Mit --run den Laufordner angeben.")


def source_guard(run, source, package11):
    from experiment_io import read_verified_run
    from review_common import read_json, file_sha256
    from verify_rule_comparison import verify_source_file
    manifest = read_json(run / "run_manifest.json")
    if manifest.get("experiment") != "cluster_rule_comparison_11":
        raise ValueError("--run muss auf einen abgeschlossenen Vergleich aus Paket 11 zeigen.")
    hashes = read_verified_run(run, list(manifest["output_sha256"]))
    for required in ("summary.json", "protocol.json", "model_table.parquet", "test_cases.parquet",
                     "all_candidates.json", "all_model_reports.json", "selection_lock.json"):
        if required not in hashes:
            raise ValueError(f"Unvollständiges Manifest: {required}")
    verify_source_file(manifest)
    for base, mapping in ((source, manifest["source_sha256"]), (package11, manifest["package_sha256"])):
        for rel, expected in mapping.items():
            path = base / rel
            if not path.resolve().is_relative_to(base.resolve()) or not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Quellcode passt nicht zum gespeicherten Vergleichslauf: {path}")
    protocol = read_json(run / "protocol.json")
    for filename, expected in protocol["code_sha256_before"].items():
        path = Path(filename)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Aufgezeichneter Vergleichscode fehlt oder wurde verändert: {filename}")
    return manifest, {**hashes, "run_manifest.json": file_sha256(run/"run_manifest.json"),
                      "run_status.json": file_sha256(run/"run_status.json")}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, help="Standard: letzter abgeschlossener Lauf unter data/regelvergleich_11")
    parser.add_argument("--output", type=Path, default=Path("data/mehrheitsvergleich_12"))
    parser.add_argument("--max-groups", type=int, default=50, help="Maximale Referenzgruppen je Seed/Ziel/Änderungsrichtung")
    return parser.parse_args()


def main():
    args = arguments()
    if args.max_groups < 1:
        raise ValueError("--max-groups muss positiv sein.")
    project = args.project.resolve(strict=True)
    source, adapter, package11 = bootstrap(project)
    import pandas as pd
    from comparison_core import Protocol, METHODS, evaluate_selected, summarize_cases
    from phase2_split import select_partition
    from experiment_io import new_output, finish_output
    from review_common import read_json, write_json, file_sha256, utc_now
    from verify_rule_comparison import verify_content, assert_same_cases
    from baseline_core import (BASELINE, majority_model, baseline_cases, classify_rules, rule_counts,
        original_comparisons, class_metrics, make_variants, assert_frozen_inputs, pair_results, mutation_metrics)
    from report_baseline import render_report
    from verify_baseline_check import verify_content as verify_new_content

    run = (args.run.resolve(strict=True) if args.run else latest_completed(project/"data/regelvergleich_11"))
    output_root = args.output if args.output.is_absolute() else project/args.output
    if output_root.resolve().is_relative_to(run):
        raise ValueError("Neuen Ausgabeordner außerhalb des bisherigen Vergleichslaufs wählen.")
    print(f"[01 Vorprüfung] Vorhandener Lauf: {run}", flush=True)
    manifest, source_hashes = source_guard(run, source, package11)
    old_verification = verify_content(run)
    package = Path(__file__).resolve().parent
    code_paths = [*package.rglob("*.py"), package/"PROTOKOLL.md"]
    initial_code = {p.relative_to(package).as_posix(): file_sha256(p) for p in sorted(code_paths)}
    prior = read_json(run/"summary.json")
    old_protocol = read_json(run/"protocol.json")
    p = Protocol(**old_protocol["parameters"])
    frame = pd.read_parquet(run/"model_table.parquet")
    source_cases = pd.read_parquet(run/"test_cases.parquet")
    out = new_output(output_root)
    try:
        write_json(out/"protocol.json", {"protocol_version": "12.1", "created_before_analysis": utc_now(),
            "source_run": str(run), "source_run_manifest_sha256": source_hashes["run_manifest.json"],
            "code_sha256_before": initial_code, "max_reference_groups_per_scenario": args.max_groups,
            "geometry_used": False, "training_performed": False, "thresholds_changed": False,
            "baseline_fit": "training_only_per_seed_target; no_gate; lexical_tie_break",
            "scenario_selection": "hash_order_one_original_case_per_fixed_reference_group",
            "routing": "reuse_saved_test_routes_after_identical_predictors_assertion",
            "experimental_scope": "exploratory_followup_on_previously_evaluated_internal_test_partition",
            "ground_truth": "none; controlled_changes_are_not_confirmed_domain_errors",
            "representation_effect_tested": "existing_rule_scope_and_selective_coverage",
            "not_tested": ["predictor_corruption", "new_records", "LOF_quality", "real_error_recall", "independent_external_generalization"]})
        (out/"PROTOKOLL.md").write_bytes((package/"PROTOKOLL.md").read_bytes())
        write_json(out/"source_verification.json", old_verification)
        print("[02 Mehrheitswerte] Bestimme Vergleichsvorhersagen ausschließlich aus Training ...", flush=True)
        bundles, models = [], []
        for info in prior["splits"]:
            seed = info["seed"]
            split = pd.read_csv(run/f"seed_{seed}"/"split.csv")
            train, validation, test = (select_partition(frame, split, name) for name in ("train", "validation", "test"))
            groups = split.set_index("record_key").group_id
            for no, target in enumerate(info["targets"], 1):
                model = {"seed": seed, **majority_model(train, target)}
                val_cases = baseline_cases(validation, target, groups, seed, model)
                model["validation"] = {**summarize_cases(val_cases), "observed_class_metrics": class_metrics(val_cases)}
                models.append(model)
                bundles.append((seed, target, run/f"seed_{seed}"/f"target_{no:02d}", train, test, groups, model))
        write_json(out/"majority_models.json", models)
        lock = {"majority_models_sha256": file_sha256(out/"majority_models.json"),
                "protocol_sha256": file_sha256(out/"protocol.json"),
                "locked_before_test_comparison": utc_now(), "source_artifact_sha256": source_hashes}
        write_json(out/"analysis_lock.json", lock)
        classified = classify_rules(read_json(run/"all_candidates.json"))
        write_json(out/"rule_classification.json", classified)
        counts = rule_counts(classified)
        write_json(out/"rule_counts.json", counts)
        model_reports = read_json(run/"all_model_reports.json")
        local_baselines = [{k: report.get(k) for k in ("seed", "method", "cluster", "target", "model_kind",
                           "selected_count", "validation_model_metrics", "validation_majority_baseline")}
                          for report in model_reports if report["method"] != "global" and report["selected_count"]]
        write_json(out/"local_majority_diagnostics.json", local_baselines)
        all_cases, all_pairs, comparison, availability, guards = [], [], [], [], []
        for seed, target, sub, train, test, groups, model in bundles:
            print(f"[03 Kontrollierte Kopien] Seed {seed}: {target}", flush=True)
            contract = read_json(sub/"feature_contract.json")
            selected = read_json(sub/"selected_rules.json")
            routes = pd.read_parquet(sub/"test_routes.parquet").set_index("record_key")
            if routes.index.tolist() != test.record_key.tolist():
                raise ValueError("Gespeicherte Routen passen nicht zu den Testkennungen.")
            original = source_cases.loc[source_cases.seed.eq(seed) & source_cases.target.eq(target)].copy()
            baseline = baseline_cases(test, target, groups, seed, model)
            original = pd.concat([original, baseline], ignore_index=True)
            all_cases.append(original)
            comparison.extend(original_comparisons(original, model))
            variants, available = make_variants(train, test, groups, seed, target, model, args.max_groups)
            availability.extend(available)
            for scenario, before, after, meta, changes in variants:
                assert_frozen_inputs(before, after, target, contract["features"], selected, changes)
                guards.append({"seed": seed, "target": target, "scenario": scenario, "pairs": len(before),
                    "unchanged_predictors": True, "unchanged_identities": True,
                    "routing_reused_by_identical_input": True, "passed": True})
                case_dir = out/"changed_copies"/f"seed_{seed}"/sub.name/scenario
                case_dir.mkdir(parents=True)
                before.to_parquet(case_dir/"original.parquet", index=False)
                after.to_parquet(case_dir/"changed.parquet", index=False)
                meta.to_parquet(case_dir/"pairs.parquet", index=False)
                for method in METHODS:
                    rules = [r for r in selected if r["scope"]["method"] == method]
                    labels = routes.loc[before.record_key, method].to_numpy()
                    before_cases, _ = evaluate_selected(before, labels, rules, target, groups, method, seed, p)
                    expected = original.loc[original.method.eq(method)].set_index("record_key", drop=False)
                    assert_same_cases(before_cases, expected.loc[before.record_key].reset_index(drop=True))
                    after_cases, _ = evaluate_selected(after, labels, rules, target, groups, method, seed, p)
                    all_pairs.append(pair_results(before_cases, after_cases, meta))
                all_pairs.append(pair_results(baseline_cases(before, target, groups, seed, model),
                    baseline_cases(after, target, groups, seed, model), meta))
        cases = pd.concat(all_cases, ignore_index=True)
        pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
        cases.to_parquet(out/"original_cases.parquet", index=False)
        pairs.to_parquet(out/"paired_changes.parquet", index=False)
        metrics = mutation_metrics(pairs)
        write_json(out/"majority_comparison.json", comparison)
        write_json(out/"mutation_metrics.json", metrics)
        write_json(out/"mutation_availability.json", availability)
        write_json(out/"frozen_input_checks.json", guards)
        same_covered = [r for r in comparison if r["stratum"] == "all" and r["method"] != BASELINE]
        selected_classes = [r for r in classified if r["selected_on_validation"]]
        summary = {"analysis_completed": True, "source_run": str(run), "sample_rows": len(frame),
            "seeds": [s["seed"] for s in prior["splits"]], "geometry_used": False,
            "training_performed": False, "thresholds_changed": False,
            "selected_rule_types": dict(Counter(r["rule_kind"] for r in selected_classes)),
            "selected_rule_types_by_method": {m: dict(Counter(r["rule_kind"] for r in selected_classes if r["method"] == m)) for m in METHODS},
            "source_candidate_count": len(classified), "source_selected_count": len(selected_classes),
            "baseline_models": len(models), "paired_changes": int(pairs.case_id.nunique()) if len(pairs) else 0,
            "paired_method_evaluations": len(pairs), "frozen_input_checks_passed": len(guards),
            "covered_predictions_differing_from_global_majority": {m: sum(r["different_predictions_from_majority"] for r in same_covered if r["method"] == m) for m in METHODS},
            "net_agreement_gain_on_same_covered": {m: sum(r["net_agreement_gain_on_same_covered"] for r in same_covered if r["method"] == m) for m in METHODS},
            "counts_are_sums_over_repeated_splits_not_independent_observations": True,
            "independent_external_evaluation": False, "real_error_detection_measured": False,
            "latent_benefit_automatically_claimed": False, "source_unchanged": True, "code_unchanged": True}
        write_json(out/"summary.json", summary)
        print("[04 Nachprüfung] Rechne gespeicherte Vergleichsfälle und Änderungen nach ...", flush=True)
        verification = verify_new_content(out, run)
        write_json(out/"verification.json", verification)
        render_report(out, summary, counts, comparison, metrics, availability, verification)
        for rel, expected in source_hashes.items():
            if file_sha256(run/rel) != expected:
                raise ValueError(f"Vorheriger Lauf hat sich während der Auswertung geändert: {rel}")
        source_guard(run, source, package11)
        for rel, expected in initial_code.items():
            if file_sha256(package/rel) != expected:
                raise ValueError(f"Paket 12 hat sich während der Auswertung geändert: {rel}")
        finish_output(out, {"experiment": "majority_and_controlled_changes_12", "source_run": str(run),
            "source_run_manifest_sha256": source_hashes["run_manifest.json"], "package_sha256": initial_code,
            "input_sha256": manifest["input_sha256"], "source": manifest["source"]})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}")
    except Exception as exc:
        write_json(out/"run_status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        raise SystemExit(f"Auswertung abgebrochen: {exc}")
