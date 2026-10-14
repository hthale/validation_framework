"""Gespeicherte Mehrheits- und Änderungsbelege ohne Training nachrechnen."""
from __future__ import annotations

from pathlib import Path


def verify_content(out, source_run):
    import pandas as pd
    from review_common import read_json, file_sha256
    from comparison_core import Protocol, METHODS, evaluate_selected
    from phase2_split import select_partition
    from verify_rule_comparison import assert_same_cases
    from baseline_core import (BASELINE, ALL_METHODS, classify_rules, rule_counts, majority_model, baseline_cases,
        original_comparisons, make_variants, assert_frozen_inputs, pair_results, mutation_metrics)
    out, run = Path(out), Path(source_run)
    lock, protocol, summary = (read_json(out/name) for name in ("analysis_lock.json", "protocol.json", "summary.json"))
    checks = []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError("Nachprüfung fehlgeschlagen: "+name)
    check("Mehrheitsmodelle seit Analysebeginn unverändert", file_sha256(out/"majority_models.json") == lock["majority_models_sha256"])
    check("Protokoll seit Analysebeginn unverändert", file_sha256(out/"protocol.json") == lock["protocol_sha256"])
    check("Vorherige Artefakte unverändert", all(file_sha256(run/name) == expected for name, expected in lock["source_artifact_sha256"].items()))
    check("Keine fachlichen Fehlerlabels oder Trainingsbehauptung", not summary["training_performed"]
          and not summary["real_error_detection_measured"] and not summary["independent_external_evaluation"])
    candidates = read_json(run/"all_candidates.json")
    classified = classify_rules(candidates)
    check("Regeltypen bewahren sämtliche Kandidaten und Scope-Bedingungen", classified == read_json(out/"rule_classification.json"))
    check("Typzahlen stimmen ohne Umwahl von Regeln", rule_counts(classified) == read_json(out/"rule_counts.json"))
    check("Kandidaten-/Auswahlzahlen unverändert", summary["source_candidate_count"] == len(candidates)
          and summary["source_selected_count"] == sum(r["selected_on_validation"] for r in candidates))
    prior = read_json(run/"summary.json")
    p = Protocol(**read_json(run/"protocol.json")["parameters"])
    frame = pd.read_parquet(run/"model_table.parquet")
    cases = pd.read_parquet(out/"original_cases.parquet")
    pairs = pd.read_parquet(out/"paired_changes.parquet")
    old_cases = pd.read_parquet(run/"test_cases.parquet")
    models = read_json(out/"majority_models.json")
    comparison, availability, replay_pairs, all_expected_cases, guard_keys = [], [], [], [], set()
    for info in prior["splits"]:
        seed = info["seed"]
        split = pd.read_csv(run/f"seed_{seed}"/"split.csv")
        train, test = (select_partition(frame, split, part) for part in ("train", "test"))
        groups = split.set_index("record_key").group_id
        for no, target in enumerate(info["targets"], 1):
            prefix = f"{seed}/{target}"
            matched = [m for m in models if m["seed"] == seed and m["target"] == target]
            check(prefix+": genau eine Trainingsmehrheit", len(matched) == 1)
            model = matched[0]
            expected_model = majority_model(train, target)
            check(prefix+": Mehrheitswert und Häufigkeiten nur aus Training", all(model[k] == v for k, v in expected_model.items()))
            baseline = baseline_cases(test, target, groups, seed, model)
            old = old_cases.loc[old_cases.seed.eq(seed) & old_cases.target.eq(target)]
            expected_cases = pd.concat([old, baseline], ignore_index=True)
            actual = cases.loc[cases.seed.eq(seed) & cases.target.eq(target)].reset_index(drop=True)
            assert_same_cases(expected_cases, actual)
            all_expected_cases.append(expected_cases)
            check(prefix+": alle Originalfälle und Methoden unverändert gepaart", set(actual.method) == set(ALL_METHODS))
            comparison.extend(original_comparisons(actual, model))
            sub = run/f"seed_{seed}"/f"target_{no:02d}"
            contract = read_json(sub/"feature_contract.json")
            rules = read_json(sub/"selected_rules.json")
            routes = pd.read_parquet(sub/"test_routes.parquet").set_index("record_key")
            variants, avail = make_variants(train, test, groups, seed, target, model, protocol["max_reference_groups_per_scenario"])
            availability.extend(avail)
            for scenario, before, after, meta, changes in variants:
                folder = out/"changed_copies"/f"seed_{seed}"/sub.name/scenario
                before_saved = pd.read_parquet(folder/"original.parquet")
                after_saved = pd.read_parquet(folder/"changed.parquet")
                meta_saved = pd.read_parquet(folder/"pairs.parquet")
                for expected, saved in ((before, before_saved), (after, after_saved), (meta, meta_saved)):
                    assert_same_cases(expected, saved)
                assert_frozen_inputs(before_saved, after_saved, target, contract["features"], rules, changes)
                check(prefix+"/"+scenario+": Kopie, Änderung und eingefrorene Eingaben stimmen", True)
                guard_keys.add((seed, target, scenario, len(before)))
                for method in METHODS:
                    selected = [r for r in rules if r["scope"]["method"] == method]
                    labels = routes.loc[before.record_key, method].to_numpy()
                    b, _ = evaluate_selected(before_saved, labels, selected, target, groups, method, seed, p)
                    a, _ = evaluate_selected(after_saved, labels, selected, target, groups, method, seed, p)
                    replay_pairs.append(pair_results(b, a, meta_saved))
                replay_pairs.append(pair_results(baseline_cases(before_saved, target, groups, seed, model),
                                                baseline_cases(after_saved, target, groups, seed, model), meta_saved))
    assert_same_cases(pd.concat(all_expected_cases, ignore_index=True), cases)
    expected_pairs = pd.concat(replay_pairs, ignore_index=True) if replay_pairs else pd.DataFrame()
    assert_same_cases(expected_pairs, pairs)
    check("Jede Änderung hat exakt dieselben vier Methoden", pairs.empty or pairs.groupby("case_id").method.apply(set).map(lambda v: v == set(ALL_METHODS)).all())
    check("Keine doppelten Methodenfälle", pairs.empty or not pairs.duplicated(["case_id", "method"]).any())
    check("Originalmehrheit, Minderheitswerte und Fehlwerte getrennt", comparison == read_json(out/"majority_comparison.json"))
    check("Änderungskennzahlen aus gespeicherten Fallpaaren nachgerechnet", mutation_metrics(pairs) == read_json(out/"mutation_metrics.json"))
    check("Leere und besetzte Änderungsrichtungen dokumentiert", availability == read_json(out/"mutation_availability.json"))
    guards = read_json(out/"frozen_input_checks.json")
    check("Alle Routing-Wiederverwendungen begründet", {(g["seed"], g["target"], g["scenario"], g["pairs"]) for g in guards} == guard_keys and all(g["passed"] for g in guards))
    check("Zusammenfassungszahlen vollständig", len(models) == summary["baseline_models"]
          and len(pairs) == summary["paired_method_evaluations"] and len(guards) == summary["frozen_input_checks_passed"]
          and (int(pairs.case_id.nunique()) if len(pairs) else 0) == summary["paired_changes"])
    return {"passed": True, "check_count": len(checks), "checks": checks,
            "scope": "artifact_replay_and_pairing; no_model_refit; no_browser_automation"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from run_baseline_check import bootstrap, source_guard
    source, _, package11 = bootstrap(args.project)
    from experiment_io import read_verified_run
    from review_common import read_json
    out = args.run.resolve(strict=True)
    manifest = read_json(out/"run_manifest.json")
    if manifest.get("experiment") != "majority_and_controlled_changes_12":
        raise ValueError("Kein Paket-12-Lauf.")
    read_verified_run(out, list(manifest["output_sha256"]))
    run = Path(manifest["source_run"])
    source_guard(run, source, package11)
    import json
    print(json.dumps(verify_content(out, run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
