"""Reproduzierbare Nachprüfung ohne erneutes Autoencoder-Training oder Pickle-Laden."""
from __future__ import annotations

from pathlib import Path


def verify_content(out):
    import numpy as np
    import pandas as pd
    from comparison_core import Protocol, METHODS, evaluate_selected, rule_evidence
    from baseline_core import BASELINE, majority_model, baseline_cases, classify_rules
    from verify_rule_comparison import assert_same_cases
    from review_common import read_json, file_sha256
    from synthetic_data import TARGET, CONTROL, SCENARIOS, create_population, scenario_frame, normal_mask, make_test_variants
    from evaluation import (summarize_predictions, fit_lof, score_lof, validation_threshold,
        lof_cases, summarize_lof, compare_methods)
    out = Path(out)
    protocol, lock = read_json(out/"protocol.json"), read_json(out/"selection_lock.json")
    p = Protocol(**protocol["rule_parameters"])
    checks = []
    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})
        if not value:
            raise AssertionError("Nachprüfung fehlgeschlagen: "+name)
    check("Protokoll unverändert", file_sha256(out/"protocol.json") == lock["protocol_sha256"])
    check("Modelle, Auswahl und Ausgangsdaten eingefroren", all(file_sha256(out/n) == h for n, h in lock["frozen_sha256"].items()))
    rule_metrics, lof_metrics, null_checks = [], [], []
    trained_models = 0
    for seed in protocol["seeds"]:
        sd = out/f"seed_{seed}"
        attributes, truth = pd.read_parquet(sd/"attributes.parquet"), pd.read_parquet(sd/"reference.parquet")
        generated, reference = create_population(seed, protocol["units_per_seed"])
        assert_same_cases(attributes, generated)
        assert_same_cases(truth, reference)
        check(f"{seed}: Normale Referenzwerte nach unabhängiger Intervallprüfung", normal_mask(attributes).all())
        check(f"{seed}: Kennungen eindeutig, Gruppen überschneiden keine Partition", attributes.record_key.is_unique and
              attributes.record_key.tolist() == truth.record_key.tolist() and truth.groupby("group_id").partition.nunique().max() == 1)
        check(f"{seed}: Seltene gültige Gruppe in allen Partitionen", set(truth.loc[truth.rare_normal, "partition"]) == {"train", "validation", "test"})
        groups = truth.set_index("record_key").group_id
        for scenario in SCENARIOS:
            sub = sd/scenario
            frame, features = scenario_frame(attributes, scenario)
            train, validation, test = (frame.loc[truth.partition.eq(part)].reset_index(drop=True) for part in ("train", "validation", "test"))
            prefix = f"{seed}/{scenario}"
            variants, labels = make_test_variants(train, test, truth, seed, protocol["perturbed_test_group_fraction"])
            assert_same_cases(labels, pd.read_parquet(sub/"test_labels.parquet"))
            for name, variant in variants.items():
                assert_same_cases(variant, pd.read_parquet(sub/f"test_{name}.parquet"))
                expected = np.zeros(len(test), dtype=bool) if name == "clean" else labels.is_error.to_numpy()
                check(prefix+"/"+name+": Fehlerlabels entsprechen Referenzdefinition", np.array_equal(~normal_mask(variant), expected))
            check(prefix+": Fehler nur in einer Beobachtung, normale Partner vorhanden", labels.loc[labels.is_error].group_id.is_unique and
                  set(labels.loc[labels.is_error, "group_id"]) <= set(labels.loc[~labels.is_error, "group_id"]))
            contracts = {}
            for name, expected_features in (("rule_representation", features), ("lof_representation", [*features, TARGET])):
                folder = sub/name
                contract = read_json(folder/"contract.json")
                contracts[name] = contract
                check(prefix+"/"+name+": ausschließlich erlaubte Merkmale", contract["features"] == expected_features
                      and CONTROL not in contract["features"] and not contract["geometry_used"])
                check(prefix+"/"+name+": ausschließlich Trainingsgruppen angepasst", contract["train_keys"] == train.record_key.tolist()
                      and contract["validation_keys"] == validation.record_key.tolist() and not set(contract["train_keys"]) & set(test.record_key))
                history = read_json(folder/"training_history.json")
                check(prefix+"/"+name+": echter endlicher Trainingsverlauf", len(history) == p.epochs and
                      [r["epoch"] for r in history] == list(range(1, p.epochs+1)) and
                      all(np.isfinite(list(r.values())).all() for r in history) and (folder/"autoencoder.pt").stat().st_size > 1000)
                trained_models += 1
                with np.load(folder/"development.npz", allow_pickle=False) as matrices:
                    for matrix_name, matrix in matrices.items():
                        n = len(train) if matrix_name.endswith("train") else len(validation)
                        d = p.latent_dim if matrix_name.startswith("latent") else contract["dimensions"]
                        check(prefix+"/"+name+"/"+matrix_name+": passende endliche Matrix", matrix.shape == (n, d) and np.isfinite(matrix).all())
            routes = pd.read_parquet(sub/"test_routes.parquet")
            val_routes = pd.read_parquet(sub/"validation_routes.parquet")
            check(prefix+": Routing-Kennungen passen", routes.record_key.tolist() == test.record_key.tolist()
                  and val_routes.record_key.tolist() == validation.record_key.tolist())
            selected = read_json(sub/"selected_rules.json")
            candidates = read_json(sub/"candidates.json")
            check(prefix+": Auswahl unverändert aus Kandidaten", selected == [r for r in candidates if r["selected_on_validation"]])
            check(prefix+": eindeutige Regelkennungen", len({r["rule_id"] for r in candidates}) == len(candidates))
            check(prefix+": Regelarten nachvollziehbar", classify_rules(candidates) == read_json(sub/"rule_classification.json"))
            reports = read_json(sub/"model_reports.json")
            gates = {(r["target"], r["method"], r["cluster"]): r["model_gate_passed"] for r in reports}
            for rule in selected:
                scope = rule["scope"]
                subset = validation if scope["method"] == "global" else validation.loc[val_routes[scope["method"]].eq(scope["cluster"])]
                evidence = rule_evidence(subset, rule["inner_rule"], groups)
                check(prefix+"/"+rule["rule_id"]+": Validierung und Scope", evidence == rule["validation"] and
                      evidence["cases"] >= p.min_rule_cases and evidence["groups"] >= p.min_rule_groups and
                      evidence["agreement_all"] >= p.min_agreement and gates[(scope["target"],scope["method"],scope["cluster"])] and
                      all(c["feature"] in features for c in rule["inner_rule"]["metadata"]["conditions"]) and not rule["test_used_for_selection"])
            majority = read_json(sub/"majority_model.json")
            check(prefix+": Mehrheitsbaseline nur aus Training", majority_model(train, TARGET) == majority)
            saved_cases = pd.read_parquet(sub/"rule_cases.parquet")
            expected_rule_cases = []
            for variant in ("clean", "code_swap"):
                expected_labels = labels.copy()
                if variant == "clean": expected_labels["is_error"] = False
                for method in (*METHODS, BASELINE):
                    if method == BASELINE:
                        cases = baseline_cases(variants[variant], TARGET, groups, seed, majority)
                    else:
                        rules = [r for r in selected if r["scope"]["target"] == TARGET and r["scope"]["method"] == method]
                        cases, _ = evaluate_selected(variants[variant], routes[method].to_numpy(), rules, TARGET, groups, method, seed, p)
                    expected_rule_cases.append(cases.assign(variant=variant))
                    rule_metrics.append({"seed": seed, "scenario": scenario, "variant": variant, "method": method,
                                         **summarize_predictions(cases, expected_labels)})
            assert_same_cases(pd.concat(expected_rule_cases, ignore_index=True), saved_cases)
            check(prefix+": alle Regelprüfungen einschließlich Enthaltungen nachgerechnet", True)
            for method in METHODS:
                controls = [r for r in selected if r["scope"]["target"] == CONTROL and r["scope"]["method"] == method]
                null_checks.append({"seed": seed, "scenario": scenario, "method": method,
                    "selected_control_rules": len(controls), "null_control_rejected": not controls})
            # LOF aus gespeicherten Matrizen frisch nachrechnen, keine Modelle laden.
            with np.load(sub/"lof_representation"/"development.npz", allow_pickle=False) as npz:
                train_matrices = {"attributes": npz["attribute_train"], "latent": npz["latent_train"]}
                val_matrices = {"attributes": npz["attribute_validation"], "latent": npz["latent_validation"]}
            with np.load(sub/"pca_development.npz", allow_pickle=False) as npz:
                train_matrices["pca"], val_matrices["pca"] = npz["train"], npz["validation"]
            thresholds = read_json(sub/"lof_thresholds.json")
            saved_val = pd.read_parquet(sub/"lof_validation_scores.parquet")
            check(prefix+": LOF-Validierung nur auf unbekannten Validierungseinheiten", saved_val.record_key.tolist() == validation.record_key.tolist())
            saved_lof = pd.read_parquet(sub/"lof_cases.parquet")
            expected_lof_cases = []
            models = {}
            for representation in ("attributes", "latent", "pca"):
                models[representation] = fit_lof(train_matrices[representation], protocol["lof_neighbors"])
                val_scores = score_lof(models[representation], val_matrices[representation])
                np.testing.assert_allclose(val_scores, saved_val[representation], rtol=1e-10, atol=1e-10)
                check(prefix+"/"+representation+": Schwelle nur aus normaler Validierung", validation_threshold(val_scores) == thresholds[representation])
            with np.load(sub/"lof_representation"/"test.npz", allow_pickle=False) as matrices:
                for variant in ("clean", "code_swap", "signal_swap"):
                    expected_labels = labels.copy()
                    if variant == "clean": expected_labels["is_error"] = False
                    for representation in ("attributes", "latent", "pca"):
                        matrix = matrices[variant+"__"+representation]
                        check(prefix+"/"+variant+"/"+representation+": Testmatrix endlich", len(matrix) == len(test) and np.isfinite(matrix).all())
                        scores = score_lof(models[representation], matrix)
                        threshold = thresholds[representation]
                        cases = lof_cases(test.record_key, scores, threshold, protocol["review_budget_fraction"])
                        expected_lof_cases.append(cases.assign(variant=variant, representation=representation))
                        lof_metrics.append({"seed": seed, "scenario": scenario, "variant": variant,
                            "representation": representation, **summarize_lof(cases, expected_labels, threshold)})
            assert_same_cases(pd.concat(expected_lof_cases, ignore_index=True), saved_lof)
            check(prefix+": LOF-Ränge, Budget und Kennzahlen unabhängig nachgerechnet", True)
    check("Alle Regelkennzahlen stimmen", rule_metrics == read_json(out/"rule_metrics.json"))
    check("Alle LOF-Kennzahlen stimmen", lof_metrics == read_json(out/"lof_metrics.json"))
    check("Alle Nullkontrollen dokumentiert", null_checks == read_json(out/"null_control.json"))
    check("Vergleiche umfassen auch negative Ergebnisse", compare_methods(rule_metrics, lof_metrics) == read_json(out/"paired_comparisons.json"))
    summary = read_json(out/"summary.json")
    check("Zusammenfassung entspricht vollständigen Läufen", summary["autoencoders_trained"] == trained_models and
          summary["rule_comparisons"] == len(rule_metrics) and summary["lof_comparisons"] == len(lof_metrics) and
          summary["null_control_rejected_everywhere"] == all(c["null_control_rejected"] for c in null_checks))
    return {"passed": True, "check_count": len(checks), "checks": checks,
            "scope": "saved_data_rule_replay_and_lof_refit_on_saved_train_matrices; no_autoencoder_retraining; no_browser_automation"}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from run_controlled_comparison import bootstrap
    bootstrap(args.project)
    from experiment_io import read_verified_run
    from review_common import read_json, file_sha256
    out = args.run.resolve(strict=True)
    manifest = read_json(out/"run_manifest.json")
    if manifest.get("experiment") != "controlled_synthetic_rules_and_lof_13":
        raise ValueError("Kein abgeschlossener Paket-13-Lauf.")
    read_verified_run(out, list(manifest["output_sha256"]))
    protocol = read_json(out/"protocol.json")
    for name, expected in protocol["code_sha256_before"].items():
        if not Path(name).is_file() or file_sha256(name) != expected:
            raise ValueError("Aufgezeichneter Quellcode hat sich geändert: "+name)
    print(json.dumps(verify_content(out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
