"""Kontrollierter Attributversuch mit bestehendem Framework-Code, ohne Installation."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys


def bootstrap(project):
    sibling = Path(__file__).resolve().parent.parent / "mehrheitsvergleich"
    if not (sibling/"run_baseline_check.py").is_file():
        raise ValueError("Paket 13 neben analysepaket_12_mehrheitsvergleich entpacken; Pakete 10/11/12 werden benötigt.")
    sys.path.insert(0, str(sibling))
    from run_baseline_check import bootstrap as previous
    source, adapter, package11 = previous(project)
    return source, adapter, package11, sibling


def parameters():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/kontrollvergleich_13"))
    parser.add_argument("--units", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.units < 1000 or args.units % 100 or args.epochs < 1:
        raise ValueError("--units mindestens 1000 und durch 100 teilbar; --epochs mindestens 1.")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s >= 2**32-1 for s in args.seeds):
        raise ValueError("Unterschiedliche gültige Zufallsseeds benötigt.")
    return args


def train_representation(train, validation, features, cfg, folder):
    import joblib
    import numpy as np
    import torch
    from comparison_core import fit_preprocessor, transform, encode
    from latentspace.main_v2 import train_model
    from review_common import write_json
    folder.mkdir(parents=True)
    prep, x_train, contract = fit_preprocessor(train, features)
    x_validation, unknown = transform(prep, validation, contract)
    with (folder/"training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        trained = train_model(train[features], cfg, prepared_tabular={"X_tab": x_train,
            "preprocessor": prep, "numeric_cols": contract["numeric"], "categorical_cols": contract["categorical"]})
    history = trained["training_history"]
    if len(history) != cfg.epochs or not all(np.isfinite(list(row.values())).all() for row in history) or trained["used_string_cols"]:
        raise AssertionError("Training unvollständig/nicht endlich oder unerwartete Textmerkmale.")
    z_train, z_validation = encode(trained["model"], x_train), encode(trained["model"], x_validation)
    write_json(folder/"contract.json", {**contract, "train_keys": train.record_key.tolist(),
        "validation_keys": validation.record_key.tolist(), "validation_unknown": unknown,
        "latent_dim": cfg.latent_dim, "fit_scope": "training_only", "geometry_used": False})
    write_json(folder/"training_history.json", history)
    write_json(folder/"model_config.json", asdict(cfg))
    joblib.dump(prep, folder/"preprocessor.joblib")
    torch.save(trained["model"].state_dict(), folder/"autoencoder.pt")
    write_json(folder/"architecture.json", {"tab_input_dim": x_train.shape[1],
        "vocab_size": trained["tokenizer"].vocab_size, "num_string_fields": 0})
    np.savez_compressed(folder/"development.npz", attribute_train=x_train, attribute_validation=x_validation,
                        latent_train=z_train, latent_validation=z_validation)
    return {"prep": prep, "contract": contract, "model": trained["model"], "x_train": x_train,
            "x_validation": x_validation, "z_train": z_train, "z_validation": z_validation}


def main():
    args = parameters()
    project = args.project.resolve(strict=True)
    source, adapter, package11, package12 = bootstrap(project)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA
    from comparison_core import Protocol, METHODS, fit_router, route, learn_method, evaluate_selected, transform, encode, targets_from_training
    from baseline_core import BASELINE, majority_model, baseline_cases, classify_rules
    from latentspace.config_latent import LatentV2Config
    from experiment_io import new_output, finish_output
    from review_common import write_json, read_json, file_sha256, utc_now
    from synthetic_data import TARGET, CONTROL, SIGNALS, SCENARIOS, create_population, scenario_frame, make_test_variants, normal_mask
    from evaluation import fit_lof, score_lof, validation_threshold, lof_cases, summarize_lof, summarize_predictions, compare_methods
    from report_controlled import render_report
    from verify_controlled_comparison import verify_content

    torch.set_num_threads(min(4, torch.get_num_threads()))
    package = Path(__file__).resolve().parent
    p = Protocol(epochs=args.epochs, latent_dim=3)
    output = args.output if args.output.is_absolute() else project/args.output
    code_paths = sorted(set([*source.rglob("*.py"), *package11.rglob("*.py"), *package12.rglob("*.py"),
                             *package.rglob("*.py"), adapter/"attribute_prep.py", package/"PROTOKOLL.md"]))
    initial_code = {str(f.resolve()): file_sha256(f) for f in code_paths}
    out = new_output(output)
    try:
        protocol = {"version": "13.1", "created_before_training": utc_now(), "seeds": args.seeds,
            "units_per_seed": args.units, "observations_per_unit": 2, "scenarios": list(SCENARIOS),
            "rule_parameters": asdict(p), "normal_profile_weights": {"A": .7, "B": .25, "C": .05},
            "primary_target": TARGET, "null_control_target": CONTROL,
            "rules_inputs": "four_measurements_and_optional_12_nuisance_columns; neither_target_nor_null_control",
            "lof_inputs": "same_measurements_plus_observed_pruefcode; null_control_excluded",
            "negative_control": "paired_balanced_K0_K1_at_identical_predictors; never_feature",
            "split": "60/20/20_entity_groups_stratified_by_synthetic_profile; no_group_overlap",
            "rule_selection": "training_and_validation_only; all_selections_frozen_before_test",
            "lof_neighbors": 20, "lof_mode": "novelty_train_only; minus_score_samples_on_unseen_groups",
            "review_budget_fraction": .10, "threshold_validation_tail_fraction": .05,
            "lof_representations": ["attributes", "pca", "latent"], "pca_dim": 3,
            "perturbed_test_group_fraction": .20, "errors_per_affected_group": 1,
            "oracle": "synthetic_intervals_and_profile_code; never_used_by_learning_functions",
            "geometry_used": False, "real_error_detection_measured": False, "hyperparameter_search": False,
            "code_sha256_before": initial_code,
            "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html"}
        write_json(out/"protocol.json", protocol)
        (out/"PROTOKOLL.md").write_bytes((package/"PROTOKOLL.md").read_bytes())
        bundles, frozen_files = [], {}
        for seed in args.seeds:
            seed_dir = out/f"seed_{seed}"
            seed_dir.mkdir()
            attributes, truth = create_population(seed, args.units)
            attributes.to_parquet(seed_dir/"attributes.parquet", index=False)
            truth.to_parquet(seed_dir/"reference.parquet", index=False)
            if not normal_mask(attributes).all() or not attributes.record_key.is_unique or truth.groupby("group_id").partition.nunique().max() != 1:
                raise AssertionError("Generator oder Gruppenaufteilung ungültig.")
            groups = truth.set_index("record_key").group_id
            for scenario in SCENARIOS:
                sub = seed_dir/scenario
                sub.mkdir()
                frame, features = scenario_frame(attributes, scenario)
                train = frame.loc[truth.partition.eq("train")].reset_index(drop=True)
                validation = frame.loc[truth.partition.eq("validation")].reset_index(drop=True)
                targets = targets_from_training(train, [*features, TARGET, CONTROL])
                if TARGET not in targets or CONTROL not in targets:
                    raise AssertionError("Trainingsbasierte Zielerkennung erkennt die Benchmarkziele nicht.")
                write_json(sub/"schema.json", {"detected_targets": targets, "evaluated_targets": [TARGET, CONTROL],
                    "primary_target_predeclared": True, "predictors": features, "features_chosen_from_training": True})
                cfg = LatentV2Config(epochs=p.epochs, random_state=seed, latent_dim=p.latent_dim,
                    string_columns=(), known_numeric_columns=(), known_date_columns=(), ignore_columns=(),
                    enable_delivery_memory=False, parquet_path=str(seed_dir/"attributes.parquet"), output_root_dir=str(sub))
                print(f"[Training {seed}/{scenario}] Regelraum und separater LOF-Raum ({p.epochs} Epochen) ...", flush=True)
                rule_rep = train_representation(train, validation, features, cfg, sub/"rule_representation")
                lof_rep = train_representation(train, validation, [*features, TARGET], cfg, sub/"lof_representation")
                routers, train_routes, validation_routes = {}, {"global": np.zeros(len(train), dtype=int)}, {"global": np.zeros(len(validation), dtype=int)}
                router_audits = {}
                for method, x, val in (("attribute_clusters", rule_rep["x_train"], rule_rep["x_validation"]),
                                       ("latent_clusters", rule_rep["z_train"], rule_rep["z_validation"])):
                    router, labels, audit = fit_router(x, p)
                    val_labels, strengths = route(router, val)
                    routers[method], train_routes[method], validation_routes[method] = router, labels, val_labels
                    router_audits[method] = {**audit, "validation_noise": int((val_labels < 0).sum())}
                    joblib.dump(router, sub/f"{method}_router.joblib")
                pd.DataFrame({"record_key": train.record_key, **train_routes}).to_parquet(sub/"training_routes.parquet", index=False)
                pd.DataFrame({"record_key": validation.record_key, **validation_routes}).to_parquet(sub/"validation_routes.parquet", index=False)
                write_json(sub/"routing_audit.json", router_audits)
                selected, candidates, reports = {}, [], []
                for target in (TARGET, CONTROL):
                    for method in METHODS:
                        found, audits = learn_method(train, validation, train_routes[method], validation_routes[method],
                            target, features, groups, method, seed, p)
                        for rule in found:
                            rule["rule_id"] = scenario+"__"+rule["rule_id"]
                            rule["inner_rule"]["rule_id"] = rule["rule_id"]+"__inner"
                            rule["scope"]["scenario"] = scenario
                        candidates.extend(found)
                        reports.extend(audits)
                        selected[(target, method)] = [r for r in found if r["selected_on_validation"]]
                write_json(sub/"candidates.json", candidates)
                write_json(sub/"selected_rules.json", [r for v in selected.values() for r in v])
                write_json(sub/"model_reports.json", reports)
                write_json(sub/"rule_classification.json", classify_rules(candidates))
                majority = majority_model(train, TARGET)
                write_json(sub/"majority_model.json", majority)
                pca = PCA(n_components=p.latent_dim, svd_solver="full").fit(lof_rep["x_train"])
                joblib.dump(pca, sub/"lof_representation"/"pca.joblib")
                matrices = {"attributes": (lof_rep["x_train"], lof_rep["x_validation"]),
                    "latent": (lof_rep["z_train"], lof_rep["z_validation"]),
                    "pca": (pca.transform(lof_rep["x_train"]), pca.transform(lof_rep["x_validation"]))}
                lof_models, thresholds, val_scores = {}, {}, {}
                for name, (x, val) in matrices.items():
                    lof_models[name] = fit_lof(x, protocol["lof_neighbors"])
                    val_scores[name] = score_lof(lof_models[name], val)
                    thresholds[name] = validation_threshold(val_scores[name])
                    joblib.dump(lof_models[name], sub/"lof_representation"/f"lof_{name}.joblib")
                pd.DataFrame({"record_key": validation.record_key, **val_scores}).to_parquet(sub/"lof_validation_scores.parquet", index=False)
                np.savez_compressed(sub/"pca_development.npz", train=matrices["pca"][0], validation=matrices["pca"][1])
                write_json(sub/"lof_thresholds.json", thresholds)
                for path in sub.rglob("*"):
                    if path.is_file(): frozen_files[path.relative_to(out).as_posix()] = file_sha256(path)
                bundles.append({"seed": seed, "scenario": scenario, "sub": sub, "frame": frame, "truth": truth,
                    "train": train, "validation": validation, "groups": groups, "features": features,
                    "rule_rep": rule_rep, "lof_rep": lof_rep, "routers": routers, "selected": selected,
                    "majority": majority, "pca": pca, "lof_models": lof_models, "thresholds": thresholds})
                print("  Gewählte Regeln: "+", ".join(f"{m}={len(selected[(TARGET,m)])}" for m in METHODS)+
                      "; Nullkontrolle="+str(sum(len(selected[(CONTROL,m)]) for m in METHODS)), flush=True)
            for name in ("attributes.parquet", "reference.parquet"):
                frozen_files[(seed_dir/name).relative_to(out).as_posix()] = file_sha256(seed_dir/name)
        write_json(out/"selection_lock.json", {"created_before_first_test_evaluation": utc_now(),
            "protocol_sha256": file_sha256(out/"protocol.json"), "frozen_sha256": frozen_files})
        rule_metrics, lof_metrics, null_checks, safeguards = [], [], [], []
        for bundle in bundles:
            seed, scenario, sub = bundle["seed"], bundle["scenario"], bundle["sub"]
            print(f"[Test {seed}/{scenario}] Eingefrorene Modelle auf unbekannten Einheiten ...", flush=True)
            test = bundle["frame"].loc[bundle["truth"].partition.eq("test")].reset_index(drop=True)
            variants, labels = make_test_variants(bundle["train"], test, bundle["truth"], seed)
            labels.to_parquet(sub/"test_labels.parquet", index=False)
            for name, frame in variants.items(): frame.to_parquet(sub/f"test_{name}.parquet", index=False)
            rule_rep = bundle["rule_rep"]
            x_test = transform(rule_rep["prep"], test, rule_rep["contract"])[0]
            z_test = encode(rule_rep["model"], x_test)
            changed_x = transform(rule_rep["prep"], variants["code_swap"], rule_rep["contract"])[0]
            if not np.array_equal(changed_x, x_test):
                raise AssertionError("Zieländerung beeinflusst den Regelrouter.")
            routes = {"global": np.zeros(len(test), dtype=int)}
            for method in ("attribute_clusters", "latent_clusters"):
                routes[method] = route(bundle["routers"][method], x_test if method == "attribute_clusters" else z_test)[0]
            pd.DataFrame({"record_key": test.record_key, **routes}).to_parquet(sub/"test_routes.parquet", index=False)
            np.savez_compressed(sub/"rule_representation"/"test.npz", attributes=x_test, latent=z_test)
            safeguards.append({"seed": seed, "scenario": scenario, "target_not_in_rule_predictors": True,
                "target_mutation_keeps_rule_representation": True, "test_groups_not_seen_during_fit": True})
            all_rule_cases = []
            for name in ("clean", "code_swap"):
                truth = labels.copy()
                if name == "clean": truth["is_error"] = False
                for method in (*METHODS, BASELINE):
                    if method == BASELINE:
                        cases = baseline_cases(variants[name], TARGET, bundle["groups"], seed, bundle["majority"])
                    else:
                        cases, _ = evaluate_selected(variants[name], routes[method], bundle["selected"][(TARGET,method)],
                            TARGET, bundle["groups"], method, seed, p)
                    metric = summarize_predictions(cases, truth)
                    rule_metrics.append({"seed": seed, "scenario": scenario, "variant": name, "method": method, **metric})
                    all_rule_cases.append(cases.assign(variant=name))
            pd.concat(all_rule_cases, ignore_index=True).to_parquet(sub/"rule_cases.parquet", index=False)
            for method in METHODS:
                control_rules = bundle["selected"][(CONTROL,method)]
                null_checks.append({"seed": seed, "scenario": scenario, "method": method,
                    "selected_control_rules": len(control_rules), "null_control_rejected": not control_rules})
            all_lof_cases, lof_test_arrays = [], {}
            for name, frame in variants.items():
                rep = bundle["lof_rep"]
                x = transform(rep["prep"], frame, rep["contract"])[0]
                z = encode(rep["model"], x)
                matrices = {"attributes": x, "latent": z, "pca": bundle["pca"].transform(x)}
                truth = labels.copy()
                if name == "clean": truth["is_error"] = False
                for representation, matrix in matrices.items():
                    scores = score_lof(bundle["lof_models"][representation], matrix)
                    threshold = bundle["thresholds"][representation]
                    cases = lof_cases(frame.record_key, scores, threshold, protocol["review_budget_fraction"])
                    metric = summarize_lof(cases, truth, threshold)
                    lof_metrics.append({"seed": seed, "scenario": scenario, "variant": name,
                        "representation": representation, **metric})
                    all_lof_cases.append(cases.assign(variant=name, representation=representation))
                    lof_test_arrays[name+"__"+representation] = matrix
            pd.concat(all_lof_cases, ignore_index=True).to_parquet(sub/"lof_cases.parquet", index=False)
            np.savez_compressed(sub/"lof_representation"/"test.npz", **lof_test_arrays)
        write_json(out/"rule_metrics.json", rule_metrics)
        write_json(out/"lof_metrics.json", lof_metrics)
        write_json(out/"null_control.json", null_checks)
        write_json(out/"safeguards.json", safeguards)
        paired = compare_methods(rule_metrics, lof_metrics)
        write_json(out/"paired_comparisons.json", paired)
        write_json(out/"summary.json", {"benchmark_completed": True, "seeds": args.seeds, "scenarios": list(SCENARIOS),
            "units_per_seed": args.units, "rows_per_seed": args.units*2, "epochs": args.epochs,
            "autoencoders_trained": len(bundles)*2, "latent_dimensions": p.latent_dim,
            "rule_comparisons": len(rule_metrics), "lof_comparisons": len(lof_metrics),
            "null_control_rejected_everywhere": all(c["null_control_rejected"] for c in null_checks),
            "positive_latent_result_required_to_pass": False, "geometry_used": False,
            "real_dop_or_alkis_performance_measured": False, "parameter_search_performed": False,
            "source_code_unchanged": True, "prior_runs_unchanged": True,
            "interpretation": "synthetic_generator_ground_truth; no_automatic_general_superiority_claim"})
        print("[Nachprüfung] Daten, ausgewählte Regeln, LOF und Kennzahlen nachrechnen ...", flush=True)
        verification = verify_content(out)
        write_json(out/"verification.json", verification)
        render_report(out)
        for name, expected in initial_code.items():
            if file_sha256(name) != expected:
                raise AssertionError("Quellcode wurde während des Versuchs geändert: "+name)
        for name, expected in frozen_files.items():
            if file_sha256(out/name) != expected:
                raise AssertionError("Eingefrorenes Artefakt wurde verändert: "+name)
        finish_output(out, {"experiment": "controlled_synthetic_rules_and_lof_13",
            "package_sha256": {f.relative_to(package).as_posix(): file_sha256(f) for f in package.rglob("*.py")},
            "protocol_sha256": file_sha256(out/"protocol.json")})
        print(json.dumps(read_json(out/"summary.json"), ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out/'bericht.html'}")
    except Exception as exc:
        write_json(out/"run_status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        raise SystemExit(f"Vergleich abgebrochen: {exc}")
