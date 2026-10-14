"""Start im bestehenden Framework; benötigt keinen Installer und ersetzt kein src."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser(description="Globale, attributlokale und latentlokale Regeln vergleichen.")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/regelvergleich_11"))
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return parser.parse_args()


def bootstrap(project):
    project = project.resolve(strict=True)
    source = project / "src"
    required = ("phase2_ml.py", "tree_rule_export.py", "generic_schema.py", "experiment_io.py",
                "latentspace/main_v2.py", "phase2_split.py")
    missing = [name for name in required if not (source/name).is_file()]
    adapter = project / "adapters" / "alkis_bahnverkehr"
    if missing or not (adapter/"attribute_prep.py").is_file():
        raise ValueError(f"Vorhandenes Framework mit Paket 07/08 und daneben Paket 10 benötigt. "
                         f"Fehlende src-Dateien: {missing}; Adapter: {adapter / 'attribute_prep.py'}")
    # Der bestehende Autoencoder besitzt noch lokale Imports wie
    # "from tabular_encoder import ...". Gleiches Suchverzeichnis wie sein
    # regulärer Programmeinstieg, ohne Dateien im Projekt umzuschreiben.
    sys.path.insert(0, str(source / "latentspace"))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(adapter))
    return source, adapter


def main():
    args = arguments()
    if args.epochs < 1 or args.rows < 0 or (args.rows and args.rows < 100):
        raise ValueError("Mindestens eine Epoche; --rows 0 oder mindestens 100.")
    if len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s < 2**32-1 for s in args.seeds):
        raise ValueError("Unterschiedliche Seeds im Bereich 0 bis 2**32-2 verwenden.")
    source_dir, adapter_dir = bootstrap(args.project)
    print("[01 Vorbereitung] Lade Framework und Vergleichsprotokoll ...", flush=True)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from attribute_prep import read_attributes, prepare_attributes, select_sample, ATTRIBUTES, DATE_MISSING
    from generic_schema import attach_identity, choose_groups
    from phase2_settings import Phase2Settings
    from phase2_split import make_grouped_split, select_partition
    from latentspace.config_latent import LatentV2Config
    from latentspace.main_v2 import train_model
    from experiment_io import new_output, finish_output
    from review_common import write_json, read_json, file_sha256, digest, utc_now
    from comparison_core import (Protocol, METHODS, model_table, eligible_columns, targets_from_training,
        predictors_for_target, fit_preprocessor, transform, encode, fit_router, route,
        learn_method, evaluate_selected, summarize_cases, paired_comparisons, DATE)
    from report_comparison import render_report
    from verify_rule_comparison import verify_content

    torch.set_num_threads(min(4, torch.get_num_threads()))
    package = Path(__file__).resolve().parent
    source = args.input.resolve(strict=True)
    p = Protocol(epochs=args.epochs)
    code_paths = [*source_dir.rglob("*.py"), *package.rglob("*.py"), adapter_dir/"attribute_prep.py", package/"PROTOKOLL.md"]
    initial_code = {str(f.resolve()): file_sha256(f) for f in sorted(code_paths)}
    source_hash = file_sha256(source)
    raw, read_audit = read_attributes(source)
    sample = select_sample(raw, args.rows, 42)
    attributes = prepare_attributes(sample)
    identified, identity_info = attach_identity(attributes, ("gml_id",))
    grouped, group_info = choose_groups(identified)
    frame = model_table(grouped)
    allowed = [c for c in (*ATTRIBUTES, DATE_MISSING) if c in frame]
    out = new_output(args.output)
    try:
        sample.to_parquet(out/"source_attributes.parquet", index=False)
        identified.to_parquet(out/"attributes_with_ids.parquet", index=False)
        frame.to_parquet(out/"model_table.parquet", index=False)
        protocol = {"protocol_version": "11.1", "created_before_training": utc_now(),
            "parameters": asdict(p), "seeds": args.seeds, "sample_seed": 42,
            "source_sha256": source_hash, "source_rows": len(raw), "sample_rows": len(sample),
            "geometry_used": False, "string_encoder_used": False,
            "representation_variant": "tabular_only_existing_autoencoder",
            "fit_scope": "training_partition_per_target", "routing": "frozen_hdbscan_approximate_predict",
            "target_selection": "automatic_training_only", "tree_parameters": "fixed_no_search",
            "test_role": "one_internal_evaluation_after_all_selections_frozen",
            "sample_keys_sha256": digest(sorted(identified.record_key)),
            "identity": identity_info, "grouping": group_info, "attribute_reading": read_audit,
            "allowed_attribute_columns": allowed, "code_sha256_before": initial_code,
            "known_limitations": ["previously_explored_dataset", "exact_duplicates_not_independent_deliveries",
                "no_error_ground_truth", "repeated_splits_not_independent_samples", "fixed_5_epoch_default_not_optimized"]}
        write_json(out/"protocol.json", protocol)
        (out/"PROTOKOLL.md").write_bytes((package/"PROTOKOLL.md").read_bytes())
        protocol_hash = file_sha256(out/"protocol.json")
        # Bündel enthalten nur Train-/Validierungsresultate. Test folgt gesammelt.
        bundles = []
        seed_info = []
        selected_paths = {}
        all_candidates, all_reports = [], []
        for seed in args.seeds:
            seed_dir = out/f"seed_{seed}"
            seed_dir.mkdir()
            settings = Phase2Settings(group_columns=("__generic_group",), random_state=seed)
            split = make_grouped_split(frame, settings)
            split.to_csv(seed_dir/"split.csv", index=False)
            train, validation = (select_partition(frame, split, part) for part in ("train", "validation"))
            groups = split.set_index("record_key").group_id
            columns, excluded = eligible_columns(train, allowed)
            targets = targets_from_training(train, columns)
            info = {"seed": seed, "targets": targets, "training_columns": columns, "excluded_columns": excluded,
                "partitions": {part: {"rows": int((split.partition == part).sum()),
                    "groups": int(split.loc[split.partition.eq(part), "group_id"].nunique())}
                    for part in ("train", "validation", "test")}}
            if not targets:
                raise ValueError(f"Seed {seed}: keine Klassifikationsziele im Training; Vergleich nicht möglich.")
            write_json(seed_dir/"selection_schema.json", info)
            seed_info.append(info)
            print(f"[Seed {seed}] {len(train)} Training, {len(validation)} Validierung; {len(targets)} Ziele", flush=True)
            for target_no, target in enumerate(targets, 1):
                sub = seed_dir/f"target_{target_no:02d}"
                sub.mkdir()
                features, target_exclusions = predictors_for_target(train, columns, target)
                if not features:
                    raise ValueError(f"{target}: keine vom Ziel getrennten Merkmale; Vergleich nicht möglich.")
                prep, x_train, schema = fit_preprocessor(train, features)
                x_val, val_unknown = transform(prep, validation, schema)
                if not np.array_equal(transform(prep, train, schema)[0], x_train):
                    raise AssertionError("Training und spätere Transformation stimmen nicht überein.")
                cfg = LatentV2Config(epochs=p.epochs, random_state=seed, latent_dim=p.latent_dim,
                    string_columns=(), known_numeric_columns=(), known_date_columns=(), ignore_columns=(),
                    enable_delivery_memory=False, parquet_path=str(source), output_root_dir=str(sub))
                print(f"  [{target_no}/{len(targets)}] {target}: Autoencoder und zwei Clusterungen ...", flush=True)
                with (sub/"training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                    trained = train_model(train[features], cfg, prepared_tabular={"X_tab": x_train,
                        "preprocessor": prep, "numeric_cols": schema["numeric"], "categorical_cols": schema["categorical"]})
                history = trained["training_history"]
                if len(history) != p.epochs or not all(np.isfinite(list(h.values())).all() for h in history):
                    raise ValueError("Unvollständiges/nicht endliches Training.")
                z_train, z_val = encode(trained["model"], x_train), encode(trained["model"], x_val)
                if trained["used_string_cols"]:
                    raise AssertionError("Unerwarteter Zeichenencoder im kontrollierten Vergleich.")
                routers, routing_train, routing_val, routing_audit = {}, {}, {}, {}
                routing_train["global"] = np.zeros(len(train), dtype=int)
                routing_val["global"] = np.zeros(len(validation), dtype=int)
                for method, matrix_train, matrix_val in (("attribute_clusters", x_train, x_val),
                                                        ("latent_clusters", z_train, z_val)):
                    router, labels_train, audit = fit_router(matrix_train, p)
                    labels_val, strengths_val = route(router, matrix_val)
                    routers[method] = router
                    routing_train[method], routing_val[method] = labels_train, labels_val
                    audit.update(validation_noise=int((labels_val < 0).sum()),
                                 training_label_semantics="hdbscan_fit_labels", validation_label_semantics="approximate_predict")
                    routing_audit[method] = audit
                    joblib.dump(router, sub/f"{method}_router.joblib")
                    pd.DataFrame({"record_key": validation.record_key, "cluster": labels_val,
                                  "strength": strengths_val}).to_parquet(sub/f"{method}_validation_routes.parquet", index=False)
                target_candidates, model_reports, selected = [], [], {}
                for method in METHODS:
                    candidates, reports = learn_method(train, validation, routing_train[method], routing_val[method],
                        target, features, groups, method, seed, p)
                    selected[method] = [r for r in candidates if r["selected_on_validation"]]
                    target_candidates.extend(candidates)
                    model_reports.extend(reports)
                selected_path = sub/"selected_rules.json"
                write_json(selected_path, [r for method in METHODS for r in selected[method]])
                selected_paths[selected_path.relative_to(out).as_posix()] = file_sha256(selected_path)
                write_json(sub/"candidate_rules.json", target_candidates)
                write_json(sub/"model_reports.json", model_reports)
                write_json(sub/"training_history.json", history)
                write_json(sub/"model_config.json", asdict(cfg))
                write_json(sub/"feature_contract.json", {**schema, "target": target, "target_exclusions": target_exclusions,
                    "train_record_keys": train.record_key.tolist(), "validation_record_keys": validation.record_key.tolist(),
                    "validation_unknown_categories": val_unknown, "fit_scope": "training_only", "geometry_used": False})
                write_json(sub/"routing_audit.json", routing_audit)
                joblib.dump(prep, sub/"preprocessor.joblib")
                torch.save(trained["model"].state_dict(), sub/"autoencoder.pt")
                write_json(sub/"autoencoder_architecture.json", {"config": asdict(cfg), "tab_input_dim": x_train.shape[1],
                    "vocab_size": trained["tokenizer"].vocab_size, "num_string_fields": 0})
                np.savez_compressed(sub/"development_representations.npz", attribute_train=x_train,
                    latent_train=z_train, attribute_validation=x_val, latent_validation=z_val)
                train_routes = pd.DataFrame({"record_key": train.record_key, **routing_train})
                train_routes.to_parquet(sub/"training_routes.parquet", index=False)
                bundles.append({"seed": seed, "target": target, "sub": sub, "split": split, "groups": groups,
                    "schema": schema, "prep": prep, "model": trained["model"], "routers": routers,
                    "selected": selected, "target_exclusions": target_exclusions})
                all_candidates.extend(target_candidates)
                all_reports.extend([{**r, "seed": seed} for r in model_reports])
                print("    Auswahl: " + ", ".join(f"{m}={len(selected[m])}" for m in METHODS), flush=True)
        write_json(out/"all_candidates.json", all_candidates)
        write_json(out/"all_model_reports.json", all_reports)
        lock = {"phase": "all_rules_frozen_before_first_test_transformation", "created_at": utc_now(),
            "protocol_sha256": protocol_hash, "selected_files_sha256": selected_paths,
            "candidate_file_sha256": file_sha256(out/"all_candidates.json")}
        write_json(out/"selection_lock.json", lock)
        print("[02 Test] Auswahl aller Ziele und Seeds eingefroren. Zurückgehaltene Fälle prüfen ...", flush=True)
        all_cases, all_test_audits, leak_checks = [], [], []
        for bundle in bundles:
            seed, target, sub = bundle["seed"], bundle["target"], bundle["sub"]
            test = select_partition(frame, bundle["split"], "test")
            x_test, unknown = transform(bundle["prep"], test, bundle["schema"])
            z_test = encode(bundle["model"], x_test)
            probe = test.copy(deep=True)
            probe[target] = "__TARGET_PROBE_NEVER_USED__"
            if target in {DATE, DATE_MISSING}:
                if DATE in probe: probe[DATE] = 999999.
                if DATE_MISSING in probe: probe[DATE_MISSING] = "__TARGET_FAMILY_PROBE__"
            probe_matrix = transform(bundle["prep"], probe, bundle["schema"])[0]
            if not np.array_equal(x_test, probe_matrix):
                raise AssertionError("Zielwerte beeinflussen die Testrepräsentation.")
            write_json(sub/"test_transformation.json", {"unknown_categories": unknown,
                "test_record_keys": test.record_key.tolist(), "target_mutation_leaves_matrix_unchanged": True})
            test_routes = pd.DataFrame({"record_key": test.record_key})
            for method in METHODS:
                if method == "global":
                    labels, strengths = np.zeros(len(test), dtype=int), np.ones(len(test))
                else:
                    matrix = x_test if method == "attribute_clusters" else z_test
                    labels, strengths = route(bundle["routers"][method], matrix)
                test_routes[method] = labels
                test_routes[method+"_strength"] = strengths
                cases, audits = evaluate_selected(test, labels, bundle["selected"][method], target,
                    bundle["groups"], method, seed, p)
                probe_cases, _ = evaluate_selected(probe, labels, bundle["selected"][method], target,
                    bundle["groups"], method, seed, p)
                invariant = cases[["prediction", "covered", "conflict", "rule_ids"]].equals(
                    probe_cases[["prediction", "covered", "conflict", "rule_ids"]])
                if not invariant:
                    raise AssertionError("Zielmutation verändert Vorhersage oder Anwendungsbereich.")
                leak_checks.append({"seed": seed, "target": target, "method": method, "passed": True})
                all_cases.append(cases)
                all_test_audits.extend([{**a, "seed": seed, "target": target, "method": method} for a in audits])
            test_routes.to_parquet(sub/"test_routes.parquet", index=False)
            np.savez_compressed(sub/"test_representations.npz", attribute_test=x_test, latent_test=z_test)
        cases = pd.concat(all_cases, ignore_index=True)
        cases.to_parquet(out/"test_cases.parquet", index=False)
        write_json(out/"test_rule_evidence.json", all_test_audits)
        write_json(out/"target_leakage_checks.json", leak_checks)
        metrics, pairs = [], []
        for seed in args.seeds:
            part = cases.loc[cases.seed.eq(seed)]
            for method in METHODS:
                selected = [r for r in all_candidates if r["scope"]["seed"] == seed and
                    r["scope"]["method"] == method and r["selected_on_validation"]]
                checked = [a for a in all_test_audits if a["seed"] == seed and a["method"] == method]
                assessments = {name: sum(a["test_assessment"] == name for a in checked) for name in
                    ("supported", "insufficient_support", "agreement_threshold_not_met")}
                metrics.append({"seed": seed, "method": method,
                    **summarize_cases(part.loc[part.method.eq(method)]), "selected_rule_count": len(selected),
                    "test_rule_assessments": assessments})
            pairs.extend([{**pair, "seed": seed} for pair in paired_comparisons(part)])
        target_metrics = [{"seed": int(seed), "method": method, "target": target, **summarize_cases(part)}
            for (seed, method, target), part in cases.groupby(["seed", "method", "target"])]
        write_json(out/"target_metrics.json", target_metrics)
        summary = {"comparison_completed": True, "source_rows": len(raw), "sample_rows": len(sample),
            "geometry_used": False, "independent_external_evaluation": False,
            "real_error_detection_measured": False, "latent_benefit_automatically_claimed": False,
            "protocol": asdict(p), "splits": seed_info, "method_metrics": metrics,
            "paired_comparisons": pairs, "target_leakage_checks_passed": len(leak_checks),
            "source_unchanged": source_hash == file_sha256(source),
            "code_unchanged": all(file_sha256(Path(path)) == value for path, value in initial_code.items())}
        if not summary["source_unchanged"] or not summary["code_unchanged"]:
            raise AssertionError("Original- oder Codedateien haben sich während des Laufs geändert.")
        write_json(out/"summary.json", summary)
        render_report(out, summary, all_candidates, all_test_audits)
        verification = verify_content(out)
        write_json(out/"verification.json", verification)
        if not verification["passed"]:
            raise AssertionError("Vergleichsartefakte erfüllen die technischen Prüfbedingungen nicht.")
        finish_output(out, {"experiment": "cluster_rule_comparison_11", "source": str(source),
            "input_sha256": source_hash, "package_sha256": {str(f.relative_to(package)): file_sha256(f)
                for f in sorted(package.rglob("*.py"))}, "protocol_sha256": protocol_hash})
        print(f"[03 Fertig] {verification['check_count']} technische Prüfungen bestanden.")
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}", flush=True)
        for metric in metrics:
            print(f"Seed {metric['seed']} {metric['method']}: {metric['selected_rule_count']} ausgewählt; "
                f"{metric['test_rule_assessments']['supported']} im Test gestützt; "
                f"{metric['covered']}/{metric['target_record_pairs']} Ziel-Datensatz-Paare abgedeckt.")
        return out
    except BaseException as exc:
        write_json(out/"run_status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
