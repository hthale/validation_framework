"""Gemeinsamer DOP-Test: vorgegebene/gelernte Regeln und zwei LOF-Repräsentationen."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


def bootstrap(project):
    project = Path(project).resolve(strict=True)
    required = ["generic_schema.py", "generic_rules.py", "phase2_ml.py", "phase2_split.py",
                "latent_comparison.py", "run_latent_comparison.py", "experiment_io.py"]
    missing = [str(Path("src") / f) for f in required if not (project / "src" / f).is_file()]
    if missing:
        raise ValueError(f"Projekt mit installiertem Paket 08 erforderlich; fehlend: {missing}")
    sys.path.insert(0, str(project / "src"))
    sys.path.insert(0, str(project / "src" / "latentspace"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    return project


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--input", type=Path, default=Path("data/check_01/dop_training_dataset_with_ids.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/dop_joint_check_09"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--cases-per-type", type=int, default=3)
    p.add_argument("--review-fraction", type=float, default=.05)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    return p


def source_hashes(project):
    from review_common import file_sha256
    package = Path(__file__).resolve().parent
    return {"project": {p.relative_to(project).as_posix(): file_sha256(p)
                        for p in sorted((project / "src").rglob("*.py"))},
            "package": {p.relative_to(package).as_posix(): file_sha256(p)
                        for p in sorted(package.rglob("*.py"))}}


def learn_rules(train, validation, contract, train_phase0, groups, settings):
    from generic_rules import learn_profiles
    from phase2_ml import get_phase2_target_candidates, train_dependency
    from rule_engine import validate_rules_on_full_dataframe
    profiles, profile_info = learn_profiles(train, contract, train_phase0, settings.random_state)
    for sample in profile_info["sampling"]:
        sample["scope"] = "training_partition_only"
    profile_info["scope"] = "training_partition_only"
    if any(m["evaluation_status"] == "invalid_rule" for m in profile_info["metrics"]):
        raise ValueError("Technisch ungültige Profiling-Regel im Training.")
    validation_results, _ = validate_rules_on_full_dataframe(validation, profiles)
    selected_profiles = []
    for rule, metric in zip(profiles, validation_results):
        if metric["evaluation_status"] == "invalid_rule":
            raise ValueError(f"Technisch ungültige Profiling-Regel: {rule['rule_id']}")
        eligible = (metric["evaluation_status"] == "evaluated" and metric["coverage_count"] >= 5
                    and metric["pass_rate"] is not None and metric["pass_rate"] >= .95)
        rule["metadata"].update(learning_scope="training_partition_only", evaluation_scope="validation_partition",
            approved=False, eligible_for_joint_check=bool(eligible), validation=metric,
            minimum_validation_cases=5, minimum_validation_pass_rate=.95,
            rejection_reasons=[] if eligible else ["validation_support_or_pass_rate_not_met"])
        if eligible: selected_profiles.append(rule)
    usable = [c for c, d in contract["columns"].items() if d["role"] in {"numeric", "categorical"}]
    train_ml, validation_ml = train[["record_key", *usable]], validation[["record_key", *usable]]
    targets = get_phase2_target_candidates(train_ml)
    ml_selected, candidates, reports = [], [], []
    for target in targets:
        duplicates = [c for c in usable if c != target and train_ml[c].equals(train_ml[target])]
        features = [c for c in usable if c != target and c not in duplicates]
        accepted, report = train_dependency(train_ml, validation_ml, target, features, groups, settings, .95, .90, .90)
        proposed = report.pop("candidate_rules")
        for rule in proposed:
            if rule["metadata"]["validation"]["evaluation_status"] == "invalid_rule":
                raise ValueError(f"Technisch ungültiger ML-Kandidat: {rule['rule_id']}")
            rule["metadata"].update(approved=False, learning_scope="training_partition_only")
        report["duplicate_target_columns_excluded"] = duplicates
        ml_selected.extend(accepted)
        candidates.extend(proposed)
        reports.append(report)
        print(f"  Ziel {target}: {len(proposed)} Kandidaten, {len(accepted)} ausgewählt", flush=True)
    return {"profiling_candidates": profiles, "profiling_selected": selected_profiles,
            "dependency_candidates": candidates, "dependency_selected": ml_selected,
            "profiling_training": profile_info, "dependency_reports": reports,
            "automatic_targets": targets, "test_used_for_rule_selection": False}


def analyze(args):
    project = bootstrap(args.project)
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    from experiment_io import new_output, finish_output
    from generic_schema import attach_identity, infer_columns, normalize_columns, tabular_preparation
    from joint_protocol import (PROTOCOL, GROUP_COLUMNS, METHOD_LABELS, require_columns, known_results,
                                inject_cases, evaluate_rules, joint_metrics)
    from phase2_settings import Phase2Settings
    from phase2_split import make_grouped_split, select_partition
    from latent_comparison import transform_frozen, baseline_matrix, source_columns, novel_lof, evaluate_pair
    from latentspace.config_latent import LatentV2Config
    from latentspace.main_v2 import train_model, encode_string_columns, compute_hdbscan_raw
    from latentspace.anomaly_scores import SCORE_SCHEMA_VERSION
    from run_latent_comparison import latent_encode
    from review_common import file_sha256, write_json
    from report_joint import render

    if min(args.epochs, args.cases_per_type, args.n_neighbors, args.threads) < 1 or not 0 < args.review_fraction <= 1:
        raise ValueError("Positive Epochen/Fallzahlen/Nachbarn/Threads und ein Budget in (0,1] verwenden.")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= s < 2**32-1 for s in args.seeds)):
        raise ValueError("Eindeutige Seeds zwischen 0 und 2**32-2 verwenden.")
    source = (project / args.input).resolve(strict=True)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Flache DOP-Parquet-Tabelle erforderlich.")
    input_hash = file_sha256(source)
    code_hash = source_hashes(project)
    raw, identity = attach_identity(pd.read_parquet(source))
    require_columns(raw)
    settings_by_seed = {s: Phase2Settings(group_columns=tuple(GROUP_COLUMNS), random_state=s) for s in args.seeds}
    splits = {s: make_grouped_split(raw, settings) for s, settings in settings_by_seed.items()}
    # Vollständigkeit aller acht Klassen prüfen, bevor ein Modell trainiert wird.
    prepared_cases = {s: inject_cases(select_partition(raw, split, "test"), select_partition(raw, split, "train"),
                                     args.cases_per_type, s) for s, split in splits.items()}
    torch.set_num_threads(args.threads)
    out = new_output(project / args.output)
    protocol = {**PROTOCOL, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_sha256": input_hash, "identity": identity, "group_columns": GROUP_COLUMNS,
                "test_labels_used_for_fitting": False, "code_at_start": code_hash}
    write_json(out / "protocol.json", protocol)
    raw.to_parquet(out / "input_with_ids.parquet", index=False)
    repetitions = []
    try:
        for seed, split in splits.items():
            print(f"[Seed {seed}] Gemeinsame Aufteilung und Testkopien", flush=True)
            settings = settings_by_seed[seed]
            sub = out / f"seed_{seed}"
            sub.mkdir()
            split.to_csv(sub / "split.csv", index=False)
            original_train = select_partition(raw, split, "train")
            original_validation = select_partition(raw, split, "validation")
            before_raw, after_raw, changes = prepared_cases[seed]
            contract = infer_columns(original_train, excluded=GROUP_COLUMNS, scope="training_partition_only")
            frames, issues = {}, {}
            for part, data in (("train", original_train), ("validation", original_validation),
                               ("before", before_raw), ("after", after_raw)):
                frames[part], issues[part] = normalize_columns(data, contract)
                frames[part].to_parquet(sub / f"{part}_normalized.parquet", index=False)
            before_raw.to_parquet(sub / "test_before.parquet", index=False)
            after_raw.to_parquet(sub / "test_after.parquet", index=False)
            write_json(sub / "changes.json", changes)
            write_json(sub / "normalization_issues.json", issues)
            model_frame, prepared, features = tabular_preparation(frames["train"], contract)
            features["fit_scope"] = "training_partition_only"
            contract.update(latent_features=features, group_columns=GROUP_COLUMNS)
            write_json(sub / "feature_contract.json", contract)
            cfg = LatentV2Config(parquet_path=str(source), output_root_dir=str(sub), epochs=args.epochs,
                random_state=seed, lof_n_neighbors=args.n_neighbors, string_columns=tuple(features["string_columns"]),
                known_numeric_columns=(), known_date_columns=(), ignore_columns=(), enable_delivery_memory=False)
            write_json(sub / "model_config.json", asdict(cfg))
            print(f"[Seed {seed}] Autoencoder nur auf {len(original_train)} Trainingszeilen", flush=True)
            trained = train_model(model_frame, cfg, prepared_tabular=prepared)
            if len(trained["training_history"]) != args.epochs or not all(
                    np.isfinite(list(h.values())).all() for h in trained["training_history"]):
                raise ValueError("Training ist unvollständig oder nicht endlich.")
            if list(trained["used_string_cols"]) != features["string_columns"]:
                raise ValueError("Textmerkmale stimmen nicht mit dem Trainingsvertrag überein.")
            tokenizer = trained["tokenizer"]
            matrices, baselines, latents = {}, {}, {}
            for part in ("train", "before", "after"):
                model_data, tab = transform_frozen(frames[part], prepared, features)
                if part == "train" and not np.array_equal(tab, prepared["X_tab"]):
                    raise ValueError("Eingefrorene Vorverarbeitung entspricht nicht dem Training.")
                strings, _ = encode_string_columns(model_data, cfg, tokenizer)
                matrices[part] = tab
                baselines[part] = baseline_matrix(tab, strings, tokenizer.vocab_size)
                latents[part] = latent_encode(trained["model"], tab, strings, torch)
            torch.save(trained["model"].state_dict(), sub / "autoencoder.pt")
            joblib.dump(prepared["preprocessor"], sub / "preprocessor.pkl")
            joblib.dump(tokenizer, sub / "tokenizer.pkl")
            write_json(sub / "training_history.json", trained["training_history"])
            np.savez_compressed(sub / "representations.npz", **{f"{method}_{part}": array
                for method, arrays in (("baseline", baselines), ("latent", latents)) for part, array in arrays.items()})

            cases = before_raw[["record_key"]].copy()
            mapping = {r["record_key"]: r for r in changes}
            cases["changed"] = cases.record_key.isin(mapping)
            cases["operator"] = cases.record_key.map(lambda k: mapping.get(k, {}).get("operator", "unchanged"))
            cases["column"] = cases.record_key.map(lambda k: mapping.get(k, {}).get("column", ""))
            cases["input_visible_change"] = np.any(baselines["before"] != baselines["after"], axis=1)
            lof_metrics, score_tables, models = {}, [], {}
            for method, arrays in (("baseline", baselines), ("latent", latents)):
                b, a, model, k = novel_lof(arrays["train"], arrays["before"], arrays["after"], args.n_neighbors)
                unchanged = ~cases.changed.to_numpy(bool)
                if not np.allclose(b[unchanged], a[unchanged], rtol=0, atol=1e-9):
                    raise ValueError("Score unveränderter Fälle hat sich geändert.")
                scores, metrics = evaluate_pair(cases.record_key.tolist(), b, a, changes, args.review_fraction)
                scores["method"] = method
                aligned = scores.set_index("record_key").loc[cases.record_key]
                for moment in ("before", "after"):
                    cases[method+"_"+moment] = aligned["selected_"+moment].to_numpy(bool)
                metrics.update(n_neighbors_effective=k, dimensions=arrays["train"].shape[1],
                    input_invisible_changes=int((cases.changed & ~cases.input_visible_change).sum()),
                    representation_unchanged_despite_change=int((cases.changed.to_numpy(bool)
                        & ~np.any(arrays["before"] != arrays["after"], axis=1)).sum()))
                models[method], lof_metrics[method] = model, metrics
                score_tables.append(scores)
                joblib.dump(model, sub / f"lof_{method}.pkl")
            pd.concat(score_tables, ignore_index=True).to_parquet(sub / "scores.parquet", index=False)
            cluster_labels, _, _ = compute_hdbscan_raw(latents["train"], cfg.hdbscan_min_cluster_size, cfg.hdbscan_min_samples)
            phase0 = pd.DataFrame({"record_key": frames["train"].record_key,
                "lof_raw": -models["latent"].negative_outlier_factor_, "cluster_label": cluster_labels,
                "primary_score_name": "lof_raw", "score_schema_version": SCORE_SCHEMA_VERSION})
            phase0.to_parquet(sub / "training_strata.parquet", index=False)
            print(f"[Seed {seed}] Profiling und automatische Abhängigkeiten", flush=True)
            rules = learn_rules(frames["train"], frames["validation"], contract, phase0,
                                split.set_index("record_key").group_id, settings)
            write_json(sub / "rules.json", rules)
            decisions = []
            for moment, raw_test in (("before", before_raw), ("after", after_raw)):
                for method in ("known", "profiling", "dependency"):
                    result = known_results(raw_test) if method == "known" else evaluate_rules(frames[moment], rules[method+"_selected"])
                    result["method"], result["moment"] = method, moment
                    flagged = set(result.loc[result.violation, "record_key"])
                    cases[method+"_"+moment] = cases.record_key.isin(flagged)
                    decisions.append(result)
                cases["rules_"+moment] = cases[[m+"_"+moment for m in ("known", "profiling", "dependency")]].any(axis=1)
                for name, members in (("rules_baseline", ("rules", "baseline")), ("rules_latent", ("rules", "latent")),
                                      ("all", ("rules", "baseline", "latent"))):
                    cases[name+"_"+moment] = cases[[m+"_"+moment for m in members]].any(axis=1)
            rule_cases = pd.concat(decisions, ignore_index=True)
            rule_cases.to_parquet(sub / "rule_cases.parquet", index=False)
            # Auch bei vorher schon markierten Zeilen können neue Regelverletzungen entstehen.
            violations = rule_cases.loc[rule_cases.violation]
            old_pairs = set(zip(violations.loc[violations.moment.eq("before"), "record_key"],
                                violations.loc[violations.moment.eq("before"), "rule_id"]))
            new_counts = {}
            for row in violations.loc[violations.moment.eq("after")].itertuples(index=False):
                if (row.record_key, row.rule_id) not in old_pairs:
                    new_counts[row.record_key] = new_counts.get(row.record_key, 0) + 1
            cases["new_rule_violations"] = cases.record_key.map(new_counts).fillna(0).astype(int)
            cases.to_parquet(sub / "case_results.parquet", index=False)
            cases.to_csv(sub / "case_results.csv", index=False, encoding="utf-8-sig")
            counts = {p: {"rows": int(split.partition.eq(p).sum()),
                "groups": int(split.loc[split.partition.eq(p), "group_id"].nunique())} for p in ("train", "validation", "test")}
            metrics = {"seed": seed, "split": counts, "epochs": args.epochs, "group_overlap_count": 0,
                "autoencoder_training_performed": True, "changed_rows": len(changes),
                "used_lof_columns": source_columns(features),
                "unused_lof_columns": sorted(set(raw)-set(source_columns(features))),
                "input_invisible_changes": int((cases.changed & ~cases.input_visible_change).sum()),
                "rule_counts": {k: len(rules[k]) for k in ("profiling_candidates", "profiling_selected", "dependency_candidates", "dependency_selected")},
                "lof": lof_metrics, **joint_metrics(cases)}
            write_json(sub / "metrics.json", metrics)
            repetitions.append(metrics)
            print(f"[Seed {seed}] Hinweise auf Änderungen: Regeln {metrics['methods']['rules']['changed_selected_after']}, "
                  f"Eingangs-LOF {metrics['methods']['baseline']['changed_selected_after']}, "
                  f"latenter LOF {metrics['methods']['latent']['changed_selected_after']} / {len(changes)}", flush=True)
        if file_sha256(source) != input_hash or source_hashes(project) != code_hash:
            raise ValueError("Eingabe oder Quellcode wurden während des Versuchs verändert.")
        summary = {"protocol": PROTOCOL["name"], "row_count": len(raw), "seeds": args.seeds, "epochs": args.epochs,
                   "repetitions": repetitions, "source_input_unchanged": True, "source_code_unchanged": True,
                   "delivery_memory_written": False, "test_labels_used_for_fitting": False,
                   "real_error_metrics_available": False, "scope": PROTOCOL["scope"],
                   "mean_recall_of_changes": {m: float(np.mean([r["methods"][m]["recall_of_changes"] for r in repetitions]))
                                              for m in METHOD_LABELS}}
        write_json(out / "summary.json", summary)
        render(out, summary, protocol)
        finish_output(out, {"input_sha256": input_hash, "source_input": str(source), "protocol": PROTOCOL["name"],
                            "test_package_sha256": code_hash["package"]})
        print(json.dumps({k: summary[k] for k in ("row_count", "seeds", "epochs", "mean_recall_of_changes")}, ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}", flush=True)
        return out
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"Unvollständiger Lauf: {out}", file=sys.stderr)
        raise


def main():
    p = parser()
    args = p.parse_args()
    try: analyze(args)
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, f"Gemeinsamer Test abgebrochen: {exc}\n")


if __name__ == "__main__":
    main()
