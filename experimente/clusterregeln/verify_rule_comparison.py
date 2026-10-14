"""Ergebnisse ohne erneutes Training anhand gespeicherter Fälle nachprüfen."""
from __future__ import annotations

import argparse
from pathlib import Path


def assert_same_cases(left, right):
    import pandas as pd
    # Pandas 3/Arrow kann eine zusammengeführte leere Textspalte als NaN
    # zurücklesen, während ein einzelner Teilbestand None enthält. Nur echte
    # Fehlwerte vereinheitlichen; Texte/Zahlen und Werte bleiben unverändert.
    def canonical_nulls(frame):
        return frame.astype(object).where(frame.notna(), None)
    pd.testing.assert_frame_equal(canonical_nulls(left), canonical_nulls(right), check_dtype=False)


def verify_source_file(manifest):
    from review_common import file_sha256
    source = Path(manifest["source"])
    # experiment_io reserviert source_sha256 für Python-Quellcode. Die
    # Parquet-Prüfsumme bekommt deshalb ausdrücklich ein anderes Feld.
    expected = manifest["input_sha256"]
    if not isinstance(expected, str) or not source.is_file() or file_sha256(source) != expected:
        raise ValueError("Originaldatei fehlt oder hat sich geändert.")


def verify_content(out):
    import numpy as np
    import pandas as pd
    from review_common import read_json, file_sha256, digest
    from comparison_core import (Protocol, METHODS, DATE, DATE_MISSING, eligible_columns,
        targets_from_training, rule_evidence, evaluate_selected, summarize_cases, paired_comparisons)
    from phase2_split import select_partition

    out = Path(out)
    protocol, summary, lock = (read_json(out/name) for name in ("protocol.json", "summary.json", "selection_lock.json"))
    p = Protocol(**protocol["parameters"])
    frame = pd.read_parquet(out/"model_table.parquet")
    cases = pd.read_parquet(out/"test_cases.parquet")
    candidates = read_json(out/"all_candidates.json")
    test_audits = {a["rule_id"]: a for a in read_json(out/"test_rule_evidence.json")}
    checks = []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(f"Vergleichsprüfung fehlgeschlagen: {name}")
    check("Originalidentitäten vollständig und eindeutig", len(frame) == summary["sample_rows"] and frame.record_key.is_unique
          and digest(sorted(frame.record_key)) == protocol["sample_keys_sha256"])
    check("Keine Geometriespalten eingelesen", not protocol["geometry_used"] and
          protocol["attribute_reading"]["geometry_columns_read"] == [])
    allowed = set(protocol["allowed_attribute_columns"])
    check("Modelltabelle enthält nur erlaubte Attribute und Kennungen",
          set(frame) <= allowed | {"gml_id", "record_key", "__generic_group"})
    check("Vor dem Test gespeichertes Protokoll unverändert", file_sha256(out/"protocol.json") == lock["protocol_sha256"])
    check("Kandidatenauswahl seit Testbeginn unverändert", file_sha256(out/"all_candidates.json") == lock["candidate_file_sha256"])
    check("Regelkennungen eindeutig", len({r["rule_id"] for r in candidates}) == len(candidates))
    check("Nur neue Scope-Umschläge statt ungeschützter lokaler Regeln",
          all(r["rule_type"] == "scoped_dependency_comparison_11" for r in candidates))
    check("Alle Regeln ohne fachliche Freigabe", all(r["status"] == "proposed" for r in candidates))
    all_selected_ids = set()
    for info in summary["splits"]:
        seed = info["seed"]
        seed_dir = out/f"seed_{seed}"
        split = pd.read_csv(seed_dir/"split.csv")
        check(f"{seed}: Identitäten der Aufteilung", split.record_key.is_unique and set(split.record_key) == set(frame.record_key))
        check(f"{seed}: Keine Gruppe überschneidet Teilmengen", split.groupby("group_id").partition.nunique().max() == 1)
        train, validation, test = (select_partition(frame, split, part) for part in ("train", "validation", "test"))
        groups = split.set_index("record_key").group_id
        columns, excluded = eligible_columns(train, protocol["allowed_attribute_columns"])
        check(f"{seed}: Ziele und Spalten nur anhand Training gewählt", columns == info["training_columns"] and
              excluded == info["excluded_columns"] and targets_from_training(train, columns) == info["targets"])
        for no, target in enumerate(info["targets"], 1):
            sub = seed_dir/f"target_{no:02d}"
            contract = read_json(sub/"feature_contract.json")
            selected = read_json(sub/"selected_rules.json")
            local_candidates = read_json(sub/"candidate_rules.json")
            prefix = f"{seed}/{target}"
            path = (sub/"selected_rules.json").relative_to(out).as_posix()
            check(prefix+": Auswahl unverändert", file_sha256(sub/"selected_rules.json") == lock["selected_files_sha256"][path])
            check(prefix+": Auswahl entspricht Kandidaten", selected == [r for r in local_candidates if r["selected_on_validation"]])
            check(prefix+": Fit-Kennungen nur aus Training", contract["train_record_keys"] == train.record_key.tolist()
                  and contract["validation_record_keys"] == validation.record_key.tolist())
            check(prefix+": Ziel und Ableitungen ausgeschlossen", target not in contract["features"] and
                set(contract["features"]) <= allowed and not (set(contract["features"]) & set(contract["target_exclusions"])))
            history = read_json(sub/"training_history.json")
            check(prefix+": Tatsächlicher endlicher Trainingsverlauf", len(history) == p.epochs and
                  [h["epoch"] for h in history] == list(range(1, p.epochs+1)) and
                  all(np.isfinite(list(h.values())).all() for h in history) and (sub/"autoencoder.pt").stat().st_size > 1000)
            for matrix_name in ("development_representations.npz", "test_representations.npz"):
                with np.load(sub/matrix_name, allow_pickle=False) as matrices:
                    expected = {"attribute_train": len(train), "latent_train": len(train),
                        "attribute_validation": len(validation), "latent_validation": len(validation),
                        "attribute_test": len(test), "latent_test": len(test)}
                    check(prefix+": Endliche passende Matrizen "+matrix_name,
                        all(a.ndim == 2 and len(a) == expected[k] and np.isfinite(a).all()
                            and a.shape[1] == (p.latent_dim if k.startswith("latent") else contract["dimensions"])
                            for k, a in matrices.items()))
            test_routes = pd.read_parquet(sub/"test_routes.parquet")
            check(prefix+": Testzuordnungen in richtiger Reihenfolge", test_routes.record_key.tolist() == test.record_key.tolist())
            model_reports = read_json(sub/"model_reports.json")
            gates = {(r["method"], r["cluster"]): r["model_gate_passed"] for r in model_reports}
            for method in METHODS:
                rules = [r for r in selected if r["scope"]["method"] == method]
                val_labels = np.zeros(len(validation), dtype=int)
                if method != "global":
                    vroutes = pd.read_parquet(sub/f"{method}_validation_routes.parquet")
                    check(prefix+"/"+method+": Validierungszuordnung", vroutes.record_key.tolist() == validation.record_key.tolist())
                    val_labels = vroutes.cluster.to_numpy()
                for r in rules:
                    scope, inner = r["scope"], r["inner_rule"]
                    all_selected_ids.add(r["rule_id"])
                    take = np.ones(len(validation), dtype=bool) if method == "global" else val_labels == scope["cluster"]
                    evidence = rule_evidence(validation.loc[take], inner, groups)
                    check(r["rule_id"]+": Validierungsbelege stimmen", evidence == r["validation"])
                    check(r["rule_id"]+": Auswahlbedingungen erfüllt", evidence["cases"] >= p.min_rule_cases and
                        evidence["groups"] >= p.min_rule_groups and evidence["agreement_all"] >= p.min_agreement
                        and gates[(method, scope["cluster"])] and not r["test_used_for_selection"])
                    check(r["rule_id"]+": Regelprädiktoren ohne Ziel", all(c["feature"] in contract["features"]
                          for c in inner["metadata"]["conditions"]))
                replay, audits = evaluate_selected(test, test_routes[method].to_numpy(), rules, target, groups, method, seed, p)
                original = cases.loc[cases.seed.eq(seed) & cases.method.eq(method) & cases.target.eq(target)].reset_index(drop=True)
                assert_same_cases(replay, original)
                check(prefix+"/"+method+": Testfälle und Scope nachgerechnet", True)
                check(prefix+"/"+method+": Testbelege nachgerechnet", all(
                    a["evidence"] == test_audits[a["rule_id"]]["evidence"] and
                    a["test_assessment"] == test_audits[a["rule_id"]]["test_assessment"] for a in audits))
        check(f"{seed}: Vergleich umfasst gleiche Fälle", paired_comparisons(cases.loc[cases.seed.eq(seed)]) ==
            [{k: v for k, v in row.items() if k != "seed"} for row in summary["paired_comparisons"] if row["seed"] == seed])
        for method in METHODS:
            metrics = summarize_cases(cases.loc[cases.seed.eq(seed) & cases.method.eq(method)])
            reported = next(m for m in summary["method_metrics"] if m["seed"] == seed and m["method"] == method)
            check(f"{seed}/{method}: Kennzahlen nachgerechnet", all(reported[k] == v for k, v in metrics.items()))
    check("Genau alle ausgewählten Regeln wurden getestet", set(test_audits) == all_selected_ids)
    leak = read_json(out/"target_leakage_checks.json")
    check("Zielmutation verändert keine Vorhersage", len(leak) == sum(len(i["targets"])*3 for i in summary["splits"])
          and all(c["passed"] for c in leak))
    check("Keine Quell-/Codeänderung im Lauf", summary["source_unchanged"] and summary["code_unchanged"])
    return {"passed": True, "check_count": len(checks), "checks": checks,
            "scope": "saved_artifacts_and_recomputed_rule_evidence; no_model_refit; no_browser_automation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from run_rule_comparison import bootstrap
    bootstrap(args.project)
    from experiment_io import read_verified_run
    from review_common import read_json, file_sha256
    out = args.run.resolve(strict=True)
    manifest = read_json(out/"run_manifest.json")
    read_verified_run(out, list(manifest["output_sha256"]))
    verify_source_file(manifest)
    protocol = read_json(out/"protocol.json")
    for name, expected in protocol["code_sha256_before"].items():
        if not Path(name).is_file() or file_sha256(name) != expected:
            raise ValueError(f"Code des Vergleichslaufs hat sich geändert: {name}")
    verified = verify_content(out)
    print(f"Bestanden: {verified['check_count']} Prüfungen; alle exportierten Dateien unverändert.")


if __name__ == "__main__":
    main()
