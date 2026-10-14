"""Prüft die drei Referenzabhängigkeiten im Ergebnis von make_example.py.

Technischer Funktionstest, keine Messung der Erkennungsleistung an echten Lieferungen.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def verify(project, run):
    sys.path.insert(0, str(project.resolve() / "src"))
    from generic_schema import choose_groups, normalize_columns
    from review_common import read_json, file_sha256, write_json
    from rule_engine import evaluate_rule_on_dataframe
    summary = read_json(run / "summary.json")
    status = read_json(run / "run_status.json")
    catalog = read_json(run / "rule_catalog.json")
    contract = read_json(run / "dependency_feature_contract.json")
    raw = pd.read_parquet(run / "input_with_ids.parquet")
    required = {"modus", "variante", "anzahl", "abstand", "ablage", "kontrolle", "entity_code"}
    if not required.issubset(raw) or len(raw) != 320:
        raise ValueError("Dieser Prüfer ist nur für das mitgelieferte Kunstprodukt mit 320 Zeilen bestimmt.")
    if status["status"] != "completed" or contract is None:
        raise ValueError("Kein vollständiger Regellauf vorhanden.")
    grouped, _ = choose_groups(raw, summary["grouping"]["columns"])
    normalized, _ = normalize_columns(grouped, contract)
    # Jede der vier Kombinationen separat prüfen, mit ansonsten echten Beispielwerten.
    probes = normalized.groupby(["modus", "variante"], sort=True).head(1).reset_index(drop=True)
    if len(probes) != 4:
        raise ValueError("Die vier vorgesehenen Kombinationen fehlen im Testbestand.")
    checks = []
    def check(name, success, **details):
        checks.append({"check": name, "passed": bool(success), **details})
    selected = catalog["eligible_dependencies"]
    relations = []
    for target in ("anzahl", "abstand", "ablage"):
        rules = [r for r in selected if r["metadata"]["target_col"] == target]
        bad = probes.copy()
        if target == "anzahl": bad[target] = np.where(bad[target].eq(2), 5, 2)
        elif target == "abstand": bad[target] = np.where(bad[target].eq(4), 12, 4)
        else: bad[target] = "WRONG"
        coverage = pd.Series(False, index=probes.index)
        alarms = coverage.copy()
        detected = coverage.copy()
        for rule in rules:
            applies, _, invalid = evaluate_rule_on_dataframe(probes, rule)
            _, _, bad_invalid = evaluate_rule_on_dataframe(bad, rule)
            coverage |= applies
            alarms |= invalid
            detected |= bad_invalid
        passed = bool(rules and coverage.all() and not alarms.any() and detected.all())
        relations.append({"target": target, "rules": len(rules), "covered_clean_cases": int(coverage.sum()),
                          "false_alarms": int(alarms.sum()), "detected_counterexamples": int(detected.sum()), "passed": passed})
        check("Referenz " + target, passed)
    check("Kontrollgröße abgewiesen", not any(r["metadata"]["target_col"] == "kontrolle" for r in selected))
    latent = pd.read_parquet(run / "latent_outlier_results_lof.parquet")
    check("Kennungen vollständig", set(raw.record_key) == set(latent.record_key) and latent.record_key.is_unique)
    check("Originalattribute erhalten", raw.sort_values("record_key").reset_index(drop=True).equals(
        latent[raw.columns].sort_values("record_key").reset_index(drop=True)))
    check("LOF-Rangfolge", latent.outlier_rank.tolist() == list(range(1, len(raw)+1)) and latent.lof_raw.is_monotonic_decreasing)
    check("Latenter Raum endlich", np.isfinite(latent.filter(regex=r"^latent_\d+$").to_numpy()).all())
    history = read_json(run / "training_history.json")
    check("Autoencoder tatsächlich trainiert", len(history) == summary["epochs"] and bool(history) and np.isfinite(pd.DataFrame(history).to_numpy()).all())
    split = pd.read_csv(run / "dependency_split.csv")
    check("Gruppen bleiben getrennt", split.groupby("group_id").partition.nunique().eq(1).all())
    check("Testanteil im Abhängigkeitslernen unbenutzt", not summary["dependency_test_evaluated"])
    check("Merkmalsrollen nur aus Training", contract["scope"] == "training_partition_only")
    cases = read_json(run / "cases.json")
    check("Keine automatische Fachentscheidung", all(c["decision"] == "unreviewed" for c in cases))
    manifest = read_json(run / "run_manifest.json")
    check("Exportierte Dateien unverändert", all(file_sha256(run / name) == digest for name, digest in manifest["output_sha256"].items()))
    report = {"example_check_passed": all(c["passed"] for c in checks), "checks": checks, "reference_relations": relations,
              "scope": "Technischer Funktionstest des Kunstprodukts; keine reale oder unabhängige Fehlererkennungsleistung."}
    write_json(run / "example_verification.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()
    try:
        report = verify(args.project, args.run)
    except (ValueError, KeyError, OSError) as exc:
        p.exit(2, f"Beispielprüfung abgebrochen: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["example_check_passed"]:
        p.exit(1, "Mindestens ein Prüfkriterium ist nicht erfüllt.\n")


if __name__ == "__main__":
    main()
