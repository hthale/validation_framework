"""Vergleichsartefakte prüfen und LOF aus gespeicherten Matrizen nachrechnen.

Dies prüft technische Konsistenz, keine fachliche Richtigkeit und keinen
positiven Mehrwert. Es werden weder Pickles noch Modellgewichte geladen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def verify(project, run):
    sys.path.insert(0, str(project.resolve() / "src"))
    from experiment_io import read_verified_run
    from review_common import write_json
    run = run.resolve(strict=True)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["output_sha256"])
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise ValueError("Manifest enthält unzulässige relative Pfade.")
    read_verified_run(run, names)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    original = pd.read_parquet(run / "input_with_ids.parquet").set_index("record_key", drop=False)
    checks = []
    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition: raise AssertionError(name)
    check("Originalkennungen eindeutig", original.index.is_unique)
    check("Keine fachlichen Fehlerkennzahlen behauptet", not summary["confirmed_error_metrics_available"])
    check("Keine Testlabels zur Parameterauswahl", not protocol["parameter_selection_using_test_labels"])
    repetitions = []
    for seed in summary["seeds"]:
        sub = run / f"seed_{seed}"
        record = json.loads((sub / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((sub / "training_history.json").read_text(encoding="utf-8"))
        split = pd.read_csv(sub / "split.csv", dtype=str)
        check(f"{seed}: Jede Kennung genau einer Teilmenge zugeordnet", split.record_key.is_unique and set(split.record_key) == set(original.index))
        check(f"{seed}: Gruppen bleiben getrennt", split.groupby("group_id").partition.nunique().max() == 1)
        check(f"{seed}: Angeforderte Trainingsepochen dokumentiert", len(history) == summary["epochs"] and np.isfinite([list(h.values()) for h in history]).all())
        train_keys = split.loc[split.partition.eq("train"), "record_key"].tolist()
        test_keys = split.loc[split.partition.eq("test"), "record_key"].tolist()
        train = pd.read_parquet(sub / "train_original.parquet").set_index("record_key", drop=False)
        pd.testing.assert_frame_equal(train.loc[train_keys], original.loc[train_keys], check_dtype=False)
        check(f"{seed}: Training enthält unveränderte Originalzeilen", True)
        before = pd.read_parquet(sub / "test_before.parquet")
        after = pd.read_parquet(sub / "test_after.parquet")
        check(f"{seed}: Prüfreihenfolge und Kennungen bleiben gleich", before.record_key.tolist() == after.record_key.tolist() == test_keys)
        log = json.loads((sub / "changes.json").read_text(encoding="utf-8"))
        log_keys = {c["record_key"] for c in log}
        differs = ~(before.eq(after) | (before.isna() & after.isna()))
        changed_mask = differs.any(axis=1)
        check(f"{seed}: Genau eine Zelle je protokollierter Änderung", differs.sum(axis=1).max() == 1 and int(differs.sum().sum()) == len(log)
              and set(before.loc[changed_mask, "record_key"]) == log_keys)
        check(f"{seed}: Änderungen liegen ausschließlich in Prüfung", log_keys <= set(test_keys) and not log_keys.intersection(train_keys))
        cases = pd.read_parquet(sub / "scores.parquet")
        reps = np.load(sub / "representations.npz", allow_pickle=False)
        for method in ("baseline", "latent"):
            m = record["methods"][method]
            train_matrix = reps[f"{method}_train"]
            before_matrix, after_matrix = reps[f"{method}_before"], reps[f"{method}_after"]
            check(f"{seed}/{method}: Repräsentationen vollständig und endlich", len(train_matrix) == len(train_keys)
                  and len(before_matrix) == len(test_keys) == len(after_matrix)
                  and all(np.isfinite(x).all() for x in (train_matrix, before_matrix, after_matrix)))
            model = LocalOutlierFactor(n_neighbors=m["n_neighbors_effective"], novelty=True, metric="euclidean", n_jobs=1).fit(train_matrix.astype(float))
            scores_before, scores_after = -model.score_samples(before_matrix.astype(float)), -model.score_samples(after_matrix.astype(float))
            table = cases.loc[cases.method.eq(method)].set_index("record_key").loc[test_keys]
            np.testing.assert_allclose(table.score_before, scores_before, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(table.score_after, scores_after, rtol=1e-7, atol=1e-8)
            check(f"{seed}/{method}: LOF unabhängig nachgerechnet", True)
            ranked = table.reset_index().sort_values(["score_after", "record_key"], ascending=[False, True])
            check(f"{seed}/{method}: Rangfolge stimmt", ranked.rank_after.tolist() == list(range(1, len(test_keys)+1)))
            k = m["review_budget_k"]
            selected = set(ranked.head(k).record_key)
            check(f"{seed}/{method}: Gleiches Budget und Kennzahlen nachgezählt", int(table.selected_after.sum()) == k
                  and len(selected.intersection(log_keys)) == m["selected_changed"]
                  and np.isclose(m["recall_of_changes_at_k"], len(selected.intersection(log_keys))/len(log)))
            check(f"{seed}/{method}: Unveränderte Zeilen erhalten unveränderte Scores", np.allclose(scores_before[~changed_mask], scores_after[~changed_mask], rtol=0, atol=1e-9))
        check(f"{seed}: Gleiche Nachbarn, Prüffälle und Budgets", all(record["methods"]["baseline"][k] == record["methods"]["latent"][k]
              for k in ("n_neighbors_effective", "review_budget_k", "test_rows", "changed_rows")))
        repetitions.append(record)
    check("Gesamtbericht enthält unveränderte Wiederholungen", repetitions == summary["repetitions"])
    result = {"comparison_check_passed": True, "checks_passed": len(checks), "checks": checks,
              "scope": "Artefaktkonsistenz und Nachrechnung; keine fachliche Fehlerbewertung, kein geforderter positiver Mehrwert."}
    # Separater Prüfbeleg gehört nicht zu den ursprünglichen Laufartefakten.
    write_json(run / "verification_08.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    try: verify(args.project, args.run)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        p.exit(2, f"Prüfung fehlgeschlagen: {exc}\n")
