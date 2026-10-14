"""Nachprüfung 09a: kompatibler Vergleich fehlender Werte nach Parquet-Import.

Diese Datei im Projektordner NEBEN src und testpaket_09_gemeinsame_pruefung
ablegen. Aufruf:
  python verify_joint_run_09a.py --project . --run data/dop_joint_check_09/ZEITORDNER

Sie verwendet den vorhandenen Prüflauf und den unveränderten Code aus Paket 09.
Prüfsummen, Kennungen, Gruppen, Änderungen, Regeln und LOF werden weiter geprüft.
Keine Änderung von Originaldaten, Modellen, Manifesten oder Paket-09-Dateien.
None, NaN und pd.NA werden nur dann gleich behandelt, wenn beide Zellen fehlen.
Vorhandene Werte, Zeilen-/Spaltenreihenfolge und fehlend/vorhanden bleiben streng.
Quelle zum String-Datentyp: https://pandas.pydata.org/docs/whatsnew/v3.0.0.html
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def original_support(project):
    import sys
    project = Path(project).resolve(strict=True)
    package = project / "experimente" / "dop_pruefung"
    required = ("run_joint_dop_check.py", "joint_protocol.py", "verify_joint_run.py")
    if not all((package / name).is_file() for name in required):
        raise ValueError("Den bestehenden Projektordner mit dem ursprünglichen Testpaket 09 als --project angeben.")
    sys.path.insert(0, str(package))
    from run_joint_dop_check import bootstrap, source_hashes
    return bootstrap(project), source_hashes


def assert_same_values(left, right):
    """Vergleicht fehlende Positionen getrennt von exakten vorhandenen Werten.

    Nur Kopien werden für den Vergleich vereinheitlicht. Kein str()-Cast,
    kein fillna(''), keine Auslassung betroffener Zeilen oder Spalten.
    """
    import numpy as np
    import pandas as pd
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise AssertionError("Zwei DataFrames erforderlich.")
    if left.shape != right.shape:
        raise AssertionError(f"Unterschiedliche Tabellenform: {left.shape} != {right.shape}")
    pd.testing.assert_index_equal(left.index, right.index, exact=False)
    pd.testing.assert_index_equal(left.columns, right.columns, exact=False)
    left_missing, right_missing = left.isna().to_numpy(bool), right.isna().to_numpy(bool)
    different_missing = left_missing != right_missing
    if different_missing.any():
        row, col = np.argwhere(different_missing)[0]
        raise AssertionError(f"Fehlend/vorhanden unterscheidet sich bei Zeilenposition {row}, "
                             f"Spalte {left.columns[col]!r}.")
    def comparable(frame, missing):
        values = frame.to_numpy(dtype=object, copy=True)
        values[missing] = None
        return pd.DataFrame(values, index=frame.index.copy(), columns=frame.columns.copy(), dtype=object)
    pd.testing.assert_frame_equal(comparable(left, left_missing), comparable(right, right_missing),
                                  check_dtype=False, check_exact=True)



def verify(project, run, control_fixture=False):
    project, source_hashes = original_support(project)
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import LocalOutlierFactor
    from experiment_io import read_verified_run
    from generic_schema import infer_columns, normalize_columns, tabular_preparation
    from latent_comparison import transform_frozen, baseline_matrix, ranked_scores, evaluate_pair
    from latentspace.string_encoder import CharTokenizer
    from phase2_settings import Phase2Settings
    from phase2_split import make_grouped_split, select_partition
    from rule_engine import validate_rules_on_full_dataframe
    from review_common import file_sha256, canonical
    from joint_protocol import GROUP_COLUMNS, METHOD_LABELS, OPERATORS, inject_cases, known_results, evaluate_rules, joint_metrics
    run = Path(run).resolve(strict=True)
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    checks = []
    def check(ok, name):
        if not ok: raise ValueError("Prüfung fehlgeschlagen: " + name)
        checks.append(name)
    manifest, protocol, summary = (read(n) for n in ("run_manifest.json", "protocol.json", "summary.json"))
    hashes = read_verified_run(run, list(manifest["output_sha256"]))
    check(bool(hashes), "Artefaktprüfsummen")
    check(summary["protocol"] == "joint_dop_check_09", "Passendes Versuchsprotokoll")
    check(source_hashes(project) == protocol["code_at_start"], "Projekt- und Testcode entsprechen dem geprüften Lauf")
    source = Path(manifest["source_input"])
    check(source.is_file() and file_sha256(source) == manifest["input_sha256"] == protocol["input_sha256"], "Originaldatei unverändert verfügbar")
    raw = pd.read_parquet(run / "input_with_ids.parquet")
    original = pd.read_parquet(source)
    assert_same_values(raw[original.columns], original)
    check(len(raw) == summary["row_count"] and raw.record_key.is_unique, "Originalattribute und vollständige eindeutige Kennungen")
    args = protocol["arguments"]
    check(not summary["test_labels_used_for_fitting"] and not summary["real_error_metrics_available"], "Auswertung richtig eingegrenzt")
    verified_reps, control_evidence = [], []
    if control_fixture:
        from make_control_fixture import make_fixture
        assert_same_values(raw, make_fixture())
        check(True, "Künstlicher Kontrollbestand entspricht den vorgegebenen Referenzdaten")
    for rep in summary["repetitions"]:
        seed = rep["seed"]
        sub = run / f"seed_{seed}"
        load = lambda name: json.loads((sub / name).read_text(encoding="utf-8"))
        prefix = f"Seed {seed}: "
        settings = Phase2Settings(group_columns=tuple(GROUP_COLUMNS), random_state=seed)
        split = pd.read_csv(sub / "split.csv")
        assert_same_values(split, make_grouped_split(raw, settings))
        check(split.groupby("group_id").partition.nunique().max() == 1, prefix+"Gruppen und reproduzierte Aufteilung")
        train = select_partition(raw, split, "train")
        validation = select_partition(raw, split, "validation")
        expected_before, expected_after, expected_changes = inject_cases(select_partition(raw, split, "test"), train, args["cases_per_type"], seed)
        before, after = (pd.read_parquet(sub / f"test_{m}.parquet") for m in ("before", "after"))
        changes = load("changes.json")
        assert_same_values(before, expected_before)
        assert_same_values(after, expected_after)
        check(canonical(changes) == canonical(expected_changes), prefix+"Alle acht Änderungsarten exakt reproduziert")
        # Unabhängige Zählung geänderter Zellen: keine zweite Änderung oder Zeilenverschiebung.
        diff = ~(before.eq(after) | (before.isna() & after.isna()))
        counts = diff.sum(axis=1)
        keys = {c["record_key"] for c in changes}
        check(counts.le(1).all() and set(before.loc[counts.eq(1), "record_key"]) == keys, prefix+"Genau eine geänderte Zelle je protokolliertem Fall")
        check(len(changes) == 8*args["cases_per_type"], prefix+"Keine Änderungsklasse ausgelassen")
        contract = load("feature_contract.json")
        inferred = infer_columns(train, excluded=GROUP_COLUMNS, scope="training_partition_only")
        check(canonical(contract["columns"]) == canonical(inferred["columns"]), prefix+"Merkmalsrollen allein aus Training reproduziert")
        normalized = {}
        for part, data in (("train", train), ("validation", validation), ("before", before), ("after", after)):
            normalized[part], _ = normalize_columns(data, inferred)
            assert_same_values(pd.read_parquet(sub / f"{part}_normalized.parquet"), normalized[part])
        check(True, prefix+"Normalisierte Trainings-, Validierungs- und Testwerte")
        _, prepared, features = tabular_preparation(normalized["train"], inferred)
        features["fit_scope"] = "training_partition_only"
        check(canonical(features) == canonical(contract["latent_features"]), prefix+"Eingefrorene Merkmalsauswahl reproduziert")
        tokenizer = CharTokenizer()
        config = load("model_config.json")
        matrices = np.load(sub / "representations.npz", allow_pickle=False)
        for part in ("train", "before", "after"):
            model_data, tab = transform_frozen(normalized[part], prepared, features)
            strings = [np.asarray([tokenizer.encode(v, config["max_seq_len"]) for v in model_data[c]], dtype=np.int64)
                       for c in features["string_columns"]]
            expected = baseline_matrix(tab, strings, tokenizer.vocab_size)
            np.testing.assert_allclose(matrices["baseline_"+part], expected, rtol=0, atol=1e-6)
        check(True, prefix+"Baseline ohne gespeicherte Pickles aus Trainingsdaten nachgerechnet")
        cases, score_table = (pd.read_parquet(sub / f"{name}.parquet") for name in ("case_results", "scores"))
        check(cases.record_key.tolist() == before.record_key.tolist() and set(cases.loc[cases.changed, "record_key"]) == keys,
              prefix+"Fallkennungen und Änderungslabels korrekt verbunden")
        used_names = set(features["numeric_columns"]+features["categorical_columns"]+features["string_columns"])
        check(not (used_names & {"record_key", "changed", "operator", "lof_raw", "cluster_label", *GROUP_COLUMNS}), prefix+"Keine Ergebnislabels, Kennungen oder Gruppierungsfelder als LOF-Merkmale")
        for method in ("baseline", "latent"):
            arrays = [matrices[method+"_"+part] for part in ("train", "before", "after")]
            check(all(np.isfinite(x).all() for x in arrays) and len(arrays[0]) == len(train), prefix+method+": endliche Matrizen mit Trainingszeilen")
            model = LocalOutlierFactor(n_neighbors=min(args["n_neighbors"], len(train)-1), novelty=True, metric="euclidean", n_jobs=1)
            model.fit(np.asarray(arrays[0], dtype=np.float64))
            scores_before, scores_after = (-model.score_samples(np.asarray(a, dtype=np.float64)) for a in arrays[1:])
            m = score_table.loc[score_table.method.eq(method)].set_index("record_key").loc[cases.record_key]
            for moment, values in (("before", scores_before), ("after", scores_after)):
                np.testing.assert_allclose(m["score_"+moment], values, rtol=1e-9, atol=1e-9)
                ranks = ranked_scores(cases.record_key, values).set_index("record_key").loc[cases.record_key, "rank"]
                np.testing.assert_array_equal(ranks, m["rank_"+moment])
                expected_selected = ranks.to_numpy() <= int(np.ceil(len(cases)*args["review_fraction"]))
                check(np.array_equal(expected_selected, cases[method+"_"+moment]), prefix+method+": "+moment+"-Ränge und Budget")
            untouched = ~cases.changed.to_numpy(bool)
            np.testing.assert_allclose(scores_before[untouched], scores_after[untouched], rtol=0, atol=1e-9)
            check(True, prefix+method+": unveränderte Fälle behalten ihren Score")
        visible = np.any(matrices["baseline_before"] != matrices["baseline_after"], axis=1)
        check(np.array_equal(visible, cases.input_visible_change), prefix+"Unsichtbare Änderungen bleiben erfasst")
        rules = load("rules.json")
        selected = {}
        for kind in ("profiling", "dependency"):
            rulelist = rules[kind+"_candidates"]
            metrics, _ = validate_rules_on_full_dataframe(normalized["validation"], rulelist)
            eligible_ids = []
            for rule, metric in zip(rulelist, metrics):
                check(metric["evaluation_status"] != "invalid_rule", prefix+"Gültiger Kandidat "+rule["rule_id"])
                md = rule["metadata"]
                # Inhaltliche Zahlen des Validierungsergebnisses separat nachrechnen.
                for key in ("coverage_count", "valid_count", "invalid_count", "pass_rate"):
                    check(metric[key] == md["validation"][key], prefix+rule["rule_id"]+": Validierung "+key)
                gate = md.get("model_gate_passed", True)
                if gate and metric["coverage_count"] >= 5 and metric["pass_rate"] is not None and metric["pass_rate"] >= .95:
                    eligible_ids.append(rule["rule_id"])
            selected[kind] = rules[kind+"_selected"]
            check(eligible_ids == [r["rule_id"] for r in selected[kind]], prefix+kind+": ausgewählte Regeln erfüllen unveränderte Gates")
            lookup = {r["rule_id"]: r for r in rulelist}
            check(all(canonical(r) == canonical(lookup[r["rule_id"]]) for r in selected[kind]), prefix+kind+": angewendete Regeln entsprechen den Kandidaten")
        for report in rules["dependency_reports"]:
            check(not report["test_evaluated"] and report["train_rows"] == len(train)
                  and report["validation_rows"] == len(validation), prefix+report["target_col"]+": Lern- und Validierungsumfang")
            if "validation_metrics" in report:
                v = report["validation_metrics"]
                expected_gate = bool(v["n_evaluated"] and v["accuracy"] >= .95 and v["f1_macro"] >= .9 and v["precision_macro"] >= .9)
                check(report["model_gate_passed"] == expected_gate, prefix+report["target_col"]+": Modellgate unverändert")
        if control_fixture:
            from rule_engine import evaluate_rule_on_dataframe
            probe_raw = raw.drop_duplicates(["spektralkanaele", "detailstufe"]).reset_index(drop=True)
            probe, _ = normalize_columns(probe_raw, inferred)
            recovered = []
            for target in ("tif_band_count", "kamera_sensor", "bodenpixelgroesse"):
                target_rules = [r for r in selected["dependency"] if r["metadata"]["target_col"] == target]
                check(bool(target_rules), prefix+"Automatische Referenzregel vorhanden: "+target)
                covered, invalid_clean, invalid_changed = np.zeros((3, len(probe)), dtype=bool)
                changed_probe = probe.copy(deep=True)
                if target == "kamera_sensor":
                    changed_probe[target] = changed_probe[target].map({"Kamera_A": "Kamera_B", "Kamera_B": "Kamera_A"})
                else:
                    changed_probe[target] = changed_probe[target] + 7
                for r in target_rules:
                    a, _, bad = evaluate_rule_on_dataframe(probe, r)
                    _, _, counter_bad = evaluate_rule_on_dataframe(changed_probe, r)
                    covered |= a.to_numpy(bool)
                    invalid_clean |= bad.to_numpy(bool)
                    invalid_changed |= counter_bad.to_numpy(bool)
                check(covered.all() and not invalid_clean.any() and invalid_changed.all(), prefix+"Referenzfälle und Gegenbeispiele: "+target)
                recovered.append({"target": target, "rules": len(target_rules), "clean_cases": len(probe),
                                  "clean_cases_flagged": int(invalid_clean.sum()), "counterexamples_flagged": int(invalid_changed.sum())})
            check(not any(r["metadata"]["target_col"] == "kontrolle" for r in selected["dependency"]), prefix+"Uninformative Kontrollgröße abgewiesen")
            control_evidence.append({"seed": seed, "reference_relations": recovered, "control_rejected": True})
        stored_rule_cases = pd.read_parquet(sub / "rule_cases.parquet")
        reconstructed = []
        for moment, data in (("before", before), ("after", after)):
            for method in ("known", "profiling", "dependency"):
                r = known_results(data) if method == "known" else evaluate_rules(normalized[moment], selected[method])
                flagged = set(r.loc[r.violation, "record_key"])
                check(np.array_equal(cases.record_key.isin(flagged), cases[method+"_"+moment]), prefix+method+": "+moment+"-Fallhinweise")
                r["method"], r["moment"] = method, moment
                reconstructed.append(r)
            for name, members in (("rules", ("known", "profiling", "dependency")), ("rules_baseline", ("rules", "baseline")),
                                  ("rules_latent", ("rules", "latent")), ("all", ("rules", "baseline", "latent"))):
                check(np.array_equal(cases[[m+"_"+moment for m in members]].any(axis=1), cases[name+"_"+moment]), prefix+name+": "+moment+"-Vereinigung")
        assert_same_values(pd.concat(reconstructed, ignore_index=True), stored_rule_cases)
        recomputed = joint_metrics(cases)
        check(all(canonical(recomputed[k]) == canonical(rep[k]) for k in recomputed), prefix+"Fallkennzahlen, Typen, Überschneidungen und Zusatzbeitrag nachgerechnet")
        history = load("training_history.json")
        check(len(history) == args["epochs"] and [h["epoch"] for h in history] == list(range(1, args["epochs"]+1))
              and all(np.isfinite(list(h.values())).all() for h in history), prefix+"Vollständiger endlicher Trainingsverlauf")
        check(canonical(load("metrics.json")) == canonical(rep), prefix+"Summary entspricht Seed-Metriken")
        verified_reps.append(recomputed)
    means = {m: float(np.mean([r["methods"][m]["recall_of_changes"] for r in verified_reps])) for m in METHOD_LABELS}
    check(canonical(means) == canonical(summary["mean_recall_of_changes"]), "Mittelwerte aus Fallkennzahlen")
    check(len(summary["repetitions"]) == len(args["seeds"]) and [r["seed"] for r in summary["repetitions"]] == args["seeds"], "Alle vorgesehenen Seeds vollständig")
    return {"joint_check_verified": True, "checks_passed": len(checks), "verified_artifact_count": len(hashes),
            "epochs": args["epochs"], "seeds": args["seeds"],
            "synthetic_control_evidence": control_evidence,
            "verifier": {"name": "joint_check_09a", "file_sha256": file_sha256(Path(__file__)),
                         "pandas_version": pd.__version__,
                         "missing_value_comparison": "same missing positions; exact nonmissing values",
                         "original_source_and_artifact_hash_checks_retained": True},
            "scope": "Technische Konsistenz, keine Gütegarantie und keine Browserbedienungsprüfung"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--control-fixture", action="store_true", help="Zusätzlich bekannte künstliche Abhängigkeiten und negative Kontrollgröße prüfen")
    args = p.parse_args()
    try: print(json.dumps(verify(args.project, args.run, args.control_fixture), ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, AssertionError, KeyError) as exc:
        p.exit(2, f"Verifikation fehlgeschlagen: {exc}\n")


if __name__ == "__main__": main()
