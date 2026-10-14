"""Kontrollierter Funktionstest der vorhandenen automatischen Regelerzeugung.

Aufruf im Projektordner:
    python testpaket_05_regelableitung/run_synthetic_rule_check.py --project .

Dieses Zusatzskript importiert den vorhandenen Regelcode. Es installiert nichts,
liest keine echten Lieferungen und ändert weder config.py noch Prüfläufe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib import import_module
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCENARIO = "synthetic_rule_reference_v1"
FEATURES = ("kanaltyp", "detailstufe", "zufall_zahl", "zufall_gruppe")
TARGETS = ("bandanzahl", "pixelgroesse_cm", "exportprofil")
CONTROL = "kontrollwert"
PROFILE_COLUMNS = ("kanaltyp", "detailstufe", *TARGETS, "crs_code")
THRESHOLDS = {"accuracy": .95, "f1_macro": .90, "precision_macro": .90}
ERROR_TYPES = (
    "bandzahl_falsche_zuordnung", "pixelgroesse_falsche_zuordnung",
    "exportprofil_falsche_zuordnung", "crs_konstanz_verletzt", "bandzahl_fehlt",
)
VALID_CHANGES = ("gueltiger_produktwechsel", "gueltige_zufallsvariation")
CASE_LABELS = {
    "unveraendert": "Unverändert",
    "bandzahl_falsche_zuordnung": "Bandanzahl: bekannter Wert, falsche Zuordnung",
    "pixelgroesse_falsche_zuordnung": "Pixelgröße: bekannter Wert, falsche Zuordnung",
    "exportprofil_falsche_zuordnung": "Exportprofil: bekannter Wert, falsche Zuordnung",
    "crs_konstanz_verletzt": "Konstanter CRS-Wert verändert",
    "bandzahl_fehlt": "Bandanzahl fehlt",
    "gueltiger_produktwechsel": "Gültiger Wechsel mit passenden abhängigen Werten",
    "gueltige_zufallsvariation": "Gültige Änderung irrelevanter Merkmale",
}
REQUIRED_SOURCES = (
    "phase2_ml.py", "phase2_settings.py", "phase2_split.py", "profiling.py",
    "rule_engine.py", "rule_conditions.py", "tree_rule_export.py",
    "record_identity.py", "date_roles.py", "latentspace/anomaly_scores.py",
    "review_common.py",
)


def load_framework(project):
    project = Path(project).resolve()
    source = project / "src"
    missing = [name for name in REQUIRED_SOURCES if not (source / name).is_file()]
    if missing:
        raise ValueError("Projekt mit Korrekturpaketen 01–04 angeben. Fehlende Dateien: " + ", ".join(missing))
    sys.path.insert(0, str(source))
    modules = {}
    for name in ("phase2_ml", "phase2_settings", "phase2_split", "profiling", "rule_engine", "review_common"):
        module = import_module(name)
        if Path(module.__file__).resolve().parent != source:
            raise ValueError(f"Modul {name} wurde aus einem anderen Projekt geladen; neuen Python-Prozess starten.")
        modules[name] = module
    return SimpleNamespace(project=project, source=source, **modules)


def record_key(group, observation, namespace=SCENARIO):
    identity = json.dumps([namespace, str(group), int(observation)], separators=(",", ":"))
    return "rk1:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def reference_values(kanaltyp, detailstufe):
    """Nur Datengenerator und Referenzprüfung verwenden diese Festlegung.

    Die Lernfunktionen erhalten weder dieses Mapping noch fertige Regeln.
    Die vier Exportprofile sind ausdrücklich frei erfundene Testannahmen.
    """
    codes = {("RGB", "standard"): "P_A", ("RGB", "fein"): "P_B",
             ("RGBI", "standard"): "P_C", ("RGBI", "fein"): "P_D"}
    return {"bandanzahl": {"RGB": 3, "RGBI": 4}[kanaltyp],
            "pixelgroesse_cm": {"standard": 20, "fein": 10}[detailstufe],
            "exportprofil": codes[(kanaltyp, detailstufe)], "crs_code": 25832}


def create_reference(seed=42):
    """300 Einheiten mit je zwei Beobachtungen; keine echten DOP/ALKIS-Daten."""
    rng = np.random.default_rng(seed)
    combinations = list(product(("RGB", "RGBI"), ("standard", "fein")))
    assignments = np.repeat(np.arange(4), 75)
    rng.shuffle(assignments)
    rows = []
    for group, combination in enumerate(assignments):
        kanaltyp, detailstufe = combinations[int(combination)]
        common = {"test_group": f"einheit-{group:04d}", "kanaltyp": kanaltyp,
                  "detailstufe": detailstufe, "zufall_zahl": float(rng.random()),
                  "zufall_gruppe": str(rng.choice(["Z_A", "Z_B", "Z_C"])),
                  **reference_values(kanaltyp, detailstufe)}
        first_label = int(rng.integers(0, 2))
        for observation in range(2):
            rows.append({"record_key": record_key(group, observation), **common,
                         CONTROL: ("K_A", "K_B")[(first_label + observation) % 2]})
    return pd.DataFrame(rows)


def scenario_spec(seed):
    return {
        "scenario": SCENARIO, "seed": seed, "row_count": 600, "group_count": 300,
        "observations_per_group": 2, "predictor_columns": list(FEATURES),
        "target_columns": list(TARGETS), "negative_control_target": CONTROL,
        "profiling_columns": list(PROFILE_COLUMNS), "model_thresholds": THRESHOLDS,
        "reference_mapping": [{"kanaltyp": channel, "detailstufe": detail, **reference_values(channel, detail)}
                              for channel, detail in product(("RGB", "RGBI"), ("standard", "fein"))],
        "control_design": "Bei identischen Prädiktoren je Gruppe einmal K_A und einmal K_B; keine vorhersagbare Zuordnung.",
        "injection_seed": seed + 1000, "records_per_change_type": 8,
        "error_types": list(ERROR_TYPES), "valid_change_types": list(VALID_CHANGES),
        "test_design": "Gepaarter Vergleich: dieselben reservierten Testdatensätze unverändert und in einer veränderten Kopie.",
        "scope": "Funktionstest von Profiling, Baumregeln und Regelprüfung; keine Messung realer DOP- oder ALKIS-Qualität.",
        "assumptions": "Alle Zuordnungen sind vorab festgelegte synthetische Testannahmen, keine Produktspezifikation.",
    }


def learn_rules(frame, split, settings, fw):
    """Nur Training und Validierung gehen an die Lernkomponente."""
    train = fw.phase2_split.select_partition(frame, split, "train")
    validation = fw.phase2_split.select_partition(frame, split, "validation")
    groups = split.set_index("record_key")["group_id"]
    profile = fw.profiling.profile_dataframe(
        train[list(PROFILE_COLUMNS)], numeric_columns=["bandanzahl", "pixelgroesse_cm", "crs_code"],
        date_columns=[], high_cardinality_threshold=20)
    profile = fw.rule_engine.attach_single_column_rule_proposals(
        profile, const_stddev_range_threshold=.05, max_numeric_distinct_for_in_rule=10,
        enable_string_rules=False, min_regex_support_ratio=.95, enable_string_length_rules=False)
    profiling = profile["proposed_rules"]
    for rule in profiling:
        rule.update({"evidence_type": "synthetic_training_profile",
                     "evidence_detail": "Automatisch aus genau einem Profil des Trainingsanteils erzeugt; synthetischer Funktionstest."})
        rule["metadata"].update({"learning_scope": "training_partition_only", "training_rows": len(train)})

    dependencies, candidates, reports = [], [], []
    for target in (*TARGETS, CONTROL):
        selected, report = fw.phase2_ml.train_dependency(
            train, validation, target, list(FEATURES), groups, settings,
            THRESHOLDS["accuracy"], THRESHOLDS["f1_macro"], THRESHOLDS["precision_macro"])
        dependencies.extend(selected)
        candidates.extend(report["candidate_rules"])
        reports.append({k: v for k, v in report.items() if k != "candidate_rules"})
    return {"profiling": profiling, "dependency_eligible": dependencies,
            "dependency_all_candidates": candidates}, reports


def make_test_copy(clean, seed):
    """Fehleretiketten bleiben separat. Geändert werden nur reservierte Testdaten."""
    clean = clean.sort_values("record_key").reset_index(drop=True).copy(deep=True)
    mixed = clean.copy(deep=True)
    mixed["bandanzahl"] = mixed["bandanzahl"].astype("Int64")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(clean))
    types = (*ERROR_TYPES, *VALID_CHANGES)
    if len(clean) < 8 * len(types):
        raise ValueError("Zu wenige Testdatensätze für die festgelegten Änderungstypen.")
    labels = {str(key): {"record_key": str(key), "is_injected_error": False, "case_type": "unveraendert"}
              for key in clean.record_key}
    changes = []
    for number, case_type in enumerate(types):
        for index in indices[number * 8:(number + 1) * 8]:
            row = clean.loc[index]
            if case_type == "bandzahl_falsche_zuordnung":
                replacements = {"bandanzahl": 7 - int(row.bandanzahl)}
            elif case_type == "pixelgroesse_falsche_zuordnung":
                replacements = {"pixelgroesse_cm": 30 - int(row.pixelgroesse_cm)}
            elif case_type == "exportprofil_falsche_zuordnung":
                choices = ["P_A", "P_B", "P_C", "P_D"]
                replacements = {"exportprofil": choices[(choices.index(row.exportprofil) + 1) % 4]}
            elif case_type == "crs_konstanz_verletzt":
                replacements = {"crs_code": 25833}
            elif case_type == "bandzahl_fehlt":
                replacements = {"bandanzahl": None}
            elif case_type == "gueltiger_produktwechsel":
                channel = "RGBI" if row.kanaltyp == "RGB" else "RGB"
                values = reference_values(channel, row.detailstufe)
                replacements = {"kanaltyp": channel, "bandanzahl": values["bandanzahl"],
                                "exportprofil": values["exportprofil"]}
            else:
                replacements = {"zufall_zahl": float((row.zufall_zahl + .5) % 1),
                                "zufall_gruppe": "Z_A" if row.zufall_gruppe != "Z_A" else "Z_B"}
            key = str(row.record_key)
            labels[key].update(is_injected_error=case_type in ERROR_TYPES, case_type=case_type)
            for field, new in replacements.items():
                changes.append({"record_key": key, "case_type": case_type,
                                "is_injected_error": case_type in ERROR_TYPES,
                                "field": field, "old_value": row[field], "new_value": new})
                mixed.at[index, field] = new
    return mixed, list(labels.values()), changes


def evaluate(frame, rules, fw):
    keys = frame.record_key.astype(str).tolist()
    if len(set(keys)) != len(keys):
        raise ValueError("Doppelte record_key in der Prüfpopulation.")
    found = {key: [] for key in keys}
    applicable = {key: [] for key in keys}
    rule_metrics = []
    rule_ids = [rule["rule_id"] for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("Regelkennungen sind nicht eindeutig.")
    for rule in rules:
        applies, valid, invalid = fw.rule_engine.evaluate_rule_on_dataframe(frame, rule)
        for key in frame.loc[applies, "record_key"].astype(str):
            applicable[key].append(rule["rule_id"])
        for key in frame.loc[invalid, "record_key"].astype(str):
            found[key].append(rule["rule_id"])
        count = int(applies.sum())
        rule_metrics.append({"rule_id": rule["rule_id"], "population_count": len(frame),
                             "coverage_count": count, "valid_count": int(valid.sum()),
                             "invalid_count": int(invalid.sum()),
                             "pass_rate": float(valid.sum() / count) if count else None})
    return {"findings": found, "applicable": applicable, "rule_metrics": rule_metrics}


def detection_metrics(findings, truth):
    truth_keys = [item["record_key"] for item in truth]
    if len(set(truth_keys)) != len(truth_keys) or set(truth_keys) != set(findings):
        raise ValueError("Fehlerreferenz und Ergebnis haben unterschiedliche oder doppelte record_key.")
    if any(type(item["is_injected_error"]) is not bool for item in truth):
        raise ValueError("Fehlerreferenz muss boolesche Werte enthalten.")
    tp = fp = tn = fn = 0
    for item in truth:
        expected, flagged = item["is_injected_error"], bool(findings[item["record_key"]])
        tp += int(expected and flagged)
        fn += int(expected and not flagged)
        fp += int(not expected and flagged)
        tn += int(not expected and not flagged)
    return {"rows": len(truth), "injected_errors": tp + fn,
            "true_positives": tp, "false_negatives": fn, "false_positives": fp, "true_negatives": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def check_relations(rules, fw):
    """Semantische Prüfung aller vier Kombinationen, unabhängig vom Baumwortlaut."""
    rows = []
    for index, (channel, detail) in enumerate(product(("RGB", "RGBI"), ("standard", "fein"))):
        rows.append({"record_key": record_key(index, 0, "semantic_probe_v1"),
                     "kanaltyp": channel, "detailstufe": detail, "zufall_zahl": .314,
                     "zufall_gruppe": "Z_A", **reference_values(channel, detail)})
    reference = pd.DataFrame(rows)
    results = []
    for target in TARGETS:
        selected = [rule for rule in rules["dependency_eligible"] if rule["metadata"]["target_col"] == target]
        clean = evaluate(reference, selected, fw)
        changed = reference.copy(deep=True)
        values = sorted(reference[target].unique().tolist())
        changed[target] = changed[target].map({value: values[(i + 1) % len(values)] for i, value in enumerate(values)})
        wrong = evaluate(changed, selected, fw)
        recovered = bool(selected) and all(clean["applicable"].values()) and not any(clean["findings"].values()) and all(wrong["findings"].values())
        results.append({"target": target, "mapping_combinations": len(reference),
                        "selected_rules": len(selected), "recovered": recovered})
    return results


def assess(clean, mixed, truth, rules, reports, fw):
    sources = {"profiling": rules["profiling"], "ml": rules["dependency_eligible"],
               "combined": rules["profiling"] + rules["dependency_eligible"]}
    clean_truth = [{"record_key": key, "is_injected_error": False} for key in clean.record_key]
    evaluations, metrics = {}, {}
    for source, selected in sources.items():
        evaluations[source] = {"clean": evaluate(clean, selected, fw), "mixed": evaluate(mixed, selected, fw)}
        metrics[source] = {"clean": detection_metrics(evaluations[source]["clean"]["findings"], clean_truth),
                           "mixed": detection_metrics(evaluations[source]["mixed"]["findings"], truth)}
    by_type = []
    for case_type in ("unveraendert", *ERROR_TYPES, *VALID_CHANGES):
        subset = [row for row in truth if row["case_type"] == case_type]
        flags = {source: sum(bool(evaluations[source]["mixed"]["findings"][row["record_key"]]) for row in subset)
                 for source in sources}
        by_type.append({"case_type": case_type, "rows": len(subset), "is_injected_error": case_type in ERROR_TYPES,
                        "flagged_by_source": flags})
    relations = check_relations(rules, fw)
    control_report = next(report for report in reports if report["target_col"] == CONTROL)
    control_rejected = not control_report["model_gate_passed"] and control_report["rule_count"] == 0
    constant_found = any(r["rule_type"] == "constant" and r["columns"] == ["crs_code"]
                         and r["metadata"].get("value") == 25832 for r in rules["profiling"])
    combined = metrics["combined"]
    checks = {
        "all_three_reference_relations_recovered": all(row["recovered"] for row in relations),
        "constant_crs_learned_by_profiling": constant_found,
        "uninformative_control_not_selected": control_rejected,
        "no_flags_on_clean_test": combined["clean"]["false_positives"] == 0,
        "all_injected_errors_detected": combined["mixed"]["false_negatives"] == 0 and combined["mixed"]["injected_errors"] == 40,
        "no_flags_on_valid_cases_in_mixed_test": combined["mixed"]["false_positives"] == 0,
    }
    cases = [{**row, "violated_rule_ids": {source: evaluations[source]["mixed"]["findings"][row["record_key"]]
                                          for source in sources}} for row in truth]
    summary = {"scenario": SCENARIO, "function_check_passed": all(checks.values()), "checks": checks,
               "profiling_rule_count": len(rules["profiling"]),
               "eligible_ml_rule_count": len(rules["dependency_eligible"]),
               "all_ml_candidate_count": len(rules["dependency_all_candidates"]),
               "relations": relations, "control_validation_accuracy": control_report["validation_metrics"]["accuracy"],
               "metrics": metrics, "by_case_type": by_type,
               "test_used_for_learning_or_selection": False,
               "synthetic_test_evaluated_after_rule_freeze": True,
               "scope": "Kontrollierter Funktionstest; keine Leistungsbewertung auf echten DOP- oder ALKIS-Lieferungen."}
    return summary, evaluations, cases


def html_table(headers, rows):
    return ("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def write_report(path, summary, spec, rules, reports, counts, changes):
    passed = summary["function_check_passed"]
    state = "Funktionstest bestanden" if passed else "Funktionstest: Abweichungen prüfen"
    document = ["<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
                "<title>Automatische Regelableitung – synthetischer Funktionstest</title>",
                "<style>body{margin:0;font:16px/1.55 system-ui,sans-serif;color:#17383f;background:#f2f6f7}header{background:#123c42;color:#fff;padding:32px max(24px,calc((100vw - 1140px)/2))}main{max-width:1140px;margin:24px auto;padding:0 24px 48px}section{background:#fff;border:1px solid #d5e0e2;border-radius:10px;padding:24px;margin:20px 0}h1{line-height:1.2}h2{font-size:1.3rem}code{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #dce4e6;padding:10px;overflow-wrap:anywhere}th{background:#ecf3f4}.table-wrap{overflow-x:auto}.note{background:#eef5f4;padding:16px;border-left:4px solid #18736b}.warning{background:#fff4d9;border-color:#b58215}.minor{font-size:14px;color:#52686d}.rule{padding:16px 0;border-top:1px solid #dce4e6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}@media print{header{background:white;color:black;padding:0}body{background:white}section{break-inside:avoid}main{margin:0;padding:0}}</style></head><body>",
                f"<header><p>VALIDIERUNGSFRAMEWORK · TESTPAKET 05</p><h1>{state}</h1><p>Automatisch gelernte Regeln gegen vorher bekannte Zusammenhänge prüfen.</p></header><main>",
                "<section><h2>Was dieser Versuch zeigt</h2><p>Der vorhandene Framework-Code erzeugt die konkreten Regeln aus Trainingsdaten. Die bekannten Zusammenhänge dienen ausschließlich der Datenerzeugung und dem anschließenden Soll-Ist-Vergleich.</p>",
                "<p class='note warning'>Alle Daten und Zuordnungen sind künstlich. Ein bestandener Versuch bestätigt das Verhalten bei diesen einfachen Fällen. Er belegt keine Erkennungsleistung für echte DOP- oder ALKIS-Lieferungen. Autoencoder, LOF, Clustering und Datei-/Geometrieeinlesung werden hier nicht geprüft.</p>"]
    document.append(html_table(["Teilmenge", "Datensätze", "Gruppen"], [[name, item["rows"], item["groups"]] for name, item in counts.items()]))
    document.append("<p>Training lernt Regeln und wählt Baumparameter durch Gruppen-Cross-Validation. Die Validierung entscheidet über die unveränderten Güteschwellen. Danach werden die Regeln eingefroren und auf den reservierten Testdaten geprüft. Beide Beobachtungen einer Einheit bleiben immer in derselben Teilmenge.</p></section>")
    document.append("<section><h2>1. Vorab festgelegte Referenz</h2><p>Die Merkmals- und Zielspalten sind für diesen Versuch festgelegt. Die konkreten Wenn-dann-Bedingungen und Werte werden vom Framework gelernt. Die Profilcodes sind frei erfunden.</p>")
    document.append(html_table(["Kanaltyp", "Detailstufe", "Bandanzahl", "Pixelgröße (cm)", "Exportprofil"],
                               [[r[k] for k in ("kanaltyp", "detailstufe", "bandanzahl", "pixelgroesse_cm", "exportprofil")] for r in spec["reference_mapping"]]))
    document.append("<p>Zusätzlich ist <code>crs_code = 25832</code> konstant. Zwei irrelevante Zufallsmerkmale stehen als mögliche Prädiktoren zur Verfügung. Die Kontrollgröße K_A/K_B ist bei identischen Prädiktoren paarweise ausgeglichen und daher nicht vorhersagbar.</p></section>")
    document.append("<section><h2>2. Tatsächlich automatisch erzeugte Regeln</h2>")
    document.append(f"<p><strong>{summary['profiling_rule_count']} Profiling-Vorschläge</strong> und <strong>{summary['eligible_ml_rule_count']} ML-Regeln mit bestandenen Güteschwellen</strong>. Die Kontrollgröße hat eine Validierungs-Accuracy von {summary['control_validation_accuracy']:.1%}. Alle Regeln bleiben Vorschläge.</p>")
    document.append(html_table(["Ziel", "Kandidaten", "Mit bestandenen Schwellen", "Accuracy", "Macro-F1"],
                               [[r["target_col"], r["candidate_rule_count"], r["rule_count"], f"{r['validation_metrics']['accuracy']:.3f}", f"{r['validation_metrics']['f1_macro']:.3f}"] for r in reports]))
    for rule in rules["dependency_eligible"]:
        validation = rule["metadata"]["validation"]
        document.append(f"<div class='rule'><strong>{escape(rule['rule_id'])}</strong><p>{escape(rule['expression'])}</p><p class='minor'>Validierung: {validation['valid_count']} von {validation['coverage_count']} anwendbaren Fällen stimmen überein.</p></div>")
    document.append("<details><summary>Automatisch erzeugte Profiling-Vorschläge ansehen</summary>" + html_table(["Kennung", "Ausdruck"], [[r["rule_id"], r["expression"]] for r in rules["profiling"]]) + "</details></section>")
    document.append("<section><h2>3. Erkennung auf reservierten Testdaten</h2><p>Verglichen werden die unveränderten Testdatensätze und eine Kopie mit 40 absichtlichen Fehlern, 16 gültigen Änderungen und 64 unveränderten Fällen. Jede Fehlerzeile hat genau einen injizierten Fehler. Gezählt werden Datensätze, nicht einzelne Regelmeldungen.</p>")
    metrics_rows = []
    for source, label in (("profiling", "Nur Profiling"), ("ml", "Nur ML-Regeln"), ("combined", "Gemeinsam")):
        current = summary["metrics"][source]
        mixed = current["mixed"]
        metrics_rows.append([label, current["clean"]["false_positives"], f"{mixed['true_positives']}/{mixed['injected_errors']}", mixed["false_negatives"], mixed["false_positives"], f"{mixed['recall']:.1%}"])
    document.append(html_table(["Prüfung", "Meldungen im sauberen Test", "Erkannte Fehler", "Übersehene Fehler", "Fehlalarme im gemischten Test", "Recall"], metrics_rows))
    document.append(html_table(["Falltyp", "Fälle", "Absichtlicher Fehler?", "Profiling meldet", "ML meldet", "Gemeinsam gemeldet"],
                               [[CASE_LABELS[r["case_type"]], r["rows"], "Ja" if r["is_injected_error"] else "Nein", *[r["flagged_by_source"][s] for s in ("profiling", "ml", "combined")]] for r in summary["by_case_type"]]))
    document.append("<p class='note'>Bei den ersten drei Fehlerarten bleibt der falsche Wert in der bekannten Wertemenge. Damit wird gezielt geprüft, ob die gelernte Abhängigkeit erkannt wird. Ein korrekt abgestimmter Produktwechsel soll dagegen keine Meldung auslösen.</p></section>")
    document.append("<section><h2>4. Nachvollziehbare Prüfkriterien</h2>")
    check_labels = {"all_three_reference_relations_recovered": "Alle drei Zuordnungen auf sämtlichen vier Referenzkombinationen wiedergefunden",
                    "constant_crs_learned_by_profiling": "Konstanz automatisch aus Trainingsprofil abgeleitet",
                    "uninformative_control_not_selected": "Uninformative Kontrollgröße hat keine ausgewählte ML-Regel",
                    "no_flags_on_clean_test": "Keine Meldung auf den 120 unveränderten Testfällen",
                    "all_injected_errors_detected": "Alle 40 gezielt eingebrachten Fehler gemeinsam erkannt",
                    "no_flags_on_valid_cases_in_mixed_test": "Keine Meldung auf gültigen Fällen einschließlich gültiger Änderungen"}
    document.append(html_table(["Prüfkriterium", "Ergebnis"], [[check_labels[k], "Bestanden" if v else "Nicht bestanden"] for k, v in summary["checks"].items()]))
    document.append("<p>Der Bezugstest prüft das Verhalten der Regeln, nicht einen bestimmten Text oder eine bestimmte Reihenfolge der Baumpfade. Auch anders formulierte, gleich wirkende Regeln können bestehen.</p></section>")
    document.append("<section><h2>5. Änderungen und Dokumentation</h2><details><summary>Alle veränderten Werte ansehen</summary>")
    document.append(html_table(["Datensatzkennung", "Falltyp", "Feld", "Vorher", "Nachher"],
                               [[r["record_key"], CASE_LABELS[r["case_type"]], r["field"], r["old_value"], r["new_value"]] for r in changes]))
    document.append("</details><p>Die Parquet-Dateien enthalten nur Eingabemerkmale. Fehleretiketten stehen separat in <code>ground_truth.json</code>; sie werden den Lernfunktionen nicht übergeben. <code>learned_rules.json</code>, <code>training_report.json</code>, <code>case_results.json</code> und <code>summary.json</code> dokumentieren Regeln und Ergebnisse. <code>run_manifest.json</code> hält Parameter, Versionen und Dateiprüfsummen fest.</p></section>")
    document.append("<section><h2>Danach: echte Lieferungen</h2><ol><li>DOP: Regeln auf einem festgelegten Referenzbestand lernen und einfrieren; Originalbestand und separat protokollierte Fehlerkopien prüfen.</li><li>ALKIS: eine Objektart auswählen und Untersuchungseinheiten, Merkmale sowie fachliche Prüfungen dafür definieren.</li><li>Ergebnisse nach Fehlerart, Fehlalarmen und Grenzen vergleichen. DOP und ALKIS getrennt modellieren; der Ablauf lässt sich wiederverwenden.</li></ol><p>Die bisherigen 500 DOP-Datensätze wurden bereits untersucht und vom Autoencoder verwendet. Ergebnisse darauf sind explorativ; eine unberührte Gesamtframework-Testmenge lässt sich daraus nachträglich nicht machen.</p></section></main></body></html>")
    path.write_text("\n".join(document), encoding="utf-8")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(project, output=None, seed=42):
    fw = load_framework(project)
    settings = fw.phase2_settings.Phase2Settings(group_columns=("test_group",), random_state=seed)
    frame = create_reference(seed)
    original = frame.copy(deep=True)
    split = fw.phase2_split.make_grouped_split(frame, settings)
    parts = {name: fw.phase2_split.select_partition(frame, split, name) for name in ("train", "validation", "test")}
    counts = {name: {"rows": len(part), "groups": part.test_group.nunique()} for name, part in parts.items()}
    # Prüfen, ob Parquet verfügbar ist, bevor ein Ausgabeverzeichnis angelegt wird.
    pd.io.parquet.get_engine("auto")
    source_hashes = {name: sha_file(fw.source / name) for name in REQUIRED_SOURCES}
    base = Path(output).resolve() if output is not None else fw.project / "data" / "synthetic_rule_check_05"
    folder = base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    write_json = fw.review_common.write_json
    digest = fw.review_common.digest
    write_json(folder / "run_status.json", {"status": "running", "scenario": SCENARIO})
    spec = scenario_spec(seed)
    write_json(folder / "reference_spec.json", spec)
    frame.to_parquet(folder / "synthetic_reference.parquet", index=False)
    split.to_csv(folder / "split_manifest.csv", index=False, encoding="utf-8")
    for name in ("train", "validation"):
        parts[name].to_parquet(folder / f"{name}.parquet", index=False)
    parts["test"].to_parquet(folder / "test_clean.parquet", index=False)
    print("Erzeuge 600 künstliche Datensätze aus 300 Einheiten.")
    print("Aufteilung:", counts)
    print("Lerne Profiling- und Baumregeln mit dem vorhandenen Framework-Code ...")
    rules, reports = learn_rules(frame, split, settings, fw)
    # Einfrieren und speichern VOR Erzeugung oder Bewertung der Fehlerkopie.
    write_json(folder / "learned_rules.json", rules)
    write_json(folder / "training_report.json", reports)
    frozen_hash = digest(rules)
    clean = parts["test"].sort_values("record_key").reset_index(drop=True)
    mixed, truth, changes = make_test_copy(clean, seed + 1000)
    summary, evaluations, cases = assess(clean, mixed, truth, rules, reports, fw)
    if not frame.equals(original) or digest(rules) != frozen_hash:
        raise RuntimeError("Referenzdaten oder eingefrorene Regeln wurden während der Prüfung verändert.")
    summary.update({"partition_counts": counts, "group_overlap_count": 0,
                    "rules_frozen_sha256": frozen_hash, "seed": seed,
                    "reference_data_unchanged": True, "frozen_rules_unchanged": True})
    mixed.to_parquet(folder / "test_mixed.parquet", index=False)
    for name, value in (("ground_truth.json", truth), ("test_changes.json", changes),
                        ("summary.json", summary), ("rule_evaluations.json", evaluations), ("case_results.json", cases)):
        write_json(folder / name, value)
    write_report(folder / "bericht.html", summary, spec, rules, reports, counts, changes)
    write_json(folder / "run_status.json", {"status": "completed", "function_check_passed": summary["function_check_passed"]})
    manifest = {"scenario": SCENARIO, "created_at": datetime.now(timezone.utc).isoformat(),
                "project": str(fw.project), "script_sha256": sha_file(__file__),
                "framework_source_sha256": source_hashes,
                "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scikit-learn")}},
                "settings": asdict(settings), "scenario_spec_sha256": digest(spec),
                "rules_frozen_sha256": frozen_hash, "selection_did_not_use_test": True,
                "evaluation_mode": "synthetic_paired_function_check",
                "artifact_sha256": {p.name: sha_file(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write_json(folder / "run_manifest.json", manifest)
    print(json.dumps({"function_check_passed": summary["function_check_passed"],
                      "profiling_rule_count": summary["profiling_rule_count"],
                      "eligible_ml_rule_count": summary["eligible_ml_rule_count"],
                      "reference_relations_recovered": sum(r["recovered"] for r in summary["relations"]),
                      "control_rejected": summary["checks"]["uninformative_control_not_selected"],
                      "detected_errors": summary["metrics"]["combined"]["mixed"]["true_positives"],
                      "missed_errors": summary["metrics"]["combined"]["mixed"]["false_negatives"],
                      "false_alarms": summary["metrics"]["combined"]["mixed"]["false_positives"]}, indent=2))
    print("Ergebnisordner:", folder)
    print("Bericht im Browser öffnen:", folder / "bericht.html")
    return summary, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Vorhandener validierungs-framework-Projektordner")
    parser.add_argument("--output", type=Path, help="Optionaler Basisordner; jeder Lauf bekommt einen neuen Unterordner")
    parser.add_argument("--seed", type=int, default=42, help="Fester Startwert, Standard 42; nicht nach günstigen Ergebnissen auswählen")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1002:
        parser.error("seed muss zwischen 0 und 4294966294 liegen.")
    try:
        summary, _ = run(args.project, args.output, args.seed)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Test nicht abgeschlossen: {exc}\n")
    return 0 if summary["function_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
