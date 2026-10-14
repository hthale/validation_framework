"""Fallbezogene Hinweise zusammenführen, ohne neue Fehlerlabels abzuleiten."""
from __future__ import annotations

import math
import pandas as pd

from phase0_join import merge_phase0_frames
from phase2_split import validate_split
from preprocessing import normalize_by_schema
from record_identity import with_record_key, validate_record_keys
from review_common import digest, json_value, utc_now
from rule_engine import evaluate_rule_on_dataframe


def integrate_results(raw, phase0, split, settings, schema, rule_sets, observations, provenance,
                      review_fraction=.05, created_at=None):
    if not 0 < review_fraction <= 1:
        raise ValueError("Prüfanteil muss zwischen 0 und 1 liegen.")
    raw = with_record_key(raw).reset_index(drop=True)
    split = validate_split(raw, split, settings)
    merged = merge_phase0_frames(raw, phase0).reset_index(drop=True)
    normalized = normalize_by_schema(merged, schema)
    partitions = split.set_index("record_key")["partition"]
    rules_by_id, catalog = {}, []
    required_sets = {"domain", "profiling", "dependency_candidates", "dependency_eligible", "cluster"}
    if set(rule_sets) != required_sets:
        raise ValueError("Unvollständige Regelgruppen für die Integration.")
    for category in ("domain", "profiling", "dependency_candidates", "cluster"):
        rules = rule_sets[category]
        if not isinstance(rules, list):
            raise ValueError(f"Regelliste erwartet: {category}")
        for rule in rules:
            rid = rule.get("rule_id")
            if not isinstance(rid, str) or not rid.strip() or rid in rules_by_id:
                raise ValueError(f"Fehlende oder doppelte Regelkennung: {rid}")
            expected_source = {"domain": "domain", "profiling": "profiling", "dependency_candidates": "ml_phase2", "cluster": "ssl_cluster"}[category]
            if rule.get("source") != expected_source:
                raise ValueError(f"Falsche Regelherkunft für {rid}")
            rules_by_id[rid] = rule
            catalog.append({"category": category, "rule": rule, "integration_use": {
                "domain": "project_rule_observation", "profiling": "profiling_candidate_observation",
                "dependency_candidates": "not_applied", "cluster": "descriptive_only"}[category]})
    candidate_ids = {r["rule_id"] for r in rule_sets["dependency_candidates"]}
    eligible_ids = []
    for rule in rule_sets["dependency_eligible"]:
        rid = rule.get("rule_id")
        if rid in eligible_ids or rid not in candidate_ids or digest(rule) != digest(rules_by_id[rid]):
            raise ValueError("ML-Auswahlliste und Kandidatenkatalog passen nicht zusammen.")
        meta = rule.get("metadata", {})
        if (meta.get("eligible_for_review") is not True or meta.get("model_gate_passed") is not True
                or meta.get("evaluation_scope") != "validation_partition" or meta.get("test_evaluated") is not False):
            raise ValueError("ML-Kandidat hat die getrennte Validierung nicht bestanden.")
        eligible_ids.append(rid)
    for item in catalog:
        if item["rule"]["rule_id"] in eligible_ids:
            item["integration_use"] = "validation_partition_only"

    budget = int(math.ceil(len(raw) * review_fraction))
    rows, by_key = [], {}
    for record in merged.to_dict("records"):
        key = str(record["record_key"])
        part = str(partitions.loc[key])
        case = {"record_key": key, "display_name": next((str(record[c]) for c in ("kachelname", "flurstueckskennzeichen", "gml_id", "xml_name") if c in record and pd.notna(record[c])), key),
                "partition": part, "outlier_rank": int(record["outlier_rank"]),
                "lof_raw": float(record["lof_raw"]), "lof_z": float(record["lof_z"]),
                "lof_review_selected": int(record["outlier_rank"]) <= budget,
                "cluster_label": json_value(record["cluster_label"]),
                "domain_deviation_count": 0, "profiling_deviation_count": 0,
                "dependency_deviation_count": 0 if part == "validation" and eligible_ids else None,
                "dependency_evaluation_status": ("validation_only" if eligible_ids else "no_eligible_rules") if part == "validation" else "not_evaluated_" + part,
                "naming_observation_count": 0, "findings": [], "cluster_hints": [],
                "raw_values": json_value({c: record[c] for c in raw.columns})}
        rows.append(case)
        by_key[key] = case
    issues, assessments = [], []

    def apply_rules(frame, rules, category):
        for rule in rules:
            rid = rule["rule_id"]
            try:
                applicable, valid, invalid = evaluate_rule_on_dataframe(frame, rule)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append({"rule_id": rid, "category": category, "error": str(exc), "evaluation_status": "invalid_rule"})
                continue
            count = int(applicable.sum())
            assessments.append({"rule_id": rid, "category": category, "population_count": len(frame),
                                "coverage_count": count, "valid_count": int(valid.sum()), "invalid_count": int(invalid.sum()),
                                "pass_rate": int(valid.sum()) / count if count else None,
                                "evaluation_scope": "validation_partition" if category == "dependency" else "descriptive_full_population"})
            for index in frame.index[invalid]:
                row = frame.loc[index]
                case = by_key[str(row["record_key"])]
                case[category + "_deviation_count"] += 1
                evidence = {c: {"original": case["raw_values"].get(c), "evaluated": json_value(row[c])}
                            for c in rule["columns"] if c in row}
                case["findings"].append({"finding_id": "finding1:" + digest([case["record_key"], category, rid]),
                                         "category": category, "rule_id": rid, "expression": rule["expression"],
                                         "rule_status": rule.get("status", "unknown"), "evidence_type": rule.get("evidence_type"),
                                         "values": evidence, "quality_decision": "unreviewed"})
    apply_rules(normalized, rule_sets["domain"], "domain")
    apply_rules(normalized, rule_sets["profiling"], "profiling")
    validation_keys = set(split.loc[split["partition"].eq("validation"), "record_key"])
    apply_rules(merged.loc[merged["record_key"].isin(validation_keys)], rule_sets["dependency_eligible"], "dependency")

    observations = observations.copy()
    if "record_key" not in observations:
        raise ValueError("Beobachtungen benötigen record_key, auch bei leerer Tabelle.")
    if len(observations):
        observations["record_key"] = validate_record_keys(observations["record_key"], context="Benennungsbeobachtungen")
        unknown = set(observations["record_key"]) - set(by_key)
        if unknown:
            raise ValueError("Benennungsbeobachtungen gehören zu einem anderen Bestand.")
    for row in observations.to_dict("records"):
        case = by_key[str(row["record_key"])]
        case["naming_observation_count"] += 1
        case["findings"].append({"finding_id": "finding1:" + digest([case["record_key"], "naming"]),
                                 "category": "naming", "rule_id": None,
                                 "expression": str(row.get("observation", "Abweichende Bezeichnungen")),
                                 "values": json_value({k: v for k, v in row.items() if k not in {"record_key", "review_status"}}),
                                 "quality_decision": "unreviewed"})

    for rule in rule_sets["cluster"]:
        meta = rule.get("metadata", {})
        if meta.get("evaluation_scope") != "descriptive_in_sample":
            raise ValueError("Clusterbeschreibung besitzt keinen eindeutigen beschreibenden Geltungsbereich.")
        subset = merged.loc[merged["cluster_label"].isin(meta.get("included_clusters", [])) & merged["cluster_label"].ne(-1)]
        try:
            applicable, valid, _ = evaluate_rule_on_dataframe(subset, rule)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"rule_id": rule["rule_id"], "category": "cluster", "error": str(exc), "evaluation_status": "invalid_rule"})
            continue
        for index in subset.index[applicable]:
            case = by_key[str(subset.loc[index, "record_key"])]
            case["cluster_hints"].append({"rule_id": rule["rule_id"], "expression": rule["expression"],
                                           "predicted_cluster": json_value(meta["predicted_value"]),
                                           "matches_assigned_cluster": bool(valid.loc[index]), "evaluation_scope": "descriptive_in_sample"})
    rows.sort(key=lambda row: row["outlier_rank"])
    summary = {"row_count": len(rows), "unique_record_keys": len(by_key), "lof_review_budget": budget,
               "domain_deviating_records": sum(r["domain_deviation_count"] > 0 for r in rows),
               "profiling_deviating_records": sum(r["profiling_deviation_count"] > 0 for r in rows),
               "dependency_deviating_records": sum((r["dependency_deviation_count"] or 0) > 0 for r in rows),
               "naming_observation_records": sum(r["naming_observation_count"] > 0 for r in rows),
               "cluster_hint_records": sum(bool(r["cluster_hints"]) for r in rows),
               "cluster_description_agreement_records": sum(any(h["matches_assigned_cluster"] for h in r["cluster_hints"]) for r in rows),
               "dependency_candidates": len(rule_sets["dependency_candidates"]), "eligible_dependency_rules": len(eligible_ids),
               "rule_evaluation_issue_count": len(issues), "dependency_test_evaluated": False,
               "sort_policy": "outlier_rank ascending, unchanged LOF ranking",
               "review_fraction": review_fraction}
    result = json_value({"schema_version": 1, "created_at": created_at or utc_now(), "provenance": provenance,
                         "summary": summary, "cases": rows, "rule_catalog": catalog,
                         "rule_assessments": assessments, "rule_evaluation_issues": issues,
                         "scope_note": "Hinweise zur fachlichen Prüfung; keine automatische Fehlerbestätigung oder Regelfreigabe. Phase-2-Abhängigkeiten nur auf Validierung; Bestandsprofiling und vorhandene Cluster sind beschreibend."})
    result["bundle_id"] = "review1:" + digest(result)
    return result
