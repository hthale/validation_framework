"""Vorhandene Profiling- und Baumverfahren ohne verpflichtendes Produktprofil."""
from __future__ import annotations

import pandas as pd

from generic_schema import infer_columns, normalize_columns
from phase2_ml import get_phase2_target_candidates, train_dependency
from generic_clusters import describe_clusters
from phase2_split import make_grouped_split, select_partition, split_summary
from profiling import profile_dataframe
from rule_engine import attach_single_column_rule_proposals, merge_stable_proposed_rules, validate_rules_on_full_dataframe
from latentspace.phase0_integration import add_phase0_stratification_columns
from sampling import stratified_sample_dataframe


def learn_profiles(normalized, contract, latent, seed=42):
    # Nur für die Auswahl: Score und Cluster gelangen nicht in Profiling-Merkmale.
    strata = add_phase0_stratification_columns(latent[["record_key", "lof_raw", "cluster_label", "primary_score_name", "score_schema_version"]])
    work = normalized.merge(strata[["record_key", "stratum_key"]], on="record_key", validate="one_to_one", sort=False)
    numeric = [c for c, d in contract["columns"].items() if d["role"] == "numeric"]
    dates = [c for c, d in contract["columns"].items() if d["role"] == "date"]
    proposals, metadata = [], []
    for n in range(3):
        sample = stratified_sample_dataframe(work, .2, seed+n, min_per_stratum=1)
        profile = profile_dataframe(sample, numeric, dates, 20, excluded_columns=["record_key", "stratum_key"])
        profile = attach_single_column_rule_proposals(profile, .05, 10, True, .95, True)
        proposals.append(profile["proposed_rules"])
        metadata.append({"run": n+1, "seed": seed+n, "rows": len(sample), "scope": "descriptive_full_population"})
    rules = merge_stable_proposed_rules(proposals, 2)
    metrics, examples = validate_rules_on_full_dataframe(normalized, rules, max_example_violations=5)
    for r in rules:
        r.setdefault("metadata", {}).update(evaluation_scope="descriptive_full_population", approved=False)
    return rules, {"sampling": metadata, "metrics": metrics, "examples": examples}


def learn_dependencies(raw_with_groups, settings, grouping, excluded=()):
    empty = {"rules": [], "candidates": [], "reports": [], "contract": None, "split": None,
             "summary": {"status": "skipped", "reason": None, "test_evaluated": False}, "evaluation_frame": None}
    try:
        split = make_grouped_split(raw_with_groups, settings)
    except ValueError as exc:
        # Schema-/Gruppenfehler müssen bereits vor dem Lauf geprüft sein. Hier nur
        # unzureichende Populationsgröße als dokumentierten Nicht-Lernfall behandeln.
        if "sechs" not in str(exc) and "zu wenige Gruppen" not in str(exc):
            raise
        empty["summary"]["reason"] = str(exc)
        return empty
    raw_train = select_partition(raw_with_groups, split, "train")
    contract = infer_columns(raw_train, excluded=excluded, scope="training_partition_only")
    normalized, issues = normalize_columns(raw_with_groups, contract)
    # Keine Text-/Datumsmerkmale als opaque oder aus Gesamtbestand kodierte Baumfeatures.
    usable = [c for c, info in contract["columns"].items() if info["role"] in {"numeric", "categorical"}]
    model_data = normalized[["record_key", *usable]].copy()
    train = select_partition(model_data, split, "train")
    validation = select_partition(model_data, split, "validation")
    groups = split.set_index("record_key")["group_id"]
    targets = get_phase2_target_candidates(train)
    rules, candidates, reports = [], [], []
    scope = "grouped_validation" if grouping["mode"] == "user_supplied_groups" else "exploratory_validation"
    for target in targets:
        # Exakte Feldduplikate können die Zielausprägung direkt verraten.
        duplicates = [c for c in usable if c != target and train[c].equals(train[target])]
        features = [c for c in usable if c != target and c not in duplicates]
        accepted, report = train_dependency(train, validation, target, features, groups, settings, .95, .90, .90)
        report.update(grouping=grouping, interpretation_scope=scope, duplicate_target_columns_excluded=duplicates)
        for candidate in report["candidate_rules"]:
            candidate["reason"] = "Automatisch aus Trainingsdaten abgeleitet; auf getrennter Teilmenge geprüft. Fachliche Freigabe erforderlich."
            candidate["evidence_type"] = scope + "_candidate"
            candidate["evidence_detail"] = "Gruppen-CV zur Parameterwahl; keine Testauswertung. " + grouping["note"]
            candidate["metadata"].update(grouping=grouping, interpretation_scope=scope,
                                         training_schema_scope=contract["scope"], approved=False)
        rules.extend(accepted)
        candidates.extend(report.pop("candidate_rules"))
        reports.append(report)
        print(f"  Ziel {target}: {report.get('candidate_rule_count', 0)} Kandidaten, {len(accepted)} mit bestandenen Schwellen", flush=True)
    summary = split_summary(split, settings)
    summary.update(status="completed", selected_targets=targets, interpretation_scope=scope, grouping=grouping,
                   normalization_issues=issues, schema_scope="training_partition_only",
                   note="Keine fachliche Regelfreigabe. Testanteil nur für Abhängigkeitslernen reserviert; Autoencoder und Profiling beschreiben den Gesamtbestand.")
    return {"rules": rules, "candidates": candidates, "reports": reports, "contract": contract,
            "split": split, "summary": summary, "evaluation_frame": model_data}

