"""Kontrollierter Vergleich. Alle fit-Funktionen erhalten ausschließlich Training.

Keine Geometrieimporte. Lokale Regeln sind Umschläge mit einer obligatorischen
Scope-Bedingung; der Framework-Regelprüfer erhält nur den korrekt zugeordneten
Teilbestand, niemals eine lokale Regel auf dem Gesamtbestand.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phase2_ml import (build_decision_tree_pipeline, evaluate_predictions,
                       prepare_features, get_phase2_target_candidates)
from rule_engine import evaluate_rule_on_dataframe
from tree_rule_export import extract_tree_rules

METHODS = ("global", "attribute_clusters", "latent_clusters")
LABELS = {"global": "Globale Regeln", "attribute_clusters": "Regeln in Attributclustern",
          "latent_clusters": "Regeln in Latentclustern"}
MISSING = "__MISSING__"
DATE = "datumderletztenueberpruefung"
DATE_MISSING = DATE + "_fehlt"


@dataclass(frozen=True)
class Protocol:
    epochs: int = 5
    latent_dim: int = 8
    min_cluster_size: int = 50
    min_samples: int = 10
    max_depth: int = 4
    min_samples_leaf: int = 5
    min_train_rows: int = 50
    min_train_groups: int = 3
    min_rule_cases: int = 5
    min_rule_groups: int = 5
    min_agreement: float = .95
    min_accuracy: float = .95
    min_precision: float = .90
    min_f1: float = .90


def model_table(attributes):
    """Deterministische Typumwandlung, keine geschätzten Statistiken."""
    result = attributes.copy(deep=True)
    if DATE in result:
        parsed = pd.to_datetime(result[DATE], format="mixed", errors="raise", utc=True)
        result[DATE] = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / 86400
    return result


def eligible_columns(train, allowed):
    selected, excluded = [], {}
    for col in allowed:
        if col not in train:
            continue
        s = train[col]
        if s.notna().sum() == 0:
            excluded[col] = "empty_in_training"
        elif s.nunique(dropna=False) <= 1:
            excluded[col] = "constant_in_training"
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
            excluded[col] = "text_over_20_training_categories"
        else:
            if not pd.api.types.is_numeric_dtype(s) and s.dropna().eq(MISSING).any():
                raise ValueError(f"Reservierter Fehlwert-Text in Originalwerten: {col}")
            selected.append(col)
    return selected, excluded


def targets_from_training(train, columns):
    # Datumswerte sind Prädiktoren, keine Klassifikationsziele.
    return sorted(c for c in get_phase2_target_candidates(train[columns]) if c != DATE)


def predictors_for_target(train, columns, target):
    excluded = {target: "target"}
    if target in {DATE, DATE_MISSING}:
        excluded.update({c: "known_target_derivative_family" for c in (DATE, DATE_MISSING)})
    for col in columns:
        if col in excluded:
            continue
        left, right = train[col], train[target]
        equal = (left.eq(right).fillna(False) | (left.isna() & right.isna())).all()
        if equal:
            excluded[col] = "exact_target_copy_in_training"
    return [c for c in columns if c not in excluded], excluded


def feature_types(frame, features):
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c])]
    return numeric, [c for c in features if c not in numeric]


def fit_preprocessor(train, features):
    if not features:
        raise ValueError("Keine vom Ziel getrennten Prädiktoren vorhanden.")
    numeric, categorical = feature_types(train, features)
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], sparse_threshold=0)
    X = prepare_features(train, features, numeric, categorical)
    matrix = np.asarray(prep.fit_transform(X), dtype=np.float32)
    if not np.isfinite(matrix).all() or not matrix.shape[1]:
        raise ValueError("Ungültige Trainingsmatrix.")
    return prep, matrix, {"features": list(features), "numeric": numeric,
                          "categorical": categorical, "dimensions": matrix.shape[1]}


def transform(prep, frame, schema):
    X = prepare_features(frame, schema["features"], schema["numeric"], schema["categorical"])
    matrix = np.asarray(prep.transform(X), dtype=np.float32)
    if not np.isfinite(matrix).all() or matrix.shape[1] != schema["dimensions"]:
        raise ValueError("Ungültige eingefrorene Transformation.")
    unknown = {}
    if schema["categorical"]:
        encoder = prep.named_transformers_["cat"]
        for col, categories in zip(schema["categorical"], encoder.categories_):
            unknown[col] = int((~X[col].isin(categories)).sum())
    return matrix, unknown


def encode(model, matrix):
    import torch
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(matrix), 256):
            z = model(torch.tensor(matrix[start:start+256], dtype=torch.float32), [])["z_shared"]
            chunks.append(z.detach().cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("Nicht endlicher latenter Raum.")
    return result


def fit_router(matrix, p):
    import hdbscan
    if len(matrix) < max(p.min_cluster_size, p.min_samples + 1):
        return None, np.full(len(matrix), -1, dtype=int), {"reason": "too_few_training_rows"}
    router = hdbscan.HDBSCAN(min_cluster_size=p.min_cluster_size, min_samples=p.min_samples,
        metric="euclidean", cluster_selection_method="eom", prediction_data=True,
        core_dist_n_jobs=1, approx_min_span_tree=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        router.fit(np.asarray(matrix, dtype=np.float64))
    labels = router.labels_.astype(int)
    return router, labels, {"warnings": [str(w.message) for w in caught],
        "cluster_sizes": {str(c): int((labels == c).sum()) for c in sorted(set(labels)) if c >= 0},
        "training_noise": int((labels == -1).sum())}


def route(router, matrix):
    import hdbscan
    if router is None or not np.any(router.labels_ >= 0):
        return np.full(len(matrix), -1, dtype=int), np.zeros(len(matrix))
    labels, strengths = hdbscan.approximate_predict(router, np.asarray(matrix, dtype=np.float64))
    if not np.isfinite(strengths).all():
        raise ValueError("Nicht endliche HDBSCAN-Zuordnungsstärke.")
    return labels.astype(int), strengths


def constant_rule(target, value, count):
    return {"rule_id": "constant", "rule_name": f"Konstante Vorhersage für {target}",
        "rule_type": "ml_dependency", "columns": [target],
        "expression": f"IF TRUE THEN {target} = {value!r}", "source": "comparison_11",
        "status": "proposed", "origin_runs": [],
        "metadata": {"target_col": target, "predicted_value": value, "conditions": [],
            "condition_schema_version": 1, "leaf_sample_count": count, "constant_prediction": True}}


def rule_evidence(frame, inner, group_map):
    applicable, valid, invalid = evaluate_rule_on_dataframe(frame, inner)
    target = inner["metadata"]["target_col"]
    missing = applicable & frame[target].isna()
    groups = frame.loc[applicable, "record_key"].map(group_map)
    if groups.isna().any():
        raise ValueError("Fehlende Gruppenzuordnung in der Regelprüfung.")
    covered, correct = int(applicable.sum()), int(valid.sum())
    return {"population": len(frame), "cases": covered, "groups": int(groups.nunique()),
        "agree": correct, "disagree_observed": int((invalid & ~missing).sum()),
        "missing_target": int(missing.sum()), "agreement_all": correct / covered if covered else None}


def learn_scope(train, validation, target, features, group_map, method, cluster, seed, p):
    """Fester Baum; ausschließlich Validierung entscheidet die Auswahl."""
    report = {"method": method, "cluster": cluster, "target": target,
        "train_rows": len(train), "validation_rows": len(validation),
        "model_gate_passed": False, "candidate_count": 0, "selected_count": 0}
    labelled = train.loc[train[target].notna()]
    report["labelled_training_rows"] = len(labelled)
    report["labelled_training_groups"] = int(labelled.record_key.map(group_map).nunique())
    if len(labelled) < p.min_train_rows or report["labelled_training_groups"] < p.min_train_groups:
        report["reason"] = "insufficient_training_support"
        return [], report
    usable = [c for c in features if labelled[c].notna().any() and labelled[c].nunique(dropna=False) > 1]
    report["rule_predictors"] = usable
    present_val = validation.loc[validation[target].notna()]
    majority = labelled[target].value_counts().index[0]
    if not usable or labelled[target].nunique() == 1:
        candidates = [constant_rule(target, majority, len(labelled))]
        predictions = np.repeat(majority, len(present_val))
        report["model_kind"] = "training_constant"
    else:
        numeric, categorical = feature_types(labelled, usable)
        model = build_decision_tree_pipeline(numeric, categorical, seed, p.max_depth, p.min_samples_leaf)
        model.fit(prepare_features(labelled, usable, numeric, categorical), labelled[target])
        predictions = model.predict(prepare_features(present_val, usable, numeric, categorical)) if len(present_val) else []
        candidates = extract_tree_rules(model, target, "fixed_tree_comparison_11", p.min_samples_leaf)
        if not candidates:
            clf = model.named_steps["classifier"]
            nodes = np.flatnonzero(clf.tree_.children_left == -1)
            classes = {int(np.argmax(clf.tree_.value[n][0])) for n in nodes}
            if len(classes) == 1:
                predicted = clf.classes_[next(iter(classes))]
                candidates = [constant_rule(target, str(predicted), len(labelled))]
        # Der exportierte Bedingungscode muss die reale Baumvorhersage wiedergeben.
        for inner in candidates:
            mask, _, _ = evaluate_rule_on_dataframe(validation, inner)
            if mask.any():
                exported = inner["metadata"]["predicted_value"]
                actual = model.predict(prepare_features(validation.loc[mask], usable, numeric, categorical))
                if not np.all(actual == exported):
                    raise AssertionError("Exportierte Regel weicht vom trainierten Baum ab.")
        report["model_kind"] = "fixed_decision_tree"
        report["leaf_count"] = int(model.named_steps["classifier"].get_n_leaves())
    metrics = evaluate_predictions(present_val[target], predictions)
    report["validation_model_metrics"] = metrics
    report["validation_majority_baseline"] = evaluate_predictions(present_val[target], np.repeat(majority, len(present_val)))
    gate = bool(metrics["n_evaluated"] and metrics["accuracy"] >= p.min_accuracy and
                metrics["precision_macro"] >= p.min_precision and metrics["f1_macro"] >= p.min_f1)
    report["model_gate_passed"] = gate
    envelopes = []
    for number, inner in enumerate(candidates):
        evidence = rule_evidence(validation, inner, group_map)
        reasons = []
        if not gate: reasons.append("model_validation_thresholds_not_met")
        if evidence["cases"] < p.min_rule_cases: reasons.append("too_few_validation_cases")
        if evidence["groups"] < p.min_rule_groups: reasons.append("too_few_validation_groups")
        if evidence["agreement_all"] is None or evidence["agreement_all"] < p.min_agreement:
            reasons.append("validation_agreement_not_met")
        rid = f"s{seed}__{method}__{target}__c{cluster}__r{number}"
        inner["rule_id"] = rid + "__inner"
        prefix = "Gesamter Bestand" if method == "global" else f"Eingefrorener Router für {target} → Cluster {cluster}"
        envelopes.append({"rule_id": rid, "rule_type": "scoped_dependency_comparison_11",
            "scope": {"method": method, "cluster": cluster, "target": target, "seed": seed},
            "expression": prefix + ": " + inner["expression"], "inner_rule": inner,
            "status": "proposed", "selected_on_validation": not reasons,
            "validation": evidence, "rejection_reasons": reasons,
            "test_used_for_selection": False,
            "note": "Statistischer Kandidat; keine fachliche Freigabe. Scope ist obligatorisch."})
    report.update(candidate_count=len(envelopes), selected_count=sum(r["selected_on_validation"] for r in envelopes),
                  reason="evaluated_on_validation")
    return envelopes, report


def learn_method(train, validation, labels_train, labels_validation, target, features, groups, method, seed, p):
    labels_train, labels_validation = np.asarray(labels_train), np.asarray(labels_validation)
    if len(labels_train) != len(train) or len(labels_validation) != len(validation):
        raise ValueError("Clusterzuordnung passt nicht zu Datensatzpositionen.")
    if method == "global" and (np.any(labels_train != 0) or np.any(labels_validation != 0)):
        raise ValueError("Globale Regeln benötigen den gesamten Bestand.")
    candidates, reports = [], []
    clusters = sorted(set(labels_train) - {-1})
    for cluster in clusters:
        found, info = learn_scope(train.loc[labels_train == cluster], validation.loc[labels_validation == cluster],
            target, features, groups, method, int(cluster), seed, p)
        candidates.extend(found)
        reports.append(info)
    return candidates, reports


def evaluate_selected(frame, labels, rules, target, groups, method, seed, p):
    labels = np.asarray(labels)
    if len(labels) != len(frame) or not frame.record_key.is_unique:
        raise ValueError("Ungültige Identität oder Clusterlänge bei Testauswertung.")
    predictions = [set() for _ in range(len(frame))]
    ids = [[] for _ in range(len(frame))]
    audits = []
    for envelope in rules:
        scope = envelope["scope"]
        if (scope["method"], scope["target"], scope["seed"]) != (method, target, seed):
            raise ValueError("Regel gehört zu einem anderen Modell/Target/Seed.")
        if not envelope["selected_on_validation"]:
            raise ValueError("Nicht ausgewählter Kandidat in der Testauswertung.")
        take = np.ones(len(frame), dtype=bool) if method == "global" else labels == scope["cluster"]
        subset = frame.loc[take]
        evidence = rule_evidence(subset, envelope["inner_rule"], groups)
        supported = (evidence["cases"] >= p.min_rule_cases and evidence["groups"] >= p.min_rule_groups)
        verdict = ("insufficient_support" if not supported else "supported" if
            evidence["agreement_all"] >= p.min_agreement else "agreement_threshold_not_met")
        audits.append({"rule_id": envelope["rule_id"], "evidence": evidence, "test_assessment": verdict})
        applies, _, _ = evaluate_rule_on_dataframe(subset, envelope["inner_rule"])
        predicted = envelope["inner_rule"]["metadata"]["predicted_value"]
        for pos in np.flatnonzero(take)[applies.to_numpy()]:
            predictions[pos].add(predicted)
            ids[pos].append(envelope["rule_id"])
    records = []
    for i, (_, row) in enumerate(frame.iterrows()):
        options = predictions[i]
        conflict = len(options) > 1
        covered = len(options) == 1
        predicted = next(iter(options)) if covered else None
        observed = pd.notna(row[target])
        agree = bool(covered and observed and predicted == row[target])
        records.append({"seed": seed, "method": method, "target": target, "record_key": row.record_key,
            "group_id": groups.loc[row.record_key], "cluster": int(labels[i]), "prediction": predicted,
            "observed_target": row[target], "covered": covered, "conflict": conflict,
            "agree": agree, "disagree_observed": bool(covered and observed and not agree),
            "missing_target_covered": bool(covered and not observed),
            "target_present": bool(observed), "rule_ids": "|".join(ids[i])})
    cases = pd.DataFrame(records)
    return cases, audits


def summarize_cases(cases):
    n, covered, agree = len(cases), int(cases.covered.sum()), int(cases.agree.sum())
    observed_covered = int((cases.covered & cases.target_present).sum())
    # Jede Gruppe erhält dasselbe Gewicht innerhalb jedes Regelziels.
    grouped = cases.groupby(["target", "group_id"])[["covered", "agree"]].mean()
    cov_g = float(grouped.covered.mean()) if len(grouped) else None
    agree_g = float(grouped.agree.mean()) if len(grouped) else None
    return {"target_record_pairs": n, "covered": covered, "agree": agree,
        "disagree_observed": int(cases.disagree_observed.sum()),
        "missing_target_covered": int(cases.missing_target_covered.sum()), "conflicts": int(cases.conflict.sum()),
        "abstained_including_conflicts": n-covered, "coverage": covered/n if n else None,
        "agreement_all_covered": agree/covered if covered else None,
        "agreement_observed_covered": agree/observed_covered if observed_covered else None,
        "correct_coverage": agree/n if n else None,
        "target_group_pairs": len(grouped), "group_mean_coverage": cov_g,
        "group_mean_correct_coverage": agree_g,
        "group_weighted_agreement": agree_g/cov_g if cov_g else None}


def paired_comparisons(cases):
    results = []
    for left, right in (("attribute_clusters", "global"), ("latent_clusters", "global"),
                        ("latent_clusters", "attribute_clusters")):
        a = cases.loc[cases.method.eq(left)].set_index(["seed", "target", "record_key"])
        b = cases.loc[cases.method.eq(right)].set_index(["seed", "target", "record_key"])
        if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
            raise ValueError("Methoden vergleichen nicht dieselben Ziel-Datensatz-Paare.")
        b = b.loc[a.index]
        added = a.covered & ~b.covered
        results.append({"method": left, "reference": right,
            "additional_covered": int(added.sum()), "additional_agree": int((added & a.agree).sum()),
            "additional_disagree_observed": int((added & a.disagree_observed).sum()),
            "additional_missing_target": int((added & a.missing_target_covered).sum()),
            "reference_only_agree": int((b.agree & ~a.covered).sum()),
            "net_agree_difference": int(a.agree.sum()-b.agree.sum()),
            "net_disagree_difference": int(a.disagree_observed.sum()-b.disagree_observed.sum())})
    return results
