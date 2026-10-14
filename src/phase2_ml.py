"""Abhängigkeitsregeln: Gruppen-CV im Training, danach getrennte Validierung.

Phase-1-Stichproben werden nicht als Trainings-/Prüfpaare wiederverwendet.
Clusterbeschreibungen sind ausdrücklich eine separate In-Sample-Auswertung.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from date_roles import DOP_DATE_COLUMNS
from latentspace.anomaly_scores import SCORE_COLUMNS
from phase2_settings import Phase2Settings
from phase2_split import make_grouped_split, select_partition, validate_split
from record_identity import IDENTITY_COLUMNS
from rule_engine import validate_rules_on_full_dataframe
from tree_rule_export import extract_tree_rules

TECHNICAL_COLUMNS = set(IDENTITY_COLUMNS) | set(SCORE_COLUMNS) | {
    "geometry", "wkb_geometry", "geometry_parsed", "geometry_parsed_geom",
    "source_folder", "xml_path", "tif_path", "xml_name", "tif_name",
    "kachelname", "kachelname_from_filename", "__index_level_0__",
}


def _eligible_column(name):
    return name not in TECHNICAL_COLUMNS and not name.startswith(("latent_", "legacy_", "stratum_"))


def get_phase2_target_candidates(df: pd.DataFrame) -> list[str]:
    """Ziele ausschließlich anhand des übergebenen Trainingsanteils wählen."""
    candidates = []
    for col in df.columns:
        if not _eligible_column(col) or col in DOP_DATE_COLUMNS:
            continue
        values = df[col].dropna()
        try:
            if len(values) >= 50 and 2 <= values.nunique() <= 20 and type_of_target(values) in {"binary", "multiclass"}:
                candidates.append(col)
        except (TypeError, ValueError):
            continue
    return candidates


def get_phase2_feature_candidates(df, target_col, excluded_columns=None, forced_feature_columns=None):
    excluded = set(excluded_columns or []) | {target_col}
    candidates = df.columns if forced_feature_columns is None else forced_feature_columns
    return list(dict.fromkeys(c for c in candidates if c in df and c not in excluded and _eligible_column(c)))


def get_phase2_cluster_feature_candidates(df, excluded_columns=None):
    return get_phase2_feature_candidates(df, "cluster_label", excluded_columns)


def _varying_features(df, features):
    usable = []
    for col in features:
        try:
            if df[col].nunique(dropna=True) > 1:
                usable.append(col)
        except TypeError:
            continue
    return usable


def prepare_features(df, feature_cols, numeric_features, categorical_features):
    """Typen aus dem Training verwenden; keine Statistiken am Prüfsatz lernen."""
    X = df[feature_cols].copy()
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        if np.isinf(X[col].to_numpy()).any():
            raise ValueError(f"Nicht endliches numerisches Feature: {col}")
    for col in categorical_features:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(object)
    return X


def prepare_phase2_dataset(df, target_col, feature_cols):
    model_df = df.loc[df[target_col].notna()]
    numeric = model_df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    X = prepare_features(model_df, feature_cols, numeric, categorical)
    return X, model_df[target_col].copy(), numeric, categorical


def build_decision_tree_pipeline(numeric_features, categorical_features, random_state=42,
                                 max_depth=4, min_samples_leaf=5):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ], remainder="drop")
    classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def extract_rules_from_decision_tree_pipeline(model, target_col, source_backend, min_leaf_samples=5):
    return extract_tree_rules(model, target_col, source_backend, min_leaf_samples)


def evaluate_predictions(y_true, y_pred):
    if len(y_true) == 0:
        return {"n_evaluated": 0, "accuracy": None, "precision_macro": None, "recall_macro": None, "f1_macro": None}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"n_evaluated": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)}


def _key_digest(values):
    return hashlib.sha256(json.dumps(sorted(str(v) for v in values), ensure_ascii=False).encode("utf-8")).hexdigest()


def choose_tree_parameters(train, target_col, features, groups, settings):
    """Jeder Fold passt Imputer und Encoder neu an, ausschließlich auf Fold-Training."""
    labelled = train.loc[train[target_col].notna()].reset_index(drop=True)
    labelled_groups = labelled["record_key"].map(groups)
    if labelled_groups.nunique() < settings.cv_folds:
        raise ValueError("Zu wenige Trainingsgruppen mit vorhandenem Target für Gruppen-CV.")
    folds = list(GroupKFold(n_splits=settings.cv_folds).split(labelled, groups=labelled_groups))
    fold_audit = []
    for number, (fit_indices, check_indices) in enumerate(folds, 1):
        fit, check = labelled.iloc[fit_indices], labelled.iloc[check_indices]
        fit_groups, check_groups = set(labelled_groups.iloc[fit_indices]), set(labelled_groups.iloc[check_indices])
        fold_audit.append({"fold": number, "fit_rows": len(fit), "check_rows": len(check),
                           "fit_groups": len(fit_groups), "check_groups": len(check_groups),
                           "group_overlap_count": len(fit_groups & check_groups),
                           "fit_record_keys_sha256": _key_digest(fit["record_key"]),
                           "check_record_keys_sha256": _key_digest(check["record_key"])})
    candidates = []
    for max_depth, min_samples_leaf in product(settings.max_depths, settings.min_samples_leaves):
        metrics = []
        for fit_indices, check_indices in folds:
            fit, check = labelled.iloc[fit_indices], labelled.iloc[check_indices]
            X, y, numeric, categorical = prepare_phase2_dataset(fit, target_col, features)
            model = build_decision_tree_pipeline(numeric, categorical, settings.random_state, max_depth, min_samples_leaf)
            model.fit(X, y)
            prediction = model.predict(prepare_features(check, features, numeric, categorical))
            metrics.append(evaluate_predictions(check[target_col], prediction))
        candidates.append({"parameters": {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf},
                           "mean_f1_macro": float(np.mean([m["f1_macro"] for m in metrics])),
                           "mean_accuracy": float(np.mean([m["accuracy"] for m in metrics])), "fold_metrics": metrics})
    # Feste Auswahl, keine erneute Suche mit den Validierungsergebnissen.
    best = max(candidates, key=lambda c: (c["mean_f1_macro"], c["mean_accuracy"],
               -c["parameters"]["max_depth"], c["parameters"]["min_samples_leaf"]))
    return dict(best["parameters"]), {"scope": "training_partition_only", "selection_metric": "mean_f1_macro",
                                    "folds": fold_audit, "candidates": candidates}


def train_dependency(train, validation, target_col, features, groups, settings,
                     accuracy_threshold, f1_threshold, precision_threshold):
    report = {"target_col": target_col, "source": "ml_phase2", "evaluation_scope": "validation_partition",
              "test_evaluated": False, "success": False, "candidate_rules": [],
              "train_rows": len(train), "validation_rows": len(validation),
              "train_record_keys_sha256": _key_digest(train["record_key"]),
              "validation_record_keys_sha256": _key_digest(validation["record_key"]),
              "thresholds": {"accuracy": accuracy_threshold, "f1_macro": f1_threshold, "precision_macro": precision_threshold}}
    features = _varying_features(train, features)
    report["feature_cols_used"] = features
    present_train = train.loc[train[target_col].notna()]
    present_val = validation.loc[validation[target_col].notna()]
    report.update({"train_labelled_rows": len(present_train), "validation_labelled_rows": len(present_val),
                   "train_missing_target_rows": len(train) - len(present_train),
                   "validation_missing_target_rows": len(validation) - len(present_val)})
    if not features or len(present_train) < 10 or present_train[target_col].nunique() < 2:
        report["reason"] = "insufficient_training_data_or_features"
        return [], report
    if type_of_target(present_train[target_col]) not in {"binary", "multiclass"}:
        report["reason"] = "target_is_not_supported_classification_label"
        return [], report
    if train.loc[train[target_col].notna(), "record_key"].map(groups).nunique() < settings.cv_folds:
        report["reason"] = "too_few_labelled_training_groups"
        return [], report
    parameters, cv = choose_tree_parameters(train, target_col, features, groups, settings)
    report.update({"selected_parameters": parameters, "cross_validation": cv})
    X, y, numeric, categorical = prepare_phase2_dataset(train, target_col, features)
    model = build_decision_tree_pipeline(numeric, categorical, settings.random_state, **parameters)
    model.fit(X, y)
    predictions = model.predict(prepare_features(present_val, features, numeric, categorical)) if len(present_val) else []
    metrics = evaluate_predictions(present_val[target_col], predictions)
    # Mehrheitsklassen-Baseline wird ebenfalls ausschließlich aus dem Training bestimmt.
    majority = y.value_counts().index[0]
    baseline = evaluate_predictions(present_val[target_col], np.repeat(majority, len(present_val)))
    report.update({"validation_metrics": metrics, "validation_majority_baseline": baseline,
                   "model_leaf_count": int(model.named_steps["classifier"].get_n_leaves()),
                   "training_class_count": int(y.nunique()),
                   "validation_unseen_target_count": int((~present_val[target_col].isin(y.unique())).sum())})
    gate = bool(metrics["n_evaluated"] and metrics["accuracy"] >= accuracy_threshold
                and metrics["precision_macro"] >= precision_threshold and metrics["f1_macro"] >= f1_threshold)
    report["model_gate_passed"] = gate
    candidates = extract_rules_from_decision_tree_pipeline(model, target_col, "group_cv_decision_tree", parameters["min_samples_leaf"])
    results, _ = validate_rules_on_full_dataframe(validation, candidates)
    accepted = []
    for rule, evaluation in zip(candidates, results):
        eligible = bool(gate and evaluation["evaluation_status"] == "evaluated"
                        and evaluation["coverage_count"] >= settings.min_validation_rule_cases
                        and evaluation["pass_rate"] is not None
                        and evaluation["pass_rate"] >= settings.min_validation_rule_pass_rate)
        reasons = []
        if not gate:
            reasons.append("model_validation_thresholds_not_met")
        if evaluation["coverage_count"] < settings.min_validation_rule_cases:
            reasons.append("too_few_applicable_validation_cases")
        if evaluation["pass_rate"] is None or evaluation["pass_rate"] < settings.min_validation_rule_pass_rate:
            reasons.append("rule_validation_pass_rate_not_met")
        rule["reason"] = "Aus Trainingsbaum abgeleitet und auf separaten Kacheln geprüft; fachliche Freigabe erforderlich."
        rule["evidence_type"] = "group_heldout_validation_candidate"
        rule["evidence_detail"] = "Parameterwahl durch Gruppen-CV im Training; Regelprüfung nur auf Validierung; keine Testauswertung."
        rule["metadata"].update({"evaluation_scope": "validation_partition", "test_evaluated": False,
                                  "selected_parameters": parameters, "model_gate_passed": gate,
                                  "eligible_for_review": eligible, "rejection_reasons": reasons,
                                  "validation": evaluation,
                                  "minimum_validation_cases": settings.min_validation_rule_cases,
                                  "minimum_validation_pass_rate": settings.min_validation_rule_pass_rate})
        if eligible:
            accepted.append(rule)
    report.update({"candidate_rules": candidates, "candidate_rule_count": len(candidates),
                   "rule_count": len(accepted), "success": bool(accepted),
                   "reason": "review_candidates_available" if accepted else "no_candidate_meets_model_and_rule_gates"})
    return accepted, report


def build_ssl_cluster_rules(df, sample_runs=None, excluded_columns=None, min_cluster_size=10, *, settings=None):
    settings = settings or Phase2Settings(cluster_min_size=min_cluster_size)
    report = {"target_col": "cluster_label", "source": "ssl_cluster", "evaluation_scope": "descriptive_in_sample",
              "independent_validation": False, "success": False, "rule_count": 0}
    if "cluster_label" not in df:
        report["reason"] = "cluster_label_missing"
        return [], [report]
    non_noise = df.loc[df["cluster_label"].notna() & df["cluster_label"].ne(-1)].copy()
    counts = non_noise["cluster_label"].value_counts()
    valid = counts[counts >= settings.cluster_min_size].index.tolist()
    work = non_noise.loc[non_noise["cluster_label"].isin(valid)].sort_values("record_key")
    report.update({"cluster_counts": {str(k): int(v) for k, v in counts.items()},
                   "included_clusters": sorted(int(c) for c in valid), "population_rows": len(df),
                   "model_rows": len(work), "excluded_noise_or_missing_rows": len(df) - len(non_noise),
                   "excluded_small_cluster_rows": len(non_noise) - len(work),
                   "population_coverage": len(work) / len(df) if len(df) else None})
    features = _varying_features(work, get_phase2_cluster_feature_candidates(work, excluded_columns))
    report["feature_cols_used"] = features
    if len(valid) < 2 or not features:
        report["reason"] = "insufficient_clusters_or_features"
        return [], [report]
    X, y, numeric, categorical = prepare_phase2_dataset(work, "cluster_label", features)
    model = build_decision_tree_pipeline(numeric, categorical, settings.random_state,
                                        settings.cluster_max_depth, settings.cluster_min_samples_leaf)
    model.fit(X, y)
    prediction = model.predict(X)
    report["in_sample_fidelity"] = evaluate_predictions(y, prediction)
    report["predicted_clusters"] = sorted(int(c) for c in np.unique(prediction))
    report["clusters_without_predicted_leaf"] = sorted(set(report["included_clusters"]) - set(report["predicted_clusters"]))
    rules = extract_rules_from_decision_tree_pipeline(model, "cluster_label", "cluster_description", settings.cluster_min_samples_leaf)
    evaluations, _ = validate_rules_on_full_dataframe(work, rules)
    for rule, evaluation in zip(rules, evaluations):
        rule.update({"rule_id": rule["rule_id"].replace("ml__", "ssl_cluster__", 1),
                     "rule_type": "ssl_cluster_rule", "source": "ssl_cluster",
                     "rule_name": "Descriptive cluster characterization", "evidence_type": "descriptive_in_sample",
                     "reason": "Beschreibung vorhandener Cluster; keine fachliche Validierungsregel.",
                     "evidence_detail": "Auf denselben Datensätzen angepasst und beschrieben; keine unabhängige Güteprüfung."})
        evaluation["rule_id"] = rule["rule_id"]
        rule["metadata"].update({"evaluation_scope": "descriptive_in_sample", "in_sample": evaluation,
                                  "included_clusters": report["included_clusters"], "eligible_for_review_as_quality_rule": False})
    report.update({"success": True, "rule_count": len(rules), "reason": "descriptive_model_only"})
    return rules, [report]


def build_phase2_ml_rules(df, sample_runs=None, target_candidates=None, accuracy_threshold=.95,
                          f1_threshold=.90, precision_threshold=.90, excluded_columns=None,
                          backend="decision_tree", dependency_suggestions=None, *, settings=None, split=None):
    if backend != "decision_tree":
        raise ValueError("Getrennte Bewertung ist für decision_tree implementiert; kein stiller Backend-Wechsel.")
    if dependency_suggestions:
        raise ValueError("LLM-Vorschläge aus Gesamtprofilen sind für diese getrennte Auswertung nicht zugelassen.")
    if sample_runs is not None:
        warnings.warn("Phase-1-Stichproben werden nicht für Phase-2-Training verwendet.", UserWarning)
    settings = settings or Phase2Settings()
    split = make_grouped_split(df, settings) if split is None else validate_split(df, split, settings)
    train, validation = select_partition(df, split, "train"), select_partition(df, split, "validation")
    groups = split.set_index("record_key")["group_id"]
    targets = get_phase2_target_candidates(train) if target_candidates is None else list(dict.fromkeys(target_candidates))
    rules, reports = [], []
    for target in targets:
        if target not in train or not _eligible_column(target):
            raise ValueError(f"Unzulässiges oder fehlendes Target: {target}")
        features = get_phase2_feature_candidates(train, target, excluded_columns)
        accepted, report = train_dependency(train, validation, target, features, groups, settings,
                                            accuracy_threshold, f1_threshold, precision_threshold)
        rules.extend(accepted)
        reports.append(report)
        metric = report.get("validation_metrics", {}).get("accuracy")
        print(f"  {target}: Validierungs-Accuracy={metric}, Kandidaten zur Prüfung={len(accepted)}")
    cluster_rules, cluster_report = build_ssl_cluster_rules(df, excluded_columns=excluded_columns, settings=settings)
    return rules + cluster_rules, reports + cluster_report
