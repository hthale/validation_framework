"""Vorab festgelegte Kennzahlen, ohne nachträgliche Auswahl der besten Methode."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from synthetic_data import digest


def ratio(a, b):
    return float(a/b) if b else None


def align_labels(cases, labels):
    if not cases.record_key.is_unique or not labels.record_key.is_unique or set(cases.record_key) != set(labels.record_key):
        raise ValueError("Kennungen der Vorhersagen und Referenzlabels stimmen nicht überein.")
    return labels.set_index("record_key").loc[cases.record_key].reset_index()


def summarize_predictions(cases, labels):
    truth = align_labels(cases, labels)
    error = truth.is_error.to_numpy(dtype=bool)
    covered = cases.covered.to_numpy(dtype=bool)
    flagged = (cases.disagree_observed | cases.missing_target_covered).to_numpy(dtype=bool)
    rare = truth.rare_normal.to_numpy(dtype=bool)
    clean = ~error
    tp, fp = int((flagged & error).sum()), int((flagged & clean).sum())
    group_frame = pd.DataFrame({"group": truth.group_id, "error": error, "flag": flagged, "covered": covered})
    clean_group_fpr = group_frame.loc[clean].groupby("group").flag.mean().mean()
    return {"rows": len(cases), "errors": int(error.sum()), "clean": int(clean.sum()),
        "covered": int(covered.sum()), "coverage": float(covered.mean()), "abstained": int((~covered).sum()),
        "true_positives": tp, "false_positives": fp, "false_negatives_including_abstentions": int(error.sum())-tp,
        "recall_all_errors": ratio(tp, int(error.sum())), "precision_flagged": ratio(tp, tp+fp),
        "false_positive_rate_clean": ratio(fp, int(clean.sum())),
        "group_mean_false_positive_rate_clean": float(clean_group_fpr) if len(group_frame.loc[clean]) else None,
        "covered_errors": int((covered & error).sum()), "covered_clean": int((covered & clean).sum()),
        "rare_clean": int((rare & clean).sum()), "rare_clean_covered": int((rare & clean & covered).sum()),
        "rare_clean_false_positives": int((rare & clean & flagged).sum()),
        "rare_clean_false_positive_rate": ratio(int((rare & clean & flagged).sum()), int((rare & clean).sum())),
        "rare_errors": int((rare & error).sum()), "rare_errors_detected": int((rare & error & flagged).sum()),
        "scope": "synthetic_oracle_only"}


def fit_lof(matrix, neighbors=20):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 3 or not np.isfinite(matrix).all():
        raise ValueError("Ungültige LOF-Trainingsmatrix.")
    model = LocalOutlierFactor(n_neighbors=min(neighbors, len(matrix)-1), novelty=True,
                              metric="euclidean", n_jobs=1)
    return model.fit(matrix)


def score_lof(model, matrix):
    result = -model.score_samples(np.asarray(matrix, dtype=np.float64))
    if not np.isfinite(result).all():
        raise ValueError("Nicht endliche LOF-Scores.")
    return result


def validation_threshold(scores, fraction=.05):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Ungültige Validierungsscores.")
    # Striktes > bei Anwendung: Gleichstände werden nicht sämtlich zu Fehlern.
    return float(np.quantile(scores, 1-fraction, method="higher"))


def rank_budget(keys, scores, budget=.10):
    keys, scores = list(keys), np.asarray(scores, dtype=float)
    if not len(keys) or len(keys) != len(scores) or len(set(keys)) != len(keys) or not np.isfinite(scores).all() or not 0 < budget <= 1:
        raise ValueError("Ungültige Rangfolge oder Prüfbudget.")
    n = int(math.ceil(len(keys)*budget))
    order = sorted(range(len(keys)), key=lambda i: (-scores[i], digest("tie13", keys[i])))
    ranks, chosen = np.empty(len(keys), dtype=int), np.zeros(len(keys), dtype=bool)
    ranks[order] = np.arange(1, len(keys)+1)
    chosen[order[:n]] = True
    return ranks, chosen


def lof_cases(keys, scores, threshold, budget=.10):
    ranks, chosen = rank_budget(keys, scores, budget)
    return pd.DataFrame({"record_key": list(keys), "score": scores, "rank": ranks,
                         "selected_budget": chosen, "above_validation_threshold": np.asarray(scores) > threshold})


def summarize_lof(cases, labels, threshold):
    truth = align_labels(cases, labels)
    error = truth.is_error.to_numpy(dtype=bool)
    rare = truth.rare_normal.to_numpy(dtype=bool)
    result = {"rows": len(cases), "errors": int(error.sum()), "error_prevalence": float(error.mean()),
        "average_precision": float(average_precision_score(error, cases.score)) if error.any() else None,
        "roc_auc": float(roc_auc_score(error, cases.score)) if error.any() and (~error).any() else None,
        "validation_threshold": threshold, "score_distinct_values": int(cases.score.nunique()),
        "scope": "synthetic_oracle_only"}
    for name, field in (("budget", "selected_budget"), ("threshold", "above_validation_threshold")):
        selected = cases[field].to_numpy(dtype=bool)
        tp, fp = int((selected & error).sum()), int((selected & ~error).sum())
        groups = pd.DataFrame({"group": truth.group_id, "selected": selected})
        result[name] = {"selected": int(selected.sum()), "true_positives": tp, "false_positives": fp,
            "precision": ratio(tp, int(selected.sum())), "recall": ratio(tp, int(error.sum())),
            "false_positive_rate_clean": ratio(fp, int((~error).sum())),
            "group_mean_false_positive_rate_clean": float(groups.loc[~error].groupby("group").selected.mean().mean()),
            "rare_clean": int((rare & ~error).sum()), "rare_clean_selected": int((rare & ~error & selected).sum()),
            "rare_errors": int((rare & error).sum()), "rare_errors_selected": int((rare & error & selected).sum())}
    return result


def compare_methods(rule_metrics, lof_metrics):
    rows = []
    for scenario in sorted({r["scenario"] for r in rule_metrics}):
        for seed in sorted({r["seed"] for r in rule_metrics if r["scenario"] == scenario}):
            r = {m["method"]: m for m in rule_metrics if m["scenario"] == scenario and m["seed"] == seed and m["variant"] == "code_swap"}
            rows.append({"scenario": scenario, "seed": seed, "task": "rules_code_swap",
                "latent_minus_attribute_true_positives": r["latent_clusters"]["true_positives"]-r["attribute_clusters"]["true_positives"],
                "latent_minus_attribute_false_positives": r["latent_clusters"]["false_positives"]-r["attribute_clusters"]["false_positives"],
                "latent_minus_global_true_positives": r["latent_clusters"]["true_positives"]-r["global"]["true_positives"]})
            for variant in ("code_swap", "signal_swap"):
                m = {x["representation"]: x for x in lof_metrics if x["scenario"] == scenario and x["seed"] == seed and x["variant"] == variant}
                rows.append({"scenario": scenario, "seed": seed, "task": "lof_"+variant,
                    "latent_minus_attribute_budget_true_positives": m["latent"]["budget"]["true_positives"]-m["attributes"]["budget"]["true_positives"],
                    "latent_minus_pca_budget_true_positives": m["latent"]["budget"]["true_positives"]-m["pca"]["budget"]["true_positives"],
                    "latent_minus_attribute_average_precision": m["latent"]["average_precision"]-m["attributes"]["average_precision"]})
    return rows
