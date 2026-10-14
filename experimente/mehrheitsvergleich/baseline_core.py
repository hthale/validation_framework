"""Mehrheitsvergleich und kontrollierte Zieländerungen, ohne Modelltraining.

Die bisherigen Regeln behalten ihre Auswahl und ihren Scope. Ein Hinweis ist
eine Abweichung von einer statistischen Erwartung, kein bestätigter Sachfehler.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

from comparison_core import DATE, DATE_MISSING, METHODS, summarize_cases
from phase2_ml import evaluate_predictions

BASELINE = "training_majority"
ALL_METHODS = (*METHODS, BASELINE)
SCENARIOS = ("majority_to_nonmajority", "nonmajority_to_majority", "present_to_missing")
KINDS = ("global_attribute_dependency", "global_constant_expectation",
         "cluster_attribute_dependency", "cluster_constant_expectation")


def token(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(*parts):
    return hashlib.sha256(token(list(parts)).encode("utf-8")).hexdigest()


def majority_model(train, target):
    """ALKIS-Adapter erzeugt Textklassen. Fehlwerte sind keine Klasse."""
    values = train[target].dropna().tolist()
    if not values or any(not isinstance(v, str) for v in values):
        raise ValueError(f"Nichtleere Textklassen aus dem Attributadapter benötigt: {target}")
    counts = Counter(values)
    classes = sorted(counts, key=lambda v: (-counts[v], token(v)))
    return {"target": target, "prediction": classes[0],
            "class_counts": [{"value": v, "count": counts[v]} for v in classes],
            "labelled_training_rows": len(values), "missing_training_rows": len(train)-len(values),
            "training_keys_sha256": stable_hash(sorted(train.record_key.tolist())),
            "fit_scope": "training_only", "tie_break": "lexical_canonical_json",
            "selection_gate": "none_unconditional_comparator_not_approved_rule"}


def baseline_cases(frame, target, groups, seed, model):
    if not frame.record_key.is_unique or model["target"] != target:
        raise ValueError("Mehrheitsmodell, Ziel oder Kennungen passen nicht zusammen.")
    present = frame[target].notna().to_numpy()
    agree = frame[target].eq(model["prediction"]).fillna(False).to_numpy(dtype=bool)
    mapped = frame.record_key.map(groups)
    if mapped.isna().any():
        raise ValueError("Fehlende Referenzgruppe.")
    return pd.DataFrame({"seed": seed, "method": BASELINE, "target": target,
        "record_key": frame.record_key.to_numpy(), "group_id": mapped.to_numpy(),
        "cluster": -2, "prediction": model["prediction"],
        "observed_target": frame[target].to_numpy(), "covered": True, "conflict": False,
        "agree": agree, "disagree_observed": present & ~agree,
        "missing_target_covered": ~present, "target_present": present, "rule_ids": ""})


def class_metrics(cases):
    present = cases.loc[cases.covered & cases.target_present]
    return evaluate_predictions(present.observed_target, present.prediction)


def classify_rules(rules):
    result = []
    for rule in rules:
        scope, inner = rule["scope"], rule["inner_rule"]
        if scope["method"] not in METHODS or inner["metadata"]["target_col"] != scope["target"]:
            raise ValueError("Unbekannter oder widersprüchlicher Regel-Scope.")
        conditions = inner["metadata"]["conditions"]
        if not isinstance(conditions, list):
            raise ValueError("Explizite Bedingungsliste benötigt.")
        prefix = "global" if scope["method"] == "global" else "cluster"
        kind = prefix + ("_attribute_dependency" if conditions else "_constant_expectation")
        result.append({"rule_id": rule["rule_id"], **scope, "rule_kind": kind,
            "selected_on_validation": bool(rule["selected_on_validation"]),
            "inner_condition_count": len(conditions), "predicted_value": inner["metadata"]["predicted_value"],
            "scope_is_required": True, "expression": rule["expression"],
            "note": "Clusterzuordnung bleibt eine Bedingung, auch ohne weitere Baumbedingung."})
    if len({r["rule_id"] for r in result}) != len(result):
        raise ValueError("Doppelte Regelkennung.")
    return result


def rule_counts(classified):
    rows = []
    for seed in sorted({r["seed"] for r in classified}):
        for method in METHODS:
            for kind in KINDS:
                subset = [r for r in classified if (r["seed"], r["method"], r["rule_kind"]) == (seed, method, kind)]
                rows.append({"seed": seed, "method": method, "rule_kind": kind,
                    "candidates": len(subset), "selected": sum(r["selected_on_validation"] for r in subset)})
    return rows


def original_comparisons(cases, model):
    """Gleiche Fälle pro Methode; keine Auswahl anhand der Testgüte."""
    if len(cases) == 0 or cases.seed.nunique() != 1 or cases.target.nunique() != 1:
        raise ValueError("Genau ein nichtleerer Seed/Ziel-Vergleich benötigt.")
    reference = cases.loc[cases.method.eq(BASELINE)].set_index("record_key")
    if not reference.index.is_unique:
        raise ValueError("Doppelte Referenzkennung.")
    rows = []
    for method in ALL_METHODS:
        cur = cases.loc[cases.method.eq(method)].set_index("record_key")
        if not cur.index.is_unique or set(cur.index) != set(reference.index):
            raise ValueError("Verglichen werden unterschiedliche Fälle.")
        ref = reference.loc[cur.index]
        observed = cur.observed_target
        strata = {"all": np.ones(len(cur), dtype=bool),
                  "original_majority": observed.eq(model["prediction"]).fillna(False).to_numpy(dtype=bool),
                  "original_nonmajority": (observed.notna() & observed.ne(model["prediction"]).fillna(False)).to_numpy(dtype=bool),
                  "original_missing": observed.isna().to_numpy()}
        for stratum, mask in strata.items():
            sub, baseline = cur.loc[mask], ref.loc[mask]
            on_same = baseline.loc[sub.covered]
            method_covered = sub.loc[sub.covered]
            different = method_covered.prediction.ne(on_same.prediction).fillna(False)
            summary = summarize_cases(sub)
            rows.append({"seed": int(cases.seed.iloc[0]), "target": model["target"], "method": method,
                "stratum": stratum, **summary, "observed_class_metrics": class_metrics(sub),
                "baseline_agree_on_same_covered": int(on_same.agree.sum()),
                "baseline_mismatch_on_same_covered": int(on_same.disagree_observed.sum()),
                "baseline_missing_on_same_covered": int(on_same.missing_target_covered.sum()),
                "different_predictions_from_majority": int(different.sum()),
                "net_agreement_gain_on_same_covered": int(method_covered.agree.sum()-on_same.agree.sum()),
                "original_values_are_verified_correct": False})
    return rows


def select_groups(frame, groups, seed, target, scenario, limit):
    """Ein Fall pro unveränderter Referenzgruppe; Reihenfolge-/Score-unabhängig."""
    if limit < 1 or not frame.record_key.is_unique:
        raise ValueError("Positives Gruppenlimit und eindeutige Kennungen benötigt.")
    choices = {}
    for key in frame.record_key:
        group = groups.loc[key]
        order = stable_hash("paired-change-v1", seed, target, scenario, str(group), key)
        if group not in choices or order < choices[group][0]:
            choices[group] = (order, key)
    # Gruppengröße darf die Chance auf Auswahl nicht über das Minimum mehrerer
    # Zeilen-Hashes erhöhen: eigener, vom Vertreter unabhängiger Gruppen-Hash.
    chosen_groups = sorted(choices, key=lambda g: stable_hash("group-order-v1", seed, target, scenario, str(g)))[:limit]
    picked = [choices[g][1] for g in chosen_groups]
    return frame.set_index("record_key", drop=False).loc[picked].reset_index(drop=True)


def make_variants(train, test, groups, seed, target, model, limit):
    """Zieländerung in beiden Richtungen sowie Fehlwert; Herkunft immer erhalten."""
    mode = model["prediction"]
    alternatives = sorted((c["value"] for c in model["class_counts"] if c["value"] != mode), key=token)
    variants, availability = [], []
    for scenario in SCENARIOS:
        if scenario == "majority_to_nonmajority":
            pool = test.loc[test[target].eq(mode).fillna(False)] if alternatives else test.iloc[:0]
        elif scenario == "nonmajority_to_majority":
            pool = test.loc[test[target].notna() & test[target].ne(mode).fillna(False)]
        else:
            pool = test.loc[test[target].notna()] if target != DATE_MISSING else test.iloc[:0]
        selected = select_groups(pool, groups, seed, target, scenario, limit)
        row = {"seed": seed, "target": target, "scenario": scenario, "eligible_rows": len(pool),
               "eligible_groups": int(pool.record_key.map(groups).nunique()), "selected_pairs": len(selected),
               "selection": "hash_order_one_row_per_reference_group", "group_limit": limit}
        if not len(selected):
            row["skip_reason"] = ("derived_presence_indicator_is_not_independently_nullable"
                if target == DATE_MISSING and scenario == "present_to_missing" else "no_eligible_original_cases_or_training_alternative")
            availability.append(row)
            continue
        changed = selected.copy(deep=True)
        changed[target] = changed[target].astype(object)
        changes = [target]
        after = []
        for key in selected.record_key:
            if scenario == "majority_to_nonmajority":
                value = alternatives[int(stable_hash("alternative-v1", seed, target, key), 16) % len(alternatives)]
            else:
                value = mode if scenario == "nonmajority_to_majority" else None
            after.append(value)
        changed[target] = pd.Series(after, dtype=object)
        if target == DATE_MISSING:
            if DATE not in changed or set(after) - {"fehlt", "vorhanden"}:
                raise ValueError("Unbekannte Datumsfamilie; keine inkonsistente Mutation erzeugen.")
            median = train.loc[train[DATE].notna(), DATE].median()
            if not np.isfinite(median):
                raise ValueError("Kein trainierter Referenzwert für vorhandenes Datum.")
            changed[DATE] = [np.nan if flag == "fehlt" else float(median) for flag in after]
            changes.append(DATE)
        meta = []
        for i, key in enumerate(selected.record_key):
            meta.append({"case_id": "change12:"+stable_hash(seed, target, scenario, key),
                "seed": seed, "target": target, "scenario": scenario, "record_key": key,
                "group_id": groups.loc[key], "original_value": selected[target].iloc[i],
                "changed_value": changed[target].iloc[i], "changed_columns": "|".join(changes),
                "original_majority": bool(selected[target].iloc[i] == mode),
                "original_verified_correct": False, "changed_value_is_confirmed_error": False})
        variants.append((scenario, selected, changed, pd.DataFrame(meta), changes))
        availability.append(row)
    return variants, availability


def assert_frozen_inputs(original, changed, target, features, rules, changes):
    """Identische Prädiktoren => gespeicherte eingefrorene Routen weiter gültig."""
    if not original.record_key.is_unique or original.record_key.tolist() != changed.record_key.tolist():
        raise ValueError("Kennungen oder Reihenfolge der Kopien wurden verändert.")
    if target in features or set(changes) & set(features):
        raise ValueError("Zielmutation berührt einen Router-Prädiktor.")
    if target == DATE_MISSING and ({DATE, DATE_MISSING} & set(features)):
        raise ValueError("Datumsableitung im Router: Ziel-Leakage.")
    if set(original.columns) != set(changed.columns) or not set(changes) <= set(original.columns):
        raise ValueError("Spalten der Mutation passen nicht zum Original.")
    from verify_rule_comparison import assert_same_cases
    unchanged = [c for c in original.columns if c not in changes]
    assert_same_cases(original[unchanged], changed[unchanged])
    for rule in rules:
        if rule["scope"]["target"] != target or any(
                c["feature"] not in features for c in rule["inner_rule"]["metadata"]["conditions"]):
            raise ValueError("Regelbedingungen passen nicht zum eingefrorenen Merkmalsvertrag.")


def pair_results(before, after, meta):
    from verify_rule_comparison import assert_same_cases
    if before.record_key.tolist() != after.record_key.tolist() or before.record_key.tolist() != meta.record_key.tolist():
        raise ValueError("Änderungsfall und unveränderte Kontrolle nicht gepaart.")
    # Eine Zielmutation darf weder Vorhersage noch Scope/Enthaltung verändern.
    assert_same_cases(before[["record_key", "prediction", "covered", "conflict", "rule_ids", "cluster"]],
                      after[["record_key", "prediction", "covered", "conflict", "rule_ids", "cluster"]])
    b = (before.disagree_observed | before.missing_target_covered).to_numpy(dtype=bool)
    a = (after.disagree_observed | after.missing_target_covered).to_numpy(dtype=bool)
    result = meta.copy()
    result["method"] = before.method.iloc[0]
    for name in ("prediction", "covered", "conflict", "rule_ids", "cluster"):
        result[name] = before[name].to_numpy()
    result["original_disagreement"] = b
    result["changed_disagreement"] = a
    result["new_disagreement"] = ~b & a
    result["disagreement_removed"] = b & ~a
    result["both_disagree"] = b & a
    result["both_without_disagreement"] = ~b & ~a
    result["original_agree"] = before.agree.to_numpy()
    result["changed_agree"] = after.agree.to_numpy()
    result["changed_missing"] = after.missing_target_covered.to_numpy()
    return result


def mutation_metrics(pairs):
    if pairs.empty:
        return []
    rows = []
    for (seed, target, scenario, method), sub in pairs.groupby(["seed", "target", "scenario", "method"], sort=True):
        if not sub.record_key.is_unique or not sub.group_id.is_unique:
            raise ValueError("Mehrere Fälle derselben Referenzgruppe in einem Szenario.")
        prior_agree = int(sub.original_agree.sum())
        rows.append({"seed": int(seed), "target": target, "scenario": scenario, "method": method,
            "paired_cases": len(sub), "reference_groups": sub.group_id.nunique(),
            "covered_pairs": int(sub.covered.sum()), "abstained_pairs": int((~sub.covered).sum()),
            **{c: int(sub[c].sum()) for c in ("original_disagreement", "changed_disagreement", "new_disagreement",
                "disagreement_removed", "both_disagree", "both_without_disagreement", "original_agree", "changed_agree", "changed_missing")},
            "new_disagreement_per_all_pairs": float(sub.new_disagreement.mean()),
            "new_disagreement_per_previously_agreeing": int(sub.new_disagreement.sum())/prior_agree if prior_agree else None,
            "interpretation": "controlled_target_change_response_not_real_error_recall"})
    return rows
