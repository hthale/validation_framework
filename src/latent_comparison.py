"""Vorab festgelegter LOF-Vergleich bei gleicher verfügbaren Eingangsinformation.

Keine fachlichen Fehlerlabels: gemessen wird die Priorisierung künstlicher
Änderungen in einer getrennten Prüfteilmenge. Der Gewinn darf negativ sein.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import LocalOutlierFactor

from record_identity import validate_record_keys


PROTOCOL = {
    "name": "paired_novelty_lof_08", "version": 1,
    "split": "70 % der Gruppen Training, 30 % Prüfung; tatsächliche Zeilenzahlen können abweichen",
    "fit_scope": "training_only: Rollen, Imputation, Skalierung, Kategorien, Autoencoder und LOF",
    "tokenizer": "Unverändertes festes Zeichenvokabular des Frameworks, keine Anpassung am Prüfsatz",
    "baseline": "Standardisierte numerische/Datumswerte + One-Hot-Kategorien + normierte Zeichenhäufigkeiten je Textfeld",
    "baseline_information": "Dieselben ausgewählten Felder, dasselbe Vokabular und dieselbe Trunkierung wie der MeanCharEncoder; Reihenfolge ist für beide Stringrepräsentationen irrelevant",
    "metric": "Euclidean, keine zusätzliche Skalierung des latenten Raums, keine nachträgliche Optimierung der Modalitätsgewichte",
    "lof": "novelty=True, Fit nur Training, -score_samples nur für Prüfdaten; kein Wiederverwenden des früheren transduktiven LOF",
    "jitter": "Kein Zufallsrauschen; gleiche Datensätze behalten bei unverändertem Inhalt exakt denselben Score",
    "review_budget": "ceil(Prüfzeilen * Budgetanteil), Gleichstand nach record_key; keine automatische Fehlerschwelle",
    "perturbations": "Eine Änderung je ausgewählter Prüfzeile: +/-3 Trainingsstandardabweichungen, +/-180 Tage, anderer beobachteter Kategorie-/Textwert oder fehlender Wert; Typen möglichst ausgeglichen",
    "comparison": "Identische Gruppen, Vorverarbeitung, Prüffälle, Änderungen, Nachbarzahl und Budget pro Methodenpaar",
    "labels": "changed=True bezeichnet ausschließlich eine tatsächlich vorgenommene künstliche Änderung",
    "scope": "Explorative Empfindlichkeit gegenüber künstlichen Änderungen; keine bestätigte reale Fehlererkennungsleistung",
    "limitations": [
        "Unveränderte Originaldaten sind nicht automatisch fehlerfrei; Auswahl solcher Zeilen ist kein nachgewiesener Fehlalarm.",
        "Änderungen können fachlich zulässig sein. Fachliche Fehlerklassen müssen gesondert definiert werden.",
        "Nur im Training variierende und vom eingefrorenen Vertrag verwendete Felder werden verändert.",
        "Nicht ausgewählte Konstanten, Kennungen, ausgeschlossene Felder und Geometrien bleiben außerhalb dieses Versuchs.",
        "Die fehlende-Wert-Änderung kann durch Imputation unsichtbar werden; solche Fälle bleiben im Nenner und werden gezählt.",
        "Bekannte Gruppen bleiben getrennt; räumliche, zeitliche oder Lieferungsunabhängigkeit folgt daraus nicht automatisch.",
        "Mehrere Seeds verwenden überlappende Daten und sind keine unabhängigen Datensätze oder Signifikanzprüfung.",
        "Nur diese feste Baseline wird geprüft; ein Sieg beweist keine allgemeine Überlegenheit des Autoencoders.",
        "Dieser Versuch bewertet keine Regelerzeugung, keine Clusterstabilität und keine ALKIS-Geometrien.",
    ],
    "documentation": "https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor",
}


def grouped_partition(raw_with_groups, seed, test_fraction=.30):
    frame = raw_with_groups.sort_values("record_key").reset_index(drop=True)
    validate_record_keys(frame.record_key)
    if frame.__generic_group.nunique() < 6:
        raise ValueError("Für den Vergleich werden mindestens sechs Gruppen benötigt.")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                                random_state=seed).split(frame, groups=frame.__generic_group))
    if len(train_idx) < 24 or len(test_idx) < 12:
        raise ValueError("Aufteilung benötigt mindestens 24 Trainings- und 12 Prüfzeilen; Gruppen prüfen.")
    split = frame[["record_key", "__generic_group"]].rename(columns={"__generic_group": "group_id"})
    split["partition"] = "train"
    split.loc[test_idx, "partition"] = "test"
    if split.groupby("group_id").partition.nunique().max() != 1:
        raise ValueError("Eine Gruppe gelangt in mehrere Teilmengen.")
    return frame.drop(columns="__generic_group"), split


def transform_frozen(normalized, prepared, metadata):
    """Gleiche Transformation wie tabular_preparation, ohne erneutes fit()."""
    model = pd.DataFrame(index=normalized.index)
    derived = {d["feature"]: d for d in metadata["derived_features"]}
    for col in metadata["numeric_columns"]:
        if col not in derived:
            model[col] = normalized[col]
        elif derived[col]["transformation"] == "UTC days since Unix epoch":
            dates = pd.to_datetime(normalized[derived[col]["source"]], errors="coerce", utc=True)
            model[col] = dates.map(lambda v: v.timestamp()/86400 if pd.notna(v) else np.nan)
        elif derived[col]["transformation"] == "string_length":
            model[col] = normalized[derived[col]["source"]].fillna("").astype(str).str.len().astype(float)
        else:
            raise ValueError(f"Unbekannte Transformation: {derived[col]}")
    for col in metadata["categorical_columns"]:
        model[col] = normalized[col].fillna("").astype(str)
    # Entspricht bewusst der vorhandenen Behandlung gemeinsam verwendeter
    # Kategorie-/Stringfelder in tabular_preparation (fehlend -> Leerstring).
    for col in metadata["string_columns"]:
        model[col] = normalized[col].fillna("").astype(str)
    matrix = np.asarray(prepared["preprocessor"].transform(model), dtype=np.float32)
    if matrix.shape[1] != prepared["X_tab"].shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Eingefrorene Vorverarbeitung liefert ungültige Merkmale.")
    return model, matrix


def baseline_matrix(tabular, encoded_strings, vocab_size):
    """Untrainierte Baseline mit derselben dem MeanCharEncoder zugänglichen Information."""
    pieces = [np.asarray(tabular, dtype=np.float32)]
    for tokens in encoded_strings:
        hist = np.zeros((len(tokens), vocab_size-1), dtype=np.float32)
        for i, row in enumerate(tokens):
            count = np.bincount(row, minlength=vocab_size)[1:]  # PAD trägt nicht bei.
            hist[i] = count / max(int(count.sum()), 1)
        pieces.append(hist)
    result = np.concatenate(pieces, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Baseline enthält nicht endliche Werte.")
    return result


def source_columns(metadata):
    derived = {d["feature"]: d["source"] for d in metadata["derived_features"]}
    return sorted({derived.get(c, c) for c in metadata["numeric_columns"]}
                  | set(metadata["categorical_columns"]) | set(metadata["string_columns"]))


def inject_changes(test, train, contract, metadata, rate, seed):
    if not 0 < rate < 1:
        raise ValueError("Änderungsanteil muss zwischen 0 und 1 liegen.")
    rng = np.random.default_rng(seed)
    changed = test.copy(deep=True).reset_index(drop=True)
    used = source_columns(metadata)
    options = {}
    for col in used:
        role = contract["columns"][col]["role"]
        vals = train[col].dropna()
        if role == "numeric" and len(vals) > 1:
            scale = float(vals.astype(float).std(ddof=0))
            if np.isfinite(scale) and scale > 0:
                options.setdefault("numeric_shift", []).append((col, scale))
        elif role == "date":
            options.setdefault("date_shift", []).append((col, 180))
        elif role in {"categorical", "text"} and vals.nunique() >= 2:
            options.setdefault("category_swap" if role == "categorical" else "text_swap", []).append(
                (col, sorted(vals.astype(str).unique().tolist())))
        if len(vals):
            options.setdefault("missing_value", []).append((col, None))
    if not options:
        raise ValueError("Keine geeigneten verwendeten Merkmale für kontrollierte Änderungen.")
    operators = sorted(options)
    desired = min(len(test)-1, max(1, math.ceil(len(test)*rate)))
    records = []
    for idx in rng.permutation(len(test)):
        # Typwahl nicht vom Score abhängig; ausgeglichene feste Rotation.
        start = len(records) % len(operators)
        success = False
        for offset in range(len(operators)):
            operator = operators[(start+offset) % len(operators)]
            candidates = options[operator]
            for pos in rng.permutation(len(candidates)):
                col, parameter = candidates[pos]
                old = changed.at[idx, col]
                if pd.isna(old):
                    continue
                if operator == "numeric_shift":
                    new = float(old) + float(rng.choice([-1, 1])) * 3 * parameter
                    if not np.isfinite(new) or new == float(old):
                        continue
                elif operator == "date_shift":
                    try:
                        new = (pd.Timestamp(old) + pd.Timedelta(days=int(rng.choice([-1, 1]))*parameter)).isoformat()
                    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
                        continue
                elif operator in {"category_swap", "text_swap"}:
                    choices = [v for v in parameter if v != str(old)]
                    if not choices: continue
                    new = choices[int(rng.integers(len(choices)))]
                else:
                    new = np.nan if contract["columns"][col]["role"] == "numeric" else None
                changed.at[idx, col] = new
                records.append({"record_key": str(changed.at[idx, "record_key"]), "column": col,
                                "operator": operator, "before": old, "after": new,
                                "is_confirmed_domain_error": False})
                success = True
                break
            if success: break
        if len(records) == desired:
            break
    if len(records) != desired:
        raise ValueError(f"Nur {len(records)} von {desired} geplanten Änderungen möglich; kein still verkleinerter Versuch.")
    return changed, records


def novel_lof(train, before, after, n_neighbors=20):
    arrays = [np.asarray(x, dtype=np.float64) for x in (train, before, after)]
    if (any(x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all() for x in arrays)
            or len({x.shape[1] for x in arrays}) != 1 or arrays[1].shape != arrays[2].shape):
        raise ValueError("Ungültige Merkmalsmatrizen für den LOF-Vergleich.")
    if len(train) < 3 or n_neighbors < 1:
        raise ValueError("Zu wenige Referenzdaten oder ungültige Nachbarzahl.")
    effective = min(n_neighbors, len(train)-1)
    model = LocalOutlierFactor(n_neighbors=effective, novelty=True, metric="euclidean", n_jobs=1)
    model.fit(arrays[0])
    clean_score, changed_score = (-model.score_samples(x) for x in arrays[1:])
    if not np.isfinite(clean_score).all() or not np.isfinite(changed_score).all():
        raise ValueError("LOF liefert nicht endliche Werte.")
    return clean_score, changed_score, model, effective


def ranked_scores(keys, scores):
    keys = validate_record_keys(pd.Series(list(keys), dtype="string"))
    scores = np.asarray(scores, dtype=float)
    if len(keys) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Ungültige Scores oder Kennungen für Rangfolge.")
    ranking = pd.DataFrame({"record_key": keys, "score": scores}).sort_values(
        ["score", "record_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking)+1)
    return ranking


def evaluate_pair(keys, before, after, changes, budget_fraction):
    if not 0 < budget_fraction <= 1:
        raise ValueError("Prüfbudget muss über 0 und höchstens 1 liegen.")
    before_table = ranked_scores(keys, before).rename(columns={"score": "score_before", "rank": "rank_before"})
    after_table = ranked_scores(keys, after).rename(columns={"score": "score_after", "rank": "rank_after"})
    cases = after_table.merge(before_table, on="record_key", validate="one_to_one", sort=False)
    mapping = {r["record_key"]: r for r in changes}
    if len(mapping) != len(changes) or not set(mapping) <= set(keys):
        raise ValueError("Änderungsprotokoll enthält doppelte oder unbekannte Kennungen.")
    if not mapping or len(mapping) == len(keys):
        raise ValueError("Geänderte und unveränderte Prüffälle erforderlich.")
    budget = min(len(keys), max(1, math.ceil(len(keys)*budget_fraction)))
    cases["changed"] = cases.record_key.isin(mapping)
    cases["operator"] = cases.record_key.map(lambda k: mapping.get(k, {}).get("operator", "unchanged"))
    cases["selected_before"] = cases.rank_before <= budget
    cases["selected_after"] = cases.rank_after <= budget
    cases["score_difference"] = cases.score_after - cases.score_before
    selected_changed = int((cases.changed & cases.selected_after).sum())
    selected_before = int((cases.changed & cases.selected_before).sum())
    per_type = {}
    for name in sorted(cases.loc[cases.changed, "operator"].unique()):
        sub = cases.loc[cases.operator.eq(name)]
        per_type[name] = {"changed_rows": len(sub), "selected_after": int(sub.selected_after.sum()),
                          "selected_before": int(sub.selected_before.sum()),
                          "recall_of_changes_at_k": float(sub.selected_after.mean())}
    metrics = {
        "test_rows": len(keys), "changed_rows": len(mapping), "review_budget_k": budget,
        "selected_changed": selected_changed, "selected_unchanged": budget-selected_changed,
        "recall_of_changes_at_k": selected_changed/len(mapping),
        "share_of_changes_in_review": selected_changed/budget,
        "maximum_possible_recall_at_budget": min(budget, len(mapping))/len(mapping),
        "random_selection_expected_recall": budget/len(keys),
        "random_selection_expected_review_share": len(mapping)/len(keys),
        "selected_same_keys_before_changes": selected_before,
        "newly_selected_changed": int((cases.changed & cases.selected_after & ~cases.selected_before).sum()),
        "no_longer_selected_changed": int((cases.changed & ~cases.selected_after & cases.selected_before).sum()),
        "average_precision_for_changes": float(average_precision_score(cases.changed, cases.score_after)),
        "roc_auc_for_changes": float(roc_auc_score(cases.changed, cases.score_after)),
        "per_operator": per_type,
        "confirmed_error_metrics_available": False,
    }
    return cases, metrics
