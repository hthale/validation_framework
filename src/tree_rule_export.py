"""Baumpfade mit Originalspalten und gespeicherter Vorverarbeitung exportieren."""
from __future__ import annotations

import numpy as np
from sklearn.tree import _tree

from rule_conditions import condition_text


def _python_value(value):
    return value.item() if isinstance(value, np.generic) else value


def _feature_descriptions(model) -> dict[int, dict]:
    """Zuordnung aus Encoder-Metadaten; Unterstriche in Namen bleiben eindeutig."""
    prep = model.named_steps["preprocessor"]
    descriptions = {}
    for name, transformer, columns in prep.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        output_slice = prep.output_indices_[name]
        if output_slice.start == output_slice.stop:
            continue
        index = output_slice.start
        if name == "num" and list(transformer.named_steps) == ["imputer"]:
            imputer = transformer.named_steps["imputer"]
            statistics = dict(zip(columns, imputer.statistics_))
            # Vollständig leere numerische Spalten können vom Imputer entfallen.
            for col in imputer.get_feature_names_out(columns):
                value = _python_value(statistics[col])
                if not np.isfinite(value):
                    raise ValueError(f"Nicht endlicher Imputationswert für {col!r}.")
                descriptions[index] = {
                    "feature": str(col), "kind": "numeric", "impute_value": value,
                }
                index += 1
        elif name == "cat" and list(transformer.named_steps) == ["imputer", "onehot"]:
            encoder = transformer.named_steps["onehot"]
            if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
                raise ValueError("Baumexport unterstützt hier nur vollständiges One-Hot-Encoding.")
            for col, categories in zip(columns, encoder.categories_):
                for category in categories:
                    if not isinstance(category, str):
                        raise ValueError("Kategoriale Features zuerst mit prepare_phase2_dataset vorbereiten.")
                    descriptions[index] = {
                        "feature": str(col), "kind": "category", "category": str(category),
                        "impute_value": str(transformer.named_steps["imputer"].fill_value),
                    }
                    index += 1
        else:
            raise ValueError(f"Nicht unterstützte Vorverarbeitung im Baumexport: {name!r}")
        if index != output_slice.stop:
            raise ValueError(f"Encoder-Zuordnung für {name!r} ist nicht vollständig.")
    return descriptions


def extract_tree_rules(model, target_col: str, source_backend: str, min_leaf_samples: int = 50) -> list[dict]:
    clf = model.named_steps["classifier"]
    if getattr(clf, "n_outputs_", 1) != 1:
        raise ValueError("Baumexport erwartet genau eine Zielspalte.")
    tree = clf.tree_
    descriptions = _feature_descriptions(model)
    leaf_predictions = {
        int(np.argmax(tree.value[node][0]))
        for node in range(tree.node_count) if tree.feature[node] == _tree.TREE_UNDEFINED
    }
    if len(leaf_predictions) <= 1:
        return []
    rules = []

    def branch_condition(feature_index: int, threshold: float, is_left: bool) -> dict:
        desc = descriptions[feature_index]
        if desc["kind"] == "numeric":
            return {
                "feature": desc["feature"], "operator": "<=" if is_left else ">",
                "value": float(threshold), "impute_value": desc["impute_value"],
                "numeric_cast": "float32",
            }
        if not 0.0 <= threshold < 1.0:
            raise ValueError("Unerwartete Schwelle auf einem One-Hot-Merkmal.")
        return {
            "feature": desc["feature"], "operator": "!=" if is_left else "==",
            "value": desc["category"], "coerce": "string",
            "impute_value": desc["impute_value"],
        }

    def recurse(node: int, conditions: list[dict]) -> None:
        feature = tree.feature[node]
        if feature != _tree.TREE_UNDEFINED:
            threshold = tree.threshold[node]
            recurse(tree.children_left[node], conditions + [branch_condition(feature, threshold, True)])
            recurse(tree.children_right[node], conditions + [branch_condition(feature, threshold, False)])
            return
        sample_count = int(tree.n_node_samples[node])
        if sample_count < min_leaf_samples or not conditions:
            return
        predicted = _python_value(clf.classes_[int(np.argmax(tree.value[node][0]))])
        rules.append({
            "rule_id": f"ml__{target_col}__leaf_{node}",
            "rule_name": f"Decision Tree rule for {target_col}",
            "rule_type": "ml_dependency",
            "columns": list(dict.fromkeys([c["feature"] for c in conditions] + [target_col])),
            "expression": "IF " + " AND ".join(condition_text(c) for c in conditions)
                + f" THEN {target_col} = {predicted!r}",
            "source": "ml_phase2", "status": "proposed",
            "reason": "Regelkandidat aus einem Decision Tree; unabhängige Validierung erforderlich",
            "origin_runs": [],
            "metadata": {
                "target_col": target_col, "predicted_value": predicted, "conditions": conditions,
                "condition_schema_version": 1, "leaf_sample_count": sample_count,
                "screening_backend": source_backend, "materialization_backend": "decision_tree",
                "preprocessing_note": "Bedingungen enthalten trainierte Imputationswerte; die Anzeige ist verkürzt.",
            },
            "evidence_type": "ml_screening_plus_tree_materialization",
            "evidence_detail": f"Screening mit {source_backend}; keine Aussage über unabhängige Testgüte",
        })

    recurse(0, [])
    return rules
