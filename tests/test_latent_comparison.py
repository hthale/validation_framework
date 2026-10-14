"""Prüft wissenschaftlich relevante Grenzen des Vergleichs und exakte Baumexporte."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment_io import read_verified_run
from generic_clusters import cluster_frame, describe_clusters, complexity_html
from generic_schema import infer_columns, normalize_columns, tabular_preparation, choose_groups
from latent_comparison import (grouped_partition, transform_frozen, baseline_matrix, inject_changes,
                               novel_lof, evaluate_pair, ranked_scores)
from phase2_settings import Phase2Settings
from review_common import file_sha256, write_json
from rule_engine import evaluate_rule_on_dataframe


class ClusterDescriptionsTests(unittest.TestCase):
    def data(self):
        frame = pd.DataFrame({"record_key": [f"k{i:03}" for i in range(80)],
                              "aufnahmedatum": ["2020-01-01"]*40+["2025-01-01"]*40,
                              "konstante": [7]*80})
        contract = infer_columns(frame)
        normalized, _ = normalize_columns(frame, contract)
        labels = pd.DataFrame({"record_key": frame.record_key, "cluster_label": [0]*40+[1]*40})
        return frame, normalized, contract, labels

    def test_dates_explain_clusters_without_changing_source_or_labels(self):
        raw, normalized, contract, latent = self.data()
        before, labels = normalized.copy(deep=True), latent.copy(deep=True)
        rules, reports = describe_clusters(normalized, contract, latent, Phase2Settings())
        report = reports[0]
        self.assertFalse(report["independent_validation"])
        self.assertFalse(report["complexity_comparison"][0]["success"])
        self.assertEqual(report["selected_variant"], "mit_datum_tiefe_4")
        self.assertEqual(report["in_sample_fidelity"]["accuracy"], 1)
        self.assertEqual(len(rules), 2)
        self.assertTrue(all("aufnahmedatum" in r["display_expression"] for r in rules))
        self.assertTrue(all(not r["metadata"]["eligible_for_review_as_quality_rule"] for r in rules))
        pd.testing.assert_frame_equal(normalized, before)
        pd.testing.assert_frame_equal(latent, labels)

    def test_numeric_date_export_covers_every_row_once_including_missing(self):
        raw, _, contract, latent = self.data()
        raw.loc[0, "aufnahmedatum"] = None
        normalized, _ = normalize_columns(raw, contract)
        rules, reports = describe_clusters(normalized, contract, latent, Phase2Settings())
        work, _ = cluster_frame(normalized, contract, latent)
        covered = np.zeros(len(work), dtype=int)
        correct = 0
        for rule in rules:
            applies, valid, _ = evaluate_rule_on_dataframe(work, rule)
            covered += applies.to_numpy(dtype=int)
            correct += int(valid.sum())
            self.assertTrue(all(c["numeric_cast"] == "float32" for c in rule["metadata"]["conditions"]))
        np.testing.assert_array_equal(covered, np.ones(len(work)))
        self.assertAlmostEqual(correct/len(work), reports[0]["in_sample_fidelity"]["accuracy"])

    def test_complexity_comparison_reports_unrepresented_clusters(self):
        # 18 Gruppen benötigen mindestens 18 Blätter; Tiefe 4 hat höchstens 16.
        n = 18*12
        raw = pd.DataFrame({"record_key": [f"k{i:04}" for i in range(n)], "wert": np.arange(n, dtype=float)})
        contract = infer_columns(raw)
        normal, _ = normalize_columns(raw, contract)
        labels = pd.DataFrame({"record_key": raw.record_key, "cluster_label": np.arange(n)//12})
        _, reports = describe_clusters(normal, contract, labels, Phase2Settings())
        variants = reports[0]["complexity_comparison"]
        self.assertTrue(variants[0]["clusters_without_predicted_leaf"])
        # Mehr Kapazität erzwingt bei einem gierig aufgebauten Baum keine
        # vollständige Beschreibung; verbliebene Lücken müssen sichtbar sein.
        self.assertGreater(max(v["predicted_cluster_count"] for v in variants[1:]), variants[0]["predicted_cluster_count"])
        for v in variants:
            self.assertEqual(len(v["clusters_without_predicted_leaf"]), 18-v["predicted_cluster_count"])
        self.assertIn("Macro-Recall", complexity_html(reports))

    def test_noise_only_is_documented_without_rules(self):
        _, normalized, contract, latent = self.data()
        latent["cluster_label"] = -1
        rules, reports = describe_clusters(normalized, contract, latent, Phase2Settings())
        self.assertFalse(rules)
        self.assertFalse(reports[0]["success"])
        self.assertEqual(reports[0]["excluded_noise_or_missing_rows"], 80)

    def test_missing_or_duplicate_identity_is_rejected(self):
        _, normal, contract, latent = self.data()
        with self.assertRaises(ValueError):
            cluster_frame(normal, contract, latent.iloc[:-1])
        latent.loc[1, "record_key"] = latent.loc[0, "record_key"]
        with self.assertRaises(ValueError):
            cluster_frame(normal, contract, latent)


class ComparisonTests(unittest.TestCase):
    def raw(self, n=120):
        return pd.DataFrame({"record_key": [f"k{i:04}" for i in range(n)],
            "einheit_id": [f"g{i//2}" for i in range(n)], "nummer": np.arange(n, dtype=float),
            "typ": ["A", "B"]*(n//2), "datum": ["2020-01-01", "2025-01-01"]*(n//2), "konstant": [3]*n})

    def prep(self):
        raw = self.raw()
        contract = infer_columns(raw)
        normal, _ = normalize_columns(raw, contract)
        frame, prep, meta = tabular_preparation(normal.iloc[:80], contract)
        return raw, contract, normal, frame, prep, meta

    def test_groups_are_disjoint_and_partition_survives_reordering(self):
        grouped, _ = choose_groups(self.raw(), ["einheit_id"])
        a, split_a = grouped_partition(grouped, 42)
        b, split_b = grouped_partition(grouped.sample(frac=1, random_state=2), 42)
        self.assertEqual(split_a.groupby("group_id").partition.nunique().max(), 1)
        pd.testing.assert_frame_equal(split_a, split_b)
        pd.testing.assert_frame_equal(a, b)

    def test_frozen_preparation_preserves_training_scale_and_unknown_categories(self):
        _, contract, normal, _, prep, meta = self.prep()
        before = copy.deepcopy(prep["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_)
        future = normal.iloc[80:].copy()
        future["nummer"] = 1e6
        future["typ"] = "new_category"
        future["datum"] = None
        _, matrix = transform_frozen(future, prep, meta)
        np.testing.assert_array_equal(before, prep["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix.shape[1], prep["X_tab"].shape[1])
        self.assertGreater(matrix[:, 1].max(), 1000)

    def test_frozen_category_missing_handling_matches_even_if_string_limit_exceeded(self):
        raw = self.raw()
        raw["zweite_kategorie"] = [None, "X", "Y"]*40
        contract = infer_columns(raw)
        normal, _ = normalize_columns(raw, contract)
        _, prep, meta = tabular_preparation(normal, contract, max_string_fields=1)
        _, again = transform_frozen(normal, prep, meta)
        np.testing.assert_array_equal(prep["X_tab"], again)

    def test_text_baseline_has_same_mean_embedding_information(self):
        tokens = np.array([[1, 2, 2, 0], [2, 1, 2, 0], [0, 0, 0, 0], [3, 0, 0, 0]])
        histogram = baseline_matrix(np.zeros((4, 1)), [tokens], 4)[:, 1:]
        embedding = np.array([[0, 0], [1, -2], [4, 1], [-7, 3]], dtype=float)
        expected = np.stack([embedding[t[t != 0]].mean(axis=0) if (t != 0).any() else np.zeros(2) for t in tokens])
        np.testing.assert_allclose(histogram @ embedding[1:], expected, rtol=1e-6)
        np.testing.assert_array_equal(histogram[0], histogram[1])

    def test_changes_are_reproducible_one_cell_per_key_and_preserve_training(self):
        _, contract, normal, _, prep, meta = self.prep()
        train, test = normal.iloc[:80].copy(), normal.iloc[80:].reset_index(drop=True)
        train_copy, test_copy = train.copy(deep=True), test.copy(deep=True)
        changed, log = inject_changes(test, train, contract, meta, .25, 42)
        repeated, log2 = inject_changes(test, train, contract, meta, .25, 42)
        pd.testing.assert_frame_equal(changed, repeated)
        self.assertEqual(len(log), 10)
        self.assertEqual(len(set(r["record_key"] for r in log)), 10)
        equal = (changed.eq(test) | (changed.isna() & test.isna()))
        counts = (~equal).sum(axis=1)
        self.assertEqual(int(counts.sum()), 10)
        self.assertLessEqual(counts.max(), 1)
        self.assertTrue(all(r["column"] not in {"konstant", "record_key", "einheit_id"} for r in log))
        pd.testing.assert_frame_equal(train, train_copy)
        pd.testing.assert_frame_equal(test, test_copy)

    def test_novel_lof_does_not_refit_on_test_and_unchanged_rows_stay_equal(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(80, 2))
        before = rng.normal(size=(12, 2))
        after = before.copy()
        after[0] = [100, 100]
        a, b, model, k = novel_lof(train, before, after, 20)
        self.assertTrue(model.novelty)
        np.testing.assert_array_equal(model._fit_X, train)
        np.testing.assert_array_equal(a[1:], b[1:])
        self.assertEqual(int(np.argmax(b)), 0)
        self.assertEqual(k, 20)

    def test_metrics_use_same_budget_count_already_selected_and_accept_negative_gain(self):
        keys = [f"k{i}" for i in range(6)]
        changes = [{"record_key": "k0", "operator": "shift"}, {"record_key": "k1", "operator": "shift"}]
        cases, m = evaluate_pair(keys, [9, 1, 8, 2, 3, 4], [9, 10, 8, 2, 3, 4], changes, 1/3)
        self.assertEqual(m["review_budget_k"], 2)
        self.assertEqual(m["selected_same_keys_before_changes"], 1)
        self.assertEqual(m["newly_selected_changed"], 1)
        self.assertEqual(m["recall_of_changes_at_k"], 1)
        _, lower = evaluate_pair(keys, [1]*6, [1, 1, 9, 10, 1, 1], changes, 1/3)
        self.assertLess(lower["recall_of_changes_at_k"]-m["recall_of_changes_at_k"], 0)
        self.assertFalse(m["confirmed_error_metrics_available"])

    def test_ties_have_stable_ids_and_invalid_scores_are_rejected(self):
        self.assertEqual(ranked_scores(["c", "a", "b"], [1, 1, 1]).record_key.tolist(), ["a", "b", "c"])
        with self.assertRaises(ValueError): ranked_scores(["a", "a"], [1, 1])
        with self.assertRaises(ValueError): ranked_scores(["a", "b"], [1, np.nan])

    def test_modified_run_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "run_status.json", {"status": "completed"})
            write_json(root / "input.json", {"value": 1})
            write_json(root / "run_manifest.json", {"output_sha256": {"input.json": file_sha256(root / "input.json")}})
            read_verified_run(root, ["input.json"])
            write_json(root / "input.json", {"value": 2})
            with self.assertRaises(ValueError): read_verified_run(root, ["input.json"])


if __name__ == "__main__":
    unittest.main()
