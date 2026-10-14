from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_controlled_comparison import bootstrap
bootstrap(Path(os.environ.get("FRAMEWORK_TEST_PROJECT", PACKAGE.parents[1])))

import numpy as np
import pandas as pd
from synthetic_data import (TARGET, CONTROL, SIGNALS, NOISE_COLUMNS, create_population,
                            scenario_frame, normal_mask, make_test_variants)
from comparison_core import Protocol, learn_method, evaluate_selected, fit_preprocessor, transform
from baseline_core import majority_model, baseline_cases
from evaluation import (summarize_predictions, align_labels, rank_budget, fit_lof, score_lof,
                        validation_threshold, lof_cases, summarize_lof)


class ControlledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attributes, cls.truth = create_population(42, 1200)
        cls.frame, cls.features = scenario_frame(cls.attributes, "basis")
        cls.train, cls.val, cls.test = (cls.frame.loc[cls.truth.partition.eq(part)].reset_index(drop=True)
                                      for part in ("train", "validation", "test"))
        cls.groups = cls.truth.set_index("record_key").group_id
        cls.variants, cls.labels = make_test_variants(cls.train, cls.test, cls.truth, 42)

    def test_groups_are_disjoint_rare_controls_present_and_labels_separate(self):
        self.assertEqual(self.truth.groupby("group_id").partition.nunique().max(), 1)
        self.assertEqual(set(self.truth.loc[self.truth.rare_normal, "partition"]), {"train", "validation", "test"})
        self.assertEqual(len(self.attributes), 2400)
        self.assertTrue(self.attributes.record_key.is_unique)
        self.assertFalse(set(self.features) & {"group_id", "profile", "partition", "is_error", "record_key", TARGET, CONTROL})
        self.assertTrue(normal_mask(self.attributes).all())

    def test_scenarios_share_signal_attributes_and_add_only_nuisance(self):
        noisy, features = scenario_frame(self.attributes, "stoermerkmale")
        pd.testing.assert_frame_equal(noisy[self.frame.columns], self.frame)
        self.assertEqual(features, SIGNALS+NOISE_COLUMNS)

    def test_reference_accepts_rare_and_noise_but_rejects_wrong_combinations(self):
        rare = self.attributes.loc[self.truth.rare_normal].copy()
        self.assertTrue(normal_mask(rare).all())
        rare[NOISE_COLUMNS] = .75
        self.assertTrue(normal_mask(rare).all())
        rare[TARGET] = "A"
        self.assertFalse(normal_mask(rare).any())
        rare[TARGET] = "C"; rare["messwert_b"] = -4.
        self.assertFalse(normal_mask(rare).any())

    def test_error_labels_exactly_match_changed_combinations_and_keep_twins(self):
        for name, frame in self.variants.items():
            expected = np.zeros(len(frame), dtype=bool) if name == "clean" else self.labels.is_error.to_numpy()
            np.testing.assert_array_equal(~normal_mask(frame), expected)
            self.assertEqual(frame.record_key.tolist(), self.test.record_key.tolist())
        bad = self.labels.loc[self.labels.is_error]
        self.assertTrue(bad.group_id.is_unique)
        self.assertTrue(set(bad.group_id) <= set(self.labels.loc[~self.labels.is_error, "group_id"]))
        self.assertGreater(bad.rare_normal.sum(), 0)
        self.assertTrue(self.variants["code_swap"].loc[self.labels.rare_normal & self.labels.is_error, TARGET].eq("A").all())

    def test_swapped_measurements_are_known_training_values(self):
        changed = self.variants["signal_swap"]
        self.assertTrue(changed.loc[self.labels.is_error, "messwert_b"].isin(self.train.messwert_b).all())
        pd.testing.assert_frame_equal(changed.drop(columns="messwert_b"), self.test.drop(columns="messwert_b"))

    def test_global_tree_recovers_conditional_values_and_detects_rare_to_majority(self):
        found, _ = learn_method(self.train, self.val, np.zeros(len(self.train)), np.zeros(len(self.val)),
            TARGET, self.features, self.groups, "global", 42, Protocol())
        selected = [r for r in found if r["selected_on_validation"]]
        self.assertGreaterEqual(len(selected), 3)
        cases, _ = evaluate_selected(self.variants["code_swap"], np.zeros(len(self.test)), selected,
            TARGET, self.groups, "global", 42, Protocol())
        metric = summarize_predictions(cases, self.labels)
        self.assertEqual(metric["recall_all_errors"], 1.)
        self.assertEqual(metric["false_positives"], 0)
        self.assertEqual(metric["rare_errors_detected"], metric["rare_errors"])

    def test_paired_uninformative_control_yields_no_selected_global_rule(self):
        found, _ = learn_method(self.train, self.val, np.zeros(len(self.train)), np.zeros(len(self.val)),
            CONTROL, self.features, self.groups, "global", 42, Protocol())
        self.assertFalse(any(r["selected_on_validation"] for r in found))
        paired = self.train.assign(group=self.train.record_key.map(self.groups))
        self.assertTrue(paired.groupby("group")[CONTROL].nunique().eq(2).all())

    def test_majority_false_alarms_are_not_mistaken_for_rule_success(self):
        m = majority_model(self.train, TARGET)
        cases = baseline_cases(self.test, TARGET, self.groups, 42, m)
        labels = self.labels.copy(); labels["is_error"] = False
        metric = summarize_predictions(cases, labels)
        self.assertGreater(metric["false_positives"], 0)
        self.assertEqual(metric["rare_clean_false_positive_rate"], 1.)
        self.assertIsNone(metric["recall_all_errors"])

    def test_target_is_invisible_to_rule_representation_but_visible_to_lof(self):
        prep, _, schema = fit_preprocessor(self.train, self.features)
        np.testing.assert_array_equal(transform(prep, self.test, schema)[0], transform(prep, self.variants["code_swap"], schema)[0])
        prep, _, schema = fit_preprocessor(self.train, [*self.features, TARGET])
        clean, _ = transform(prep, self.test, schema)
        changed, _ = transform(prep, self.variants["code_swap"], schema)
        np.testing.assert_array_equal((clean != changed).any(axis=1), self.labels.is_error.to_numpy())

    def test_abstention_stays_in_error_denominator(self):
        labels = pd.DataFrame({"record_key": ["a", "b", "c"], "group_id": ["g1", "g2", "g3"],
                               "is_error": [True, True, False], "rare_normal": [False, True, True]})
        cases = pd.DataFrame({"record_key": ["a", "b", "c"], "covered": [True, False, False],
                              "disagree_observed": [True, False, False], "missing_target_covered": False})
        m = summarize_predictions(cases, labels)
        self.assertEqual(m["recall_all_errors"], .5)
        self.assertEqual(m["abstained"], 2)
        self.assertEqual(m["rare_clean_covered"], 0)

    def test_equal_scores_have_exact_budget_and_order_independent_ties(self):
        keys = [f"id{i}" for i in range(20)]
        ranks, selected = rank_budget(keys, np.ones(20))
        reverse_ranks, reverse_selected = rank_budget(keys[::-1], np.ones(20))
        self.assertEqual(selected.sum(), 2)
        self.assertEqual(dict(zip(keys, ranks)), dict(zip(keys[::-1], reverse_ranks)))
        threshold = validation_threshold(np.ones(20))
        self.assertFalse(lof_cases(keys, np.ones(20), threshold).above_validation_threshold.any())

    def test_unseen_lof_queries_do_not_change_reference_or_other_scores(self):
        rng = np.random.default_rng(4)
        train = rng.normal(0, .2, (70, 3)); query = rng.normal(0, .2, (8, 3))
        model = fit_lof(train)
        clean = score_lof(model, query)
        changed = query.copy(); changed[0] = 100.
        modified = score_lof(model, changed)
        np.testing.assert_array_equal(clean[1:], modified[1:])
        self.assertGreater(modified[0], max(clean))
        self.assertEqual(model.n_samples_fit_, 70)

    def test_reference_label_alignment_rejects_missing_or_duplicate_keys(self):
        cases = pd.DataFrame({"record_key": ["a", "b"]})
        with self.assertRaises(ValueError): align_labels(cases, pd.DataFrame({"record_key": ["a", "a"]}))
        with self.assertRaises(ValueError): align_labels(cases, pd.DataFrame({"record_key": ["a", "c"]}))

    def test_lof_budget_precision_recall_and_rare_false_positive_are_separate(self):
        labels = pd.DataFrame({"record_key": list("abcd"), "group_id": list("1234"),
                               "is_error": [True, False, True, False], "rare_normal": [False, True, False, False]})
        cases = lof_cases(labels.record_key, [4., 3., 2., 1.], threshold=3.5, budget=.5)
        metrics = summarize_lof(cases, labels, 3.5)
        self.assertEqual(metrics["budget"]["precision"], .5)
        self.assertEqual(metrics["budget"]["recall"], .5)
        self.assertEqual(metrics["budget"]["rare_clean_selected"], 1)
        self.assertEqual(metrics["threshold"]["false_positives"], 0)


if __name__ == "__main__":
    unittest.main()
