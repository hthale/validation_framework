"""Konkrete Risiken: Zirkelschluss, lokale Reichweite, Nullkontrolle und Evidenz."""
import os
from pathlib import Path
import sys
import unittest
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_rule_comparison import bootstrap
bootstrap(Path(os.environ.get("FRAMEWORK_TEST_PROJECT", PACKAGE.parents[1])))

import numpy as np
import pandas as pd
from comparison_core import (Protocol, DATE, DATE_MISSING, eligible_columns, predictors_for_target,
    fit_preprocessor, transform, fit_router, route, learn_method, evaluate_selected,
    paired_comparisons, summarize_cases)
from rule_engine import evaluate_rule_on_dataframe
from verify_rule_comparison import assert_same_cases, verify_source_file


def data(n=120, offset=0, constant=False):
    x = np.array(["A", "B"]*(n//2))
    y = np.repeat("yes", n) if constant else np.where(x == "A", "yes", "no")
    return pd.DataFrame({"record_key": [f"rk:{i+offset}" for i in range(n)], "x": x, "y": y,
                         "age": np.arange(n, dtype=float)})


def groups_for(*frames):
    keys = [k for frame in frames for k in frame.record_key]
    return pd.Series(["g:"+k for k in keys], index=keys)


class ComparisonTests(unittest.TestCase):
    def test_project_manifest_keeps_input_hash_distinct_from_code_hashes(self):
        from experiment_io import new_output, finish_output
        from review_common import read_json, file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"original.txt"
            source.write_text("unchanged", encoding="utf-8")
            out = new_output(Path(tmp)/"results")
            finish_output(out, {"source": str(source), "input_sha256": file_sha256(source)})
            manifest = read_json(out/"run_manifest.json")
            self.assertIsInstance(manifest["source_sha256"], dict)
            verify_source_file(manifest)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_source_file(manifest)

    def test_empty_predictions_survive_arrow_roundtrip(self):
        empty = pd.DataFrame({"prediction": [None, None], "covered": [False, False]})
        nonempty = pd.DataFrame({"prediction": ["yes"], "covered": [True]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cases.parquet"
            pd.concat([empty, nonempty], ignore_index=True).to_parquet(path, index=False)
            loaded = pd.read_parquet(path).iloc[:2].reset_index(drop=True)
            assert_same_cases(empty, loaded)
            loaded.loc[0, "prediction"] = "a_real_change"
            with self.assertRaises(AssertionError):
                assert_same_cases(empty, loaded)

    def test_target_copies_and_date_family_are_excluded(self):
        train = data()
        train[DATE] = train.age
        train[DATE_MISSING] = ["yes", "no"]*60
        train["copy"] = train[DATE_MISSING]
        columns = ["x", "age", DATE, DATE_MISSING, "copy"]
        features, excluded = predictors_for_target(train, columns, DATE_MISSING)
        self.assertEqual(features, ["x", "age"])
        self.assertIn("copy", excluded)
        prep, original, schema = fit_preprocessor(train, features)
        changed = train.copy()
        changed[DATE], changed[DATE_MISSING], changed["copy"] = 9999., "changed", "changed"
        np.testing.assert_array_equal(transform(prep, changed, schema)[0], original)

    def test_preprocessing_uses_training_statistics_and_counts_unknown(self):
        train = data()
        prep, original, schema = fit_preprocessor(train, ["x", "age"])
        val = data(20, 1000)
        val["age"], val["x"] = 1000000., "new"
        matrix, unknown = transform(prep, val, schema)
        self.assertEqual(unknown["x"], 20)
        np.testing.assert_array_equal(transform(prep, train, schema)[0], original)
        self.assertTrue((matrix[:, 0] > 1000).all())

    def test_missingness_of_otherwise_constant_category_survives(self):
        train = pd.DataFrame({"optional": ["v", None]*60, "constant": ["v"]*120})
        columns, _ = eligible_columns(train, list(train))
        self.assertEqual(columns, ["optional"])
        _, matrix, _ = fit_preprocessor(train, columns)
        self.assertEqual(matrix.shape[1], 2)
        self.assertFalse(np.array_equal(matrix[0], matrix[1]))

    def test_known_relation_export_matches_and_target_change_is_detected(self):
        train, val, test = data(), data(60, 1000), data(60, 2000)
        groups = groups_for(train, val, test)
        rules, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x"], groups, "global", 42, Protocol())
        selected = [r for r in rules if r["selected_on_validation"]]
        self.assertEqual(len(selected), 2)
        clean, _ = evaluate_selected(test, np.zeros(len(test)), selected, "y", groups, "global", 42, Protocol())
        self.assertTrue(clean.agree.all())
        changed = test.copy()
        changed["y"] = changed["y"].map({"yes": "no", "no": "yes"})
        bad, _ = evaluate_selected(changed, np.zeros(len(test)), selected, "y", groups, "global", 42, Protocol())
        self.assertTrue(bad.disagree_observed.all())
        self.assertTrue(clean.prediction.equals(bad.prediction))

    def test_constant_cluster_rule_cannot_escape_scope_or_cover_noise(self):
        train, val, test = data(constant=True), data(60, 1000, True), data(60, 2000, True)
        groups = groups_for(train, val, test)
        candidates, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x"], groups, "latent_clusters", 42, Protocol())
        rules = [r for r in candidates if r["selected_on_validation"]]
        self.assertEqual(len(rules), 1)
        labels = np.repeat([0, 1, -1], 20)
        cases, _ = evaluate_selected(test, labels, rules, "y", groups, "latent_clusters", 42, Protocol())
        self.assertEqual(int(cases.covered.sum()), 20)
        with self.assertRaises(ValueError):
            evaluate_selected(test, labels, rules, "y", groups, "attribute_clusters", 42, Protocol())
        with self.assertRaises((ValueError, KeyError)):
            evaluate_rule_on_dataframe(test, rules[0])

    def test_many_duplicates_are_not_five_validation_groups(self):
        train, val = data(constant=True), data(60, 1000, True)
        groups = groups_for(train, val)
        groups.loc[val.record_key] = "one-group"
        rules, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x"], groups, "global", 42, Protocol())
        self.assertTrue(rules)
        self.assertFalse(any(r["selected_on_validation"] for r in rules))
        self.assertIn("too_few_validation_groups", rules[0]["rejection_reasons"])

    def test_balanced_uninformative_control_is_not_selected(self):
        train, val = data(), data(60, 1000)
        # Identische Prädiktoren besitzen beide Zielwerte gleich häufig.
        train["x"], val["x"] = "same", "same"
        groups = groups_for(train, val)
        rules, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x"], groups, "global", 42, Protocol())
        self.assertTrue(rules)
        self.assertFalse(any(r["selected_on_validation"] for r in rules))

    def test_local_tree_can_express_known_contextual_relation(self):
        # Isoliert die lokale Regelsuche; Clusterlabels hier bewusst vorgegeben,
        # kein behaupteter Autoencoder-Nachweis.
        train, val = data(240), data(120, 1000)
        for df in (train, val):
            df["context"] = np.repeat(["C", "D"], len(df)//2)
            flip = df.context.eq("D")
            df.loc[flip, "y"] = df.loc[flip, "y"].map({"yes": "no", "no": "yes"})
        groups = groups_for(train, val)
        p = Protocol(max_depth=1)
        global_rules, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x", "context"], groups, "global", 42, p)
        local_rules, _ = learn_method(train, val, train.context.eq("D").astype(int), val.context.eq("D").astype(int),
            "y", ["x", "context"], groups, "attribute_clusters", 42, p)
        self.assertFalse(any(r["selected_on_validation"] for r in global_rules))
        self.assertEqual(sum(r["selected_on_validation"] for r in local_rules), 4)

    def test_real_hdbscan_frozen_prediction_and_all_noise(self):
        rng = np.random.default_rng(17)
        x = np.vstack([rng.normal(-4, .1, (100, 2)), rng.normal(4, .1, (100, 2))])
        p = Protocol(min_cluster_size=15, min_samples=5)
        router, labels, _ = fit_router(x, p)
        self.assertGreaterEqual(len(set(labels)-{-1}), 2)
        before = router.labels_.copy()
        query = np.array([[-4., -4.], [4., 4.]])
        a, _ = route(router, query)
        b, _ = route(router, query[::-1])
        np.testing.assert_array_equal(a, b[::-1])
        np.testing.assert_array_equal(router.labels_, before)
        noise, all_labels, _ = fit_router(np.zeros((120, 2)), p)
        self.assertTrue((all_labels == -1).all())
        result, strengths = route(noise, query)
        self.assertTrue((result == -1).all())
        self.assertTrue((strengths == 0).all())

    def test_missing_target_is_separate_from_observed_disagreement(self):
        train, val, test = data(constant=True), data(60, 1000, True), data(60, 2000, True)
        groups = groups_for(train, val, test)
        rules, _ = learn_method(train, val, np.zeros(len(train)), np.zeros(len(val)),
            "y", ["x"], groups, "global", 42, Protocol())
        test.loc[:9, "y"] = None
        cases, _ = evaluate_selected(test, np.zeros(len(test)), rules, "y", groups, "global", 42, Protocol())
        metrics = summarize_cases(cases)
        self.assertEqual(metrics["missing_target_covered"], 10)
        self.assertEqual(metrics["disagree_observed"], 0)
        self.assertEqual(metrics["agreement_observed_covered"], 1.)

    def test_comparisons_reject_mismatched_cases(self):
        frames = []
        for method in ("global", "attribute_clusters", "latent_clusters"):
            frames.append(pd.DataFrame({"seed": [42], "target": ["y"], "record_key": [method], "method": [method],
                "covered": [False], "agree": [False], "disagree_observed": [False], "missing_target_covered": [False]}))
        with self.assertRaises(ValueError):
            paired_comparisons(pd.concat(frames))


if __name__ == "__main__":
    unittest.main()
