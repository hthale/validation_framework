"""Regressionstests der ersten Korrekturstufe; keine Aussage zur ML-Testgüte."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification import classify_and_prioritize_rules
from check_foundation import dop_name_observations, json_compatible
from phase0_join import merge_phase0_frames
from phase2_ml import (
    build_decision_tree_pipeline, extract_rules_from_decision_tree_pipeline,
    get_phase2_feature_candidates, get_phase2_cluster_feature_candidates,
    get_phase2_target_candidates, prepare_phase2_dataset,
)
from preprocessing import normalize_by_schema
from record_identity import build_record_key, with_record_key
from rule_engine import build_domain_specific_rules, evaluate_rule_on_dataframe, make_rule, validate_rules_on_full_dataframe


def rule(kind="ml_dependency", conditions=None, columns=None, **metadata):
    if kind == "ml_dependency":
        metadata = {"target_col": "target", "predicted_value": "yes", "conditions": conditions or [], **metadata}
    return make_rule("test", "Test", kind, columns or ["target"], "test", "profiling", "Test", metadata=metadata)


def result_for(df, item):
    return validate_rules_on_full_dataframe(df, [item])[0][0]


class RuleMetricsTests(unittest.TestCase):
    def test_conditional_denominator(self):
        df = pd.DataFrame({"x": [0] * 200 + [1] * 800, "target": ["yes"] * 180 + ["no"] * 820})
        result = result_for(df, rule(conditions=["x <= 0.5"]))
        self.assertEqual((result["coverage_count"], result["valid_count"], result["invalid_count"]), (200, 180, 20))
        self.assertEqual((result["coverage_rate"], result["pass_rate"], result["violation_rate"]), (0.2, 0.9, 0.1))

    def test_not_null_checks_missing_values(self):
        result = result_for(pd.DataFrame({"target": [1, None, 3]}), rule("not_null"))
        self.assertEqual((result["coverage_count"], result["invalid_count"]), (3, 1))

    def test_missing_target_is_violation_inside_antecedent(self):
        df = pd.DataFrame({"x": [0, 1], "target": [None, None]})
        result = result_for(df, rule(conditions=["x <= 0.5"]))
        self.assertEqual((result["coverage_count"], result["invalid_count"]), (1, 1))

    def test_no_applicable_cases_are_not_good_evidence(self):
        result = result_for(pd.DataFrame({"x": [1], "target": ["yes"]}), rule(conditions=["x <= 0"]))
        self.assertEqual(result["evaluation_status"], "not_applicable")
        self.assertIsNone(result["pass_rate"])
        classified = classify_and_prioritize_rules([result])[0]
        self.assertEqual(classified["rule_class"], "not_evaluated")
        self.assertIsNone(classified["violation_rate"])
        json.dumps(classified, allow_nan=False)

    def test_missing_feature_is_explicit_invalid_rule(self):
        result = result_for(pd.DataFrame({"target": ["yes"]}), rule(conditions=["onehot_old_blue > 0.5"]))
        self.assertEqual(result["evaluation_status"], "invalid_rule")
        self.assertIsNone(result["coverage_count"])
        self.assertEqual(classify_and_prioritize_rules([result])[0]["recommended_action"], "repair_rule")

    def test_unsupported_condition_is_not_silently_ignored(self):
        result = result_for(pd.DataFrame({"x": [1], "target": ["yes"]}), rule(conditions=["x < 2"]))
        self.assertEqual(result["evaluation_status"], "invalid_rule")

    def test_violation_examples_include_identity_and_condition(self):
        df = pd.DataFrame({"record_key": ["a"], "x": [0], "target": ["no"]})
        _, examples = validate_rules_on_full_dataframe(df, [rule(conditions=["x <= 0.5"])])
        self.assertEqual(examples["test"], [{"record_key": "a", "target": "no", "x": 0}])

    def test_unknown_mapping_input_is_not_applicable(self):
        df = pd.DataFrame({"bodenpixelgroesse": [20, 50, 10], "standardabweichung": [40, 80, None]})
        result = result_for(df, build_domain_specific_rules(df)[0])
        self.assertEqual((result["coverage_count"], result["invalid_count"]), (2, 1))

    def test_empty_population(self):
        result = result_for(pd.DataFrame({"target": pd.Series(dtype="string")}), rule("not_null"))
        self.assertEqual(result["coverage_count"], 0)
        self.assertIsNone(result["pass_rate"])


class IdentityTests(unittest.TestCase):
    def test_ids_stable_when_attributes_change(self):
        df = pd.DataFrame({"xml_name": ["tile_2020.xml", "tile_2023.xml"], "kachelname": ["old", "old"], "value": [1, 2]})
        before = build_record_key(df)
        df["kachelname"] = "changed"
        df["value"] = 999
        pd.testing.assert_series_equal(before, build_record_key(df))

    def test_existing_ids_survive_filename_changes(self):
        df = with_record_key(pd.DataFrame({"xml_name": ["old.xml"]}))
        before = df["record_key"].copy()
        df["xml_name"] = "new.xml"
        pd.testing.assert_series_equal(before, with_record_key(df)["record_key"])

    def test_reordering_and_repeated_assignment_are_stable(self):
        df = with_record_key(pd.DataFrame({"xml_name": ["a.xml", "b.xml"]}))
        pd.testing.assert_frame_equal(with_record_key(df.iloc[::-1]), df.iloc[::-1])

    def test_ambiguous_or_empty_identity_rejected(self):
        for values in [["a", "a"], ["a", None], ["a", " "]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_record_key(pd.DataFrame({"gml_id": values}))

    def test_delivery_namespace_allows_repeated_filename(self):
        df = pd.DataFrame({"xml_name": ["a.xml", "a.xml"], "delivery_id": ["D1", "D2"]})
        self.assertEqual(build_record_key(df).nunique(), 2)

    def test_no_hash_of_arbitrary_attributes(self):
        with self.assertRaises(ValueError):
            build_record_key(pd.DataFrame({"value": [1, 2]}))

    def test_explicit_composite_identity(self):
        df = pd.DataFrame({"tile": ["a", "a"], "date": ["2020", "2023"]})
        self.assertEqual(build_record_key(df, identity_columns=["tile", "date"]).nunique(), 2)

    def test_normalization_preserves_ids_and_missing_values(self):
        df = pd.DataFrame({"record_key": ["0001", "0002"], "category": ["a", None], "empty": [None, None]})
        result = normalize_by_schema(df, {"record_key": "numeric", "category": "categorical", "empty": "unknown"})
        self.assertEqual(result["record_key"].tolist(), ["0001", "0002"])
        self.assertTrue(pd.isna(result.loc[1, "category"]))
        self.assertTrue(result["empty"].isna().all())


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.left = pd.DataFrame({"record_key": ["a", "b"], "value": [10, 20]}, index=[9, 4])
        self.right = pd.DataFrame({"record_key": ["b", "a"], "cluster_label": [2, 1], "combined_outlier_score": [0.2, 0.1], "reconstruction_error": [2.0, 1.0]})

    def test_merge_preserves_order_count_and_index(self):
        result = merge_phase0_frames(self.left, self.right)
        self.assertEqual(result.index.tolist(), [9, 4])
        self.assertEqual(result["cluster_label"].tolist(), [1, 2])
        self.assertEqual(len(result), 2)

    def test_duplicate_phase0_ids_rejected(self):
        self.right["record_key"] = ["a", "a"]
        with self.assertRaises(ValueError):
            merge_phase0_frames(self.left, self.right)

    def test_duplicate_input_ids_rejected(self):
        self.left["record_key"] = ["a", "a"]
        with self.assertRaises(ValueError):
            merge_phase0_frames(self.left, self.right)

    def test_wrong_dataset_rejected(self):
        self.right.loc[0, "record_key"] = "c"
        with self.assertRaises(ValueError):
            merge_phase0_frames(self.left, self.right)

    def test_repeated_merge_rejected(self):
        result = merge_phase0_frames(self.left, self.right)
        with self.assertRaises(ValueError):
            merge_phase0_frames(result, self.right)


class TreeExportTests(unittest.TestCase):
    def assert_rules_match_model(self, train, probe):
        features = [c for c in train if c != "target"]
        X, y, numeric, categorical = prepare_phase2_dataset(train, "target", features)
        model = build_decision_tree_pipeline(numeric, categorical)
        model.fit(X, y)
        rules = extract_rules_from_decision_tree_pipeline(model, "target", "test", min_leaf_samples=1)
        self.assertGreater(len(rules), 1)
        probe = probe.copy()
        probe["target"] = "placeholder"
        prepared, _, _, _ = prepare_phase2_dataset(probe, "target", features)
        expected = model.predict(prepared)
        hits = np.zeros(len(probe), dtype=int)
        actual = np.empty(len(probe), dtype=object)
        for item in rules:
            self.assertTrue(all(isinstance(c, dict) for c in item["metadata"]["conditions"]))
            mask, _, _ = evaluate_rule_on_dataframe(probe, item)
            hits += mask.to_numpy(dtype=int)
            actual[mask.to_numpy()] = item["metadata"]["predicted_value"]
        np.testing.assert_array_equal(hits, np.ones(len(probe), dtype=int))
        np.testing.assert_array_equal(actual, expected)
        json.dumps(rules, allow_nan=False)
        return rules

    def test_categories_with_underscores_missing_and_unknown_values(self):
        train = pd.DataFrame({"sensor_name_type": ["blue_green"] * 100 + ["red"] * 100 + [None] * 100, "target": ["a"] * 100 + ["b"] * 100 + ["c"] * 100})
        rules = self.assert_rules_match_model(train, pd.DataFrame({"sensor_name_type": ["blue_green", "red", None, "never_seen"]}))
        self.assertTrue(all(c["feature"] == "sensor_name_type" for r in rules for c in r["metadata"]["conditions"]))

    def test_numeric_threshold_and_imputation_match_tree(self):
        train = pd.DataFrame({"x": [1.0000001] * 100 + [1.0000035] * 100 + [np.nan] * 100, "target": ["a"] * 100 + ["b"] * 100 + ["a"] * 100})
        self.assert_rules_match_model(train, pd.DataFrame({"x": list(np.linspace(1.0, 1.000004, 101)) + [np.nan]}))

    def test_dropped_all_missing_numeric_feature_keeps_encoder_alignment(self):
        train = pd.DataFrame({"empty": [np.nan] * 200, "x": [1] * 100 + [3] * 100, "target": ["a"] * 100 + ["b"] * 100})
        self.assert_rules_match_model(train, pd.DataFrame({"empty": [np.nan, np.nan], "x": [1, 3]}))

    def test_identity_cannot_be_forced_as_predictor(self):
        df = pd.DataFrame({"record_key": ["a"], "delivery_id": ["d"], "gml_id": ["g"], "umap_x": [1], "x": [2], "target": [3]})
        self.assertEqual(get_phase2_feature_candidates(df, "target", forced_feature_columns=list(df)), ["x"])
        self.assertNotIn("delivery_id", get_phase2_cluster_feature_candidates(df))
        self.assertNotIn("gml_id", get_phase2_target_candidates(pd.concat([df] * 60, ignore_index=True)))


class ProductProfileTests(unittest.TestCase):
    def test_missing_values_in_violation_examples_are_json_null(self):
        data = {"examples": [{"target": float("nan")}, {"target": pd.NA}]}
        encoded = json.dumps(json_compatible(data), allow_nan=False)
        self.assertEqual(json.loads(encoded), {"examples": [{"target": None}, {"target": None}]})

    def test_no_implicit_rgbi_requirement(self):
        df = pd.DataFrame({"spektralkanaele": ["RGB"]})
        self.assertEqual(build_domain_specific_rules(df), [])
        self.assertEqual(result_for(df, build_domain_specific_rules(df, expected_spectral_channels="RGB")[0])["invalid_count"], 0)
        self.assertEqual(result_for(df, build_domain_specific_rules(df, expected_spectral_channels="RGBI")[0])["invalid_count"], 1)

    def test_observation_does_not_modify_original_data(self):
        df = pd.DataFrame({"record_key": ["a"], "xml_name": ["dop_rgb.xml"], "tif_name": ["dop_rgb.tif"], "kachelname": ["dop_rgbi"]})
        before = df.copy(deep=True)
        observations = dop_name_observations(df)
        self.assertEqual(observations["review_status"].tolist(), ["fachlich_zu_klaeren"])
        pd.testing.assert_frame_equal(df, before)


if __name__ == "__main__":
    unittest.main()
