import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from date_roles import apply_date_roles, date_diagnostics
from phase2_settings import Phase2Settings
from phase2_split import load_or_create_split, make_grouped_split, select_partition, validate_split
from phase2_ml import build_phase2_ml_rules, build_ssl_cluster_rules, get_phase2_feature_candidates
from preprocessing import normalize_by_schema
from profiling import profile_dataframe


def fixture():
    groups = np.repeat(np.arange(120), 2)
    labels = np.array(["a", "b", "c"])[groups % 3]
    return pd.DataFrame({"record_key": [f"record-{i:04d}" for i in range(len(groups))],
                         "tile": groups, "target": labels, "signal": labels,
                         "numeric": np.arange(len(groups), dtype=float)})


def settings():
    return Phase2Settings(group_columns=("tile",), max_depths=(2, 5), min_samples_leaves=(2, 8),
                          min_validation_rule_cases=2)


def dependency_run(df, split, protocol=None):
    return build_phase2_ml_rules(df, target_candidates=["target"], excluded_columns=["tile", "numeric"],
                                settings=protocol or settings(), split=split)


class SplitTests(unittest.TestCase):
    def test_same_tile_across_years_never_crosses_partitions(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        joined = df.merge(split, on="record_key", validate="one_to_one")
        self.assertEqual(joined.groupby("tile")["partition"].nunique().max(), 1)
        self.assertEqual(set(split.partition), {"train", "validation", "test"})
        self.assertEqual(len(split), len(df))
        pd.testing.assert_frame_equal(split, make_grouped_split(df.sample(frac=1, random_state=2), settings()))

    def test_persisted_split_reused_but_changed_groups_rejected(self):
        df = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            first = load_or_create_split(df, path, settings())
            before = path.read_bytes()
            second = load_or_create_split(df.iloc[::-1], path, settings())
            self.assertEqual(first.to_dict("records"), second.to_dict("records"))
            self.assertEqual(path.read_bytes(), before)
            changed = df.copy()
            changed.loc[0, "tile"] = 9999
            with self.assertRaises(ValueError):
                load_or_create_split(changed, path, settings())
            with self.assertRaises(ValueError):
                load_or_create_split(df, path, replace(settings(), random_state=1))
            with self.assertRaises(ValueError):
                load_or_create_split(df.iloc[1:], path, settings())

    def test_missing_group_and_cross_partition_group_rejected(self):
        df = fixture()
        with self.assertRaises(ValueError):
            make_grouped_split(df.drop(columns="tile"), settings())
        broken = df.copy()
        broken.loc[0, "tile"] = np.nan
        with self.assertRaises(ValueError):
            make_grouped_split(broken, settings())
        split = make_grouped_split(df, settings())
        split.loc[0, "partition"] = "validation" if split.loc[1, "partition"] != "validation" else "test"
        with self.assertRaises(ValueError):
            validate_split(df, split, settings())


class ValidationTests(unittest.TestCase):
    def test_multiclass_training_and_group_cv_produce_review_candidates(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        rules, reports = dependency_run(df, split)
        report = reports[0]
        self.assertGreaterEqual(report["model_leaf_count"], 3)
        self.assertTrue(report["model_gate_passed"])
        self.assertEqual(report["validation_metrics"]["accuracy"], 1)
        self.assertEqual(len(rules), 3)
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(report["cross_validation"]["scope"], "training_partition_only")
        self.assertTrue(all(f["group_overlap_count"] == 0 for f in report["cross_validation"]["folds"]))
        self.assertTrue(all(r["status"] == "proposed" for r in rules))
        self.assertTrue(all(r["metadata"]["validation"]["population_count"] == report["validation_rows"] for r in rules))
        json.dumps(reports, allow_nan=False)

    def test_reserved_test_values_do_not_change_model_selection_or_rules(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        rules, reports = dependency_run(df, split)
        changed = df.copy()
        is_test = changed.record_key.isin(split.loc[split.partition.eq("test"), "record_key"])
        changed.loc[is_test, "target"] = "unseen-test-label"
        changed.loc[is_test, "signal"] = "completely-different"
        changed.loc[is_test, "numeric"] = 1e20
        other_rules, other_reports = dependency_run(changed, split)
        self.assertEqual(reports, other_reports)
        self.assertEqual(rules, other_rules)

    def test_validation_labels_do_not_select_parameters_and_bad_model_is_rejected(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        _, reports = dependency_run(df, split)
        changed = df.copy()
        is_val = changed.record_key.isin(split.loc[split.partition.eq("validation"), "record_key"])
        changed.loc[is_val, "target"] = "wrong"
        rules, changed_reports = dependency_run(changed, split)
        other = changed_reports[0]
        self.assertEqual(reports[0]["cross_validation"], other["cross_validation"])
        self.assertEqual(reports[0]["selected_parameters"], other["selected_parameters"])
        self.assertEqual([r["expression"] for r in reports[0]["candidate_rules"]], [r["expression"] for r in other["candidate_rules"]])
        self.assertFalse(other["model_gate_passed"])
        self.assertEqual(rules, [])
        self.assertTrue(other["candidate_rules"])

    def test_rule_support_gate_and_missing_validation_targets(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        val_key = split.loc[split.partition.eq("validation"), "record_key"].iloc[0]
        df.loc[df.record_key.eq(val_key), "target"] = None
        _, reports = dependency_run(df, split)
        report = reports[0]
        self.assertEqual(report["validation_missing_target_rows"], 1)
        evaluations = [r["metadata"]["validation"] for r in report["candidate_rules"]]
        self.assertEqual(sum(r["coverage_count"] for r in evaluations), report["validation_rows"])
        self.assertEqual(sum(r["invalid_count"] for r in evaluations), 1)
        rules, low_support = dependency_run(df, split, replace(settings(), min_validation_rule_cases=1000))
        self.assertEqual(rules, [])
        self.assertTrue(low_support[0]["model_gate_passed"])
        self.assertTrue(all("too_few_applicable_validation_cases" in r["metadata"]["rejection_reasons"] for r in low_support[0]["candidate_rules"]))

    def test_empty_labelled_validation_is_not_positive_evidence(self):
        df = fixture()
        split = make_grouped_split(df, settings())
        val_keys = split.loc[split.partition.eq("validation"), "record_key"]
        df.loc[df.record_key.isin(val_keys), "target"] = None
        rules, reports = dependency_run(df, split)
        self.assertEqual(rules, [])
        self.assertFalse(reports[0]["model_gate_passed"])
        self.assertIsNone(reports[0]["validation_metrics"]["accuracy"])

    def test_technical_paths_cannot_be_forced_and_unsupported_backend_fails(self):
        df = fixture()
        df["xml_path"] = "path"
        df["kachelname"] = "name"
        features = get_phase2_feature_candidates(df, "target", forced_feature_columns=list(df))
        self.assertNotIn("record_key", features)
        self.assertNotIn("xml_path", features)
        self.assertNotIn("kachelname", features)
        with self.assertRaises(ValueError):
            build_phase2_ml_rules(df, backend="flaml", settings=settings())

    def test_cluster_descriptions_report_all_eligible_clusters_and_noise(self):
        df = fixture()
        df["cluster_label"] = np.repeat(np.arange(120) % 3, 2)
        df.loc[:3, "cluster_label"] = -1
        rules, reports = build_ssl_cluster_rules(df, excluded_columns=["tile", "numeric"], settings=settings())
        report = reports[0]
        self.assertEqual(report["included_clusters"], [0, 1, 2])
        self.assertEqual(report["model_rows"], 236)
        self.assertEqual(report["excluded_noise_or_missing_rows"], 4)
        self.assertFalse(report["independent_validation"])
        self.assertTrue(rules)
        self.assertTrue(all(r["rule_type"] == "ssl_cluster_rule" for r in rules))
        self.assertTrue(all(r["metadata"]["evaluation_scope"] == "descriptive_in_sample" for r in rules))


class DateRoleTests(unittest.TestCase):
    def test_low_cardinality_dates_override_categorical_and_preserve_invalid_values(self):
        df = pd.DataFrame({"aktualitaet": ["2020-01-03", "bad-date", None, "2020-01-03"]})
        original = df.copy(deep=True)
        schema = apply_date_roles({"aktualitaet": "categorical"}, df.columns)
        self.assertEqual(schema["aktualitaet"], "date")
        normalized = normalize_by_schema(df, schema)
        self.assertEqual(normalized.loc[1, "aktualitaet"], "bad-date")
        self.assertTrue(pd.isna(normalized.loc[2, "aktualitaet"]))
        detail = date_diagnostics(df, ["aktualitaet"])[0]
        self.assertEqual((detail["parseable_count"], detail["invalid_count"], detail["missing_count"]), (2, 1, 1))
        profile = profile_dataframe(df, [], ["aktualitaet"], 20)["column_profiles"]["aktualitaet"]
        self.assertEqual(profile["dtype_group"], "date")
        self.assertEqual(profile["profile"]["invalid_count"], 1)
        pd.testing.assert_frame_equal(df, original)

    def test_arbitrary_integer_is_not_a_valid_nanosecond_date(self):
        detail = date_diagnostics(pd.DataFrame({"aktualitaet": [123456]}), ["aktualitaet"])[0]
        self.assertEqual(detail["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
