"""Gezielte Tests für Referenzdaten, Testtrennung und ehrliche Fehlerzählung.

Aufruf: python testpaket_05_regelableitung/tests/test_synthetic_rule_check.py --project .
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_synthetic_rule_check as check

PROJECT = Path.cwd()


class SyntheticRuleCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fw = check.load_framework(PROJECT)
        cls.settings = cls.fw.phase2_settings.Phase2Settings(group_columns=("test_group",))
        cls.frame = check.create_reference()
        cls.split = cls.fw.phase2_split.make_grouped_split(cls.frame, cls.settings)
        cls.clean = cls.fw.phase2_split.select_partition(cls.frame, cls.split, "test")
        cls.rules, cls.reports = check.learn_rules(cls.frame, cls.split, cls.settings, cls.fw)
        cls.mixed, cls.truth, cls.changes = check.make_test_copy(cls.clean, 1042)

    def test_known_relations_are_learned_and_control_is_rejected(self):
        result, _, _ = check.assess(self.clean, self.mixed, self.truth, self.rules, self.reports, self.fw)
        self.assertTrue(result["function_check_passed"], result["checks"])
        self.assertEqual(sum(r["recovered"] for r in result["relations"]), 3)
        self.assertEqual(result["metrics"]["combined"]["mixed"]["true_positives"], 40)
        self.assertEqual(result["metrics"]["combined"]["mixed"]["false_positives"], 0)
        self.assertEqual(result["control_validation_accuracy"], .5)
        self.assertTrue(result["checks"]["uninformative_control_not_selected"])

    def test_only_declared_predictors_are_used_and_groups_do_not_overlap(self):
        for report in self.reports:
            self.assertEqual(report["feature_cols_used"], list(check.FEATURES))
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(report["thresholds"], check.THRESHOLDS)
            self.assertTrue(all(f["group_overlap_count"] == 0 for f in report["cross_validation"]["folds"]))
        joined = self.frame.merge(self.split, on="record_key", validate="one_to_one")
        self.assertEqual(joined.groupby("test_group").partition.nunique().max(), 1)
        self.assertEqual(self.frame.record_key.nunique(), 600)

    def test_changing_reserved_test_values_does_not_change_learned_rules(self):
        changed = self.frame.copy(deep=True)
        test_keys = self.split.loc[self.split.partition.eq("test"), "record_key"]
        is_test = changed.record_key.isin(test_keys)
        changed.loc[is_test, "bandanzahl"] = 999
        changed.loc[is_test, "pixelgroesse_cm"] = -200
        changed.loc[is_test, "exportprofil"] = "TEST_ONLY"
        changed.loc[is_test, "crs_code"] = 123
        changed.loc[is_test, "kanaltyp"] = "UNSEEN_TEST"
        changed.loc[is_test, check.CONTROL] = "CHANGED"
        rules, reports = check.learn_rules(changed, self.split, self.settings, self.fw)
        self.assertEqual(rules, self.rules)
        self.assertEqual(reports, self.reports)

    def test_injection_log_and_labels_match_actual_changes(self):
        original = self.clean.sort_values("record_key").reset_index(drop=True).copy(deep=True)
        reconstructed = original.copy(deep=True).set_index("record_key", drop=False)
        reconstructed.bandanzahl = reconstructed.bandanzahl.astype("Int64")
        for item in self.changes:
            self.assertEqual(reconstructed.at[item["record_key"], item["field"]], item["old_value"])
            reconstructed.at[item["record_key"], item["field"]] = item["new_value"]
        pd.testing.assert_frame_equal(reconstructed.reset_index(drop=True), self.mixed)
        pd.testing.assert_frame_equal(original, self.clean.sort_values("record_key").reset_index(drop=True))
        truth = {r["record_key"]: r["is_injected_error"] for r in self.truth}
        for row in self.mixed.to_dict("records"):
            expected = check.reference_values(row["kanaltyp"], row["detailstufe"])
            mismatch = any(pd.isna(row[field]) or row[field] != value for field, value in expected.items())
            self.assertEqual(bool(mismatch), truth[row["record_key"]])
        self.assertEqual(sum(truth.values()), 40)
        self.assertEqual(sum(r["case_type"] in check.VALID_CHANGES for r in self.truth), 16)

    def test_metrics_count_false_alarms_and_reject_missing_or_duplicate_truth(self):
        findings = {"error": ["r"], "clean": ["r"]}
        truth = [{"record_key": "clean", "is_injected_error": False},
                 {"record_key": "error", "is_injected_error": True}]
        metrics = check.detection_metrics(findings, truth)
        self.assertEqual((metrics["true_positives"], metrics["false_positives"]), (1, 1))
        self.assertEqual(metrics["precision"], .5)
        with self.assertRaises(ValueError):
            check.detection_metrics(findings, truth[:1])
        with self.assertRaises(ValueError):
            check.detection_metrics(findings, truth + truth[:1])

    def test_missing_dependency_rules_make_the_function_check_fail(self):
        broken = copy.deepcopy(self.rules)
        broken["dependency_eligible"] = [r for r in broken["dependency_eligible"]
                                         if r["metadata"]["target_col"] != "pixelgroesse_cm"]
        result, _, _ = check.assess(self.clean, self.mixed, self.truth, broken, self.reports, self.fw)
        self.assertFalse(result["function_check_passed"])
        self.assertEqual(result["metrics"]["combined"]["mixed"]["false_negatives"], 8)
        self.assertFalse(next(r for r in result["relations"] if r["target"] == "pixelgroesse_cm")["recovered"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args, rest = parser.parse_known_args()
    PROJECT = args.project.resolve()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
