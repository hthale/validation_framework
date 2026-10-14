"""Relevante Grenzen des neuen Versuchs, ohne Änderungen an Projekttests."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_joint_dop_check import bootstrap, analyze, parser
PROJECT = Path(os.environ.get("FRAMEWORK_PROJECT", PACKAGE.parents[1])).resolve()
bootstrap(PROJECT)
from joint_protocol import GROUP_COLUMNS, KNOWN, OPERATORS, inject_cases, known_results, joint_metrics
from make_control_fixture import make_fixture
from generic_schema import infer_columns, normalize_columns, tabular_preparation
from latent_comparison import transform_frozen, novel_lof


class MutationTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_fixture()
        self.train, self.test = self.frame.iloc[:240].copy(), self.frame.iloc[240:].copy()

    def test_exact_single_cell_changes_and_source_unchanged(self):
        original = self.test.copy(deep=True)
        before, after, changes = inject_cases(self.test, self.train, 3, 42)
        pd.testing.assert_frame_equal(self.test, original)
        delta = ~(before.eq(after) | (before.isna() & after.isna()))
        self.assertEqual(int(delta.to_numpy().sum()), 24)
        self.assertTrue(delta.sum(axis=1).le(1).all())
        self.assertEqual({op["name"]: 3 for op in OPERATORS}, pd.Series([c["operator"] for c in changes]).value_counts().to_dict())
        pre, post = (known_results(f).set_index(["record_key", "rule_id"]) for f in (before, after))
        for change in changes:
            if change["test_reference_rule"]:
                pos = (change["record_key"], change["test_reference_rule"])
                self.assertFalse(pre.at[pos, "violation"])
                self.assertTrue(post.at[pos, "violation"])

    def test_reordering_does_not_choose_other_cases(self):
        b1, a1, c1 = inject_cases(self.test, self.train, 3, 43)
        b2, a2, c2 = inject_cases(self.test.sample(frac=1, random_state=9), self.train, 3, 43)
        pd.testing.assert_frame_equal(b1, b2)
        pd.testing.assert_frame_equal(a1, a2)
        self.assertEqual(c1, c2)

    def test_no_silent_skipping_of_an_unavailable_operator(self):
        broken = self.test.assign(spektralkanaele="unknown")
        with self.assertRaisesRegex(ValueError, "band_count"):
            inject_cases(broken, self.train, 3, 42)
        with self.assertRaises(ValueError):
            inject_cases(self.test.iloc[:24], self.train, 3, 42)
        with self.assertRaises(ValueError):
            inject_cases(self.test, self.train.assign(xml_size_bytes=1), 3, 42)

    def test_unknown_reference_is_not_applicable_and_missing_target_violates(self):
        frame = self.test.iloc[:3].copy().reset_index(drop=True)
        frame.loc[0, "spektralkanaele"] = "unknown"
        frame.loc[1, "tif_band_count"] = np.nan
        frame.loc[2, "bodenpixelgroesse"] = np.nan
        results = known_results(frame).set_index(["record_key", "rule_id"])
        self.assertFalse(results.at[(frame.at[0, "record_key"], KNOWN[0]["rule_id"]), "applicable"])
        self.assertTrue(results.at[(frame.at[1, "record_key"], KNOWN[0]["rule_id"]), "violation"])
        self.assertFalse(results.at[(frame.at[2, "record_key"], KNOWN[1]["rule_id"]), "applicable"])

    def test_constant_changes_invisible_to_current_lof_are_retained(self):
        train = self.train.assign(spektralkanaele="RGB", tif_band_count=3, bodenpixelgroesse=20,
                                  tif_pixel_size_x=.2, tif_pixel_size_y=-.2)
        test = self.test.assign(spektralkanaele="RGB", tif_band_count=3, bodenpixelgroesse=20,
                                tif_pixel_size_x=.2, tif_pixel_size_y=-.2)
        before, after, changes = inject_cases(test, train, 3, 42)
        contract = infer_columns(train, excluded=GROUP_COLUMNS)
        train_norm, _ = normalize_columns(train, contract)
        _, prep, meta = tabular_preparation(train_norm, contract)
        b, _ = normalize_columns(before, contract)
        a, _ = normalize_columns(after, contract)
        _, bt = transform_frozen(b, prep, meta)
        _, at = transform_frozen(a, prep, meta)
        keys = {c["record_key"] for c in changes if c["operator"] in {"band_count", "pixel_size", "crs", "raster_width"}}
        invisible = before.record_key.isin(keys).to_numpy()
        self.assertEqual(int(invisible.sum()), 12)
        np.testing.assert_array_equal(bt[invisible], at[invisible])
        scores_b, scores_a, _, _ = novel_lof(prep["X_tab"], bt, at, 20)
        np.testing.assert_allclose(scores_b[invisible], scores_a[invisible], rtol=0, atol=1e-9)

    def test_changed_test_does_not_change_training_schema(self):
        before = infer_columns(self.train, excluded=GROUP_COLUMNS)
        _, altered, _ = inject_cases(self.test, self.train, 3, 44)
        normalize_columns(altered, before)
        self.assertEqual(before, infer_columns(self.train, excluded=GROUP_COLUMNS))

    def test_invalid_input_creates_no_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "duplicate.parquet"
            data = self.frame.copy()
            data.loc[1, "record_key"] = data.loc[0, "record_key"]
            data.to_parquet(source, index=False)
            output = root / "runs"
            args = parser().parse_args(["--project", str(PROJECT), "--input", str(source), "--output", str(output), "--seeds", "42", "--epochs", "1"])
            with self.assertRaises(ValueError): analyze(args)
            self.assertFalse(output.exists())


class IntegrationMetricsTests(unittest.TestCase):
    def test_overlap_and_new_findings_are_not_double_counted(self):
        cases = pd.DataFrame({"changed": [True, True, False, True], "operator": ["crs", "crs", "unchanged", "camera_swap"]})
        for name in ("known", "profiling", "dependency", "baseline", "latent", "rules", "rules_baseline", "rules_latent", "all"):
            cases[name+"_before"] = False
            cases[name+"_after"] = False
        cases.loc[[0], "known_after"] = True
        cases.loc[[0], "profiling_after"] = True
        cases.loc[[0, 2], "baseline_after"] = True
        cases.loc[[1, 2, 3], "latent_after"] = True
        cases.loc[[3], "latent_before"] = True
        for moment in ("before", "after"):
            cases["rules_"+moment] = cases[[m+"_"+moment for m in ("known", "profiling", "dependency")]].any(axis=1)
            cases["rules_baseline_"+moment] = cases["rules_"+moment] | cases["baseline_"+moment]
            cases["rules_latent_"+moment] = cases["rules_"+moment] | cases["latent_"+moment]
            cases["all_"+moment] = cases["rules_baseline_"+moment] | cases["latent_"+moment]
        result = joint_metrics(cases)
        self.assertEqual(result["methods"]["rules"]["changed_selected_after"], 1)
        self.assertEqual(result["methods"]["all"]["changed_selected_after"], 3)
        self.assertEqual(result["methods"]["all"]["selected_after"], 4)
        extra = result["incremental"]["latent_beyond_rules_baseline"]
        self.assertEqual(extra["additional_changed_rows"], 2)
        self.assertEqual(extra["newly_selected_additional_changed"], 1)
        self.assertEqual(extra["additional_review_rows"], 2)


if __name__ == "__main__": unittest.main()
