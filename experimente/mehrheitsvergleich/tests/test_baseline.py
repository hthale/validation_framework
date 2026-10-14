from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from run_baseline_check import bootstrap, latest_completed
bootstrap(Path(os.environ.get("FRAMEWORK_TEST_PROJECT", PACKAGE.parents[1])))

import numpy as np
import pandas as pd
from comparison_core import Protocol, constant_rule, evaluate_selected, DATE, DATE_MISSING, METHODS
from baseline_core import (BASELINE, majority_model, baseline_cases, classify_rules, original_comparisons,
    make_variants, select_groups, assert_frozen_inputs, pair_results, mutation_metrics)
from verify_rule_comparison import assert_same_cases


def sample():
    return pd.DataFrame({"record_key": [f"r:{i}" for i in range(8)],
                         "x": ["left"]*5+["right"]*3, "y": ["A"]*5+["B"]*2+[None]})


def groups(frame):
    return pd.Series(["g:"+k for k in frame.record_key], index=frame.record_key)


def rule(value, method="global", cluster=0, conditions=None, rid="r"):
    inner = constant_rule("y", value, 10)
    inner["metadata"]["conditions"] = conditions or []
    return {"rule_id": rid, "scope": {"method": method, "cluster": cluster, "target": "y", "seed": 42},
            "inner_rule": inner, "selected_on_validation": True, "expression": "test scoped rule"}


class BaselineTests(unittest.TestCase):
    def test_training_only_mode_tie_break_missing_and_reordering(self):
        train = pd.DataFrame({"record_key": list("abcde"), "y": ["B", "A", "B", "A", None]})
        m = majority_model(train, "y")
        self.assertEqual(m["prediction"], "A")
        self.assertEqual(m["missing_training_rows"], 1)
        self.assertEqual(m, majority_model(train.iloc[::-1], "y"))
        test = pd.DataFrame({"record_key": ["t1", "t2"], "y": ["B", None]})
        c = baseline_cases(test, "y", groups(test), 42, m)
        self.assertEqual(c.prediction.tolist(), ["A", "A"])
        self.assertEqual(c.disagree_observed.tolist(), [True, False])
        self.assertEqual(c.missing_target_covered.tolist(), [False, True])
        self.assertEqual(m["class_counts"], [{"value": "A", "count": 2}, {"value": "B", "count": 2}])

    def test_scope_constant_and_conditional_are_distinct_and_preserved(self):
        rules = [rule("A", rid="1"), rule("A", "latent_clusters", 7, rid="2"),
                 rule("B", conditions=[{"feature": "x", "operator": "==", "value": "right"}], rid="3"),
                 rule("B", "attribute_clusters", 8, [{"feature": "x", "operator": "==", "value": "right"}], "4")]
        saved = copy.deepcopy(rules)
        classified = classify_rules(rules)
        self.assertEqual({r["rule_kind"] for r in classified}, {"global_constant_expectation", "cluster_constant_expectation",
            "global_attribute_dependency", "cluster_attribute_dependency"})
        self.assertEqual(classified[1]["cluster"], 7)
        self.assertEqual(rules, saved)

    def test_real_conditional_prediction_can_beat_majority_on_minority_controls(self):
        df = sample(); group = groups(df); m = majority_model(df, "y")
        all_cases = [baseline_cases(df, "y", group, 42, m)]
        for method in METHODS:
            rules = [rule("A", method, conditions=[{"feature": "x", "operator": "==", "value": "left"}], rid="a"),
                     rule("B", method, conditions=[{"feature": "x", "operator": "==", "value": "right"}], rid="b")]
            c, _ = evaluate_selected(df, np.zeros(len(df), dtype=int), rules, "y", group, method, 42, Protocol())
            all_cases.append(c)
        stats = original_comparisons(pd.concat(all_cases, ignore_index=True), m)
        cur = next(r for r in stats if r["method"] == "latent_clusters" and r["stratum"] == "original_nonmajority")
        self.assertEqual(cur["net_agreement_gain_on_same_covered"], 2)
        self.assertEqual(cur["disagree_observed"], 0)
        base = next(r for r in stats if r["method"] == BASELINE and r["stratum"] == "original_nonmajority")
        self.assertEqual(base["disagree_observed"], 2)

    def test_both_directions_distinguish_new_disagreement_from_removed(self):
        df = sample(); group = groups(df); m = majority_model(df, "y")
        variants, _ = make_variants(df, df, group, 42, "y", m, 50)
        outputs = []
        for scenario, before, after, meta, changes in variants:
            assert_frozen_inputs(before, after, "y", ["x"], [], changes)
            outputs.append(pair_results(baseline_cases(before, "y", group, 42, m),
                baseline_cases(after, "y", group, 42, m), meta))
        stats = {r["scenario"]: r for r in mutation_metrics(pd.concat(outputs, ignore_index=True))}
        self.assertEqual(stats["majority_to_nonmajority"]["new_disagreement"], 5)
        self.assertEqual(stats["nonmajority_to_majority"]["disagreement_removed"], 2)
        self.assertEqual(stats["nonmajority_to_majority"]["new_disagreement"], 0)
        self.assertEqual(stats["present_to_missing"]["both_disagree"], 2)
        self.assertEqual(stats["present_to_missing"]["new_disagreement"], 5)
        self.assertEqual(stats["present_to_missing"]["changed_missing"], 7)

    def test_noise_and_outside_scope_remain_abstentions_not_successes(self):
        df = sample(); group = groups(df)
        selected = [rule("A", "latent_clusters", 2)]
        labels = np.array([2, 2, -1, 3, 3, -1, -1, -1])
        before, _ = evaluate_selected(df, labels, selected, "y", group, "latent_clusters", 42, Protocol())
        after_df = df.copy(); after_df["y"] = "B"
        after, _ = evaluate_selected(after_df, labels, selected, "y", group, "latent_clusters", 42, Protocol())
        meta = pd.DataFrame({"record_key": df.record_key, "group_id": group.values, "seed": 42, "target": "y", "scenario": "test"})
        metric = mutation_metrics(pair_results(before, after, meta))[0]
        self.assertEqual(metric["new_disagreement"], 2)
        self.assertEqual(metric["abstained_pairs"], 6)
        self.assertEqual(metric["new_disagreement_per_all_pairs"], .25)

    def test_one_per_group_selection_is_order_independent_and_limited(self):
        df = sample(); group = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"], index=df.record_key)
        a = select_groups(df, group, 42, "y", "test", 3)
        b = select_groups(df.iloc[::-1], group, 42, "y", "test", 3)
        assert_same_cases(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(a.record_key.map(group).nunique(), 3)

    def test_no_alternative_and_empty_strata_are_explicit(self):
        df = sample().iloc[:4].copy(); m = majority_model(df, "y")
        variants, availability = make_variants(df, df, groups(df), 42, "y", m, 3)
        self.assertEqual([v[0] for v in variants], ["present_to_missing"])
        self.assertEqual(sum("skip_reason" in r for r in availability), 2)
        all_cases = []
        for method in (*METHODS, BASELINE):
            c = baseline_cases(df, "y", groups(df), 42, m); c["method"] = method; all_cases.append(c)
        rows = original_comparisons(pd.concat(all_cases, ignore_index=True), m)
        self.assertTrue(all(r["agreement_all_covered"] is None for r in rows if r["stratum"] == "original_nonmajority"))

    def test_date_family_changed_consistently_and_excluded_from_features(self):
        df = sample().iloc[:7].rename(columns={"y": DATE_MISSING})
        df[DATE_MISSING] = ["vorhanden"]*5+["fehlt"]*2
        df[DATE] = [1., 2., 3., 4., 5., np.nan, np.nan]
        variants, availability = make_variants(df, df, groups(df), 42, DATE_MISSING, majority_model(df, DATE_MISSING), 50)
        for _, before, after, _, changes in variants:
            self.assertEqual(after[DATE].isna().tolist(), after[DATE_MISSING].eq("fehlt").tolist())
            assert_frozen_inputs(before, after, DATE_MISSING, ["x"], [], changes)
            with self.assertRaises(ValueError):
                assert_frozen_inputs(before, after, DATE_MISSING, ["x", DATE], [], changes)
        self.assertIn("derived_presence_indicator", availability[-1]["skip_reason"])

    def test_predictor_or_original_attribute_change_rejected(self):
        df = sample(); changed = df.copy(); changed["y"] = "B"
        with self.assertRaises(ValueError):
            assert_frozen_inputs(df, changed, "y", ["x", "y"], [], ["y"])
        changed.loc[0, "x"] = "corrupted"
        with self.assertRaises(AssertionError):
            assert_frozen_inputs(df, changed, "y", ["x"], [], ["y"])

    def test_prediction_changes_and_unpaired_keys_rejected(self):
        df = sample(); m = majority_model(df, "y"); before = baseline_cases(df, "y", groups(df), 42, m)
        after = before.copy(); after.loc[0, "prediction"] = "B"
        with self.assertRaises(AssertionError):
            pair_results(before, after, pd.DataFrame({"record_key": df.record_key}))
        with self.assertRaises(ValueError):
            pair_results(before, before.iloc[::-1], pd.DataFrame({"record_key": df.record_key}))

    def test_saved_nulls_roundtrip_preserves_pairs(self):
        df = sample(); m = majority_model(df, "y")
        scenario, before, after, meta, _ = make_variants(df, df, groups(df), 42, "y", m, 50)[0][-1]
        pair = pair_results(baseline_cases(before, "y", groups(df), 42, m), baseline_cases(after, "y", groups(df), 42, m), meta)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"pairs.parquet"; pair.to_parquet(path, index=False)
            assert_same_cases(pair, pd.read_parquet(path))

    def test_latest_completed_skips_running_and_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, status in (("01", "completed"), ("02", "failed"), ("03", "running")):
                sub = root/name; sub.mkdir(); (sub/"run_status.json").write_text(json.dumps({"status": status}))
            self.assertEqual(latest_completed(root).name, "01")
            with self.assertRaises(ValueError):
                latest_completed(root/"missing")


if __name__ == "__main__":
    unittest.main()
