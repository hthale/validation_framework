from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_review import build, choose_phase12_run, REQUIRED_FILES, verify_phase2_provenance
from phase2_settings import Phase2Settings
from phase2_split import make_grouped_split, split_summary
from result_integration import integrate_results
from review_common import digest, file_sha256, read_json, verify_bundle, write_json
from review_decisions import initial_history, apply_decisions, replay_history
from review_output import build_html, csv_text, write_views


def inputs():
    raw = pd.DataFrame({"record_key": [f"key-{i:03d}" for i in range(80)], "tile": np.repeat(np.arange(40), 2),
                        "x": np.ones(80), "target": ["actual"] * 80, "kachelname": [f"tile-{i}" for i in range(80)]})
    phase0 = pd.DataFrame({"record_key": raw.record_key, "lof_raw": np.linspace(1, 2, 80), "cluster_label": [0] * 79 + [-1]})
    settings = Phase2Settings(group_columns=("tile",))
    split = make_grouped_split(raw, settings)
    schema = {"x": "numeric", "target": "categorical", "kachelname": "text"}
    rules = {k: [] for k in ["domain", "profiling", "dependency_candidates", "dependency_eligible", "cluster"]}
    observations = pd.DataFrame(columns=["record_key", "observation"])
    return raw, phase0, split, settings, schema, rules, observations


def basic_rule():
    return {"rule_id": "x_constant", "rule_name": "x must be zero", "rule_type": "constant",
            "columns": ["x"], "expression": "x = 0", "source": "profiling", "status": "proposed",
            "metadata": {"value": 0}, "evidence_type": "profiling_runs"}


def dependency_rule(eligible=True):
    return {"rule_id": "ml_target", "rule_name": "target dependency", "rule_type": "ml_dependency",
            "columns": ["x", "target"], "expression": "IF x > 0 THEN target = expected", "source": "ml_phase2", "status": "proposed",
            "metadata": {"target_col": "target", "predicted_value": "expected", "conditions": [{"feature": "x", "operator": ">", "value": 0}],
                         "eligible_for_review": eligible, "model_gate_passed": eligible, "evaluation_scope": "validation_partition", "test_evaluated": False}}


def make_bundle(parts=None):
    return integrate_results(*(parts or inputs()), provenance={"fixture": True}, created_at="2026-09-22T00:00:00+00:00")


class ReviewIntegrationTests(unittest.TestCase):
    def test_full_population_and_all_violations_are_preserved_beyond_example_limit(self):
        parts = inputs()
        parts[5]["profiling"].append(basic_rule())
        bundle = make_bundle(parts)
        self.assertEqual(len(bundle["cases"]), 80)
        self.assertEqual(bundle["summary"]["profiling_deviating_records"], 80)
        self.assertTrue(all(row["profiling_deviation_count"] == 1 for row in bundle["cases"]))
        self.assertEqual(sum(len(r["findings"]) for r in bundle["cases"]), 80)
        self.assertEqual(bundle["cases"][0]["record_key"], "key-079")
        self.assertEqual(sum(r["lof_review_selected"] for r in bundle["cases"]), 4)
        self.assertEqual(replay_history(bundle, initial_history(bundle)), {})

    def test_shuffling_inputs_keeps_identity_scores_and_findings(self):
        parts = inputs()
        parts[5]["profiling"].append(basic_rule())
        original = parts[0].copy(deep=True)
        first = make_bundle(parts)
        other = list(parts)
        other[0] = parts[0].sample(frac=1, random_state=9)
        other[1] = parts[1].sample(frac=1, random_state=7)
        second = make_bundle(other)
        self.assertEqual(first["cases"], second["cases"])
        pd.testing.assert_frame_equal(parts[0], original)

    def test_missing_or_duplicate_phase0_keys_are_rejected(self):
        parts = list(inputs())
        parts[1] = parts[1].iloc[:-1]
        with self.assertRaises(ValueError): make_bundle(parts)
        parts = list(inputs())
        parts[1] = pd.concat([parts[1], parts[1].iloc[[0]]])
        with self.assertRaises(ValueError): make_bundle(parts)

    def test_ml_candidates_apply_only_to_validation_and_rejected_ones_never_apply(self):
        parts = inputs()
        rule = dependency_rule()
        rejected = dependency_rule(False)
        rejected["rule_id"] = "rejected"
        parts[5]["dependency_candidates"] += [rule, rejected]
        parts[5]["dependency_eligible"].append(rule)
        bundle = make_bundle(parts)
        for row in bundle["cases"]:
            if row["partition"] == "validation":
                self.assertEqual(row["dependency_deviation_count"], 1)
            else:
                self.assertIsNone(row["dependency_deviation_count"])
                self.assertFalse(any(f["category"] == "dependency" for f in row["findings"]))
            self.assertFalse(any(f["rule_id"] == "rejected" for f in row["findings"]))
        self.assertFalse(bundle["summary"]["dependency_test_evaluated"])

    def test_false_eligibility_or_different_candidate_catalog_rejected(self):
        parts = inputs()
        rule = dependency_rule(False)
        parts[5]["dependency_candidates"].append(rule)
        parts[5]["dependency_eligible"].append(rule)
        with self.assertRaises(ValueError): make_bundle(parts)
        parts = inputs()
        rule = dependency_rule(True)
        parts[5]["dependency_candidates"].append(rule)
        changed = deepcopy(rule)
        changed["metadata"]["predicted_value"] = "different"
        parts[5]["dependency_eligible"].append(changed)
        with self.assertRaises(ValueError): make_bundle(parts)

    def test_cluster_disagreement_and_naming_are_not_quality_violations(self):
        parts = list(inputs())
        rule = dependency_rule()
        rule.update(rule_id="cluster_description", rule_type="ssl_cluster_rule", source="ssl_cluster", columns=["x", "cluster_label"])
        rule["metadata"].update(target_col="cluster_label", predicted_value=1, evaluation_scope="descriptive_in_sample", included_clusters=[0])
        parts[5]["cluster"].append(rule)
        parts[6] = pd.DataFrame({"record_key": ["key-079"], "observation": ["Names differ"]})
        bundle = make_bundle(parts)
        self.assertEqual(bundle["summary"]["cluster_hint_records"], 79)
        self.assertEqual(bundle["summary"]["cluster_description_agreement_records"], 0)
        self.assertEqual(bundle["summary"]["domain_deviating_records"], 0)
        self.assertEqual(bundle["summary"]["naming_observation_records"], 1)
        self.assertFalse(bundle["cases"][0]["cluster_hints"])
        self.assertEqual(bundle["cases"][0]["findings"][0]["category"], "naming")

    def test_invalid_rules_report_an_issue_instead_of_silently_passing(self):
        parts = inputs()
        rule = basic_rule()
        rule["columns"] = ["missing_feature"]
        parts[5]["profiling"].append(rule)
        bundle = make_bundle(parts)
        self.assertEqual(bundle["summary"]["rule_evaluation_issue_count"], 1)
        self.assertEqual(bundle["rule_evaluation_issues"][0]["evaluation_status"], "invalid_rule")
        self.assertFalse(bundle["rule_assessments"])

    def test_duplicate_rules_and_foreign_observations_rejected(self):
        parts = inputs()
        parts[5]["profiling"] += [basic_rule(), basic_rule()]
        with self.assertRaises(ValueError): make_bundle(parts)
        parts = list(inputs())
        parts[6] = pd.DataFrame({"record_key": ["unknown"], "observation": ["test"]})
        with self.assertRaises(ValueError): make_bundle(parts)

    def test_source_file_mismatch_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            phase12 = root / "phase12"
            phase12.mkdir()
            for name in REQUIRED_FILES:
                (phase12 / name).write_text("{}")
            (root / "input").write_text("input")
            (root / "phase0").write_text("phase0")
            with self.assertRaises(ValueError):
                build(root / "input", root / "phase0", phase12, root / "out")
            self.assertFalse((root / "out").exists())

    def test_mismatched_training_manifest_is_rejected(self):
        parts = inputs()
        split, settings = parts[2], parts[3]
        summary = split_summary(split, settings)
        report = {"source": "ml_phase2", "target_col": "target", "evaluation_scope": "validation_partition",
                  "test_evaluated": False, "train_record_keys_sha256": "wrong", "validation_record_keys_sha256": "wrong"}
        with self.assertRaises(ValueError):
            verify_phase2_provenance({"split": summary}, summary, split, [report], parts[5], settings)

    def test_latest_complete_run_is_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ["20260922_100000", "20260922_110000", "20260922_120000"]:
                (root / name).mkdir()
            for folder in [root / "20260922_100000", root / "20260922_110000"]:
                for name in REQUIRED_FILES: (folder / name).write_text("{}")
            self.assertEqual(choose_phase12_run(root).name, "20260922_110000")


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = make_bundle()
        write_json(self.root / "review_bundle.json", self.bundle)
        write_json(self.root / "review_history.json", initial_history(self.bundle))
        write_views(self.root, self.bundle, initial_history(self.bundle))

    def request(self, key="key-079", decision="confirmed_error", reason="Kontrollierter Testfall.", base=0):
        return {"schema_version": 1, "bundle_id": self.bundle["bundle_id"], "base_revision": base,
                "reviewer": "Testprüfer", "decisions": [{"record_key": key, "decision": decision, "reason": reason}]}

    def test_confirm_and_revise_preserve_evidence_and_prior_decisions(self):
        before = (self.root / "review_bundle.json").read_bytes()
        result = apply_decisions(self.root, self.request())
        self.assertEqual(result["revision"], 1)
        summary = read_json(self.root / "review_summary.json")
        self.assertEqual(summary["review_counts"]["confirmed_error"], 1)
        self.assertFalse(summary["rule_approval_performed"])
        apply_decisions(self.root, self.request(decision="needs_clarification", reason="Neue Unterlage fehlt.", base=1))
        history = read_json(self.root / "review_history.json")
        self.assertEqual(len(history["events"]), 2)
        self.assertEqual(history["events"][0]["decisions"][0]["decision"], "confirmed_error")
        self.assertEqual(replay_history(self.bundle, history)["key-079"]["decision"], "needs_clarification")
        self.assertEqual((self.root / "review_bundle.json").read_bytes(), before)

    def test_stale_revision_and_cross_run_import_do_not_overwrite(self):
        apply_decisions(self.root, self.request())
        before = (self.root / "review_history.json").read_bytes()
        with self.assertRaises(ValueError): apply_decisions(self.root, self.request())
        other = self.request(base=1)
        other["bundle_id"] = "another_run"
        with self.assertRaises(ValueError): apply_decisions(self.root, other)
        self.assertEqual((self.root / "review_history.json").read_bytes(), before)

    def test_all_changes_validated_before_write(self):
        cases = [self.request(key="unknown"), self.request(reason=""), self.request(decision="automatic_pass")]
        duplicate = self.request()
        duplicate["decisions"] *= 2
        cases.append(duplicate)
        bad_extra = self.request()
        bad_extra["decisions"][0]["lof_raw"] = 999
        cases.append(bad_extra)
        no_reviewer = self.request()
        no_reviewer["reviewer"] = ""
        cases.append(no_reviewer)
        mixed = self.request()
        mixed["decisions"].append({"record_key": "unknown", "decision": "plausible", "reason": "reason"})
        cases.append(mixed)
        before = (self.root / "review_history.json").read_bytes()
        for request in cases:
            with self.subTest(request=request):
                with self.assertRaises(ValueError): apply_decisions(self.root, request)
                self.assertEqual((self.root / "review_history.json").read_bytes(), before)
        self.assertFalse((self.root / ".review.lock").exists())

    def test_dry_run_and_unchanged_decisions_do_not_add_events(self):
        before = (self.root / "review_history.json").read_bytes()
        result = apply_decisions(self.root, self.request(), dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual((self.root / "review_history.json").read_bytes(), before)
        apply_decisions(self.root, self.request())
        with self.assertRaises(ValueError): apply_decisions(self.root, self.request(base=1))
        self.assertEqual(read_json(self.root / "review_history.json")["revision"], 1)

    def test_changed_evidence_or_history_is_rejected(self):
        altered = deepcopy(self.bundle)
        altered["cases"][0]["lof_raw"] = 100
        with self.assertRaises(ValueError): verify_bundle(altered)
        apply_decisions(self.root, self.request())
        history = read_json(self.root / "review_history.json")
        history["events"][0]["decisions"][0]["reason"] = "Changed"
        with self.assertRaises(ValueError): replay_history(self.bundle, history)

    def test_existing_writer_lock_is_respected(self):
        lock = self.root / ".review.lock"
        lock.write_text("another writer")
        with self.assertRaises(ValueError): apply_decisions(self.root, self.request())
        self.assertEqual(lock.read_text(), "another writer")

    def test_html_embeds_values_as_data_and_csv_does_not_execute_formulas(self):
        bundle = deepcopy(self.bundle)
        bundle["cases"][0]["display_name"] = '</script><script>alert("test")</script>'
        bundle["bundle_id"] = "review1:" + digest({k: v for k, v in bundle.items() if k != "bundle_id"})
        html = build_html(bundle, initial_history(bundle), {})
        self.assertNotIn('</script><script>alert', html)
        self.assertIn('\\u003c/script\\u003e', html)
        content = csv_text([{"record_key": "safe", "reason": "=1+2"}], ["record_key", "reason"])
        rows = list(csv.DictReader(io.StringIO(content.lstrip('\ufeff')), delimiter=';'))
        self.assertEqual(rows[0]["reason"], "'=1+2")


if __name__ == "__main__": unittest.main()
