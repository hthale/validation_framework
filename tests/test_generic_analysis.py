from pathlib import Path
import json
import re
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generic_schema import attach_identity, infer_columns, normalize_columns, tabular_preparation, choose_groups
from generic_rules import learn_dependencies, learn_profiles
from generic_report import build_cases, render_report, original_value
from phase2_settings import Phase2Settings
from phase2_split import make_grouped_split
from rule_engine import evaluate_rule_on_dataframe
from latentspace.anomaly_scores import rank_by_lof


def unknown_product():
    group = np.repeat(np.arange(160), 2)
    first = group % 2
    second = (group // 2) % 2
    return pd.DataFrame({
        "record_key": [f"r{i:04d}" for i in range(len(group))],
        "entity_code": [f"E-{g}" for g in group], "lokal_id": np.arange(len(group)),
        "modus": np.array(["M", "N"])[first], "variante": np.array(["grob", "fein"])[second],
        "anzahl": np.where(first, 5, 2), "abstand": np.where(second, 4, 12),
        "ablage": [f"P-{a}-{b}" for a, b in zip(first, second)],
        "kontrolle": np.tile(["K_A", "K_B"], 160), "konstante": "META",
    })


def protocol():
    return Phase2Settings(group_columns=("__generic_group",), max_depths=(4,), min_samples_leaves=(5,))


class GenericSchemaTests(unittest.TestCase):
    def test_new_identity_is_persisted_not_attribute_hash(self):
        source = pd.DataFrame({"wert": [1, 2, 3]})
        a, info = attach_identity(source)
        b, _ = attach_identity(source)
        self.assertEqual(info["method"], "persisted_import_uuid")
        self.assertTrue(set(a.record_key).isdisjoint(b.record_key))
        changed = a.iloc[::-1].copy()
        changed["wert"] += 100
        again, _ = attach_identity(changed)
        self.assertEqual(again.record_key.tolist(), changed.record_key.tolist())
        assert_frame_equal(source, pd.DataFrame({"wert": [1, 2, 3]}))

    def test_explicit_identity_and_duplicates(self):
        raw = pd.DataFrame({"code": ["a", "b"], "wert": [1, 2]})
        a, _ = attach_identity(raw, ["code"])
        b, _ = attach_identity(raw.assign(wert=9), ["code"])
        self.assertEqual(a.record_key.tolist(), b.record_key.tolist())
        for keys in (["a", "a"], ["a", ""]):
            with self.assertRaises(ValueError):
                attach_identity(raw.assign(record_key=keys))

    def test_artifact_columns_cannot_be_reused_as_input(self):
        for col in ("lof_raw", "cluster_label", "latent_00", "__generic_date_0", "stratum_key"):
            with self.assertRaises(ValueError):
                attach_identity(pd.DataFrame({col: [1, 2]}))

    def test_roles_normalization_and_original_preservation(self):
        raw = pd.DataFrame({"datum": ["31.12.2020", "2021-01-01"], "zahl": ["1.5", "2.5"],
                            "code": ["001", "002"], "wert": [np.inf, 2], "leer": [None, None],
                            "some_path": ["a", "b"], "lokal_id": [1, 2], "bin": [b"abc", b"xyz"]})
        original = raw.copy(deep=True)
        work, _ = attach_identity(raw)
        contract = infer_columns(work)
        roles = {k: v["role"] for k,v in contract["columns"].items()}
        self.assertEqual(roles, {"bin": "unsupported", "code": "categorical", "datum": "date", "leer": "empty",
                                 "lokal_id": "technical", "record_key": "technical", "some_path": "technical", "wert": "numeric", "zahl": "numeric"})
        normalized, issues = normalize_columns(work, contract)
        self.assertEqual(normalized.code.tolist(), ["001", "002"])
        self.assertTrue(normalized.datum.iloc[0].startswith("2020-12-31"))
        self.assertTrue(pd.isna(normalized.wert.iloc[0]))
        self.assertEqual(issues[0]["column"], "wert")
        assert_frame_equal(original, raw)

    def test_late_nested_value_is_excluded(self):
        raw = pd.DataFrame({"nested": [None]*501 + [[1, 2]], "wert": np.arange(502)})
        self.assertEqual(infer_columns(raw)["columns"]["nested"]["role"], "unsupported")

    def test_numeric_only_dates_and_text_only_preparation(self):
        for raw in (pd.DataFrame({"wert": [1., 2., None]}),
                    pd.DataFrame({"tag": ["2020-01-01", "2020-01-02", None]}),
                    pd.DataFrame({"prosa": [f"Beschreibung Nummer {i}" for i in range(30)]})):
            raw, _ = attach_identity(raw)
            contract = infer_columns(raw)
            norm, _ = normalize_columns(raw, contract)
            model, prep, metadata = tabular_preparation(norm, contract)
            self.assertEqual(prep["X_tab"].shape[0], len(raw))
            self.assertTrue(np.isfinite(prep["X_tab"]).all())
            self.assertNotIn("record_key", model)
            if "prosa" in raw:
                self.assertEqual(metadata["derived_features"][0]["transformation"], "string_length")

    def test_constant_population_has_no_silent_fake_features(self):
        raw, _ = attach_identity(pd.DataFrame({"wert": [5]*12}))
        contract = infer_columns(raw)
        norm, _ = normalize_columns(raw, contract)
        with self.assertRaisesRegex(ValueError, "Keine variierenden"):
            tabular_preparation(norm, contract)

    def test_named_groups_and_explicit_group_exclusion(self):
        raw = unknown_product()
        grouped, info = choose_groups(raw)
        self.assertEqual(info["mode"], "heuristic_named_groups")
        self.assertEqual(info["group_count"], 160)
        split = make_grouped_split(grouped, protocol())
        self.assertTrue(split.groupby("group_id").partition.nunique().eq(1).all())
        renamed = raw.rename(columns={"entity_code": "zusammenhang"})
        _, explicit = choose_groups(renamed, ["zusammenhang"])
        self.assertEqual(explicit["mode"], "user_supplied_groups")
        self.assertFalse(explicit["independence_confirmed"])
        self.assertEqual(infer_columns(renamed, excluded=["zusammenhang"])["columns"]["zusammenhang"]["role"], "technical")

    def test_fallback_holds_duplicates_and_distinguishes_value_types(self):
        raw = pd.DataFrame({"record_key": ["a", "b", "c"], "wert": [1, 1, "1"]})
        grouped, info = choose_groups(raw)
        self.assertEqual(info["mode"], "exploratory_rows_and_exact_duplicates")
        self.assertEqual(grouped["__generic_group"].iloc[0], grouped["__generic_group"].iloc[1])
        self.assertNotEqual(grouped["__generic_group"].iloc[0], grouped["__generic_group"].iloc[2])

    def test_ambiguous_hierarchies_are_not_claimed_independent(self):
        raw = unknown_product().assign(parent_code=np.repeat(np.arange(80), 4))
        _, info = choose_groups(raw)
        self.assertEqual(info["mode"], "exploratory_rows_and_exact_duplicates")
        self.assertFalse(info["independence_confirmed"])


class GenericRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = unknown_product()
        cls.raw["zweites_modusfeld"] = cls.raw.modus
        cls.grouped, cls.grouping = choose_groups(cls.raw)
        split = make_grouped_split(cls.grouped, protocol())
        later = set(split.loc[split.partition.ne("train"), "record_key"])
        cls.grouped["nur_spaeter"] = ["neuer wert" if k in later else None for k in cls.grouped.record_key]
        cls.result = learn_dependencies(cls.grouped, protocol(), cls.grouping)

    def test_three_relations_found_by_real_tree_code(self):
        selected = self.result["rules"]
        for target in ("anzahl", "abstand", "ablage"):
            rules = [r for r in selected if r["metadata"]["target_col"] == target]
            self.assertTrue(rules, target)
            clean = self.result["evaluation_frame"].copy()
            changed = clean.copy()
            if target == "anzahl": changed[target] = np.where(changed[target].eq(2), 5, 2)
            elif target == "abstand": changed[target] = np.where(changed[target].eq(4), 12, 4)
            else: changed[target] = "WRONG"
            covered = pd.Series(False, index=clean.index)
            detected = covered.copy()
            for rule in rules:
                _, _, false_alarm = evaluate_rule_on_dataframe(clean, rule)
                applies, _, invalid = evaluate_rule_on_dataframe(changed, rule)
                self.assertFalse(false_alarm.any())
                covered |= applies
                detected |= invalid
            self.assertTrue(covered.all(), target)
            self.assertTrue(detected.all(), target)

    def test_control_rejected_and_no_technical_predictors(self):
        self.assertFalse(any(r["metadata"]["target_col"] == "kontrolle" for r in self.result["rules"]))
        for report in self.result["reports"]:
            self.assertFalse(set(report["feature_cols_used"]) & {"entity_code", "lokal_id", "record_key", "__generic_group"})
            self.assertFalse(report["test_evaluated"])

    def test_training_only_schema_and_duplicate_target_exclusion(self):
        self.assertEqual(self.result["contract"]["scope"], "training_partition_only")
        self.assertEqual(self.result["contract"]["columns"]["nur_spaeter"]["role"], "empty")
        self.assertNotIn("nur_spaeter", self.result["summary"]["selected_targets"])
        report = next(r for r in self.result["reports"] if r["target_col"] == "modus")
        self.assertIn("zweites_modusfeld", report["duplicate_target_columns_excluded"])
        self.assertNotIn("zweites_modusfeld", report["feature_cols_used"])

    def test_small_number_of_groups_skips_dependencies_explicitly(self):
        grouped, info = choose_groups(self.raw.assign(wenige=1), ["wenige"])
        result = learn_dependencies(grouped, protocol(), info, excluded=["wenige"])
        self.assertEqual(result["summary"]["status"], "skipped")
        self.assertIsNone(result["split"])
        self.assertEqual(result["rules"], [])

    def test_profiles_are_observations_and_html_values_are_escaped(self):
        contract = infer_columns(self.raw)
        norm, _ = normalize_columns(self.raw, contract)
        latent = rank_by_lof(pd.DataFrame({"record_key": self.raw.record_key, "lof_raw": np.linspace(1, 2, len(self.raw)), "cluster_label": 0}))
        profiles, _ = learn_profiles(norm, contract, latent)
        self.assertTrue(profiles)
        self.assertTrue(all(r["metadata"]["approved"] is False for r in profiles))
        raw = self.raw.copy()
        raw.loc[0, "konstante"] = "</script><script>alert(1)</script>"
        cases, _, issues = build_cases(raw, norm, latent, profiles, self.result)
        self.assertFalse(issues)
        self.assertTrue(all(c["decision"] == "unreviewed" for c in cases))
        self.assertTrue(all(not c["findings"] or c["partition"] == "validation" or all(f["family"] == "profiling" for f in c["findings"]) for c in cases))
        summary = {"row_count": len(raw), "eligible_ml_rule_count": len(self.result["rules"]), "scope_note": "Test", "grouping": self.grouping}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bericht.html"
            render_report(path, summary, contract, self.result["candidates"], profiles, [], cases, [])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", text)
            payload = json.loads(re.search(r'id="analysis-data">(.*?)</script>', text).group(1))
            self.assertEqual(payload["case_count"], len(raw))


class GenericTrainingTests(unittest.TestCase):
    def test_extra_parquet_scalar_types_remain_readable(self):
        from decimal import Decimal
        from datetime import time
        self.assertEqual(original_value(Decimal("123.00100"))["value"], "123.00100")
        self.assertEqual(original_value(time(12, 34))["value"], "12:34:00")

    def test_prepared_matrix_wrong_length_is_rejected(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "latentspace"))
        from latentspace.main_v2 import train_model
        from latentspace.config_latent import LatentV2Config
        with self.assertRaisesRegex(ValueError, "Merkmalsmatrix"):
            train_model(pd.DataFrame({"wert": [1., 2.]}), LatentV2Config(), prepared_tabular={"X_tab": np.ones((3, 1))})

    def test_all_padding_text_batch_has_finite_loss_and_gradient(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "latentspace"))
        import torch
        from latentspace.main_v2 import compute_string_reconstruction_loss
        logits = torch.randn(2, 3, 5, requires_grad=True)
        loss = compute_string_reconstruction_loss([logits], [torch.zeros((2, 3), dtype=torch.long)])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
