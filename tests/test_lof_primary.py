"""Regressionstests für Score-Semantik, Migration und Pipeline-Schnittstellen."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentspace.anomaly_scores import (
    SCORE_COLUMNS, build_outlier_scores, compute_lof_raw, lof_metadata, rank_by_lof, robust_zscore,
)
from latentspace.evaluate_injected_errors import evaluate_injections
from latentspace.phase0_integration import add_phase0_stratification_columns
from latentspace.rerank_lof import convert_file, rerank_existing_results
from latentspace.result_exports import build_neighbor_preview
from phase0_join import merge_phase0_frames
from phase2_ml import get_phase2_feature_candidates, get_phase2_target_candidates, get_phase2_cluster_feature_candidates


def old_result() -> pd.DataFrame:
    """Unabhängig über sklearn berechneter kleiner Export im bisherigen Format."""
    Z = np.random.default_rng(8).normal(size=(30, 4)).astype(np.float32)
    Z[-1] = 25
    jitter = np.random.default_rng(42).normal(0, 1e-6, size=Z.shape).astype(np.float32)
    estimator = LocalOutlierFactor(n_neighbors=5, metric="euclidean", novelty=False).fit(Z + jitter)
    raw = -estimator.negative_outlier_factor_
    frame = pd.DataFrame(Z, columns=[f"latent_{i:02d}" for i in range(4)])
    frame["row_id"] = np.arange(len(Z))
    frame["record_key"] = [f"rk:{i:03d}" for i in range(len(Z))]
    frame["lof_z"] = robust_zscore(raw)
    frame["cluster_label"] = [0] * 29 + [-1]
    frame["centroid_distance_z"] = 0.5
    frame["reconstruction_error"] = np.arange(len(Z)) / 30
    frame["combined_outlier_score"] = np.linspace(1, 0, len(Z))
    frame["outlier_rank"] = np.arange(1, len(Z) + 1)
    frame["source_attribute"] = ["unchanged"] * 29 + [None]
    return frame


class ScoreTests(unittest.TestCase):
    def test_isolated_point_receives_highest_lof(self):
        data = old_result()
        raw = compute_lof_raw(data[[f"latent_{i:02d}" for i in range(4)]].to_numpy(), 5, 42)
        self.assertEqual(int(np.argmax(raw)), 29)
        self.assertGreater(raw[29], 10)
        np.testing.assert_allclose(robust_zscore(raw), data["lof_z"], rtol=0, atol=0)

    def test_diagnostics_do_not_change_ranking(self):
        scores = build_outlier_scores([1000, 0], [1000, 0], [1, 4], [1, 0], [999, 0])
        scores["record_key"] = ["a", "b"]
        ranked = rank_by_lof(scores)
        self.assertEqual(ranked["record_key"].tolist(), ["b", "a"])
        self.assertNotIn("combined_outlier_score", scores)
        self.assertEqual(ranked["primary_score_name"].unique().tolist(), ["lof_raw"])

    def test_ties_are_reproducible_and_not_all_review_priority(self):
        frame = pd.DataFrame({"record_key": ["c", "a", "b"], "lof_raw": [1, 1, 1]})
        for shuffled in (frame, frame.sample(frac=1, random_state=2)):
            ranked = rank_by_lof(shuffled)
            self.assertEqual(ranked["record_key"].tolist(), ["a", "b", "c"])
            self.assertEqual(ranked["lof_z"].tolist(), [0, 0, 0])
            strata = add_phase0_stratification_columns(shuffled)
            self.assertEqual(strata.loc[strata["stratum_outlier"].eq("review_priority"), "record_key"].tolist(), ["a"])

    def test_invalid_latents_rejected(self):
        for Z in (np.empty((0, 2)), np.ones((1, 2)), np.ones((3, 0)), np.array([[1], [np.nan]]), np.array([[1], [np.inf]])):
            with self.subTest(Z=Z), self.assertRaises(ValueError):
                compute_lof_raw(Z, 5, 42)

    def test_invalid_scores_and_ambiguous_schema_rejected(self):
        base = pd.DataFrame({"record_key": ["a", "b"], "lof_raw": [1.0, 2.0]})
        for invalid in (np.nan, np.inf, -1, 0):
            frame = base.copy()
            frame.loc[0, "lof_raw"] = invalid
            with self.subTest(score=invalid), self.assertRaises(ValueError):
                rank_by_lof(frame)
        for name, value in (("combined_outlier_score", 0.5), ("score_schema_version", "future"), ("primary_score_name", "wrong")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                rank_by_lof(base.assign(**{name: value}))

    def test_small_population_records_effective_neighborhood(self):
        raw = compute_lof_raw(np.array([[0, 0], [1, 1]]), 20, 42)
        self.assertTrue(np.isfinite(raw).all())
        self.assertEqual(lof_metadata(2, 20, 42)["lof"]["n_neighbors_effective"], 1)
        with self.assertRaises(ValueError):
            compute_lof_raw(np.ones((3, 2)), 0, 42)


class MigrationTests(unittest.TestCase):
    def test_row_order_restored_and_original_values_preserved(self):
        original = old_result().sample(frac=1, random_state=1)
        before = original.copy(deep=True)
        ranked, info = rerank_existing_results(original, 5, 42)
        self.assertTrue(info["source_lof_z_matches"])
        self.assertEqual(info["source_lof_z_max_abs_difference"], 0)
        self.assertFalse(info["training_performed"])
        self.assertEqual(int(ranked.iloc[0]["row_id"]), 29)
        columns = [c for c in original if c not in {"lof_z", "outlier_rank", "combined_outlier_score"}]
        pd.testing.assert_frame_equal(original[columns].sort_values("record_key").reset_index(drop=True), ranked[columns].sort_values("record_key").reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(original, before)
        self.assertIn("legacy_combined_outlier_score", ranked)
        self.assertNotIn("combined_outlier_score", ranked)
        self.assertTrue(bool(ranked.iloc[0]["centroid_distance_is_fallback"]))

    def test_reordered_input_produces_same_ranking(self):
        a, _ = rerank_existing_results(old_result(), 5, 42)
        b, _ = rerank_existing_results(old_result().sample(frac=1, random_state=7), 5, 42)
        pd.testing.assert_frame_equal(a, b)
        as_text = old_result().assign(row_id=lambda frame: frame["row_id"].astype(str))
        c, _ = rerank_existing_results(as_text, 5, 42)
        pd.testing.assert_frame_equal(a, c)

    def test_different_neighbors_require_explicit_change_flag(self):
        with self.assertRaisesRegex(ValueError, "LOF-Reproduktion"):
            rerank_existing_results(old_result(), 10, 42)
        _, info = rerank_existing_results(old_result(), 10, 42, allow_lof_change=True)
        self.assertFalse(info["source_lof_z_matches"])
        self.assertTrue(info["allow_lof_change"])

    def test_incomplete_latents_or_positions_rejected(self):
        data = old_result()
        variants = [data.drop(columns="latent_01"), data.drop(columns="row_id"), data.iloc[1:].copy()]
        duplicate = data.copy()
        duplicate.loc[0, "row_id"] = 1
        variants.append(duplicate)
        for frame in variants:
            with self.subTest(columns=list(frame)), self.assertRaises(ValueError):
                rerank_existing_results(frame, 5, 42)

    def test_duplicate_keys_and_repeated_migration_rejected(self):
        data = old_result()
        data.loc[0, "record_key"] = data.loc[1, "record_key"]
        with self.assertRaises(ValueError):
            rerank_existing_results(data, 5, 42)
        ranked, _ = rerank_existing_results(old_result(), 5, 42)
        with self.assertRaisesRegex(ValueError, "bereits"):
            rerank_existing_results(ranked, 5, 42)

    def test_file_conversion_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source.parquet", root / "converted"
            old_result().to_parquet(source, index=False)
            before = source.read_bytes()
            ranked, _ = convert_file(source, target, 5, 42)
            self.assertEqual(before, source.read_bytes())
            pd.testing.assert_frame_equal(ranked, pd.read_parquet(target / "latent_outlier_results_lof.parquet"))
            self.assertEqual(json.loads((target / "rerank_info.json").read_text())["row_count"], 30)
            with self.assertRaises(FileExistsError):
                convert_file(source, target, 5, 42)
            self.assertEqual(before, source.read_bytes())

    def test_mismatch_creates_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source.parquet", root / "converted"
            old_result().to_parquet(source, index=False)
            with self.assertRaises(ValueError):
                convert_file(source, target, 10, 42)
            self.assertFalse(target.exists())


class PipelineTests(unittest.TestCase):
    def test_join_keeps_left_order_and_score_identity(self):
        results, _ = rerank_existing_results(old_result(), 5, 42)
        left = old_result()[["record_key", "source_attribute"]].sample(frac=1, random_state=3)
        left.index = [3] * len(left)
        joined = merge_phase0_frames(left, results)
        self.assertEqual(joined.index.tolist(), left.index.tolist())
        self.assertEqual(joined["record_key"].tolist(), left["record_key"].tolist())
        np.testing.assert_array_equal(joined["lof_raw"], results.set_index("record_key").loc[left["record_key"], "lof_raw"])
        self.assertNotIn("legacy_combined_outlier_score", joined)

    def test_legacy_join_and_legacy_stratification_rejected(self):
        old = old_result()
        with self.assertRaisesRegex(ValueError, "lof_raw fehlt"):
            merge_phase0_frames(old[["record_key"]], old)
        with self.assertRaises(ValueError):
            add_phase0_stratification_columns(old)

    def test_review_budget_preserves_order_and_is_not_error_label(self):
        ranked, _ = rerank_existing_results(old_result(), 5, 42)
        frame = ranked.sample(frac=1, random_state=3)
        strata = add_phase0_stratification_columns(frame)
        self.assertEqual(strata["record_key"].tolist(), frame["record_key"].tolist())
        self.assertEqual(int(strata["stratum_outlier"].eq("review_priority").sum()), 2)
        self.assertEqual(set(strata.loc[strata["stratum_outlier"].eq("review_priority"), "record_key"]), set(ranked.head(2)["record_key"]))
        self.assertEqual(set(strata["stratum_outlier"]), {"review_priority", "other"})
        self.assertEqual(int(add_phase0_stratification_columns(frame, 0)["stratum_outlier"].eq("review_priority").sum()), 0)

    def test_score_columns_never_become_ml_features_or_targets(self):
        frame = pd.DataFrame({c: [0, 1] * 30 for c in SCORE_COLUMNS})
        frame["legacy_unknown_score"] = [0, 1] * 30
        frame["latent_00"] = [0, 1] * 30
        frame["original_feature"] = [0, 1] * 30
        self.assertEqual(get_phase2_target_candidates(frame), ["original_feature"])
        self.assertEqual(get_phase2_feature_candidates(frame, "target", forced_feature_columns=list(frame)), ["original_feature"])
        self.assertEqual(get_phase2_cluster_feature_candidates(frame), ["original_feature"])

    def test_neighbors_belong_to_top_lof_queries_and_have_stable_keys(self):
        ranked, _ = rerank_existing_results(old_result(), 5, 42)
        preview = build_neighbor_preview(ranked, top_k=3, k=5)
        self.assertEqual([q["record_key"] for q in preview["queries"]], ranked.head(3)["record_key"].tolist())
        known = set(ranked["record_key"])
        for query in preview["queries"]:
            self.assertEqual(len(query["neighbors"]), 5)
            for neighbor in query["neighbors"]:
                self.assertIn(neighbor["record_key"], known)
                self.assertNotEqual(neighbor["record_key"], query["record_key"])
                self.assertGreaterEqual(neighbor["distance"], 0)
        json.dumps(preview, allow_nan=False)

    def test_injection_evaluation_uses_lof_and_reports_budget(self):
        ranked, _ = rerank_existing_results(old_result(), 5, 42)
        injected = pd.DataFrame({"record_key_after": [ranked.iloc[0]["record_key"], ranked.iloc[-1]["record_key"]], "error_type": ["synthetic", "synthetic"]})
        _, summary = evaluate_injections(injected, ranked)
        self.assertEqual(summary["top10_hits"], 1)
        self.assertEqual(summary["top10_entry_recall"], 0.5)
        self.assertEqual(summary["top500_effective_budget"], 30)
        self.assertEqual(summary["matched_entries"], 2)

    def test_unmatched_injection_is_explicit_error(self):
        ranked, _ = rerank_existing_results(old_result(), 5, 42)
        with self.assertRaisesRegex(ValueError, "nicht zugeordnet"):
            evaluate_injections(pd.DataFrame({"record_key_after": ["missing"], "error_type": ["test"]}), ranked)


if __name__ == "__main__":
    unittest.main()
