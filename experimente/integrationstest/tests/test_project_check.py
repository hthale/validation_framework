"""Gezielte Absicherung des Testläufers; die echten Programmstufen prüft der Hauptlauf."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_project_check as check

PROJECT = Path.cwd()


class ProjectCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check.prepare_imports(PROJECT)

    def test_fixture_has_known_relations_and_uninformative_control(self):
        from dop_observations import dop_name_observations
        from record_identity import with_record_key
        from phase2_ml import get_phase2_feature_candidates

        data = check.create_input()
        self.assertEqual(len(data), 320)
        identified = with_record_key(data)
        self.assertTrue(identified.record_key.is_unique)
        self.assertEqual(dop_name_observations(identified).xml_name.tolist(), ["synthetic_0000_0.xml"])
        features = get_phase2_feature_candidates(identified, check.CONTROL)
        labels_by_input = identified.groupby(features, dropna=False)[check.CONTROL].agg(list)
        self.assertEqual(len(labels_by_input), 160)
        self.assertTrue(all(sorted(labels) == ["K_A", "K_B"] for labels in labels_by_input))
        for row in data.to_dict("records"):
            for name, value in check.expected_values(row["spektralkanaele"], row["detailstufe"]).items():
                self.assertEqual(row[name], value)

    def test_no_rules_cannot_pass_the_reference_check(self):
        results = check.check_generated_relations(check.create_input(), [])
        self.assertEqual(len(results), 3)
        self.assertTrue(all(not row["recovered"] for row in results))

    def test_unexpected_crash_does_not_count_as_expected_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            command = [sys.executable, "-c", "print('different failure'); raise SystemExit(2)"]
            with self.assertRaises(RuntimeError):
                check.execute(folder, "wrong_reason", command, folder, expected_failure="required rejection")
            accepted = [sys.executable, "-c", "print('required rejection'); raise SystemExit(2)"]
            self.assertTrue(check.execute(folder, "correct_reason", accepted, folder, expected_failure="required rejection")["passed"])
            zero_exit = [sys.executable, "-c", "print('required rejection')"]
            with self.assertRaises(RuntimeError):
                check.execute(folder, "wrong_exit", zero_exit, folder, expected_failure="required rejection")

    def test_existing_ids_survive_attribute_change(self):
        from record_identity import with_record_key
        data = with_record_key(check.create_input())
        keys = data.record_key.copy()
        data.loc[0, "bodenpixelgroesse"] = 999
        data.loc[0, "xml_name"] = "changed.xml"
        pd.testing.assert_series_equal(keys, with_record_key(data).record_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args, rest = parser.parse_known_args()
    PROJECT = args.project.resolve()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
