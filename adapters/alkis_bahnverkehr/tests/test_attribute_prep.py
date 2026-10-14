from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribute_prep import (prepare_attributes, read_attributes, select_sample, encode_list,
                            encode_code, DATE_MISSING, FORBIDDEN)


def fixture(n=30):
    return pd.DataFrame({
        "gml_id": [f"TEST_{i:04d}" for i in range(n)],
        "funktion": [2321 if i % 2 else 2322 for i in range(n)],
        "bahnkategorie": [["1100"] if i % 3 else None for i in range(n)],
        "ergebnisderueberpruefung": [2000] * n,
        "datumderletztenueberpruefung": [None if i % 7 == 0 else "2023-11-30T10:21:32Z" for i in range(n)],
        "wkb_geometry": [b"DO_NOT_READ_GEOMETRY"] * n,
        "geometry": [b"DO_NOT_READ_GEOMETRY"] * n,
        "objektkoordinaten": ["DO_NOT_READ_COORDINATES"] * n,
        "nummerderbahnstrecke": ["1234"] * n,
        "unexpected_target": list(range(n)),
    })


class AttributeTests(unittest.TestCase):
    def test_reader_never_loads_geometry_identifiers_or_unapproved_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.parquet"
            fixture().to_parquet(path, index=False)
            original_reader = pq.read_table
            with patch("attribute_prep.pq.read_table", wraps=original_reader) as reader:
                frame, info = read_attributes(path)
            self.assertFalse(set(reader.call_args.kwargs["columns"]) & FORBIDDEN)
            self.assertNotIn("unexpected_target", frame)
            self.assertEqual(info["geometry_columns_read"], [])
            self.assertEqual(frame.gml_id.nunique(), 30)

    def test_code_prefix_prevents_numeric_interpretation_and_preserves_zeros(self):
        self.assertEqual(encode_code(2322.0), 'code:"2322"')
        self.assertEqual(encode_code("0012"), 'code:"0012"')
        self.assertNotEqual(encode_code("0012"), encode_code(12))

    def test_lists_preserve_all_values_and_distinguish_missing_from_empty(self):
        self.assertIsNone(encode_list(None))
        self.assertIsNone(encode_list(np.nan))
        self.assertEqual(encode_list([]), "liste:[]")
        self.assertEqual(encode_list(np.array(["1100", "1400", "1100"])),
                         'liste:["1100","1400","1100"]')

    def test_malformed_codes_and_nested_lists_rejected(self):
        for value in (1.2, np.inf, True, b"2322"):
            with self.assertRaises(ValueError):
                encode_code(value)
        for value in ('["1100"]', [["1100"]], [None]):
            with self.assertRaises(ValueError):
                encode_list(value)

    def test_preparation_does_not_mutate_original(self):
        frame = fixture()
        before = frame.copy(deep=True)
        prepared = prepare_attributes(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertFalse(set(prepared) & FORBIDDEN)

    def test_sampling_ignores_row_order_and_attribute_changes(self):
        frame = prepare_attributes(fixture(100))
        sample = select_sample(frame, 20, 42)
        reordered = frame.sample(frac=1, random_state=90).reset_index(drop=True)
        reordered["funktion"] = 'code:"changed"'
        self.assertEqual(sample.gml_id.tolist(), select_sample(reordered, 20, 42).gml_id.tolist())
        self.assertNotEqual(sample.gml_id.tolist(), select_sample(frame, 20, 43).gml_id.tolist())

    def test_all_rows_and_invalid_sample_parameters(self):
        frame = prepare_attributes(fixture())
        self.assertEqual(len(select_sample(frame, 0, 42)), 30)
        for rows in (-1, 1, 11):
            with self.assertRaises(ValueError):
                select_sample(frame, rows, 42)

    def test_duplicate_and_empty_ids_rejected(self):
        for value in (None, "", "  ", "TEST_0001"):
            frame = fixture()
            frame.loc[0, "gml_id"] = value
            with self.assertRaises(ValueError):
                prepare_attributes(frame)

    def test_missing_date_has_explicit_flag_invalid_date_is_rejected(self):
        frame = fixture()
        prepared = prepare_attributes(frame)
        self.assertEqual(prepared.loc[0, DATE_MISSING], "fehlt")
        self.assertEqual(prepared.loc[1, DATE_MISSING], "vorhanden")
        self.assertTrue(pd.isna(prepared.loc[0, "datumderletztenueberpruefung"]))
        frame.loc[0, "datumderletztenueberpruefung"] = "not-a-date"
        with self.assertRaises(ValueError):
            prepare_attributes(frame)

    def test_parquet_roundtrip_retains_missing_and_categories(self):
        prepared = prepare_attributes(fixture())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attributes.parquet"
            prepared.to_parquet(path, index=False)
            restored = pd.read_parquet(path)
            pd.testing.assert_frame_equal(prepared, restored, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
