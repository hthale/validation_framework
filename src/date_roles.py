"""Explizite DOP-Datumsrollen; Originalwerte bleiben zur Prüfung erhalten."""
from __future__ import annotations

import pandas as pd

DOP_DATE_COLUMNS = ("aktualitaet", "aktualitaet_kachelinformationen")


def apply_date_roles(schema: dict, columns, known_date_columns=DOP_DATE_COLUMNS) -> dict:
    result = dict(schema)
    for name in known_date_columns:
        if name in columns:
            result[name] = "date"
    return result


def date_diagnostics(df: pd.DataFrame, date_columns) -> list[dict]:
    results = []
    for name in date_columns:
        values = df[name]
        # Keine stillschweigende Korrektur: Parser dient nur der Diagnose.
        parsed = pd.to_datetime(values.astype("string"), errors="coerce", format="mixed")
        missing = values.isna()
        invalid = ~missing & parsed.isna()
        results.append({"column": name, "row_count": len(values), "missing_count": int(missing.sum()),
                        "parseable_count": int(parsed.notna().sum()), "invalid_count": int(invalid.sum()),
                        "invalid_examples": values[invalid].astype(str).head(5).tolist(),
                        "role_source": "explicit product profile", "format_conformance_checked": False})
    return results
