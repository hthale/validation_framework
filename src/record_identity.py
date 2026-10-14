"""Datensatzidentität: einmal am Eingang erzeugen, danach unverändert mitführen."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import pandas as pd


# Diese Spalten dienen der Zuordnung, nicht als Modellmerkmale.
IDENTITY_COLUMNS = frozenset({
    "record_key", "row_id", "unit_id", "entity_id", "gml_id",
    "flurstueckskennzeichen", "delivery_id", "delivery_run_id",
    "delivery_group_id", "group_id", "tile_group_id",
})


def validate_record_keys(keys: pd.Series, *, context: str = "Daten") -> pd.Series:
    """Leere oder doppelte Schlüssel sind ein Eingabefehler, kein Merge-Fallback."""
    result = keys.astype("string").rename("record_key")
    empty = result.isna() | result.str.strip().eq("").fillna(False)
    if empty.any():
        raise ValueError(f"{context}: {int(empty.sum())} leere record_key-Werte.")
    duplicated = result.duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"{context}: {int(duplicated.sum())} Zeilen mit doppelten record_key-Werten. "
            "Lieferungs-ID oder explizite zusammengesetzte Identität erforderlich."
        )
    return result


def build_record_key(
    df: pd.DataFrame, *, identity_columns: Sequence[str] | None = None
) -> pd.Series:
    """Vorhandene IDs erhalten; sonst nur aus ausgewählten Identitätsfeldern ableiten.

    Automatisch: gml_id, flurstueckskennzeichen oder DOP-Dateiname inklusive Datum.
    delivery_id wird, sofern vorhanden, als Namensraum einbezogen. Ohne diese
    Spalte ist die Eindeutigkeit nur innerhalb des übergebenen Bestands geprüft.
    Kein Hash über alle Attribute und kein DataFrame-Index als Identität.
    """
    if "record_key" in df.columns:
        return validate_record_keys(df["record_key"])

    if identity_columns is not None:
        columns = list(identity_columns)
    else:
        candidate = next((c for c in (
            "gml_id", "flurstueckskennzeichen", "xml_name", "tif_name", "xml", "tif"
        ) if c in df.columns), None)
        if candidate is None:
            raise ValueError(
                "Kein Identitätsfeld vorhanden. record_key bereitstellen oder "
                "build_record_key(..., identity_columns=[...]) verwenden."
            )
        columns = (["delivery_id"] if "delivery_id" in df.columns else []) + [candidate]

    if not columns or len(set(columns)) != len(columns):
        raise ValueError("identity_columns muss unterschiedliche Spalten enthalten.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Identitätsspalten fehlen: {missing}")

    values = df[columns].astype("string")
    for col in columns:
        # Alte DOP-Spalten können vollständige Windows- oder POSIX-Pfade enthalten.
        if identity_columns is None and col in {"xml", "tif"}:
            values[col] = values[col].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
        invalid = values[col].isna() | values[col].str.strip().eq("").fillna(False)
        if invalid.any():
            raise ValueError(f"Identitätsspalte {col!r}: {int(invalid.sum())} leere Werte.")

    def digest(row: tuple) -> str:
        payload = json.dumps(["record-key-v1", columns, list(row)], ensure_ascii=False, separators=(",", ":"))
        return "rk1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    keys = pd.Series(
        [digest(row) for row in values.itertuples(index=False, name=None)],
        index=df.index, dtype="string", name="record_key",
    )
    return validate_record_keys(keys)


def with_record_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["record_key"] = build_record_key(df)
    return result
