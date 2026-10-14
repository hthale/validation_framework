"""Produktspezifische Attributaufbereitung; keine Geometrieverarbeitung.

Die Feldrollen sind eine offengelegte Auswahl für diesen Versuch, keine aus
ALKIS abgeleiteten Pflichtregeln. Vorhandene fachliche Codes bleiben unverändert.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCALAR_CODES = ("ergebnisderueberpruefung", "istweiterenutzung", "funktion", "zustand")
LIST_CODES = ("anlass", "advstandardmodell", "sonstigesmodell",
              "processstep_ax_datenerhebung", "bahnkategorie")
DATES = ("datumderletztenueberpruefung",)
TEXTS = ("zweitname", "unverschluesselt")
ATTRIBUTES = (*SCALAR_CODES, *LIST_CODES, *DATES, *TEXTS)
REQUIRED = ("gml_id", "bahnkategorie", "funktion", "ergebnisderueberpruefung")
DATE_MISSING = "datumderletztenueberpruefung_fehlt"
FORBIDDEN = {"geometry", "wkb_geometry", "objektkoordinaten", "nummerderbahnstrecke",
             "quellobjektid", "beginnt", "endet", "hatdirektunten", "istteilvon"}


def is_missing(value):
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def code_atom(value):
    if not pd.api.types.is_scalar(value) or isinstance(value, (bytes, bytearray, bool)):
        raise ValueError("Fachcode muss ein skalarer Text oder eine ganze Zahl sein.")
    if is_missing(value):
        raise ValueError("Fehlender Eintrag innerhalb einer Codeliste; Originalwert prüfen.")
    if isinstance(value, str):
        return value  # Führende Nullen und Leerzeichen nicht still entfernen.
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    raise ValueError(f"Kein unterstützter ALKIS-Code: {value!r}")


def encode_code(value):
    if is_missing(value):
        return None
    return "code:" + json.dumps(code_atom(value), ensure_ascii=False)


def encode_list(value):
    if is_missing(value):
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("Codeliste muss eine Liste sein; keine stillschweigende String-Auswertung.")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError("Verschachtelte Codeliste wird nicht unterstützt.")
    # Vollständige Kombination; Reihenfolge und Wiederholungen bleiben erhalten.
    return "liste:" + json.dumps([code_atom(v) for v in value], ensure_ascii=False,
                                 separators=(",", ":"))


def validate_ids(frame):
    if not frame.columns.is_unique or "gml_id" not in frame:
        raise ValueError("Eindeutige Spalten mit vorhandener gml_id erforderlich.")
    ids = frame.gml_id
    if not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("gml_id enthält fehlende, leere oder nichttextuelle Werte.")
    if ids.duplicated().any():
        raise ValueError("Doppelte gml_id: keine automatische Zusammenführung oder Umnummerierung.")


def read_attributes(path):
    path = Path(path)
    schema = pq.read_schema(path)
    names = schema.names
    if len(names) != len(set(names)):
        raise ValueError("Parquet enthält doppelte Spaltennamen.")
    missing = sorted(set(REQUIRED) - set(names))
    if missing:
        raise ValueError(f"Erwartete Bahnverkehr-Spalten fehlen: {missing}")
    selected = ["gml_id", *[c for c in ATTRIBUTES if c in names]]
    # Nur diese Spalten lesen. Kein WKB-Parser, keine Geometrie, keine Koordinaten.
    frame = pq.read_table(path, columns=selected).to_pandas(ignore_metadata=True)
    validate_ids(frame)
    return frame, {"source_schema": [{"column": f.name, "arrow_type": str(f.type)} for f in schema],
                   "read_columns": selected, "excluded_columns": sorted(set(names)-set(selected)),
                   "absent_optional_attributes": sorted(set(ATTRIBUTES)-set(names)),
                   "geometry_columns_read": [], "geometry_features_created": []}


def prepare_attributes(frame):
    validate_ids(frame)
    result = frame[["gml_id"]].copy().reset_index(drop=True)
    for col in ATTRIBUTES:
        if col not in frame:
            continue
        values = frame[col].reset_index(drop=True)
        if col in SCALAR_CODES:
            result[col] = values.map(encode_code)
        elif col in LIST_CODES:
            result[col] = values.map(encode_list)
        else:
            if not values.map(lambda v: is_missing(v) or isinstance(v, str)).all():
                raise ValueError(f"{col}: Text/ISO-Datum erwartet; keine stille Konvertierung.")
            # Leertext bleibt im Originalexport sichtbar; als Lernwert ist er fehlend.
            result[col] = values.map(lambda v: None if is_missing(v) or not v.strip() else v)
    if DATES[0] in result:
        parsed = pd.to_datetime(result[DATES[0]], format="mixed", utc=True, errors="coerce")
        malformed = result[DATES[0]].notna() & parsed.isna()
        if malformed.any():
            raise ValueError(f"{int(malformed.sum())} nicht lesbare Prüfzeitpunkte; keine stille Imputation.")
        result[DATES[0]] = parsed.map(lambda v: v.isoformat() if pd.notna(v) else None)
        # Eigene Belegungsinformation, da die spätere Medianimputation sie sonst verliert.
        result[DATE_MISSING] = parsed.isna().map({True: "fehlt", False: "vorhanden"})
    if any(c in result for c in FORBIDDEN):
        raise AssertionError("Verbotenes Merkmal gelangt in die Attributtabelle.")
    return result


def select_sample(frame, rows, seed):
    validate_ids(frame)
    if rows < 0 or (rows and rows < 12) or not 0 <= seed < 2**32:
        raise ValueError("--rows muss 0 oder mindestens 12 sein; Seed 0 bis 2**32-1.")
    # Auswahl ausschließlich über Kennung und Seed; unabhängig von Attributen und Zeilenfolge.
    rank = frame.gml_id.map(lambda v: hashlib.sha256(
        json.dumps(["bahn-attribute-sample-v1", seed, v], ensure_ascii=False).encode()).hexdigest())
    order = pd.DataFrame({"rank": rank, "id": frame.gml_id}).sort_values(["rank", "id"]).index
    chosen = order[:rows] if rows else order
    return frame.loc[chosen].sort_values("gml_id").reset_index(drop=True)


def column_statistics(frame):
    result = []
    for col in frame:
        if col == "gml_id":
            continue
        s = frame[col]
        result.append({"column": col, "nonmissing": int(s.notna().sum()),
                       "missing": int(s.isna().sum()), "distinct": int(s.nunique(dropna=True)),
                       "value_counts": [{"value": str(k), "count": int(v)}
                                        for k, v in s.value_counts(dropna=True).items()]})
    return result


def check_contract(frame, contract):
    for col in (*SCALAR_CODES, *LIST_CODES, DATE_MISSING):
        if col in frame and frame[col].notna().any():
            role = contract["columns"][col]["role"]
            if role != "categorical":
                raise ValueError(f"{col} wird als {role} erkannt. Zu viele Kombinationen für diesen "
                                 "Adapter; Rolle explizit erweitern, keinen Lauf mit falscher Rolle starten.")
    if DATES[0] in frame and frame[DATES[0]].notna().any():
        if contract["columns"][DATES[0]]["role"] != "date":
            raise ValueError("Prüfzeitpunkt wurde nicht als Datum erkannt.")
    if contract["columns"]["gml_id"]["role"] != "technical":
        raise ValueError("gml_id darf kein Lernmerkmal sein.")

