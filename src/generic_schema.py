"""Datenbasierte Spaltenrollen für unbekannte, flache Parquet-Tabellen.

Rollen sind technische Heuristiken, keine fachliche Produktspezifikation.
Originalwerte bleiben in der Eingabekopie erhalten. Zwei getrennte Verträge:
Bestandsbeschreibung (alle Zeilen) und Abhängigkeitslernen (nur Training).
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid

import numpy as np
import pandas as pd

from latentspace.anomaly_scores import SCORE_COLUMNS
from record_identity import IDENTITY_COLUMNS, build_record_key, validate_record_keys

DATE_SHAPE = r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\.\d{1,2}\.\d{4})(?:[ T].*)?"
INTERNAL_PREFIX = "__generic_"


def is_id_name(name):
    return name in IDENTITY_COLUMNS or bool(re.search(
        r"(^id$|(?:^|_)(?:id|uuid|guid|key|identifier|kennung|kennzeichen)$|kennzeichen$)", name.lower()))


def is_group_name(name):
    return bool(re.search(r"(?:^|_)(?:group|gruppe|entity|object|objekt|parent|unit|file|datei|tile|kachel)(?:_|$)", name.lower()))


def is_path_name(name):
    return bool(re.search(r"(?:^|_)(?:path|pfad|folder|directory|verzeichnis|filename|dateiname)(?:_|$)", name.lower()))


def validate_input(frame):
    if not len(frame) or not frame.columns.is_unique or any(not isinstance(c, str) or not c.strip() for c in frame):
        raise ValueError("Eine nichtleere Tabelle mit eindeutigen, nichtleeren Spaltennamen wird benötigt.")
    reserved = [c for c in frame if c in SCORE_COLUMNS or c.startswith((INTERNAL_PREFIX, "latent_", "legacy_", "stratum_"))]
    if reserved:
        raise ValueError(f"Eingabe enthält reservierte Analyse-/Ergebnisspalten: {reserved}. Originaltabelle verwenden.")


def attach_identity(frame, id_columns=None):
    validate_input(frame)
    work = frame.copy(deep=True).reset_index(drop=True)
    if "record_key" in work:
        work["record_key"] = validate_record_keys(work["record_key"])
        return work, {"method": "existing_record_key", "scope": "as_supplied"}
    if id_columns:
        work["record_key"] = build_record_key(work, identity_columns=id_columns)
        return work, {"method": "explicit_identity_columns", "columns": list(id_columns),
                      "scope": "identity_values; include delivery column if needed"}
    # Neue Importidentität, KEIN Hash über Attribute oder veränderliche Zeilenwerte.
    work["record_key"] = pd.Series(["rk7:" + uuid.uuid4().hex for _ in range(len(work))], dtype="string")
    return work, {"method": "persisted_import_uuid", "scope": "saved_input_copy",
                  "note": "Für Folgeläufe input_with_ids.parquet verwenden. Ein erneuter Import der Datei ohne IDs erhält neue Kennungen."}


def _has_structure(series):
    return any(not pd.api.types.is_scalar(v) or isinstance(v, (bytes, bytearray, memoryview))
               for v in series.dropna())


def infer_columns(frame, *, excluded=(), scope="descriptive_full_population"):
    columns = {}
    excluded = set(excluded)
    for name in sorted(frame.columns):
        s = frame[name]
        base = {"dtype": str(s.dtype), "missing_count": int(s.isna().sum())}
        if name in excluded or name in IDENTITY_COLUMNS or name.startswith(INTERNAL_PREFIX):
            role, reason = "technical", "identity_or_explicit_exclusion"
        elif _has_structure(s):
            role, reason = "unsupported", "nested_or_binary; raw values retained, geometry not interpreted"
        elif is_id_name(name) or is_group_name(name) or is_path_name(name):
            role, reason = "technical", "name_heuristic_identity_group_or_path"
        elif s.dropna().empty:
            role, reason = "empty", "no_observed_values"
        elif pd.api.types.is_datetime64_any_dtype(s):
            role, reason = "date", "datetime_dtype"
        elif pd.api.types.is_bool_dtype(s):
            role, reason = "categorical", "boolean_dtype"
        elif pd.api.types.is_numeric_dtype(s):
            role, reason = "numeric", "numeric_dtype"
        else:
            values = s.dropna().astype(str).str.strip()
            date_share = values.str.fullmatch(DATE_SHAPE).mean()
            leading_zeros = values.str.fullmatch(r"[+-]?0\d+").any()
            numeric_share = pd.to_numeric(values, errors="coerce").notna().mean()
            if date_share >= .95:
                role, reason = "date", "date_shape_heuristic"
            elif numeric_share >= .95 and not leading_zeros:
                role, reason = "numeric", "numeric_text_heuristic"
            elif values.nunique() <= 20:
                role, reason = "categorical", "at_most_20_distinct_values"
            elif values.nunique() / len(values) >= .98 and values.str.fullmatch(r"[A-Za-z0-9_.:-]+").all():
                role, reason = "technical", "nearly_unique_code_heuristic; possibly_identifier"
            else:
                role, reason = "text", "text_values"
        base.update(role=role, reason=reason)
        if role not in {"unsupported", "technical"}:
            base["distinct_count"] = int(s.nunique(dropna=True))
        columns[name] = base
    return {"schema_version": 1, "scope": scope, "columns": columns,
            "note": "Heuristische Rollen; keine fachliche Bedeutungs- oder Unabhängigkeitserkennung."}


def normalize_columns(frame, contract):
    result = pd.DataFrame(index=frame.index)
    issues = []
    for name, info in contract["columns"].items():
        if name not in frame:
            raise ValueError(f"Merkmal des gespeicherten Vertrags fehlt: {name}")
        role = info["role"]
        if role in {"technical", "unsupported"}:
            continue
        s = frame[name]
        if role == "numeric":
            numeric = pd.to_numeric(s, errors="coerce").astype(float)
            result[name] = numeric.replace([np.inf, -np.inf], np.nan)
        elif role == "date":
            text = s.astype("string")
            # Europäische Punktnotation bewusst getrennt von ISO-Daten auswerten.
            dot = text.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}", na=False)
            parsed = pd.to_datetime(text.where(~dot), errors="coerce", format="mixed", utc=True)
            parsed.loc[dot] = pd.to_datetime(text[dot], errors="coerce", format="mixed", dayfirst=True, utc=True)
            result[name] = parsed.map(lambda x: x.isoformat() if pd.notna(x) else None)
        else:
            result[name] = s.astype("string").replace(r"^\s*$", pd.NA, regex=True).astype(object)
            result[name] = result[name].where(pd.notna(result[name]), None)
        lost = s.notna() & result[name].isna()
        if lost.any():
            issues.append({"column": name, "normalization_missing_count": int(lost.sum()),
                           "examples": s[lost].astype(str).head(5).tolist(),
                           "record_keys": frame.loc[lost, "record_key"].astype(str).tolist() if "record_key" in frame else []})
    if "record_key" in frame:
        result["record_key"] = frame["record_key"].astype("string")
    return result, issues


def choose_groups(frame, explicit=None):
    """Heuristische Gruppen sind keine behauptete fachliche Unabhängigkeit."""
    if explicit:
        missing = set(explicit) - set(frame)
        if missing:
            raise ValueError(f"Gruppenspalten fehlen: {sorted(missing)}")
        selected, mode = list(explicit), "user_supplied_groups"
    else:
        candidates = []
        for c in sorted(frame):
            if c == "record_key" or not (is_group_name(c) or is_id_name(c)) or _has_structure(frame[c]):
                continue
            s = frame[c]
            n = s.nunique(dropna=True)
            if s.notna().all() and 6 <= n < len(frame) and not s.astype(str).str.strip().eq("").any():
                candidates.append(c)
        # Mehrere widersprüchliche Hierarchien nicht automatisch gleichsetzen.
        if len(candidates) == 1:
            selected, mode = candidates, "heuristic_named_groups"
        else:
            selected, mode = [], "exploratory_rows_and_exact_duplicates"
    work = frame.copy()
    if selected:
        from phase2_split import group_mapping
        mapping = group_mapping(work, tuple(selected)).set_index("record_key")["group_id"]
        work["__generic_group"] = work["record_key"].map(mapping)
    else:
        # Gleiche Originalzeilen zusammenhalten; Identität bleibt davon getrennt.
        cols = [c for c in sorted(frame) if c != "record_key" and not is_id_name(c) and not is_path_name(c)]
        def fingerprint(row):
            def value(v):
                if pd.api.types.is_scalar(v) and pd.isna(v): return ["missing", None]
                if isinstance(v, np.ndarray): return ["array", [value(x) for x in v.tolist()]]
                if isinstance(v, (list, tuple)): return ["list", [value(x) for x in v]]
                if isinstance(v, dict): return ["dict", sorted((str(k), value(x)) for k, x in v.items())]
                if isinstance(v, (bytes, bytearray, memoryview)): return ["binary", bytes(v).hex()]
                return [type(v).__name__, str(v)]
            return hashlib.sha256(json.dumps([value(v) for v in row], ensure_ascii=False).encode()).hexdigest()
        work["__generic_group"] = [fingerprint(row) for row in frame[cols].itertuples(index=False, name=None)]
    return work, {"mode": mode, "columns": selected, "group_count": int(work["__generic_group"].nunique()),
                  "independence_confirmed": False,
                  "note": "Keine automatisch bestätigte räumliche, zeitliche oder lieferungsbezogene Unabhängigkeit. Bekannte Gruppenspalten können mit --group-columns angegeben werden."}


def tabular_preparation(normalized, contract, max_string_fields=12):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    model = pd.DataFrame(index=normalized.index)
    numeric, categorical, strings, derived = [], [], [], []
    for col, info in contract["columns"].items():
        role = info["role"]
        if col not in normalized or normalized[col].nunique(dropna=True) < 2:
            continue
        if role == "numeric":
            model[col] = normalized[col]
            numeric.append(col)
        elif role == "categorical":
            model[col] = normalized[col].fillna("__MISSING__").astype(str)
            categorical.append(col)
        elif role == "date":
            d = pd.to_datetime(normalized[col], errors="coerce", utc=True)
            name = f"__generic_date_{len(derived)}"
            model[name] = d.map(lambda v: v.timestamp() / 86400 if pd.notna(v) else np.nan)
            numeric.append(name)
            derived.append({"source": col, "feature": name, "transformation": "UTC days since Unix epoch"})
        if role in {"text", "categorical"}:
            strings.append(col)
            model[col] = normalized[col].fillna("").astype(str)
    string_selected = sorted(strings, key=lambda c: (contract["columns"][c]["role"] != "text", c))[:max_string_fields]
    if not numeric and not categorical:
        if not string_selected:
            raise ValueError("Keine variierenden skalaren Lernmerkmale gefunden; Rollen in der Eingabe prüfen.")
        for i, col in enumerate(string_selected):
            name = f"__generic_text_length_{i}"
            model[name] = model[col].str.len().astype(float)
            numeric.append(name)
            derived.append({"source": col, "feature": name, "transformation": "string_length"})
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")
    matrix = np.asarray(preprocessor.fit_transform(model), dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Generische Vorverarbeitung erzeugt keine endliche Merkmalsmatrix.")
    prepared = {"X_tab": matrix, "preprocessor": preprocessor, "numeric_cols": numeric, "categorical_cols": categorical}
    metadata = {"numeric_columns": numeric, "categorical_columns": categorical,
                "string_columns": string_selected, "string_columns_not_used": sorted(set(strings)-set(string_selected)),
                "derived_features": derived, "encoded_tabular_dimensions": matrix.shape[1],
                "fit_scope": "descriptive_full_population", "string_encoder": "existing_mean_char_encoder; order-insensitive"}
    return model, prepared, metadata
