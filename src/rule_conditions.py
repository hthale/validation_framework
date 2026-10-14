"""Strukturierte Baum-Bedingungen auf Originalspalten auswerten, ohne eval()."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalize_condition(condition: dict | str) -> dict:
    if isinstance(condition, str):
        # Rückwärtskompatibilität für einfache numerische Regeln; kein Raten
        # von Originalspalten aus alten One-Hot-Namen wie kamera_sensor_X_Y.
        match = re.fullmatch(r"\s*(.+?)\s*(<=|>)\s*([-+0-9.eE]+)\s*", condition)
        if not match:
            raise ValueError(f"Nicht unterstützte alte Regelbedingung: {condition!r}")
        result = {"feature": match[1], "operator": match[2], "value": float(match[3])}
    elif isinstance(condition, dict):
        result = dict(condition)
    else:
        raise ValueError("Regelbedingungen müssen Objekte oder numerische Altbedingungen sein.")
    if not isinstance(result.get("feature"), str) or not result["feature"]:
        raise ValueError("Regelbedingung ohne Originalspalte.")
    if result.get("operator") not in {"<=", ">", "==", "!="} or "value" not in result:
        raise ValueError(f"Nicht unterstützte Regelbedingung: {result}")
    if result["operator"] in {"<=", ">"} and not np.isfinite(float(result["value"])):
        raise ValueError("Numerische Regelschwelle muss endlich sein.")
    return result


def evaluate_conditions(df: pd.DataFrame, conditions: list[dict | str]) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for raw in conditions:
        condition = normalize_condition(raw)
        col, op, value = condition["feature"], condition["operator"], condition["value"]
        if col not in df.columns:
            raise ValueError(f"Bedingungsspalte {col!r} fehlt; Baumregeln gegebenenfalls neu erzeugen.")
        series = df[col]
        if condition.get("coerce") == "string":
            series = series.astype("string")
        if op in {"<=", ">"}:
            series = pd.to_numeric(series, errors="coerce")
        if "impute_value" in condition:
            series = series.fillna(condition["impute_value"])
        if condition.get("numeric_cast") == "float32":
            # sklearn-Entscheidungsbäume vergleichen float32-Eingaben mit ihrer
            # gespeicherten Schwelle. Auch direkt an der Grenze nachbilden.
            series = series.astype("float32").astype("float64")
        if op == "<=":
            current = series <= value
        elif op == ">":
            current = series > value
        elif op == "==":
            current = series == value
        else:
            current = series != value
        mask &= (series.notna() & current.fillna(False)).astype(bool)
    return mask


def condition_text(condition: dict) -> str:
    c = normalize_condition(condition)
    return f"{c['feature']} {c['operator']} {c['value']!r}"
