"""Striktes JSON und atomare Schreibvorgänge für nachvollziehbare Prüfläufe."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def json_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Nicht unterstützter JSON-Wert: {type(value).__name__}")


def canonical(value):
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject_constant(text):
        raise ValueError(f"Ungültige JSON-Zahl: {text}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def write_json(path, value):
    atomic_text(path, json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_bundle(bundle):
    if bundle.get("schema_version") != 1 or bundle.get("bundle_id") != "review1:" + digest({k: v for k, v in bundle.items() if k != "bundle_id"}):
        raise ValueError("Prüfpaket wurde verändert oder verwendet ein unbekanntes Schema.")
    keys = [row["record_key"] for row in bundle["cases"]]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Prüfpaket enthält keine oder doppelte Datensatzkennungen.")
    return bundle
