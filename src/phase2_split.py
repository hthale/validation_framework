"""Feste Gruppenaufteilung ohne Verwendung von Zielwerten oder Anomaliescores."""
from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phase2_settings import Phase2Settings
from record_identity import validate_record_keys


def _group_value(value):
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise ValueError("Gruppenspalten dürfen keine fehlenden oder verschachtelten Werte enthalten.")
    if isinstance(value, Real):
        if not np.isfinite(value):
            raise ValueError("Nicht endlicher Gruppenwert.")
        return format(float(value), ".17g")
    value = str(value).strip()
    if not value:
        raise ValueError("Leerer Gruppenwert.")
    return value


def group_mapping(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if "record_key" not in df or not df.columns.is_unique:
        raise ValueError("Eindeutige Spalten und vorhandene record_key sind für die Aufteilung erforderlich.")
    keys = validate_record_keys(df["record_key"], context="Phase-2-Aufteilung")
    missing = [c for c in columns if c not in df]
    if missing:
        raise ValueError(f"Explizite Gruppenspalten fehlen: {missing}")
    groups = []
    for row in df[list(columns)].itertuples(index=False, name=None):
        payload = json.dumps([list(columns), [_group_value(v) for v in row]], ensure_ascii=False, separators=(",", ":"))
        groups.append("grp1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return pd.DataFrame({"record_key": keys.to_numpy(), "group_id": groups}).sort_values("record_key").reset_index(drop=True)


def split_protocol(settings: Phase2Settings) -> dict:
    return {"schema_version": 1, "group_columns": list(settings.group_columns),
            "validation_fraction": settings.validation_fraction, "test_fraction": settings.test_fraction,
            "random_state": settings.random_state, "fraction_unit": "groups"}


def make_grouped_split(df: pd.DataFrame, settings: Phase2Settings) -> pd.DataFrame:
    mapping = group_mapping(df, settings.group_columns)
    if mapping["group_id"].nunique() < 6:
        raise ValueError("Mindestens sechs unabhängige Gruppen für Training, Validierung und reservierten Test erforderlich.")
    outer = GroupShuffleSplit(n_splits=1, test_size=settings.test_fraction, random_state=settings.random_state)
    development, test = next(outer.split(mapping, groups=mapping["group_id"]))
    dev = mapping.iloc[development]
    inner = GroupShuffleSplit(n_splits=1, test_size=settings.validation_fraction / (1 - settings.test_fraction), random_state=settings.random_state + 1)
    train_local, val_local = next(inner.split(dev, groups=dev["group_id"]))
    result = mapping.copy()
    result["partition"] = "test"
    result.loc[development[train_local], "partition"] = "train"
    result.loc[development[val_local], "partition"] = "validation"
    return validate_split(df, result, settings)


def validate_split(df: pd.DataFrame, split: pd.DataFrame, settings: Phase2Settings) -> pd.DataFrame:
    if not {"record_key", "group_id", "partition"}.issubset(split.columns):
        raise ValueError("Unvollständiges Aufteilungsmanifest.")
    current = group_mapping(df, settings.group_columns)
    manifest = split[["record_key", "group_id", "partition"]].copy()
    manifest["record_key"] = validate_record_keys(manifest["record_key"], context="Aufteilungsmanifest")
    if set(current["record_key"]) != set(manifest["record_key"]):
        raise ValueError("Aufteilungsmanifest und Eingabebestand haben unterschiedliche record_key.")
    if manifest.isna().any().any() or set(manifest["partition"]) != {"train", "validation", "test"}:
        raise ValueError("Alle drei Partitionen train, validation und test müssen belegt sein.")
    if manifest.groupby("group_id")["partition"].nunique().gt(1).any():
        raise ValueError("Dieselbe Gruppe steht in mehreren Partitionen.")
    manifest = manifest.sort_values("record_key").reset_index(drop=True)
    if not current["group_id"].eq(manifest["group_id"]).all():
        raise ValueError("Gruppenzuordnung hat sich geändert. Referenzgruppen vor Fehlerinjektionen festlegen und erhalten.")
    if manifest.loc[manifest["partition"].eq("train"), "group_id"].nunique() < settings.cv_folds:
        raise ValueError("Trainingsanteil enthält zu wenige Gruppen für die konfigurierte Gruppen-CV.")
    return manifest


def load_or_create_split(df: pd.DataFrame, path: Path, settings: Phase2Settings) -> pd.DataFrame:
    """Eine vorhandene Aufteilung wiederverwenden; Änderungen nicht still ersetzen."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Aufteilungsmanifest benötigt eine .csv-Datei; das Begleitprotokoll verwendet .json.")
    protocol_path = path.with_suffix(".json")
    if path.exists() or protocol_path.exists():
        if not path.is_file() or not protocol_path.is_file():
            raise ValueError("Aufteilungsmanifest oder zugehöriges Protokoll fehlt.")
        if json.loads(protocol_path.read_text(encoding="utf-8")) != split_protocol(settings):
            raise ValueError("Aufteilungsparameter unterscheiden sich vom gespeicherten Protokoll.")
        return validate_split(df, pd.read_csv(path, dtype="string"), settings)
    result = make_grouped_split(df, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exklusives Anlegen verhindert das Überschreiben einer bestehenden Zuordnung.
    with path.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump(split_protocol(settings), stream, ensure_ascii=False, indent=2)
    return result


def select_partition(df: pd.DataFrame, split: pd.DataFrame, partition: str) -> pd.DataFrame:
    keys = split.loc[split["partition"].eq(partition), "record_key"]
    work = df.copy()
    work["record_key"] = validate_record_keys(work["record_key"], context="Partitionsauswahl")
    return work.set_index("record_key", drop=False).loc[keys].reset_index(drop=True)


def split_summary(split: pd.DataFrame, settings: Phase2Settings) -> dict:
    return {
        "protocol": split_protocol(settings),
        "counts": {part: {"rows": int(split["partition"].eq(part).sum()),
                           "groups": int(split.loc[split["partition"].eq(part), "group_id"].nunique())}
                   for part in ("train", "validation", "test")},
        "group_overlap_count": 0,
        "test_evaluated": False,
        "scope": "Supervised raw-attribute dependency rules only; not the previously trained full-population autoencoder or cluster descriptions.",
    }
