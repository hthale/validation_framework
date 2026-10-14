from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

from latentspace.config_latent import LatentV2Config


CFG = LatentV2Config()

# =========================================================
# ZENTRALE KONFIGURATION
# =========================================================

DELIVERY_GROUP_ID = getattr(CFG, "delivery_group_id", Path(CFG.parquet_path).stem)
INPUT_PARQUET = Path(CFG.parquet_path)

MEMORY_DIR = Path(
    getattr(
        CFG,
        "delivery_memory_dir",
        r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\delivery_memory_tabular",
    )
)

PROFILE_MEMORY_PARQUET = MEMORY_DIR / "tabular_delivery_profiles.parquet"
RULES_JSON = MEMORY_DIR / "learned_tabular_delivery_rules.json"
REPORT_JSON = MEMORY_DIR / "latest_delivery_check_report.json"

FOLDER_PROFILE_MEMORY_PARQUET = MEMORY_DIR / "folder_delivery_profiles.parquet"
FOLDER_RULES_JSON = MEMORY_DIR / "learned_folder_delivery_rules.json"
FOLDER_REPORT_JSON = MEMORY_DIR / "latest_folder_delivery_check_report.json"


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def make_hashable_value(value):
    """
    Wandelt komplexe Werte in vergleichbare Werte um.
    Wichtig für Spalten mit Listen, Arrays oder Dicts.
    """
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def infer_simple_type(series: pd.Series) -> str:
    """
    Erkennt einen einfachen Datentyp für eine Spalte.

    Reihenfolge ist wichtig:
    - numerisch zuerst
    - boolean danach
    - Datum nur bei echtem Datumsformat
    - sonst string

    Dadurch werden Code-Spalten wie land=03 nicht fälschlich als Datum erkannt.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    s = series.dropna().astype(str)

    if len(s) == 0:
        return "string"

    date_ratio = s.str.match(r"^\d{4}-\d{2}-\d{2}").mean()

    if date_ratio > 0.8:
        return "date"

    return "string"


# =========================================================
# TABELLEN-MEMORY
# =========================================================

def build_dataset_profile(
    df: pd.DataFrame,
    delivery_group_id: str,
    input_path: Path,
) -> pd.DataFrame:
    """
    Erstellt ein Profil der tabellarischen Datei.

    Pro Spalte werden gespeichert:
    - Spaltenname
    - erkannter Typ
    - Nullrate
    - Anzahl eindeutiger Werte
    - numerischer Wertebereich, falls möglich
    """
    rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for col in df.columns:
        s = df[col]
        safe_s = s.map(make_hashable_value)
        dtype_group = infer_simple_type(s)

        row = {
            "delivery_group_id": delivery_group_id,
            "delivery_run_id": run_id,
            "source_file": str(input_path),
            "column_name": col,
            "dtype_group": dtype_group,
            "pandas_dtype": str(s.dtype),
            "row_count": len(df),
            "null_count": int(s.isna().sum()),
            "null_rate": float(s.isna().mean()),
            "unique_count": int(safe_s.nunique(dropna=True)),
        }

        if dtype_group == "numeric":
            numeric = pd.to_numeric(s, errors="coerce")
            row.update({
                "min_value": float(numeric.min()) if numeric.notna().any() else None,
                "max_value": float(numeric.max()) if numeric.notna().any() else None,
                "mean_value": float(numeric.mean()) if numeric.notna().any() else None,
            })
        else:
            row.update({
                "min_value": None,
                "max_value": None,
                "mean_value": None,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def load_memory() -> pd.DataFrame:
    """
    Lädt das bisherige Tabellen-Memory.
    Falls die Datei fehlt oder ungültig ist, wird leer gestartet.
    """
    if PROFILE_MEMORY_PARQUET.exists():
        try:
            df = pd.read_parquet(PROFILE_MEMORY_PARQUET)
            required_cols = ["delivery_group_id", "delivery_run_id", "column_name"]

            if all(c in df.columns for c in required_cols):
                return df

            print("WARNUNG: Tabellen-Memory ungültig. Es wird neu aufgebaut.")
            return pd.DataFrame()

        except Exception as e:
            print(f"WARNUNG: Tabellen-Memory konnte nicht geladen werden: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def save_memory(memory_df: pd.DataFrame) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_df.to_parquet(PROFILE_MEMORY_PARQUET, index=False)


def compare_with_memory(new_profile: pd.DataFrame, memory_df: pd.DataFrame) -> dict:
    """
    Vergleicht den aktuellen Tabellenlauf mit früheren Läufen derselben Liefergruppe.
    """
    group_id = new_profile["delivery_group_id"].iloc[0]

    if memory_df.empty or "delivery_group_id" not in memory_df.columns:
        return {
            "status": "first_run_for_group",
            "message": "Für diese Liefergruppe gibt es noch keine gespeicherten Vergleichsdaten.",
            "new_columns": sorted(new_profile["column_name"].tolist()),
            "missing_columns": [],
            "changed_types": [],
            "numeric_range_warnings": [],
        }

    old = memory_df[memory_df["delivery_group_id"] == group_id].copy()

    if old.empty:
        return {
            "status": "first_run_for_group",
            "message": "Für diese Liefergruppe gibt es noch keine gespeicherten Vergleichsdaten.",
            "new_columns": sorted(new_profile["column_name"].tolist()),
            "missing_columns": [],
            "changed_types": [],
            "numeric_range_warnings": [],
        }

    old_columns = set(old["column_name"].unique())
    new_columns = set(new_profile["column_name"].unique())

    added_columns = sorted(new_columns - old_columns)
    missing_columns = sorted(old_columns - new_columns)

    changed_types = []

    for _, row in new_profile.iterrows():
        col = row["column_name"]
        old_types = set(
            old.loc[old["column_name"] == col, "dtype_group"]
            .dropna()
            .astype(str)
        )

        if old_types and row["dtype_group"] not in old_types:
            changed_types.append({
                "column": col,
                "old_types": sorted(old_types),
                "new_type": row["dtype_group"],
            })

    numeric_range_warnings = []

    for _, row in new_profile.iterrows():
        if row["dtype_group"] != "numeric":
            continue

        col = row["column_name"]
        old_col = old[
            (old["column_name"] == col)
            & (old["dtype_group"] == "numeric")
        ]

        if old_col.empty:
            continue

        old_min = old_col["min_value"].min()
        old_max = old_col["max_value"].max()

        new_min = row["min_value"]
        new_max = row["max_value"]

        if pd.notna(new_min) and pd.notna(old_min) and new_min < old_min:
            numeric_range_warnings.append({
                "column": col,
                "issue": "new_min_below_previous_min",
                "previous_min": float(old_min),
                "new_min": float(new_min),
            })

        if pd.notna(new_max) and pd.notna(old_max) and new_max > old_max:
            numeric_range_warnings.append({
                "column": col,
                "issue": "new_max_above_previous_max",
                "previous_max": float(old_max),
                "new_max": float(new_max),
            })

    return {
        "status": "compared_with_previous_runs",
        "delivery_group_id": group_id,
        "previous_runs": int(old["delivery_run_id"].nunique()),
        "new_columns": added_columns,
        "missing_columns": missing_columns,
        "changed_types": changed_types,
        "numeric_range_warnings": numeric_range_warnings,
    }


def learn_rules(memory_df: pd.DataFrame, min_support_ratio: float = 0.8) -> list[dict]:
    """
    Leitet einfache Tabellenstruktur-Regeln aus mehreren Lieferungen ab.
    """
    rules = []

    if memory_df.empty:
        return rules

    for group_id, group_df in memory_df.groupby("delivery_group_id"):
        run_count = group_df["delivery_run_id"].nunique()

        if run_count < 2:
            continue

        for col in sorted(group_df["column_name"].unique()):
            col_df = group_df[group_df["column_name"] == col]

            support_count = col_df["delivery_run_id"].nunique()
            support_ratio = support_count / run_count

            if support_ratio < min_support_ratio:
                continue

            dtype_values = col_df["dtype_group"].dropna().astype(str).value_counts()
            dominant_type = dtype_values.index[0]

            rule = {
                "rule_id": f"{group_id}__column_exists__{col}",
                "rule_type": "tabular_delivery_structure",
                "delivery_group_id": group_id,
                "column": col,
                "expression": f"COLUMN_EXISTS('{col}') AND TYPE('{col}') = '{dominant_type}'",
                "support_count": int(support_count),
                "run_count": int(run_count),
                "support_ratio": float(support_ratio),
                "status": "proposed",
                "reason": "Diese Spalte kam stabil in mehreren Lieferungen dieser Gruppe vor.",
            }

            if dominant_type == "numeric":
                rule["observed_min"] = float(col_df["min_value"].min())
                rule["observed_max"] = float(col_df["max_value"].max())

            rules.append(rule)

    return rules


# =========================================================
# ORDNER-/DATEI-MEMORY
# =========================================================

def build_folder_profile(delivery_root: Path, delivery_group_id: str) -> pd.DataFrame:
    """
    Erstellt ein Profil der gesamten Lieferstruktur.

    Erfasst werden:
    - Gesamtanzahl Dateien
    - Gesamtanzahl Unterordner
    - Dateiendungen und deren Anzahl
    - relative Dateipfade
    - relative Ordnerpfade
    - Dateinamen
    - Dateigrößen
    """
    rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    files = [p for p in delivery_root.rglob("*") if p.is_file()]
    folders = [p for p in delivery_root.rglob("*") if p.is_dir()]

    extension_counts = {}

    for file in files:
        ext = file.suffix.lower() if file.suffix else "__no_extension__"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    rows.append({
        "delivery_group_id": delivery_group_id,
        "delivery_run_id": run_id,
        "profile_type": "summary",
        "relative_path": ".",
        "name": delivery_root.name,
        "file_stem": None,
        "file_extension": None,
        "file_size_bytes": None,
        "file_count_total": len(files),
        "folder_count_total": len(folders),
        "extension_counts_json": json.dumps(extension_counts, ensure_ascii=False),
    })

    for folder in folders:
        rel_path = str(folder.relative_to(delivery_root))

        rows.append({
            "delivery_group_id": delivery_group_id,
            "delivery_run_id": run_id,
            "profile_type": "folder",
            "relative_path": rel_path,
            "name": folder.name,
            "file_stem": None,
            "file_extension": None,
            "file_size_bytes": None,
            "file_count_total": None,
            "folder_count_total": None,
            "extension_counts_json": None,
        })

    for file in files:
        rel_path = str(file.relative_to(delivery_root))

        rows.append({
            "delivery_group_id": delivery_group_id,
            "delivery_run_id": run_id,
            "profile_type": "file",
            "relative_path": rel_path,
            "name": file.name,
            "file_stem": file.stem,
            "file_extension": file.suffix.lower() if file.suffix else "__no_extension__",
            "file_size_bytes": int(file.stat().st_size),
            "file_count_total": None,
            "folder_count_total": None,
            "extension_counts_json": None,
        })

    return pd.DataFrame(rows)


def load_folder_memory() -> pd.DataFrame:
    """
    Lädt das bisherige Ordner-/Datei-Memory.
    """
    if FOLDER_PROFILE_MEMORY_PARQUET.exists():
        try:
            df = pd.read_parquet(FOLDER_PROFILE_MEMORY_PARQUET)
            required_cols = ["delivery_group_id", "delivery_run_id", "profile_type"]

            if all(c in df.columns for c in required_cols):
                return df

            print("WARNUNG: Folder-Memory ungültig. Es wird neu aufgebaut.")
            return pd.DataFrame()

        except Exception as e:
            print(f"WARNUNG: Folder-Memory konnte nicht geladen werden: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def save_folder_memory(memory_df: pd.DataFrame) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_df.to_parquet(FOLDER_PROFILE_MEMORY_PARQUET, index=False)


def compare_folder_with_memory(new_profile: pd.DataFrame, memory_df: pd.DataFrame) -> dict:
    """
    Vergleicht den aktuellen Lieferordner mit früheren Lieferordnern derselben Gruppe.
    """
    group_id = new_profile["delivery_group_id"].iloc[0]

    if memory_df.empty or "delivery_group_id" not in memory_df.columns:
        return {
            "status": "first_folder_run_for_group",
            "message": "Für diese Liefergruppe gibt es noch keine gespeicherten Ordner-Vergleichsdaten.",
            "new_files": sorted(
                new_profile.loc[new_profile["profile_type"] == "file", "relative_path"]
                .dropna()
                .tolist()
            ),
            "missing_files": [],
            "new_folders": sorted(
                new_profile.loc[new_profile["profile_type"] == "folder", "relative_path"]
                .dropna()
                .tolist()
            ),
            "missing_folders": [],
            "file_count_warnings": [],
            "extension_count_warnings": [],
        }

    old = memory_df[memory_df["delivery_group_id"] == group_id].copy()

    if old.empty:
        return {
            "status": "first_folder_run_for_group",
            "message": "Für diese Liefergruppe gibt es noch keine gespeicherten Ordner-Vergleichsdaten.",
            "new_files": sorted(
                new_profile.loc[new_profile["profile_type"] == "file", "relative_path"]
                .dropna()
                .tolist()
            ),
            "missing_files": [],
            "new_folders": sorted(
                new_profile.loc[new_profile["profile_type"] == "folder", "relative_path"]
                .dropna()
                .tolist()
            ),
            "missing_folders": [],
            "file_count_warnings": [],
            "extension_count_warnings": [],
        }

    old_files = set(
        old.loc[old["profile_type"] == "file", "relative_path"]
        .dropna()
        .astype(str)
    )
    new_files = set(
        new_profile.loc[new_profile["profile_type"] == "file", "relative_path"]
        .dropna()
        .astype(str)
    )

    old_folders = set(
        old.loc[old["profile_type"] == "folder", "relative_path"]
        .dropna()
        .astype(str)
    )
    new_folders = set(
        new_profile.loc[new_profile["profile_type"] == "folder", "relative_path"]
        .dropna()
        .astype(str)
    )

    new_summary = new_profile[new_profile["profile_type"] == "summary"].iloc[0]
    old_summary = old[old["profile_type"] == "summary"]

    file_count_warnings = []

    if not old_summary.empty:
        old_min = int(old_summary["file_count_total"].min())
        old_max = int(old_summary["file_count_total"].max())
        new_count = int(new_summary["file_count_total"])

        if new_count < old_min or new_count > old_max:
            file_count_warnings.append({
                "issue": "file_count_outside_previous_range",
                "previous_min": old_min,
                "previous_max": old_max,
                "new_count": new_count,
            })

    return {
        "status": "compared_folder_with_previous_runs",
        "delivery_group_id": group_id,
        "previous_runs": int(old["delivery_run_id"].nunique()),
        "new_files": sorted(new_files - old_files),
        "missing_files": sorted(old_files - new_files),
        "new_folders": sorted(new_folders - old_folders),
        "missing_folders": sorted(old_folders - new_folders),
        "file_count_warnings": file_count_warnings,
        "extension_count_warnings": [],
    }


def learn_folder_rules(memory_df: pd.DataFrame, min_support_ratio: float = 0.8) -> list[dict]:
    """
    Leitet Regeln zur Lieferstruktur ab.

    Gelernt werden:
    - typische Gesamtanzahl Dateien
    - stabile Unterordner
    - stabile Dateinamen
    - stabile Dateiendungsanzahlen
    """
    rules = []

    if memory_df.empty:
        return rules

    for group_id, group_df in memory_df.groupby("delivery_group_id"):
        run_count = group_df["delivery_run_id"].nunique()

        if run_count < 2:
            continue

        summary_df = group_df[group_df["profile_type"] == "summary"]

        if not summary_df.empty:
            min_files = int(summary_df["file_count_total"].min())
            max_files = int(summary_df["file_count_total"].max())

            rules.append({
                "rule_id": f"{group_id}__file_count_total",
                "rule_type": "folder_delivery_structure",
                "delivery_group_id": group_id,
                "expression": f"FILE_COUNT_TOTAL BETWEEN {min_files} AND {max_files}",
                "observed_min": min_files,
                "observed_max": max_files,
                "run_count": int(run_count),
                "status": "proposed",
                "reason": "Gesamtanzahl der Dateien wurde über mehrere Lieferungen beobachtet.",
            })

        folder_df = group_df[group_df["profile_type"] == "folder"]

        for folder_path in sorted(folder_df["relative_path"].dropna().unique()):
            support_count = folder_df.loc[
                folder_df["relative_path"] == folder_path,
                "delivery_run_id",
            ].nunique()

            support_ratio = support_count / run_count

            if support_ratio >= min_support_ratio:
                rules.append({
                    "rule_id": f"{group_id}__folder_exists__{folder_path}",
                    "rule_type": "required_folder",
                    "delivery_group_id": group_id,
                    "folder_path": folder_path,
                    "expression": f"FOLDER_EXISTS('{folder_path}')",
                    "support_count": int(support_count),
                    "run_count": int(run_count),
                    "support_ratio": float(support_ratio),
                    "status": "proposed",
                    "reason": "Dieser Unterordner kam stabil in mehreren Lieferungen vor.",
                })

        file_df = group_df[group_df["profile_type"] == "file"]

        for file_name in sorted(file_df["name"].dropna().unique()):
            support_count = file_df.loc[
                file_df["name"] == file_name,
                "delivery_run_id",
            ].nunique()

            support_ratio = support_count / run_count

            if support_ratio >= min_support_ratio:
                rules.append({
                    "rule_id": f"{group_id}__file_exists__{file_name}",
                    "rule_type": "required_file",
                    "delivery_group_id": group_id,
                    "file_name": file_name,
                    "expression": f"FILE_EXISTS('{file_name}')",
                    "support_count": int(support_count),
                    "run_count": int(run_count),
                    "support_ratio": float(support_ratio),
                    "status": "proposed",
                    "reason": "Diese Datei kam stabil in mehreren Lieferungen vor.",
                })

        for ext in sorted(file_df["file_extension"].dropna().unique()):
            ext_df = file_df[file_df["file_extension"] == ext]
            counts_per_run = ext_df.groupby("delivery_run_id").size()

            rules.append({
                "rule_id": f"{group_id}__extension_count__{ext}",
                "rule_type": "file_extension_count",
                "delivery_group_id": group_id,
                "file_extension": ext,
                "expression": (
                    f"EXTENSION_COUNT('{ext}') "
                    f"BETWEEN {int(counts_per_run.min())} AND {int(counts_per_run.max())}"
                ),
                "observed_min": int(counts_per_run.min()),
                "observed_max": int(counts_per_run.max()),
                "run_count": int(run_count),
                "status": "proposed",
                "reason": "Anzahl dieser Dateiendung wurde über mehrere Lieferungen beobachtet.",
            })

    return rules


# =========================================================
# GEMEINSAMER HAUPTAUFRUF FÜR main_v2.py
# =========================================================

def run_delivery_memory(
    df: pd.DataFrame,
    input_parquet: Path,
    delivery_group_id: str,
    memory_dir: Path,
    delivery_root_dir: Path | None = None,   # explizit übergeben
) -> None:
    """
    Aktualisiert das Delivery-Memory.

    Dieser Aufruf wird aus main_v2.py genutzt.
    Er speichert:
    - tabellarische Strukturinformationen
    - Ordner- und Dateiinformationen, sofern delivery_root_dir gesetzt ist
    """
    df = df.copy()
    df = df.drop(columns=["geometry", "__index_level_0__"], errors="ignore")

    print("\nAktualisiere Delivery-Memory ...")
    print(f"Datei: {input_parquet}")
    print(f"Liefergruppe: {delivery_group_id}")

    # -----------------------------
    # Tabellenstruktur speichern
    # -----------------------------
    new_profile = build_dataset_profile(
        df=df,
        delivery_group_id=delivery_group_id,
        input_path=input_parquet,
    )

    memory_df = load_memory()
    report = compare_with_memory(new_profile, memory_df)

    combined_memory = pd.concat([memory_df, new_profile], ignore_index=True)
    save_memory(combined_memory)

    rules = learn_rules(combined_memory, min_support_ratio=0.8)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    with open(RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Delivery-Memory gespeichert:")
    print(f"  Tabellen-Profil:   {PROFILE_MEMORY_PARQUET}")
    print(f"  Tabellen-Regeln:   {RULES_JSON}")
    print(f"  Tabellen-Bericht:  {REPORT_JSON}")
    print(f"  Tabellen-Regeln:   {len(rules)}")

    # -----------------------------
    # Ordnerstruktur speichern
    # -----------------------------
    delivery_root_dir = getattr(CFG, "delivery_root_dir", None)

    if not delivery_root_dir:
        print("  Kein delivery_root_dir gesetzt. Folder-Memory wird übersprungen.")
        return

    delivery_root = Path(delivery_root_dir)

    if not delivery_root.exists():
        print(f"  WARNUNG: delivery_root_dir existiert nicht: {delivery_root}")
        return

    folder_profile = build_folder_profile(
        delivery_root=delivery_root,
        delivery_group_id=delivery_group_id,
    )

    folder_memory_df = load_folder_memory()
    folder_report = compare_folder_with_memory(folder_profile, folder_memory_df)

    combined_folder_memory = pd.concat(
        [folder_memory_df, folder_profile],
        ignore_index=True,
    )

    save_folder_memory(combined_folder_memory)

    folder_rules = learn_folder_rules(
        combined_folder_memory,
        min_support_ratio=0.8,
    )

    with open(FOLDER_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(folder_rules, f, ensure_ascii=False, indent=2)

    with open(FOLDER_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(folder_report, f, ensure_ascii=False, indent=2)

    print(f"  Folder-Profil:     {FOLDER_PROFILE_MEMORY_PARQUET}")
    print(f"  Folder-Regeln:     {FOLDER_RULES_JSON}")
    print(f"  Folder-Bericht:    {FOLDER_REPORT_JSON}")
    print(f"  Folder-Regeln:     {len(folder_rules)}")


# =========================================================
# DIREKTER TESTAUFRUF
# =========================================================

def main():
    """
    Direkter Testlauf nur für diese Datei.
    Normalerweise wird run_delivery_memory(...) aus main_v2.py genutzt.
    """
    print("Lade neuen Datensatz ...")

    df = pd.read_parquet(INPUT_PARQUET)
    run_delivery_memory(
        df=df,
        input_parquet=INPUT_PARQUET,
        delivery_group_id=DELIVERY_GROUP_ID,
    )


if __name__ == "__main__":
    main()