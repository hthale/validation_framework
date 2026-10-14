from pathlib import Path
import hashlib
import json
import pandas as pd

# Für jede Lieferung hier auf 002, 003 etc. anpassen, damit die Ergebnisse in getrennten Dateien landen und nicht überschrieben werden
DELIVERY_ROOT = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\parquetdata\lieferung_001"
)

OUTPUT_PARQUET = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\parquetdata\delivery_structure_lieferung_001.parquet"
)

MEMORY_PARQUET = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\delivery_structure_memory\delivery_profiles.parquet"
)

RULES_JSON = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\delivery_structure_memory\learned_delivery_structure_rules.json"
)

def build_folder_profile(delivery_root: Path, delivery_group_id: str) -> pd.DataFrame:
    rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    files = [p for p in delivery_root.rglob("*") if p.is_file()]
    dirs = [p for p in delivery_root.rglob("*") if p.is_dir()]

    extension_counts = {}
    for file in files:
        ext = file.suffix.lower() or "__no_extension__"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    rows.append({
        "delivery_group_id": delivery_group_id,
        "delivery_run_id": run_id,
        "profile_type": "folder_summary",
        "root_path": str(delivery_root),
        "file_count_total": len(files),
        "folder_count_total": len(dirs),
        "extension_counts": json.dumps(extension_counts, ensure_ascii=False),
    })

    for file in files:
        rel_path = file.relative_to(delivery_root)
        rows.append({
            "delivery_group_id": delivery_group_id,
            "delivery_run_id": run_id,
            "profile_type": "file",
            "root_path": str(delivery_root),
            "relative_path": str(rel_path),
            "parent_folder": str(rel_path.parent),
            "file_name": file.name,
            "file_stem": file.stem,
            "file_extension": file.suffix.lower(),
            "file_size_bytes": file.stat().st_size,
        })

    return pd.DataFrame(rows)

def stable_file_id(relative_path: str) -> str:
    return hashlib.md5(relative_path.encode("utf-8")).hexdigest()


def filename_pattern(file_name: str) -> str:
    pattern = []

    for ch in file_name:
        if ch.isdigit():
            pattern.append("9")
        elif ch.isalpha():
            pattern.append("A")
        elif ch in {"_", "-", "."}:
            pattern.append(ch)
        else:
            pattern.append("X")

    return "".join(pattern)


def guess_object_type(path: Path) -> str:
    text = str(path).lower()

    if "flurstueck" in text or "flurstück" in text:
        return "flurstueck"
    if "gebaeude" in text or "gebäude" in text:
        return "gebaeude"
    if "buchungsblatt" in text:
        return "buchungsblatt"
    if "nas" in text:
        return "nas"
    if "meta" in text:
        return "metadata"

    return "unknown"


def build_structure_dataframe(root: Path) -> pd.DataFrame:
    rows = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)
        parts = relative_path.parts
        stat = file_path.stat()

        rows.append({
            "delivery_id": root.name,
            "record_key": stable_file_id(str(relative_path)),

            "relative_path": str(relative_path),
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_extension": file_path.suffix.lower(),

            "parent_folder": file_path.parent.name,
            "top_folder": parts[0] if len(parts) > 0 else "",
            "folder_depth": len(parts) - 1,

            "file_size_bytes": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 4),

            "object_type_from_path": guess_object_type(relative_path),
            "filename_pattern": filename_pattern(file_path.name),

            "is_xml": file_path.suffix.lower() == ".xml",
            "is_gml": file_path.suffix.lower() == ".gml",
            "is_zip": file_path.suffix.lower() == ".zip",
            "is_tif": file_path.suffix.lower() in {".tif", ".tiff"},
            "is_metadata_file": "meta" in str(relative_path).lower(),
            "is_nas_related": "nas" in str(relative_path).lower(),
        })

    return pd.DataFrame(rows)


def append_to_memory(df_structure: pd.DataFrame) -> pd.DataFrame:
    MEMORY_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if MEMORY_PARQUET.exists():
        old = pd.read_parquet(MEMORY_PARQUET)

        # gleiche Lieferung vorher entfernen, damit erneute Läufe nicht doppeln
        old = old[old["delivery_id"] != df_structure["delivery_id"].iloc[0]]

        combined = pd.concat([old, df_structure], ignore_index=True)
    else:
        combined = df_structure.copy()

    combined.to_parquet(MEMORY_PARQUET, index=False)
    return combined


def learn_structure_rules(memory_df: pd.DataFrame, min_support_ratio: float = 0.8) -> list[dict]:
    rules = []
    delivery_count = memory_df["delivery_id"].nunique()

    if delivery_count < 2:
        return rules

    columns_to_check = [
        "top_folder",
        "parent_folder",
        "file_extension",
        "object_type_from_path",
        "filename_pattern",
    ]

    for col in columns_to_check:
        if col not in memory_df.columns:
            continue

        values_per_delivery = (
            memory_df.groupby("delivery_id")[col]
            .apply(lambda s: set(s.dropna().astype(str)))
        )

        all_values = sorted(set().union(*values_per_delivery))

        for value in all_values:
            support_count = sum(value in values for values in values_per_delivery)
            support_ratio = support_count / delivery_count

            if support_ratio >= min_support_ratio:
                rules.append({
                    "rule_id": f"delivery_structure__{col}__{value}",
                    "rule_name": f"Repeated delivery structure value for {col}",
                    "rule_type": "delivery_structure_presence",
                    "column": col,
                    "value": value,
                    "expression": f"{col} CONTAINS '{value}'",
                    "support_count": int(support_count),
                    "delivery_count": int(delivery_count),
                    "support_ratio": float(support_ratio),
                    "status": "proposed",
                    "reason": (
                        "Dieses Strukturmerkmal kam in mehreren Lieferungen stabil vor "
                        "und wird deshalb als Regelvorschlag gespeichert."
                    ),
                })

    return rules


def main():
    print("Lese Lieferstruktur ein ...")
    df_structure = build_structure_dataframe(DELIVERY_ROOT)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_structure.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Strukturdatensatz gespeichert: {OUTPUT_PARQUET}")
    print(f"Dateien in aktueller Lieferung: {len(df_structure)}")

    print("Aktualisiere Struktur-Gedächtnis ...")
    memory_df = append_to_memory(df_structure)

    print(f"Lieferungen im Gedächtnis: {memory_df['delivery_id'].nunique()}")
    print(f"Dateizeilen im Gedächtnis: {len(memory_df)}")

    print("Lerne stabile Strukturregeln ...")
    rules = learn_structure_rules(memory_df, min_support_ratio=0.8)

    RULES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"Regelvorschläge gespeichert: {RULES_JSON}")
    print(f"Anzahl Regelvorschläge: {len(rules)}")


if __name__ == "__main__":
    main()