from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd


@dataclass
class InjectionConfig:
    input_parquet: str = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\dopparwrong\dop20-rgbi.parquet"
    output_dir: str = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\dopparwrong\injected_runs"
    output_filename_prefix: str = "dop20-rgbi_injected"
    random_state: int = 42


CFG = InjectionConfig()


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_record_key(df: pd.DataFrame) -> pd.Series:
    preferred_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["gml_id", "kennzeichen", "uuid", "id"])
        and df[c].notna().mean() > 0.9
        and df[c].nunique(dropna=True) / max(len(df), 1) > 0.9
    ]

    if preferred_candidates:
        selected = preferred_candidates[:2]
        print(f"Record-Key-Spalten: {selected}")
        return df[selected].astype("string").fillna("").agg("||".join, axis=1)

    fallback = pd.util.hash_pandas_object(
        df.astype("string").fillna(""),
        index=False,
    ).astype(str)

    print("Record-Key: Fallback-Hash über ganze Zeile")
    return fallback


def log_change(
    logs: list[dict],
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    idx: int,
    column: str,
    old_value,
    new_value,
    error_type: str,
    note: str,
) -> None:
    logs.append({
        "row_index": int(idx),
        "record_key_before": str(df_before.loc[idx, "record_key"]),
        "record_key_after": str(df_after.loc[idx, "record_key"]),
        "column": column,
        "old_value": old_value,
        "new_value": new_value,
        "error_type": error_type,
        "note": note,
    })


def replace_first_date_in_text(text: str, new_date: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}", new_date, text, count=1)


def replace_first_nine_digit_prefix(text: str, new_prefix: str) -> str:
    return re.sub(r"^\d{9}", new_prefix, text, count=1)


def replace_coords_in_kachelname(kachelname: str, new_east_3: str, new_north_4: str) -> str:
    # Beispiel: dop20rgbi_32_612_5800_2_ni_2019-04-01
    parts = kachelname.split("_")
    if len(parts) >= 5:
        parts[2] = new_east_3
        parts[3] = new_north_4
    return "_".join(parts)


def select_first_match(df: pd.DataFrame, kachelname: str) -> int:
    matches = df.index[df["kachelname"] == kachelname].tolist()
    if not matches:
        raise ValueError(f"Kachelname nicht gefunden: {kachelname}")
    return matches[0]


def update_record_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["record_key"] = build_record_key(df)
    return df


def inject_errors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    before = df.copy()
    before = update_record_key(before)

    work = df.copy()
    work = update_record_key(work)

    logs: list[dict] = []

    # ---------------------------------------------------------
    # 1) Standardabweichung falsch für DOP20 (20m Pixel -> sollte 40 sein)
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_612_5800_2_ni_2019-04-01")
    old = work.loc[idx, "standardabweichung"]
    work.loc[idx, "standardabweichung"] = "20"
    log_change(
        logs, before, work, idx, "standardabweichung", old, "20",
        "pixel_stddev_mismatch",
        "Für bodenpixelgroesse=20 absichtlich falsche standardabweichung gesetzt."
    )

    # ---------------------------------------------------------
    # 2) XML-Datum absichtlich falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_388_5866_2_ni_2025-03-08")
    old = work.loc[idx, "xml"]
    new = replace_first_date_in_text(str(old), "2024-03-08")
    work.loc[idx, "xml"] = new
    log_change(
        logs, before, work, idx, "xml", old, new,
        "xml_date_mismatch",
        "Datum im XML-Pfad stimmt absichtlich nicht mehr mit aktualitaet überein."
    )

    # ---------------------------------------------------------
    # 3) TIF-Datum absichtlich falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop10rgbi_32_418_5954_1_ni_2024-07-25")
    old = work.loc[idx, "tif"]
    new = replace_first_date_in_text(str(old), "2023-07-25")
    work.loc[idx, "tif"] = new
    log_change(
        logs, before, work, idx, "tif", old, new,
        "tif_date_mismatch",
        "Datum im TIF-Pfad stimmt absichtlich nicht mehr mit aktualitaet überein."
    )

    # ---------------------------------------------------------
    # 4) Kachelname-Datum absichtlich falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_470_5920_2_ni_2024-05-15")
    old = work.loc[idx, "kachelname"]
    new = replace_first_date_in_text(str(old), "2023-05-15")
    work.loc[idx, "kachelname"] = new
    log_change(
        logs, before, work, idx, "kachelname", old, new,
        "kachelname_date_mismatch",
        "Datum im Kachelnamen stimmt absichtlich nicht mehr mit aktualitaet überein."
    )

    # ---------------------------------------------------------
    # 5) East-Koordinate absichtlich falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_592_5838_2_ni_2025-03-09")
    old = work.loc[idx, "koordinatenursprung_east"]
    work.loc[idx, "koordinatenursprung_east"] = "591000"
    log_change(
        logs, before, work, idx, "koordinatenursprung_east", old, "591000",
        "east_coordinate_mismatch",
        "Ursprungskoordinate East absichtlich inkonsistent zur Pfad-/Kachelkennung gesetzt."
    )

    # ---------------------------------------------------------
    # 6) North-Koordinate absichtlich falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_410_5916_2_ni_2014-07-11")
    old = work.loc[idx, "koordinatenursprung_north"]
    work.loc[idx, "koordinatenursprung_north"] = "5915000"
    log_change(
        logs, before, work, idx, "koordinatenursprung_north", old, "5915000",
        "north_coordinate_mismatch",
        "Ursprungskoordinate North absichtlich inkonsistent zur Pfad-/Kachelkennung gesetzt."
    )

    # ---------------------------------------------------------
    # 7) Dateiformat untypisch/falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_612_5800_2_ni_2019-04-01")
    old = work.loc[idx, "dateiformat"]
    work.loc[idx, "dateiformat"] = "GeoTIFF"
    log_change(
        logs, before, work, idx, "dateiformat", old, "GeoTIFF",
        "unexpected_dateiformat",
        "Erwartetes Format absichtlich auf untypischen Wert gesetzt."
    )

    # ---------------------------------------------------------
    # 8) Spektralkanäle untypisch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_388_5866_2_ni_2025-03-08")
    old = work.loc[idx, "spektralkanaele"]
    work.loc[idx, "spektralkanaele"] = "RGB"
    log_change(
        logs, before, work, idx, "spektralkanaele", old, "RGB",
        "unexpected_spectral_channels",
        "Spektralkanäle absichtlich geändert."
    )

    # ---------------------------------------------------------
    # 9) CRS Lage falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop10rgbi_32_418_5954_1_ni_2024-07-25")
    old = work.loc[idx, "koordinatenreferenzsystem_lage"]
    work.loc[idx, "koordinatenreferenzsystem_lage"] = "4326"
    log_change(
        logs, before, work, idx, "koordinatenreferenzsystem_lage", old, "4326",
        "wrong_crs_lage",
        "CRS Lage absichtlich auf untypischen/falschen Wert gesetzt."
    )

    # ---------------------------------------------------------
    # 10) Kamera-Sensor mit untypischer Kombination
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_470_5920_2_ni_2024-05-15")
    old = work.loc[idx, "kamera_sensor"]
    work.loc[idx, "kamera_sensor"] = "Unknown Sensor Prototype X"
    log_change(
        logs, before, work, idx, "kamera_sensor", old, "Unknown Sensor Prototype X",
        "unexpected_camera_sensor",
        "Kamera-Sensor absichtlich auf sehr untypischen Wert gesetzt."
    )

    # ---------------------------------------------------------
    # 11) XML 9-stellige Prefix-Kennung falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_592_5838_2_ni_2025-03-09")
    old = work.loc[idx, "xml"]
    new = replace_first_nine_digit_prefix(str(old), "325915838")
    work.loc[idx, "xml"] = new
    log_change(
        logs, before, work, idx, "xml", old, new,
        "xml_prefix_coordinate_mismatch",
        "9-stellige Prefix-Kennung im XML-Pfad absichtlich geändert."
    )

    # ---------------------------------------------------------
    # 12) Kachelname-Koordinaten falsch
    # ---------------------------------------------------------
    idx = select_first_match(work, "dop20rgbi_32_410_5916_2_ni_2014-07-11")
    old = work.loc[idx, "kachelname"]
    new = replace_coords_in_kachelname(str(old), "411", "5917")
    work.loc[idx, "kachelname"] = new
    log_change(
        logs, before, work, idx, "kachelname", old, new,
        "kachelname_coordinate_mismatch",
        "East/North im Kachelnamen absichtlich inkonsistent gesetzt."
    )

    # Nach Änderungen record_key neu bauen
    work = update_record_key(work)

    return work.drop(columns=["record_key"]), logs


def save_outputs(df_out: pd.DataFrame, logs: list[dict], cfg: InjectionConfig) -> None:
    out_root = ensure_dir(cfg.output_dir)
    run_id = now_run_id()
    run_dir = ensure_dir(out_root / run_id)

    parquet_path = run_dir / f"{cfg.output_filename_prefix}_{run_id}.parquet"
    log_csv_path = run_dir / "injected_errors_log.csv"
    log_json_path = run_dir / "injected_errors_log.json"
    config_path = run_dir / "injection_config.json"
    summary_path = run_dir / "injection_summary.json"

    df_out.to_parquet(parquet_path, index=False)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_csv_path, index=False, encoding="utf-8")

    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2, default=str)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2, default=str)

    summary = {
        "run_id": run_id,
        "input_parquet": cfg.input_parquet,
        "output_parquet": str(parquet_path),
        "log_csv": str(log_csv_path),
        "log_json": str(log_json_path),
        "num_injected_changes": len(logs),
        "error_types": sorted(pd.DataFrame(logs)["error_type"].unique().tolist()) if logs else [],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("Fertig.")
    print(f"Manipulierte Datei: {parquet_path}")
    print(f"Fehler-Log CSV: {log_csv_path}")
    print(f"Fehler-Log JSON: {log_json_path}")
    print(f"Zusammenfassung: {summary_path}")


def main() -> None:
    print("Lade Originaldatei ...")
    df = load_data(CFG.input_parquet)
    print(f"Zeilen: {len(df)} | Spalten: {len(df.columns)}")

    print("Injiziere Testfehler ...")
    df_wrong, logs = inject_errors(df)

    print(f"Anzahl injizierter Änderungen: {len(logs)}")
    save_outputs(df_wrong, logs, CFG)


if __name__ == "__main__":
    main()