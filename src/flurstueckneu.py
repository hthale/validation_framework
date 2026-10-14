import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\parquetdata\geodata-frame-ax_flurstueck-0.parquet"
OUTPUT_PATH = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\parquetdata\geodata-frame-ax_flurstueck-0_synthetic.parquet"

INJECT_FULLY_WRONG_ROW = True   # auf False setzen, um ohne Fehlerzeile zu speichern


def add_synthetic_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    amtlicheflaeche = (
        pd.to_numeric(df["amtlicheflaeche"], errors="coerce")
        if "amtlicheflaeche" in df.columns
        else None
    )

    zaehler = (
        pd.to_numeric(df["zaehler"], errors="coerce")
        if "zaehler" in df.columns
        else None
    )

    nenner = (
        pd.to_numeric(df["nenner"], errors="coerce")
        if "nenner" in df.columns
        else None
    )

    if amtlicheflaeche is not None:
        df["flaechenklasse"] = (amtlicheflaeche > 500).astype(int)

    if zaehler is not None and nenner is not None:
        ratio = zaehler / nenner.replace(0, np.nan)
        df["dichteklasse"] = (ratio > 1).fillna(False).astype(int)

    if amtlicheflaeche is not None and zaehler is not None and nenner is not None:
        ratio = zaehler / nenner.replace(0, np.nan)
        df["komplexklasse"] = (
            (amtlicheflaeche > 500) & (ratio > 1)
        ).fillna(False).astype(int)

    return df


def add_fully_wrong_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    wrong_row = {}

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_integer_dtype(s):
            wrong_row[col] = -999999

        elif pd.api.types.is_float_dtype(s):
            wrong_row[col] = -999999.0

        elif pd.api.types.is_bool_dtype(s):
            wrong_row[col] = True

        elif pd.api.types.is_datetime64_any_dtype(s):
            wrong_row[col] = pd.Timestamp("1800-01-01", tz="UTC")

        else:
            wrong_row[col] = None

    # Nur Werte setzen, die zum Spaltentyp passen.
    safe_values = {
        "gml_id": "SYNTHETIC_FULLY_WRONG_ROW_0001",
        "anlass": ["999999"],
        "beginnt": "1900-01-01T00:00:00Z",
        "endet": None,
        "advstandardmodell": ["WRONG_MODEL"],
        "abweichenderrechtszustand": "maybe",
        "amtlicheflaeche": -999999,
        "flurnummer": -999,
        "flurstueckskennzeichen": "WRONG_FLURSTUECKS_KENNZEICHEN",
        "nenner": "0",
        "zaehler": "-999",
        "gemarkungsnummer": "XXXX",
        "land": "99",
        "gemeindezugehoerigkeit_gemeinde": "999",
        "gemeindezugehoerigkeit_kreis": "999",
        "gemeindezugehoerigkeit_land": "99",
        "gemeindezugehoerigkeit_regierungsbezirk": "9",
        "rechtsbehelfsverfahren": "maybe",
        "zustaendigestelle_land": ["99"],
        "zustaendigestelle_stelle": ["9999"],
        "zweifelhafterflurstuecksnachweis": "maybe",
        "istgebucht": "WRONG_BOOKING_ID",
        "zeigtauf": ["WRONG_REFERENCE_ID"],
        "weistauf": ["WRONG_REFERENCE_ID"],
        "wkb_geometry": "INVALID_WKB_GEOMETRY",
        "objektkoordinaten": None,
        "ALKIS_TABLE_NAME": "wrong_table",
        "geometry": None,
        "flaechenklasse": 9,
        "dichteklasse": 9,
        "komplexklasse": 9,
    }

    for col, value in safe_values.items():
        if col in wrong_row:
            wrong_row[col] = value

    wrong_df = pd.DataFrame([wrong_row], columns=df.columns)

    # Wichtig: Datentypen der Originaldatei wieder angleichen
    for col in df.columns:
        try:
            wrong_df[col] = wrong_df[col].astype(df[col].dtype)
        except Exception:
            pass

    return pd.concat([df, wrong_df], ignore_index=True)

def main():
    print("Lade Datei...")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Original Shape: {df.shape}")

    df_new = add_synthetic_columns(df)

    if INJECT_FULLY_WRONG_ROW:
        df_new = add_fully_wrong_row(df_new)
        print("Vollständig falsche synthetische Zeile wurde eingefügt.")
    else:
        print("Keine vollständig falsche Zeile eingefügt.")

    print(f"Neue Shape: {df_new.shape}")

    print("Neue Spalten:")
    new_cols = set(df_new.columns) - set(df.columns)
    print(list(new_cols))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_new.to_parquet(OUTPUT_PATH, index=False)

    print("\nFertig.")
    print(f"Gespeichert unter:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()