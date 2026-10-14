import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from latentspace.schema_interference import infer_schema
from preprocessing import normalize_by_schema


# -------------------------------------------------------------
# Hilfsfunktion: Erweiterte Datum-Features (nur wenn erkannt)
# -------------------------------------------------------------
def expand_date_features(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in date_cols:
        s = pd.to_datetime(df[col], errors="coerce", format="mixed")
        df[f"{col}__year"] = s.dt.year
        df[f"{col}__month"] = s.dt.month
        df[f"{col}__day"] = s.dt.day
        df[f"{col}__weekday"] = s.dt.weekday
        df[f"{col}__dayofyear"] = s.dt.dayofyear
    return df


# -------------------------------------------------------------
# Hauptfunktion: automatische Feature-Pipeline
# -------------------------------------------------------------
def prepare_tabular_matrix(df: pd.DataFrame):

    # ------------------------
    # 1. Schema automatisch erkennen
    # ------------------------
    print("Erkenne Schema...")
    schema = infer_schema(df)
    print("Erkanntes Schema (Auszug):")
    for c, t in list(schema.items())[:10]:
        print(f"  {c}: {t}")
    if len(schema) > 10:
        print(f"  ... ({len(schema)} Spalten gesamt)")

    # ------------------------
    # 2. Normalisieren nach Schema
    # ------------------------
    df = normalize_by_schema(df, schema)

    # Liste der Spalten nach Typen
    numeric_cols = [c for c, t in schema.items() if t == "numeric"]
    categorical_cols = [c for c, t in schema.items() if t == "categorical"]
    text_cols = [c for c, t in schema.items() if t == "text"]
    geometry_cols = [c for c in df.columns if c.endswith("_geom")]

    # ------------------------
    # 3. Datumsspalten erkennen (über schema)
    # → jedes text/string-Feld, das wie ein Datum aussieht, wird erweitert
    # ------------------------
    detected_date_cols = []
    for col, t in schema.items():
        if t == "text":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9:  # 90%+ gültige Datumswerte
                    detected_date_cols.append(col)
            except:
                pass

    # Datum-Features erweitern
    df = expand_date_features(df, detected_date_cols)

    # abgeleitete Datumsspalten hinzufügen
    date_derived_cols = [
        c for c in df.columns if "__" in c and any(c.startswith(f"{d}__") for d in detected_date_cols)
    ]
    numeric_cols = numeric_cols + date_derived_cols

    # ------------------------
    # 4. Geometrie-Features erzeugen
    # ------------------------
    for gc in geometry_cols:
        df[gc + "_area"] = df[gc].apply(lambda g: g.area if g else None)
        df[gc + "_perimeter"] = df[gc].apply(lambda g: g.length if g else None)
        df[gc + "_compactness"] = df[gc].apply(
            lambda g: (4 * np.pi * g.area) / (g.length ** 2)
            if g and g.length > 0 else None
        )
        df[gc + "_vertices"] = df[gc].apply(
            lambda g: len(g.exterior.coords) if hasattr(g, "exterior") else None
        )

    # Geometrie-Feature-Spalten zu numerischen hinzufügen
    geom_feature_cols = []
    for gc in geometry_cols:
        geom_feature_cols += [
            gc + "_area",
            gc + "_perimeter",
            gc + "_compactness",
            gc + "_vertices",
        ]
    numeric_cols = numeric_cols + geom_feature_cols

    # ------------------------
    # 5. Sicherstellen, dass alle verwendeten Spalten existieren
    # ------------------------
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    print(f"Tabellarische Features (numerisch): {len(numeric_cols)}")
    print(f"Tabellarische Features (kategorisch): {len(categorical_cols)}")
    print(f"Textfeatures: {len(text_cols)}")
    print(f"Geometriefeatures: {len(geometry_cols)}")

    # ------------------------
    # 6. ColumnTransformer bauen
    # ------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    # ------------------------
    # 7. Fit/Transform tabellarische Matrix
    # ------------------------
    X_tab = preprocessor.fit_transform(df)
    X_tab = np.asarray(X_tab, dtype=np.float32)

    # ------------------------
    # 8. Rückgabe
    # ------------------------
    return {
        "X_tab": X_tab,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "text_cols": text_cols,
        "geometry_cols": geometry_cols,
        "date_cols": detected_date_cols,
    }
