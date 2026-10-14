import numpy as np
import pandas as pd
from record_identity import IDENTITY_COLUMNS

def normalize_by_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()

    for col, t in schema.items():
        if col in IDENTITY_COLUMNS:
            continue

        if t == "boolean":
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})

        elif t == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif t == "categorical":
            df[col] = df[col].astype("string").astype(object).where(df[col].notna(), np.nan)

        elif t == "text":
            df[col] = df[col].astype("string")

        elif t == "json_struct":
            df[col] = df[col].apply(
                lambda x: x[0]
                if isinstance(x, (list, np.ndarray)) and len(x) > 0
                else None
            )

        elif t == "geometry_wkb":
            import shapely.wkb as wkb
            df[col + "_geom"] = df[col].apply(lambda x: wkb.loads(bytes.fromhex(x)) if isinstance(x,str) else None)

        else:
            df[col] = df[col].astype("string")

    return df
