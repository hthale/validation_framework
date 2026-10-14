from pathlib import Path
import pandas as pd


def load_data(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)