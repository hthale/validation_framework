from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import umap
import hdbscan

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =========================================================
# Konfiguration
# =========================================================
@dataclass
class LatentConfig:
    parquet_path: str = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\dopparwrong\injected_runs\20260423_143012\dop20-rgbi_injected_20260423_143012.parquet"
    output_root_dir: str = r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\output_latentspace\dop20-rgbi"

    random_state: int = 42

    known_numeric_columns: tuple[str, ...] = (
        "bildflugnummer",
        "farbtiefe",
        "belaubungszustand",
        "kompression",
        "komprimierung",
        "hintergrundwert",
        "quelldatenqualitaet",
        "koordinatenursprung_north",
        "koordinatenursprung_east",
        "koordinatenreferenzsystem_hoehe",
        "koordinatenreferenzsystem_lage",
        "anzahl_zeilen",
        "anzahl_spalten",
        "bodenpixelgroesse",
        "standardabweichung",
        "hintergrund",
        "erfassungsmethode",
    )

    known_date_columns: tuple[str, ...] = (
        "aktualitaet",
        "aktualitaet_kachelinformationen",
    )

    force_id_like_columns: tuple[str, ...] = (
        "xml",
        "tif",
        "kachelname",
    )

    ignore_columns: tuple[str, ...] = ()

    latent_dim: int = 32
    batch_size: int = 256
    epochs: int = 25
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.03

    lof_n_neighbors: int = 50
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: int = 20
    normal_neighbor_count: int = 5
    top_k_outliers: int = 250

    umap_n_neighbors: int = 20
    umap_min_dist: float = 0.1

    max_onehot_categories: int = 200
    text_avg_length_threshold: int = 50

    save_html_umap: bool = True


CFG = LatentConfig()


# =========================================================
# Allgemeine Hilfsfunktionen
# =========================================================

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_parquet(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def normalize_known_types(df: pd.DataFrame, cfg: LatentConfig) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*$", np.nan, regex=True)

    for col in cfg.known_numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in cfg.known_date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def is_path_like_series(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).head(200)
    if vals.empty:
        return False
    ratio = vals.str.contains(r"[\\/]|\.xml$|\.tif$|\.tiff$|\.json$|^\d{9}/", regex=True).mean()
    return bool(ratio >= 0.5)


def is_probably_text(s: pd.Series, avg_len_threshold: int = 50) -> bool:
    vals = s.dropna().astype(str)
    if vals.empty:
        return False
    avg_len = vals.str.len().mean()
    return bool(avg_len >= avg_len_threshold)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))

    if mad == 0:
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values, dtype=float)
        return (values - np.mean(values)) / std

    return 0.6745 * (values - med) / mad


def safe_rank_01(values: np.ndarray) -> np.ndarray:
    """
    Rangbasierte Skalierung auf [0, 1].
    Stabiler als z-score + clipping für stark schiefe Outlier-Verteilungen.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    ranks = s.rank(method="average", pct=True).to_numpy()
    return ranks


def robust_component(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """
    Robuster z-score nur noch für Report-Zwecke.
    """
    z = robust_zscore(values)
    return np.clip(z, -clip, clip)


# =========================================================
# Schema / Features
# =========================================================

def infer_schema(df: pd.DataFrame, cfg: LatentConfig) -> dict[str, list[str]]:
    numeric_cols: list[str] = []
    date_cols: list[str] = []
    categorical_cols: list[str] = []
    text_cols: list[str] = []
    id_like_cols: list[str] = []

    known_numeric = set(cfg.known_numeric_columns)
    known_date = set(cfg.known_date_columns)
    forced_id_like = set(cfg.force_id_like_columns)

    for col in df.columns:
        if col in cfg.ignore_columns:
            continue

        s = df[col]

        if col in known_date:
            date_cols.append(col)
            continue

        if col in known_numeric:
            numeric_cols.append(col)
            continue

        if col in forced_id_like:
            id_like_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        if is_path_like_series(s):
            id_like_cols.append(col)
            continue

        non_null = s.dropna()
        nunique = non_null.nunique()
        unique_ratio = float(nunique / len(non_null)) if len(non_null) > 0 else 0.0

        if unique_ratio > 0.95 and nunique > 100:
            id_like_cols.append(col)
            continue

        if nunique <= cfg.max_onehot_categories and not is_probably_text(s, cfg.text_avg_length_threshold):
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    return {
        "numeric": sorted(numeric_cols),
        "date": sorted(date_cols),
        "categorical": sorted(categorical_cols),
        "text": sorted(text_cols),
        "id_like": sorted(id_like_cols),
    }


def expand_date_features(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in date_cols:
        s = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}__year"] = s.dt.year
        df[f"{col}__month"] = s.dt.month
        df[f"{col}__day"] = s.dt.day
        df[f"{col}__weekday"] = s.dt.weekday
        df[f"{col}__dayofyear"] = s.dt.dayofyear
        df[f"{col}__is_month_start"] = s.dt.is_month_start.astype(float)
        df[f"{col}__is_month_end"] = s.dt.is_month_end.astype(float)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    schema: dict[str, list[str]],
) -> tuple[np.ndarray, ColumnTransformer, list[str], dict[str, tuple[int, int]], pd.DataFrame]:
    work_df = expand_date_features(df.copy(), schema["date"])

    date_feature_cols = []
    for dcol in schema["date"]:
        for suffix in [
            "__year",
            "__month",
            "__day",
            "__weekday",
            "__dayofyear",
            "__is_month_start",
            "__is_month_end",
        ]:
            candidate = f"{dcol}{suffix}"
            if candidate in work_df.columns:
                date_feature_cols.append(candidate)

    numeric_cols = list(schema["numeric"]) + date_feature_cols
    categorical_cols = list(schema["categorical"])

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(work_df)
    X = np.asarray(X, dtype=np.float32)
    feature_names = list(preprocessor.get_feature_names_out())

    num_count = len(preprocessor.named_transformers_["num"].get_feature_names_out(numeric_cols))
    cat_count = len(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols))

    group_slices = {
        "numeric_and_dates": (0, num_count),
        "categorical": (num_count, num_count + cat_count),
    }

    return X, preprocessor, feature_names, group_slices, work_df


# =========================================================
# Autoencoder
# =========================================================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.03):
        super().__init__()

        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(X: np.ndarray, cfg: LatentConfig) -> tuple[AutoEncoder, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cpu")

    model = AutoEncoder(
        input_dim=X.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss(reduction="none")

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch)

        epoch_loss /= len(dataset)
        print(f"[AE] Epoch {epoch + 1:02d}/{cfg.epochs} - loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        X_recon_t, Z_t = model(X_t)

    X_recon = X_recon_t.cpu().numpy()
    Z = Z_t.cpu().numpy()
    reconstruction_error = np.mean((X - X_recon) ** 2, axis=1)

    return model, Z, X_recon, reconstruction_error


# =========================================================
# Outlier-Komponenten
# =========================================================

def compute_lof_raw(Z: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 1e-6, size=Z.shape).astype(np.float32)
    Z_safe = Z + jitter

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric="euclidean",
        novelty=False,
    )
    lof.fit_predict(Z_safe)
    raw = -lof.negative_outlier_factor_
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def compute_hdbscan_raw(
    Z: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )

    labels = clusterer.fit_predict(Z)
    outlier_scores = getattr(clusterer, "outlier_scores_", np.zeros(len(Z), dtype=float))
    outlier_scores = np.nan_to_num(outlier_scores, nan=0.0, posinf=0.0, neginf=0.0)

    probabilities = getattr(clusterer, "probabilities_", np.ones(len(Z), dtype=float))
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    return labels, outlier_scores, probabilities


def compute_centroid_distance_raw(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(Z), dtype=float)
    valid_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    for lbl in valid_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 3:
            continue

        cluster_points = Z[idx]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        scores[idx] = dists

    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        non_noise = scores[labels != -1]
        fallback = np.percentile(non_noise, 95) if len(non_noise) > 0 else 1.0
        scores[noise_idx] = np.maximum(scores[noise_idx], fallback)

    return scores


def combine_scores(
    reconstruction_error: np.ndarray,
    lof_raw: np.ndarray,
    hdbscan_raw: np.ndarray,
    centroid_raw: np.ndarray,
) -> pd.DataFrame:
    reconstruction_z = robust_component(reconstruction_error)
    lof_z = robust_component(lof_raw)
    hdbscan_z = robust_component(hdbscan_raw)
    centroid_z = robust_component(centroid_raw)

    reconstruction_01 = safe_rank_01(reconstruction_error)
    lof_01 = safe_rank_01(lof_raw)
    hdbscan_01 = safe_rank_01(hdbscan_raw)
    centroid_01 = safe_rank_01(centroid_raw)

    combined = (
        0.45 * reconstruction_01 +
        0.20 * lof_01 +
        0.15 * hdbscan_01 +
        0.20 * centroid_01
    )

    return pd.DataFrame({
        "reconstruction_z": reconstruction_z,
        "lof_z": lof_z,
        "hdbscan_outlier_z": hdbscan_z,
        "centroid_distance_z": centroid_z,
        "reconstruction_01": reconstruction_01,
        "lof_01": lof_01,
        "hdbscan_outlier_01": hdbscan_01,
        "centroid_distance_01": centroid_01,
        "combined_outlier_score": combined,
    })


# =========================================================
# Erklärbarkeit / Nachbarn / UMAP
# =========================================================

def compute_group_reconstruction_contributions(
    X: np.ndarray,
    X_recon: np.ndarray,
    group_slices: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    sqerr = (X - X_recon) ** 2
    out: dict[str, np.ndarray] = {}

    for group_name, (start, end) in group_slices.items():
        if end <= start:
            out[group_name] = np.zeros(X.shape[0], dtype=float)
        else:
            out[group_name] = sqerr[:, start:end].mean(axis=1)

    return out


def project_umap(Z: np.ndarray, cfg: LatentConfig) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state,
        init="random",
    )
    return reducer.fit_transform(Z)


def find_nearest_normal_neighbors(
    Z: np.ndarray,
    combined_score: np.ndarray,
    top_idx: np.ndarray,
    k: int,
) -> dict[int, list[int]]:
    normal_mask = combined_score <= np.quantile(combined_score, 0.5)
    normal_idx = np.where(normal_mask)[0]

    if len(normal_idx) == 0:
        return {int(i): [] for i in top_idx}

    nn = NearestNeighbors(
        n_neighbors=min(k, len(normal_idx)),
        metric="euclidean",
    )
    nn.fit(Z[normal_idx])

    _, indices = nn.kneighbors(Z[top_idx])

    result: dict[int, list[int]] = {}
    for ridx, neigh in zip(top_idx, indices):
        result[int(ridx)] = [int(normal_idx[j]) for j in neigh]

    return result


def save_umap_png(results_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        results_df["umap_x"],
        results_df["umap_y"],
        c=results_df["combined_outlier_score"],
        s=8,
        alpha=0.8,
    )
    plt.colorbar(scatter, label="combined_outlier_score")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP 2D des Autoencoder-Latent-Space")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_umap_html(results_df: pd.DataFrame, output_path: Path) -> None:
    if not PLOTLY_AVAILABLE:
        return

    hover_cols = [
        c for c in [
            "record_key",
            "row_id",
            "combined_outlier_score",
            "cluster_label",
            "kachelname",
            "aktualitaet",
            "bildflugname",
            "kamera_sensor",
        ] if c in results_df.columns
    ]

    fig = px.scatter(
        results_df,
        x="umap_x",
        y="umap_y",
        color="combined_outlier_score",
        hover_data=hover_cols,
        title="Interaktiver UMAP-Plot des Latent Space",
    )
    fig.write_html(str(output_path))


# =========================================================
# Versionsverwaltung / Vergleich
# =========================================================

def build_record_key(df: pd.DataFrame) -> pd.Series:
    preferred_keys = ["kachelname", "xml", "tif"]

    available = [c for c in preferred_keys if c in df.columns]
    if available:
        return df[available].astype("string").fillna("").agg("||".join, axis=1)

    # Fallback: Hash über ganze Zeile
    hashed = pd.util.hash_pandas_object(
        df.astype("string").fillna(""),
        index=False,
    ).astype(str)

    return hashed


def get_existing_run_dirs(output_root: Path) -> list[Path]:
    return sorted([p for p in output_root.iterdir() if p.is_dir()])


def get_previous_run_dir(output_root: Path, current_run_dir: Path) -> Path | None:
    runs = get_existing_run_dirs(output_root)
    previous = [r for r in runs if r != current_run_dir]
    if not previous:
        return None
    return previous[-1]


def summarize_scores(df: pd.DataFrame) -> dict[str, Any]:
    s = df["combined_outlier_score"]
    return {
        "row_count": int(len(df)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "noise_count": int((df["cluster_label"] == -1).sum()) if "cluster_label" in df.columns else None,
    }


def compare_schema(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key in ["numeric", "date", "categorical", "text", "id_like"]:
        old_set = set(old_schema.get("full_schema", {}).get(key, []))
        new_set = set(new_schema.get("full_schema", {}).get(key, []))
        out[key] = {
            "added": sorted(list(new_set - old_set)),
            "removed": sorted(list(old_set - new_set)),
            "unchanged_count": len(old_set & new_set),
        }

    return out


def compare_runs(
    previous_run_dir: Path,
    current_run_dir: Path,
    top_k: int = 250,
) -> dict[str, Any]:
    prev_results_path = previous_run_dir / "latent_outlier_results.parquet"
    curr_results_path = current_run_dir / "latent_outlier_results.parquet"
    prev_schema_path = previous_run_dir / "schema_summary.json"
    curr_schema_path = current_run_dir / "schema_summary.json"

    if not prev_results_path.exists() or not curr_results_path.exists():
        return {
            "comparison_available": False,
            "reason": "missing_previous_or_current_results",
        }

    prev_df = pd.read_parquet(prev_results_path)
    curr_df = pd.read_parquet(curr_results_path)

    with open(prev_schema_path, "r", encoding="utf-8") as f:
        prev_schema = json.load(f)
    with open(curr_schema_path, "r", encoding="utf-8") as f:
        curr_schema = json.load(f)

    prev_top = prev_df.nlargest(top_k, "combined_outlier_score")[["record_key", "combined_outlier_score"]].copy()
    curr_top = curr_df.nlargest(top_k, "combined_outlier_score")[["record_key", "combined_outlier_score"]].copy()

    prev_top_keys = set(prev_top["record_key"])
    curr_top_keys = set(curr_top["record_key"])

    intersection = prev_top_keys & curr_top_keys
    only_prev = prev_top_keys - curr_top_keys
    only_curr = curr_top_keys - prev_top_keys

    overlap_ratio = len(intersection) / top_k if top_k > 0 else 0.0

    prev_score_summary = summarize_scores(prev_df)
    curr_score_summary = summarize_scores(curr_df)

    score_delta = {
        k: (
            None
            if prev_score_summary.get(k) is None or curr_score_summary.get(k) is None
            else curr_score_summary[k] - prev_score_summary[k]
        )
        for k in curr_score_summary.keys()
    }

    joined = curr_top.merge(
        prev_top,
        on="record_key",
        how="outer",
        suffixes=("_current", "_previous"),
    )

    joined["status"] = np.select(
        [
            joined["combined_outlier_score_previous"].isna(),
            joined["combined_outlier_score_current"].isna(),
        ],
        [
            "new_in_top_k",
            "dropped_from_top_k",
        ],
        default="still_in_top_k",
    )

    joined["score_change"] = joined["combined_outlier_score_current"] - joined["combined_outlier_score_previous"]
    joined = joined.sort_values(
        by=["status", "combined_outlier_score_current"],
        ascending=[True, False],
        na_position="last",
    )

    diff_csv_path = current_run_dir / "difference_to_previous_topk.csv"
    joined.to_csv(diff_csv_path, index=False, encoding="utf-8")

    comparison = {
        "comparison_available": True,
        "previous_run_dir": str(previous_run_dir),
        "current_run_dir": str(current_run_dir),
        "top_k_compared": int(top_k),
        "score_summary_previous": prev_score_summary,
        "score_summary_current": curr_score_summary,
        "score_summary_delta": score_delta,
        "schema_difference": compare_schema(prev_schema, curr_schema),
        "top_k_overlap": {
            "intersection_count": int(len(intersection)),
            "only_previous_count": int(len(only_prev)),
            "only_current_count": int(len(only_curr)),
            "overlap_ratio": float(overlap_ratio),
        },
        "difference_table_csv": str(diff_csv_path),
    }

    return comparison


def write_latest_pointer(output_root: Path, current_run_dir: Path) -> None:
    latest_info = {
        "latest_run_dir": str(current_run_dir),
        "latest_run_name": current_run_dir.name,
        "written_at": datetime.now().isoformat(),
    }
    with open(output_root / "latest_run.json", "w", encoding="utf-8") as f:
        json.dump(latest_info, f, ensure_ascii=False, indent=2)

def save_grouped_outlier_blocks(results: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Fasst ähnliche Outlier zu Blöcken zusammen.
    Für DOP-Daten: Datum + Bildflugname + Kamera + Pixelgröße + Standardabweichung.
    """
    group_cols = [
        c for c in [
            "aktualitaet",
            "bildflugname",
            "kamera_sensor",
            "bodenpixelgroesse",
            "standardabweichung",
            "cluster_label",
            "dominant_group",
        ] if c in results.columns
    ]

    if not group_cols:
        grouped = pd.DataFrame({
            "note": ["no_group_columns_available"]
        })
        grouped.to_csv(out_path, index=False, encoding="utf-8")
        return grouped

    grouped = (
        results.groupby(group_cols, dropna=False)
        .agg(
            count=("row_id", "size"),
            mean_score=("combined_outlier_score", "mean"),
            max_score=("combined_outlier_score", "max"),
            min_east=("koordinatenursprung_east", "min") if "koordinatenursprung_east" in results.columns else ("row_id", "min"),
            max_east=("koordinatenursprung_east", "max") if "koordinatenursprung_east" in results.columns else ("row_id", "max"),
            min_north=("koordinatenursprung_north", "min") if "koordinatenursprung_north" in results.columns else ("row_id", "min"),
            max_north=("koordinatenursprung_north", "max") if "koordinatenursprung_north" in results.columns else ("row_id", "max"),
        )
        .reset_index()
        .sort_values(["max_score", "count"], ascending=[False, False])
    )

    grouped.to_csv(out_path, index=False, encoding="utf-8")
    return grouped

# =========================================================
# Main
# =========================================================

def main() -> None:
    set_seeds(CFG.random_state)

    output_root = ensure_dir(CFG.output_root_dir)
    run_id = now_run_id()
    run_dir = ensure_dir(output_root / run_id)

    print("Lade Parquet ...")
    df = load_parquet(CFG.parquet_path)
    df = normalize_known_types(df, CFG)

    print(f"Zeilen: {len(df)} | Spalten: {len(df.columns)}")

    schema = infer_schema(df, CFG)
    print("Schema:")
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    used_schema = {
        "numeric": schema["numeric"],
        "date": schema["date"],
        "categorical": schema["categorical"],
        "text": [],
        "id_like": schema["id_like"],
    }

    print("Baue Features ...")
    X, preprocessor, feature_names, group_slices, work_df = build_feature_matrix(df, used_schema)
    print(f"Feature-Matrix: {X.shape}")

    print("Trainiere Autoencoder ...")
    model, Z, X_recon, reconstruction_error = train_autoencoder(X, CFG)

    print("Berechne Outlier-Komponenten ...")
    lof_raw = compute_lof_raw(Z, CFG.lof_n_neighbors, CFG.random_state)
    labels, hdbscan_raw, cluster_prob = compute_hdbscan_raw(
        Z,
        CFG.hdbscan_min_cluster_size,
        CFG.hdbscan_min_samples,
    )
    centroid_raw = compute_centroid_distance_raw(Z, labels)

    score_df = combine_scores(
        reconstruction_error=reconstruction_error,
        lof_raw=lof_raw,
        hdbscan_raw=hdbscan_raw,
        centroid_raw=centroid_raw,
    )

    print("Projiziere UMAP 2D ...")
    Z_2d = project_umap(Z, CFG)

    print("Berechne Nachbarn und Gruppenbeiträge ...")
    group_contribs = compute_group_reconstruction_contributions(X, X_recon, group_slices)

    results = pd.DataFrame(index=df.index)
    results["row_id"] = np.arange(len(df))
    results["record_key"] = build_record_key(df)
    results["reconstruction_error"] = reconstruction_error
    results["cluster_label"] = labels
    results["cluster_probability"] = cluster_prob
    results["umap_x"] = Z_2d[:, 0]
    results["umap_y"] = Z_2d[:, 1]

    for col in score_df.columns:
        results[col] = score_df[col].values

    for i in range(Z.shape[1]):
        results[f"latent_{i:02d}"] = Z[:, i]

    for group_name, values in group_contribs.items():
        results[f"group_contrib__{group_name}"] = values

    contrib_cols = [c for c in results.columns if c.startswith("group_contrib__")]
    results["dominant_group"] = results[contrib_cols].idxmax(axis=1).str.replace("group_contrib__", "", regex=False)

    preview_cols = [
        c for c in [
            "kachelname",
            "xml",
            "tif",
            "aktualitaet",
            "bildflugname",
            "kamera_sensor",
            "bodenpixelgroesse",
            "standardabweichung",
            "koordinatenursprung_east",
            "koordinatenursprung_north",
        ] if c in df.columns
    ]
    if preview_cols:
        results = results.join(df[preview_cols], how="left")

    top_idx = results["combined_outlier_score"].nlargest(CFG.top_k_outliers).index.to_numpy()
    neighbor_map = find_nearest_normal_neighbors(
        Z=Z,
        combined_score=results["combined_outlier_score"].to_numpy(),
        top_idx=top_idx,
        k=CFG.normal_neighbor_count,
    )

    top_outliers = results.loc[top_idx].copy().sort_values("combined_outlier_score", ascending=False)
    top_outliers["normal_neighbors"] = [neighbor_map[int(i)] for i in top_outliers.index]

    print("Speichere Ergebnisse ...")
    results_path = run_dir / "latent_outlier_results.parquet"
    top_path = run_dir / "top_outliers.csv"
    umap_csv_path = run_dir / "umap_points.csv"
    schema_path = run_dir / "schema_summary.json"
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "run_summary.json"
    png_path = run_dir / "umap_plot.png"
    html_path = run_dir / "umap_plot.html"
    grouped_blocks_path = run_dir / "grouped_outlier_blocks.csv"
    grouped_blocks = save_grouped_outlier_blocks(results, grouped_blocks_path)

    results.to_parquet(results_path, index=False)
    top_outliers.to_csv(top_path, index=False, encoding="utf-8")
    results[[
        "record_key",
        "row_id",
        "umap_x",
        "umap_y",
        "combined_outlier_score",
        "cluster_label",
        "cluster_probability",
        "dominant_group",
    ] + [c for c in preview_cols if c in results.columns]].to_csv(
        umap_csv_path,
        index=False,
        encoding="utf-8",
    )

    schema_summary = {
        "full_schema": schema,
        "used_schema": used_schema,
        "feature_matrix_shape": list(X.shape),
        "latent_dim": CFG.latent_dim,
        "group_slices": {k: [int(v[0]), int(v[1])] for k, v in group_slices.items()},
        "preview_columns": preview_cols,
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_summary, f, ensure_ascii=False, indent=2, default=str)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2, default=str)

    run_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_file": CFG.parquet_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_matrix_shape": list(X.shape),
        "score_summary": summarize_scores(results),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2, default=str)

    save_umap_png(results, png_path)
    if CFG.save_html_umap:
        save_umap_html(results, html_path)
    

    previous_run_dir = get_previous_run_dir(output_root, run_dir)
    comparison_path = run_dir / "difference_to_previous_summary.json"

    if previous_run_dir is not None:
        comparison = compare_runs(
            previous_run_dir=previous_run_dir,
            current_run_dir=run_dir,
            top_k=CFG.top_k_outliers,
        )
    else:
        comparison = {
            "comparison_available": False,
            "reason": "no_previous_run_found",
        }

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)

    write_latest_pointer(output_root, run_dir)

    print("\nFertig.")
    print(f"Run-Ordner: {run_dir}")
    print(f"Ergebnisse: {results_path}")
    print(f"Top-Outlier: {top_path}")
    print(f"Schema: {schema_path}")
    print(f"Run-Zusammenfassung: {summary_path}")
    print(f"Vergleich zur vorherigen Datei: {comparison_path}")
    print(f"UMAP CSV: {umap_csv_path}")
    print(f"UMAP Plot PNG: {png_path}")
    if CFG.save_html_umap and PLOTLY_AVAILABLE:
        print(f"UMAP Plot HTML: {html_path}")
    print(f"Gruppierte Outlier-Blöcke: {grouped_blocks_path}")

    show_cols = [
        "row_id",
        "record_key",
        "combined_outlier_score",
        "reconstruction_error",
        "reconstruction_z",
        "lof_z",
        "hdbscan_outlier_z",
        "centroid_distance_z",
        "cluster_label",
        "dominant_group",
    ] + preview_cols

    print("\nTop 10 Outlier:")
    print(top_outliers[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()