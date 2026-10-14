"""Konfiguration für den technischen DOP-Durchlauf nach Korrekturpaket 02.

Diese Datei vollständig nach <Projekt>/src/latentspace/config_latent.py kopieren.
Die Datenpfade werden aus dieser Position abgeleitet, unabhängig vom Terminalpfad.
Der Lauf verwendet die geprüfte Parquet-Kopie mit IDs und schreibt keine Historie.

Trainingsparameter und wirksame Feature-Einstellungen entsprechen weiterhin dem
bisherigen Prototyp. Dieses Update korrigiert die Laufkonfiguration; die fachliche
Feature-Auswahl, Datumsverarbeitung folgen separat. Die Ergebnisrangliste verwendet jetzt ausschließlich LOF.
"""
from dataclasses import dataclass
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LatentV2Config:
    # Eingabekopie aus check_foundation.py; der vorhandene record_key bleibt erhalten.
    parquet_path: str = str(
        FRAMEWORK_ROOT / "data" / "check_01" / "dop_training_dataset_with_ids.parquet"
    )
    output_root_dir: str = str(
        FRAMEWORK_ROOT / "data" / "output_latentspace_v2" / "dop_check_02"
    )

    # Im technischen Prüflauf keine unbestätigten Ergebnisse als Historie speichern.
    enable_delivery_memory: bool = False
    delivery_group_id: str = "dop_training_001"
    delivery_root_dir: str = ""
    delivery_memory_dir: str = str(FRAMEWORK_ROOT / "data" / "delivery_memory_tabular")

    # Jede Einstellung wird genau einmal definiert.
    # Die leere Datumsliste entspricht dem zuvor tatsächlich wirksamen Wert.
    known_numeric_columns: tuple[str, ...] = ()
    known_date_columns: tuple[str, ...] = ()
    string_columns: tuple[str, ...] = (
        "kachelname",
        "xml_name",
        "tif_name",
        "bildflugname",
        "kamera_sensor",
        "spektralkanaele",
        "dateiformat",
        "komprimierung",
    )
    ignore_columns: tuple[str, ...] = (
        "geometry",
        "__index_level_0__",
    )

    # Modell und Training
    latent_dim: int = 64
    random_state: int = 42
    batch_size: int = 256
    epochs: int = 25
    learning_rate: float = 1e-3
    tab_hidden_dims: tuple[int, ...] = (256, 128)
    fusion_hidden_dims: tuple[int, ...] = (128,)
    dropout: float = 0.05

    # Zeichenbasierte Eingaben
    max_seq_len: int = 96
    char_emb_dim: int = 16
    string_emb_dim: int = 32

    # Anomalieerkennung und Clustering
    lof_n_neighbors: int = 20
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5

    # UMAP-Darstellung
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Gewichte der Trainingsziele
    weight_tab_recon: float = 1.0
    weight_str_recon: float = 0.3
    weight_contrastive: float = 0.1
    weight_alignment: float = 0.05

    # Trainingsaugmentation
    tab_mask_prob: float = 0.15
    tab_jitter_std: float = 0.02
    str_mask_prob: float = 0.10
    contrastive_temperature: float = 0.2
