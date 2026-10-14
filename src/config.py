"""Regelpipeline für den geprüften DOP-Bestand und seine LOF-Rangliste.

PHASE0_RESULTS_PARQUET verweist bewusst auf den vorhandenen Lauf vom
21.09.2026 nach Umstellung durch latentspace.rerank_lof.
"""
from pathlib import Path

from phase2_settings import Phase2Settings
from record_identity import IDENTITY_COLUMNS
from latentspace.anomaly_scores import SCORE_COLUMNS

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = FRAMEWORK_ROOT / "data" / "check_01" / "dop_training_dataset_with_ids.parquet"
PHASE0_RESULTS_PARQUET = (
    FRAMEWORK_ROOT / "data" / "output_latentspace_v2" / "dop_check_01"
    / "20260921_201231" / "lof_primary" / "latent_outlier_results_lof.parquet"
)
OUTPUT_DIR = FRAMEWORK_ROOT / "data" / "output_phase1_phase2_dop_check_03"
USE_PHASE0 = True

INTERNAL_COLUMNS = sorted(set(IDENTITY_COLUMNS) | set(SCORE_COLUMNS) | {
    "stratum_product_type", "stratum_pixel_size", "stratum_year", "stratum_key",
    "stratum_latent_cluster", "stratum_outlier", "geometry", "wkb_geometry",
    "geometry_parsed", "geometry_parsed_geom",
})

# Phase 1 beschreibt den vorhandenen Bestand; Phase 2 verwendet eine eigene Aufteilung.
HIGH_CARDINALITY_THRESHOLD = 20
CONST_STDDEV_RANGE_THRESHOLD = 0.05
MAX_NUMERIC_DISTINCT_FOR_IN_RULE = 10
SAMPLE_FRACTION = 0.2
SAMPLE_RUNS = 3
BASE_RANDOM_STATE = 42
MIN_RULE_OCCURRENCES = 2
MAX_EXAMPLE_VIOLATIONS = 20
ENABLE_STRING_RULES = True
MIN_REGEX_SUPPORT_RATIO = 0.95
ENABLE_STRING_LENGTH_RULES = True
PHASE2_BACKEND = "decision_tree"
USE_LLM_SUGGESTIONS = False
PHASE2_ACCURACY_THRESHOLD = 0.95
PHASE2_F1_THRESHOLD = 0.90
PHASE2_PRECISION_THRESHOLD = 0.90

PHASE2_SETTINGS = Phase2Settings()
PHASE2_SPLIT_PATH = FRAMEWORK_ROOT / "data" / "splits" / "dop_check_01_phase2_v1.csv"
