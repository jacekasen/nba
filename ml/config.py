"""Configuration for the NBA modeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELING_DIR = DATA_DIR / "05-modeling"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures" / "modeling"

RAW_STATS_PATH = DATA_DIR / "nba_player_stats.csv"
PLAYER_SEASONS_PATH = MODELING_DIR / "player_seasons.csv"
MODEL_FEATURES_PATH = MODELING_DIR / "model_features.csv"
CURRENT_PREDICTIONS_PATH = MODELING_DIR / "current_player_predictions.csv"

METRICS_PATH = MODELS_DIR / "metrics.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "trajectory_classifier.joblib"
REGRESSOR_MODEL_PATH = MODELS_DIR / "bpm_delta_regressor.joblib"
PEAK_MODEL_PATH = MODELS_DIR / "near_peak_classifier.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
LABEL_ORDER_PATH = MODELS_DIR / "label_order.json"
CONFUSION_MATRIX_CSV_PATH = MODELS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = MODELS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

TRAJECTORY_LABELS = ["improving", "stable", "regressing"]
TRAJECTORY_FEATURE_COLUMNS = [
    "age",
    "experience",
    "qualified_seasons_to_date",
    "season_start",
    "season_gap_from_prev",
    "is_consecutive_from_prev",
    "changed_team",
    "games",
    "mp",
    "mpg",
    "per",
    "bpm",
    "vorp",
    "ws",
    "ws_per_48",
    "bpm_delta_1",
    "per_delta_1",
    "ws_per_48_delta_1",
    "games_delta_1",
    "mp_delta_1",
    "mpg_delta_1",
    "bpm_delta_2",
    "bpm_roll2",
    "bpm_roll3",
    "per_roll2",
    "per_roll3",
    "ws_per_48_roll2",
    "ws_per_48_roll3",
    "bpm_slope_3",
    "prior_career_high_bpm",
    "career_high_bpm_through_t",
    "distance_from_career_high_bpm",
]


@dataclass(frozen=True)
class ModelingThresholds:
    """Filters to define qualified player seasons for modeling."""

    min_games: int = 15
    min_minutes: int = 250
    require_non_null_metrics: tuple[str, ...] = ("bpm", "per", "ws", "ws_per_48")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature and target settings for the modeling table."""

    trajectory_threshold: float = 0.5


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split boundaries for train/validation/test."""

    train_end: int = 2018
    val_start: int = 2019
    val_end: int = 2022
    test_start: int = 2023


@dataclass(frozen=True)
class PeakModelConfig:
    """Settings for near-peak probability model."""

    completion_gap_years: int = 5
    smoothing_window: int = 3
    near_peak_tolerance: float = 0.5


RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"
