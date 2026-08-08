from __future__ import annotations

import pandas as pd

from ml.config import TRAJECTORY_FEATURE_COLUMNS
from ml.features import build_features, classify_trajectory


def _sample_player_seasons() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "alpha01",
                "player_name": "Player A",
                "player_url": "https://www.basketball-reference.com/players/a/alpha01.html",
                "season": "2020-21",
                "season_start": 2020,
                "season_end": 2021,
                "age": 23,
                "team_name_abbr": "BOS",
                "games": 60,
                "mp": 1800,
                "per": 18.0,
                "bpm": 1.0,
                "vorp": 1.1,
                "ws": 4.0,
                "ws_per_48": 0.120,
            },
            {
                "player_id": "alpha01",
                "player_name": "Player A",
                "player_url": "https://www.basketball-reference.com/players/a/alpha01.html",
                "season": "2021-22",
                "season_start": 2021,
                "season_end": 2022,
                "age": 24,
                "team_name_abbr": "BOS",
                "games": 62,
                "mp": 1900,
                "per": 19.0,
                "bpm": 1.8,
                "vorp": 1.5,
                "ws": 4.6,
                "ws_per_48": 0.130,
            },
            {
                "player_id": "alpha01",
                "player_name": "Player A",
                "player_url": "https://www.basketball-reference.com/players/a/alpha01.html",
                "season": "2023-24",
                "season_start": 2023,
                "season_end": 2024,
                "age": 26,
                "team_name_abbr": "LAL",
                "games": 58,
                "mp": 1700,
                "per": 17.0,
                "bpm": 0.5,
                "vorp": 0.8,
                "ws": 3.0,
                "ws_per_48": 0.100,
            },
        ]
    )


def test_trajectory_threshold_boundaries() -> None:
    assert classify_trajectory(0.5, threshold=0.5) == "stable"
    assert classify_trajectory(-0.5, threshold=0.5) == "stable"
    assert classify_trajectory(0.51, threshold=0.5) == "improving"
    assert classify_trajectory(-0.51, threshold=0.5) == "regressing"


def test_no_target_across_non_consecutive_gap() -> None:
    features = build_features(_sample_player_seasons(), threshold=0.5)

    row_2021 = features.loc[features["season_start"] == 2021].iloc[0]
    assert pd.isna(row_2021["target_bpm_change"])
    assert pd.isna(row_2021["target_trajectory"])


def test_rolling_resets_after_gap() -> None:
    features = build_features(_sample_player_seasons(), threshold=0.5)
    row_2023 = features.loc[features["season_start"] == 2023].iloc[0]

    assert row_2023["season_gap_from_prev"] == 2
    assert bool(row_2023["is_consecutive_from_prev"]) is False
    # Gap-aware rolling should restart from current row, not average old seasons.
    assert float(row_2023["bpm_roll2"]) == float(row_2023["bpm"])


def test_feature_columns_are_reused_from_config() -> None:
    features = build_features(_sample_player_seasons(), threshold=0.5)

    feature_columns_in_output = [col for col in features.columns if col in TRAJECTORY_FEATURE_COLUMNS]
    assert feature_columns_in_output == TRAJECTORY_FEATURE_COLUMNS
