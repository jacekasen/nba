"""Feature engineering for trajectory and BPM-delta prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ml.config import (
    FeatureConfig,
    MODEL_FEATURES_PATH,
    PEAK_AGE,
    PLAYER_SEASONS_PATH,
    TRAJECTORY_FEATURE_COLUMNS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PLAYER_SEASONS_PATH, help="Canonical player-seasons CSV")
    parser.add_argument("--output", type=Path, default=MODEL_FEATURES_PATH, help="Output features CSV")
    parser.add_argument(
        "--improving-threshold",
        type=float,
        default=FeatureConfig().improving_threshold,
        help="BPM gain beyond which next season counts as improving",
    )
    parser.add_argument(
        "--regressing-threshold",
        type=float,
        default=FeatureConfig().regressing_threshold,
        help="BPM decline magnitude beyond which next season counts as regressing",
    )
    return parser.parse_args()


def classify_trajectory(
    delta: float | None,
    improving_threshold: float,
    regressing_threshold: float,
) -> str | None:
    """Map next-season BPM change to improving/stable/regressing.

    Bands are asymmetric: minor declines (within `regressing_threshold`) are
    considered stable, since they sit inside the metric's noise floor.
    """
    if delta is None or pd.isna(delta):
        return None
    if delta > improving_threshold:
        return "improving"
    if delta < -regressing_threshold:
        return "regressing"
    return "stable"


def _rolling_slope(values: Iterable[float]) -> float | np.nan:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 3 or np.isnan(arr).any():
        return np.nan
    x = np.arange(len(arr), dtype=float)
    slope = np.polyfit(x, arr, deg=1)[0]
    return float(slope)


def _add_gap_aware_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["season_gap_from_prev"] = group["season_start"].diff().fillna(1).astype(int)
    group["is_consecutive_from_prev"] = group["season_gap_from_prev"].eq(1)
    group["streak_id"] = (~group["is_consecutive_from_prev"]).cumsum()

    metric_cols = ["bpm", "per", "ws_per_48"]
    for metric in metric_cols:
        group[f"{metric}_roll2"] = (
            group.groupby("streak_id", group_keys=False)[metric]
            .rolling(window=2, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        group[f"{metric}_roll3"] = (
            group.groupby("streak_id", group_keys=False)[metric]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    group["bpm_slope_3"] = (
        group.groupby("streak_id", group_keys=False)["bpm"]
        .rolling(window=3, min_periods=3)
        .apply(_rolling_slope, raw=False)
        .reset_index(level=0, drop=True)
    )

    return group


def build_features(
    player_seasons: pd.DataFrame,
    improving_threshold: float = FeatureConfig().improving_threshold,
    regressing_threshold: float = FeatureConfig().regressing_threshold,
) -> pd.DataFrame:
    """Create leakage-safe features and next-season targets."""
    df = player_seasons.copy()
    df = df.sort_values(["player_url", "season_start", "season_end"]).reset_index(drop=True)

    df["mpg"] = np.where(df["games"] > 0, df["mp"] / df["games"], np.nan)
    df["experience"] = df.groupby("player_url").cumcount()
    df["qualified_seasons_to_date"] = df.groupby("player_url").cumcount() + 1
    df["changed_team"] = (
        df.groupby("player_url")["team_name_abbr"]
        .transform(lambda s: s.ne(s.shift(1)).astype(int))
        .fillna(0)
        .astype(int)
    )

    lag_features = {
        "bpm": "bpm_delta_1",
        "per": "per_delta_1",
        "ws_per_48": "ws_per_48_delta_1",
        "games": "games_delta_1",
        "mp": "mp_delta_1",
        "mpg": "mpg_delta_1",
    }

    season_diff_1 = df.groupby("player_url")["season_start"].diff()
    consecutive_1 = season_diff_1.eq(1)
    for col, out_col in lag_features.items():
        delta = df[col] - df.groupby("player_url")[col].shift(1)
        df[out_col] = np.where(consecutive_1, delta, np.nan)

    season_diff_2 = df["season_start"] - df.groupby("player_url")["season_start"].shift(2)
    bpm_delta_2 = df["bpm"] - df.groupby("player_url")["bpm"].shift(2)
    df["bpm_delta_2"] = np.where(season_diff_2.eq(2), bpm_delta_2, np.nan)

    rolled_frames = []
    for _, player_group in df.groupby("player_url", sort=False):
        rolled_frames.append(_add_gap_aware_rolling_features(player_group))
    df = pd.concat(rolled_frames, ignore_index=True)

    df["career_high_bpm_through_t"] = df.groupby("player_url")["bpm"].cummax()
    df["prior_career_high_bpm"] = df.groupby("player_url")["career_high_bpm_through_t"].shift(1)
    df["distance_from_career_high_bpm"] = df["career_high_bpm_through_t"] - df["bpm"]

    # Availability relative to that season's schedule (handles lockout/COVID years).
    season_max_games = df.groupby("season_start")["games"].transform("max")
    df["games_pct"] = np.where(season_max_games > 0, df["games"] / season_max_games, np.nan)

    # Age-curve terms let linear models express that mean reversion and
    # minute changes carry different meaning on either side of the peak age.
    age_offset = df["age"] - PEAK_AGE
    df["age_curve_sq"] = age_offset**2
    df["age_x_bpm_delta_1"] = age_offset * df["bpm_delta_1"]
    df["age_x_mp_delta_1"] = age_offset * df["mp_delta_1"]

    next_season_start = df.groupby("player_url")["season_start"].shift(-1)
    next_bpm = df.groupby("player_url")["bpm"].shift(-1)
    has_consecutive_next = (next_season_start - df["season_start"]).eq(1)

    df["target_bpm_change"] = np.where(has_consecutive_next, next_bpm - df["bpm"], np.nan)

    # Exit-aware trajectory target: if the league played a season after t but
    # the player logged no qualified consecutive season, that is treated as
    # "regressing" rather than dropped. Without this, training only sees
    # survivors and learns that aging players with bad seasons bounce back.
    max_season_start = int(df["season_start"].max())
    df["next_season_observable"] = df["season_start"] < max_season_start
    df["target_played_next"] = np.where(
        df["next_season_observable"], has_consecutive_next.astype(float), np.nan
    )

    trajectory_if_played = df["target_bpm_change"].apply(
        lambda x: classify_trajectory(x, improving_threshold, regressing_threshold)
    )
    df["target_trajectory"] = np.select(
        [has_consecutive_next, df["next_season_observable"]],
        [trajectory_if_played, "regressing"],
        default=None,
    )
    df["target_trajectory"] = df["target_trajectory"].where(df["target_trajectory"].notna(), np.nan)

    metadata_cols = [
        "player_id",
        "player_name",
        "player_url",
        "season",
        "season_end",
    ]
    target_cols = [
        "target_bpm_change",
        "target_trajectory",
        "target_played_next",
        "next_season_observable",
    ]
    ordered_cols = metadata_cols + list(TRAJECTORY_FEATURE_COLUMNS) + target_cols

    return df[ordered_cols].copy()


def main() -> None:
    """CLI entry point for feature generation."""
    args = _parse_args()
    input_df = pd.read_csv(args.input)
    features = build_features(
        input_df,
        improving_threshold=args.improving_threshold,
        regressing_threshold=args.regressing_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    total_rows = len(features)
    target_rows = features["target_trajectory"].notna().sum()
    print("=== Feature Engineering Summary ===")
    print(f"Rows written: {total_rows:,}")
    print(f"Rows with next-season target: {target_rows:,}")
    print(f"Unique players: {features['player_url'].nunique():,}")
    print(f"Season range: {int(features['season_start'].min())} -> {int(features['season_start'].max())}")
    print(f"Saved features to: {args.output}")


if __name__ == "__main__":
    main()
