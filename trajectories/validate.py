"""Validate normalized player game logs dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from trajectories.config import (
    CSV_OUTPUT_PATH,
    PARQUET_OUTPUT_PATH,
    SHORTENED_SEASONS,
    STANDARD_SEASON_GAMES,
    STANDARD_SEASON_MIN_REQUIRED,
    VALIDATION_REPORT_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_trajectories(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run validation checks on the normalized player game logs."""
    if df is None:
        if PARQUET_OUTPUT_PATH.exists():
            df = pd.read_parquet(PARQUET_OUTPUT_PATH)
        elif CSV_OUTPUT_PATH.exists():
            df = pd.read_csv(CSV_OUTPUT_PATH)
        else:
            raise FileNotFoundError(f"Neither {PARQUET_OUTPUT_PATH} nor {CSV_OUTPUT_PATH} exists.")

    report: dict[str, Any] = {
        "total_rows": len(df),
        "total_players": int(df["player_id"].nunique()),
        "total_seasons": int(df["season"].nunique()),
        "seasons_summary": {},
        "issues": [],
    }

    # 1. Check required non-null columns
    required_cols = ["player_id", "player_name", "season", "game_id", "team_abbreviation", "team_game_number"]
    for col in required_cols:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report["issues"].append(f"Column '{col}' has {null_count} null rows.")

    # 2. Check team_game_number bounds
    invalid_game_nums = int((df["team_game_number"] <= 0).sum())
    if invalid_game_nums > 0:
        report["issues"].append(f"Found {invalid_game_nums} rows with team_game_number <= 0.")

    # 3. Check plus-minus bounds (sanity check: single-game plus-minus shouldn't exceed +/- 80)
    extreme_pm = int(((df["plus_minus"] > 75) | (df["plus_minus"] < -75)).sum())
    if extreme_pm > 0:
        report["issues"].append(f"Found {extreme_pm} rows with single-game plus_minus outside [-75, 75].")

    # 4. Per-season validation
    for season, s_df in df.groupby("season"):
        season_str = str(season)
        if season_str in SHORTENED_SEASONS:
            expected_max = SHORTENED_SEASONS[season_str]["expected_max"]
            min_required = SHORTENED_SEASONS[season_str]["min_required"]
        else:
            expected_max = STANDARD_SEASON_GAMES
            min_required = STANDARD_SEASON_MIN_REQUIRED

        actual_max_team_games = int(s_df["team_game_number"].max())

        s_summary = {
            "rows": len(s_df),
            "players": int(s_df["player_id"].nunique()),
            "teams": int(s_df["team_abbreviation"].nunique()),
            "max_team_game_number": actual_max_team_games,
            "expected_max_games": expected_max,
            "min_required_games": min_required,
        }
        report["seasons_summary"][season_str] = s_summary

        if actual_max_team_games > expected_max + 1:  # Allow 1 game tolerance for edge cases
            report["issues"].append(
                f"Season {season} max team_game_number ({actual_max_team_games}) exceeds expected max ({expected_max})."
            )
        if actual_max_team_games < min_required:
            report["issues"].append(
                f"Season {season} max team_game_number ({actual_max_team_games}) is below required minimum ({min_required}). Data appears truncated or missing restart games."
            )

    report["status"] = "PASS" if not report["issues"] else "WARN"

    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Validation report written to %s (Status: %s)", VALIDATION_REPORT_PATH, report["status"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    validate_trajectories()


if __name__ == "__main__":
    main()
