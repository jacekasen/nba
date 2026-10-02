"""Normalize raw player game logs, assign team game numbers (1-82), and compute rolling metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trajectories.config import (
    CSV_OUTPUT_PATH,
    NORMALIZED_DIR,
    PARQUET_OUTPUT_PATH,
    RAW_DIR,
    TEAM_SCHEDULES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_minutes(val: Any) -> float:
    """Parse minutes into a clean decimal float."""
    if pd.isna(val) or val is None or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    s = str(val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return round(float(parts[0]) + float(parts[1]) / 60.0, 2)
        except Exception:
            return 0.0
    try:
        return round(float(s), 2)
    except Exception:
        return 0.0


def parse_matchup(matchup: str) -> tuple[bool, str]:
    """Parse matchup into (is_home, opponent_abbreviation)."""
    if not matchup:
        return False, ""
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        return True, parts[1].strip() if len(parts) > 1 else ""
    elif " @ " in matchup:
        parts = matchup.split(" @ ")
        return False, parts[1].strip() if len(parts) > 1 else ""
    return False, ""


def load_raw_season_logs(season_dir: Path) -> pd.DataFrame:
    """Load and concatenate all month JSON files for a season."""
    all_rows = []
    headers = None

    for month_file in sorted(season_dir.glob("month_*.json")):
        try:
            data = json.loads(month_file.read_text(encoding="utf-8"))
            result_sets = data.get("resultSets", [])
            if not result_sets:
                continue
            first_set = result_sets[0]
            if headers is None:
                headers = first_set.get("headers", [])
            rows = first_set.get("rowSet", [])
            all_rows.extend(rows)
        except Exception as exc:
            logger.warning("Error reading %s: %s", month_file, exc)

    if not all_rows or not headers:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=headers)
    # Deduplicate in case of overlapping queries
    if {"PLAYER_ID", "GAME_ID", "TEAM_ID"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    return df


def assign_team_game_numbers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign team_game_number (1..82) based on chronological order of games for each team."""
    team_schedules: dict[str, Any] = {}

    # Standardize date to YYYY-MM-DD
    df["game_date"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    # Map team schedules
    team_game_map: dict[tuple[str, str], int] = {}

    for (season, team), group in df.groupby(["SEASON_YEAR", "TEAM_ABBREVIATION"]):
        unique_games = (
            group[["game_date", "GAME_ID"]]
            .drop_duplicates()
            .sort_values(by=["game_date", "GAME_ID"])
            .reset_index(drop=True)
        )
        unique_games["team_game_number"] = unique_games.index + 1

        sched_key = f"{season}_{team}"
        team_schedules[sched_key] = [
            {"game_id": r["GAME_ID"], "game_date": r["game_date"], "team_game_number": int(r["team_game_number"])}
            for _, r in unique_games.iterrows()
        ]

        for _, r in unique_games.iterrows():
            team_game_map[(r["GAME_ID"], team)] = int(r["team_game_number"])

    df["team_game_number"] = df.apply(
        lambda r: team_game_map.get((r["GAME_ID"], r["TEAM_ABBREVIATION"]), 0), axis=1
    )
    return df, team_schedules


def compute_player_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player appearance numbers and rolling plus-minus metrics."""
    # Ensure chronological order per player
    df = df.sort_values(by=["PLAYER_ID", "game_date", "GAME_ID"]).reset_index(drop=True)

    df["player_season_game_number"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"]).cumcount() + 1

    # Rolling metrics
    df["rolling_pm_5"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["plus_minus"].transform(
        lambda s: s.rolling(5, min_periods=1).mean().round(2)
    )
    df["rolling_pm_10"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["plus_minus"].transform(
        lambda s: s.rolling(10, min_periods=1).mean().round(2)
    )
    df["rolling_pm_15"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["plus_minus"].transform(
        lambda s: s.rolling(15, min_periods=1).mean().round(2)
    )

    # Minutes-weighted rolling 10
    prod = df["plus_minus"] * df["minutes"]
    sum_prod = prod.groupby([df["PLAYER_ID"], df["SEASON_YEAR"]]).transform(
        lambda s: s.rolling(10, min_periods=1).sum()
    )
    sum_min = df["minutes"].groupby([df["PLAYER_ID"], df["SEASON_YEAR"]]).transform(
        lambda s: s.rolling(10, min_periods=1).sum()
    )
    df["minutes_weighted_pm_10"] = np.where(
        sum_min > 0, (sum_prod / sum_min).round(2), df["rolling_pm_10"]
    )

    # Identify traded segment transitions
    prev_team = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["TEAM_ABBREVIATION"].shift(1)
    df["is_trade_transition"] = (df["TEAM_ABBREVIATION"] != prev_team) & prev_team.notna()

    return df


def normalize_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Run normalization across all collected season logs."""
    all_season_dfs = []
    combined_schedules = {}

    season_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    logger.info("Found %s season directories in %s", len(season_dirs), raw_dir)

    for s_dir in season_dirs:
        season_name = s_dir.name
        raw_df = load_raw_season_logs(s_dir)
        if raw_df.empty:
            logger.info("Season %s has no raw rows, skipping", season_name)
            continue

        logger.info("Loaded %s raw rows for %s", len(raw_df), season_name)
        season_df, schedules = assign_team_game_numbers(raw_df)
        combined_schedules.update(schedules)
        all_season_dfs.append(season_df)

    if not all_season_dfs:
        logger.warning("No data found to normalize")
        return pd.DataFrame()

    full_df = pd.concat(all_season_dfs, ignore_index=True)

    # Standardize column names and types
    full_df["minutes"] = full_df["MIN"].apply(parse_minutes)
    full_df["plus_minus"] = pd.to_numeric(full_df["PLUS_MINUS"], errors="coerce").fillna(0).astype(int)
    full_df["points"] = pd.to_numeric(full_df.get("PTS", 0), errors="coerce").fillna(0).astype(int)
    full_df["rebounds"] = pd.to_numeric(full_df.get("REB", 0), errors="coerce").fillna(0).astype(int)
    full_df["assists"] = pd.to_numeric(full_df.get("AST", 0), errors="coerce").fillna(0).astype(int)
    full_df["win_loss"] = full_df.get("WL", "").fillna("")

    # Parse matchups
    matchup_res = full_df["MATCHUP"].apply(parse_matchup)
    full_df["is_home"] = [m[0] for m in matchup_res]
    full_df["opponent_abbreviation"] = [m[1] for m in matchup_res]

    # Compute rolling metrics
    logger.info("Computing rolling trajectory metrics...")
    full_df = compute_player_trajectories(full_df)

    # Select and order final canonical columns
    canonical_columns = [
        "SEASON_YEAR",
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_ID",
        "game_date",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "opponent_abbreviation",
        "is_home",
        "team_game_number",
        "player_season_game_number",
        "win_loss",
        "minutes",
        "plus_minus",
        "points",
        "rebounds",
        "assists",
        "rolling_pm_5",
        "rolling_pm_10",
        "rolling_pm_15",
        "minutes_weighted_pm_10",
        "is_trade_transition",
    ]

    clean_df = full_df[[c for c in canonical_columns if c in full_df.columns]].copy()
    clean_df.rename(
        columns={
            "SEASON_YEAR": "season",
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "GAME_ID": "game_id",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbreviation",
        },
        inplace=True,
    )

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    TEAM_SCHEDULES_PATH.write_text(json.dumps(combined_schedules, indent=2), encoding="utf-8")
    clean_df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
    clean_df.to_csv(CSV_OUTPUT_PATH, index=False)
    logger.info("Saved normalized dataset: %s rows -> %s", len(clean_df), PARQUET_OUTPUT_PATH)
    return clean_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    normalize_dataset()


if __name__ == "__main__":
    main()
