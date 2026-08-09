"""Normalize raw player salaries, join salary caps, and build derived tables."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from salary.config import (
    MIN_TEAM_ROWS_FOR_PAYROLL_SHARE,
    MODELING_DIR,
    PLAYER_SALARIES_PATH,
    PLAYER_SALARY_COLUMNS,
    PLAYER_SALARY_HISTORY_PATH,
    RAW_PLAYER_SALARIES_PATH,
    SALARY_CAPS_PATH,
    TEAM_SEASON_SALARIES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PLAYER_SALARIES_PATH, help="Raw scraped salary CSV")
    parser.add_argument("--caps", type=Path, default=SALARY_CAPS_PATH, help="Normalized salary-cap CSV")
    parser.add_argument("--output", type=Path, default=PLAYER_SALARIES_PATH, help="Canonical player salaries CSV")
    parser.add_argument(
        "--team-output",
        type=Path,
        default=TEAM_SEASON_SALARIES_PATH,
        help="Derived team-season salary summary CSV",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=PLAYER_SALARY_HISTORY_PATH,
        help="Derived player career salary history CSV",
    )
    parser.add_argument(
        "--min-team-rows",
        type=int,
        default=MIN_TEAM_ROWS_FOR_PAYROLL_SHARE,
        help="Minimum known salaries on a team-season before computing team_payroll_share",
    )
    return parser.parse_args(argv)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def normalize_salaries(
    raw: pd.DataFrame,
    caps: pd.DataFrame,
    *,
    min_team_rows: int = MIN_TEAM_ROWS_FOR_PAYROLL_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join caps, compute cap_share / payroll share, and build derived frames."""
    _require_columns(
        raw,
        [
            "player_id",
            "player_name",
            "player_url",
            "season",
            "season_start",
            "season_end",
            "team",
            "salary",
            "salary_source",
            "salary_quality",
        ],
        "raw salaries",
    )
    _require_columns(caps, ["season", "salary_cap"], "salary caps")

    working = raw.copy()
    working["player_id"] = working["player_id"].astype(str)
    working["player_url"] = working["player_url"].astype(str)
    working["season"] = working["season"].astype(str)
    working["team"] = working["team"].astype(str).str.upper()
    working["salary"] = pd.to_numeric(working["salary"], errors="coerce").astype("Int64")

    # Deterministic collapse of accidental duplicate source rows.
    working = (
        working.sort_values(["player_id", "season", "team", "salary"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["player_id", "season", "team"], keep="first")
        .reset_index(drop=True)
    )

    caps_small = caps[["season", "salary_cap"]].drop_duplicates(subset=["season"], keep="last")
    merged = working.merge(caps_small, on="season", how="left", indicator=True)
    merged["cap_join_failed"] = merged["_merge"].eq("left_only")
    merged = merged.drop(columns=["_merge"])

    merged["salary_cap"] = pd.to_numeric(merged["salary_cap"], errors="coerce").astype("Int64")
    merged["cap_share"] = pd.NA
    has_both = merged["salary"].notna() & merged["salary_cap"].notna() & merged["salary_cap"].gt(0)
    merged.loc[has_both, "cap_share"] = (
        merged.loc[has_both, "salary"].astype(float) / merged.loc[has_both, "salary_cap"].astype(float) * 100.0
    )
    merged["cap_share"] = pd.to_numeric(merged["cap_share"], errors="coerce")

    # Team payroll share only when the team-season looks sufficiently populated.
    known = merged.loc[merged["salary"].notna()].copy()
    team_totals = (
        known.groupby(["team", "season"], as_index=False)
        .agg(
            team_known_salary_total=("salary", "sum"),
            team_known_player_count=("player_id", "nunique"),
        )
    )
    merged = merged.merge(team_totals, on=["team", "season"], how="left")
    merged["team_known_salary_total"] = pd.to_numeric(merged["team_known_salary_total"], errors="coerce").astype("Int64")
    merged["team_known_player_count"] = pd.to_numeric(merged["team_known_player_count"], errors="coerce").astype("Int64")
    merged["team_payroll_share"] = pd.NA
    eligible = (
        merged["salary"].notna()
        & merged["team_known_salary_total"].notna()
        & merged["team_known_salary_total"].gt(0)
        & merged["team_known_player_count"].fillna(0).ge(min_team_rows)
    )
    merged.loc[eligible, "team_payroll_share"] = (
        merged.loc[eligible, "salary"].astype(float)
        / merged.loc[eligible, "team_known_salary_total"].astype(float)
        * 100.0
    )
    merged["team_payroll_share"] = pd.to_numeric(merged["team_payroll_share"], errors="coerce")

    player_salaries = merged[PLAYER_SALARY_COLUMNS].copy()
    player_salaries = player_salaries.sort_values(
        ["season_start", "cap_share", "player_id", "team"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    history = player_salaries.sort_values(
        ["player_id", "season_start", "team"]
    ).reset_index(drop=True)

    team_season = (
        player_salaries.groupby(["team", "season", "season_start", "season_end"], as_index=False)
        .agg(
            salary_cap=("salary_cap", "max"),
            team_known_salary_total=("salary", "sum"),
            team_known_player_count=("player_id", "nunique"),
            total_cap_share=("cap_share", "sum"),
            max_cap_share=("cap_share", "max"),
        )
        .sort_values(["season_start", "team"])
        .reset_index(drop=True)
    )
    team_season["team_known_salary_total"] = pd.to_numeric(
        team_season["team_known_salary_total"], errors="coerce"
    ).astype("Int64")
    team_season["payroll_share_available"] = team_season["team_known_player_count"].ge(min_team_rows)
    team_season["team_payroll_vs_cap"] = pd.NA
    cap_ok = team_season["salary_cap"].notna() & team_season["salary_cap"].gt(0) & team_season[
        "team_known_salary_total"
    ].notna()
    team_season.loc[cap_ok, "team_payroll_vs_cap"] = (
        team_season.loc[cap_ok, "team_known_salary_total"].astype(float)
        / team_season.loc[cap_ok, "salary_cap"].astype(float)
        * 100.0
    )

    join_failures = int(merged["cap_join_failed"].sum())
    logger.info(
        "Normalized %s salary rows (%s players); cap join failures=%s",
        len(player_salaries),
        player_salaries["player_id"].nunique(),
        join_failures,
    )
    return player_salaries, team_season, history


def run_normalize(
    *,
    raw_path: Path,
    caps_path: Path,
    output_path: Path,
    team_output_path: Path,
    history_output_path: Path,
    min_team_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw salaries not found: {raw_path}. Run python -m salary.scrape first.")
    if not caps_path.exists():
        raise FileNotFoundError(f"Salary caps not found: {caps_path}. Run python -m salary.caps first.")

    raw = pd.read_csv(raw_path)
    caps = pd.read_csv(caps_path)
    player_salaries, team_season, history = normalize_salaries(raw, caps, min_team_rows=min_team_rows)

    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    player_salaries.to_csv(output_path, index=False)
    team_season.to_csv(team_output_path, index=False)
    history.to_csv(history_output_path, index=False)
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", team_output_path)
    logger.info("Wrote %s", history_output_path)
    return player_salaries, team_season, history


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_normalize(
        raw_path=args.raw,
        caps_path=args.caps,
        output_path=args.output,
        team_output_path=args.team_output,
        history_output_path=args.history_output,
        min_team_rows=args.min_team_rows,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
