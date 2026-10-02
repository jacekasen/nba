"""Orchestrator for the trajectories pipeline: collect -> normalize -> validate."""

from __future__ import annotations

import argparse
import logging
import sys

from trajectories.collect import collect_all_seasons, collect_season
from trajectories.config import ALL_SEASONS
from trajectories.normalize import normalize_dataset
from trajectories.validate import validate_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    season: str = "2025-26",
    *,
    all_seasons: bool = False,
    refresh: bool = False,
    delay: float = 1.2,
) -> None:
    logger.info("=== STEP 1: COLLECTING GAME LOGS ===")
    if all_seasons:
        collect_all_seasons(ALL_SEASONS, refresh=refresh, delay=delay)
    else:
        collect_season(season, refresh=refresh, delay=delay)

    logger.info("=== STEP 2: NORMALIZING DATASET ===")
    df = normalize_dataset()
    if df.empty:
        logger.error("Normalization returned empty dataset. Exiting.")
        sys.exit(1)

    logger.info("=== STEP 3: VALIDATING DATASET ===")
    report = validate_trajectories(df)
    if report["status"] == "FAIL":
        logger.error("Validation failed with issues: %s", report["issues"])
        sys.exit(1)

    logger.info("Pipeline completed successfully! %s records processed.", report["total_rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=str, default="2025-26", help="Season to run")
    parser.add_argument("--all-seasons", action="store_true", help="Run across all play-by-play seasons")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch raw checkpoints")
    parser.add_argument("--delay", type=float, default=1.2, help="Request delay seconds")
    args = parser.parse_args()

    run_pipeline(season=args.season, all_seasons=args.all_seasons, refresh=args.refresh, delay=args.delay)


if __name__ == "__main__":
    main()
