"""Collect play-by-play era player game logs from NBA Stats in monthly chunks with disk caching."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from curl_cffi import requests

from trajectories.config import (
    ALL_SEASONS,
    CURL_IMPERSONATE,
    MAX_RETRIES,
    RAW_DIR,
    REGULAR_SEASON_MONTHS,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    STATS_HEADERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_url(season: str, month: int) -> str:
    """Build NBA Stats playergamelogs URL for a season and month."""
    base = "https://stats.nba.com/stats/playergamelogs"
    params = [
        "DateFrom=",
        "DateTo=",
        "GameSegment=",
        "LastNGames=0",
        "LeagueID=00",
        "Location=",
        "MeasureType=Base",
        f"Month={month}",
        "OpponentTeamID=0",
        "Outcome=",
        "PORound=0",
        "PaceAdjust=N",
        "PerMode=Totals",
        "Period=0",
        "PlayerID=",
        "PlusMinus=N",
        "Rank=N",
        f"Season={season}",
        "SeasonSegment=",
        "SeasonType=Regular+Season",
        "ShotClockRange=",
        "TeamID=",
        "VsConference=",
        "VsDivision=",
    ]
    return f"{base}?{'&'.join(params)}"


def fetch_month_logs(
    season: str,
    month: int,
    *,
    delay: float = REQUEST_DELAY_SECONDS,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
) -> dict[str, Any] | None:
    """Fetch game logs for one season and month with retries."""
    url = build_url(season, month)
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            time.sleep(delay + random.uniform(0.1, 0.4))
            resp = requests.get(
                url,
                headers=STATS_HEADERS,
                impersonate=CURL_IMPERSONATE,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data
            elif resp.status_code in {429, 500, 502, 503, 504}:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "HTTP %s for %s Month %s (attempt %s/%s). Backing off %.1fs",
                    resp.status_code,
                    season,
                    month,
                    attempt,
                    retries,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error("Unexpected status %s for %s Month %s", resp.status_code, season, month)
                return None
        except Exception as exc:
            last_err = exc
            wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(
                "Request error for %s Month %s: %s (attempt %s/%s). Backing off %.1fs",
                season,
                month,
                exc,
                attempt,
                retries,
                wait_time,
            )
            time.sleep(wait_time)

    logger.error("Failed fetching %s Month %s after %s retries: %s", season, month, retries, last_err)
    return None


def collect_season(
    season: str,
    *,
    output_dir: Path = RAW_DIR,
    refresh: bool = False,
    delay: float = REQUEST_DELAY_SECONDS,
) -> int:
    """Collect all monthly game logs for a season, saving to disk checkpoints."""
    season_dir = output_dir / season
    season_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    logger.info("Starting collection for season %s", season)
    for month in REGULAR_SEASON_MONTHS:
        month_file = season_dir / f"month_{month}.json"
        if month_file.exists() and not refresh:
            try:
                cached = json.loads(month_file.read_text(encoding="utf-8"))
                rows_in_file = len(cached.get("resultSets", [{}])[0].get("rowSet", []))
                total_rows += rows_in_file
                logger.info("Using cached %s Month %s (%s rows)", season, month, rows_in_file)
                continue
            except Exception:
                pass

        logger.info("Fetching %s Month %s...", season, month)
        data = fetch_month_logs(season, month, delay=delay)
        if data is None:
            logger.warning("No data returned for %s Month %s", season, month)
            continue

        result_sets = data.get("resultSets", [])
        rows = result_sets[0].get("rowSet", []) if result_sets else []
        month_file.write_text(json.dumps(data), encoding="utf-8")
        total_rows += len(rows)
        logger.info("Saved %s Month %s: %s rows -> %s", season, month, len(rows), month_file)

    logger.info("Completed season %s with %s total rows", season, total_rows)
    return total_rows


def collect_all_seasons(
    seasons: list[str] | None = None,
    *,
    refresh: bool = False,
    delay: float = REQUEST_DELAY_SECONDS,
) -> dict[str, int]:
    """Collect all requested play-by-play seasons."""
    target_seasons = seasons or ALL_SEASONS
    summary = {}
    for season in target_seasons:
        rows = collect_season(season, refresh=refresh, delay=delay)
        summary[season] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=str, default="2025-26", help="Specific season to collect (e.g. 2025-26)")
    parser.add_argument("--all-seasons", action="store_true", help="Collect all seasons back to 1996-97")
    parser.add_argument("--start-season", type=str, default=None, help="Start season (e.g. 1996-97)")
    parser.add_argument("--end-season", type=str, default=None, help="End season (e.g. 2025-26)")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch of existing checkpoints")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Delay between requests in seconds")
    args = parser.parse_args()

    if args.all_seasons:
        collect_all_seasons(ALL_SEASONS, refresh=args.refresh, delay=args.delay)
    elif args.start_season and args.end_season:
        start_idx = ALL_SEASONS.index(args.start_season)
        end_idx = ALL_SEASONS.index(args.end_season)
        target = ALL_SEASONS[start_idx : end_idx + 1]
        collect_all_seasons(target, refresh=args.refresh, delay=args.delay)
    else:
        collect_season(args.season, refresh=args.refresh, delay=args.delay)


if __name__ == "__main__":
    main()
