"""Scrape Basketball Reference player salary tables with resume-safe caching."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from salary.config import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    DEFAULT_START_SEASON,
    INTERMEDIATE_SALARIES_DIR,
    PLAYER_HTML_DIR,
    PLAYER_URLS_PATH,
    RAW_PLAYER_SALARIES_PATH,
    RAW_SALARY_COLUMNS,
    RAW_STATS_PATH,
    SAMPLE_PLAYERS,
    SALARY_SOURCE,
    SCRAPE_FAILURES_PATH,
    SCRAPE_PROGRESS_PATH,
)
from ml.data import normalize_player_url, parse_year_id
from salary.http import AccessBlockedError, HttpClient
from salary.parsing import (
    classify_salary_quality,
    extract_player_name,
    normalize_player_url_or_raise,
    parse_all_salaries_table,
    player_id_from_url,
    season_sort_key,
)
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players-csv", type=Path, default=PLAYER_URLS_PATH, help="Player URL inventory CSV")
    parser.add_argument("--stats-csv", type=Path, default=RAW_STATS_PATH, help="Optional stats CSV used to filter cap-era players")
    parser.add_argument("--output", type=Path, default=RAW_PLAYER_SALARIES_PATH, help="Raw salary rows CSV")
    parser.add_argument("--html-dir", type=Path, default=PLAYER_HTML_DIR, help="Directory for cached player HTML")
    parser.add_argument("--progress", type=Path, default=SCRAPE_PROGRESS_PATH, help="JSON progress file")
    parser.add_argument("--failures", type=Path, default=SCRAPE_FAILURES_PATH, help="Failure log CSV")
    parser.add_argument("--start-season", default=DEFAULT_START_SEASON, help="Keep seasons on/after this label")
    parser.add_argument("--end-season", default=None, help="Optional inclusive end season filter")
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="Seconds between network requests")
    parser.add_argument("--refresh", action="store_true", help="Redownload HTML even when cached")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip players already present in output (default)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Reparse all selected players from cache/network")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of players to process")
    parser.add_argument(
        "--player",
        action="append",
        default=[],
        help="Restrict to player_id or player_url (repeatable)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the built-in representative validation sample",
    )
    return parser.parse_args(argv)


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_player_ids": [], "stopped_reason": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_existing_raw(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=RAW_SALARY_COLUMNS)


def _season_in_range(season: str, start_season: str, end_season: str | None) -> bool:
    if season_sort_key(season) < season_sort_key(start_season):
        return False
    if end_season is not None and season_sort_key(season) > season_sort_key(end_season):
        return False
    return True


def candidate_players(
    players_csv: Path,
    stats_csv: Path,
    *,
    sample: bool,
    player_filters: list[str],
    start_season: str,
) -> pd.DataFrame:
    """Build the player scrape queue."""
    if sample:
        return pd.DataFrame(SAMPLE_PLAYERS)

    if not players_csv.exists():
        raise FileNotFoundError(f"Player URL inventory not found: {players_csv}")

    players = pd.read_csv(players_csv)
    if "player_url" not in players.columns:
        raise ValueError(f"{players_csv} missing player_url column")

    players = players.copy()
    players["player_url"] = players["player_url"].map(normalize_player_url)
    invalid_players = int(players["player_url"].isna().sum())
    if invalid_players:
        logger.warning("Dropping %s player inventory rows with invalid player_url", invalid_players)
        players = players.loc[players["player_url"].notna()].copy()
    players["player_id"] = players["player_url"].map(player_id_from_url)
    if "player_name" not in players.columns:
        players["player_name"] = players["player_id"]

    if player_filters:
        wanted = {item.strip().lower() for item in player_filters if item.strip()}
        mask = players["player_id"].str.lower().isin(wanted) | players["player_url"].str.lower().isin(wanted)
        players = players.loc[mask].copy()
        missing = wanted - set(players["player_id"].str.lower()) - set(players["player_url"].str.lower())
        # Allow direct player_id filters even if inventory row is absent.
        extra_rows = []
        for item in sorted(missing):
            if item.startswith("http"):
                url = normalize_player_url_or_raise(item)
                extra_rows.append(
                    {
                        "player_id": player_id_from_url(url),
                        "player_name": player_id_from_url(url),
                        "player_url": url,
                    }
                )
            elif re_fullmatch_player_id(item):
                letter = item[0]
                url = f"https://www.basketball-reference.com/players/{letter}/{item}.html"
                extra_rows.append({"player_id": item, "player_name": item, "player_url": url})
        if extra_rows:
            players = pd.concat([players, pd.DataFrame(extra_rows)], ignore_index=True)

    elif stats_csv.exists():
        stats = pd.read_csv(stats_csv, usecols=["player_url", "year_id"])
        stats["player_url"] = stats["player_url"].map(normalize_player_url)
        invalid_stats = int(stats["player_url"].isna().sum())
        if invalid_stats:
            logger.warning("Dropping %s stats rows with invalid player_url while filtering candidates", invalid_stats)
            stats = stats.loc[stats["player_url"].notna()].copy()
        parsed = stats["year_id"].map(parse_year_id)
        stats["season"] = parsed.map(lambda item: item[0])
        stats["season_start"] = parsed.map(lambda item: item[1])
        start_year = parse_year_id(start_season)[1]
        if start_year is None:
            raise ValueError(f"Invalid start season: {start_season}")
        eligible_urls = set(stats.loc[stats["season_start"].ge(start_year - 5), "player_url"].dropna())
        # Keep a small lookback so late-career/pre-cap players who overlap the early
        # cap years are still considered, while skipping obviously pre-modern careers.
        players = players.loc[players["player_url"].isin(eligible_urls)].copy()
        logger.info("Filtered to %s cap-era-overlapping players via stats inventory", len(players))

    players = players.drop_duplicates(subset=["player_id"]).sort_values("player_id").reset_index(drop=True)
    return players[["player_id", "player_name", "player_url"]]


def re_fullmatch_player_id(value: str) -> bool:
    return bool(value) and value.replace("_", "").isalnum() and value[0].isalpha()


def parse_player_html(
    html: str,
    *,
    player_id: str,
    player_url: str,
    player_name: str | None,
    html_cache_path: Path,
    start_season: str,
    end_season: str | None,
) -> list[dict[str, Any]]:
    """Parse one cached/downloaded player page into raw salary rows."""
    soup = BeautifulSoup(html, "html.parser")
    resolved_name = extract_player_name(soup) or player_name or player_id
    parsed_rows = parse_all_salaries_table(html)

    output: list[dict[str, Any]] = []
    for row in parsed_rows:
        season = row.get("season")
        if not season or row.get("season_start") is None or row.get("season_end") is None:
            continue
        if not _season_in_range(season, start_season, end_season):
            continue
        # Keep only NBA rows when league is present.
        league = (row.get("league") or "").upper()
        if league and league != "NBA":
            continue

        salary = row.get("salary")
        team = row.get("team") or "UNK"

        output.append(
            {
                "player_id": player_id,
                "player_name": resolved_name,
                "player_url": player_url,
                "season": season,
                "season_start": row["season_start"],
                "season_end": row["season_end"],
                "team": team,
                "team_name": row.get("team_name"),
                "league": row.get("league") or "NBA",
                "salary": salary,
                "salary_source": SALARY_SOURCE,
                "salary_quality": classify_salary_quality(salary if isinstance(salary, int) else None),
                "html_cache_path": str(html_cache_path),
            }
        )
    return output


def scrape_players(
    players: pd.DataFrame,
    *,
    output_path: Path,
    html_dir: Path,
    progress_path: Path,
    failures_path: Path,
    start_season: str,
    end_season: str | None,
    delay: float,
    refresh: bool,
    resume: bool,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Scrape/parse player salary pages and persist continuously."""
    INTERMEDIATE_SALARIES_DIR.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_raw(output_path)
    progress = _load_progress(progress_path)
    completed = set(progress.get("completed_player_ids", []))
    if resume and not existing.empty:
        completed |= set(existing["player_id"].astype(str))

    todo = players.copy()
    if resume:
        todo = todo.loc[~todo["player_id"].isin(completed)].copy()

    http = client or HttpClient(delay_seconds=delay)
    new_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    blocked = False

    logger.info("Players selected=%s todo=%s resume=%s refresh=%s", len(players), len(todo), resume, refresh)

    try:
        for idx, player in enumerate(todo.itertuples(index=False), start=1):
            player_id = str(player.player_id)
            player_url = str(player.player_url)
            player_name = str(player.player_name)
            cache_path = html_dir / f"{player_id}.html"
            logger.info("[%s/%s] %s (%s)", idx, len(todo), player_name, player_id)

            try:
                html = http.fetch_html(player_url, cache_path, refresh=refresh)
                rows = parse_player_html(
                    html,
                    player_id=player_id,
                    player_url=player_url,
                    player_name=player_name,
                    html_cache_path=cache_path,
                    start_season=start_season,
                    end_season=end_season,
                )
                # Even players with zero in-range salary rows count as completed so
                # resume does not endlessly re-hit empty pages.
                new_rows.extend(rows)
                completed.add(player_id)

                # Persist after every player for interrupt safety.
                combined = _merge_raw(existing, new_rows)
                combined.to_csv(output_path, index=False)
                existing = combined
                new_rows = []
                _save_progress(
                    progress_path,
                    {
                        "completed_player_ids": sorted(completed),
                        "stopped_reason": None,
                        "last_player_id": player_id,
                        "rows_written": int(len(combined)),
                    },
                )
            except AccessBlockedError as exc:
                blocked = True
                failures.append({"player_id": player_id, "player_url": player_url, "reason": str(exc)})
                logger.error("%s", exc)
                _save_progress(
                    progress_path,
                    {
                        "completed_player_ids": sorted(completed),
                        "stopped_reason": "access_blocked",
                        "last_player_id": player_id,
                        "rows_written": int(len(existing)),
                    },
                )
                break
            except Exception as exc:  # noqa: BLE001 - persist and continue for non-block failures
                logger.exception("Failed parsing/scraping %s", player_id)
                failures.append({"player_id": player_id, "player_url": player_url, "reason": str(exc)})
    finally:
        if failures:
            fail_df = pd.DataFrame(failures)
            if failures_path.exists():
                prior = pd.read_csv(failures_path)
                fail_df = pd.concat([prior, fail_df], ignore_index=True)
            fail_df.to_csv(failures_path, index=False)
            logger.info("Wrote %s failure rows to %s", len(failures), failures_path)

    final = _load_existing_raw(output_path)
    logger.info(
        "Scrape complete blocked=%s players_completed=%s salary_rows=%s output=%s",
        blocked,
        len(completed),
        len(final),
        output_path,
    )
    if blocked:
        raise SystemExit(2)
    return final


def _merge_raw(existing: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    incoming = pd.DataFrame(new_rows, columns=RAW_SALARY_COLUMNS)
    if existing.empty:
        merged = incoming
    elif incoming.empty:
        merged = existing
    else:
        # Replace any prior rows for players present in this scrape batch.
        touched = set(incoming["player_id"].astype(str))
        kept = existing.loc[~existing["player_id"].astype(str).isin(touched)]
        merged = pd.concat([kept, incoming], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=RAW_SALARY_COLUMNS)
    merged = merged.drop_duplicates(subset=["player_id", "season", "team"], keep="last")
    return merged.sort_values(["player_id", "season_start", "team"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    players = candidate_players(
        args.players_csv,
        args.stats_csv,
        sample=args.sample,
        player_filters=args.player,
        start_season=args.start_season,
    )
    if args.limit is not None:
        players = players.head(args.limit).copy()
    if players.empty:
        logger.error("No players selected.")
        raise SystemExit(1)

    scrape_players(
        players,
        output_path=args.output,
        html_dir=args.html_dir,
        progress_path=args.progress,
        failures_path=args.failures,
        start_season=args.start_season,
        end_season=args.end_season,
        delay=args.delay,
        refresh=args.refresh,
        resume=args.resume,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
