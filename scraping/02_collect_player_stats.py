"""Collect season-by-season advanced stats for each NBA/ABA player from their Basketball-Reference profile page."""

import argparse
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

RATE_LIMIT_SECONDS = 3.0  # Basketball-Reference allows 20 requests/minute
CHECKPOINT_EVERY = 25
REFRESH_LOOKBACK = 2  # rescrape players whose latest season is among the N most recent seasons in the data, since a new season may have been added for them since last run
TARGET_STATS = ["year_id", "age", "team_name_abbr", "games", "mp", "per", "bpm", "vorp", "ws", "ws_per_48"]
COLUMNS = ["player_name", "player_url"] + TARGET_STATS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URLS = REPO_ROOT / "data" / "01-pages" / "all_nba_players.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "nba_player_stats.csv"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / "02-player-stats-extraction" / "checkpoints"

# player_name is not a stable key: Basketball-Reference has dozens of distinct players who share
# a name (e.g. two different "Bobby Jones"s). player_url is the real unique identifier, so all
# resume/refresh/merge logic below keys off it instead of the name.


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_player(driver: webdriver.Chrome, player_name: str, player_url: str) -> pd.DataFrame:
    driver.get(player_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "advanced")))

    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", id="advanced")
    if table is None:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for row in table.find("tbody").find_all("tr"):
        season_cell = row.find("th", {"data-stat": "year_id"})
        season = season_cell.text.strip() if season_cell else ""
        if not season or season.startswith("Career"):
            continue

        season_data = {"player_name": player_name, "player_url": player_url}
        for stat in TARGET_STATS:
            cell = row.find(["th", "td"], {"data-stat": stat})
            value = cell.text.strip() if cell else ""
            season_data[stat] = value or None
        rows.append(season_data)

    return pd.DataFrame(rows, columns=COLUMNS)


def load_existing(output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path)
    return pd.DataFrame(columns=COLUMNS)


def urls_needing_refresh(existing: pd.DataFrame, lookback: int) -> set[str]:
    """player_urls whose latest recorded season is recent enough that a new season may since have been added."""
    if existing.empty:
        return set()
    recent_seasons = set(sorted(existing["year_id"].dropna().unique())[-lookback:])
    latest_by_url = existing.groupby("player_url")["year_id"].max()
    return set(latest_by_url[latest_by_url.isin(recent_seasons)].index)


def collect(
    urls_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    delay: float,
    force: bool,
    limit: int | None,
    refresh_lookback: int,
) -> None:
    df_players = pd.read_csv(urls_path)
    existing = load_existing(output_path)

    if force:
        todo_urls = set(df_players["player_url"])
    else:
        have_stats = set(existing["player_url"])
        missing = set(df_players["player_url"]) - have_stats
        needs_refresh = urls_needing_refresh(existing, refresh_lookback)
        todo_urls = missing | needs_refresh
        print(f"{len(missing)} new players, {len(needs_refresh)} recently-active players to refresh")

    todo = df_players[df_players["player_url"].isin(todo_urls)]
    if limit:
        todo = todo.head(limit)

    print(f"{len(df_players)} players total, {len(todo)} to scrape")
    if todo.empty:
        print("Nothing to do.")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    driver = build_driver()
    new_frames = []
    failures = []

    try:
        for idx, (_, player) in enumerate(todo.iterrows(), 1):
            name, url = player["player_name"], player["player_url"]
            print(f"[{idx}/{len(todo)}] {name}")
            try:
                new_frames.append(scrape_player(driver, name, url))
            except (TimeoutException, WebDriverException) as exc:
                print(f"  Failed: {exc}")
                failures.append({"player_name": name, "player_url": url, "reason": str(exc)})

            if idx % CHECKPOINT_EVERY == 0:
                df_checkpoint = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(columns=COLUMNS)
                df_checkpoint.to_csv(checkpoint_dir / f"progress_{idx}.csv", index=False)
                print(f"  -> checkpoint saved ({len(df_checkpoint)} seasons so far)")

            time.sleep(delay)
    finally:
        driver.quit()

    df_new = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(columns=COLUMNS)
    scraped_urls = set(df_new["player_url"].unique())
    existing_kept = existing[~existing["player_url"].isin(scraped_urls)]
    df_final = pd.concat([existing_kept, df_new], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_final)} season rows ({df_final['player_url'].nunique()} players) to {output_path}")

    if failures:
        failures_path = output_path.parent / "stats_collection_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"{len(failures)} player(s) failed, logged to {failures_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, default=DEFAULT_URLS, help="CSV of player_name/player_url to scrape")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for the combined stats CSV")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Progress checkpoint directory")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_SECONDS, help="Seconds to wait between player requests")
    parser.add_argument("--force", action="store_true", help="Re-scrape every player, ignoring existing stats")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N pending players")
    parser.add_argument(
        "--refresh-lookback",
        type=int,
        default=REFRESH_LOOKBACK,
        help="Rescrape players whose latest season is among the N most recent seasons in the existing data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.urls, args.output, args.checkpoint_dir, args.delay, args.force, args.limit, args.refresh_lookback)


if __name__ == "__main__":
    main()
