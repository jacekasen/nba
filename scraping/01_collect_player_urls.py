"""Collect NBA/ABA player names and profile URLs from Basketball-Reference's A-Z player index pages."""

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

BASE_URL = "https://www.basketball-reference.com"
ALL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
RATE_LIMIT_SECONDS = 3.0  # Basketball-Reference allows 20 requests/minute

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "player_pages.csv"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / ".cache" / "player_url_checkpoints"


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_letter(driver: webdriver.Chrome, letter: str) -> pd.DataFrame:
    driver.get(f"{BASE_URL}/players/{letter}/")
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "players")))

    columns = ["player_name", "player_url", "letter"]
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", id="players")
    if table is None:
        return pd.DataFrame(columns=columns)

    rows = []
    for row in table.find("tbody").find_all("tr"):
        name_cell = row.find("th", {"data-stat": "player"})
        link = name_cell.find("a") if name_cell else None
        if link is None:
            continue
        rows.append(
            {
                "player_name": link.text.strip(),
                "player_url": BASE_URL + link["href"],
                "letter": letter.upper(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def collect(letters: str, output_path: Path, checkpoint_dir: Path, delay: float, force: bool) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    driver = build_driver()
    failures = []

    try:
        for letter in letters:
            checkpoint_path = checkpoint_dir / f"letter_{letter}.csv"
            if checkpoint_path.exists() and not force:
                print(f"[{letter.upper()}] Skipping, checkpoint already exists")
                continue

            print(f"[{letter.upper()}] Fetching player index...")
            try:
                df_letter = scrape_letter(driver, letter)
            except (TimeoutException, WebDriverException) as exc:
                print(f"[{letter.upper()}] Failed: {exc}")
                failures.append({"letter": letter.upper(), "reason": str(exc)})
                time.sleep(delay)
                continue

            df_letter.to_csv(checkpoint_path, index=False)
            print(f"[{letter.upper()}] Saved {len(df_letter)} players")
            time.sleep(delay)
    finally:
        driver.quit()

    checkpoints = sorted(checkpoint_dir.glob("letter_*.csv"))
    if not checkpoints:
        print("No checkpoints found, nothing to combine.")
        return

    df_all = pd.concat((pd.read_csv(p) for p in checkpoints), ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_all)} players across {len(checkpoints)} letters to {output_path}")

    if failures:
        failures_path = output_path.parent / "url_collection_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"{len(failures)} letter(s) failed, logged to {failures_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for the combined CSV")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Per-letter checkpoint directory")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_SECONDS, help="Seconds to wait between letter requests")
    parser.add_argument("--letters", default=ALL_LETTERS, help="Letters to scrape, e.g. 'abc' (default: all 26)")
    parser.add_argument("--force", action="store_true", help="Re-scrape letters even if a checkpoint already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.letters.lower(), args.output, args.checkpoint_dir, args.delay, args.force)


if __name__ == "__main__":
    main()
