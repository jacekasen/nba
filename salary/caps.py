"""Download and normalize NBA salary-cap history from Basketball Reference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from salary.config import (
    CAP_HTML_PATH,
    DEFAULT_REQUEST_DELAY_SECONDS,
    INTERMEDIATE_SALARIES_DIR,
    SALARY_CAP_URL,
    SALARY_CAPS_PATH,
)
from salary.http import AccessBlockedError, HttpClient
from salary.parsing import parse_salary_cap_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SALARY_CAPS_PATH, help="Normalized salary-cap CSV")
    parser.add_argument("--html-path", type=Path, default=CAP_HTML_PATH, help="Cached salary-cap HTML path")
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS, help="Request delay seconds")
    parser.add_argument("--refresh", action="store_true", help="Redownload cap history HTML")
    return parser.parse_args(argv)


def build_salary_caps(
    *,
    output_path: Path = SALARY_CAPS_PATH,
    html_path: Path = CAP_HTML_PATH,
    delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    refresh: bool = False,
    client: HttpClient | None = None,
) -> pd.DataFrame:
    """Fetch (or reuse cache) and write salary_caps.csv."""
    INTERMEDIATE_SALARIES_DIR.mkdir(parents=True, exist_ok=True)
    http = client or HttpClient(delay_seconds=delay)
    try:
        html = http.fetch_html(SALARY_CAP_URL, html_path, refresh=refresh)
    except AccessBlockedError:
        logger.error("Access blocked while fetching salary-cap history; preserving prior outputs.")
        raise

    rows = parse_salary_cap_table(html)
    if not rows:
        raise RuntimeError("Parsed zero salary-cap rows; refusing to overwrite output")

    df = pd.DataFrame(rows, columns=["season", "season_start", "season_end", "salary_cap"])
    df = df.drop_duplicates(subset=["season"], keep="last").sort_values("season_start").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Wrote %s salary-cap seasons (%s -> %s) to %s",
        len(df),
        df["season"].iloc[0],
        df["season"].iloc[-1],
        output_path,
    )
    return df


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        build_salary_caps(
            output_path=args.output,
            html_path=args.html_path,
            delay=args.delay,
            refresh=args.refresh,
        )
    except AccessBlockedError:
        raise SystemExit(2) from None


if __name__ == "__main__":
    main(sys.argv[1:])
