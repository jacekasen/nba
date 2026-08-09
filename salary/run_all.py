"""Orchestrate the salary analysis pipeline in logical order."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from salary.config import (
    PLAYER_HTML_DIR,
    PLAYER_SALARIES_PATH,
    RAW_PLAYER_SALARIES_PATH,
    SALARY_CAPS_PATH,
    VALIDATION_REPORT_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--sample", action="store_true", help="Scrape only the representative validation sample")
    parser.add_argument("--limit", type=int, default=None, help="Optional player scrape limit")
    parser.add_argument("--player", action="append", default=[], help="Optional player_id/url filter (repeatable)")
    parser.add_argument("--start-season", default="1984-85")
    parser.add_argument("--end-season", default=None)
    parser.add_argument("--refresh", action="store_true", help="Force HTML redownload")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reparse selected players even if already present in raw output",
    )
    parser.add_argument("--skip-scrape", action="store_true", help="Reuse existing raw salary CSV / HTML cache")
    parser.add_argument("--skip-caps", action="store_true", help="Reuse existing salary_caps.csv")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation step")
    return parser.parse_args(argv)


def _run(python_executable: Path, module_name: str, module_args: list[str] | None = None) -> None:
    cmd = [str(python_executable), "-m", module_name, *(module_args or [])]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.skip_caps:
        caps_args = ["--refresh"] if args.refresh else []
        # Avoid network if cache and output already exist and refresh was not requested.
        if SALARY_CAPS_PATH.exists() and not args.refresh:
            logger.info("Salary caps already present; running caps module (cache-friendly).")
        _run(args.python_executable, "salary.caps", caps_args)
    else:
        logger.info("Skipping salary.caps")

    if not args.skip_scrape:
        scrape_args: list[str] = [
            "--start-season",
            args.start_season,
        ]
        if args.no_resume:
            scrape_args.append("--no-resume")
        else:
            scrape_args.append("--resume")
        if args.end_season:
            scrape_args.extend(["--end-season", args.end_season])
        if args.refresh:
            scrape_args.append("--refresh")
        if args.sample:
            scrape_args.append("--sample")
        if args.limit is not None:
            scrape_args.extend(["--limit", str(args.limit)])
        for player in args.player:
            scrape_args.extend(["--player", player])
        if RAW_PLAYER_SALARIES_PATH.exists() and PLAYER_HTML_DIR.exists() and not args.refresh:
            logger.info("Raw salaries / HTML cache present; scrape will resume without redownload when possible.")
        _run(args.python_executable, "salary.scrape", scrape_args)
    else:
        logger.info("Skipping salary.scrape")

    _run(args.python_executable, "salary.normalize")

    if not args.skip_validate:
        try:
            _run(args.python_executable, "salary.validate")
        except subprocess.CalledProcessError as exc:
            logger.error("Validation reported a structural failure; stopping run_all.")
            raise SystemExit(exc.returncode) from exc
    else:
        logger.info("Skipping salary.validate")

    print("=== Salary Pipeline Complete ===")
    print(f"Raw salaries: {RAW_PLAYER_SALARIES_PATH}")
    print(f"Salary caps: {SALARY_CAPS_PATH}")
    print(f"Canonical salaries: {PLAYER_SALARIES_PATH}")
    print(f"Validation report: {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
