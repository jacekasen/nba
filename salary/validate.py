"""Validate canonical player salary datasets and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data import PLAYER_URL_PATTERN, parse_year_id
from salary.config import (
    KNOWN_PROBLEMATIC_SEASONS,
    PLAYER_SALARIES_PATH,
    RAW_PLAYER_SALARIES_PATH,
    SALARY_CAPS_PATH,
    SCRAPE_FAILURES_PATH,
    VALIDATION_REPORT_PATH,
)
from salary.parsing import season_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PLAYER_SALARIES_PATH, help="Canonical player salaries CSV")
    parser.add_argument("--raw", type=Path, default=RAW_PLAYER_SALARIES_PATH, help="Raw scraped salaries CSV")
    parser.add_argument("--caps", type=Path, default=SALARY_CAPS_PATH, help="Salary caps CSV")
    parser.add_argument("--failures", type=Path, default=SCRAPE_FAILURES_PATH, help="Scrape failures CSV")
    parser.add_argument("--report", type=Path, default=VALIDATION_REPORT_PATH, help="JSON validation report path")
    parser.add_argument(
        "--fail-on-structural",
        action="store_true",
        default=True,
        help="Exit non-zero on serious structural failures (default)",
    )
    parser.add_argument(
        "--no-fail-on-structural",
        action="store_false",
        dest="fail_on_structural",
        help="Always exit 0 after writing the report",
    )
    return parser.parse_args(argv)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def validate_salary_dataset(
    player_salaries: pd.DataFrame,
    *,
    raw: pd.DataFrame | None = None,
    caps: pd.DataFrame | None = None,
    failures: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run structural and data-quality checks; return a report dictionary."""
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    required = [
        "player_id",
        "player_name",
        "player_url",
        "season",
        "season_start",
        "season_end",
        "team",
        "salary",
        "salary_cap",
        "cap_share",
        "salary_source",
        "salary_quality",
    ]
    missing_cols = [col for col in required if col not in player_salaries.columns]
    if missing_cols:
        issues.append({"code": "missing_columns", "detail": missing_cols})

    df = player_salaries.copy()
    n_rows = int(len(df))
    n_players = int(df["player_id"].nunique()) if "player_id" in df.columns and n_rows else 0

    # Duplicate player-season-team rows.
    dup_mask = df.duplicated(subset=["player_id", "season", "team"], keep=False) if n_rows else pd.Series(dtype=bool)
    duplicate_count = int(dup_mask.sum()) if n_rows else 0
    if duplicate_count:
        issues.append(
            {
                "code": "duplicate_player_season_team",
                "count": duplicate_count,
                "examples": df.loc[dup_mask, ["player_id", "season", "team"]].head(10).to_dict(orient="records"),
            }
        )

    # Malformed seasons / inconsistent parsed bounds.
    malformed_seasons = 0
    inconsistent_bounds = 0
    if n_rows:
        for _, row in df.iterrows():
            season, start, end = parse_year_id(row.get("season"))
            if season is None or start is None or end is None:
                malformed_seasons += 1
                continue
            if int(row.get("season_start")) != start or int(row.get("season_end")) != end:
                inconsistent_bounds += 1
    if malformed_seasons:
        issues.append({"code": "malformed_seasons", "count": malformed_seasons})
    if inconsistent_bounds:
        issues.append({"code": "inconsistent_season_bounds", "count": inconsistent_bounds})

    # Missing player IDs / invalid URLs / identity collisions.
    missing_player_ids = int(df["player_id"].isna().sum()) if n_rows else 0
    if missing_player_ids:
        issues.append({"code": "missing_player_ids", "count": missing_player_ids})

    invalid_urls = 0
    if n_rows:
        invalid_urls = int((~df["player_url"].astype(str).map(lambda u: bool(PLAYER_URL_PATTERN.match(u)))).sum())
    if invalid_urls:
        issues.append({"code": "invalid_player_urls", "count": invalid_urls})

    identity_collisions = 0
    if n_rows:
        names_per_id = df.groupby("player_id")["player_name"].nunique()
        identity_collisions = int((names_per_id > 1).sum())
        if identity_collisions:
            warnings.append(
                {
                    "code": "player_identity_name_collisions",
                    "count": identity_collisions,
                    "examples": names_per_id[names_per_id > 1].head(10).to_dict(),
                }
            )
        urls_per_id = df.groupby("player_id")["player_url"].nunique()
        url_collisions = int((urls_per_id > 1).sum())
        if url_collisions:
            issues.append({"code": "player_id_url_collisions", "count": url_collisions})

    # Salary value checks.
    negative_salaries = int((pd.to_numeric(df["salary"], errors="coerce") < 0).sum()) if n_rows else 0
    if negative_salaries:
        issues.append({"code": "negative_salaries", "count": negative_salaries})

    zero_salaries = int((pd.to_numeric(df["salary"], errors="coerce") == 0).sum()) if n_rows else 0
    if zero_salaries:
        warnings.append({"code": "zero_salaries", "count": zero_salaries})

    missing_salary_count = int(pd.to_numeric(df["salary"], errors="coerce").isna().sum()) if n_rows else 0
    known_salary_count = n_rows - missing_salary_count

    # Cap join failures.
    missing_caps = int(pd.to_numeric(df["salary_cap"], errors="coerce").isna().sum()) if n_rows else 0
    if missing_caps:
        warnings.append({"code": "missing_salary_caps", "count": missing_caps})

    # Extreme cap shares (not automatically invalid).
    cap_share = pd.to_numeric(df["cap_share"], errors="coerce") if n_rows else pd.Series(dtype=float)
    extreme_high = int((cap_share > 100).sum()) if n_rows else 0
    extreme_low = int((cap_share < 0).sum()) if n_rows else 0
    if extreme_high:
        warnings.append(
            {
                "code": "extreme_cap_share_gt_100",
                "count": extreme_high,
                "examples": df.loc[cap_share > 100, ["player_id", "season", "team", "salary", "salary_cap", "cap_share"]]
                .head(10)
                .to_dict(orient="records"),
            }
        )
    if extreme_low:
        issues.append({"code": "negative_cap_share", "count": extreme_low})

    # Season coverage / sparsity.
    rows_per_season: dict[str, int] = {}
    if n_rows:
        rows_per_season = {
            str(season): int(count)
            for season, count in df.groupby("season").size().sort_index(key=lambda s: s.map(season_sort_key)).items()
        }
    first_season = next(iter(rows_per_season), None)
    last_season = next(reversed(rows_per_season), None) if rows_per_season else None

    sparse_seasons = []
    # Only evaluate league-wide sparsity once the dataset is large enough that a
    # tiny sample (e.g. --sample) would not falsely look sparse every season.
    if rows_per_season and n_players >= 50:
        counts = list(rows_per_season.values())
        median = sorted(counts)[len(counts) // 2]
        threshold = max(20, int(median * 0.15)) if median else 20
        for season, count in rows_per_season.items():
            if count < threshold:
                sparse_seasons.append({"season": season, "rows": count, "threshold": threshold})
    if sparse_seasons:
        warnings.append({"code": "unexpectedly_sparse_seasons", "seasons": sparse_seasons})

    problematic_season_stats = []
    for season in KNOWN_PROBLEMATIC_SEASONS:
        count = rows_per_season.get(season, 0)
        problematic_season_stats.append({"season": season, "rows": count})
        if count == 0:
            warnings.append({"code": "known_problematic_season_empty", "season": season})

    # Salary quality breakdown.
    quality_breakdown: dict[str, int] = {}
    if n_rows and "salary_quality" in df.columns:
        quality_breakdown = {str(k): int(v) for k, v in df["salary_quality"].fillna("missing").value_counts().items()}

    # Raw duplicate source rows.
    duplicate_source_rows = 0
    if raw is not None and not raw.empty:
        duplicate_source_rows = int(raw.duplicated(subset=["player_id", "season", "team"], keep=False).sum())
        if duplicate_source_rows:
            warnings.append({"code": "duplicate_source_rows", "count": duplicate_source_rows})

    # Caps continuity.
    if caps is not None and not caps.empty:
        cap_seasons = sorted(caps["season"].astype(str), key=season_sort_key)
        if cap_seasons:
            start = parse_year_id(cap_seasons[0])[1]
            end = parse_year_id(cap_seasons[-1])[1]
            if start is not None and end is not None:
                expected = end - start + 1
                if len(cap_seasons) < expected:
                    warnings.append(
                        {
                            "code": "salary_cap_season_gaps",
                            "expected_seasons": expected,
                            "observed_seasons": len(cap_seasons),
                        }
                    )

    failed_pages = 0
    failed_page_examples: list[dict[str, str]] = []
    if failures is not None and not failures.empty:
        failed_pages = int(len(failures))
        failed_page_examples = failures.head(20).to_dict(orient="records")
        warnings.append({"code": "scrape_failures", "count": failed_pages})

    # Parsing failures approximated by players with HTML but zero extractable rows are
    # not always knowable here; rely on scrape failures + empty sample checks.
    parsing_failures = failed_pages

    structural_failure_codes = {
        "missing_columns",
        "duplicate_player_season_team",
        "malformed_seasons",
        "missing_player_ids",
        "invalid_player_urls",
        "player_id_url_collisions",
        "negative_salaries",
        "negative_cap_share",
    }
    structural_failures = [item for item in issues if item.get("code") in structural_failure_codes]

    report: dict[str, Any] = {
        "number_of_players": n_players,
        "number_of_salary_rows": n_rows,
        "first_season": first_season,
        "last_season": last_season,
        "rows_per_season": rows_per_season,
        "known_salary_count": known_salary_count,
        "missing_salary_count": missing_salary_count,
        "known_salary_percentage": _pct(known_salary_count, n_rows),
        "missing_salary_percentage": _pct(missing_salary_count, n_rows),
        "salary_quality_breakdown": quality_breakdown,
        "salary_cap_join_failures": missing_caps,
        "duplicate_count": duplicate_count,
        "duplicate_source_rows": duplicate_source_rows,
        "parsing_failures": parsing_failures,
        "failed_pages": failed_pages,
        "failed_page_examples": failed_page_examples,
        "known_problematic_seasons": problematic_season_stats,
        "issues": issues,
        "warnings": warnings,
        "structural_failure": bool(structural_failures) or n_rows == 0,
        "structural_failures": structural_failures,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=== Salary Validation Summary ===")
    print(f"Players: {report['number_of_players']:,}")
    print(f"Salary rows: {report['number_of_salary_rows']:,}")
    print(f"First season: {report['first_season']}")
    print(f"Last season: {report['last_season']}")
    print(f"Known salary %: {report['known_salary_percentage']}")
    print(f"Missing salary %: {report['missing_salary_percentage']}")
    print(f"Salary-quality breakdown: {report['salary_quality_breakdown']}")
    print(f"Salary-cap join failures: {report['salary_cap_join_failures']:,}")
    print(f"Duplicate player-season-team rows: {report['duplicate_count']:,}")
    print(f"Parsing/scrape failures: {report['parsing_failures']:,}")
    print(f"Known problematic seasons: {report['known_problematic_seasons']}")
    if report["issues"]:
        print(f"Issues ({len(report['issues'])}):")
        for item in report["issues"]:
            print(f"  - {item}")
    if report["warnings"]:
        print(f"Warnings ({len(report['warnings'])}):")
        for item in report["warnings"]:
            print(f"  - {item.get('code')}: { {k: v for k, v in item.items() if k != 'code'} }")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Canonical salaries not found: {args.input}. Run normalize first.")

    player_salaries = pd.read_csv(args.input)
    raw = pd.read_csv(args.raw) if args.raw.exists() else None
    caps = pd.read_csv(args.caps) if args.caps.exists() else None
    failures = pd.read_csv(args.failures) if args.failures.exists() else None

    report = validate_salary_dataset(player_salaries, raw=raw, caps=caps, failures=failures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_report(report)
    print(f"Wrote validation report: {args.report}")

    if args.fail_on_structural and report["structural_failure"]:
        logger.error("Structural validation failures detected.")
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
