"""Validate and upload Dunks & Threes EPM exports to Supabase."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

DEFAULT_INPUT_DIR = Path("data/EPM")
TABLE = "nba_player_epm"
CONFLICT_COLUMNS = ("nba_player_id", "season", "season_type")
FILENAME_PATTERN = re.compile(r"epm_actual_(?P<start>\d{4})-(?P<suffix>\d{2})\.csv")

COLUMN_MAP = {
    "age": "age",
    "seasontype": "season_type",
    "team_id": "team_id",
    "team_alias": "team_alias",
    "team_alias_all": "team_alias_all",
    "player_id": "nba_player_id",
    "player_name": "player_name",
    "pos_text": "position",
    "gp": "games",
    "roster_games": "roster_games",
    "start": "starts",
    "mp": "minutes",
    "rookie_year": "rookie_year",
    "inches": "height_inches",
    "weight": "weight_lbs",
    "mpg": "minutes_per_game",
    "off": "offensive_epm",
    "def": "defensive_epm",
    "tot": "epm",
    "ewins": "estimated_wins",
    "usg": "usage_rate",
    "tspct": "true_shooting_pct",
    "efg": "effective_fg_pct",
    "fgpct_rim": "rim_fg_pct",
    "fgpct_mid": "midrange_fg_pct",
    "fg2pct": "two_point_fg_pct",
    "fg3pct": "three_point_fg_pct",
    "ftpct": "free_throw_pct",
    "orbpct": "offensive_rebound_pct",
    "drbpct": "defensive_rebound_pct",
    "astpct": "assist_pct",
    "topct": "turnover_pct",
    "stlpct": "steal_pct",
    "blkpct": "block_pct",
    "fga_rim_75": "rim_fga_per_75",
    "fga_mid_75": "midrange_fga_per_75",
    "fg3a_75": "three_point_fga_per_75",
    "fta_75": "fta_per_75",
    "fga_75": "fga_per_75",
}

INTEGER_FIELDS = {
    "nba_player_id",
    "season_start",
    "season_end",
    "season_type",
    "age",
    "team_id",
    "games",
    "roster_games",
    "starts",
    "rookie_year",
    "height_inches",
    "weight_lbs",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate without network writes (default).",
    )
    parser.add_argument("--apply", action="store_true", help="Perform remote upserts.")
    return parser.parse_args()


def _season_from_filename(path: Path) -> tuple[str, int, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected EPM filename: {path.name}")
    season_start = int(match.group("start"))
    season_end = season_start + 1
    expected_suffix = season_end % 100
    if int(match.group("suffix")) != expected_suffix:
        raise ValueError(f"Invalid season suffix in {path.name}")
    return f"{season_start}-{expected_suffix:02d}", season_start, season_end


def load_epm_exports(input_dir: Path) -> pd.DataFrame:
    """Load exports, validate their seasons, and map columns to database names."""
    paths = sorted(input_dir.glob("epm_actual_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No EPM exports found in {input_dir}")

    frames: list[pd.DataFrame] = []
    uploaded_at = datetime.now(timezone.utc).isoformat()
    for path in paths:
        season, season_start, season_end = _season_from_filename(path)
        raw = pd.read_csv(path)
        missing = sorted(set(COLUMN_MAP) - set(raw.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")

        raw_seasons = set(pd.to_numeric(raw["season"], errors="raise").astype(int))
        if raw_seasons != {season_end}:
            raise ValueError(f"{path.name} contains season values {sorted(raw_seasons)}")

        frame = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
        frame.insert(1, "season", season)
        frame.insert(2, "season_start", season_start)
        frame.insert(3, "season_end", season_end)
        frame["source_file"] = path.name
        frame["updated_at"] = uploaded_at
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_mask = combined.duplicated(list(CONFLICT_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicates = combined.loc[duplicate_mask, list(CONFLICT_COLUMNS)].head(20)
        raise ValueError(f"Duplicate EPM keys:\n{duplicates.to_string(index=False)}")
    return combined


def _json_safe(column: str, value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if column in INTEGER_FIELDS:
        return int(value)
    return value


def build_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _json_safe(column, value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _upsert_batch(
    supabase_url: str,
    service_role_key: str,
    batch: list[dict[str, Any]],
) -> None:
    conflict = ",".join(CONFLICT_COLUMNS)
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{TABLE}?on_conflict={conflict}"
    req = request.Request(
        endpoint,
        data=json.dumps(batch, allow_nan=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    try:
        with request.urlopen(req, timeout=90):
            return
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase EPM upsert failed (HTTP {exc.code}): {details}") from exc


def remote_row_count(supabase_url: str, service_role_key: str) -> int:
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{TABLE}?select=nba_player_id"
    req = request.Request(
        endpoint,
        method="GET",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Prefer": "count=exact",
            "Range": "0-0",
        },
    )
    with request.urlopen(req, timeout=30) as response:
        content_range = response.headers.get("Content-Range")
    if not content_range or "/" not in content_range:
        raise RuntimeError("Supabase did not return an exact EPM row count")
    return int(content_range.rsplit("/", 1)[1])


def main() -> None:
    args = _parse_args()
    frame = load_epm_exports(args.input_dir)
    payload = build_payload(frame)

    print("=== Supabase EPM Upload Summary ===")
    print(f"Files: {frame['source_file'].nunique():,}")
    print(f"Rows: {len(payload):,}")
    print(f"Players: {frame['nba_player_id'].nunique():,}")
    print(f"Season range: {frame['season'].min()} -> {frame['season'].max()}")

    if not args.apply:
        print("Dry-run enabled; skipping network writes.")
        return

    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for --apply")

    for start in range(0, len(payload), args.batch_size):
        batch = payload[start : start + args.batch_size]
        _upsert_batch(supabase_url, service_role_key, batch)
        print(f"Uploaded rows {start + 1:,}-{start + len(batch):,}")

    remote_count = remote_row_count(supabase_url, service_role_key)
    print(f"Remote table row count: {remote_count:,}")


if __name__ == "__main__":
    main()
