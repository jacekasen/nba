"""Build the canonical player-season modeling dataset from raw scraped stats."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import MODELING_DIR, ModelingThresholds, PLAYER_SEASONS_PATH, RAW_STATS_PATH

NUMERIC_COLUMNS = ["age", "games", "mp", "per", "bpm", "vorp", "ws", "ws_per_48"]
YEAR_PATTERN = re.compile(r"^(?P<start>\d{4})-(?P<suffix>\d{2})$")
NTM_PATTERN = re.compile(r"^\d+TM$")
PLAYER_URL_PATTERN = re.compile(r"^https://www\.basketball-reference\.com/players/[a-z]/(?P<id>[a-z0-9]+)\.html$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_STATS_PATH, help="Raw season-by-season stats CSV")
    parser.add_argument("--output", type=Path, default=PLAYER_SEASONS_PATH, help="Output modeling player-seasons CSV")
    parser.add_argument("--min-games", type=int, default=15, help="Minimum games required")
    parser.add_argument("--min-minutes", type=int, default=250, help="Minimum total minutes required")
    parser.add_argument(
        "--required-metrics",
        nargs="*",
        default=["bpm", "per", "ws", "ws_per_48"],
        help="Metrics that must be non-null for a qualified season",
    )
    parser.add_argument(
        "--qa-json",
        type=Path,
        default=MODELING_DIR / "player_seasons_qa.json",
        help="Path to write QA summary JSON",
    )
    return parser.parse_args()


def normalize_player_url(url: Any) -> str | None:
    """Normalize Basketball-Reference player URL formats and reject invalid values."""
    if pd.isna(url):
        return None
    text = str(url).strip()
    if not text:
        return None
    if text.startswith("/players/"):
        text = f"https://www.basketball-reference.com{text}"
    text = text.split("?")[0].split("#")[0]
    if PLAYER_URL_PATTERN.match(text):
        return text
    return None


def parse_year_id(year_id: Any) -> tuple[str | None, int | None, int | None]:
    """Parse season string like '2024-25' into season/start/end values."""
    if pd.isna(year_id):
        return None, None, None
    season = str(year_id).strip()
    match = YEAR_PATTERN.match(season)
    if not match:
        return None, None, None
    season_start = int(match.group("start"))
    suffix = int(match.group("suffix"))
    century = season_start // 100
    season_end = century * 100 + suffix
    if season_end < season_start:
        season_end += 100
    return season, season_start, season_end


def derive_player_id(player_url: str) -> str:
    """Extract stable Basketball-Reference player id from player URL."""
    match = PLAYER_URL_PATTERN.match(player_url)
    if match is None:
        raise ValueError(f"Invalid player_url cannot derive player_id: {player_url}")
    return match.group("id")


def collapse_player_seasons(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse rows to one record per player_url+season with deterministic tie-breaking."""

    qa = {
        "groups_with_multiple_rows": 0,
        "groups_using_ntm_row": 0,
        "groups_resolved_without_ntm": 0,
        "groups_with_duplicate_ntm_rows": 0,
    }

    def choose_group_row(group: pd.DataFrame) -> pd.Series:
        if len(group) > 1:
            qa["groups_with_multiple_rows"] += 1

        ntm_mask = group["team_name_abbr"].astype(str).str.match(NTM_PATTERN, na=False)
        if ntm_mask.any():
            ntm_rows = group.loc[ntm_mask].copy()
            if len(ntm_rows) > 1:
                qa["groups_with_duplicate_ntm_rows"] += 1
            qa["groups_using_ntm_row"] += 1
            chosen_pool = ntm_rows
        else:
            chosen_pool = group
            if len(group) > 1:
                qa["groups_resolved_without_ntm"] += 1

        order_cols = ["games", "mp", "bpm", "per", "ws", "ws_per_48", "team_name_abbr"]
        ascending = [False, False, False, False, False, False, True]
        chosen = chosen_pool.sort_values(order_cols, ascending=ascending).iloc[0]
        return chosen

    collapsed = (
        df.groupby(["player_url", "season"], sort=False, as_index=False)
        .apply(choose_group_row, include_groups=False)
        .reset_index(drop=True)
    )
    return collapsed, qa


def build_modeling_dataset(raw_path: Path, thresholds: ModelingThresholds) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, clean, and collapse raw stats into a canonical modeling table."""
    raw = pd.read_csv(raw_path)
    qa: dict[str, Any] = {
        "input_rows": int(len(raw)),
        "input_unique_players": int(raw["player_url"].nunique(dropna=True)),
    }

    raw["player_url"] = raw["player_url"].apply(normalize_player_url)
    invalid_url_mask = raw["player_url"].isna()
    qa["rejected_invalid_player_url_rows"] = int(invalid_url_mask.sum())
    cleaned = raw.loc[~invalid_url_mask].copy()

    dnp_mask = cleaned["team_name_abbr"].astype(str).str.startswith("Did not play", na=False)
    qa["removed_dnp_rows"] = int(dnp_mask.sum())
    cleaned = cleaned.loc[~dnp_mask].copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    parsed = cleaned["year_id"].apply(parse_year_id)
    cleaned[["season", "season_start", "season_end"]] = pd.DataFrame(parsed.tolist(), index=cleaned.index)

    invalid_season_mask = cleaned["season_start"].isna() | cleaned["season_end"].isna()
    qa["rejected_invalid_season_rows"] = int(invalid_season_mask.sum())
    cleaned = cleaned.loc[~invalid_season_mask].copy()

    collapsed, collapse_qa = collapse_player_seasons(cleaned)
    qa.update(collapse_qa)

    collapsed["player_id"] = collapsed["player_url"].map(derive_player_id)

    incomplete_mask = (
        (collapsed["games"] < thresholds.min_games)
        | (collapsed["mp"] < thresholds.min_minutes)
        | collapsed[list(thresholds.require_non_null_metrics)].isna().any(axis=1)
    )
    qa["removed_incomplete_rows"] = int(incomplete_mask.sum())
    output = collapsed.loc[~incomplete_mask].copy()

    output = output.sort_values(["player_url", "season_start", "season_end"]).reset_index(drop=True)

    duplicate_keys = output.duplicated(subset=["player_url", "season"], keep=False)
    qa["duplicate_player_season_rows_post_collapse"] = int(duplicate_keys.sum())
    if duplicate_keys.any():
        output = output.drop_duplicates(subset=["player_url", "season"], keep="first").copy()

    output["consecutive_next"] = output.groupby("player_url")["season_start"].shift(-1) - output["season_start"]
    qa["consecutive_season_pairs"] = int((output["consecutive_next"] == 1).sum())
    output = output.drop(columns=["consecutive_next"])

    qa["output_rows"] = int(len(output))
    qa["output_unique_players"] = int(output["player_url"].nunique())
    qa["season_start_min"] = int(output["season_start"].min()) if not output.empty else None
    qa["season_start_max"] = int(output["season_start"].max()) if not output.empty else None

    return output, qa


def print_qa(qa: dict[str, Any]) -> None:
    """Print concise QA summary for command-line runs."""
    print("=== Modeling Dataset QA Summary ===")
    print(f"Input rows: {qa['input_rows']:,}")
    print(f"Output rows: {qa['output_rows']:,}")
    print(f"Unique players: {qa['output_unique_players']:,}")
    print(f"Duplicate player-seasons post-collapse: {qa['duplicate_player_season_rows_post_collapse']:,}")
    print(f"Removed DNP records: {qa['removed_dnp_rows']:,}")
    print(f"Rejected invalid player_url rows: {qa['rejected_invalid_player_url_rows']:,}")
    print(f"Rejected invalid season rows: {qa['rejected_invalid_season_rows']:,}")
    print(f"Removed incomplete records: {qa['removed_incomplete_rows']:,}")
    print(f"Consecutive season pairs: {qa['consecutive_season_pairs']:,}")
    print(f"Season range: {qa['season_start_min']} -> {qa['season_start_max']}")
    print("Collapse details:")
    print(f"  Groups with multiple rows: {qa['groups_with_multiple_rows']:,}")
    print(f"  Groups using nTM row: {qa['groups_using_ntm_row']:,}")
    print(f"  Groups resolved without nTM: {qa['groups_resolved_without_ntm']:,}")
    print(f"  Groups with duplicate nTM rows: {qa['groups_with_duplicate_ntm_rows']:,}")


def main() -> None:
    """CLI entry point for modeling dataset creation."""
    args = _parse_args()
    thresholds = ModelingThresholds(
        min_games=args.min_games,
        min_minutes=args.min_minutes,
        require_non_null_metrics=tuple(args.required_metrics),
    )
    output_df, qa = build_modeling_dataset(args.input, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    args.qa_json.parent.mkdir(parents=True, exist_ok=True)
    qa_payload = {**qa, "thresholds": asdict(thresholds)}
    args.qa_json.write_text(json.dumps(qa_payload, indent=2), encoding="utf-8")

    print_qa(qa)
    print(f"Saved modeling dataset to: {args.output}")
    print(f"Saved QA summary to: {args.qa_json}")


if __name__ == "__main__":
    main()
