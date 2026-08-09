"""Upload salary datasets to Supabase with safe dry-run defaults."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

from salary.config import PLAYER_SALARIES_PATH, SALARY_CAPS_PATH, TEAM_SEASON_SALARIES_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salaries", type=Path, default=PLAYER_SALARIES_PATH, help="Canonical player salaries CSV")
    parser.add_argument("--caps", type=Path, default=SALARY_CAPS_PATH, help="Salary caps CSV")
    parser.add_argument(
        "--team-seasons",
        type=Path,
        default=TEAM_SEASON_SALARIES_PATH,
        help="Derived team-season salaries CSV",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per upsert batch")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate payload and print summary without network writes (default true).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform remote upsert. Must be explicitly set to override dry-run behavior.",
    )
    parser.add_argument(
        "--skip-caps",
        action="store_true",
        help="Do not upload salary_caps rows.",
    )
    parser.add_argument(
        "--skip-team-seasons",
        action="store_true",
        help="Do not upload team_season_salaries rows.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:  # noqa: BLE001
            return None
    return value


def _json_int(value: Any) -> int | None:
    """Coerce CSV/pandas numerics to JSON integers (never 2125000.0)."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return int(float(text))
    return int(value)


def _json_float(value: Any) -> float | None:
    raw = _json_safe(value)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(raw)


def build_player_salary_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = [
        "player_id",
        "player_name",
        "player_url",
        "season",
        "season_start",
        "season_end",
        "team",
        "salary_source",
        "salary_quality",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required salary columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "player_id": str(row["player_id"]),
                "player_name": str(row["player_name"]),
                "player_url": str(row["player_url"]),
                "season": str(row["season"]),
                "season_start": int(row["season_start"]),
                "season_end": int(row["season_end"]),
                "team": str(row["team"]),
                "salary": _json_int(row.get("salary")),
                "salary_cap": _json_int(row.get("salary_cap")),
                "cap_share": _json_float(row.get("cap_share")),
                "salary_source": str(row["salary_source"]),
                "salary_quality": str(row["salary_quality"]),
                "team_payroll_share": _json_float(row.get("team_payroll_share")),
                "team_known_salary_total": _json_int(row.get("team_known_salary_total")),
                "team_known_player_count": _json_int(row.get("team_known_player_count")),
            }
        )
    return rows


def build_salary_cap_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = ["season", "season_start", "season_end", "salary_cap"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required salary-cap columns: {missing}")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "season": str(row["season"]),
                "season_start": int(row["season_start"]),
                "season_end": int(row["season_end"]),
                "salary_cap": _json_int(row["salary_cap"]),
            }
        )
    return rows


def build_team_season_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = ["team", "season", "season_start", "season_end"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required team-season columns: {missing}")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "team": str(row["team"]),
                "season": str(row["season"]),
                "season_start": int(row["season_start"]),
                "season_end": int(row["season_end"]),
                "salary_cap": _json_int(row.get("salary_cap")),
                "team_known_salary_total": _json_int(row.get("team_known_salary_total")),
                "team_known_player_count": _json_int(row.get("team_known_player_count")),
                "total_cap_share": _json_float(row.get("total_cap_share")),
                "max_cap_share": _json_float(row.get("max_cap_share")),
                "payroll_share_available": bool(row.get("payroll_share_available"))
                if pd.notna(row.get("payroll_share_available"))
                else False,
                "team_payroll_vs_cap": _json_float(row.get("team_payroll_vs_cap")),
            }
        )
    return rows


def _validate_player_payload(payload: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for item in payload:
        key = (item["player_id"], item["season"], item["team"])
        if key in seen:
            raise ValueError(f"Duplicate payload key: {key}")
        seen.add(key)
        if item["salary"] is not None and item["salary"] < 0:
            raise ValueError(f"Negative salary for {key}")
        if item["cap_share"] is not None and item["cap_share"] < 0:
            raise ValueError(f"Negative cap_share for {key}")


def _upsert_batch(supabase_url: str, service_role_key: str, table: str, on_conflict: str, batch: list[dict[str, Any]]) -> None:
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table}?on_conflict={on_conflict}"
    body = json.dumps(batch).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    try:
        with request.urlopen(req, timeout=60):
            return
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase upsert failed for {table} (HTTP {exc.code}): {details}") from exc


def upload_table(
    *,
    table: str,
    on_conflict: str,
    payload: list[dict[str, Any]],
    batch_size: int,
    apply: bool,
) -> None:
    print(f"{table}: prepared {len(payload):,} rows")
    if not apply:
        return
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for --apply uploads.")
    for start in range(0, len(payload), batch_size):
        batch = payload[start : start + batch_size]
        _upsert_batch(supabase_url, service_role_key, table, on_conflict, batch)
        print(f"  uploaded {table} rows {start + 1}-{start + len(batch)}")


def main() -> None:
    args = _parse_args()
    apply = bool(args.apply)

    salaries = pd.read_csv(args.salaries)
    player_payload = build_player_salary_payload(salaries)
    _validate_player_payload(player_payload)

    caps_payload: list[dict[str, Any]] = []
    if not args.skip_caps and args.caps.exists():
        caps_payload = build_salary_cap_payload(pd.read_csv(args.caps))

    team_payload: list[dict[str, Any]] = []
    if not args.skip_team_seasons and args.team_seasons.exists():
        team_payload = build_team_season_payload(pd.read_csv(args.team_seasons))

    print("=== Supabase Salary Upload Summary ===")
    print(f"Dry-run: {not apply}")
    upload_table(
        table="player_salaries",
        on_conflict="player_id,season,team",
        payload=player_payload,
        batch_size=args.batch_size,
        apply=apply,
    )
    if caps_payload:
        upload_table(
            table="salary_caps",
            on_conflict="season",
            payload=caps_payload,
            batch_size=args.batch_size,
            apply=apply,
        )
    if team_payload:
        upload_table(
            table="team_season_salaries",
            on_conflict="team,season",
            payload=team_payload,
            batch_size=args.batch_size,
            apply=apply,
        )
    if not apply:
        print("Dry-run enabled; skipping network writes.")


if __name__ == "__main__":
    main()
