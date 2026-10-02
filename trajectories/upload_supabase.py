"""Upload normalized player game logs to Supabase with dry-run protection."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pandas as pd

from trajectories.config import PARQUET_OUTPUT_PATH, REPO_ROOT, SUPABASE_TABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def get_credentials(use_publishable_key: bool = False) -> tuple[str, str]:
    local_env = _load_env_file(REPO_ROOT / ".env.local")
    url = os.getenv("SUPABASE_URL") or local_env.get("SUPABASE_URL")
    if use_publishable_key:
        key = (
            os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
            or local_env.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
        )
    else:
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_KEY")
            or local_env.get("SUPABASE_SERVICE_ROLE_KEY")
            or local_env.get("SUPABASE_KEY")
        )
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or API key in environment or .env.local")
    return url.rstrip("/"), key


def build_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build payload records ready for Supabase upsert."""
    records = df.to_dict(orient="records")
    clean_records = []
    for r in records:
        clean_row = {}
        for k, v in r.items():
            if pd.isna(v) or v is None:
                clean_row[k] = None
            elif isinstance(v, (bool, int, float, str)):
                clean_row[k] = v
            else:
                clean_row[k] = str(v)
        clean_records.append(clean_row)
    return clean_records


def upsert_rows(
    supabase_url: str,
    service_key: str,
    table: str,
    rows: list[dict[str, Any]],
    batch_size: int = 500,
) -> int:
    """Upsert records into Supabase in batches."""
    endpoint = f"{supabase_url}/rest/v1/{table}?on_conflict=player_id,game_id,team_id"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    total_uploaded = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        data = json.dumps(chunk).encode("utf-8")
        req = request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 201, 204):
                    raise RuntimeError(f"Supabase returned status {resp.status}")
                total_uploaded += len(chunk)
                logger.info("Batch %s/%s uploaded (%s rows)", batch_num, total_batches, len(chunk))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("HTTP %s on batch %s: %s", exc.code, batch_num, body)
            raise

    return total_uploaded


def upload_dataset(
    *,
    input_path: Path = PARQUET_OUTPUT_PATH,
    season: str | None = None,
    batch_size: int = 500,
    dry_run: bool = True,
    apply: bool = False,
    use_publishable_key: bool = False,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    logger.info("Reading %s...", input_path)
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)

    if season:
        df = df[df["season"] == season].reset_index(drop=True)
        logger.info("Filtered to season %s (%s rows)", season, len(df))

    rows = build_payload(df)
    logger.info("Prepared %s records for table '%s'", len(rows), SUPABASE_TABLE)

    if not apply or dry_run:
        logger.info("DRY RUN: skipping remote upload. Pass --apply to perform upload.")
        return

    supabase_url, key = get_credentials(use_publishable_key=use_publishable_key)
    uploaded = upsert_rows(supabase_url, key, SUPABASE_TABLE, rows, batch_size=batch_size)
    logger.info("Successfully uploaded %s rows to Supabase table '%s'", uploaded, SUPABASE_TABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PARQUET_OUTPUT_PATH, help="Path to input parquet/csv")
    parser.add_argument("--season", type=str, default=None, help="Filter to single season")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Validate without writing")
    parser.add_argument("--apply", action="store_true", help="Perform remote upload")
    parser.add_argument(
        "--use-publishable-key",
        action="store_true",
        help="Use NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY instead of service role key",
    )
    args = parser.parse_args()

    upload_dataset(
        input_path=args.input,
        season=args.season,
        batch_size=args.batch_size,
        dry_run=args.dry_run and not args.apply,
        apply=args.apply,
        use_publishable_key=args.use_publishable_key,
    )


if __name__ == "__main__":
    main()
