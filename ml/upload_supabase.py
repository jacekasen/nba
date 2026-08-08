"""Upload player prediction rows to Supabase with safe dry-run defaults."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import pandas as pd

from ml.config import CURRENT_PREDICTIONS_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CURRENT_PREDICTIONS_PATH, help="Predictions CSV to upload")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per batch")
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
    return parser.parse_args()


def _normalize_prediction_factors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def build_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Map CSV columns to Supabase table payload."""
    rows: list[dict[str, Any]] = []
    required = [
        "player_id",
        "player_name",
        "season",
        "trajectory",
        "improving_probability",
        "stable_probability",
        "regressing_probability",
        "predicted_bpm_delta",
        "model_version",
        "updated_at",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for _, row in df.iterrows():
        item = {
            "player_id": str(row["player_id"]),
            "player_name": str(row["player_name"]),
            "season": str(row["season"]),
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "current_bpm": float(row["current_bpm"]) if pd.notna(row.get("current_bpm")) else None,
            "trajectory": str(row["trajectory"]),
            "improving_probability": float(row["improving_probability"]),
            "stable_probability": float(row["stable_probability"]),
            "regressing_probability": float(row["regressing_probability"]),
            "peak_probability": float(row["peak_probability"]) if pd.notna(row.get("peak_probability")) else None,
            "predicted_bpm_delta": float(row["predicted_bpm_delta"]),
            "model_version": str(row["model_version"]),
            "prediction_factors": _normalize_prediction_factors(row.get("prediction_factors")),
            "updated_at": str(row["updated_at"]),
        }
        rows.append(item)
    return rows


def _validate_payload(payload: list[dict[str, Any]]) -> None:
    for item in payload:
        probs = [
            item["improving_probability"],
            item["stable_probability"],
            item["regressing_probability"],
        ]
        for prob in probs:
            if prob < 0 or prob > 1:
                raise ValueError(f"Out-of-range probability in payload for {item['player_id']}")
        if abs(sum(probs) - 1.0) > 1e-3:
            raise ValueError(f"Probabilities do not sum to 1 for {item['player_id']}")


def _upsert_batch(supabase_url: str, service_role_key: str, batch: list[dict[str, Any]]) -> None:
    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/player_predictions?on_conflict=player_id"
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
        with request.urlopen(req, timeout=30):
            return
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase upsert failed (HTTP {exc.code}): {details}") from exc


def upload_payload(payload: list[dict[str, Any]], batch_size: int, apply: bool) -> None:
    """Upload payload to Supabase, unless running in dry mode."""
    if not apply:
        print("Dry-run enabled; skipping network writes.")
        return

    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for --apply uploads.")

    for start in range(0, len(payload), batch_size):
        batch = payload[start : start + batch_size]
        _upsert_batch(supabase_url=supabase_url, service_role_key=service_role_key, batch=batch)
        print(f"Uploaded rows {start + 1}-{start + len(batch)}")


def main() -> None:
    """CLI entry point for payload validation and optional upload."""
    args = _parse_args()
    apply = bool(args.apply)

    df = pd.read_csv(args.input)
    payload = build_payload(df)
    _validate_payload(payload)

    print("=== Supabase Upload Summary ===")
    print(f"Rows prepared: {len(payload):,}")
    if payload:
        print(f"Sample player_id: {payload[0]['player_id']}")

    upload_payload(payload=payload, batch_size=args.batch_size, apply=apply)


if __name__ == "__main__":
    main()
