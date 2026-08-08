from __future__ import annotations

import pandas as pd

from ml.predict import _validate_probabilities
from ml.upload_supabase import build_payload, upload_payload


def test_probability_validation_passes() -> None:
    df = pd.DataFrame(
        [
            {
                "improving_probability": 0.6,
                "stable_probability": 0.3,
                "regressing_probability": 0.1,
            }
        ]
    )
    _validate_probabilities(df, ["improving", "stable", "regressing"])


def test_supabase_payload_formatting() -> None:
    df = pd.DataFrame(
        [
            {
                "player_id": "jamesle01",
                "player_name": "LeBron James",
                "season": "2024-25",
                "age": 40,
                "current_bpm": 6.2,
                "trajectory": "stable",
                "improving_probability": 0.2,
                "stable_probability": 0.6,
                "regressing_probability": 0.2,
                "peak_probability": 0.45,
                "predicted_bpm_delta": -0.1,
                "model_version": "1.0.0",
                "prediction_factors": '["Factor A", "Factor B"]',
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    payload = build_payload(df)
    assert payload[0]["player_id"] == "jamesle01"
    assert payload[0]["prediction_factors"] == ["Factor A", "Factor B"]


def test_upload_dry_run_does_not_require_env() -> None:
    payload = [
        {
            "player_id": "jamesle01",
            "player_name": "LeBron James",
            "season": "2024-25",
            "age": 40,
            "current_bpm": 6.2,
            "trajectory": "stable",
            "improving_probability": 0.2,
            "stable_probability": 0.6,
            "regressing_probability": 0.2,
            "peak_probability": 0.45,
            "predicted_bpm_delta": -0.1,
            "model_version": "1.0.0",
            "prediction_factors": ["Factor"],
            "updated_at": "2026-01-01T00:00:00Z",
        }
    ]
    upload_payload(payload=payload, batch_size=100, apply=False)
