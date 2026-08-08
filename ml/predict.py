"""Generate latest-player predictions for trajectory, BPM delta, and near-peak probability."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import (
    CLASSIFIER_MODEL_PATH,
    CURRENT_PREDICTIONS_PATH,
    FEATURE_COLUMNS_PATH,
    LABEL_ORDER_PATH,
    MODEL_FEATURES_PATH,
    MODEL_VERSION,
    PEAK_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=MODEL_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=CURRENT_PREDICTIONS_PATH)
    parser.add_argument(
        "--allow-incomplete-current-season",
        action="store_true",
        help="Include rows from the max season_end in source data.",
    )
    return parser.parse_args()


def _latest_rows(features_df: pd.DataFrame, allow_incomplete_current: bool) -> pd.DataFrame:
    df = features_df.copy()
    if not allow_incomplete_current:
        max_end = int(df["season_end"].max())
        df = df[df["season_end"] < max_end].copy()

    latest = (
        df.sort_values(["player_url", "season_start", "season_end"])
        .groupby("player_url", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def _format_factors(row: pd.Series) -> list[str]:
    factors: list[str] = []

    if pd.notna(row.get("bpm_delta_1")):
        delta = float(row["bpm_delta_1"])
        if delta >= 0.5:
            factors.append(f"BPM rose by {delta:+.1f} last qualified season.")
        elif delta <= -0.5:
            factors.append(f"BPM fell by {delta:+.1f} last qualified season.")
        else:
            factors.append(f"BPM was stable last season ({delta:+.1f}).")

    if pd.notna(row.get("mp_delta_1")) and pd.notna(row.get("mp")) and row["mp"] > 0:
        mp_delta = float(row["mp_delta_1"])
        pct = mp_delta / max(float(row["mp"] - mp_delta), 1.0)
        factors.append(f"Minutes changed by {pct:+.0%} vs prior season.")

    if pd.notna(row.get("distance_from_career_high_bpm")):
        dist = float(row["distance_from_career_high_bpm"])
        factors.append(f"Current BPM is {dist:.1f} below career high through this season.")

    if pd.notna(row.get("bpm_slope_3")):
        slope = float(row["bpm_slope_3"])
        factors.append(f"Three-season BPM slope is {slope:+.2f} per season.")

    if pd.notna(row.get("age")):
        factors.append(f"Player age is {int(row['age'])}.")

    # Keep a compact set for UI card readability.
    return factors[:5]


def _validate_probabilities(df: pd.DataFrame, labels: list[str]) -> None:
    for label in labels:
        col = f"{label}_probability"
        if col not in df:
            raise ValueError(f"Missing class probability column: {col}")
        if df[col].isna().any():
            raise ValueError(f"NaN values found in {col}")
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"Out-of-range probability values found in {col}")

    row_sums = df[[f"{label}_probability" for label in labels]].sum(axis=1)
    if not np.allclose(row_sums.to_numpy(), 1.0, atol=1e-3):
        raise ValueError("Class probabilities do not sum to 1 within tolerance.")


def main() -> None:
    """CLI entry point for generating current player predictions."""
    args = _parse_args()

    features_df = pd.read_csv(args.features)
    feature_cols = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    label_order = json.loads(LABEL_ORDER_PATH.read_text(encoding="utf-8"))

    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    regressor = joblib.load(REGRESSOR_MODEL_PATH)
    peak_model = joblib.load(PEAK_MODEL_PATH) if PEAK_MODEL_PATH.exists() else None

    latest = _latest_rows(features_df, allow_incomplete_current=args.allow_incomplete_current_season)
    x = latest[feature_cols]

    class_probs = classifier.predict_proba(x)
    bpm_delta_pred = regressor.predict(x)

    out = latest[["player_id", "player_name", "season", "age", "bpm"]].copy()
    out = out.rename(columns={"bpm": "current_bpm"})

    for idx, label in enumerate(label_order):
        out[f"{label}_probability"] = class_probs[:, idx]

    out["predicted_bpm_delta"] = bpm_delta_pred

    if peak_model is not None:
        peak_prob = peak_model.predict_proba(x)[:, 1]
        out["peak_probability"] = peak_prob
    else:
        out["peak_probability"] = np.nan

    prob_cols = [f"{label}_probability" for label in label_order]
    out["trajectory"] = out[prob_cols].idxmax(axis=1).str.replace("_probability", "", regex=False)
    out["prediction_factors"] = latest.apply(lambda r: json.dumps(_format_factors(r)), axis=1)
    out["model_version"] = MODEL_VERSION
    out["updated_at"] = datetime.now(timezone.utc).isoformat()

    _validate_probabilities(out, label_order)
    if out["predicted_bpm_delta"].isna().any():
        raise ValueError("NaN values found in predicted_bpm_delta")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("=== Prediction Summary ===")
    print(f"Players predicted: {len(out):,}")
    print(f"Trajectory classes present: {sorted(out['trajectory'].unique())}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
