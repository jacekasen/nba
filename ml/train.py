"""Train and evaluate trajectory, BPM-delta, continuation, and near-peak models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    CLASSIFIER_MODEL_PATH,
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    CONTINUATION_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_IMPORTANCE_PATH,
    LABEL_ORDER_PATH,
    METRICS_PATH,
    MODEL_FEATURES_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    PEAK_MODEL_PATH,
    PeakModelConfig,
    RANDOM_SEED,
    REGRESSOR_MODEL_PATH,
    SplitConfig,
    TRAINING_METADATA_PATH,
    TRAJECTORY_FEATURE_COLUMNS,
    TRAJECTORY_LABELS,
)
from ml.evaluate import ClassificationOutputs, evaluate_classifier, evaluate_regressor, grouped_regression_error

# Small validation-selected grid; defaults overfit the noisy BPM targets.
HGB_PARAM_GRID: list[dict[str, Any]] = [
    {
        "learning_rate": lr,
        "max_leaf_nodes": leaves,
        "min_samples_leaf": min_leaf,
        "l2_regularization": l2,
    }
    for lr in (0.05, 0.1)
    for leaves in (15, 31)
    for min_leaf in (20, 50)
    for l2 in (0.0, 1.0)
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=MODEL_FEATURES_PATH, help="Feature table CSV")
    parser.add_argument("--train-end", type=int, default=SplitConfig().train_end)
    parser.add_argument("--val-start", type=int, default=SplitConfig().val_start)
    parser.add_argument("--val-end", type=int, default=SplitConfig().val_end)
    parser.add_argument("--test-start", type=int, default=SplitConfig().test_start)
    parser.add_argument("--completion-gap-years", type=int, default=PeakModelConfig().completion_gap_years)
    parser.add_argument("--peak-tolerance", type=float, default=PeakModelConfig().near_peak_tolerance)
    parser.add_argument("--peak-smoothing-window", type=int, default=PeakModelConfig().smoothing_window)
    return parser.parse_args()


def _build_numeric_pipeline(with_scaling: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaling:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _build_preprocessor(feature_cols: list[str], with_scaling: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", _build_numeric_pipeline(with_scaling=with_scaling), feature_cols)],
        remainder="drop",
    )


def _split_masks(df: pd.DataFrame, split: SplitConfig) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    train_mask = df["season_start"] <= split.train_end
    val_mask = (df["season_start"] >= split.val_start) & (df["season_start"] <= split.val_end)
    test_mask = df["season_start"] >= split.test_start

    meta = {
        "method": "configured_boundaries",
        "train_end": split.train_end,
        "val_start": split.val_start,
        "val_end": split.val_end,
        "test_start": split.test_start,
    }

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        unique_years = sorted(df["season_start"].dropna().unique())
        if len(unique_years) < 6:
            raise ValueError("Not enough seasonal coverage for chronological split.")
        train_idx = int(len(unique_years) * 0.7)
        val_idx = int(len(unique_years) * 0.85)
        train_end = unique_years[max(train_idx - 1, 0)]
        val_end = unique_years[max(val_idx - 1, train_idx)]
        val_start = unique_years[train_idx]
        test_start = unique_years[min(val_idx, len(unique_years) - 1)]

        train_mask = df["season_start"] <= train_end
        val_mask = (df["season_start"] >= val_start) & (df["season_start"] <= val_end)
        test_mask = df["season_start"] >= test_start

        meta = {
            "method": "fallback_quantile_boundaries",
            "train_end": int(train_end),
            "val_start": int(val_start),
            "val_end": int(val_end),
            "test_start": int(test_start),
        }

    return train_mask, val_mask, test_mask, meta


def _trajectory_feature_columns() -> list[str]:
    return list(TRAJECTORY_FEATURE_COLUMNS)


def _hgb_classifier_pipeline(feature_cols: list[str], params: dict[str, Any] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, **(params or {}))),
        ]
    )


def _hgb_regressor_pipeline(feature_cols: list[str], params: dict[str, Any] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_SEED, **(params or {}))),
        ]
    )


def _calibrated_refit(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Refit a classifier on the given data with cross-fitted sigmoid calibration."""
    calibrator = CalibratedClassifierCV(clone(pipeline), method="sigmoid", cv=5)
    calibrator.fit(x, y)
    return Pipeline([("calibrated", calibrator)])


def _fit_classifier_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], str, Pipeline]:
    x_train = train_df[feature_cols]
    y_train = train_df["target_trajectory"]
    x_val = val_df[feature_cols]
    y_val = val_df["target_trajectory"]
    x_test = test_df[feature_cols]
    y_test = test_df["target_trajectory"]

    candidates: dict[str, Pipeline] = {
        "dummy_prior": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "logistic_multinomial": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": _hgb_classifier_pipeline(feature_cols),
    }

    metrics: dict[str, Any] = {"classifier": {"validation": {}, "test": {}}}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model

        model_classes = [str(c) for c in model.classes_]
        val_outputs = ClassificationOutputs(
            y_pred=model.predict(x_val), y_proba=model.predict_proba(x_val), classes=model_classes
        )
        test_outputs = ClassificationOutputs(
            y_pred=model.predict(x_test), y_proba=model.predict_proba(x_test), classes=model_classes
        )

        metrics["classifier"]["validation"][name] = evaluate_classifier(y_val, val_outputs, TRAJECTORY_LABELS)
        metrics["classifier"]["test"][name] = evaluate_classifier(y_test, test_outputs, TRAJECTORY_LABELS)

    best_tuned_score = -np.inf
    best_tuned_params: dict[str, Any] | None = None
    best_tuned_model: Pipeline | None = None
    for params in HGB_PARAM_GRID:
        model = _hgb_classifier_pipeline(feature_cols, params)
        model.fit(x_train, y_train)
        score = evaluate_classifier(
            y_val,
            ClassificationOutputs(
                y_pred=model.predict(x_val),
                y_proba=model.predict_proba(x_val),
                classes=[str(c) for c in model.classes_],
            ),
            TRAJECTORY_LABELS,
        )["macro_f1"]
        if score > best_tuned_score:
            best_tuned_score = score
            best_tuned_params = params
            best_tuned_model = model

    if best_tuned_model is not None:
        name = "hist_gradient_boosting_tuned"
        fitted_models[name] = best_tuned_model
        candidates[name] = best_tuned_model
        tuned_classes = [str(c) for c in best_tuned_model.classes_]
        val_outputs = ClassificationOutputs(
            y_pred=best_tuned_model.predict(x_val),
            y_proba=best_tuned_model.predict_proba(x_val),
            classes=tuned_classes,
        )
        test_outputs = ClassificationOutputs(
            y_pred=best_tuned_model.predict(x_test),
            y_proba=best_tuned_model.predict_proba(x_test),
            classes=tuned_classes,
        )
        metrics["classifier"]["validation"][name] = evaluate_classifier(y_val, val_outputs, TRAJECTORY_LABELS)
        metrics["classifier"]["test"][name] = evaluate_classifier(y_test, test_outputs, TRAJECTORY_LABELS)
        metrics["classifier"]["tuned_params"] = best_tuned_params

    best_name = max(
        metrics["classifier"]["validation"],
        key=lambda k: metrics["classifier"]["validation"][k]["macro_f1"],
    )

    # Final model: refit the selected architecture on train+validation so the
    # shipped model sees the most recent completed seasons. Tree models get
    # cross-fitted sigmoid calibration for honest probabilities.
    x_final = pd.concat([x_train, x_val])
    y_final = pd.concat([y_train, y_val])
    if best_name.startswith("hist_gradient_boosting"):
        final_model = _calibrated_refit(candidates[best_name], x_final, y_final)
        calibration_info = {"used": True, "method": "sigmoid", "cv": 5, "fit_on": "train+validation"}
    else:
        final_model = clone(candidates[best_name])
        final_model.fit(x_final, y_final)
        calibration_info = {"used": False, "reason": "linear_model_refit_without_calibration"}

    final_outputs = ClassificationOutputs(
        y_pred=final_model.predict(x_test),
        y_proba=final_model.predict_proba(x_test),
        classes=[str(c) for c in final_model.classes_],
    )
    metrics["classifier"]["test"]["final_refit"] = evaluate_classifier(y_test, final_outputs, TRAJECTORY_LABELS)
    metrics["classifier"]["selected_model"] = best_name
    metrics["classifier"]["calibration"] = calibration_info

    return metrics, best_name, final_model


def _fit_regressor_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], str, Pipeline, pd.DataFrame]:
    x_train = train_df[feature_cols]
    y_train = train_df["target_bpm_change"]
    x_val = val_df[feature_cols]
    y_val = val_df["target_bpm_change"]
    x_test = test_df[feature_cols]
    y_test = test_df["target_bpm_change"]

    candidates: dict[str, Pipeline] = {
        "dummy_zero": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", DummyRegressor(strategy="constant", constant=0.0)),
            ]
        ),
        "dummy_mean": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "ridge": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=True)),
                ("model", Ridge(random_state=RANDOM_SEED)),
            ]
        ),
        "hist_gradient_boosting": _hgb_regressor_pipeline(feature_cols),
    }

    metrics: dict[str, Any] = {"regression": {"validation": {}, "test": {}}}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        metrics["regression"]["validation"][name] = evaluate_regressor(y_val, model.predict(x_val))
        metrics["regression"]["test"][name] = evaluate_regressor(y_test, model.predict(x_test))

    best_tuned_mae = np.inf
    best_tuned_params: dict[str, Any] | None = None
    best_tuned_model: Pipeline | None = None
    for params in HGB_PARAM_GRID:
        model = _hgb_regressor_pipeline(feature_cols, params)
        model.fit(x_train, y_train)
        mae = evaluate_regressor(y_val, model.predict(x_val))["mae"]
        if mae < best_tuned_mae:
            best_tuned_mae = mae
            best_tuned_params = params
            best_tuned_model = model

    if best_tuned_model is not None:
        name = "hist_gradient_boosting_tuned"
        fitted_models[name] = best_tuned_model
        candidates[name] = best_tuned_model
        metrics["regression"]["validation"][name] = evaluate_regressor(y_val, best_tuned_model.predict(x_val))
        metrics["regression"]["test"][name] = evaluate_regressor(y_test, best_tuned_model.predict(x_test))
        metrics["regression"]["tuned_params"] = best_tuned_params

    best_name = min(
        metrics["regression"]["validation"],
        key=lambda k: metrics["regression"]["validation"][k]["mae"],
    )

    final_model = clone(candidates[best_name])
    final_model.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    metrics["regression"]["test"]["final_refit"] = evaluate_regressor(y_test, final_model.predict(x_test))

    test_eval = test_df[["season_start", "age", "bpm", "target_bpm_change"]].copy()
    test_eval["predicted_bpm_change"] = final_model.predict(x_test)

    age_bins = pd.cut(test_eval["age"], bins=[0, 22, 26, 30, 35, 60], labels=["<=22", "23-26", "27-30", "31-35", "36+"])
    bpm_bins = pd.cut(
        test_eval["bpm"],
        bins=[-20, -2, 0, 2, 5, 20],
        labels=["< -2", "-2 to 0", "0 to 2", "2 to 5", ">= 5"],
    )
    test_eval["age_group"] = age_bins
    test_eval["bpm_tier"] = bpm_bins

    metrics["regression"]["selected_model"] = best_name
    metrics["regression"]["test_error_by_age_group"] = grouped_regression_error(
        test_eval.dropna(subset=["age_group"]),
        y_true_col="target_bpm_change",
        y_pred_col="predicted_bpm_change",
        group_col="age_group",
    )
    metrics["regression"]["test_error_by_bpm_tier"] = grouped_regression_error(
        test_eval.dropna(subset=["bpm_tier"]),
        y_true_col="target_bpm_change",
        y_pred_col="predicted_bpm_change",
        group_col="bpm_tier",
    )

    return metrics, best_name, final_model, test_eval


def _fit_continuation_model(
    features_df: pd.DataFrame,
    split: SplitConfig,
    feature_cols: list[str],
) -> tuple[dict[str, Any], Pipeline | None]:
    """Fit P(player logs a qualified consecutive season next year)."""
    cont_df = features_df.dropna(subset=["target_played_next"]).copy()
    cont_df["target_played_next"] = cont_df["target_played_next"].astype(int)

    metrics: dict[str, Any] = {"continuation_model": {}}
    if cont_df.empty or cont_df["target_played_next"].nunique() < 2:
        metrics["continuation_model"]["status"] = "not_trained"
        metrics["continuation_model"]["reason"] = "target_not_binary"
        return metrics, None

    train_mask, val_mask, test_mask, split_meta = _split_masks(cont_df, split)
    train_df = cont_df[train_mask]
    val_df = cont_df[val_mask]
    test_df = cont_df[test_mask]

    if train_df.empty or val_df.empty or test_df.empty:
        metrics["continuation_model"]["status"] = "not_trained"
        metrics["continuation_model"]["reason"] = "insufficient_split_samples"
        return metrics, None

    x_train = train_df[feature_cols]
    y_train = train_df["target_played_next"]
    x_val = val_df[feature_cols]
    y_val = val_df["target_played_next"]
    x_test = test_df[feature_cols]
    y_test = test_df["target_played_next"]

    candidates: dict[str, Pipeline] = {
        "dummy_prior": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "logistic": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=True)),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
            ]
        ),
        "hist_gradient_boosting": _hgb_classifier_pipeline(feature_cols),
    }

    result: dict[str, Any] = {"validation": {}, "test": {}, "split": split_meta}
    fitted: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted[name] = model
        for split_name, (x_s, y_s) in {"validation": (x_val, y_val), "test": (x_test, y_test)}.items():
            proba = model.predict_proba(x_s)[:, 1]
            pred = model.predict(x_s)
            result[split_name][name] = {
                "accuracy": float((pred == y_s).mean()),
                "brier": float(np.mean((y_s.to_numpy() - proba) ** 2)),
            }

    selected = min(result["validation"], key=lambda k: result["validation"][k]["brier"])

    x_final = pd.concat([x_train, x_val])
    y_final = pd.concat([y_train, y_val])
    if selected == "hist_gradient_boosting":
        final_model = _calibrated_refit(candidates[selected], x_final, y_final)
    else:
        final_model = clone(candidates[selected])
        final_model.fit(x_final, y_final)

    final_proba = final_model.predict_proba(x_test)[:, 1]
    result["test"]["final_refit"] = {
        "accuracy": float((final_model.predict(x_test) == y_test).mean()),
        "brier": float(np.mean((y_test.to_numpy() - final_proba) ** 2)),
    }
    result["selected_model"] = selected
    result["positive_rate_train"] = float(y_train.mean())
    metrics["continuation_model"].update(result)

    return metrics, final_model


def _build_peak_training_frame(
    features_df: pd.DataFrame,
    completion_gap_years: int,
    smoothing_window: int,
    near_peak_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = features_df.copy()
    global_max_end = int(df["season_end"].max())
    completion_cutoff = global_max_end - completion_gap_years

    latest_by_player = df.groupby("player_url")["season_end"].max()
    completed_players = set(latest_by_player[latest_by_player <= completion_cutoff].index)

    candidate = df[df["player_url"].isin(completed_players)].copy()

    def label_near_peak(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("season_start").copy()
        g["bpm_smooth_peak_model"] = g["bpm"].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        eventual_peak = g["bpm_smooth_peak_model"].max()
        g["eventual_bpm_smooth_peak"] = eventual_peak
        g["near_peak"] = (g["bpm_smooth_peak_model"] >= (eventual_peak - near_peak_tolerance)).astype(int)
        return g

    if candidate.empty:
        return candidate, {
            "global_max_season_end": global_max_end,
            "completion_cutoff": completion_cutoff,
            "completed_players": 0,
            "valid": False,
            "reason": "no_completed_players",
        }

    labeled = (
        candidate.groupby("player_url", group_keys=False)
        .apply(label_near_peak, include_groups=False)
        .reset_index(drop=True)
    )

    meta = {
        "global_max_season_end": global_max_end,
        "completion_cutoff": completion_cutoff,
        "completed_players": int(len(completed_players)),
        "rows": int(len(labeled)),
        "positive_rate": float(labeled["near_peak"].mean()),
        "valid": labeled["near_peak"].nunique() == 2,
    }
    if not meta["valid"]:
        meta["reason"] = "target_not_binary"

    return labeled, meta


def _fit_peak_model(
    features_df: pd.DataFrame,
    split: SplitConfig,
    feature_cols: list[str],
    completion_gap_years: int,
    smoothing_window: int,
    near_peak_tolerance: float,
) -> tuple[dict[str, Any], Pipeline | None]:
    peak_df, peak_meta = _build_peak_training_frame(
        features_df,
        completion_gap_years=completion_gap_years,
        smoothing_window=smoothing_window,
        near_peak_tolerance=near_peak_tolerance,
    )

    metrics: dict[str, Any] = {"peak_model": {"meta": peak_meta}}
    if not peak_meta.get("valid", False):
        metrics["peak_model"]["status"] = "not_trained"
        return metrics, None

    train_mask, val_mask, test_mask, split_meta = _split_masks(peak_df, split)
    train_df = peak_df[train_mask]
    val_df = peak_df[val_mask]
    test_df = peak_df[test_mask]

    if train_df.empty or val_df.empty or test_df.empty:
        metrics["peak_model"]["status"] = "not_trained"
        metrics["peak_model"]["reason"] = "insufficient_split_samples"
        metrics["peak_model"]["split"] = split_meta
        return metrics, None

    x_train = train_df[feature_cols]
    y_train = train_df["near_peak"]
    x_val = val_df[feature_cols]
    y_val = val_df["near_peak"]
    x_test = test_df[feature_cols]
    y_test = test_df["near_peak"]

    candidates: dict[str, Pipeline] = {
        "dummy_prior": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "logistic": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=True)),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
            ]
        ),
        "hist_gradient_boosting": _hgb_classifier_pipeline(feature_cols),
    }

    result = {"validation": {}, "test": {}, "split": split_meta}
    fitted: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted[name] = model
        val_proba = model.predict_proba(x_val)
        test_proba = model.predict_proba(x_test)

        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        val_loss = float(np.mean((y_val.to_numpy() - val_proba[:, 1]) ** 2))
        test_loss = float(np.mean((y_test.to_numpy() - test_proba[:, 1]) ** 2))

        result["validation"][name] = {
            "accuracy": float((val_pred == y_val).mean()),
            "f1": float((2 * ((val_pred & y_val).sum())) / max((val_pred.sum() + y_val.sum()), 1)),
            "brier": val_loss,
        }
        result["test"][name] = {
            "accuracy": float((test_pred == y_test).mean()),
            "f1": float((2 * ((test_pred & y_test).sum())) / max((test_pred.sum() + y_test.sum()), 1)),
            "brier": test_loss,
        }

    selected = min(result["validation"], key=lambda k: result["validation"][k]["brier"])
    result["selected_model"] = selected
    metrics["peak_model"].update(result)

    return metrics, fitted[selected]


def _save_confusion_matrix_png(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Trajectory Confusion Matrix (Test)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _permutation_importance_table(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    scoring: str,
    model_name: str,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=10,
        random_state=RANDOM_SEED,
        scoring=scoring,
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def main() -> None:
    """Train all models and persist metrics/artifacts."""
    args = _parse_args()
    split_cfg = SplitConfig(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
    )

    df = pd.read_csv(args.features)
    feature_cols = _trajectory_feature_columns()

    # The classifier target covers exits (no qualified next season -> regressing);
    # the regression target only exists when a qualified next season was played.
    clf_df = df.dropna(subset=["target_trajectory"]).copy()
    reg_df = df.dropna(subset=["target_bpm_change"]).copy()

    clf_train_mask, clf_val_mask, clf_test_mask, split_meta = _split_masks(clf_df, split_cfg)
    clf_train_df = clf_df[clf_train_mask].copy()
    clf_val_df = clf_df[clf_val_mask].copy()
    clf_test_df = clf_df[clf_test_mask].copy()

    reg_train_mask, reg_val_mask, reg_test_mask, _ = _split_masks(reg_df, split_cfg)
    reg_train_df = reg_df[reg_train_mask].copy()
    reg_val_df = reg_df[reg_val_mask].copy()
    reg_test_df = reg_df[reg_test_mask].copy()

    for frame_set in ((clf_train_df, clf_val_df, clf_test_df), (reg_train_df, reg_val_df, reg_test_df)):
        if any(frame.empty for frame in frame_set):
            raise ValueError("Chronological split produced empty train/validation/test set.")

    clf_metrics, clf_selected_name, classifier_model = _fit_classifier_candidates(
        clf_train_df, clf_val_df, clf_test_df, feature_cols
    )
    reg_metrics, reg_selected_name, regressor_model, reg_test_eval = _fit_regressor_candidates(
        reg_train_df, reg_val_df, reg_test_df, feature_cols
    )
    cont_metrics, continuation_model = _fit_continuation_model(
        features_df=df,
        split=split_cfg,
        feature_cols=feature_cols,
    )
    peak_metrics, peak_model = _fit_peak_model(
        features_df=df,
        split=split_cfg,
        feature_cols=feature_cols,
        completion_gap_years=args.completion_gap_years,
        smoothing_window=args.peak_smoothing_window,
        near_peak_tolerance=args.peak_tolerance,
    )

    clf_test_preds = classifier_model.predict(clf_test_df[feature_cols])

    cm = pd.crosstab(
        clf_test_df["target_trajectory"],
        pd.Series(clf_test_preds, index=clf_test_df.index, name="predicted"),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    ).reindex(index=TRAJECTORY_LABELS, columns=TRAJECTORY_LABELS, fill_value=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cm.to_csv(CONFUSION_MATRIX_CSV_PATH)
    _save_confusion_matrix_png(cm.to_numpy(), TRAJECTORY_LABELS, CONFUSION_MATRIX_PNG_PATH)

    importance_frames = []
    try:
        importance_frames.append(
            _permutation_importance_table(
                classifier_model,
                clf_test_df[feature_cols],
                clf_test_df["target_trajectory"],
                feature_cols,
                scoring="f1_macro",
                model_name=f"classifier:{clf_selected_name}",
            )
        )
    except Exception:
        pass

    try:
        importance_frames.append(
            _permutation_importance_table(
                regressor_model,
                reg_test_df[feature_cols],
                reg_test_df["target_bpm_change"],
                feature_cols,
                scoring="neg_mean_absolute_error",
                model_name=f"regressor:{reg_selected_name}",
            )
        )
    except Exception:
        pass

    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "split": split_meta,
        "final_models_fit_on": "train+validation",
        "n_rows": {
            "classifier": {
                "train": int(len(clf_train_df)),
                "validation": int(len(clf_val_df)),
                "test": int(len(clf_test_df)),
            },
            "regressor": {
                "train": int(len(reg_train_df)),
                "validation": int(len(reg_val_df)),
                "test": int(len(reg_test_df)),
            },
        },
        "season_range": {
            "train": [int(clf_train_df["season_start"].min()), int(clf_train_df["season_start"].max())],
            "validation": [int(clf_val_df["season_start"].min()), int(clf_val_df["season_start"].max())],
            "test": [int(clf_test_df["season_start"].min()), int(clf_test_df["season_start"].max())],
        },
        "selected_models": {
            "classifier": clf_selected_name,
            "regressor": reg_selected_name,
            "continuation": cont_metrics["continuation_model"].get("selected_model"),
            "peak": peak_metrics["peak_model"].get("selected_model"),
        },
    }

    metrics = {
        **clf_metrics,
        **reg_metrics,
        **cont_metrics,
        **peak_metrics,
        "baseline_comparison": {
            "classifier_final_beats_dummy_macro_f1": (
                clf_metrics["classifier"]["test"]["final_refit"]["macro_f1"]
                > clf_metrics["classifier"]["test"]["dummy_prior"]["macro_f1"]
            ),
            "regressor_final_beats_zero_mae": (
                reg_metrics["regression"]["test"]["final_refit"]["mae"]
                < reg_metrics["regression"]["test"]["dummy_zero"]["mae"]
            ),
        },
    }

    joblib.dump(classifier_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(regressor_model, REGRESSOR_MODEL_PATH)
    if continuation_model is not None:
        joblib.dump(continuation_model, CONTINUATION_MODEL_PATH)
    if peak_model is not None:
        joblib.dump(peak_model, PEAK_MODEL_PATH)

    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    LABEL_ORDER_PATH.write_text(json.dumps(TRAJECTORY_LABELS, indent=2), encoding="utf-8")
    TRAINING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== Training Complete ===")
    print(f"Classifier selected: {clf_selected_name}")
    print(f"Regressor selected: {reg_selected_name}")
    print(f"Continuation selected: {cont_metrics['continuation_model'].get('selected_model')}")
    print("Classifier validation macro F1 by model:")
    for name, row in clf_metrics["classifier"]["validation"].items():
        print(f"  {name}: {row['macro_f1']:.4f}")
    print("Regressor validation MAE by model:")
    for name, row in reg_metrics["regression"]["validation"].items():
        print(f"  {name}: {row['mae']:.4f}")
    print("Final refit test metrics vs baselines:")
    print(
        "  classifier macro_f1 "
        f"{clf_metrics['classifier']['test']['final_refit']['macro_f1']:.4f} "
        f"(dummy {clf_metrics['classifier']['test']['dummy_prior']['macro_f1']:.4f})"
    )
    print(
        "  regressor mae "
        f"{reg_metrics['regression']['test']['final_refit']['mae']:.4f} "
        f"(dummy_zero {reg_metrics['regression']['test']['dummy_zero']['mae']:.4f})"
    )
    if "validation" in cont_metrics["continuation_model"]:
        print(
            "  continuation brier "
            f"{cont_metrics['continuation_model']['test']['final_refit']['brier']:.4f} "
            f"(dummy {cont_metrics['continuation_model']['test']['dummy_prior']['brier']:.4f})"
        )
    print(f"Saved artifacts in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
