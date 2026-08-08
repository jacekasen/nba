"""Train and evaluate trajectory, BPM-delta, and near-peak models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _fit_classifier_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], str, Pipeline, dict[str, Any]]:
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
        "hist_gradient_boosting": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
            ]
        ),
    }

    metrics: dict[str, Any] = {"classifier": {"validation": {}, "test": {}}}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model

        val_outputs = ClassificationOutputs(y_pred=model.predict(x_val), y_proba=model.predict_proba(x_val))
        test_outputs = ClassificationOutputs(y_pred=model.predict(x_test), y_proba=model.predict_proba(x_test))

        metrics["classifier"]["validation"][name] = evaluate_classifier(y_val, val_outputs, TRAJECTORY_LABELS)
        metrics["classifier"]["test"][name] = evaluate_classifier(y_test, test_outputs, TRAJECTORY_LABELS)

    best_name = max(
        metrics["classifier"]["validation"],
        key=lambda k: metrics["classifier"]["validation"][k]["macro_f1"],
    )
    best_model = fitted_models[best_name]

    calibration_info: dict[str, Any] = {"used": False, "reason": "not_applied"}
    if best_name == "hist_gradient_boosting":
        try:
            try:
                from sklearn.frozen import FrozenEstimator

                calibrator = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
            except Exception:
                calibrator = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
            calibrator.fit(x_val, y_val)
            calibrated_outputs = ClassificationOutputs(
                y_pred=calibrator.predict(x_test),
                y_proba=calibrator.predict_proba(x_test),
            )
            metrics["classifier"]["test"]["hist_gradient_boosting_calibrated"] = evaluate_classifier(
                y_test,
                calibrated_outputs,
                TRAJECTORY_LABELS,
            )
            best_name = "hist_gradient_boosting_calibrated"
            best_model = Pipeline([("calibrated", calibrator)])
            calibration_info = {"used": True, "method": "sigmoid", "calibration_split": "validation"}
        except Exception as exc:  # pragma: no cover - depends on sklearn version
            calibration_info = {"used": False, "reason": f"calibration_not_supported: {exc}"}

    metrics["classifier"]["selected_model"] = best_name
    metrics["classifier"]["calibration"] = calibration_info

    return metrics, best_name, best_model, {
        "x_test": x_test,
        "y_test": y_test,
        "x_val": x_val,
        "y_val": y_val,
    }


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
        "hist_gradient_boosting": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
            ]
        ),
    }

    metrics: dict[str, Any] = {"regression": {"validation": {}, "test": {}}}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        metrics["regression"]["validation"][name] = evaluate_regressor(y_val, val_pred)
        metrics["regression"]["test"][name] = evaluate_regressor(y_test, test_pred)

    best_name = min(
        metrics["regression"]["validation"],
        key=lambda k: metrics["regression"]["validation"][k]["mae"],
    )
    best_model = fitted_models[best_name]

    test_eval = test_df[["season_start", "age", "bpm", "target_bpm_change"]].copy()
    test_eval["predicted_bpm_change"] = best_model.predict(x_test)

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

    return metrics, best_name, best_model, test_eval


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
        "hist_gradient_boosting": Pipeline(
            [
                ("pre", _build_preprocessor(feature_cols, with_scaling=False)),
                ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
            ]
        ),
    }

    result = {"validation": {}, "test": {}, "split": split_meta}
    fitted: dict[str, Pipeline] = {}
    labels = [0, 1]

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
    target_df = df.dropna(subset=["target_trajectory", "target_bpm_change"]).copy()

    train_mask, val_mask, test_mask, split_meta = _split_masks(target_df, split_cfg)
    train_df = target_df[train_mask].copy()
    val_df = target_df[val_mask].copy()
    test_df = target_df[test_mask].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Chronological split produced empty train/validation/test set.")

    clf_metrics, clf_selected_name, classifier_model, clf_eval_data = _fit_classifier_candidates(
        train_df, val_df, test_df, feature_cols
    )
    reg_metrics, reg_selected_name, regressor_model, reg_test_eval = _fit_regressor_candidates(
        train_df, val_df, test_df, feature_cols
    )
    peak_metrics, peak_model = _fit_peak_model(
        features_df=df,
        split=split_cfg,
        feature_cols=feature_cols,
        completion_gap_years=args.completion_gap_years,
        smoothing_window=args.peak_smoothing_window,
        near_peak_tolerance=args.peak_tolerance,
    )

    if clf_selected_name == "hist_gradient_boosting_calibrated":
        clf_test_probs = classifier_model.predict_proba(clf_eval_data["x_test"])
        clf_test_preds = classifier_model.predict(clf_eval_data["x_test"])
    else:
        clf_test_probs = classifier_model.predict_proba(clf_eval_data["x_test"])
        clf_test_preds = classifier_model.predict(clf_eval_data["x_test"])

    cm = pd.crosstab(
        clf_eval_data["y_test"],
        pd.Series(clf_test_preds, index=clf_eval_data["y_test"].index, name="predicted"),
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
                clf_eval_data["x_val"],
                clf_eval_data["y_val"],
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
                val_df[feature_cols],
                val_df["target_bpm_change"],
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
        "n_rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "season_range": {
            "train": [int(train_df["season_start"].min()), int(train_df["season_start"].max())],
            "validation": [int(val_df["season_start"].min()), int(val_df["season_start"].max())],
            "test": [int(test_df["season_start"].min()), int(test_df["season_start"].max())],
        },
        "selected_models": {
            "classifier": clf_selected_name,
            "regressor": reg_selected_name,
            "peak": peak_metrics["peak_model"].get("selected_model"),
        },
    }

    metrics = {
        **clf_metrics,
        **reg_metrics,
        **peak_metrics,
        "baseline_comparison": {
            "classifier_selected_beats_dummy_macro_f1": (
                clf_metrics["classifier"]["test"][clf_selected_name]["macro_f1"]
                > clf_metrics["classifier"]["test"]["dummy_prior"]["macro_f1"]
            ),
            "regressor_selected_beats_zero_mae": (
                reg_metrics["regression"]["test"][reg_selected_name]["mae"]
                < reg_metrics["regression"]["test"]["dummy_zero"]["mae"]
            ),
        },
    }

    joblib.dump(classifier_model, CLASSIFIER_MODEL_PATH)
    joblib.dump(regressor_model, REGRESSOR_MODEL_PATH)
    if peak_model is not None:
        joblib.dump(peak_model, PEAK_MODEL_PATH)

    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    LABEL_ORDER_PATH.write_text(json.dumps(TRAJECTORY_LABELS, indent=2), encoding="utf-8")
    TRAINING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== Training Complete ===")
    print(f"Classifier selected: {clf_selected_name}")
    print(f"Regressor selected: {reg_selected_name}")
    print("Classifier validation macro F1 by model:")
    for name, row in clf_metrics["classifier"]["validation"].items():
        print(f"  {name}: {row['macro_f1']:.4f}")
    print("Regressor validation MAE by model:")
    for name, row in reg_metrics["regression"]["validation"].items():
        print(f"  {name}: {row['mae']:.4f}")
    print("Selected model test metrics vs baselines:")
    print(
        "  classifier macro_f1 "
        f"{clf_metrics['classifier']['test'][clf_selected_name]['macro_f1']:.4f} "
        f"(dummy {clf_metrics['classifier']['test']['dummy_prior']['macro_f1']:.4f})"
    )
    print(
        "  regressor mae "
        f"{reg_metrics['regression']['test'][reg_selected_name]['mae']:.4f} "
        f"(dummy_zero {reg_metrics['regression']['test']['dummy_zero']['mae']:.4f})"
    )
    print(f"Saved artifacts in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
