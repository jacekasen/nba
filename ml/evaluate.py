"""Evaluation helpers for classification and regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


@dataclass
class ClassificationOutputs:
    """Container for classification prediction outputs."""

    y_pred: np.ndarray
    y_proba: np.ndarray


def multiclass_brier_score(y_true: pd.Series, y_proba: np.ndarray, labels: list[str]) -> float:
    """Compute multiclass Brier-style score (lower is better)."""
    y_index = pd.Categorical(y_true, categories=labels)
    y_one_hot = np.zeros((len(y_true), len(labels)), dtype=float)
    valid_mask = y_index.codes >= 0
    y_one_hot[np.arange(len(y_true))[valid_mask], y_index.codes[valid_mask]] = 1.0
    return float(np.mean(np.sum((y_one_hot - y_proba) ** 2, axis=1)))


def evaluate_classifier(
    y_true: pd.Series,
    outputs: ClassificationOutputs,
    labels: list[str],
) -> dict[str, Any]:
    """Evaluate multiclass classifier on a fixed label order."""
    y_pred = outputs.y_pred
    y_proba = outputs.y_proba

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "multiclass_brier": multiclass_brier_score(y_true, y_proba, labels=labels),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_regressor(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate regression model with key numeric metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def grouped_regression_error(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> list[dict[str, Any]]:
    """Compute MAE/RMSE by group for diagnostics."""
    rows: list[dict[str, Any]] = []
    for group_name, group in df.groupby(group_col, observed=False):
        if group.empty:
            continue
        y_true = group[y_true_col]
        y_pred = group[y_pred_col]
        rows.append(
            {
                "group": str(group_name),
                "n": int(len(group)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            }
        )
    return rows
