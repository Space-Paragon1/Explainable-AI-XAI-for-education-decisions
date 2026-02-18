from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def evaluate_binary(model, X_test, y_test) -> dict:
    """Full suite of binary classification metrics for a trained model."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "mastery_rate_test": float(np.mean(y_test)),
    }
    return metrics


def cross_validate_model(
    model_factory: Callable,
    X,
    y,
    n_splits: int = 5,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable that returns a fresh unfitted sklearn Pipeline.
    X : array-like
    y : array-like
    n_splits : int

    Returns
    -------
    dict with mean ± std for accuracy, f1, roc_auc across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: dict[str, list[float]] = {"accuracy": [], "f1": [], "roc_auc": []}

    X_arr = np.array(X)
    y_arr = np.array(y)

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics["accuracy"].append(float(accuracy_score(y_val, pred)))
        fold_metrics["f1"].append(float(f1_score(y_val, pred, zero_division=0)))
        fold_metrics["roc_auc"].append(float(roc_auc_score(y_val, proba)))

    return {
        metric: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "folds": vals,
        }
        for metric, vals in fold_metrics.items()
    }


def get_calibration_data(model, X_test, y_test, n_bins: int = 10) -> dict:
    """
    Compute calibration curve data for a reliability diagram.

    Returns fraction_of_positives and mean_predicted_value arrays
    (both as lists for JSON serialization).
    """
    proba = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, proba, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }


def save_metrics(metrics: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
