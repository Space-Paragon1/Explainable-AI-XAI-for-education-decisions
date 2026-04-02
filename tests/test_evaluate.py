"""
test_evaluate.py — Tests for evaluation metrics and statistical comparison.
"""
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.evaluate import (
    evaluate_binary,
    cross_validate_model,
    compare_models_mcnemar,
    get_calibration_data,
)
from src.config import FEATURE_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dummy_pipeline(strategy="most_frequent"):
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), slice(0, None))],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", DummyClassifier(strategy=strategy, random_state=0)),
    ])


# ── evaluate_binary ───────────────────────────────────────────────────────────

def test_evaluate_binary_keys(X_y, trained_logreg):
    X, y = X_y
    metrics = evaluate_binary(trained_logreg, X, y)
    expected_keys = {"accuracy", "precision", "recall", "f1",
                     "roc_auc", "brier_score", "mcc", "mastery_rate_test"}
    assert expected_keys == set(metrics.keys())


def test_evaluate_binary_ranges(X_y, trained_logreg):
    X, y = X_y
    m = evaluate_binary(trained_logreg, X, y)
    assert 0.0 <= m["accuracy"]  <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"]    <= 1.0
    assert 0.0 <= m["f1"]        <= 1.0
    assert 0.0 <= m["roc_auc"]   <= 1.0
    assert 0.0 <= m["brier_score"] <= 1.0
    assert -1.0 <= m["mcc"]      <= 1.0


def test_evaluate_binary_perfect_model(X_y):
    """A perfect predictor should score 1.0 on accuracy, f1, auc."""
    from unittest.mock import MagicMock
    import pandas as pd

    X, y = X_y
    model = MagicMock()
    y_arr = np.array(y)
    # Perfect proba: 1.0 for positives, 0.0 for negatives
    proba = np.where(y_arr == 1, 1.0, 0.0).reshape(-1, 1)
    model.predict_proba.return_value = np.hstack([1 - proba, proba])

    m = evaluate_binary(model, X, y)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"]       == pytest.approx(1.0)
    assert m["roc_auc"]  == pytest.approx(1.0)


# ── cross_validate_model ──────────────────────────────────────────────────────

def test_cross_validate_model_structure(X_y):
    X, y = X_y
    results = cross_validate_model(
        model_factory=_make_dummy_pipeline,
        X=X, y=y, n_splits=3,
    )
    for metric in ["accuracy", "f1", "roc_auc", "mcc", "brier_score"]:
        assert metric in results
        assert "mean" in results[metric]
        assert "std"  in results[metric]
        assert "folds" in results[metric]
        assert len(results[metric]["folds"]) == 3


def test_cross_validate_std_nonnegative(X_y):
    X, y = X_y
    results = cross_validate_model(
        model_factory=_make_dummy_pipeline, X=X, y=y, n_splits=3
    )
    for metric in results.values():
        assert metric["std"] >= 0.0


# ── compare_models_mcnemar ────────────────────────────────────────────────────

def test_mcnemar_identical_models(X_y, trained_logreg):
    """Two identical models should have p-value = 1.0 (not significant)."""
    X, y = X_y
    result = compare_models_mcnemar(trained_logreg, trained_logreg, X, y)
    assert result["n01"] == 0
    assert result["n10"] == 0
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False


def test_mcnemar_keys(X_y, trained_logreg):
    X, y = X_y
    result = compare_models_mcnemar(trained_logreg, trained_logreg, X, y)
    assert {"n01", "n10", "statistic", "p_value", "significant"} == set(result.keys())


def test_mcnemar_p_value_range(X_y, trained_logreg):
    X, y = X_y
    dummy = _make_dummy_pipeline(strategy="stratified")
    dummy.fit(np.array(X), np.array(y))
    result = compare_models_mcnemar(trained_logreg, dummy, X, y)
    assert 0.0 <= result["p_value"] <= 1.0


# ── get_calibration_data ──────────────────────────────────────────────────────

def test_calibration_data_structure(X_y, trained_logreg):
    X, y = X_y
    cal = get_calibration_data(trained_logreg, X, y, n_bins=5)
    assert "fraction_of_positives" in cal
    assert "mean_predicted_value"  in cal
    assert len(cal["fraction_of_positives"]) == len(cal["mean_predicted_value"])
    assert len(cal["fraction_of_positives"]) <= 5


def test_calibration_values_in_range(X_y, trained_logreg):
    X, y = X_y
    cal = get_calibration_data(trained_logreg, X, y)
    for v in cal["fraction_of_positives"]:
        assert 0.0 <= v <= 1.0
    for v in cal["mean_predicted_value"]:
        assert 0.0 <= v <= 1.0
