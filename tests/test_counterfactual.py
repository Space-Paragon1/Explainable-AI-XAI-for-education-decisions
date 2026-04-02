"""
test_counterfactual.py — Tests for the counterfactual explanation engine.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.counterfactual import (
    generate_counterfactual,
    CounterfactualConfig,
    CONTROLLABLE,
)
from src.data_gen import generate_student_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def at_risk_row(trained_logreg):
    """Find a student who is actually predicted at-risk."""
    df = generate_student_dataset(n=500, seed=77)
    X  = df[FEATURE_COLUMNS]
    proba = trained_logreg.predict_proba(X)[:, 1]
    at_risk_idx = np.where(proba < 0.4)[0]
    if len(at_risk_idx) == 0:
        pytest.skip("No at-risk student found in fixture data.")
    return X.iloc[at_risk_idx[0]]


@pytest.fixture(scope="module")
def on_track_row(trained_logreg):
    """Find a student who is predicted on-track."""
    df = generate_student_dataset(n=500, seed=77)
    X  = df[FEATURE_COLUMNS]
    proba = trained_logreg.predict_proba(X)[:, 1]
    on_track_idx = np.where(proba >= 0.7)[0]
    if len(on_track_idx) == 0:
        pytest.skip("No on-track student found in fixture data.")
    return X.iloc[on_track_idx[0]]


# ── CONTROLLABLE features ─────────────────────────────────────────────────────

def test_controllable_features_are_in_feature_columns():
    for f in CONTROLLABLE:
        assert f in FEATURE_COLUMNS, f"CONTROLLABLE feature '{f}' not in FEATURE_COLUMNS"


def test_controllable_has_at_least_five_features():
    assert len(CONTROLLABLE) >= 5


# ── generate_counterfactual — on-track student ───────────────────────────────

def test_already_on_track_status(trained_logreg, on_track_row):
    result = generate_counterfactual(trained_logreg, on_track_row.copy())
    assert result["status"] == "already_on_track"
    assert result["edits"] == {}
    assert result["delta_prob"] == 0.0


def test_already_on_track_base_prob_positive(trained_logreg, on_track_row):
    result = generate_counterfactual(trained_logreg, on_track_row.copy())
    assert result["base_prob"] >= 0.5


# ── generate_counterfactual — at-risk student ────────────────────────────────

def test_result_keys(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    assert {"base_prob", "status", "edits", "new_prob", "delta_prob", "steps_taken"} \
           == set(result.keys())


def test_status_is_valid(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    assert result["status"] in {"flipped", "not_flipped", "already_on_track"}


def test_new_prob_geq_base_prob(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    assert result["new_prob"] >= result["base_prob"] - 1e-6


def test_delta_prob_consistent(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    assert abs(result["delta_prob"] - (result["new_prob"] - result["base_prob"])) < 1e-4


def test_flipped_new_prob_above_threshold(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    if result["status"] == "flipped":
        assert result["new_prob"] >= 0.5


def test_edits_only_controllable_features(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    for feat in result["edits"]:
        assert feat in CONTROLLABLE, f"Edit applied to non-controllable feature: {feat}"


def test_edit_values_within_bounds(trained_logreg, at_risk_row):
    cfg    = CounterfactualConfig()
    result = generate_counterfactual(trained_logreg, at_risk_row.copy(), cfg=cfg)
    for feat, val in result["edits"].items():
        lo, hi = cfg.bounds[feat]
        assert lo <= val <= hi, f"{feat}: {val} outside bounds [{lo}, {hi}]"


def test_steps_taken_nonnegative(trained_logreg, at_risk_row):
    result = generate_counterfactual(trained_logreg, at_risk_row.copy())
    assert result["steps_taken"] >= 0


def test_steps_taken_leq_max_steps(trained_logreg, at_risk_row):
    cfg    = CounterfactualConfig(max_steps=10)
    result = generate_counterfactual(trained_logreg, at_risk_row.copy(), cfg=cfg)
    assert result["steps_taken"] <= 10


# ── CounterfactualConfig ──────────────────────────────────────────────────────

def test_config_default_step_sizes():
    cfg = CounterfactualConfig()
    assert cfg.step_sizes["practice_completion_rate"] > 0
    assert cfg.step_sizes["study_time_min"] > 0
    assert cfg.step_sizes["hint_usage_rate"] < 0   # decrease
    assert cfg.step_sizes["stress_index"] < 0      # decrease


def test_config_bounds_valid():
    cfg = CounterfactualConfig()
    for feat, (lo, hi) in cfg.bounds.items():
        assert lo < hi, f"Bound for '{feat}' is invalid: [{lo}, {hi}]"
