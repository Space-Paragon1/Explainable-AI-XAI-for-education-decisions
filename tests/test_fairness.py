"""
test_fairness.py — Tests for group-level fairness metric computation.
"""
import numpy as np
import pandas as pd
import pytest

from src.fairness import compute_fairness_metrics, _bin_continuous


# ── _bin_continuous ───────────────────────────────────────────────────────────

def test_bin_continuous_produces_three_groups():
    s = pd.Series(np.linspace(0, 1, 300))
    binned = _bin_continuous(s)
    assert set(binned.dropna().astype(str).unique()) == {"Low", "Medium", "High"}


def test_bin_continuous_roughly_equal_sizes():
    s = pd.Series(np.linspace(0, 1, 300))
    binned = _bin_continuous(s)
    counts = binned.value_counts()
    # Each group should have ~100 members (±30 for edge effects)
    for count in counts.values:
        assert 70 <= count <= 130


# ── compute_fairness_metrics ──────────────────────────────────────────────────

class _PerfectBinaryModel:
    """Mock model that predicts perfectly."""
    def predict_proba(self, X):
        # Not actually using X — this fixture overrides predictions externally
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


class _MockModel:
    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - self._proba[:n], self._proba[:n]])


def _make_fairness_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X   = pd.DataFrame(np.zeros((n, 3)), columns=["a", "b", "c"])
    y   = pd.Series(rng.binomial(1, 0.7, size=n))
    return X, y


def test_compute_fairness_returns_required_keys(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "gender")
    required = {
        "group_col", "groups", "group_metrics",
        "disparate_impact_ratio", "equal_opportunity_gap",
        "demographic_parity_gap", "fairness_flags",
    }
    assert required.issubset(set(result.keys()))


def test_compute_fairness_group_metrics_have_correct_keys(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "gender")
    for group, metrics in result["group_metrics"].items():
        for key in ["n", "accuracy", "tpr", "fpr", "precision", "positive_rate"]:
            assert key in metrics, f"Group '{group}' missing metric '{key}'"


def test_disparate_impact_ratio_range(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "gender")
    di = result["disparate_impact_ratio"]
    assert 0.0 <= di <= 1.0


def test_gap_metrics_nonnegative(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "first_gen")
    assert result["equal_opportunity_gap"]  >= 0.0
    assert result["demographic_parity_gap"] >= 0.0


def test_fairness_flags_are_bool(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "gender")
    for flag_name, flag_val in result["fairness_flags"].items():
        assert isinstance(flag_val, bool), f"Flag '{flag_name}' is not bool: {flag_val}"


def test_ses_index_auto_binned(trained_logreg, X_y, demo_df):
    """ses_index is continuous — should be auto-binned and is_binned=True."""
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "ses_index")
    assert result["is_binned"] is True
    assert set(result["groups"]).issubset({"Low", "Medium", "High"})


def test_categorical_column_not_binned(trained_logreg, X_y, demo_df):
    X, y = X_y
    result = compute_fairness_metrics(trained_logreg, X, y, demo_df, "first_gen")
    assert result["is_binned"] is False


def test_perfect_model_disparate_impact_near_one():
    """A model that gives every group the same positive rate should have DI ≈ 1."""
    n = 300
    X = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    y = pd.Series(np.ones(n, dtype=int))
    demo = pd.DataFrame({"gender": np.tile([0, 1, 2], n // 3)})
    model = _MockModel(np.ones(n))
    result = compute_fairness_metrics(model, X, y, demo, "gender")
    assert result["disparate_impact_ratio"] == pytest.approx(1.0, abs=0.01)
    assert result["demographic_parity_gap"] == pytest.approx(0.0, abs=0.01)


def test_known_gap_detected():
    """
    Construct a scenario where group 0 gets high positive rate and
    group 1 gets low positive rate — disparate impact should be < 0.8.
    """
    n = 200
    X    = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    y    = pd.Series(np.zeros(n, dtype=int))
    demo = pd.DataFrame({"group": np.repeat([0, 1], n // 2)})

    # Group 0: always predicted positive (prob=0.9)
    # Group 1: always predicted negative (prob=0.1)
    proba = np.where(demo["group"] == 0, 0.9, 0.1)
    model = _MockModel(proba)

    result = compute_fairness_metrics(model, X, y, demo, "group")
    assert result["disparate_impact_ratio"] < 0.5
    assert result["fairness_flags"]["disparate_impact_ok"] is False
