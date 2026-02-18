"""
fairness.py — Group-level fairness analysis for XAI-ED.

Computes standard algorithmic fairness metrics across demographic subgroups:
  - Equal Opportunity (True Positive Rate parity)
  - Demographic Parity (Positive Prediction Rate parity)
  - Predictive Parity (Precision parity)
  - Disparate Impact Ratio (80% rule)

Usage:
    from src.fairness import compute_fairness_metrics
    results = compute_fairness_metrics(model, X_test, y_test, demo_df, group_col="gender")
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute per-group metrics for a single group subset."""
    n = len(y_true)
    if n == 0:
        return {}

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    tpr = tp / max(tp + fn, 1)         # Equal Opportunity / Recall
    fpr = fp / max(fp + tn, 1)         # False Positive Rate
    precision = tp / max(tp + fp, 1)   # Predictive Parity
    positive_rate = (tp + fp) / n      # Demographic Parity
    accuracy = (tp + tn) / n

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "positive_rate": round(positive_rate, 4),
    }


def compute_fairness_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    demographic_df: pd.DataFrame,
    group_col: str,
) -> dict:
    """
    Compute group-level fairness metrics.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    X_test : feature matrix (same index as demographic_df)
    y_test : true labels (same index as demographic_df)
    demographic_df : DataFrame with demographic columns, aligned with X_test
    group_col : column in demographic_df to group by (e.g., "gender", "first_gen")

    Returns
    -------
    dict with:
        - group_metrics: per-group metric breakdown
        - disparate_impact_ratio: min/max positive prediction rate
        - equal_opportunity_gap: max - min TPR across groups
        - demographic_parity_gap: max - min positive rate across groups
        - fairness_flags: boolean indicators for common thresholds
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    y_true = np.array(y_test)

    groups = demographic_df[group_col].values
    unique_groups = sorted(np.unique(groups))

    group_results = {}
    for g in unique_groups:
        mask = groups == g
        gm = _group_metrics(y_true[mask], y_pred[mask], proba[mask])
        group_results[str(g)] = gm

    # Aggregate summary statistics
    tpr_values = [v["tpr"] for v in group_results.values() if "tpr" in v]
    positive_rates = [v["positive_rate"] for v in group_results.values() if "positive_rate" in v]

    equal_opportunity_gap = float(max(tpr_values) - min(tpr_values)) if tpr_values else 0.0
    demographic_parity_gap = float(max(positive_rates) - min(positive_rates)) if positive_rates else 0.0

    # Disparate Impact Ratio: min(positive_rate) / max(positive_rate) — should be >= 0.8
    if positive_rates and max(positive_rates) > 0:
        disparate_impact_ratio = float(min(positive_rates) / max(positive_rates))
    else:
        disparate_impact_ratio = 1.0

    return {
        "group_col": group_col,
        "groups": list(map(str, unique_groups)),
        "group_metrics": group_results,
        "disparate_impact_ratio": round(disparate_impact_ratio, 4),
        "equal_opportunity_gap": round(equal_opportunity_gap, 4),
        "demographic_parity_gap": round(demographic_parity_gap, 4),
        "fairness_flags": {
            "disparate_impact_ok": disparate_impact_ratio >= 0.8,
            "equal_opportunity_ok": equal_opportunity_gap <= 0.1,
            "demographic_parity_ok": demographic_parity_gap <= 0.1,
        },
    }
