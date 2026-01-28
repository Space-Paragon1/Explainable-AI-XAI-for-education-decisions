from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ExplanationConfig:
    # How many top positive/negative factors to mention
    top_k: int = 3
    # Thresholds for turning numeric features into friendly language
    low_study_time_min: float = 120.0
    low_practice_rate: float = 0.4
    low_attendance_rate: float = 0.7
    high_days_inactive: float = 7.0
    high_hint_usage: float = 0.65


# Friendly names for users (student view)
FRIENDLY_NAMES: Dict[str, str] = {
    "study_time_min": "study time",
    "practice_completion_rate": "practice completion",
    "avg_quiz_score": "average quiz score",
    "quiz_attempts": "quiz attempts",
    "hint_usage_rate": "hint usage",
    "attendance_rate": "attendance",
    "days_since_last_activity": "days since last activity",
    "stress_index": "stress level",
    "prereq_mastery": "prerequisite mastery",
    "device_reliability": "device reliability",
}


def _bucket_feature_value(feature: str, value: float, cfg: ExplanationConfig) -> str:
    """Turn raw numeric value into a short, human-friendly phrase."""
    if feature == "study_time_min":
        return "low" if value < cfg.low_study_time_min else "okay"
    if feature == "practice_completion_rate":
        return "low" if value < cfg.low_practice_rate else "good"
    if feature == "attendance_rate":
        return "low" if value < cfg.low_attendance_rate else "good"
    if feature == "days_since_last_activity":
        return "high" if value > cfg.high_days_inactive else "low"
    if feature == "hint_usage_rate":
        return "high" if value > cfg.high_hint_usage else "low"
    return "noted"


def summarize_shap_to_text(
    feature_names: List[str],
    feature_values: Dict[str, float],
    shap_values: List[float],
    predicted_prob_mastery: float,
    cfg: ExplanationConfig | None = None,
) -> str:
    """
    Build a student-friendly explanation using top positive/negative SHAP contributors.
    """
    cfg = cfg or ExplanationConfig()

    # Pair up feature -> shap contribution
    import numpy as np
    pairs: List[Tuple[str, float]] = list(zip(feature_names, shap_values))
    # Ensure the SHAP value is a scalar for sorting
    pairs_sorted = sorted(pairs, key=lambda x: abs(np.ravel(x[1])[0]), reverse=True)

    top = pairs_sorted[: max(cfg.top_k * 2, 6)]
    positives = [p for p in top if np.ravel(p[1])[0] > 0][: cfg.top_k]
    negatives = [p for p in top if np.ravel(p[1])[0] < 0][: cfg.top_k]

    label = "on track" if predicted_prob_mastery >= 0.5 else "at risk"

    def phrase(f: str, s: float) -> str:
        import numpy as np
        s = np.ravel(s)[0]
        name = FRIENDLY_NAMES.get(f, f)
        val_bucket = _bucket_feature_value(f, float(feature_values.get(f, 0.0)), cfg)
        direction = "helped" if s > 0 else "hurt"
        return f"{name} ({val_bucket}) {direction}"

    parts = [f"Prediction: **{label}** (mastery probability ≈ {predicted_prob_mastery:.2f})."]

    if negatives:
        parts.append("Main factors holding you back: " + "; ".join(phrase(f, s) for f, s in negatives) + ".")
    if positives:
        parts.append("Main factors helping you: " + "; ".join(phrase(f, s) for f, s in positives) + ".")

    # Add a gentle actionable hint if common issues appear
    actionable = []
    if feature_values.get("practice_completion_rate", 1.0) < cfg.low_practice_rate:
        actionable.append("finish more practice problems")
    if feature_values.get("study_time_min", 999.0) < cfg.low_study_time_min:
        actionable.append("increase study time")
    if feature_values.get("days_since_last_activity", 0.0) > cfg.high_days_inactive:
        actionable.append("study more consistently (avoid long gaps)")

    if actionable:
        parts.append("Action you can try this week: " + ", ".join(actionable) + ".")

    return " ".join(parts)
