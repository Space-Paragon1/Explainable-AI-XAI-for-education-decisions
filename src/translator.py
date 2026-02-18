"""
translator.py — Human-friendly explanation text for XAI-ED.

Two audience modes:
  - summarize_shap_to_text()  : Student-friendly, actionable, encouraging language.
  - summarize_for_instructor(): Technical, data-rich instructor/advisor view.
"""

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

# Units / display suffixes for instructor view
FEATURE_UNITS: Dict[str, str] = {
    "study_time_min": "min/wk",
    "practice_completion_rate": "",
    "avg_quiz_score": "/100",
    "quiz_attempts": "attempts",
    "hint_usage_rate": "",
    "attendance_rate": "",
    "days_since_last_activity": "days",
    "stress_index": "",
    "prereq_mastery": "",
    "device_reliability": "",
}

# Risk tier thresholds (mastery probability)
_RISK_TIERS = [
    (0.30, "High Risk"),
    (0.50, "At Risk"),
    (0.70, "Borderline"),
    (0.85, "On Track"),
    (1.01, "Strong"),
]


def _risk_tier(prob: float) -> str:
    for threshold, label in _RISK_TIERS:
        if prob < threshold:
            return label
    return "Strong"


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


def summarize_for_instructor(
    feature_names: List[str],
    feature_values: Dict[str, float],
    shap_values: List[float],
    predicted_prob_mastery: float,
    model_name: str,
    counterfactual: Dict | None = None,
    cfg: ExplanationConfig | None = None,
) -> str:
    """
    Build a technical, data-rich explanation for instructors / academic advisors.

    Includes:
    - Risk tier and exact probability
    - Model used
    - Top contributing features with raw values and SHAP magnitudes
    - Counterfactual summary (what would flip the prediction)
    - Recommended intervention category
    """
    cfg = cfg or ExplanationConfig()
    import numpy as np

    tier = _risk_tier(predicted_prob_mastery)
    pairs: List[Tuple[str, float]] = list(zip(feature_names, shap_values))
    pairs_sorted = sorted(pairs, key=lambda x: abs(np.ravel(x[1])[0]), reverse=True)
    top_factors = pairs_sorted[:5]

    lines = [
        f"=== INSTRUCTOR REPORT (Model: {model_name.upper()}) ===",
        f"Risk Tier    : {tier}",
        f"Mastery Prob : {predicted_prob_mastery:.3f} ({'ON TRACK' if predicted_prob_mastery >= 0.5 else 'AT RISK'})",
        "",
        "Top 5 Influential Features (SHAP):",
    ]

    for rank, (feat, shap_val) in enumerate(top_factors, start=1):
        sv = float(np.ravel(shap_val)[0])
        raw_val = feature_values.get(feat, float("nan"))
        unit = FEATURE_UNITS.get(feat, "")
        direction = "positive" if sv > 0 else "negative"
        lines.append(
            f"  {rank}. {FRIENDLY_NAMES.get(feat, feat):<28}"
            f"value={raw_val:.2f} {unit:<10}"
            f"SHAP={sv:+.4f}  ({direction})"
        )

    # Intervention recommendation
    lines.append("")
    if predicted_prob_mastery < 0.30:
        rec = "URGENT: schedule 1-on-1 advising session immediately."
    elif predicted_prob_mastery < 0.50:
        rec = "RECOMMENDED: outreach within 48 hours; review practice/attendance trends."
    elif predicted_prob_mastery < 0.70:
        rec = "MONITOR: check in at next scheduled session; encourage consistency."
    else:
        rec = "LOW PRIORITY: student is on track; positive reinforcement suggested."
    lines.append(f"Recommended Action: {rec}")

    # Counterfactual summary
    if counterfactual:
        lines.append("")
        status = counterfactual.get("status", "unknown")
        if status == "already_on_track":
            lines.append("Counterfactual: Student is already predicted on-track.")
        elif status == "flipped":
            edits = counterfactual.get("edits", {})
            new_prob = counterfactual.get("new_prob", 0.0)
            edit_str = ", ".join(
                f"{FRIENDLY_NAMES.get(k, k)} -> {v:.2f}" for k, v in edits.items()
            )
            lines.append(
                f"Counterfactual: Prediction flips to on-track (p={new_prob:.2f}) "
                f"if student improves: {edit_str}."
            )
        elif status == "not_flipped":
            new_prob = counterfactual.get("new_prob", 0.0)
            lines.append(
                f"Counterfactual: Could not flip prediction with controllable features alone "
                f"(best p={new_prob:.2f}). Consider structural support."
            )

    return "\n".join(lines)
