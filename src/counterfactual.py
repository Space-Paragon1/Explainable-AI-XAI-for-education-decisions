from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CounterfactualConfig:
    max_steps: int = 40
    # Step sizes per feature (one greedy step per iteration)
    step_sizes: Dict[str, float] = field(default_factory=lambda: {
        "practice_completion_rate": 0.05,
        "study_time_min":           20.0,
        "attendance_rate":          0.03,
        "quiz_attempts":            1.0,
        "prereq_mastery":           0.04,
        "hint_usage_rate":          -0.04,   # negative: we *decrease* hint dependence
        "stress_index":             -0.03,   # negative: we *decrease* stress
    })
    # Hard bounds per feature
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "practice_completion_rate": (0.0, 1.0),
        "study_time_min":           (0.0, 900.0),
        "attendance_rate":          (0.0, 1.0),
        "quiz_attempts":            (0.0, 25.0),
        "prereq_mastery":           (0.0, 1.0),
        "hint_usage_rate":          (0.0, 1.0),
        "stress_index":             (0.0, 1.0),
    })


# Ordered by expected impact — the greedy algorithm will try all and pick best
CONTROLLABLE = list(CounterfactualConfig().step_sizes.keys())


def generate_counterfactual(
    model_pipeline,
    x_row: pd.Series,
    cfg: CounterfactualConfig | None = None,
) -> Dict:
    """
    Generate a minimal counterfactual explanation using greedy hill-climbing.

    Iteratively applies the single best incremental change across all
    controllable features until the prediction flips to on-track (p >= 0.5)
    or the maximum number of steps is reached.

    Controllable features (and their directions):
        practice_completion_rate  ↑  (do more practice)
        study_time_min            ↑  (study longer)
        attendance_rate           ↑  (attend more sessions)
        quiz_attempts             ↑  (take more quizzes)
        prereq_mastery            ↑  (review prerequisites)
        hint_usage_rate           ↓  (rely less on hints)
        stress_index              ↓  (reduce stress)

    Returns
    -------
    dict with keys: base_prob, status, edits, new_prob, delta_prob, steps_taken
    """
    cfg = cfg or CounterfactualConfig()

    x = x_row.copy()
    base_prob = float(model_pipeline.predict_proba(pd.DataFrame([x]))[:, 1][0])

    if base_prob >= 0.5:
        return {
            "base_prob": round(base_prob, 4),
            "status": "already_on_track",
            "edits": {},
            "new_prob": round(base_prob, 4),
            "delta_prob": 0.0,
            "steps_taken": 0,
        }

    edits: Dict[str, float] = {}
    current_prob = base_prob
    steps_taken = 0

    for _ in range(cfg.max_steps):
        candidates: List[Tuple[str, float, float]] = []

        for f in CONTROLLABLE:
            if f not in x.index:
                continue
            step = cfg.step_sizes[f]
            lo, hi = cfg.bounds[f]
            current_val = float(x[f])
            new_val = float(np.clip(current_val + step, lo, hi))

            # Skip if already at the bound and no more movement possible
            if abs(new_val - current_val) < 1e-9:
                continue

            x_try = x.copy()
            x_try[f] = new_val
            p = float(model_pipeline.predict_proba(pd.DataFrame([x_try]))[:, 1][0])
            candidates.append((f, new_val, p))

        if not candidates:
            break

        best_f, best_val, best_p = max(candidates, key=lambda t: t[2])

        if best_p <= current_prob + 1e-6:
            break

        x[best_f] = best_val
        edits[best_f] = round(best_val, 4)
        current_prob = best_p
        steps_taken += 1

        if current_prob >= 0.5:
            return {
                "base_prob":   round(base_prob, 4),
                "status":      "flipped",
                "edits":       edits,
                "new_prob":    round(current_prob, 4),
                "delta_prob":  round(current_prob - base_prob, 4),
                "steps_taken": steps_taken,
            }

    return {
        "base_prob":   round(base_prob, 4),
        "status":      "not_flipped",
        "edits":       edits,
        "new_prob":    round(current_prob, 4),
        "delta_prob":  round(current_prob - base_prob, 4),
        "steps_taken": steps_taken,
    }
