from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CounterfactualConfig:
    max_steps: int = 25
    step_size_practice: float = 0.05
    step_size_study_min: float = 20.0
    step_size_attendance: float = 0.03
    # clamp ranges
    study_min_max: float = 900.0


CONTROLLABLE = ["practice_completion_rate", "study_time_min", "attendance_rate"]


def generate_counterfactual(
    model_pipeline,
    x_row: pd.Series,
    cfg: CounterfactualConfig | None = None,
) -> Dict:
    """
    Returns minimal edits (heuristic) that change prediction from at-risk -> on track.
    """
    cfg = cfg or CounterfactualConfig()

    x = x_row.copy()
    base_prob = float(model_pipeline.predict_proba(pd.DataFrame([x]))[:, 1][0])

    if base_prob >= 0.5:
        return {
            "base_prob": base_prob,
            "status": "already_on_track",
            "edits": {},
            "new_prob": base_prob,
        }

    edits: Dict[str, float] = {}
    current_prob = base_prob

    for _ in range(cfg.max_steps):
        # try increasing each controllable feature a little and choose best improvement
        candidates: List[Tuple[str, float, float]] = []  # (feature, new_value, prob)
        for f in CONTROLLABLE:
            x_try = x.copy()
            if f == "practice_completion_rate":
                x_try[f] = float(min(1.0, x_try[f] + cfg.step_size_practice))
            elif f == "study_time_min":
                x_try[f] = float(min(cfg.study_min_max, x_try[f] + cfg.step_size_study_min))
            elif f == "attendance_rate":
                x_try[f] = float(min(1.0, x_try[f] + cfg.step_size_attendance))

            p = float(model_pipeline.predict_proba(pd.DataFrame([x_try]))[:, 1][0])
            candidates.append((f, float(x_try[f]), p))

        # pick the single best edit this step
        best_f, best_val, best_p = max(candidates, key=lambda t: t[2])

        # stop if no improvement
        if best_p <= current_prob + 1e-6:
            break

        # apply
        x[best_f] = best_val
        edits[best_f] = best_val
        current_prob = best_p

        if current_prob >= 0.5:
            return {
                "base_prob": base_prob,
                "status": "flipped",
                "edits": edits,
                "new_prob": current_prob,
            }

    return {
        "base_prob": base_prob,
        "status": "not_flipped",
        "edits": edits,
        "new_prob": current_prob,
    }
