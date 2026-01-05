from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    outputs_dir: Path = project_root / "outputs"
    models_dir: Path = outputs_dir / "models"
    metrics_dir: Path = outputs_dir / "metrics"
    explanations_dir: Path = outputs_dir / "explanations"

FEATURE_COLUMNS = [
    "study_time_min",
    "practice_completion_rate",
    "avg_quiz_score",
    "quiz_attempts",
    "hint_usage_rate",
    "attendance_rate",
    "days_since_last_activity",
    "stress_index",
    "prereq_mastery",
    "device_reliability",
]

TARGET_COLUMN = "mastery"  # 1 = mastered / on-track, 0 = at-risk
