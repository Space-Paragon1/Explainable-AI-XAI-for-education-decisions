import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_student_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a privacy-safe synthetic dataset that mimics common learning analytics signals.
    Target: mastery (1) vs at-risk (0)

    Includes demographic columns (gender, ses_index, first_gen) for fairness analysis.
    Demographics are NOT used in model training — only for post-hoc bias evaluation.
    """
    rng = np.random.default_rng(seed)

    # ── Academic / behavioral features ──────────────────────────────────────

    # Study time per week (minutes)
    study_time_min = np.clip(rng.normal(loc=220, scale=110, size=n), 0, 900)

    # Practice completion (0-1)
    practice_completion_rate = np.clip(rng.beta(a=2.2, b=1.8, size=n), 0, 1)

    # Avg quiz score (0-100)
    avg_quiz_score = np.clip(rng.normal(loc=72, scale=14, size=n), 0, 100)

    # Quiz attempts (count)
    quiz_attempts = np.clip(rng.poisson(lam=4.5, size=n), 0, 25)

    # Hint usage rate (0-1)
    hint_usage_rate = np.clip(rng.beta(a=1.7, b=3.0, size=n), 0, 1)

    # Attendance rate (0-1)
    attendance_rate = np.clip(rng.beta(a=4.0, b=1.6, size=n), 0, 1)

    # Days since last activity (0-30)
    days_since_last_activity = np.clip(rng.gamma(shape=2.2, scale=2.2, size=n), 0, 30)

    # Stress index (0-1) (proxy for workload/pressure)
    stress_index = np.clip(rng.beta(a=2.0, b=2.2, size=n), 0, 1)

    # Prerequisite mastery (0-1)
    prereq_mastery = np.clip(rng.beta(a=2.5, b=2.0, size=n), 0, 1)

    # Device reliability (0-1) (connectivity/tech issues)
    device_reliability = np.clip(rng.beta(a=5.0, b=1.8, size=n), 0, 1)

    # ── Mastery label ────────────────────────────────────────────────────────
    # Positive contributors: practice, quiz score, prereq mastery, attendance, study time
    # Negative contributors: days inactive, high stress, excessive hint dependence
    linear = (
        0.006 * study_time_min
        + 2.0 * practice_completion_rate
        + 0.05 * avg_quiz_score
        + 0.10 * np.log1p(quiz_attempts)
        - 1.2 * hint_usage_rate
        + 1.8 * attendance_rate
        - 0.12 * days_since_last_activity
        - 0.9 * stress_index
        + 2.2 * prereq_mastery
        + 0.9 * device_reliability
        - 6.0  # bias term to set overall base rate
    )

    p_mastery = sigmoid(linear)
    mastery = rng.binomial(n=1, p=p_mastery, size=n).astype(int)

    # ── Demographic features (for fairness analysis only) ────────────────────
    # gender: 0=female, 1=male, 2=non-binary  (roughly uniform)
    gender = rng.integers(0, 3, size=n)

    # ses_index: socioeconomic status 0.0–1.0; Beta(2,2) centered around 0.5
    ses_index = np.clip(rng.beta(a=2.0, b=2.0, size=n), 0, 1)

    # first_gen: first-generation college student (35% prevalence)
    first_gen = rng.binomial(n=1, p=0.35, size=n)

    df = pd.DataFrame(
        {
            # Academic features (used in model)
            "study_time_min": study_time_min.round(1),
            "practice_completion_rate": practice_completion_rate.round(3),
            "avg_quiz_score": avg_quiz_score.round(1),
            "quiz_attempts": quiz_attempts.astype(int),
            "hint_usage_rate": hint_usage_rate.round(3),
            "attendance_rate": attendance_rate.round(3),
            "days_since_last_activity": days_since_last_activity.round(1),
            "stress_index": stress_index.round(3),
            "prereq_mastery": prereq_mastery.round(3),
            "device_reliability": device_reliability.round(3),
            # Target
            "mastery": mastery,
            # Demographic features (fairness analysis only)
            "gender": gender,
            "ses_index": ses_index.round(3),
            "first_gen": first_gen,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic student dataset for XAI-ED.")
    parser.add_argument("--out", type=str, default="data/student_data.csv", help="Output CSV path.")
    parser.add_argument("--n", type=int, default=3000, help="Number of rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_student_dataset(n=args.n, seed=args.seed)
    df.to_csv(out_path, index=False)

    mastery_rate = df["mastery"].mean()
    print(f"Saved dataset to {out_path} | n={len(df)} | mastery_rate={mastery_rate:.3f}")
    print(f"Demographic columns added: gender, ses_index, first_gen")


if __name__ == "__main__":
    main()
