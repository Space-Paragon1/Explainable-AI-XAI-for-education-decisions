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

    Demographics (gender, ses_index, first_gen) are generated first and then used to
    realistically shift academic feature distributions — producing genuine fairness gaps
    that reflect known equity patterns in education research:

        - Low SES students: less study time, lower attendance, weaker device access
        - First-generation students: lower prereq mastery, fewer quiz attempts,
          slightly higher hint dependence

    Demographics are NOT used in model training — only for post-hoc bias evaluation.
    """
    rng = np.random.default_rng(seed)

    # ── Demographic features (generated first so we can condition on them) ────
    # gender: 0=female, 1=male, 2=non-binary  (roughly uniform)
    gender = rng.integers(0, 3, size=n)

    # ses_index: socioeconomic status 0.0–1.0; Beta(2,2) centred around 0.5
    ses_index = np.clip(rng.beta(a=2.0, b=2.0, size=n), 0, 1)

    # first_gen: first-generation college student (35% prevalence)
    first_gen = rng.binomial(n=1, p=0.35, size=n)

    # ── Equity adjustment factors ─────────────────────────────────────────────
    # Low SES (bottom tercile) and first-gen students face structural disadvantages.
    # These are additive shifts applied *before* clipping, calibrated to reflect
    # effect sizes reported in educational equity literature.
    low_ses = (ses_index < 0.33).astype(float)   # bottom third
    high_ses = (ses_index > 0.67).astype(float)  # top third

    # ── Academic / behavioral features ───────────────────────────────────────

    # Study time per week (minutes) — SES effect: low SES → ~50 min/wk less
    study_time_min = np.clip(
        rng.normal(loc=220, scale=110, size=n)
        - 55.0 * low_ses
        + 30.0 * high_ses,
        0, 900,
    )

    # Practice completion (0–1)
    practice_completion_rate = np.clip(
        rng.beta(a=2.2, b=1.8, size=n)
        - 0.07 * low_ses * rng.uniform(0.5, 1.5, size=n),
        0, 1,
    )

    # Avg quiz score (0–100) — SES gap: ~5-point deficit for low SES
    avg_quiz_score = np.clip(
        rng.normal(loc=72, scale=14, size=n)
        - 5.5 * low_ses
        + 3.0 * high_ses,
        0, 100,
    )

    # Quiz attempts (count) — first-gen students attempt slightly fewer
    quiz_attempts = np.clip(
        rng.poisson(lam=4.5, size=n)
        - first_gen * rng.binomial(n=1, p=0.40, size=n),
        0, 25,
    )

    # Hint usage rate (0–1) — first-gen rely on hints slightly more
    hint_usage_rate = np.clip(
        rng.beta(a=1.7, b=3.0, size=n)
        + 0.06 * first_gen * rng.uniform(0.5, 1.5, size=n),
        0, 1,
    )

    # Attendance rate (0–1) — low SES: transport, work obligations → ~7% lower
    attendance_rate = np.clip(
        rng.beta(a=4.0, b=1.6, size=n)
        - 0.07 * low_ses
        + 0.03 * high_ses,
        0, 1,
    )

    # Days since last activity (0–30) — low SES slightly more inactive
    days_since_last_activity = np.clip(
        rng.gamma(shape=2.2, scale=2.2, size=n)
        + 1.5 * low_ses,
        0, 30,
    )

    # Stress index (0–1) — low SES and first-gen carry more financial/social stress
    stress_index = np.clip(
        rng.beta(a=2.0, b=2.2, size=n)
        + 0.07 * low_ses
        + 0.05 * first_gen,
        0, 1,
    )

    # Prerequisite mastery (0–1) — first-gen gap in prior academic preparation
    prereq_mastery = np.clip(
        rng.beta(a=2.5, b=2.0, size=n)
        - 0.09 * first_gen * rng.uniform(0.5, 1.5, size=n),
        0, 1,
    )

    # Device reliability (0–1) — low SES: less reliable devices/connectivity
    device_reliability = np.clip(
        rng.beta(a=5.0, b=1.8, size=n)
        - 0.12 * low_ses,
        0, 1,
    )

    # ── Mastery label ─────────────────────────────────────────────────────────
    # Positive contributors: practice, quiz score, prereq mastery, attendance, study time
    # Negative contributors: days inactive, high stress, excessive hint dependence
    # Bias term -6.0 calibrated to produce ~82% overall mastery rate.
    linear = (
        0.006 * study_time_min
        + 2.0  * practice_completion_rate
        + 0.05 * avg_quiz_score
        + 0.10 * np.log1p(quiz_attempts)
        - 1.2  * hint_usage_rate
        + 1.8  * attendance_rate
        - 0.12 * days_since_last_activity
        - 0.9  * stress_index
        + 2.2  * prereq_mastery
        + 0.9  * device_reliability
        - 6.0
    )

    p_mastery = sigmoid(linear)
    mastery = rng.binomial(n=1, p=p_mastery, size=n).astype(int)

    df = pd.DataFrame(
        {
            # Academic features (used in model)
            "study_time_min":            study_time_min.round(1),
            "practice_completion_rate":  practice_completion_rate.round(3),
            "avg_quiz_score":            avg_quiz_score.round(1),
            "quiz_attempts":             quiz_attempts.astype(int),
            "hint_usage_rate":           hint_usage_rate.round(3),
            "attendance_rate":           attendance_rate.round(3),
            "days_since_last_activity":  days_since_last_activity.round(1),
            "stress_index":              stress_index.round(3),
            "prereq_mastery":            prereq_mastery.round(3),
            "device_reliability":        device_reliability.round(3),
            # Target
            "mastery": mastery,
            # Demographic features (fairness analysis only — NOT used in training)
            "gender":    gender,
            "ses_index": ses_index.round(3),
            "first_gen": first_gen,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic student dataset for XAI-ED."
    )
    parser.add_argument("--out",  type=str, default="data/student_data.csv")
    parser.add_argument("--n",    type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_student_dataset(n=args.n, seed=args.seed)
    df.to_csv(out_path, index=False)

    mastery_rate = df["mastery"].mean()
    low_ses_mastery  = df[df["ses_index"] < 0.33]["mastery"].mean()
    high_ses_mastery = df[df["ses_index"] > 0.67]["mastery"].mean()
    fg_mastery       = df[df["first_gen"] == 1]["mastery"].mean()
    cont_mastery     = df[df["first_gen"] == 0]["mastery"].mean()

    print(f"Saved dataset to {out_path} | n={len(df)} | mastery_rate={mastery_rate:.3f}")
    print(f"Fairness gaps (mastery rate by group):")
    print(f"  Low SES:        {low_ses_mastery:.3f}  |  High SES: {high_ses_mastery:.3f}")
    print(f"  First-gen:      {fg_mastery:.3f}  |  Continuing: {cont_mastery:.3f}")


if __name__ == "__main__":
    main()
