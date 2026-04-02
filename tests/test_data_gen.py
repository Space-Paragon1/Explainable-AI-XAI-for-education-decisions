"""
test_data_gen.py — Tests for synthetic dataset generation with equity gaps.
"""
import numpy as np
import pytest
from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_gen import generate_student_dataset


def test_output_shape():
    df = generate_student_dataset(n=500, seed=1)
    assert len(df) == 500
    assert TARGET_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_demographic_columns_present():
    df = generate_student_dataset(n=100, seed=1)
    for col in ["gender", "ses_index", "first_gen"]:
        assert col in df.columns


def test_feature_ranges():
    df = generate_student_dataset(n=500, seed=1)
    assert df["study_time_min"].between(0, 900).all()
    assert df["practice_completion_rate"].between(0, 1).all()
    assert df["avg_quiz_score"].between(0, 100).all()
    assert df["quiz_attempts"].between(0, 25).all()
    assert df["hint_usage_rate"].between(0, 1).all()
    assert df["attendance_rate"].between(0, 1).all()
    assert df["days_since_last_activity"].between(0, 30).all()
    assert df["stress_index"].between(0, 1).all()
    assert df["prereq_mastery"].between(0, 1).all()
    assert df["device_reliability"].between(0, 1).all()


def test_target_is_binary():
    df = generate_student_dataset(n=200, seed=1)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_mastery_rate_reasonable():
    df = generate_student_dataset(n=1000, seed=42)
    rate = df[TARGET_COLUMN].mean()
    # Overall mastery rate should be between 70% and 90%
    assert 0.70 <= rate <= 0.90, f"Mastery rate out of expected range: {rate:.3f}"


def test_ses_equity_gap():
    """Low-SES students should have a lower mastery rate than high-SES students."""
    df = generate_student_dataset(n=1500, seed=42)
    low_ses_rate  = df[df["ses_index"] < 0.33][TARGET_COLUMN].mean()
    high_ses_rate = df[df["ses_index"] > 0.67][TARGET_COLUMN].mean()
    assert low_ses_rate < high_ses_rate, (
        f"Expected SES equity gap: low={low_ses_rate:.3f} should be < high={high_ses_rate:.3f}"
    )


def test_first_gen_equity_gap():
    """First-generation students should have a lower mastery rate on average."""
    df = generate_student_dataset(n=1500, seed=42)
    first_gen_rate = df[df["first_gen"] == 1][TARGET_COLUMN].mean()
    cont_rate      = df[df["first_gen"] == 0][TARGET_COLUMN].mean()
    assert first_gen_rate < cont_rate, (
        f"Expected first-gen gap: {first_gen_rate:.3f} should be < {cont_rate:.3f}"
    )


def test_ses_affects_study_time():
    """Low-SES students should have lower average study time."""
    df = generate_student_dataset(n=1500, seed=42)
    low_ses_study  = df[df["ses_index"] < 0.33]["study_time_min"].mean()
    high_ses_study = df[df["ses_index"] > 0.67]["study_time_min"].mean()
    assert low_ses_study < high_ses_study


def test_reproducibility():
    df1 = generate_student_dataset(n=100, seed=99)
    df2 = generate_student_dataset(n=100, seed=99)
    assert df1.equals(df2)


def test_different_seeds_differ():
    df1 = generate_student_dataset(n=100, seed=1)
    df2 = generate_student_dataset(n=100, seed=2)
    assert not df1.equals(df2)
