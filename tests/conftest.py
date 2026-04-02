"""
conftest.py — Shared pytest fixtures for XAI-ED test suite.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_gen import generate_student_dataset


@pytest.fixture(scope="session")
def synthetic_df():
    """Small synthetic dataset (200 rows) for fast tests."""
    return generate_student_dataset(n=200, seed=0)


@pytest.fixture(scope="session")
def X_y(synthetic_df):
    X = synthetic_df[FEATURE_COLUMNS].copy()
    y = synthetic_df[TARGET_COLUMN].astype(int).copy()
    return X, y


@pytest.fixture(scope="session")
def demo_df(synthetic_df):
    return synthetic_df[["gender", "ses_index", "first_gen"]].copy()


@pytest.fixture(scope="session")
def trained_logreg(X_y):
    """A fast LogisticRegression pipeline fitted on the test data."""
    X, y = X_y
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), slice(0, None))],
        remainder="drop",
    )
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced",
                                   random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe
