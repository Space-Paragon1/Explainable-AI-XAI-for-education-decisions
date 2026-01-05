from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["logreg", "rf"]


@dataclass
class TrainedModel:
    name: ModelName
    pipeline: Pipeline


def build_model(model_name: ModelName) -> Pipeline:
    # All features are numeric in this Phase 1 dataset
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), slice(0, None)),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def train(model_name: ModelName, X_train, y_train) -> TrainedModel:
    pipe = build_model(model_name)
    pipe.fit(X_train, y_train)
    return TrainedModel(name=model_name, pipeline=pipe)


def save_model(trained: TrainedModel, path: str) -> None:
    joblib.dump(trained, path)


def load_model(path: str) -> TrainedModel:
    return joblib.load(path)
