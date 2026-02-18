from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelName = Literal["logreg", "rf", "xgb", "lgbm"]


@dataclass
class TrainedModel:
    name: ModelName
    pipeline: Pipeline


def build_model(model_name: ModelName, y_train=None) -> Pipeline:
    """
    Build a sklearn Pipeline for the given model name.

    For logreg: uses StandardScaler preprocessor.
    For rf, xgb, lgbm: uses passthrough preprocessor (tree models don't need scaling).
    XGBoost scale_pos_weight is set from y_train class ratio if provided.
    """
    if model_name == "logreg":
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), slice(0, None))],
            remainder="drop",
        )
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )

    elif model_name == "rf":
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), slice(0, None))],
            remainder="drop",
        )
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "xgb":
        from xgboost import XGBClassifier

        # Compute scale_pos_weight from training label ratio (negatives / positives)
        if y_train is not None:
            n_neg = int(np.sum(y_train == 0))
            n_pos = int(np.sum(y_train == 1))
            scale_pos_weight = n_neg / max(n_pos, 1)
        else:
            scale_pos_weight = 1.0

        preprocessor = ColumnTransformer(
            transformers=[("passthrough", "passthrough", slice(0, None))],
            remainder="drop",
        )
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier

        preprocessor = ColumnTransformer(
            transformers=[("passthrough", "passthrough", slice(0, None))],
            remainder="drop",
        )
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}. Choose from: logreg, rf, xgb, lgbm")

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def train(model_name: ModelName, X_train, y_train) -> TrainedModel:
    pipe = build_model(model_name, y_train=y_train)
    pipe.fit(X_train, y_train)
    return TrainedModel(name=model_name, pipeline=pipe)


def save_model(trained: TrainedModel, path: str) -> None:
    joblib.dump(trained, path)


def load_model(path: str) -> TrainedModel:
    return joblib.load(path)
