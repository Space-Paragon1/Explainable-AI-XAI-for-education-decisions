"""
app.py — FastAPI REST API for XAI-ED.

Run with:
    uvicorn app:app --reload

Interactive docs at:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, Paths
from src.counterfactual import generate_counterfactual
from src.data_loader import load_dataset
from src.evaluate import evaluate_binary, get_calibration_data
from src.fairness import compute_fairness_metrics
from src.train_model import TrainedModel, load_model
from src.translator import summarize_for_instructor, summarize_shap_to_text

PATHS = Paths()
MODEL_NAMES = ["logreg", "rf", "xgb", "lgbm"]

# ── Global model registry (loaded once at startup) ───────────────────────────
_MODELS: Dict[str, TrainedModel] = {}
_X_TEST: Optional[pd.DataFrame] = None
_Y_TEST: Optional[pd.Series] = None
_DEMO_TEST: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data at startup."""
    global _MODELS, _X_TEST, _Y_TEST, _DEMO_TEST

    for name in MODEL_NAMES:
        path = PATHS.models_dir / f"{name}.joblib"
        if path.exists():
            _MODELS[name] = load_model(str(path))

    csv_path = PATHS.data_dir / "student_data.csv"
    if csv_path.exists():
        from sklearn.model_selection import train_test_split
        df_full = pd.read_csv(csv_path)
        train_idx, test_idx = train_test_split(
            range(len(df_full)), test_size=0.2, random_state=42,
            stratify=df_full[TARGET_COLUMN]
        )
        df_test = df_full.iloc[test_idx].reset_index(drop=True)
        _X_TEST = df_test[FEATURE_COLUMNS]
        _Y_TEST = df_test[TARGET_COLUMN]

        demo_cols = [c for c in ["gender", "ses_index", "first_gen"] if c in df_test.columns]
        if demo_cols:
            _DEMO_TEST = df_test[demo_cols]

    yield
    # Cleanup (none needed)


app = FastAPI(
    title="XAI-ED API",
    description=(
        "Explainable AI for Education — REST API for student performance "
        "prediction, SHAP/LIME explanations, counterfactuals, and fairness analysis."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    study_time_min: float = Field(..., ge=0, le=900, example=250.0)
    practice_completion_rate: float = Field(..., ge=0, le=1, example=0.72)
    avg_quiz_score: float = Field(..., ge=0, le=100, example=75.0)
    quiz_attempts: int = Field(..., ge=0, le=25, example=5)
    hint_usage_rate: float = Field(..., ge=0, le=1, example=0.3)
    attendance_rate: float = Field(..., ge=0, le=1, example=0.88)
    days_since_last_activity: float = Field(..., ge=0, le=30, example=2.0)
    stress_index: float = Field(..., ge=0, le=1, example=0.4)
    prereq_mastery: float = Field(..., ge=0, le=1, example=0.65)
    device_reliability: float = Field(..., ge=0, le=1, example=0.9)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class PredictRequest(BaseModel):
    model: str = Field("rf", example="rf", description="One of: logreg, rf, xgb, lgbm")
    features: StudentFeatures


class ExplainSHAPRequest(BaseModel):
    model: str = Field("rf", example="rf")
    features: StudentFeatures
    student_index: Optional[int] = Field(None, description="If provided, loads pre-computed SHAP from CSV")


class CounterfactualRequest(BaseModel):
    model: str = Field("rf", example="rf")
    features: StudentFeatures


class LIMERequest(BaseModel):
    model: str = Field("rf", example="rf")
    features: StudentFeatures
    num_samples: int = Field(300, ge=50, le=2000)


class FairnessRequest(BaseModel):
    model: str = Field("rf", example="rf")
    group_col: str = Field("gender", example="gender", description="One of: gender, ses_index, first_gen")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model(name: str) -> TrainedModel:
    if name not in _MODELS:
        available = list(_MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {available}. "
                   "Train the model first using train_model.py."
        )
    return _MODELS[name]


def _compute_shap(pipeline, X_background, x_explain, feature_names: list[str]) -> list[float]:
    """Quick SHAP computation for a single instance."""
    import shap

    preprocess = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]

    Xb = preprocess.transform(X_background[:200])
    Xe = preprocess.transform(x_explain)

    is_tree = clf.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}

    if is_tree:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xe)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        f = lambda z: clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, Xb)
        sv = explainer.shap_values(Xe, nsamples=100)

    return [float(np.ravel(v)[0]) for v in sv[0]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check and model list")
async def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "available_models": list(_MODELS.keys()),
        "feature_columns": FEATURE_COLUMNS,
        "version": "2.0.0",
    }


@app.get("/models/metrics", summary="Metrics for all available models on test set")
async def get_metrics() -> Dict[str, Any]:
    if _X_TEST is None:
        raise HTTPException(status_code=503, detail="Test dataset not loaded.")

    results = {}
    for name, trained in _MODELS.items():
        metrics = evaluate_binary(trained.pipeline, _X_TEST, _Y_TEST)
        results[name] = metrics
    return {"metrics": results}


@app.post("/predict", summary="Predict mastery probability for a student")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)
    X = req.features.to_dataframe()
    prob = float(trained.pipeline.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)

    return {
        "model": req.model,
        "prediction": "on-track" if pred else "at-risk",
        "mastery_probability": round(prob, 4),
        "features": req.features.model_dump(),
    }


@app.post("/explain/shap", summary="SHAP explanation for a student")
async def explain_shap(req: ExplainSHAPRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)

    # Try pre-computed CSV first
    if req.student_index is not None:
        csv_path = PATHS.explanations_dir / f"{req.model}_local_explanations.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if req.student_index < len(df):
                row = df.iloc[req.student_index]
                feats = row["top_features"].split("; ")
                contribs = [float(c) for c in row["top_contributions"].split("; ")]
                prob = float(trained.pipeline.predict_proba(req.features.to_dataframe())[:, 1][0])
                return {
                    "model": req.model,
                    "student_index": req.student_index,
                    "mastery_probability": round(prob, 4),
                    "top_features": feats,
                    "top_contributions": contribs,
                    "source": "precomputed_csv",
                }

    # Compute on the fly
    if _X_TEST is None:
        raise HTTPException(status_code=503, detail="Test dataset not loaded for background SHAP computation.")

    X = req.features.to_dataframe()
    prob = float(trained.pipeline.predict_proba(X)[:, 1][0])

    shap_vals = _compute_shap(
        trained.pipeline,
        np.array(_X_TEST),
        np.array(X),
        FEATURE_COLUMNS,
    )

    sorted_pairs = sorted(
        zip(FEATURE_COLUMNS, shap_vals),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    text = summarize_shap_to_text(
        FEATURE_COLUMNS,
        req.features.model_dump(),
        shap_vals,
        prob,
    )

    return {
        "model": req.model,
        "mastery_probability": round(prob, 4),
        "shap_values": {f: round(v, 6) for f, v in zip(FEATURE_COLUMNS, shap_vals)},
        "top_features": [f for f, _ in sorted_pairs[:5]],
        "top_contributions": [round(v, 6) for _, v in sorted_pairs[:5]],
        "student_explanation": text,
        "source": "computed",
    }


@app.post("/explain/counterfactual", summary="Counterfactual explanation — minimal edits to flip prediction")
async def explain_counterfactual(req: CounterfactualRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)
    x_row = req.features.to_dataframe().iloc[0]
    result = generate_counterfactual(trained.pipeline, x_row)
    return {
        "model": req.model,
        **result,
    }


@app.post("/explain/lime", summary="LIME local explanation for a student")
async def explain_lime(req: LIMERequest) -> Dict[str, Any]:
    trained = _get_model(req.model)

    if _X_TEST is None:
        raise HTTPException(status_code=503, detail="Test dataset not loaded for LIME background.")

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise HTTPException(status_code=503, detail="lime package not installed.")

    preprocess = trained.pipeline.named_steps["preprocess"]
    clf = trained.pipeline.named_steps["clf"]

    X_train_raw = np.array(_X_TEST)
    x_explain = np.array(req.features.to_dataframe())[0]

    def predict_fn(x_raw: np.ndarray) -> np.ndarray:
        return clf.predict_proba(preprocess.transform(x_raw))

    explainer = LimeTabularExplainer(
        training_data=X_train_raw,
        feature_names=FEATURE_COLUMNS,
        class_names=["at-risk", "on-track"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    exp = explainer.explain_instance(
        x_explain, predict_fn,
        num_features=len(FEATURE_COLUMNS),
        num_samples=req.num_samples,
        labels=(1,),
    )

    weights = dict(exp.as_list(label=1))
    feature_weight_map: dict[str, float] = {}
    for label, w in weights.items():
        for fn in FEATURE_COLUMNS:
            if fn in label:
                feature_weight_map[fn] = float(w)
                break

    sorted_lime = sorted(feature_weight_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    prob = float(trained.pipeline.predict_proba(req.features.to_dataframe())[:, 1][0])

    return {
        "model": req.model,
        "mastery_probability": round(prob, 4),
        "lime_weights": {f: round(v, 6) for f, v in feature_weight_map.items()},
        "top_features": [f for f, _ in sorted_lime[:5]],
        "top_weights": [round(v, 6) for _, v in sorted_lime[:5]],
    }


@app.post("/fairness", summary="Group-level fairness analysis")
async def fairness(req: FairnessRequest) -> Dict[str, Any]:
    if _X_TEST is None or _DEMO_TEST is None:
        raise HTTPException(
            status_code=503,
            detail="Test dataset with demographic columns not loaded. "
                   "Re-run data_gen.py to add demographic features."
        )

    if req.group_col not in _DEMO_TEST.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{req.group_col}' not found in demographic data. "
                   f"Available: {list(_DEMO_TEST.columns)}"
        )

    trained = _get_model(req.model)
    result = compute_fairness_metrics(
        trained.pipeline, _X_TEST, _Y_TEST, _DEMO_TEST, req.group_col
    )
    return {"model": req.model, **result}
