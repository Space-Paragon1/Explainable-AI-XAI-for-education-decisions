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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEMOGRAPHIC_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, Paths
from src.counterfactual import generate_counterfactual
from src.data_loader import load_dataset
from src.evaluate import evaluate_binary, get_calibration_data, compare_models_mcnemar
from src.fairness import compute_fairness_metrics
from src.train_model import TrainedModel, load_model
from src.translator import summarize_for_instructor, summarize_shap_to_text

PATHS       = Paths()
MODEL_NAMES = ["logreg", "rf", "xgb", "lgbm"]

# ── Global registry (loaded once at startup) ──────────────────────────────────
_MODELS:          Dict[str, TrainedModel] = {}
_X_TEST:          Optional[pd.DataFrame]  = None
_Y_TEST:          Optional[pd.Series]     = None
_DEMO_TEST:       Optional[pd.DataFrame]  = None
# Pre-computed transformed backgrounds for fast SHAP — keyed by model name
_SHAP_BACKGROUND: Dict[str, Any]          = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models, data, and pre-compute SHAP backgrounds at startup."""
    global _MODELS, _X_TEST, _Y_TEST, _DEMO_TEST, _SHAP_BACKGROUND

    # Load all available trained models
    import joblib
    for name in MODEL_NAMES:
        path = PATHS.models_dir / f"{name}.joblib"
        if path.exists():
            obj = joblib.load(str(path))
            # Handle both TrainedModel wrapper and raw Pipeline (legacy saves)
            if isinstance(obj, TrainedModel):
                _MODELS[name] = obj
            else:
                _MODELS[name] = TrainedModel(name=name, pipeline=obj)

    # Load test split (same split as training: 80/20, random_state=42)
    csv_path = PATHS.data_dir / "student_data.csv"
    if csv_path.exists():
        from sklearn.model_selection import train_test_split
        df_full = pd.read_csv(csv_path)
        _, test_idx = train_test_split(
            range(len(df_full)), test_size=0.2, random_state=42,
            stratify=df_full[TARGET_COLUMN],
        )
        df_test   = df_full.iloc[test_idx].reset_index(drop=True)
        _X_TEST   = df_test[FEATURE_COLUMNS]
        _Y_TEST   = df_test[TARGET_COLUMN]

        demo_cols = [c for c in DEMOGRAPHIC_COLUMNS if c in df_test.columns]
        if demo_cols:
            _DEMO_TEST = df_test[demo_cols]

        # Pre-compute transformed background arrays for each model (200 samples)
        # Avoids re-transforming on every API call for KernelExplainer
        bg_raw = _X_TEST.sample(n=min(200, len(_X_TEST)), random_state=42)
        for name, trained in _MODELS.items():
            try:
                preprocess = trained.pipeline.named_steps["preprocess"]
                _SHAP_BACKGROUND[name] = preprocess.transform(bg_raw)
            except Exception:
                _SHAP_BACKGROUND[name] = np.array(bg_raw)

    yield  # Application runs here


app = FastAPI(
    title="XAI-ED API",
    description=(
        "Explainable AI for Education — REST API for student performance "
        "prediction, SHAP/LIME explanations, counterfactuals, and fairness analysis."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    study_time_min:            float = Field(..., ge=0,   le=900,  example=250.0)
    practice_completion_rate:  float = Field(..., ge=0,   le=1,    example=0.72)
    avg_quiz_score:            float = Field(..., ge=0,   le=100,  example=75.0)
    quiz_attempts:             int   = Field(..., ge=0,   le=25,   example=5)
    hint_usage_rate:           float = Field(..., ge=0,   le=1,    example=0.3)
    attendance_rate:           float = Field(..., ge=0,   le=1,    example=0.88)
    days_since_last_activity:  float = Field(..., ge=0,   le=30,   example=2.0)
    stress_index:              float = Field(..., ge=0,   le=1,    example=0.4)
    prereq_mastery:            float = Field(..., ge=0,   le=1,    example=0.65)
    device_reliability:        float = Field(..., ge=0,   le=1,    example=0.9)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class PredictRequest(BaseModel):
    model:    str = Field("rf", example="rf", description="One of: logreg, rf, xgb, lgbm")
    features: StudentFeatures


class ExplainSHAPRequest(BaseModel):
    model:         str          = Field("rf", example="rf")
    features:      StudentFeatures
    student_index: Optional[int] = Field(
        None, description="If provided, loads pre-computed SHAP from CSV (faster)."
    )


class CounterfactualRequest(BaseModel):
    model:    str = Field("rf", example="rf")
    features: StudentFeatures


class LIMERequest(BaseModel):
    model:       str = Field("rf", example="rf")
    features:    StudentFeatures
    num_samples: int = Field(300, ge=50, le=2000)


class FairnessRequest(BaseModel):
    model:     str = Field("rf",     example="rf")
    group_col: str = Field("gender", example="gender",
                           description="One of: gender, ses_index, first_gen")


class CompareRequest(BaseModel):
    model_a: str = Field("rf",     example="rf")
    model_b: str = Field("logreg", example="logreg")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model(name: str) -> TrainedModel:
    if name not in _MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {list(_MODELS.keys())}. "
                   "Train the model first using scripts/run_all.py.",
        )
    return _MODELS[name]


def _validate_features(features: StudentFeatures) -> pd.DataFrame:
    """Validate and return feature DataFrame, checking for NaN/Inf."""
    df = features.to_dataframe()
    if df.isnull().any(axis=None).any() or np.isinf(df.values).any():
        raise HTTPException(
            status_code=422,
            detail="Input features contain NaN or Inf values.",
        )
    return df


def _compute_shap_live(
    pipeline, model_name: str, x_explain: np.ndarray
) -> list[float]:
    """
    Fast SHAP computation using pre-computed background.
    TreeExplainer for RF/XGB/LGBM; KernelExplainer for LogReg.
    """
    import shap
    from src.explain import _extract_class1_shap

    preprocess = pipeline.named_steps["preprocess"]
    clf        = pipeline.named_steps["clf"]
    Xe         = preprocess.transform(x_explain)

    is_tree = clf.__class__.__name__ in {
        "RandomForestClassifier", "XGBClassifier", "LGBMClassifier"
    }

    if is_tree:
        from src.explain import _fix_xgb_base_score
        if clf.__class__.__name__ == "XGBClassifier":
            _fix_xgb_base_score(clf)
        explainer = shap.TreeExplainer(clf)
        sv_raw    = explainer.shap_values(np.ascontiguousarray(Xe, dtype=np.float64))
        sv        = _extract_class1_shap(sv_raw)
    else:
        Xb = _SHAP_BACKGROUND.get(model_name)
        if Xb is None:
            raise HTTPException(
                status_code=503,
                detail="SHAP background not available. Ensure test data is loaded.",
            )
        f         = lambda z: clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, Xb)
        sv_raw    = explainer.shap_values(Xe, nsamples=100)
        sv        = _extract_class1_shap(sv_raw)

    sv = np.array(sv, dtype=float)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return [float(np.array(sv[0, j]).ravel()[0]) for j in range(len(FEATURE_COLUMNS))]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check and model list")
async def root() -> Dict[str, Any]:
    return {
        "status":           "ok",
        "version":          "2.1.0",
        "available_models": list(_MODELS.keys()),
        "feature_columns":  FEATURE_COLUMNS,
        "endpoints": [
            "GET  /",
            "GET  /models/metrics",
            "POST /predict",
            "POST /explain/shap",
            "POST /explain/counterfactual",
            "POST /explain/lime",
            "POST /fairness",
            "POST /models/compare",
        ],
    }


@app.get("/models/metrics", summary="Metrics for all available models on test set")
async def get_metrics() -> Dict[str, Any]:
    if _X_TEST is None:
        raise HTTPException(status_code=503, detail="Test dataset not loaded.")
    results = {}
    for name, trained in _MODELS.items():
        results[name] = evaluate_binary(trained.pipeline, _X_TEST, _Y_TEST)
    return {"metrics": results}


@app.post("/predict", summary="Predict mastery probability for a student")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)
    X       = _validate_features(req.features)
    prob    = float(trained.pipeline.predict_proba(X)[:, 1][0])
    pred    = int(prob >= 0.5)

    from src.translator import _risk_tier
    return {
        "model":               req.model,
        "prediction":          "on-track" if pred else "at-risk",
        "mastery_probability": round(prob, 4),
        "risk_tier":           _risk_tier(prob),
        "features":            req.features.model_dump(),
    }


@app.post("/explain/shap", summary="SHAP explanation for a student")
async def explain_shap(req: ExplainSHAPRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)

    # Fast path: serve from pre-computed CSV
    if req.student_index is not None:
        csv_path = PATHS.explanations_dir / f"{req.model}_local_explanations.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if req.student_index < len(df):
                row      = df.iloc[req.student_index]
                feats    = row["top_features"].split("; ")
                contribs = [float(c) for c in row["top_contributions"].split("; ")]
                prob     = float(
                    trained.pipeline.predict_proba(
                        _validate_features(req.features)
                    )[:, 1][0]
                )
                text = summarize_shap_to_text(
                    FEATURE_COLUMNS,
                    req.features.model_dump(),
                    dict(zip(feats, contribs)),
                    prob,
                )
                return {
                    "model":               req.model,
                    "student_index":       req.student_index,
                    "mastery_probability": round(prob, 4),
                    "top_features":        feats,
                    "top_contributions":   contribs,
                    "student_explanation": text,
                    "source":              "precomputed_csv",
                }

    # Live SHAP computation
    if _X_TEST is None:
        raise HTTPException(
            status_code=503,
            detail="Test dataset not loaded for background SHAP computation.",
        )
    X    = _validate_features(req.features)
    prob = float(trained.pipeline.predict_proba(X)[:, 1][0])

    shap_vals = _compute_shap_live(trained.pipeline, req.model, np.array(X))
    sorted_pairs = sorted(
        zip(FEATURE_COLUMNS, shap_vals), key=lambda kv: abs(kv[1]), reverse=True
    )
    text = summarize_shap_to_text(
        FEATURE_COLUMNS, req.features.model_dump(), shap_vals, prob
    )

    return {
        "model":               req.model,
        "mastery_probability": round(prob, 4),
        "shap_values":         {f: round(v, 6) for f, v in zip(FEATURE_COLUMNS, shap_vals)},
        "top_features":        [f for f, _ in sorted_pairs[:5]],
        "top_contributions":   [round(v, 6) for _, v in sorted_pairs[:5]],
        "student_explanation": text,
        "source":              "computed",
    }


@app.post("/explain/counterfactual",
          summary="Counterfactual — minimal edits to flip prediction")
async def explain_counterfactual(req: CounterfactualRequest) -> Dict[str, Any]:
    trained = _get_model(req.model)
    x_row   = _validate_features(req.features).iloc[0]
    result  = generate_counterfactual(trained.pipeline, x_row)
    return {"model": req.model, **result}


@app.post("/explain/lime", summary="LIME local explanation for a student")
async def explain_lime(req: LIMERequest) -> Dict[str, Any]:
    trained = _get_model(req.model)

    if _X_TEST is None:
        raise HTTPException(
            status_code=503, detail="Test dataset not loaded for LIME background."
        )

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise HTTPException(status_code=503, detail="lime package not installed.")

    preprocess = trained.pipeline.named_steps["preprocess"]
    clf        = trained.pipeline.named_steps["clf"]

    X_train_raw = np.array(_X_TEST)
    x_explain   = np.array(_validate_features(req.features))[0]

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
    exp     = explainer.explain_instance(
        x_explain, predict_fn,
        num_features=len(FEATURE_COLUMNS),
        num_samples=req.num_samples,
        labels=(1,),
    )
    weights: dict[str, float] = {}
    for label, w in dict(exp.as_list(label=1)).items():
        for fn in FEATURE_COLUMNS:
            if fn in label:
                weights[fn] = float(w)
                break

    sorted_lime = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
    prob        = float(trained.pipeline.predict_proba(
        _validate_features(req.features)
    )[:, 1][0])

    return {
        "model":               req.model,
        "mastery_probability": round(prob, 4),
        "lime_weights":        {f: round(v, 6) for f, v in weights.items()},
        "top_features":        [f for f, _ in sorted_lime[:5]],
        "top_weights":         [round(v, 6) for _, v in sorted_lime[:5]],
    }


@app.post("/fairness", summary="Group-level fairness analysis")
async def fairness(req: FairnessRequest) -> Dict[str, Any]:
    if _X_TEST is None or _DEMO_TEST is None:
        raise HTTPException(
            status_code=503,
            detail="Test dataset with demographic columns not loaded.",
        )
    if req.group_col not in _DEMO_TEST.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{req.group_col}' not found. "
                   f"Available: {list(_DEMO_TEST.columns)}",
        )
    trained = _get_model(req.model)
    result  = compute_fairness_metrics(
        trained.pipeline, _X_TEST, _Y_TEST, _DEMO_TEST, req.group_col
    )
    return {"model": req.model, **result}


@app.post("/models/compare",
          summary="McNemar significance test between two models")
async def compare_models(req: CompareRequest) -> Dict[str, Any]:
    if _X_TEST is None:
        raise HTTPException(status_code=503, detail="Test dataset not loaded.")
    trained_a = _get_model(req.model_a)
    trained_b = _get_model(req.model_b)
    result    = compare_models_mcnemar(
        trained_a.pipeline, trained_b.pipeline, _X_TEST, _Y_TEST
    )
    return {
        "model_a": req.model_a,
        "model_b": req.model_b,
        **result,
    }
