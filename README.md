# Explainable AI for Educational Decision-Making (XAI-ED)

> A research-grade, end-to-end explainable AI pipeline for student performance prediction — with interactive dashboard, fairness analysis, REST API, and dual-audience explanations.

---

## Overview

XAI-ED bridges the gap between powerful machine learning and transparent educational decision-making. Opaque "black-box" predictions raise ethical, pedagogical, and trust concerns in education. This project embeds explainability directly into every layer of the system — from model selection and evaluation to student-facing language and instructor reports.

**Key contributions:**
- Four ML models with unified training API (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Two complementary XAI methods: **SHAP** + **LIME** — with comparison artifacts
- **Counterfactual explanations** — what a student would need to change to flip their prediction
- **Dual-audience explanations** — friendly student view + technical instructor view
- **Fairness & bias analysis** — group-level metrics across demographic subgroups
- **Interactive Streamlit dashboard** — 7-tab interface for exploration and demonstration
- **FastAPI REST API** — deployable service with auto-generated docs

---

## Architecture

```
XAI-ED
├── src/
│   ├── config.py           Configuration, paths, feature definitions
│   ├── data_gen.py         Synthetic student dataset generator (with demographics)
│   ├── data_loader.py      CSV loading with column validation
│   ├── train_model.py      Model training: LogReg, RF, XGBoost, LightGBM
│   ├── evaluate.py         Metrics: accuracy, F1, AUC, MCC, Brier + cross-validation + calibration
│   ├── explain.py          SHAP explanations (global + local)
│   ├── lime_explain.py     LIME explanations (global + local)
│   ├── counterfactual.py   Minimal-edit counterfactual engine
│   ├── translator.py       Student view + Instructor view text generation
│   └── fairness.py         Group-level fairness metrics
├── data/
│   └── student_data.csv    3,000 synthetic student records (with demographics)
├── outputs/
│   ├── models/             Trained model artifacts (.joblib)
│   ├── metrics/            Evaluation metrics (.json)
│   └── explanations/       SHAP/LIME plots, local explanation CSVs
├── dashboard.py            Streamlit interactive dashboard (7 tabs)
└── app.py                  FastAPI REST API
```

---

## Features

### Models
| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Interpretable | Baseline, StandardScaler |
| Random Forest | Ensemble | 300 trees, balanced classes |
| XGBoost | Gradient Boosting | scale_pos_weight auto-computed |
| LightGBM | Gradient Boosting | Fast, memory-efficient |

### Explainability
| Method | Scope | Output |
|--------|-------|--------|
| SHAP (TreeExplainer) | Global + Local | Summary plot, per-student CSV |
| SHAP (KernelExplainer) | Global + Local | For non-tree models |
| LIME | Global + Local | Summary bar chart, per-student CSV |
| Counterfactual | Local | Minimal edits to flip prediction |

### Evaluation (per model)
- Accuracy, Precision, Recall, F1 (weighted)
- ROC-AUC, Brier Score, Matthews Correlation Coefficient
- 5-fold Stratified Cross-Validation (mean +/- std)
- Calibration Curves (reliability diagram)

### Fairness Analysis
Computes group-level metrics across demographic subgroups:
- **Equal Opportunity** (TPR parity)
- **Demographic Parity** (positive prediction rate parity)
- **Predictive Parity** (precision parity)
- **Disparate Impact Ratio** (80% rule)

### Dashboard (7 Tabs)
1. **Student Overview** — prediction badge, probability, features, student explanation
2. **SHAP Explanations** — global summary + local waterfall bar chart + LIME comparison
3. **Counterfactual** — before/after table showing minimal changes needed
4. **What-If Simulator** — live sliders with real-time prediction update
5. **Model Comparison** — metrics table, ROC curves overlay, calibration curves
6. **Fairness Analysis** — per-group bar charts, disparate impact, parity gaps
7. **Instructor Report** — technical view + downloadable JSON

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python -m src.data_gen --out data/student_data.csv --n 3000
```

### 3. Train all models
```python
from src.data_loader import load_dataset
from src.train_model import train, save_model
from src.config import Paths
from sklearn.model_selection import train_test_split

PATHS = Paths()
X, y = load_dataset(PATHS.data_dir / "student_data.csv")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for model_name in ["logreg", "rf", "xgb", "lgbm"]:
    trained = train(model_name, X_train, y_train)
    save_model(trained, PATHS.models_dir / f"{model_name}.joblib")
    print(f"Saved {model_name}")
```

### 4. Launch the dashboard
```bash
streamlit run dashboard.py
```

### 5. Launch the API
```bash
uvicorn app:app --reload
```
API docs: http://127.0.0.1:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check, model list |
| GET | `/models/metrics` | All model metrics on test set |
| POST | `/predict` | Predict mastery probability |
| POST | `/explain/shap` | SHAP explanation |
| POST | `/explain/counterfactual` | Counterfactual (minimal edits) |
| POST | `/explain/lime` | LIME local explanation |
| POST | `/fairness` | Group-level fairness analysis |

### Example: Predict
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rf",
    "features": {
      "study_time_min": 250,
      "practice_completion_rate": 0.72,
      "avg_quiz_score": 75,
      "quiz_attempts": 5,
      "hint_usage_rate": 0.3,
      "attendance_rate": 0.88,
      "days_since_last_activity": 2,
      "stress_index": 0.4,
      "prereq_mastery": 0.65,
      "device_reliability": 0.9
    }
  }'
```

---

## Dataset

3,000 synthetic student records with:
- **10 academic features**: study time, quiz scores, practice completion, attendance, etc.
- **3 demographic features** (fairness analysis only, NOT used in model training):
  - `gender` (0=female, 1=male, 2=non-binary)
  - `ses_index` (socioeconomic status 0-1)
  - `first_gen` (first-generation student: 0/1)
- **1 target**: `mastery` (1=on-track, 0=at-risk, ~82% base rate)

---

## Research Contributions

1. **Dual XAI Methods**: SHAP and LIME explanations for every model enable comparison of explanation fidelity and stability.
2. **Dual Audience Design**: Student-friendly explanations use approachable language and actionable suggestions; instructor reports provide technical detail, risk tiers, and intervention guidance.
3. **Controllable Counterfactuals**: The counterfactual engine only modifies features students can realistically change (practice completion, study time, attendance).
4. **Fairness-First Evaluation**: All models evaluated on demographic subgroup fairness, addressing a critical gap in educational AI research.
5. **Production-Ready API**: Full REST API with Pydantic validation enables integration with LMS platforms.

---

## Future Work
- Integration with real LMS data (Canvas, Moodle)
- Temporal / longitudinal modeling (student trajectories over time)
- DiCE diverse counterfactuals for richer what-if analysis
- LLM-powered natural language explanation generation
- Automated fairness mitigation (reweighting, post-processing)
- Learning path recommendations based on counterfactual analysis

---

*XAI-ED — Phase 2 Complete*
