# Explainable AI for Educational Decision-Making (XAI-ED)

> A research-grade, end-to-end explainable AI pipeline for student performance prediction —
> with realistic equity gaps, multi-method XAI, statistical model comparison, interactive
> dashboard, REST API, reproducible notebooks, and a full test suite.

---

## Overview

XAI-ED bridges the gap between powerful machine learning and transparent educational
decision-making. Opaque "black-box" predictions raise ethical, pedagogical, and trust
concerns in education. This project embeds explainability at every layer of the system —
from model training and evaluation to student-facing language, instructor reports,
counterfactual recommendations, and group-level fairness auditing.

**Key research contributions:**

1. **Dual-Audience Explanations** — The same SHAP computation produces a student-friendly
   actionable message *and* a technical instructor report with risk tiers, SHAP feature
   tables, intervention urgency, and counterfactual summaries.
2. **Multi-Method XAI** — SHAP (TreeExplainer / KernelExplainer) and LIME are both
   computed for every model, enabling cross-method agreement analysis to measure
   explanation stability.
3. **Equity-Aware Dataset** — Demographic features (gender, SES, first-gen) are generated
   with realistic structural correlations (low-SES → less study time / attendance /
   device reliability; first-gen → lower prereq mastery) producing genuine fairness gaps
   for meaningful bias evaluation.
4. **Expanded Counterfactual Guidance** — Greedy hill-climbing over **7 actionable
   student behaviors** (practice completion, study time, attendance, quiz attempts,
   prereq mastery, hint dependence, stress) produces prescriptive "what must change"
   recommendations.
5. **Statistical Rigor** — 5-fold CV with mean ± std for 5 metrics, plus McNemar's test
   for pairwise model significance, and calibration reliability diagrams.
6. **SES Fairness Axis** — `ses_index` (continuous 0–1) is auto-binned into
   Low / Medium / High terciles for fairness analysis, alongside gender and first-gen.

---

## Project Structure

```
XAI-ED/
├── src/
│   ├── config.py           Paths, feature lists, demographic constants
│   ├── data_gen.py         Synthetic dataset generator (realistic equity gaps)
│   ├── data_loader.py      CSV loading with column validation
│   ├── train_model.py      4 sklearn Pipelines: LogReg, RF, XGBoost, LightGBM
│   ├── evaluate.py         7 metrics + 5-fold CV + McNemar test + calibration
│   ├── explain.py          SHAP global/local (handles all output formats)
│   ├── lime_explain.py     LIME global/local explanations
│   ├── counterfactual.py   Greedy hill-climbing over 7 controllable features
│   ├── translator.py       Student view + Instructor view text generation
│   └── fairness.py         Group-level fairness metrics + SES auto-binning
├── scripts/
│   └── run_all.py          Full pipeline: train 4 models + SHAP + LIME + CV + McNemar
├── notebooks/
│   ├── 01_data_exploration.ipynb    EDA, distributions, equity gap analysis
│   ├── 02_model_comparison.ipynb    Training, CV, ROC, calibration, McNemar
│   ├── 03_xai_deep_dive.ipynb       SHAP, LIME, agreement analysis, fairness
│   └── 04_case_studies.ipynb        End-to-end walkthroughs for 4 risk tiers
├── tests/
│   ├── conftest.py                  Shared pytest fixtures
│   ├── test_data_gen.py             Dataset shape, ranges, equity gap assertions
│   ├── test_evaluate.py             Metric correctness, CV structure, McNemar
│   ├── test_fairness.py             Group metrics, SES binning, known-gap detection
│   └── test_counterfactual.py       Flip logic, bounds, controllable features
├── data/
│   └── student_data.csv    3,000 synthetic student records with demographics
├── outputs/
│   ├── models/             Trained .joblib pipelines (4 models)
│   ├── metrics/            metrics.json (7 metrics + CV + McNemar per model)
│   └── explanations/       SHAP/LIME summary PNGs + local explanation CSVs
├── dashboard.py            Streamlit interactive dashboard (9 tabs)
├── app.py                  FastAPI REST API (8 endpoints)
└── requirements.txt
```

---

## Features

### Models

| Model | Type | Class Imbalance Handling |
|-------|------|--------------------------|
| Logistic Regression | Linear | `class_weight="balanced"` |
| Random Forest | Ensemble (300 trees) | `class_weight="balanced"` |
| XGBoost | Gradient Boosting | `scale_pos_weight` auto-computed |
| LightGBM | Gradient Boosting | `class_weight="balanced"` |

### Explainability

| Method | Scope | Model Types | Output |
|--------|-------|-------------|--------|
| SHAP TreeExplainer | Global + Local | RF, XGBoost, LightGBM | Beeswarm PNG, per-student CSV |
| SHAP KernelExplainer | Global + Local | Logistic Regression | Beeswarm PNG, per-student CSV |
| LIME | Global + Local | All (model-agnostic) | Importance bar chart PNG, per-student CSV |
| Counterfactual | Local | All | Minimal-edit table with steps and delta |

### Evaluation

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision / Recall / F1 | Classification performance |
| ROC-AUC | Discrimination ability |
| Brier Score | Probability calibration |
| Matthews Correlation Coefficient | Balanced metric for imbalanced classes |
| 5-fold CV (mean ± std) | Generalisation stability |
| McNemar's Test | Pairwise statistical significance (χ² continuity-corrected) |
| Calibration Curves | Reliability diagrams |

### Fairness Analysis

Group-level metrics across three demographic axes:

| Axis | Type | Groups |
|------|------|--------|
| Gender | Categorical | Female / Male / Non-binary |
| First-Generation Student | Binary | First-Gen / Continuing |
| SES Index | Continuous (auto-binned) | Low / Medium / High |

Computed metrics per group:
- **Equal Opportunity Gap** — max − min TPR across groups (threshold ≤ 0.10)
- **Demographic Parity Gap** — max − min positive prediction rate (threshold ≤ 0.10)
- **Disparate Impact Ratio** — min/max positive rate (80% rule: ≥ 0.80)

### Dashboard (9 Tabs)

| Tab | Contents |
|-----|----------|
| Home | Project abstract, architecture diagram, tech stack, quick metrics snapshot |
| Student Overview | Prediction badge, probability, feature table, student-friendly explanation |
| SHAP Explanations | Global SHAP + LIME side-by-side, local waterfall charts, feature agreement table |
| Counterfactual | Minimal edit table with current/suggested values, steps taken, delta probability |
| What-If Simulator | 10 live sliders with real-time prediction update and change bar chart |
| Model Comparison | Metrics table with CV results, McNemar table, ROC curves, calibration curves |
| Fairness Analysis | Gender / First-Gen / SES group metrics, pass/fail indicators, per-group bar chart |
| Research Metrics | CV stability table, SHAP–LIME agreement histogram, counterfactual flip-rate analysis |
| Instructor Report | Technical risk report + downloadable JSON |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (train + evaluate + explain)

```bash
python scripts/run_all.py
```

This will:
- Regenerate `data/student_data.csv` if it doesn't exist
- Train all 4 models and save to `outputs/models/`
- Evaluate with 7 metrics + 5-fold CV + McNemar tests
- Generate SHAP summary plots and local explanation CSVs for all 4 models
- Generate LIME summary plots and local explanation CSVs for all 4 models
- Save `outputs/metrics/metrics.json`

**Options:**
```bash
python scripts/run_all.py --skip-lime        # Skip LIME (faster, ~2 min vs ~10 min)
python scripts/run_all.py --models logreg,rf # Train specific models only
python scripts/run_all.py --regen-data       # Force-regenerate student_data.csv
```

### 3. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 4. Launch the API

```bash
uvicorn app:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### 5. Run the test suite

```bash
python -m pytest tests/ -v
```

Expected: **46 tests passing**.

### 6. Run the research notebooks

Open any notebook in `notebooks/` with Jupyter or VS Code. Run cells top-to-bottom.
Each notebook is self-contained and imports from `src/`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check, model list, endpoint index |
| GET | `/models/metrics` | All model metrics on test set |
| POST | `/predict` | Predict mastery probability + risk tier |
| POST | `/explain/shap` | SHAP explanation (pre-computed or live) |
| POST | `/explain/counterfactual` | Minimal edits to flip prediction |
| POST | `/explain/lime` | LIME local explanation |
| POST | `/fairness` | Group-level fairness analysis |
| POST | `/models/compare` | McNemar significance test between two models |

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

### Example: Compare Models (McNemar)

```bash
curl -X POST http://127.0.0.1:8000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"model_a": "rf", "model_b": "logreg"}'
```

---

## Dataset

3,000 synthetic student records — 10 academic features + 3 demographic features + 1 target.

### Academic Features (used in model training)

| Feature | Range | Distribution |
|---------|-------|--------------|
| `study_time_min` | 0–900 min/wk | Normal(220, 110), SES-adjusted |
| `practice_completion_rate` | 0–1 | Beta(2.2, 1.8) |
| `avg_quiz_score` | 0–100 | Normal(72, 14), SES-adjusted |
| `quiz_attempts` | 0–25 | Poisson(4.5), first-gen adjusted |
| `hint_usage_rate` | 0–1 | Beta(1.7, 3.0), first-gen adjusted |
| `attendance_rate` | 0–1 | Beta(4.0, 1.6), SES-adjusted |
| `days_since_last_activity` | 0–30 days | Gamma(2.2, 2.2), SES-adjusted |
| `stress_index` | 0–1 | Beta(2.0, 2.2), SES+first-gen adjusted |
| `prereq_mastery` | 0–1 | Beta(2.5, 2.0), first-gen adjusted |
| `device_reliability` | 0–1 | Beta(5.0, 1.8), SES-adjusted |

### Demographic Features (fairness analysis only — NOT used in training)

| Feature | Values | Equity Effect |
|---------|--------|---------------|
| `gender` | 0=female, 1=male, 2=non-binary | Independent (no feature adjustment) |
| `ses_index` | 0.0–1.0 continuous | Low SES: −55 min study, −7% attendance, −12% device reliability |
| `first_gen` | 0/1 (35% prevalence) | First-gen: −9% prereq mastery, fewer quiz attempts, higher hint usage |

### Target

`mastery` — 1 = on-track, 0 = at-risk. Generated via sigmoid of a linear combination
of academic features with a bias term calibrated to ~77–83% mastery rate. Low-SES and
first-generation students show materially lower mastery rates, enabling realistic
fairness gap analysis.

---

## Counterfactual Engine

The engine performs greedy hill-climbing over **7 controllable features**:

| Feature | Direction | Interpretation |
|---------|-----------|----------------|
| `practice_completion_rate` | Increase | Do more practice problems |
| `study_time_min` | Increase | Study longer each week |
| `attendance_rate` | Increase | Attend more sessions |
| `quiz_attempts` | Increase | Take more quizzes |
| `prereq_mastery` | Increase | Review prerequisite material |
| `hint_usage_rate` | Decrease | Attempt problems without hints first |
| `stress_index` | Decrease | Use stress management strategies |

At each step, the algorithm tries all 7 incremental changes and applies the one that
most increases the mastery probability, stopping when the prediction flips (p ≥ 0.5)
or the maximum step count is reached. The result includes `steps_taken`, `delta_prob`,
and the full edit history.

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_data_exploration.ipynb` | Feature distributions by class, correlation heatmap, equity gap bar charts, feature–mastery correlations |
| `02_model_comparison.ipynb` | Training all 4 models, test metrics, 5-fold CV, ROC + calibration curves, McNemar tests, learning curves |
| `03_xai_deep_dive.ipynb` | SHAP global/local, LIME comparison, SHAP–LIME agreement distribution, counterfactual flip rates, fairness across all 3 axes |
| `04_case_studies.ipynb` | End-to-end pipeline for 4 students (Strong / Borderline / At Risk / High Risk) — feature profiles, SHAP waterfalls, counterfactuals, instructor reports, radar chart |

---

## Test Suite

46 tests across 4 modules:

| File | Coverage |
|------|----------|
| `test_data_gen.py` | Shape, ranges, target balance, SES gap, first-gen gap, reproducibility |
| `test_evaluate.py` | Metric keys/ranges, perfect model, CV structure, McNemar keys/p-value, calibration |
| `test_fairness.py` | Binning, group metric keys, DI ratio range, SES auto-bin, known-gap detection |
| `test_counterfactual.py` | Controllable features, flip logic, bounds, delta consistency, step limits |

---

## Technology Stack

| Layer | Library |
|-------|---------|
| ML Models | scikit-learn, XGBoost ≥ 2.0, LightGBM ≥ 4.0 |
| XAI | SHAP ≥ 0.51, LIME ≥ 0.2 |
| Dashboard | Streamlit ≥ 1.35, Plotly ≥ 5.22 |
| API | FastAPI ≥ 0.111, Pydantic v2, uvicorn |
| Data | pandas ≥ 2.2, numpy ≥ 2.1 |
| Statistics | scipy ≥ 1.13 (McNemar, calibration) |
| Testing | pytest ≥ 9.0 |

---

## Future Work

- Integration with real LMS data (Canvas, Moodle export)
- Temporal / longitudinal modeling (student trajectories over time)
- DiCE diverse counterfactuals for richer what-if analysis
- LLM-powered natural language explanation generation
- Automated fairness mitigation (reweighting, post-processing threshold calibration)
- Learning path recommendations derived from counterfactual paths
- Multi-institution federated learning with differential privacy

---

*XAI-ED — Phase 3 Complete*
