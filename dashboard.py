"""
dashboard.py — Interactive Streamlit dashboard for XAI-ED.

Run with:
    streamlit run dashboard.py

Tabs:
  1. Home               — project overview, architecture, quick stats
  2. Student Overview   — prediction badge, probability, feature table, explanation
  3. SHAP Explanations  — global summary + local waterfall + SHAP vs LIME side-by-side
  4. Counterfactual     — minimal edits to flip at-risk → on-track
  5. What-If Simulator  — live sliders with real-time prediction update
  6. Model Comparison   — metrics table, CV results, ROC curves, calibration curves
  7. Fairness Analysis  — group-level bias metrics with pass/fail indicators
  8. Research Metrics   — SHAP stability, LIME vs SHAP agreement, flip-rate analysis
  9. Instructor Report  — technical view + downloadable JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEMOGRAPHIC_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, Paths
from src.counterfactual import generate_counterfactual
from src.data_loader import load_dataset
from src.evaluate import evaluate_binary, get_calibration_data
from src.fairness import compute_fairness_metrics
from src.train_model import TrainedModel, load_model
from src.translator import summarize_for_instructor, summarize_shap_to_text

PATHS = Paths()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-ED Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .risk-badge-on  { background:#1b5e20; color:white; border-radius:8px;
                      padding:6px 18px; font-size:1.1rem; font-weight:700; }
    .risk-badge-off { background:#b71c1c; color:white; border-radius:8px;
                      padding:6px 18px; font-size:1.1rem; font-weight:700; }
    .metric-card    { background:#1e1e2e; border-radius:10px; padding:14px; margin:6px; }
    .section-header { font-size:1.15rem; font-weight:600; margin-top:1rem; }
    .home-chip      { display:inline-block; background:#1e3a5f; color:#90caf9;
                      border-radius:20px; padding:4px 14px; margin:3px;
                      font-size:0.85rem; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FRIENDLY = {
    "study_time_min":            "Study Time (min/wk)",
    "practice_completion_rate":  "Practice Completion",
    "avg_quiz_score":            "Avg Quiz Score",
    "quiz_attempts":             "Quiz Attempts",
    "hint_usage_rate":           "Hint Usage Rate",
    "attendance_rate":           "Attendance Rate",
    "days_since_last_activity":  "Days Since Last Activity",
    "stress_index":              "Stress Index",
    "prereq_mastery":            "Prereq Mastery",
    "device_reliability":        "Device Reliability",
}

MODEL_NAMES  = ["logreg", "rf", "xgb", "lgbm"]
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf":     "Random Forest",
    "xgb":    "XGBoost",
    "lgbm":   "LightGBM",
}

FEATURE_RANGES = {
    "study_time_min":            (0.0,  900.0, 1.0),
    "practice_completion_rate":  (0.0,    1.0, 0.01),
    "avg_quiz_score":            (0.0,  100.0, 0.5),
    "quiz_attempts":             (0.0,   25.0, 1.0),
    "hint_usage_rate":           (0.0,    1.0, 0.01),
    "attendance_rate":           (0.0,    1.0, 0.01),
    "days_since_last_activity":  (0.0,   30.0, 0.5),
    "stress_index":              (0.0,    1.0, 0.01),
    "prereq_mastery":            (0.0,    1.0, 0.01),
    "device_reliability":        (0.0,    1.0, 0.01),
}

DARK_LAYOUT = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_all_models() -> dict[str, TrainedModel]:
    import joblib
    models = {}
    for name in MODEL_NAMES:
        path = PATHS.models_dir / f"{name}.joblib"
        if path.exists():
            obj = joblib.load(str(path))
            # Handle both TrainedModel wrapper and raw Pipeline (legacy saves)
            if isinstance(obj, TrainedModel):
                models[name] = obj
            else:
                models[name] = TrainedModel(name=name, pipeline=obj)
    return models


@st.cache_data
def load_data():
    csv_path = PATHS.data_dir / "student_data.csv"
    df_full  = pd.read_csv(csv_path)

    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(
        range(len(df_full)), test_size=0.2, random_state=42,
        stratify=df_full[TARGET_COLUMN],
    )
    df_test  = df_full.iloc[test_idx].reset_index(drop=True)
    X_test   = df_test[FEATURE_COLUMNS]
    y_test   = df_test[TARGET_COLUMN]
    demo_test = (
        df_test[DEMOGRAPHIC_COLUMNS]
        if all(c in df_test.columns for c in DEMOGRAPHIC_COLUMNS)
        else None
    )
    return df_test, X_test, y_test, demo_test


@st.cache_data
def load_metrics():
    path = PATHS.metrics_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@st.cache_data
def get_shap_csv(model_name: str):
    csv_path = PATHS.explanations_dir / f"{model_name}_local_explanations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def get_lime_csv(model_name: str):
    csv_path = PATHS.explanations_dir / f"{model_name}_lime_local_explanations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ── Chart helpers ─────────────────────────────────────────────────────────────

def shap_bar_chart(feature_names, shap_vals, title="Local SHAP Contributions"):
    colors = ["#ef5350" if v < 0 else "#42a5f5" for v in shap_vals]
    fig = go.Figure(
        go.Bar(
            x=shap_vals, y=feature_names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (impact on mastery probability)",
        height=380,
        margin=dict(l=180, r=30, t=40, b=30),
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


def lime_bar_chart(feature_names, lime_vals, title="Local LIME Contributions"):
    colors = ["#ef5350" if v < 0 else "#66bb6a" for v in lime_vals]
    fig = go.Figure(
        go.Bar(
            x=lime_vals, y=feature_names, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in lime_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="LIME Weight (impact on mastery probability)",
        height=380,
        margin=dict(l=180, r=30, t=40, b=30),
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


def _parse_shap_row(shap_df, student_idx):
    """Parse a SHAP CSV row into (feats, contribs, full_shap_list)."""
    if shap_df is None or student_idx >= len(shap_df):
        return None, None, None
    row    = shap_df.iloc[student_idx]
    feats  = row["top_features"].split("; ")
    contribs = [float(c) for c in row["top_contributions"].split("; ")]
    full_shap = [0.0] * len(FEATURE_COLUMNS)
    for feat, contrib in zip(feats, contribs):
        if feat in FEATURE_COLUMNS:
            full_shap[FEATURE_COLUMNS.index(feat)] = contrib
    return feats, contribs, full_shap


def _parse_lime_row(lime_df, student_idx):
    """Parse a LIME CSV row into (feats, weights, full_lime_list)."""
    if lime_df is None or student_idx >= len(lime_df):
        return None, None, None
    row    = lime_df.iloc[student_idx]
    feats  = row["top_features"].split("; ")
    weights = [float(w) for w in row["top_lime_weights"].split("; ")]
    full_lime = [0.0] * len(FEATURE_COLUMNS)
    for feat, w in zip(feats, weights):
        if feat in FEATURE_COLUMNS:
            full_lime[FEATURE_COLUMNS.index(feat)] = w
    return feats, weights, full_lime


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 XAI-ED Dashboard")
    st.markdown("**Explainable AI for Education**")
    st.markdown("---")

    all_models = load_all_models()
    available  = [m for m in MODEL_NAMES if m in all_models]

    if not available:
        st.error(
            "No trained models found in `outputs/models/`.\n\n"
            "Run `python scripts/run_all.py` first."
        )
        st.stop()

    selected_model_name = st.selectbox(
        "Select Model",
        options=available,
        format_func=lambda n: MODEL_LABELS.get(n, n),
    )
    trained        = all_models[selected_model_name]
    model_pipeline = trained.pipeline

    df_test, X_test, y_test, demo_test = load_data()
    n_students = len(df_test)

    student_idx = st.slider(
        "Student ID (test set)",
        min_value=0, max_value=n_students - 1, value=0, step=1,
    )

    st.markdown("---")
    st.markdown(f"**Test set size:** {n_students} students")
    st.markdown(f"**Model:** {MODEL_LABELS.get(selected_model_name, selected_model_name)}")

# ── Current student ───────────────────────────────────────────────────────────
student_row  = X_test.iloc[student_idx]
student_true = int(y_test.iloc[student_idx])
student_prob = float(model_pipeline.predict_proba(pd.DataFrame([student_row]))[:, 1][0])
student_pred = int(student_prob >= 0.5)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Home",
    "👤 Student Overview",
    "🔍 SHAP Explanations",
    "🔄 Counterfactual",
    "🎛️ What-If Simulator",
    "📊 Model Comparison",
    "⚖️ Fairness Analysis",
    "🔬 Research Metrics",
    "📋 Instructor Report",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — Home
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("🎓 XAI-ED: Explainable AI for Educational Decision-Making")
    st.markdown(
        """
        **XAI-ED** is a research-grade, end-to-end pipeline that predicts student academic
        mastery and provides transparent, human-understandable explanations at every level —
        from global feature importance to individual student counterfactuals.
        """
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Students in Dataset", "3,000")
    with col_b:
        st.metric("XAI Methods", "SHAP + LIME + Counterfactual")
    with col_c:
        st.metric("Models Trained", str(len(available)))
    with col_d:
        st.metric("Fairness Axes", "Gender · SES · First-Gen")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Research Contributions")
        st.markdown(
            """
            1. **Dual-Audience Explanations** — The same SHAP computation generates
               student-friendly actionable text *and* a technical instructor report with
               risk tiers, intervention urgency, and counterfactual summaries.

            2. **Multi-Method XAI** — SHAP (TreeExplainer / KernelExplainer) and LIME
               are both computed, enabling cross-method agreement analysis to measure
               explanation stability.

            3. **Equity-Aware Evaluation** — Demographic features (gender, SES, first-gen)
               are generated with realistic equity gaps and evaluated with Disparate Impact
               Ratio, Equal Opportunity Gap, and Demographic Parity Gap.

            4. **Counterfactual Guidance** — Greedy hill-climbing over 7 controllable
               student behaviors produces actionable "what must change" recommendations,
               going beyond prediction to prescription.
            """
        )

    with col_right:
        st.subheader("System Architecture")
        st.markdown(
            """
            ```
            ┌─────────────────────────────────────┐
            │          data_gen.py                │
            │  Synthetic dataset (3,000 students) │
            │  with realistic equity gaps         │
            └──────────────┬──────────────────────┘
                           │
            ┌──────────────▼──────────────────────┐
            │         train_model.py              │
            │  4 Pipelines: LogReg · RF · XGB     │
            │              · LightGBM             │
            └──────────────┬──────────────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
            ┌──▼──────────┐   ┌────────▼────────┐
            │  explain.py │   │ lime_explain.py │
            │  SHAP       │   │ LIME            │
            └──┬──────────┘   └────────┬────────┘
               │                       │
            ┌──▼───────────────────────▼────────┐
            │         translator.py             │
            │  Student view  |  Instructor view │
            └──────────────┬────────────────────┘
                           │
            ┌──────────────▼──────────────────────┐
            │  counterfactual.py + fairness.py    │
            │  Prescriptive advice + Bias audit   │
            └──────────────┬──────────────────────┘
                           │
               ┌───────────┴──────────┐
            ┌──▼─────────┐   ┌────────▼──────┐
            │ dashboard  │   │   app.py      │
            │ (Streamlit)│   │   (FastAPI)   │
            └────────────┘   └───────────────┘
            ```
            """
        )

    st.markdown("---")
    st.subheader("Technology Stack")
    techs = [
        "Python 3", "scikit-learn", "XGBoost", "LightGBM",
        "SHAP 0.46", "LIME", "Streamlit", "FastAPI", "Plotly", "pandas", "numpy",
    ]
    chips = "".join(f'<span class="home-chip">{t}</span>' for t in techs)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Model Snapshot")
    saved = load_metrics()
    if saved:
        rows = []
        for mname in MODEL_NAMES:
            if mname in saved and "metrics" in saved[mname]:
                m = saved[mname]["metrics"]
                cv = saved[mname].get("cross_validation", {})
                auc_str = (
                    f"{m.get('roc_auc', 0):.3f}"
                    + (f" ± {cv['roc_auc']['std']:.3f}" if "roc_auc" in cv else "")
                )
                rows.append({
                    "Model":   MODEL_LABELS.get(mname, mname),
                    "ROC-AUC": auc_str,
                    "F1":      f"{m.get('f1', 0):.3f}",
                    "MCC":     f"{m.get('mcc', 0):.3f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python scripts/run_all.py` to generate model metrics.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Student Overview
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader(f"Student #{student_idx} — Overview")

    col_badge, col_prob, col_true = st.columns(3)
    with col_badge:
        if student_pred == 1:
            st.markdown('<span class="risk-badge-on">ON TRACK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-badge-off">AT RISK</span>', unsafe_allow_html=True)
    with col_prob:
        st.metric("Mastery Probability", f"{student_prob:.3f}")
    with col_true:
        st.metric("True Label", "On Track" if student_true == 1 else "At Risk")

    st.markdown("---")
    col_table, col_text = st.columns([1, 1])

    with col_table:
        st.markdown("**Feature Values**")
        feat_df = pd.DataFrame({
            "Feature": [FRIENDLY.get(f, f) for f in FEATURE_COLUMNS],
            "Value":   [round(float(student_row[f]), 3) for f in FEATURE_COLUMNS],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_text:
        st.markdown("**Student-Friendly Explanation**")
        shap_df = get_shap_csv(selected_model_name)
        feats_s, contribs_s, full_shap = _parse_shap_row(shap_df, student_idx)

        if full_shap is not None:
            explanation = summarize_shap_to_text(
                FEATURE_COLUMNS, student_row.to_dict(), full_shap, student_prob,
            )
            st.info(explanation)
        else:
            st.info(
                f"Prediction: {'**on track**' if student_pred else '**at risk**'} "
                f"(mastery probability ≈ {student_prob:.2f}). "
                "Load SHAP explanations to see detailed feature contributions."
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP Explanations
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("SHAP & LIME Explanations")

    # ── Global plots side by side ────────────────────────────────────────────
    st.markdown("#### Global Feature Importance")
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.markdown("**SHAP Summary (all test students)**")
        summary_png = PATHS.explanations_dir / f"{selected_model_name}_shap_summary.png"
        if summary_png.exists():
            st.image(str(summary_png), use_container_width=True)
        else:
            st.warning(f"No SHAP summary for **{selected_model_name}**. Run `scripts/run_all.py`.")

    with gcol2:
        st.markdown("**LIME Global Feature Importance**")
        lime_png = PATHS.explanations_dir / f"{selected_model_name}_lime_summary.png"
        if lime_png.exists():
            st.image(str(lime_png), use_container_width=True)
        else:
            st.info("LIME summary not yet generated. Run `scripts/run_all.py`.")

    # ── Local explanations side by side ─────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Local Explanations — Student #{student_idx}")
    lcol1, lcol2 = st.columns(2)

    shap_df = get_shap_csv(selected_model_name)
    lime_df = get_lime_csv(selected_model_name)
    feats_s, contribs_s, full_shap = _parse_shap_row(shap_df, student_idx)
    feats_l, weights_l, full_lime  = _parse_lime_row(lime_df, student_idx)

    with lcol1:
        if feats_s is not None:
            friendly_feats = [FRIENDLY.get(f, f) for f in feats_s]
            fig = shap_bar_chart(
                friendly_feats, contribs_s,
                title=f"SHAP — Student #{student_idx}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No local SHAP data for this student. Run `scripts/run_all.py`.")

    with lcol2:
        if feats_l is not None:
            friendly_feats_l = [FRIENDLY.get(f, f) for f in feats_l]
            fig = lime_bar_chart(
                friendly_feats_l, weights_l,
                title=f"LIME — Student #{student_idx}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LIME data for this student. Run `scripts/run_all.py`.")

    # ── Feature agreement table ──────────────────────────────────────────────
    if feats_s is not None and feats_l is not None:
        st.markdown("---")
        st.markdown("#### SHAP vs LIME — Top Feature Agreement")
        shap_set = set(feats_s[:5])
        lime_set = set(feats_l[:5])
        overlap  = shap_set & lime_set
        agree_pct = len(overlap) / 5 * 100
        st.metric("Top-5 Feature Overlap", f"{len(overlap)}/5 ({agree_pct:.0f}%)")
        agree_df = pd.DataFrame({
            "SHAP Top 5": [FRIENDLY.get(f, f) for f in feats_s[:5]],
            "LIME Top 5": [FRIENDLY.get(f, f) for f in feats_l[:5]],
            "Agrees?": [
                "✅" if feats_s[i] in lime_set else "❌"
                for i in range(min(5, len(feats_s)))
            ],
        })
        st.dataframe(agree_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Counterfactual
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Counterfactual Explanation")
    st.markdown(
        "Shows the **minimal changes** across 7 actionable behaviors "
        "needed to flip an *at-risk* prediction to *on-track*."
    )

    with st.spinner("Computing counterfactual..."):
        cf = generate_counterfactual(model_pipeline, student_row.copy())

    status   = cf["status"]
    base_prob = cf["base_prob"]
    new_prob  = cf["new_prob"]
    edits     = cf.get("edits", {})

    col_status, col_before, col_after, col_steps = st.columns(4)
    with col_status:
        if status == "already_on_track":
            st.success("Already **ON TRACK**. No changes needed.")
        elif status == "flipped":
            st.success(f"Prediction flipped! New probability: **{new_prob:.3f}**")
        else:
            st.warning(f"Could not fully flip. Best achieved: **{new_prob:.3f}**")
    with col_before:
        st.metric("Original Probability", f"{base_prob:.3f}")
    with col_after:
        st.metric("After Changes", f"{new_prob:.3f}", delta=f"{new_prob - base_prob:+.3f}")
    with col_steps:
        st.metric("Steps Taken", cf.get("steps_taken", "—"))

    if edits:
        st.markdown("**Required Changes:**")
        edit_rows = []
        for feat, new_val in edits.items():
            old_val = float(student_row.get(feat, 0))
            edit_rows.append({
                "Feature":        FRIENDLY.get(feat, feat),
                "Current Value":  round(old_val, 3),
                "Suggested Value": round(new_val, 3),
                "Change":         f"{new_val - old_val:+.3f}",
                "Direction":      "↑ Increase" if new_val > old_val else "↓ Decrease",
            })
        st.dataframe(pd.DataFrame(edit_rows), use_container_width=True, hide_index=True)
    else:
        if status not in ("already_on_track",):
            st.info("No edits were applied (student may already be at feature bounds).")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — What-If Simulator
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("What-If Simulator")
    st.markdown("Adjust sliders to see how the prediction changes in real time.")

    sim_cols   = st.columns(2)
    sim_values = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        col = sim_cols[i % 2]
        lo, hi, step = FEATURE_RANGES[feat]
        default_val = float(round(student_row[feat], 2))
        default_val = max(lo, min(hi, default_val))
        sim_values[feat] = col.slider(
            FRIENDLY.get(feat, feat),
            min_value=float(lo), max_value=float(hi),
            value=default_val, step=float(step),
            key=f"sim_{feat}",
        )

    sim_df   = pd.DataFrame([sim_values])
    sim_prob  = float(model_pipeline.predict_proba(sim_df)[:, 1][0])
    sim_pred  = int(sim_prob >= 0.5)

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if sim_pred == 1:
            st.markdown('<span class="risk-badge-on">ON TRACK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-badge-off">AT RISK</span>', unsafe_allow_html=True)
        st.metric(
            "Simulated Mastery Probability", f"{sim_prob:.3f}",
            delta=f"{sim_prob - student_prob:+.3f}",
        )

    with res_col2:
        diffs = {
            FRIENDLY.get(f, f): sim_values[f] - float(student_row[f])
            for f in FEATURE_COLUMNS
        }
        nonzero_diffs = {k: v for k, v in diffs.items() if abs(v) > 1e-6}
        if nonzero_diffs:
            fig_diff = go.Figure(go.Bar(
                x=list(nonzero_diffs.values()),
                y=list(nonzero_diffs.keys()),
                orientation="h",
                marker_color=["#42a5f5" if v > 0 else "#ef5350"
                              for v in nonzero_diffs.values()],
            ))
            fig_diff.update_layout(
                title="Feature Changes vs. Original",
                height=300,
                margin=dict(l=180, r=10, t=40, b=30),
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("No changes from original student values yet.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Model Comparison")

    saved_metrics = load_metrics()
    rows = []
    roc_fig = go.Figure()
    cal_fig = go.Figure()

    for mname in available:
        m = all_models[mname]
        metrics = evaluate_binary(m.pipeline, X_test, y_test)

        # Pull CV data from saved metrics if available
        cv_auc_str = ""
        if mname in saved_metrics and "cross_validation" in saved_metrics[mname]:
            cv = saved_metrics[mname]["cross_validation"]
            cv_auc_str = f"{cv['roc_auc']['mean']:.3f}±{cv['roc_auc']['std']:.3f}"

        rows.append({
            "Model":        MODEL_LABELS.get(mname, mname),
            "Accuracy":     f"{metrics['accuracy']:.3f}",
            "Precision":    f"{metrics['precision']:.3f}",
            "Recall":       f"{metrics['recall']:.3f}",
            "F1":           f"{metrics['f1']:.3f}",
            "ROC-AUC":      f"{metrics['roc_auc']:.3f}",
            "CV ROC-AUC":   cv_auc_str or "—",
            "Brier Score":  f"{metrics['brier_score']:.3f}",
            "MCC":          f"{metrics['mcc']:.3f}",
        })

        from sklearn.metrics import roc_curve
        proba = m.pipeline.predict_proba(X_test)[:, 1]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, proba)
        roc_fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, mode="lines",
            name=f"{MODEL_LABELS.get(mname, mname)} (AUC={metrics['roc_auc']:.3f})",
        ))

        cal_data = get_calibration_data(m.pipeline, X_test, y_test)
        cal_fig.add_trace(go.Scatter(
            x=cal_data["mean_predicted_value"],
            y=cal_data["fraction_of_positives"],
            name=MODEL_LABELS.get(mname, mname),
            mode="lines+markers",
        ))

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # McNemar results if available
    if "mcnemar_tests" in saved_metrics:
        with st.expander("📐 McNemar Significance Tests"):
            mc_rows = []
            for mc in saved_metrics["mcnemar_tests"]:
                mc_rows.append({
                    "Model A": MODEL_LABELS.get(mc["model_a"], mc["model_a"]),
                    "Model B": MODEL_LABELS.get(mc["model_b"], mc["model_b"]),
                    "p-value": mc["p_value"],
                    "Significant (α=0.05)": "✅ Yes" if mc["significant"] else "❌ No",
                })
            st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_roc, col_cal = st.columns(2)

    with col_roc:
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        roc_fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, **DARK_LAYOUT,
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_cal:
        cal_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        cal_fig.update_layout(
            title="Calibration Curves (Reliability Diagram)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=380, **DARK_LAYOUT,
        )
        st.plotly_chart(cal_fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Fairness Analysis
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Fairness & Bias Analysis")

    if demo_test is None:
        st.warning(
            "Demographic columns not found. "
            "Re-run `python src/data_gen.py` to regenerate the dataset."
        )
    else:
        group_options = {
            "Gender (Female / Male / Non-binary)":    "gender",
            "First-Generation Student":               "first_gen",
            "Socioeconomic Status (Low/Med/High)":    "ses_index",
        }
        selected_group_label = st.selectbox(
            "Analyze fairness by:", list(group_options.keys())
        )
        group_col = group_options[selected_group_label]

        fairness = compute_fairness_metrics(
            model_pipeline, X_test, y_test, demo_test, group_col
        )

        if fairness.get("is_binned"):
            st.info(
                f"**{group_col}** is a continuous variable and has been "
                "automatically binned into Low / Medium / High terciles."
            )

        flags  = fairness["fairness_flags"]
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            di   = fairness["disparate_impact_ratio"]
            icon = "✅" if flags["disparate_impact_ok"] else "❌"
            st.metric(
                f"{icon} Disparate Impact Ratio", f"{di:.3f}",
                help="≥ 0.8 required (80% rule). Ratio of min/max positive prediction rate.",
            )
        with col_f2:
            eo_gap = fairness["equal_opportunity_gap"]
            icon   = "✅" if flags["equal_opportunity_ok"] else "❌"
            st.metric(
                f"{icon} Equal Opportunity Gap", f"{eo_gap:.3f}",
                help="Max - Min TPR across groups. Should be ≤ 0.10.",
            )
        with col_f3:
            dp_gap = fairness["demographic_parity_gap"]
            icon   = "✅" if flags["demographic_parity_ok"] else "❌"
            st.metric(
                f"{icon} Demographic Parity Gap", f"{dp_gap:.3f}",
                help="Max - Min positive rate across groups. Should be ≤ 0.10.",
            )

        st.markdown("---")
        group_data   = fairness["group_metrics"]
        group_labels = fairness["groups"]

        metric_choices = ["accuracy", "tpr", "fpr", "precision", "positive_rate"]
        metric_labels  = {
            "accuracy":      "Accuracy",
            "tpr":           "True Positive Rate (Equal Opportunity)",
            "fpr":           "False Positive Rate",
            "precision":     "Precision (Predictive Parity)",
            "positive_rate": "Positive Rate (Demographic Parity)",
        }
        selected_metric = st.selectbox(
            "Show metric:", metric_choices,
            format_func=lambda m: metric_labels[m],
        )

        gender_map    = {"0": "Female", "1": "Male",       "2": "Non-binary"}
        first_gen_map = {"0": "Continuing", "1": "First-Gen"}

        def label_group(g):
            if group_col == "gender":
                return gender_map.get(g, g)
            if group_col == "first_gen":
                return first_gen_map.get(g, g)
            return g   # for ses_index binned groups: "Low" / "Medium" / "High"

        display_labels = [label_group(g) for g in group_labels]
        vals = [group_data[g].get(selected_metric, 0.0) for g in group_labels]
        ns   = [group_data[g].get("n", 0) for g in group_labels]

        bar_fig = go.Figure(go.Bar(
            x=display_labels, y=vals,
            text=[f"{v:.3f} (n={n})" for v, n in zip(vals, ns)],
            textposition="outside",
            marker_color=["#42a5f5"] * len(vals),
        ))
        bar_fig.update_layout(
            title=f"{metric_labels[selected_metric]} by {selected_group_label}",
            yaxis=dict(range=[0, 1.15]),
            height=380, **DARK_LAYOUT,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        with st.expander("View raw group metrics table"):
            table_rows = []
            for g in group_labels:
                row = {"Group": label_group(g)}
                row.update({
                    metric_labels.get(k, k): round(v, 4)
                    for k, v in group_data[g].items()
                })
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Research Metrics
# ════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Research Metrics")
    st.markdown(
        "Deep-dive analytics for assessing explanation quality, stability, and coverage "
        "across the full test set."
    )

    # ── Cross-validation stability ───────────────────────────────────────────
    st.markdown("#### Cross-Validation Stability (from saved metrics.json)")
    saved_metrics = load_metrics()
    cv_rows = []
    for mname in MODEL_NAMES:
        if mname in saved_metrics and "cross_validation" in saved_metrics[mname]:
            cv = saved_metrics[mname]["cross_validation"]
            cv_rows.append({
                "Model":        MODEL_LABELS.get(mname, mname),
                "CV AUC Mean":  f"{cv['roc_auc']['mean']:.4f}",
                "CV AUC Std":   f"{cv['roc_auc']['std']:.4f}",
                "CV F1 Mean":   f"{cv['f1']['mean']:.4f}",
                "CV F1 Std":    f"{cv['f1']['std']:.4f}",
                "CV MCC Mean":  f"{cv['mcc']['mean']:.4f}",
                "CV Brier Mean": f"{cv['brier_score']['mean']:.4f}",
            })
    if cv_rows:
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Low CV Std → stable model. High Std → potentially overfitting or "
            "sensitive to data splits."
        )
    else:
        st.info("No cross-validation data found. Run `scripts/run_all.py` to generate.")

    # ── SHAP vs LIME agreement across all students ────────────────────────────
    st.markdown("---")
    st.markdown("#### SHAP vs LIME Top-5 Agreement (across test set)")
    shap_df_r = get_shap_csv(selected_model_name)
    lime_df_r = get_lime_csv(selected_model_name)

    if shap_df_r is not None and lime_df_r is not None:
        agree_counts = []
        n_compared = min(len(shap_df_r), len(lime_df_r))
        for i in range(n_compared):
            s_feats = set(shap_df_r.iloc[i]["top_features"].split("; ")[:5])
            l_feats = set(lime_df_r.iloc[i]["top_features"].split("; ")[:5])
            agree_counts.append(len(s_feats & l_feats))

        agree_arr = np.array(agree_counts)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Mean Agreement", f"{agree_arr.mean():.2f} / 5 features")
        with a2:
            st.metric("Full Agreement (5/5)", f"{(agree_arr == 5).mean():.1%} of students")
        with a3:
            st.metric("Low Agreement (≤ 2/5)", f"{(agree_arr <= 2).mean():.1%} of students")

        fig_agree = go.Figure(go.Histogram(
            x=agree_counts,
            nbinsx=6,
            marker_color="#42a5f5",
            opacity=0.8,
        ))
        fig_agree.update_layout(
            title=f"Distribution of SHAP–LIME Top-5 Agreement — {MODEL_LABELS.get(selected_model_name)}",
            xaxis=dict(title="# Features in common (out of 5)", tickvals=list(range(6))),
            yaxis_title="# Students",
            height=320, **DARK_LAYOUT,
        )
        st.plotly_chart(fig_agree, use_container_width=True)
    else:
        st.info(
            "SHAP and LIME CSVs not found for this model. "
            "Run `scripts/run_all.py` to generate both."
        )

    # ── Counterfactual flip rate by risk tier ────────────────────────────────
    st.markdown("---")
    st.markdown("#### Counterfactual Flip Rate by Risk Tier")

    from src.translator import _risk_tier
    n_sample_cf = st.slider(
        "Students to sample for CF analysis (higher = slower)",
        min_value=20, max_value=200, value=50, step=10,
        key="cf_sample_size",
    )

    if st.button("Run Counterfactual Flip Analysis"):
        sample_idx = np.random.default_rng(0).choice(len(X_test), size=n_sample_cf, replace=False)
        cf_results = []
        prog = st.progress(0)
        for ii, idx_i in enumerate(sample_idx):
            row_i   = X_test.iloc[int(idx_i)]
            prob_i  = float(model_pipeline.predict_proba(pd.DataFrame([row_i]))[:, 1][0])
            tier_i  = _risk_tier(prob_i)
            cf_i    = generate_counterfactual(model_pipeline, row_i.copy())
            cf_results.append({
                "tier":    tier_i,
                "status":  cf_i["status"],
                "base_prob": cf_i["base_prob"],
            })
            prog.progress((ii + 1) / n_sample_cf)

        cf_df = pd.DataFrame(cf_results)
        at_risk = cf_df[cf_df["status"] != "already_on_track"]
        if len(at_risk) > 0:
            flip_rate = (at_risk["status"] == "flipped").mean()
            st.metric("Overall Flip Rate (among at-risk students)", f"{flip_rate:.1%}")

            tier_order = ["High Risk", "At Risk", "Borderline"]
            tier_rows  = []
            for tier in tier_order:
                t_sub = at_risk[at_risk["tier"] == tier]
                if len(t_sub) == 0:
                    continue
                tier_rows.append({
                    "Risk Tier": tier,
                    "# Students Sampled": len(t_sub),
                    "Flip Rate": f"{(t_sub['status'] == 'flipped').mean():.1%}",
                    "Not Flipped": f"{(t_sub['status'] == 'not_flipped').mean():.1%}",
                })
            if tier_rows:
                st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "Higher flip rate = more actionable interventions available. "
                    "'Not Flipped' students may need structural support beyond behavioral changes."
                )
        else:
            st.success("All sampled students are already on track!")
    else:
        st.info("Click **Run Counterfactual Flip Analysis** to compute flip rates.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Instructor Report
# ════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader(f"Instructor Report — Student #{student_idx}")

    shap_df = get_shap_csv(selected_model_name)
    feats_r, contribs_r, full_shap_r = _parse_shap_row(shap_df, student_idx)

    if full_shap_r is not None:
        with st.spinner("Generating counterfactual for instructor report..."):
            cf = generate_counterfactual(model_pipeline, student_row.copy())

        report = summarize_for_instructor(
            FEATURE_COLUMNS,
            student_row.to_dict(),
            full_shap_r,
            student_prob,
            model_name=selected_model_name,
            counterfactual=cf,
        )
        st.code(report, language=None)

        report_json = {
            "student_index":        student_idx,
            "model":                selected_model_name,
            "mastery_probability":  student_prob,
            "predicted":            "on-track" if student_pred else "at-risk",
            "true_label":           "on-track" if student_true else "at-risk",
            "feature_values":       student_row.to_dict(),
            "shap_top_features":    feats_r,
            "shap_top_contributions": contribs_r,
            "counterfactual":       cf,
            "instructor_report":    report,
        }
        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(report_json, indent=2),
            file_name=f"student_{student_idx}_report.json",
            mime="application/json",
        )
    else:
        # Fallback: show basic info even without SHAP
        from src.translator import _risk_tier
        tier = _risk_tier(student_prob)

        st.info(
            "SHAP explanations not found for this model/student. "
            "Run `scripts/run_all.py` to generate them."
        )
        st.markdown(f"**Risk Tier:** {tier}")
        st.markdown(f"**Mastery Probability:** {student_prob:.3f}")
        st.markdown(f"**Model:** {MODEL_LABELS.get(selected_model_name, selected_model_name)}")
