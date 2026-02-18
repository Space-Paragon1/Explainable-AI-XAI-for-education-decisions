"""
dashboard.py — Interactive Streamlit dashboard for XAI-ED.

Run with:
    streamlit run dashboard.py

Tabs:
  1. Student Overview    — prediction badge, probability, feature table, student explanation
  2. SHAP Explanations   — global summary plot + local waterfall bar chart
  3. Counterfactual      — minimal edits to flip at-risk → on-track
  4. What-If Simulator   — live sliders with real-time prediction update
  5. Model Comparison    — metrics table, ROC curves, calibration curves
  6. Fairness Analysis   — group-level bias metrics with pass/fail indicators
  7. Instructor Report   — technical view + downloadable JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-ED Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .risk-badge-on  { background:#1b5e20; color:white; border-radius:8px; padding:6px 18px; font-size:1.1rem; font-weight:700; }
    .risk-badge-off { background:#b71c1c; color:white; border-radius:8px; padding:6px 18px; font-size:1.1rem; font-weight:700; }
    .metric-card    { background:#1e1e2e; border-radius:10px; padding:14px; margin:6px; }
    .section-header { font-size:1.15rem; font-weight:600; margin-top:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

FRIENDLY = {
    "study_time_min": "Study Time (min/wk)",
    "practice_completion_rate": "Practice Completion",
    "avg_quiz_score": "Avg Quiz Score",
    "quiz_attempts": "Quiz Attempts",
    "hint_usage_rate": "Hint Usage Rate",
    "attendance_rate": "Attendance Rate",
    "days_since_last_activity": "Days Since Last Activity",
    "stress_index": "Stress Index",
    "prereq_mastery": "Prereq Mastery",
    "device_reliability": "Device Reliability",
}

MODEL_NAMES = ["logreg", "rf", "xgb", "lgbm"]
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

FEATURE_RANGES = {
    "study_time_min":           (0.0,   900.0, 1.0),
    "practice_completion_rate": (0.0,   1.0,   0.01),
    "avg_quiz_score":           (0.0,   100.0, 0.5),
    "quiz_attempts":            (0.0,   25.0,  1.0),
    "hint_usage_rate":          (0.0,   1.0,   0.01),
    "attendance_rate":          (0.0,   1.0,   0.01),
    "days_since_last_activity": (0.0,   30.0,  0.5),
    "stress_index":             (0.0,   1.0,   0.01),
    "prereq_mastery":           (0.0,   1.0,   0.01),
    "device_reliability":       (0.0,   1.0,   0.01),
}


@st.cache_resource
def load_all_models() -> dict[str, TrainedModel]:
    models = {}
    for name in MODEL_NAMES:
        path = PATHS.models_dir / f"{name}.joblib"
        if path.exists():
            models[name] = load_model(str(path))
    return models


@st.cache_data
def load_data():
    csv_path = PATHS.data_dir / "student_data.csv"
    df_full = pd.read_csv(csv_path)

    # Split same as training: 80/20, random_state=42
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(df_full)), test_size=0.2, random_state=42,
        stratify=df_full[TARGET_COLUMN]
    )
    df_test = df_full.iloc[test_idx].reset_index(drop=True)
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN]
    demo_test = df_test[DEMOGRAPHIC_COLUMNS] if all(c in df_test.columns for c in DEMOGRAPHIC_COLUMNS) else None
    return df_test, X_test, y_test, demo_test


@st.cache_data
def load_metrics():
    path = PATHS.metrics_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@st.cache_data
def get_shap_values_cached(model_name: str):
    """Load pre-computed SHAP CSVs if available, else return None."""
    csv_path = PATHS.explanations_dir / f"{model_name}_local_explanations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def shap_bar_chart(feature_names, shap_vals, title="Local SHAP Contributions"):
    """Plotly horizontal bar chart for local SHAP values."""
    colors = ["#ef5350" if v < 0 else "#42a5f5" for v in shap_vals]
    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=feature_names,
            orientation="h",
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
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 XAI-ED Dashboard")
    st.markdown("**Explainable AI for Education**")
    st.markdown("---")

    all_models = load_all_models()
    available = [m for m in MODEL_NAMES if m in all_models]

    if not available:
        st.error(
            "No trained models found in `outputs/models/`.\n\n"
            "Run the training pipeline first."
        )
        st.stop()

    selected_model_name = st.selectbox(
        "Select Model",
        options=available,
        format_func=lambda n: MODEL_LABELS.get(n, n),
    )
    trained = all_models[selected_model_name]
    model_pipeline = trained.pipeline

    df_test, X_test, y_test, demo_test = load_data()
    n_students = len(df_test)

    student_idx = st.slider(
        "Student ID (test set)",
        min_value=0,
        max_value=n_students - 1,
        value=0,
        step=1,
    )

    st.markdown("---")
    st.markdown(f"**Test set size:** {n_students} students")
    st.markdown(f"**Model:** {MODEL_LABELS.get(selected_model_name, selected_model_name)}")

# ── Get current student data ──────────────────────────────────────────────────
student_row = X_test.iloc[student_idx]
student_true = int(y_test.iloc[student_idx])
student_prob = float(model_pipeline.predict_proba(pd.DataFrame([student_row]))[:, 1][0])
student_pred = int(student_prob >= 0.5)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Student Overview",
    "SHAP Explanations",
    "Counterfactual",
    "What-If Simulator",
    "Model Comparison",
    "Fairness Analysis",
    "Instructor Report",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Student Overview
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
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

    # Feature values table
    col_table, col_text = st.columns([1, 1])

    with col_table:
        st.markdown("**Feature Values**")
        feat_df = pd.DataFrame(
            {"Feature": [FRIENDLY.get(f, f) for f in FEATURE_COLUMNS],
             "Value": [round(float(student_row[f]), 3) for f in FEATURE_COLUMNS]}
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_text:
        st.markdown("**Student-Friendly Explanation**")

        # Try to get SHAP values from cached CSV
        shap_df = get_shap_values_cached(selected_model_name)
        student_explanation = ""

        if shap_df is not None and student_idx < len(shap_df):
            row = shap_df.iloc[student_idx]
            feats = row["top_features"].split("; ")
            contribs_raw = row["top_contributions"].split("; ")
            contribs = [float(c) for c in contribs_raw]

            # Rebuild full SHAP array (zeros for non-top features)
            full_shap = [0.0] * len(FEATURE_COLUMNS)
            for feat, contrib in zip(feats, contribs):
                if feat in FEATURE_COLUMNS:
                    full_shap[FEATURE_COLUMNS.index(feat)] = contrib

            student_explanation = summarize_shap_to_text(
                FEATURE_COLUMNS,
                student_row.to_dict(),
                full_shap,
                student_prob,
            )
            st.info(student_explanation)
        else:
            st.info(
                f"Prediction: {'**on track**' if student_pred else '**at risk**'} "
                f"(mastery probability ≈ {student_prob:.2f}). "
                "Load SHAP explanations to see detailed feature contributions."
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP Explanations
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("SHAP Explanations")

    col_global, col_local = st.columns(2)

    with col_global:
        st.markdown("**Global SHAP Summary (all test students)**")
        summary_png = PATHS.explanations_dir / f"{selected_model_name}_shap_summary.png"
        if summary_png.exists():
            st.image(str(summary_png), use_container_width=True)
        else:
            st.warning(f"No SHAP summary plot found for **{selected_model_name}**. Run `explain.py` first.")

    with col_local:
        st.markdown(f"**Local SHAP — Student #{student_idx}**")
        shap_df = get_shap_values_cached(selected_model_name)

        if shap_df is not None and student_idx < len(shap_df):
            row = shap_df.iloc[student_idx]
            feats_raw = row["top_features"].split("; ")
            contribs_raw = row["top_contributions"].split("; ")
            feats = [FRIENDLY.get(f, f) for f in feats_raw]
            contribs = [float(c) for c in contribs_raw]
            fig = shap_bar_chart(feats, contribs, title=f"Student #{student_idx} — Top SHAP Contributors")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No local SHAP data available for this student index.")

    # LIME comparison
    st.markdown("---")
    st.markdown("**LIME Global Feature Importance**")
    lime_png = PATHS.explanations_dir / f"{selected_model_name}_lime_summary.png"
    if lime_png.exists():
        st.image(str(lime_png), caption=f"LIME — {MODEL_LABELS.get(selected_model_name)}", use_container_width=True)
    else:
        st.info("LIME summary not yet generated. Run `lime_explain.py` to add LIME explanations.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Counterfactual
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Counterfactual Explanation")
    st.markdown(
        "Shows the **minimal changes** a student can make to their behavior "
        "to flip an *at-risk* prediction to *on-track*."
    )

    with st.spinner("Computing counterfactual..."):
        cf = generate_counterfactual(model_pipeline, student_row.copy())

    status = cf["status"]
    base_prob = cf["base_prob"]
    new_prob = cf["new_prob"]
    edits = cf.get("edits", {})

    col_status, col_before, col_after = st.columns(3)
    with col_status:
        if status == "already_on_track":
            st.success("Student is already predicted **ON TRACK**. No changes needed.")
        elif status == "flipped":
            st.success(f"Prediction can be flipped! New mastery probability: **{new_prob:.3f}**")
        else:
            st.warning(f"Could not fully flip prediction. Best achieved: **{new_prob:.3f}**")

    with col_before:
        st.metric("Original Probability", f"{base_prob:.3f}")

    with col_after:
        st.metric("After Changes", f"{new_prob:.3f}", delta=f"{new_prob - base_prob:+.3f}")

    if edits:
        st.markdown("**Required Changes:**")
        edit_rows = []
        for feat, new_val in edits.items():
            old_val = float(student_row[feat])
            edit_rows.append({
                "Feature": FRIENDLY.get(feat, feat),
                "Current Value": round(old_val, 3),
                "Suggested Value": round(new_val, 3),
                "Change": f"{new_val - old_val:+.3f}",
            })
        st.dataframe(pd.DataFrame(edit_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — What-If Simulator
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("What-If Simulator")
    st.markdown("Adjust feature sliders to see how the prediction changes in real time.")

    sim_cols = st.columns(2)
    sim_values = {}

    for i, feat in enumerate(FEATURE_COLUMNS):
        col = sim_cols[i % 2]
        lo, hi, step = FEATURE_RANGES[feat]
        default_val = float(round(student_row[feat], 2))
        default_val = max(lo, min(hi, default_val))
        sim_values[feat] = col.slider(
            FRIENDLY.get(feat, feat),
            min_value=float(lo),
            max_value=float(hi),
            value=default_val,
            step=float(step),
            key=f"sim_{feat}",
        )

    sim_df = pd.DataFrame([sim_values])
    sim_prob = float(model_pipeline.predict_proba(sim_df)[:, 1][0])
    sim_pred = int(sim_prob >= 0.5)

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if sim_pred == 1:
            st.markdown('<span class="risk-badge-on">ON TRACK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-badge-off">AT RISK</span>', unsafe_allow_html=True)
        st.metric("Simulated Mastery Probability", f"{sim_prob:.3f}",
                  delta=f"{sim_prob - student_prob:+.3f}")

    with res_col2:
        # Quick bar chart: difference in feature values from original
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
                marker_color=["#42a5f5" if v > 0 else "#ef5350" for v in nonzero_diffs.values()],
            ))
            fig_diff.update_layout(
                title="Feature Changes vs. Original",
                height=300,
                margin=dict(l=180, r=10, t=40, b=30),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("No changes from original student values yet.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Model Comparison")

    saved_metrics = load_metrics()

    rows = []
    roc_fig = go.Figure()
    cal_fig = go.Figure()

    for mname in available:
        m = all_models[mname]
        metrics = evaluate_binary(m.pipeline, X_test, y_test)
        rows.append({
            "Model": MODEL_LABELS.get(mname, mname),
            "Accuracy": f"{metrics['accuracy']:.3f}",
            "Precision": f"{metrics['precision']:.3f}",
            "Recall": f"{metrics['recall']:.3f}",
            "F1": f"{metrics['f1']:.3f}",
            "ROC-AUC": f"{metrics['roc_auc']:.3f}",
            "Brier": f"{metrics['brier_score']:.3f}",
            "MCC": f"{metrics['mcc']:.3f}",
        })

        # ROC curve
        from sklearn.metrics import roc_curve
        proba = m.pipeline.predict_proba(X_test)[:, 1]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, proba)
        roc_fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr,
            name=f"{MODEL_LABELS.get(mname, mname)} (AUC={metrics['roc_auc']:.3f})",
            mode="lines",
        ))

        # Calibration
        cal_data = get_calibration_data(m.pipeline, X_test, y_test)
        cal_fig.add_trace(go.Scatter(
            x=cal_data["mean_predicted_value"],
            y=cal_data["fraction_of_positives"],
            name=MODEL_LABELS.get(mname, mname),
            mode="lines+markers",
        ))

    # Metrics table
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_roc, col_cal = st.columns(2)

    with col_roc:
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        roc_fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=380,
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_cal:
        cal_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        cal_fig.update_layout(
            title="Calibration Curves (Reliability Diagram)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=380,
        )
        st.plotly_chart(cal_fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Fairness Analysis
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Fairness & Bias Analysis")

    if demo_test is None:
        st.warning(
            "Demographic columns not found in dataset. "
            "Re-run `data_gen.py` to regenerate the dataset with demographic features."
        )
    else:
        group_options = {
            "Gender (0=female, 1=male, 2=non-binary)": "gender",
            "First-Generation Student": "first_gen",
        }
        selected_group_label = st.selectbox("Analyze fairness by:", list(group_options.keys()))
        group_col = group_options[selected_group_label]

        fairness = compute_fairness_metrics(
            model_pipeline, X_test, y_test, demo_test, group_col
        )

        # Summary indicators
        flags = fairness["fairness_flags"]
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            di = fairness["disparate_impact_ratio"]
            icon = "✅" if flags["disparate_impact_ok"] else "❌"
            st.metric(f"{icon} Disparate Impact Ratio", f"{di:.3f}",
                      help="Should be >= 0.8 (80% rule). Measures if one group is significantly less likely to get a positive prediction.")

        with col_f2:
            eo_gap = fairness["equal_opportunity_gap"]
            icon = "✅" if flags["equal_opportunity_ok"] else "❌"
            st.metric(f"{icon} Equal Opportunity Gap", f"{eo_gap:.3f}",
                      help="Max - Min TPR across groups. Should be <= 0.10.")

        with col_f3:
            dp_gap = fairness["demographic_parity_gap"]
            icon = "✅" if flags["demographic_parity_ok"] else "❌"
            st.metric(f"{icon} Demographic Parity Gap", f"{dp_gap:.3f}",
                      help="Max - Min positive prediction rate across groups. Should be <= 0.10.")

        st.markdown("---")

        # Per-group breakdown
        group_data = fairness["group_metrics"]
        group_labels = fairness["groups"]

        metric_choices = ["accuracy", "tpr", "fpr", "precision", "positive_rate"]
        metric_labels = {
            "accuracy": "Accuracy",
            "tpr": "True Positive Rate (Equal Opportunity)",
            "fpr": "False Positive Rate",
            "precision": "Precision (Predictive Parity)",
            "positive_rate": "Positive Rate (Demographic Parity)",
        }

        selected_metric = st.selectbox("Show metric:", metric_choices,
                                       format_func=lambda m: metric_labels[m])

        vals = [group_data[g].get(selected_metric, 0.0) for g in group_labels]
        ns = [group_data[g].get("n", 0) for g in group_labels]

        gender_map = {"0": "Female", "1": "Male", "2": "Non-binary"}
        first_gen_map = {"0": "Continuing", "1": "First-Gen"}

        def label_group(g):
            if group_col == "gender":
                return gender_map.get(g, g)
            elif group_col == "first_gen":
                return first_gen_map.get(g, g)
            return g

        display_labels = [label_group(g) for g in group_labels]

        bar_fig = go.Figure(go.Bar(
            x=display_labels,
            y=vals,
            text=[f"{v:.3f} (n={n})" for v, n in zip(vals, ns)],
            textposition="outside",
            marker_color=["#42a5f5"] * len(vals),
        ))
        bar_fig.update_layout(
            title=f"{metric_labels[selected_metric]} by {selected_group_label}",
            yaxis=dict(range=[0, 1.1]),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=380,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Raw table
        with st.expander("View raw group metrics table"):
            table_rows = []
            for g in group_labels:
                row = {"Group": label_group(g)}
                row.update({metric_labels.get(k, k): round(v, 4) for k, v in group_data[g].items()})
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Instructor Report
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader(f"Instructor Report — Student #{student_idx}")

    shap_df = get_shap_values_cached(selected_model_name)

    if shap_df is not None and student_idx < len(shap_df):
        row = shap_df.iloc[student_idx]
        feats_raw = row["top_features"].split("; ")
        contribs_raw = row["top_contributions"].split("; ")
        contribs = [float(c) for c in contribs_raw]

        full_shap = [0.0] * len(FEATURE_COLUMNS)
        for feat, contrib in zip(feats_raw, contribs):
            if feat in FEATURE_COLUMNS:
                full_shap[FEATURE_COLUMNS.index(feat)] = contrib

        with st.spinner("Generating counterfactual for instructor report..."):
            cf = generate_counterfactual(model_pipeline, student_row.copy())

        report = summarize_for_instructor(
            FEATURE_COLUMNS,
            student_row.to_dict(),
            full_shap,
            student_prob,
            model_name=selected_model_name,
            counterfactual=cf,
        )
        st.code(report, language=None)

        # Download report
        report_json = {
            "student_index": student_idx,
            "model": selected_model_name,
            "mastery_probability": student_prob,
            "predicted": "on-track" if student_pred else "at-risk",
            "true_label": "on-track" if student_true else "at-risk",
            "feature_values": student_row.to_dict(),
            "shap_top_features": feats_raw,
            "shap_top_contributions": contribs,
            "counterfactual": cf,
            "instructor_report": report,
        }
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report_json, indent=2),
            file_name=f"student_{student_idx}_report.json",
            mime="application/json",
        )
    else:
        st.info(
            "SHAP explanations not found for this model/student. "
            "Run `explain.py` to generate them first."
        )
        # Still show basic instructor info
        from src.translator import _risk_tier
        tier = _risk_tier(student_prob)
        st.write(f"**Risk Tier:** {tier}")
        st.write(f"**Mastery Probability:** {student_prob:.3f}")
        st.write(f"**Model:** {MODEL_LABELS.get(selected_model_name, selected_model_name)}")
