"""
lime_explain.py — LIME-based local explanations for XAI-ED.

Complements SHAP explanations by providing a second, model-agnostic
local explanation method (Local Interpretable Model-agnostic Explanations).

LIME fits a simple linear model locally around each prediction, making
it easy to understand which features mattered *for that specific student*.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def explain_with_lime(
    trained_pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_dir: str | Path,
    model_name: str,
    max_local: int = 25,
    num_samples: int = 1000,
) -> dict:
    """
    Generate LIME explanations and save summary artifacts.

    Parameters
    ----------
    trained_pipeline : fitted sklearn Pipeline (preprocess + clf)
    X_train : training data used to fit LIME's background distribution
    X_explain : instances to explain (numpy array, already in feature space)
    feature_names : list of feature name strings
    out_dir : directory for output files
    model_name : string prefix for output files (e.g., "logreg", "rf")
    max_local : how many instances to explain
    num_samples : LIME perturbation samples per instance

    Returns
    -------
    dict with paths to saved artifacts
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise ImportError("lime is required: pip install lime") from e

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # LIME needs the raw predict_proba on original feature scale
    preprocess = trained_pipeline.named_steps["preprocess"]
    clf = trained_pipeline.named_steps["clf"]

    def predict_fn(x_raw: np.ndarray) -> np.ndarray:
        x_transformed = preprocess.transform(x_raw)
        return clf.predict_proba(x_transformed)

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["at-risk", "on-track"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    k = min(max_local, X_explain.shape[0])
    rows = []
    all_weights: dict[str, list[float]] = {f: [] for f in feature_names}

    for i in range(k):
        exp = explainer.explain_instance(
            X_explain[i],
            predict_fn,
            num_features=len(feature_names),
            num_samples=num_samples,
            labels=(1,),  # explain class 1 (on-track / mastery)
        )
        weights = dict(exp.as_list(label=1))

        # Map LIME's discretized feature labels back to base feature names
        # LIME returns strings like "feature_name <= 0.50" — we extract the base name
        feature_weight_map: dict[str, float] = {}
        for label, w in weights.items():
            matched = None
            for fn in feature_names:
                if fn in label:
                    matched = fn
                    break
            if matched:
                feature_weight_map[matched] = w
                all_weights[matched].append(w)

        # Sort by absolute weight
        sorted_weights = sorted(feature_weight_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_features = sorted_weights[:5]

        rows.append(
            {
                "row_index": i,
                "top_features": "; ".join(f for f, _ in top_features),
                "top_lime_weights": "; ".join(f"{w:.4f}" for _, w in top_features),
            }
        )

    # Save local explanations CSV
    local_csv = out_dir / f"{model_name}_lime_local_explanations.csv"
    with local_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "top_features", "top_lime_weights"])
        writer.writeheader()
        writer.writerows(rows)

    # Compute mean absolute LIME weight per feature across explained instances
    mean_abs_weights = {
        fn: float(np.mean(np.abs(vals))) if vals else 0.0
        for fn, vals in all_weights.items()
    }
    sorted_features = sorted(mean_abs_weights.items(), key=lambda kv: kv[1], reverse=True)

    # Save global importance bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    feat_labels = [fn for fn, _ in sorted_features]
    feat_vals = [v for _, v in sorted_features]
    colors = ["#2196F3"] * len(feat_labels)
    ax.barh(feat_labels[::-1], feat_vals[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |LIME weight| (class: on-track)")
    ax.set_title(f"LIME Global Feature Importance — {model_name.upper()}")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    summary_path = out_dir / f"{model_name}_lime_summary.png"
    plt.savefig(summary_path, dpi=200)
    plt.close()

    return {
        "lime_summary_plot": str(summary_path),
        "lime_local_explanations_csv": str(local_csv),
        "mean_abs_weights": mean_abs_weights,
    }
