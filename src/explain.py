from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def explain_with_shap(
    trained_pipeline,
    X_background,
    X_explain,
    feature_names: list[str],
    out_dir: str | Path,
    model_name: str,
    max_local: int = 25,
) -> dict:
    """
    Produces:
    - SHAP summary plot (global)
    - Local explanations for up to max_local rows (CSV)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Pipeline = preprocess + classifier.
    preprocess = trained_pipeline.named_steps["preprocess"]
    clf = trained_pipeline.named_steps["clf"]

    # Transform data for SHAP (works best on the model input space).
    Xb = preprocess.transform(X_background)
    Xe = preprocess.transform(X_explain)

    # If preprocessing yields numpy arrays, feature order matches feature_names
    # because we used a simple scaler over all columns.
    # SHAP explainers differ depending on model type:
    is_tree = clf.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier"}

    if is_tree:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Xe)
        # For binary classification, shap_values may be list [class0, class1]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        # KernelExplainer can be slow; use a smaller background set.
        # Use model's probability of class 1.
        f = lambda z: clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, Xb[:200])
        sv = explainer.shap_values(Xe[: min(len(Xe), 300)], nsamples=200)
        Xe = Xe[: min(len(Xe), 300)]

    # Global summary plot
    summary_path = out_dir / f"{model_name}_shap_summary.png"
    plt.figure()
    shap.summary_plot(sv, Xe, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # Local explanations CSV (top contributors)
    local_csv = out_dir / f"{model_name}_local_explanations.csv"
    k = min(max_local, Xe.shape[0])

    rows = []
    for i in range(k):
        contrib = sv[i]
        # top features by absolute impact
        idx = np.argsort(np.abs(contrib))[::-1]
        top = idx[:5]
        rows.append(
            {
                "row_index": int(i),
                "top_features": "; ".join([feature_names[j] for j in top]),
                "top_contributions": "; ".join([f"{contrib[j]:.4f}" for j in top]),
            }
        )

    with local_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_index", "top_features", "top_contributions"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "summary_plot": str(summary_path),
        "local_explanations_csv": str(local_csv),
    }
