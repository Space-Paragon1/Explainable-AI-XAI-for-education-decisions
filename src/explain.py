from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for scripts / CI
import matplotlib.pyplot as plt
import numpy as np
import shap


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fix_xgb_base_score(clf) -> None:
    """
    Patch XGBoost 2.x + SHAP incompatibility.

    XGBoost >= 2.0 serialises base_score as '[5E-1]' (JSON array notation).
    SHAP's TreeEnsemble loader calls float('[5E-1]') which raises ValueError.
    We patch the booster config in-place to store a plain float string instead.
    """
    try:
        booster = clf.get_booster()
        config  = json.loads(booster.save_config())
        bs = config["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("["):
            # '[5E-1]' -> strip brackets -> '5E-1' -> float -> '0.5'
            val = float(bs.strip("[]"))
            config["learner"]["learner_model_param"]["base_score"] = str(val)
            booster.load_config(json.dumps(config))
    except Exception:
        pass  # If patching fails, let SHAP raise its own error


def _extract_class1_shap(shap_output) -> np.ndarray:
    """
    Robustly extract class-1 SHAP values from any SHAP output format.

    Handles:
    - Old SHAP API: list of arrays [class0, class1], each (n_samples, n_features)
    - New SHAP API: single 3-D array (n_samples, n_features, n_classes)
    - SHAP Explanation object (shap >= 0.40)
    - 2-D array passed directly (KernelExplainer output)
    """
    # SHAP Explanation object (shap >= 0.40 with TreeExplainer)
    if hasattr(shap_output, "values"):
        vals = np.array(shap_output.values)
        if vals.ndim == 3:
            return vals[:, :, 1]
        return vals

    # Old-style list: [class0_array, class1_array]
    if isinstance(shap_output, list):
        return np.array(shap_output[1])

    arr = np.array(shap_output)
    if arr.ndim == 3:
        return arr[:, :, 1]   # (n_samples, n_features, n_classes) -> class 1
    return arr                 # already (n_samples, n_features)


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
    - SHAP beeswarm summary plot (global, PNG)
    - Local explanations for up to max_local rows (CSV)

    Robust to all SHAP output formats including 3-D arrays from newer
    TreeExplainer on RandomForestClassifier.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    preprocess = trained_pipeline.named_steps["preprocess"]
    clf = trained_pipeline.named_steps["clf"]

    Xb = preprocess.transform(X_background)
    Xe = preprocess.transform(X_explain)

    feature_names_arr = np.array(feature_names)
    is_tree = clf.__class__.__name__ in {
        "RandomForestClassifier", "XGBClassifier", "LGBMClassifier"
    }

    if is_tree:
        if clf.__class__.__name__ == "XGBClassifier":
            _fix_xgb_base_score(clf)
        explainer = shap.TreeExplainer(clf)
        # Convert to plain float64 numpy array — avoids feature-name mismatches
        # with XGBoost/LightGBM when the preprocessor added column-name metadata.
        Xe_arr = np.ascontiguousarray(Xe, dtype=np.float64)
        shap_raw = explainer.shap_values(Xe_arr)
        sv = _extract_class1_shap(shap_raw)
    else:
        # KernelExplainer — sample a small background to keep it tractable
        f = lambda z: clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, Xb[:200])
        n_explain = min(len(Xe), 300)
        shap_raw = explainer.shap_values(Xe[:n_explain], nsamples=200)
        Xe = Xe[:n_explain]
        sv = _extract_class1_shap(shap_raw)

    # Guarantee 2-D: (n_samples, n_features)
    sv = np.array(sv, dtype=float)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    if sv.shape[1] != len(feature_names):
        sv = sv[:, : len(feature_names)]

    # ── Global summary plot ──────────────────────────────────────────────────
    summary_path = out_dir / f"{model_name}_shap_summary.png"
    plt.figure()
    shap.summary_plot(sv, np.array(Xe, dtype=float), feature_names=feature_names_arr, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # ── Local explanations CSV ───────────────────────────────────────────────
    local_csv = out_dir / f"{model_name}_local_explanations.csv"
    k = min(max_local, sv.shape[0])
    rows = []

    for i in range(k):
        contrib = np.array(sv[i]).ravel()          # guaranteed 1-D
        idx = np.argsort(np.abs(contrib))[::-1]
        top = idx[:5]
        rows.append(
            {
                "row_index": int(i),
                "top_features": "; ".join(
                    [str(feature_names_arr[int(j)]) for j in top]
                ),
                "top_contributions": "; ".join(
                    [f"{float(contrib[int(j)]):.4f}" for j in top]
                ),
            }
        )

    with local_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["row_index", "top_features", "top_contributions"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "summary_plot": str(summary_path),
        "local_explanations_csv": str(local_csv),
        "n_explained": k,
    }
