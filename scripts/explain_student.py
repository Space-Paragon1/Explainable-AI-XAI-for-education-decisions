import sys
import os
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import shap


# Allow running as script by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import FEATURE_COLUMNS
from src.translator import summarize_shap_to_text
from src.counterfactual import generate_counterfactual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .joblib")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--row", type=int, default=0, help="Row index to explain")
    parser.add_argument("--out", type=str, default="outputs/explanations/student_report.json")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    x = df.loc[args.row, FEATURE_COLUMNS]
    X = df[FEATURE_COLUMNS]

    # Predict
    prob = float(model.predict_proba(pd.DataFrame([x]))[:, 1][0])

    # SHAP (tree vs non-tree)
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    Xb = preprocess.transform(X.sample(n=min(600, len(X)), random_state=42))
    x_t = preprocess.transform(pd.DataFrame([x]))

    is_tree = clf.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier"}
    if is_tree:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(x_t)
        shap_row = (sv[1][0] if isinstance(sv, list) else sv[0]).tolist()
    else:
        f = lambda z: clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, Xb[:200])
        sv = explainer.shap_values(x_t, nsamples=200)
        shap_row = (sv[0] if isinstance(sv, list) else sv).tolist()

    # Student-friendly text
    feature_values = {k: float(x[k]) for k in FEATURE_COLUMNS}
    explanation_text = summarize_shap_to_text(
        feature_names=FEATURE_COLUMNS,
        feature_values=feature_values,
        shap_values=shap_row,
        predicted_prob_mastery=prob,
    )

    # Counterfactual
    cf = generate_counterfactual(model, x)

    report = {
        "row_index": args.row,
        "mastery_probability": prob,
        "feature_values": feature_values,
        "shap_values": shap_row,
        "student_explanation": explanation_text,
        "counterfactual": cf,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("✅ Saved report:", out_path)
    print("\n--- Student Explanation ---\n")
    print(explanation_text)
    print("\n--- Counterfactual ---\n")
    print(cf)


if __name__ == "__main__":
    main()
