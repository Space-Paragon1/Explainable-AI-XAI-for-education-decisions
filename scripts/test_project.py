"""
test_project.py — Quick end-to-end smoke test for XAI-ED.

Usage:
    python scripts/test_project.py

Checks: data, models, prediction, counterfactual, fairness, SHAP artifacts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import FEATURE_COLUMNS, DEMOGRAPHIC_COLUMNS, TARGET_COLUMN
from src.counterfactual import generate_counterfactual
from src.fairness import compute_fairness_metrics

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition

errors = 0

print("\n=== XAI-ED Smoke Test ===\n")

# ── Data ──────────────────────────────────────────────────────────────────────
print("[ Data ]")
csv = Path("data/student_data.csv")
ok = check("student_data.csv exists", csv.exists())
if ok:
    df = pd.read_csv(csv)
    check("3000 rows", len(df) == 3000, f"found {len(df)}")
    check("mastery column present", "mastery" in df.columns)
    check("demographic columns present", all(c in df.columns for c in DEMOGRAPHIC_COLUMNS))
    check("SES equity gap exists",
          df[df["ses_index"] < 0.33]["mastery"].mean() <
          df[df["ses_index"] > 0.67]["mastery"].mean())

# ── Models ────────────────────────────────────────────────────────────────────
print("\n[ Models ]")
models = {}
for name in ["logreg", "rf", "xgb", "lgbm"]:
    path = Path(f"outputs/models/{name}.joblib")
    ok = check(f"{name}.joblib exists", path.exists())
    if ok:
        models[name] = joblib.load(path)

# ── Prediction ────────────────────────────────────────────────────────────────
print("\n[ Prediction ]")
if "rf" in models:
    _, test_idx = train_test_split(range(len(df)), test_size=0.2,
                                   random_state=42, stratify=df[TARGET_COLUMN])
    df_test = df.iloc[test_idx].reset_index(drop=True)
    X_test  = df_test[FEATURE_COLUMNS]
    y_test  = df_test[TARGET_COLUMN]
    demo    = df_test[DEMOGRAPHIC_COLUMNS]

    prob = float(models["rf"].predict_proba(X_test.iloc[:1])[:, 1][0])
    check("RF prediction returns float in [0,1]", 0 <= prob <= 1, f"p={prob:.3f}")

    for name, model in models.items():
        p = float(model.predict_proba(X_test.iloc[:1])[:, 1][0])
        check(f"{name} predicts without error", 0 <= p <= 1, f"p={p:.3f}")

# ── Counterfactual ────────────────────────────────────────────────────────────
print("\n[ Counterfactual ]")
if "rf" in models:
    cf = generate_counterfactual(models["rf"], X_test.iloc[0].copy())
    check("status is valid",   cf["status"] in {"flipped","not_flipped","already_on_track"})
    check("new_prob >= base",  cf["new_prob"] >= cf["base_prob"] - 1e-6,
          f"base={cf['base_prob']:.3f} new={cf['new_prob']:.3f}")

# ── Fairness ──────────────────────────────────────────────────────────────────
print("\n[ Fairness ]")
if "rf" in models:
    for group in ["gender", "first_gen", "ses_index"]:
        r = compute_fairness_metrics(models["rf"], X_test, y_test, demo, group)
        di = r["disparate_impact_ratio"]
        check(f"{group} — DI ratio in [0,1]", 0 <= di <= 1, f"DI={di:.3f}")

# ── SHAP artifacts ────────────────────────────────────────────────────────────
print("\n[ SHAP Artifacts ]")
for name in ["logreg", "rf", "xgb", "lgbm"]:
    check(f"{name}_shap_summary.png",
          Path(f"outputs/explanations/{name}_shap_summary.png").exists())
    check(f"{name}_local_explanations.csv",
          Path(f"outputs/explanations/{name}_local_explanations.csv").exists())

# ── Metrics JSON ──────────────────────────────────────────────────────────────
print("\n[ Metrics ]")
metrics_path = Path("outputs/metrics/metrics.json")
ok = check("metrics.json exists", metrics_path.exists())
if ok:
    m = json.loads(metrics_path.read_text())
    for name in ["logreg", "rf", "xgb", "lgbm"]:
        if name in m:
            auc = m[name]["metrics"]["roc_auc"]
            cv  = m[name]["cross_validation"]["roc_auc"]
            check(f"{name} AUC > 0.70", auc > 0.70,
                  f"AUC={auc:.3f}  CV={cv['mean']:.3f}+-{cv['std']:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=========================")
