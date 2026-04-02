"""
run_all.py — Full XAI-ED pipeline: train all 4 models, evaluate, explain.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --models logreg,rf
    python scripts/run_all.py --skip-lime          # faster, skips LIME
    python scripts/run_all.py --regen-data         # re-generate student_data.csv first
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, Paths
from src.data_gen import generate_student_dataset
from src.data_loader import load_dataset
from src.evaluate import (
    cross_validate_model,
    evaluate_binary,
    compare_models_mcnemar,
    save_metrics,
)
from src.explain import explain_with_shap
from src.lime_explain import explain_with_lime
from src.train_model import build_model, train

ALL_MODELS = ["logreg", "rf", "xgb", "lgbm"]


def main():
    parser = argparse.ArgumentParser(
        description="Run full XAI-ED pipeline: train all 4 models, evaluate, explain."
    )
    parser.add_argument("--data",       type=str, default="data/student_data.csv")
    parser.add_argument("--out",        type=str, default="outputs")
    parser.add_argument("--models",     type=str, default="all",
                        help="Comma-separated model names, or 'all'.")
    parser.add_argument("--skip-lime",  action="store_true",
                        help="Skip LIME generation (faster).")
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenerate student_data.csv before training.")
    args = parser.parse_args()

    model_list = ALL_MODELS if args.models == "all" else args.models.split(",")

    # ── Output directories ───────────────────────────────────────────────────
    out_root        = Path(args.out)
    models_dir      = out_root / "models"
    metrics_dir     = out_root / "metrics"
    explanations_dir = out_root / "explanations"
    for d in [models_dir, metrics_dir, explanations_dir]:
        if d.exists() and not d.is_dir():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if args.regen_data or not data_path.exists():
        print("Generating synthetic dataset...")
        df = generate_student_dataset(n=3000, seed=42)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"  Saved {len(df)} rows -> {data_path}")

    print(f"Loading data from {data_path}...")
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Mastery rate: {y.mean():.3f}")

    bg = X_train.sample(n=min(600, len(X_train)), random_state=42)
    ex = X_test.sample(n=min(300, len(X_test)), random_state=42)

    trained_models = {}
    results = {}

    # ── Train, evaluate, explain each model ──────────────────────────────────
    for model_name in model_list:
        print(f"\n{'=' * 55}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'=' * 55}")

        print(f"  Training...")
        trained = train(model_name, X_train, y_train)
        model   = trained.pipeline
        trained_models[model_name] = model

        from src.train_model import save_model
        save_model(trained, models_dir / f"{model_name}.joblib")
        print(f"  Saved -> {models_dir / f'{model_name}.joblib'}")

        print(f"  Evaluating on test set...")
        metrics = evaluate_binary(model, X_test, y_test)
        print(
            f"  Accuracy={metrics['accuracy']:.3f} | "
            f"ROC-AUC={metrics['roc_auc']:.3f} | "
            f"F1={metrics['f1']:.3f} | "
            f"MCC={metrics['mcc']:.3f}"
        )

        print(f"  Running 5-fold cross-validation...")
        cv = cross_validate_model(
            model_factory=lambda mn=model_name: build_model(mn, y_train),
            X=X_train,
            y=y_train,
        )
        print(
            f"  CV ROC-AUC: {cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f} | "
            f"CV F1: {cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}"
        )

        print(f"  Generating SHAP explanations...")
        try:
            shap_paths = explain_with_shap(
                trained_pipeline=model,
                X_background=bg,
                X_explain=ex,
                feature_names=FEATURE_COLUMNS,
                out_dir=explanations_dir,
                model_name=model_name,
                max_local=25,
            )
            print(f"  SHAP summary -> {shap_paths['summary_plot']}")
        except Exception as e:
            print(f"  SHAP failed: {e}")
            shap_paths = {}

        lime_paths = {}
        if not args.skip_lime:
            print(f"  Generating LIME explanations (this may take a moment)...")
            try:
                lime_paths = explain_with_lime(
                    trained_pipeline=model,
                    X_train=np.array(X_train),
                    X_explain=np.array(ex),
                    feature_names=FEATURE_COLUMNS,
                    out_dir=explanations_dir,
                    model_name=model_name,
                    max_local=25,
                )
                print(f"  LIME summary -> {lime_paths.get('lime_summary_plot', 'n/a')}")
            except Exception as e:
                print(f"  LIME failed: {e}")
        else:
            print("  Skipping LIME (--skip-lime flag set).")

        results[model_name] = {
            "metrics":          metrics,
            "cross_validation": cv,
            "artifacts":        {**shap_paths, **lime_paths},
        }

    # ── Pairwise McNemar significance tests ──────────────────────────────────
    if len(trained_models) >= 2:
        print(f"\n{'=' * 55}")
        print("  Pairwise McNemar significance tests")
        print(f"{'=' * 55}")
        model_pairs = []
        names = list(trained_models.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                mc = compare_models_mcnemar(
                    trained_models[a], trained_models[b], X_test, y_test
                )
                sig = "SIGNIFICANT" if mc["significant"] else "not significant"
                print(f"  {a} vs {b}: p={mc['p_value']:.4f} ({sig})")
                model_pairs.append({
                    "model_a": a,
                    "model_b": b,
                    **mc,
                })
        results["mcnemar_tests"] = model_pairs

    # ── Save all metrics ──────────────────────────────────────────────────────
    save_metrics(results, metrics_dir / "metrics.json")
    print(f"\nDone.")
    print(f"    Metrics  -> {metrics_dir / 'metrics.json'}")
    print(f"    Models   -> {models_dir}")
    print(f"    Plots    -> {explanations_dir}")


if __name__ == "__main__":
    main()
