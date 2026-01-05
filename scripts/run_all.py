import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, Paths
from src.data_loader import load_dataset
from src.evaluate import evaluate_binary, save_metrics
from src.train_model import train
from src.explain import explain_with_shap


def main():
    parser = argparse.ArgumentParser(description="Run Phase-1 pipeline: train, evaluate, explain.")
    parser.add_argument("--data", type=str, default="data/student_data.csv", help="Path to CSV dataset.")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory root.")
    args = parser.parse_args()

    # Resolve output dirs
    paths = Paths()
    out_root = Path(args.out)
    models_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    explanations_dir = out_root / "explanations"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_dataset(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Background sample for SHAP (use subset of train)
    bg = X_train.sample(n=min(600, len(X_train)), random_state=42)
    ex = X_test.sample(n=min(300, len(X_test)), random_state=42)

    results = {}

    for model_name in ["logreg", "rf"]:
        trained = train(model_name, X_train, y_train)
        model = trained.pipeline

        metrics = evaluate_binary(model, X_test, y_test)

        explain_paths = explain_with_shap(
            trained_pipeline=model,
            X_background=bg,
            X_explain=ex,
            feature_names=FEATURE_COLUMNS,
            out_dir=explanations_dir,
            model_name=model_name,
            max_local=25,
        )

        results[model_name] = {
            "metrics": metrics,
            "artifacts": explain_paths,
        }

    save_metrics(results, metrics_dir / "metrics.json")
    print(f"✅ Done. Metrics saved to {metrics_dir / 'metrics.json'}")
    print(f"📊 SHAP plots + local explanations in {explanations_dir}")


if __name__ == "__main__":
    main()
