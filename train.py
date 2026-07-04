import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

from src.preprocessing import load_data, split_data
from src.pipeline import (
    build_preprocessor,
    build_pipeline,
    get_models,
)
from src.evaluation import (
    cross_validate_model,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():

    # ======================================================
    # Load Data
    # ======================================================

    print("=" * 50)
    print("Loading data...")
    print("=" * 50)

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ======================================================
    # Build Preprocessor
    # ======================================================

    preprocessor = build_preprocessor(X_train)

    # ======================================================
    # Load Models
    # ======================================================

    models = get_models()

    pipelines = {}
    results = {}

    # ======================================================
    # Cross Validation
    # ======================================================

    print("\nRunning Cross Validation...\n")

    for name, model in models.items():

        print(f"Training: {name}")

        pipeline = build_pipeline(
            preprocessor,
            model,
        )

        cv_mean, cv_std, cv_scores = cross_validate_model(
            pipeline,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=5,
        )

        pipelines[name] = pipeline

        results[name] = {
            "CV Mean AUC": cv_mean,
            "CV Std": cv_std,
        }

    # ======================================================
    # Results
    # ======================================================

    results_df = (
        pd.DataFrame(results)
        .T
        .sort_values(
            by="CV Mean AUC",
            ascending=False,
        )
    )

    print("\nCross Validation Results\n")
    print(results_df.round(4))

    results_df.to_csv(
        RESULTS_DIR / "cv_results.csv",
        index=True,
    )

    # ======================================================
    # Best Model
    # ======================================================

    best_model_name = results_df.index[0]
    best_pipeline = pipelines[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    # ======================================================
    # Train Best Model
    # ======================================================

    print("\nTraining Best Pipeline...\n")

    best_pipeline.fit(
        X_train,
        y_train,
    )

    # ======================================================
    # Save Pipeline
    # ======================================================

    joblib.dump(
        best_pipeline,
        MODELS_DIR / "best_pipeline.pkl",
    )

    with open(RESULTS_DIR / "best_model.txt", "w") as f:
        f.write(best_model_name)

    # ======================================================
    # Evaluation
    # ======================================================

    print("\nEvaluating on Test Set...\n")

    metrics = evaluate_model(
        best_pipeline,
        X_test,
        y_test,
    )

    print("Metrics\n")

    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.4f}")

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    print("\nClassification Report\n")

    print_classification_report(
        best_pipeline,
        X_test,
        y_test,
    )

    # ======================================================
    # Plots
    # ======================================================

    plot_confusion_matrix(
        best_pipeline,
        X_test,
        y_test,
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )

    plot_roc_curve(
        best_pipeline,
        X_test,
        y_test,
        save_path=RESULTS_DIR / "roc_curve.png",
    )

    # ======================================================
    # Done
    # ======================================================

    print("\n" + "=" * 50)
    print("Training Completed Successfully")
    print("=" * 50)

    print(f"\nBest Model : {best_model_name}")
    print(f"Pipeline   : {MODELS_DIR / 'best_pipeline.pkl'}")
    print(f"Results    : {RESULTS_DIR}")


if __name__ == "__main__":
    main()