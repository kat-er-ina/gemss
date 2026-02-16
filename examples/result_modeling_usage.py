"""
Example usage of the result_modeling module for evaluating feature selection solutions.

This script demonstrates how to use nested cross-validation to evaluate discovered
solutions from the GEMSS feature selection algorithm.
"""

import numpy as np
import pandas as pd
from gemss.postprocessing.result_modeling import (
    evaluate_with_nested_cv,
    evaluate_all_solutions,
    _get_available_models,
)


# Example 1: Evaluate a single solution
def example_single_solution():
    """Evaluate a single feature set with nested CV."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Evaluate a Single Solution")
    print("=" * 70)

    # Load your data
    # df = pd.read_csv('your_data.csv')
    # y = df['response_column']
    # X = df.drop('response_column', axis=1)

    # For demonstration, create synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])

    # Select a subset of features (e.g., from GEMSS solution)
    selected_features = [f"feature_{i}" for i in [0, 1, 5, 7, 12]]
    X_selected = df[selected_features]

    # Evaluate with nested CV
    result = evaluate_with_nested_cv(
        X=X_selected,
        y=y,
        model_name="logistic_l2",  # or 'random_forest', 'gradient_boosting', etc.
        apply_scaling="standard",
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    print("\nFinal metrics:")
    print(pd.Series(result["metrics"]))


# Example 2: Evaluate multiple solutions (from GEMSS)
def example_multiple_solutions():
    """Evaluate multiple feature selection solutions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Evaluate Multiple Solutions")
    print("=" * 70)

    # Load your data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=30, n_informative=15, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(30)])

    # Define solutions (these would typically come from GEMSS recovery)
    # Example: solutions = recover_solutions(history, ...)
    solutions = {
        "component_1": [f"feature_{i}" for i in [0, 1, 2, 5, 8]],
        "component_2": [f"feature_{i}" for i in [3, 4, 6, 9, 12, 15]],
        "component_3": [f"feature_{i}" for i in [10, 11, 13, 14]],
        "full_solution": [f"feature_{i}" for i in range(15)],
    }

    # Evaluate all solutions
    results_df = evaluate_all_solutions(
        solutions=solutions,
        df=df,
        response=y,
        model_name="auto",  # Auto-selects logistic_l2 for classification
        apply_scaling="standard",
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
        use_markdown=False,
    )

    print("\n" + "=" * 70)
    print("COMPARISON OF SOLUTIONS")
    print("=" * 70)
    # Select available columns for display
    display_cols = [
        c
        for c in ["n_samples", "accuracy", "balanced_accuracy", "roc_auc", "f1_score"]
        if c in results_df.columns
    ]
    print(results_df[display_cols])


# Example 3: Compare different models
def example_compare_models():
    """Compare different models for the same solution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Compare Different Models")
    print("=" * 70)

    # Generate data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])

    # Single solution
    selected_features = [f"feature_{i}" for i in range(10)]
    X_selected = df[selected_features]

    # Test different models
    models_to_test = [
        "logistic_l1",
        "logistic_l2",
        "random_forest",
        "xgboost",
    ]

    comparison = []
    for model_name in models_to_test:
        print(f"\nEvaluating with {model_name}...")
        result = evaluate_with_nested_cv(
            X=X_selected,
            y=y,
            model_name=model_name,
            apply_scaling="standard",
            outer_cv_folds=5,
            random_state=42,
            verbose=False,
        )
        comparison.append(
            {
                "model": model_name,
                "accuracy": result["metrics"]["accuracy"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "f1_score": result["metrics"]["f1_score"],
                "precision": result["metrics"][
                    "precision"
                ],  # Binary classification convenience metric
                "recall": result["metrics"][
                    "recall"
                ],  # Binary classification convenience metric
                "roc_auc": result["metrics"]["roc_auc"],
            }
        )

    comparison_df = pd.DataFrame(comparison)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(
        "\nBest model by F1 score:",
        comparison_df.loc[comparison_df["f1_score"].idxmax(), "model"],
    )


# Example 4: Leave-One-Out for small datasets
def example_leave_one_out():
    """Use leave-one-out CV for small datasets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Leave-One-Out CV (Small Dataset)")
    print("=" * 70)

    # Small dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=40, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

    solutions = {
        "solution_1": [f"feature_{i}" for i in [0, 1, 2, 3]],
        "solution_2": [f"feature_{i}" for i in [4, 5, 6, 7, 8]],
    }

    # Use leave-one-out CV
    results_df = evaluate_all_solutions(
        solutions=solutions,
        df=df,
        response=y,
        model_name="logistic_l2",
        outer_cv_folds="loo",  # Leave-One-Out
        random_state=42,
        verbose=True,
        use_markdown=False,
    )

    print("\nResults with LOO:")
    # Select available columns
    display_cols = [
        c
        for c in ["n_samples", "accuracy", "balanced_accuracy"]
        if c in results_df.columns
    ]
    print(results_df[display_cols])


# Example 5: Regression task
def example_regression():
    """Evaluate regression solutions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Regression Task")
    print("=" * 70)

    # Generate regression data
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=200, n_features=25, n_informative=10, noise=15.0, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(25)])

    solutions = {
        "linear_solution": [f"feature_{i}" for i in [0, 1, 2, 5, 8, 12]],
        "complex_solution": [f"feature_{i}" for i in range(15)],
    }

    # Evaluate regression solutions
    results_df = evaluate_all_solutions(
        solutions=solutions,
        df=df,
        response=y,
        model_name="auto",  # Auto-selects linear_l2 for regression
        apply_scaling="standard",
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
        use_markdown=False,
    )

    print("\nRegression results:")
    # Select available columns
    display_cols = [
        c
        for c in ["n_samples", "n_features", "r2_score", "adjusted_r2", "RMSE", "MAE"]
        if c in results_df.columns
    ]
    print(results_df[display_cols])


# Example 6: List available models
def example_available_models():
    """Show all available models."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Available Models")
    print("=" * 70)

    class_models = _get_available_models("classification")
    reg_models = _get_available_models("regression")

    print("\nCLASSIFICATION MODELS:")
    for i, model in enumerate(class_models, 1):
        print(f"  {i}. {model}")

    print("\nREGRESSION MODELS:")
    for i, model in enumerate(reg_models, 1):
        print(f"  {i}. {model}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RESULT_MODELING USAGE EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_available_models()
    example_single_solution()
    example_multiple_solutions()
    example_compare_models()
    example_leave_one_out()
    example_regression()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
