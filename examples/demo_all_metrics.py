"""
Demonstration of all available metrics in result_modeling module.
Shows that all metrics from simple_regressions.py are now included.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from gemss.postprocessing.result_modeling import evaluate_with_nested_cv

print("=" * 70)
print("CLASSIFICATION METRICS DEMONSTRATION")
print("=" * 70)

# Generate binary classification data
X_class, y_class = make_classification(
    n_samples=150,
    n_features=10,
    n_informative=6,
    n_classes=2,
    random_state=42,
)
X_class_df = pd.DataFrame(X_class, columns=[f"feature_{i}" for i in range(10)])

# Evaluate
result_class = evaluate_with_nested_cv(
    X=X_class_df,
    y=y_class,
    model_name="logistic_l2",
    apply_scaling="standard",
    outer_cv_folds=5,
    random_state=42,
    verbose=False,
)

print("\nAll Classification Metrics Available:")
print("-" * 70)
metrics = result_class["metrics"]
for key, value in metrics.items():
    if key == "confusion_matrix":
        print(f"  {key}:")
        print(f"    {value}")
    elif key == "confusion_matrix [TN, FP, FN, TP]":
        print(f"  {key}: {value}")
    elif key == "class_distribution":
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("Key Classification Metrics (matching simple_regressions.py):")
print("=" * 70)
print(f"  F1 Score:              {metrics['f1_score']} (MOST IMPORTANT)")
print(f"  Accuracy:              {metrics['accuracy']}")
print(f"  Balanced Accuracy:     {metrics['balanced_accuracy']}")
print(f"  ROC-AUC:               {metrics['roc_auc']}")
print(f"  Precision (Class 0):   {metrics['precision_class_0']}")
print(f"  Precision (Class 1):   {metrics['precision_class_1']}")
print(f"  Precision (Binary):    {metrics['precision']} (convenience metric)")
print(f"  Recall (Class 0):      {metrics['recall_class_0']}")
print(f"  Recall (Class 1):      {metrics['recall_class_1']}")
print(f"  Recall (Binary):       {metrics['recall']} (convenience metric)")
print(f"  Confusion Matrix:      Available (2D and flattened)")

print("\n" + "=" * 70)
print("REGRESSION METRICS DEMONSTRATION")
print("=" * 70)

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=150,
    n_features=10,
    n_informative=6,
    noise=10.0,
    random_state=42,
)
X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])

# Evaluate
result_reg = evaluate_with_nested_cv(
    X=X_reg_df,
    y=y_reg,
    model_name="linear_l2",
    apply_scaling="standard",
    outer_cv_folds=5,
    random_state=42,
    verbose=False,
)

print("\nAll Regression Metrics Available:")
print("-" * 70)
metrics_reg = result_reg["metrics"]
for key, value in metrics_reg.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("Key Regression Metrics (matching simple_regressions.py):")
print("=" * 70)
print(f"  R² Score:              {metrics_reg['r2_score']}")
print(f"  Adjusted R²:           {metrics_reg['adjusted_r2']}")
print(f"  MSE:                   {metrics_reg['MSE']}")
print(f"  RMSE:                  {metrics_reg['RMSE']}")
print(f"  MAE:                   {metrics_reg['MAE']}")
print(f"  MAPE:                  {metrics_reg['MAPE']}")
print(f"  Number of Features:    {metrics_reg['n_features']}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("All metrics from simple_regressions.py are now available in nested CV!")
print("\nClassification: F1, accuracy, precision/recall per class, confusion matrix")
print("Regression: R², adjusted R², MSE, RMSE, MAE, MAPE")
print("Plus: Proper generalization estimates through nested cross-validation")
print("=" * 70)
