from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabpfn import TabPFNClassifier, TabPFNRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import shap


def detect_task(y, n_class_threshold=10):
    """
    Detect if the task should be treated as classification or regression.

    Parameters
    ----------
    y : array-like
        Target vector.
    n_class_threshold : int, optional
        Maximum number of unique integer values to consider as classification.

    Returns
    -------
    str
        "classification" or "regression"
    """
    y = np.asarray(y)
    unique = np.unique(y)
    if (pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y)) and len(
        unique
    ) <= n_class_threshold:
        return "classification"
    return "regression"


def regression_metrics(y_true, y_pred, n_features):
    """
    Compute regression metrics for predictions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    n_features : int
        Number of features used in the model.

    Returns
    -------
    dict
        Regression performance metrics.
    """
    residuals = y_true - y_pred
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = (
        1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        if n_samples > n_features + 1
        else np.nan
    )
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = (
        np.mean(np.abs(residuals / np.where(y_true != 0, y_true, 1e-8))) * 100
        if not np.any(y_true == 0)
        else np.nan
    )
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "r2_score": np.round(r2, 3),
        "adjusted_r2": np.round(adj_r2, 3) if not np.isnan(adj_r2) else np.nan,
        "MSE": np.round(mse, 3),
        "RMSE": np.round(rmse, 3),
        "MAE": np.round(mae, 3),
        "MAPE": np.round(mape, 3) if not np.isnan(mape) else np.nan,
    }


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics for predictions.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    dict
        Dictionary with relevant classification metrics.
    """
    n_samples = len(y_true)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        try:
            roc = roc_auc_score(y_true, y_pred)
        except Exception:
            roc = np.nan
    else:
        roc = np.nan
    class_dist = {
        f"class_{v}": np.round(np.mean(y_true == v), 3) for v in unique_classes
    }
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precisions = {
        f"precision_class_{v}": np.round(
            precision_score(y_true, y_pred, pos_label=v, zero_division=0), 3
        )
        for v in unique_classes
    }
    recalls = {
        f"recall_class_{v}": np.round(
            recall_score(y_true, y_pred, pos_label=v, zero_division=0), 3
        )
        for v in unique_classes
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "n_samples": int(n_samples),
        "class_distribution": class_dist,
        "accuracy": np.round(acc, 3),
        "balanced_accuracy": np.round(bal_acc, 3),
        "roc_auc": np.round(roc, 3) if not np.isnan(roc) else np.nan,
        "f1_score": np.round(f1, 3),
        "confusion_matrix": cm,
    }
    metrics.update(precisions)
    metrics.update(recalls)
    return metrics


def _compute_shap_explanation(model, X, feature_names=None):
    """
    Compute SHAP values for a fitted model and feature matrix X.

    Parameters
    ----------
    model : fitted estimator
        Model that implements a `predict` method.
    X : np.ndarray
        Feature matrix.
    feature_names : list of str, optional
        Feature names; used for output.

    Returns
    -------
    shap_importance : dict
        Mean absolute SHAP values per feature.
    shap_values : array
        Raw SHAP values for each sample/feature.
    """
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    shap_importance = dict(zip(feature_names, mean_shap))
    return shap_importance, shap_values


def tabpfn_evaluate(
    X,
    y,
    apply_scaling: Literal["standard", "minmax", None] = None,
    outer_cv_folds=5,
    inner_cv_folds=3,
    tabpfn_kwargs=None,
    random_state=None,
    verbose: bool = False,
    explain: bool = False,
    shap_sample_size: int = None,  # optional max number of samples to use for SHAP explanations
):
    """
    Evaluate TabPFN Classifier or Regressor using nested cross-validation.
    Optionally standardizes features and computes SHAP (Shapley) explanations.
    Metrics are inspired by gemss.diagnostics.simple_regressions.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target values.
    apply_scaling: Literal["standard", "minmax", None] = None
        Whether to apply feature scaling. Options are:
        - "standard": apply standard scaling.
        - "minmax": apply Min-Max scaling.
        - None: do not apply any scaling.
    outer_cv_folds : int, optional
        Number of outer cross-validation folds (default: 5).
    inner_cv_folds : int, optional
        For future hyper-parameter search (currently unused).
    tabpfn_kwargs : dict, optional
        Custom arguments for TabPFNClassifier/Regressor.
    random_state : int, optional
        Random seed.
    verbose : bool, optional
        If True, prints metrics per fold.
    explain : bool, optional
        If True, computes SHAP feature explanations.
    shap_sample_size : int, optional
        If set, subsample up to this many samples for SHAP explanations.

    Returns
    -------
    result : dict
        Dictionary with task, average scores, scores per fold, and optionally SHAP explanations.
    """
    task = detect_task(y)
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    outer_cv = (
        StratifiedKFold(
            n_splits=outer_cv_folds, shuffle=True, random_state=random_state
        )
        if task == "classification"
        else KFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    )

    all_scores = []
    explanations = []
    rs = np.random.RandomState(random_state)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if apply_scaling == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif apply_scaling == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if task == "classification":
            model = TabPFNClassifier(
                **(tabpfn_kwargs or {}),
                balance_probabilities=True,
                ignore_pretraining_limits=True,
            )
        else:
            model = TabPFNRegressor(
                **(tabpfn_kwargs or {}),
                ignore_pretraining_limits=True,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        n_features = X_train.shape[1]
        if task == "classification":
            scores = classification_metrics(y_test, y_pred)
            scores["n_features"] = n_features
        else:
            scores = regression_metrics(y_test, y_pred, n_features=n_features)

        all_scores.append(scores)
        if verbose:
            print(f"Fold {fold+1}/{outer_cv_folds}\n{pd.Series(scores)}\n")

        if explain:
            # Subsample for SHAP if requested
            if shap_sample_size is not None and X_train.shape[0] > shap_sample_size:
                idx = rs.choice(X_train.shape[0], shap_sample_size, replace=False)
                background = X_train[idx, :]
            else:
                background = X_train
            shap_importance, _ = _compute_shap_explanation(
                model, background, feature_names=feature_names
            )
            explanations.append(shap_importance)

    # Aggregate (average except for confusion matrices)
    keys = [k for k in all_scores[0].keys() if k != "confusion_matrix"]
    avg_scores = {k: np.mean([s[k] for s in all_scores]) for k in keys}
    if "confusion_matrix" in all_scores[0]:
        avg_scores["confusion_matrix_sum"] = np.sum(
            [s["confusion_matrix"] for s in all_scores], axis=0
        )

    result = {"task": task, "average_scores": avg_scores, "fold_scores": all_scores}
    if explain:
        result["shap_explanations_per_fold"] = explanations
    return result
