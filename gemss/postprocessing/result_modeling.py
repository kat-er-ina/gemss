"""
Use scikit's algorithms with nested CV to evaluate the performance of the discovered solutions.

This module provides functions to evaluate feature selection solutions using nested cross-validation
with various scikit-learn models. Unlike simple_regressions which only uses training data,
this module properly estimates generalization performance through nested CV.

Main functions:
- evaluate_with_nested_cv: Evaluate a single solution with nested CV
- evaluate_all_solutions: Evaluate multiple solutions (like solve_any_regression)
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from gemss.postprocessing.simple_regressions import detect_task
from gemss.utils.utils import myprint

# Constants from simple_regressions
MAX_ALLOWED_NAN_RATIO = 0.9
MIN_ALLOWED_SAMPLES = 15


def _get_classification_model_registry(
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Get registry of classification models with their default configurations.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping model names to model instances.
    """
    return {
        "logistic_l2": LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l2",
            solver="saga",
            scoring="roc_auc",
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "logistic_l1": LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "logistic_elasticnet": LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            scoring="roc_auc",
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(n_neighbors=3, weights="uniform"),
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight="balanced",
        ),
        "naive_bayes": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
    }


def _get_regression_model_registry(random_state: int | None = None) -> dict[str, Any]:
    """
    Get registry of regression models with their default configurations.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping model names to model instances.
    """
    return {
        "linear_l2": RidgeCV(cv=5),
        "linear_l1": LassoCV(cv=5, max_iter=2000, random_state=random_state),
        "linear_elasticnet": ElasticNetCV(
            cv=5,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            max_iter=2000,
            random_state=random_state,
        ),
        "svm": SVR(kernel="rbf", C=1.0, gamma="scale"),
        "knn": KNeighborsRegressor(n_neighbors=3, weights="uniform"),
        "xgboost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        ),
    }


def _get_available_models(task: str) -> list[str]:
    """
    Get list of available model names for a given task.

    Parameters
    ----------
    task : str
        Either 'classification' or 'regression'.

    Returns
    -------
    list[str]
        List of available model names.
    """
    if task == "classification":
        return list(_get_classification_model_registry().keys())
    else:
        return list(_get_regression_model_registry().keys())


def _create_model(model_name: str, task: str, random_state: int | None = None) -> Any:
    """
    Create a model instance based on name and task type.

    Parameters
    ----------
    model_name : str
        Name of the model from the registry.
    task : str
        Either 'classification' or 'regression'.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    estimator
        Scikit-learn estimator instance.

    Raises
    ------
    ValueError
        If model_name is not in the registry for the given task.
    """
    if task == "classification":
        registry = _get_classification_model_registry(random_state)
    else:
        registry = _get_regression_model_registry(random_state)

    if model_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Model '{model_name}' not found for {task}. Available: {available}"
        )

    return registry[model_name]


def _create_cv_splitter(
    outer_cv_folds: int | Literal["loo"],
    y: np.ndarray,
    task: str,
    random_state: int | None = None,
    stratify: np.ndarray | pd.Series | None = None,
):
    """
    Create appropriate CV splitter based on parameters.

    Parameters
    ----------
    outer_cv_folds : int or 'loo'
        Number of outer CV folds or 'loo' for leave-one-out.
    y : np.ndarray
        Target values (used for default stratification in classification).
    task : str
        Either 'classification' or 'regression'.
    random_state : int, optional
        Random seed for reproducibility.
    stratify : np.ndarray or pd.Series, optional
        Stratification vector. If provided, used for stratification regardless of task type.
        If None and task is classification, y is used for stratification.
        If None and task is regression, no stratification is applied.

    Returns
    -------
    cv_splitter
        Scikit-learn CV splitter instance.
    """
    if outer_cv_folds == "loo":
        return LeaveOneOut()

    # Determine stratification vector
    stratify_vector = None
    if stratify is not None:
        # Use provided stratification vector
        stratify_vector = stratify
    elif task == "classification":
        # Use response for classification
        stratify_vector = y
    # else: no stratification for regression (stratify_vector remains None)

    if stratify_vector is not None:
        return StratifiedKFold(
            n_splits=outer_cv_folds, shuffle=True, random_state=random_state
        )
    else:
        return KFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> dict[str, Any]:
    """
    Compute regression metrics for predictions (adapted from tabpfn_evaluation).

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
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
        "adjusted_r2": np.round(adj_r2, 3) if not np.isnan(adj_r2) else np.nan,
        "r2_score": np.round(r2, 3),
        "MSE": np.round(mse, 3),
        "RMSE": np.round(rmse, 3),
        "MAE": np.round(mae, 3),
        "MAPE": np.round(mape, 3) if not np.isnan(mape) else np.nan,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> dict[str, Any]:
    """
    Compute classification metrics for predictions (matching simple_regressions.py format).

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted class labels.
    n_features : int
        Number of features used in the model.

    Returns
    -------
    dict
        Dictionary with classification metrics.
    """
    n_samples = len(y_true)
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    # ROC-AUC (only for binary classification)
    if n_classes == 2:
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
    cm = confusion_matrix(y_true, y_pred)

    # Class-specific precision and recall
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

    metrics = {
        "f1_score": np.round(f1, 3),
        "balanced_accuracy": np.round(bal_acc, 3),
        "accuracy": np.round(acc, 3),
        "roc_auc": np.round(roc, 3) if not np.isnan(roc) else np.nan,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "class_distribution": class_dist,
    }

    # Add class-specific metrics
    metrics.update(precisions)
    metrics.update(recalls)

    # Add convenience metrics for binary classification (class 1 is typically positive class)
    # if n_classes == 2:
    #     # Assume classes are 0 and 1, or take the second unique value as positive
    #     positive_class = (
    #         unique_classes[1] if 1 in unique_classes else unique_classes[-1]
    #     )
    #     metrics["precision"] = metrics[f"precision_class_{positive_class}"]
    #     metrics["recall"] = metrics[f"recall_class_{positive_class}"]

    # # Add confusion matrix (both 2D and flattened for binary)
    # metrics["confusion_matrix"] = cm
    if n_classes == 2:
        metrics["confusion_matrix [TN, FP, FN, TP]"] = cm.ravel()

    return metrics


def evaluate_with_nested_cv(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_name: str = "logistic_l2",
    apply_scaling: Literal["standard", "minmax", None] = None,
    outer_cv_folds: int | Literal["loo"] = 5,
    inner_cv_folds: int = 5,
    random_state: int | None = None,
    verbose: bool = False,
    stratify: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single solution using nested cross-validation.

    This function performs nested cross-validation:
    - Outer loop: Splits data, evaluates generalization performance
    - Inner loop: Fits model with hyperparameter tuning (via CV in the model itself)

    Metrics are computed on the aggregated predictions from all outer CV folds,
    similar to the TabPFN evaluation approach.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target values.
    model_name : str, optional
        Name of the model to use. Default is 'logistic_l2'.
        Available models depend on task type (auto-detected from y).
    apply_scaling : Literal['standard', 'minmax', None], optional
        Feature scaling method. Options:
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - None: no scaling
        Default is None.
    outer_cv_folds : int or 'loo', optional
        Number of outer CV folds or 'loo' for leave-one-out. Default is 5.
    inner_cv_folds : int, optional
        Number of inner CV folds for hyperparameter tuning. Default is 5.
        Note: Some models (LogisticRegressionCV, etc.) use their own internal CV.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print progress information. Default is False.
    stratify : np.ndarray or pd.Series, optional
        Stratification vector for cross-validation splits. If provided, used for
        stratification regardless of task type. If None and task is classification,
        y is used for stratification. If None and task is regression, no stratification
        is applied. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'task': 'classification' or 'regression'
        - 'model': model name
        - 'cv_type': description of CV used
        - 'metrics': aggregated performance metrics
        - 'n_features': number of features

    Raises
    ------
    ValueError
        If model_name is not available for the detected task.
    """
    # Detect task type
    task = detect_task(y)

    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    n_features = X.shape[1]

    # Create CV splitter
    outer_cv = _create_cv_splitter(outer_cv_folds, y, task, random_state, stratify)
    cv_description = (
        f"Leave-One-Out" if outer_cv_folds == "loo" else f"{outer_cv_folds}-fold CV"
    )

    if verbose:
        print(f"Task: {task}")
        print(f"Model: {model_name}")
        print(f"Outer CV: {cv_description}")
        print(f"Features: {n_features}")
        print(f"Samples: {len(y)}")
        print("-" * 50)

    # Collect all predictions across folds
    all_y_true = []
    all_y_pred = []

    # Perform outer cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply scaling
        if apply_scaling == "standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif apply_scaling == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Create model instance (fresh for each fold)
        model = _create_model(model_name, task, random_state)

        # Fit model (inner CV happens inside LogisticRegressionCV, etc.)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Store predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        if verbose:
            fold_num = fold_idx + 1
            total_folds = len(y) if outer_cv_folds == "loo" else outer_cv_folds
            print(f"Fold {fold_num}/{total_folds} completed")

    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Compute aggregated metrics
    if task == "classification":
        metrics = _compute_classification_metrics(all_y_true, all_y_pred, n_features)
    else:
        metrics = _compute_regression_metrics(all_y_true, all_y_pred, n_features)

    result = {
        "task": task,
        "model": model_name,
        "cv_type": cv_description,
        "metrics": metrics,
        "n_features": n_features,
    }

    if verbose:
        print("-" * 50)
        print("Aggregated metrics:")
        print(pd.Series(metrics))

    return result


def evaluate_all_solutions(
    solutions: dict[str, list[str]],
    df: pd.DataFrame,
    response: pd.Series | np.ndarray,
    model_name: str | Literal["auto"] = "auto",
    apply_scaling: Literal["standard", "minmax", None] = None,
    outer_cv_folds: int | Literal["loo"] = 5,
    inner_cv_folds: int = 5,
    random_state: int | None = None,
    verbose: bool = False,
    use_markdown: bool = True,
    stratify: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple solutions using nested cross-validation.

    This function is analogous to solve_any_regression but uses proper nested CV
    to estimate generalization performance. Each solution is evaluated independently.

    Parameters
    ----------
    solutions : dict[str, list[str]]
        Dictionary mapping solution names to lists of feature names.
    df : pd.DataFrame
        Feature matrix with all features.
    response : pd.Series or np.ndarray
        Target values.
    model_name : str or 'auto', optional
        Name of the model to use. If 'auto', uses:
        - 'logistic_l2' for classification
        - 'linear_l2' for regression
        Default is 'auto'.
    apply_scaling : Literal['standard', 'minmax', None], optional
        Feature scaling method. Default is None.
    outer_cv_folds : int or 'loo', optional
        Number of outer CV folds or 'loo' for leave-one-out. Default is 5.
    inner_cv_folds : int, optional
        Number of inner CV folds (currently not used, models use internal CV). Default is 5.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print detailed information for each solution. Default is False.
    use_markdown : bool, optional
        Whether to use markdown formatting for verbose output. Default is True.
    stratify : np.ndarray or pd.Series, optional
        Stratification vector for cross-validation splits. If provided, used for
        stratification regardless of task type. If None and task is classification,
        response is used for stratification. If None and task is regression, no
        stratification is applied. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with solutions as index and metrics as columns.
        Each row contains the nested CV performance metrics for one solution.

    Notes
    -----
    - Solutions with too few samples (< MIN_ALLOWED_SAMPLES after dropping NaNs) are skipped.
    - Metrics are computed on aggregated predictions from all CV folds.
    - This provides a more realistic estimate of generalization performance compared to
      training-only evaluation in solve_any_regression.
    """
    # Convert response to pandas Series if numpy array
    if isinstance(response, np.ndarray):
        response = pd.Series(response)

    # Detect task
    task = detect_task(response)

    # Auto-select model if requested
    if model_name == "auto":
        model_name = "logistic_l2" if task == "classification" else "linear_l2"

    if verbose:
        myprint(
            msg=f"Evaluating solutions with nested CV using {model_name}",
            use_markdown=use_markdown,
            header=2,
        )
        cv_desc = (
            "Leave-One-Out" if outer_cv_folds == "loo" else f"{outer_cv_folds}-fold"
        )
        myprint(
            msg=f"Task: {task.capitalize()}, Outer CV: {cv_desc}",
            use_markdown=use_markdown,
        )
        myprint(msg="=" * 60, use_markdown=use_markdown)

    results = {}

    # Evaluate each solution
    for component, features in solutions.items():
        if verbose:
            myprint(
                msg=f"Evaluating **{component}**",
                use_markdown=use_markdown,
                header=3,
            )
            myprint(
                f"- {len(features)} features: {features}",
                use_markdown=use_markdown,
            )

        # Filter data to selected features and drop NaNs
        df_filtered = df[features].copy()
        df_filtered["response"] = response

        # Handle stratification vector if provided
        if stratify is not None:
            df_filtered["__stratify__"] = stratify

        df_filtered = df_filtered.dropna()
        y_filtered = df_filtered.pop("response")

        # Extract filtered stratification vector if it was provided
        stratify_filtered = None
        if stratify is not None:
            stratify_filtered = df_filtered.pop("__stratify__")

        # Check if we have enough samples
        if df_filtered.shape[0] < MIN_ALLOWED_SAMPLES:
            n_left = df_filtered.shape[0]
            myprint(
                msg=(
                    f"**Cannot evaluate {component}.** "
                    f"After dropping NaNs: {n_left} samples "
                    f"(need ≥ {MIN_ALLOWED_SAMPLES})."
                ),
                use_markdown=use_markdown,
            )
            continue

        # Check if we have enough samples for CV
        if outer_cv_folds != "loo" and df_filtered.shape[0] < outer_cv_folds:
            myprint(
                msg=(
                    f"**Cannot evaluate {component}.** "
                    f"Only {df_filtered.shape[0]} samples, need ≥ {outer_cv_folds} for {outer_cv_folds}-fold CV. "
                    f'Consider using fewer folds or "loo".'
                ),
                use_markdown=use_markdown,
            )
            continue

        # Evaluate with nested CV
        try:
            result = evaluate_with_nested_cv(
                X=df_filtered[features],
                y=y_filtered,
                model_name=model_name,
                apply_scaling=apply_scaling,
                outer_cv_folds=outer_cv_folds,
                inner_cv_folds=inner_cv_folds,
                random_state=random_state,
                verbose=False,  # We handle verbosity here
                stratify=stratify_filtered,
            )
            results[component] = result["metrics"]

            if verbose:
                print(f"OK - Completed: {result['cv_type']}")
                print()

        except Exception as e:
            myprint(
                msg=f"**Error evaluating {component}:** {str(e)}",
                use_markdown=use_markdown,
            )
            if verbose:
                import traceback

                traceback.print_exc()
            continue

        if verbose:
            myprint(msg="-" * 60, use_markdown=use_markdown)

    # Convert results to DataFrame
    if not results:
        myprint(
            msg="No solutions could be evaluated.",
            use_markdown=use_markdown,
        )
        return pd.DataFrame()

    metrics_df = pd.DataFrame.from_dict(results, orient="index")

    if verbose:
        myprint(
            msg=f"## Summary: Nested CV Results ({model_name}, {cv_desc})",
            use_markdown=use_markdown,
            header=2,
        )
        display_df = metrics_df.copy()
        # Drop complex nested objects for cleaner display, but keep all numeric metrics
        cols_to_drop = [
            c
            for c in display_df.columns
            if c
            in [
                "confusion_matrix",
                "confusion_matrix [TN, FP, FN, TP]",
                "class_distribution",
            ]
        ]
        if cols_to_drop:
            display_df = display_df.drop(columns=cols_to_drop)
        print(display_df.to_string())

        # Display confusion matrices separately if present
        if "confusion_matrix" in metrics_df.columns:
            print("\nConfusion Matrices:")
            for sol_name in metrics_df.index:
                cm = metrics_df.loc[sol_name, "confusion_matrix"]
                print(f"  {sol_name}:")
                print(f"    {cm}")
        print()

    return metrics_df
