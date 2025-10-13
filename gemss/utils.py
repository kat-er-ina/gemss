"""
Utility functions for feature selection project.
Includes:
- Sampling
- Metrics
- Data handling
"""

from IPython.display import display, Markdown
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from gemss.visualizations import (
    show_confusion_matrix,
    show_predicted_vs_actual_response,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def batch_data(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample a batch of data.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    batch_size : int
        Number of samples to select.

    Returns
    -------
    X_batch : np.ndarray
        Batch of data features.
    y_batch : np.ndarray
        Batch of data targets.
    """
    n = X.shape[0]
    idx = np.random.choice(n, batch_size, replace=False)
    return X[idx], y[idx]


def print_optimization_setting(
    n_components: int,
    regularize: bool,
    lambda_jaccard: float,
    regularization_threshold: float,
    n_iterations: int,
    prior_settings: Dict[str, Any],
) -> None:
    """
    Print the optimization settings for the Bayesian Feature Selector.

    Parameters
    ----------
    n_components : int
        Number of mixture components (desired solutions).
    regularize : bool
        Whether regularization is applied.
    lambda_jaccard : float
        Regularization strength for Jaccard similarity penalty.
    regularization_threshold : float
        Threshold for support computation.
    n_iterations : int
        Number of optimization iterations.
    prior_settings : Dict[str, Any]
        Dictionary containing prior settings such as prior name and parameters.

    Returns
    -------
    None
    """
    display(Markdown(f"#### Running Bayesian Feature Selector:"))
    display(Markdown(f"- desired number of solutions: {n_components}"))
    display(Markdown(f"- number of iterations: {n_iterations}"))

    if regularize:
        display(Markdown("- regularization parameters:"))
        display(Markdown(f"  - Jaccard penalization: {lambda_jaccard}"))
        display(Markdown(f"  - threshold for support: {regularization_threshold}"))
    else:
        display(Markdown("- no regularization"))

    display(Markdown("##### Algorithm settings:"))

    prior_name = prior_settings.get("prior_name", "N/A")
    prior_settings_to_display = {"prior name": prior_name}
    if prior_name == "StructuredSpikeAndSlabPrior":
        prior_settings_to_display.update(
            {
                "prior_sparsity": prior_settings.get("prior_sparsity", "N/A"),
                "var_slab": prior_settings.get("var_slab", "N/A"),
                "var_spike": prior_settings.get("var_spike", "N/A"),
            }
        )
    if prior_name == "SpikeAndSlabPrior":
        prior_settings_to_display.update(
            {
                "var_slab": prior_settings.get("var_slab", "N/A"),
                "var_spike": prior_settings.get("var_spike", "N/A"),
                "weight_slab": prior_settings.get("weight_slab", "N/A"),
                "weight_spike": prior_settings.get("weight_spike", "N/A"),
            }
        )
    elif prior_name == "StudentTPrior":
        prior_settings_to_display.update(
            {
                "student_df": prior_settings.get("student_df", "N/A"),
                "student_scale": prior_settings.get("student_scale", "N/A"),
            }
        )

    for key, value in prior_settings_to_display.items():
        display(Markdown(f" - {key.lower()}: {value}"))

    return


def save_history(
    history: Dict[str, Any],
    fname: str,
) -> None:
    """
    Save optimization history to a file using pickle.

    Parameters
    ----------
    history : dict
        Dictionary containing optimization history.
    fname : str
        Output file path.

    Returns
    -------
    None
    """
    import pickle

    with open(fname, "wb") as f:
        pickle.dump(history, f)


def solve_with_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    verbose: bool = True,
    show_cm_figure: bool = True,
) -> Dict[str, Any]:
    """
    Solve a logistic regression problem with the specified penalty.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Binary response vector.
    penalty : str
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l2".
    verbose : bool, optional
        Whether to print detailed metrics, coefficients and confusion matrix.
        Default is False.
    show_cm_figure : bool, optional
        Whether to display the confusion matrix figure using Plotly
        (otherwise it is just printed). Default is True.
        Only applicable if verbose is True.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing performance metrics and model coefficients.
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty=penalty,
            solver="saga",
            scoring="roc_auc",
            max_iter=2000,
            random_state=42,
            refit=True,
            class_weight="balanced",
        ),
    )

    # Solve the full problem
    # Ignore warnings about convergence for this demo
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X, y)

    # Predictions and evaluation
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)[:, 1]

    stats = {
        "accuracy": np.round(accuracy_score(y, y_pred), 3),
        "balanced_accuracy": np.round(balanced_accuracy_score(y, y_pred), 3),
        "roc_auc": np.round(roc_auc_score(y, y_pred_prob), 3),
        "f1_score": np.round(f1_score(y, y_pred, average="weighted"), 3),
        "recall_class_0": np.round(
            np.sum((y_pred == 0) & (y == 0)) / np.sum(y == 0), 3
        ),
        "recall_class_1": np.round(
            np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1), 3
        ),
    }

    # Print results
    if verbose:
        display(
            Markdown(
                f"### Logistic regression with {penalty.upper()} penalty - performance on training set"
            )
        )
        display(Markdown(f"**Accuracy:** {stats['accuracy']}"))
        display(Markdown(f"**Balanced Accuracy:** {stats['balanced_accuracy']}"))
        display(Markdown(f"**ROC-AUC:** {stats['roc_auc']}"))
        display(Markdown(f"**Balanced F1 Score:** {stats['f1_score']}"))
        display(Markdown(f"**Recall on class 0:** {stats['recall_class_0']}"))
        display(Markdown(f"**Recall on class 1:** {stats['recall_class_1']}"))

        display(Markdown("**Coefficients of the logistic regression model:**"))

    coefficients = (
        pd.Series(clf.named_steps["logisticregressioncv"].coef_[0], index=X.columns)
        .rename("Coefficient")
        .sort_values(ascending=False)
    )
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)
    stats["n_nonzero_coefficients"] = len(coefficients)
    stats["nonzero_coefficients"] = coefficients

    if verbose:
        display(stats["nonzero_coefficients"])

    if verbose:
        cm = confusion_matrix(y, y_pred)
        if show_cm_figure:
            # Show confusion matrix using Plotly
            show_confusion_matrix(confusion_matrix=cm)
        else:
            display(Markdown("**Confusion Matrix:**"))
            display(cm)

    return stats


def solve_with_linear_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    verbose: bool = True,
    illustrate_predicted_vs_actual: bool = False,
) -> Dict[str, Any]:
    """
    Solve a linear regression problem with the specified penalty.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Continuous response vector.
    penalty : str
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l2".
    show_cm_figure : bool, optional
        Whether to display the confusion matrix figure using Plotly (otherwise it is just printed).
    verbose : bool, optional
        Whether to print detailed metrics and coefficients.
    illustrate_predicted_vs_actual : bool, optional
        Whether to illustrate predicted vs actual response using Plotly. Default is False.
        Only applicable if verbose is True.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing performance metrics and model coefficients.
    """
    # Choose model based on penalty
    if penalty == "l1":
        model = LassoCV(cv=5, random_state=42, max_iter=2000)
        model_name = "Lasso (L1)"
    elif penalty == "l2":
        model = RidgeCV(cv=5)
        model_name = "Ridge (L2)"
    elif penalty == "elasticnet":
        model = ElasticNetCV(cv=5, random_state=42, max_iter=2000)
        model_name = "ElasticNet"
    else:
        raise ValueError("Penalty must be 'l1', 'l2', or 'elasticnet'.")

    pipeline = make_pipeline(StandardScaler(), model)

    # Fit the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipeline.fit(X, y)

    # Predictions and evaluation
    y_pred = pipeline.predict(X)

    stats = {
        "r2_score": np.round(r2_score(y, y_pred), 3),
        "mean_squared_error": np.round(mean_squared_error(y, y_pred), 3),
    }
    # Print results
    if verbose:
        display(
            Markdown(
                f"### Linear regression with {model_name} penalty - performance on training set"
            )
        )
        display(Markdown(f"**R² Score:** {stats['r2_score']}"))
        display(Markdown(f"**Mean Squared Error:** {stats['mean_squared_error']}"))

    # Get coefficients, handle difference for RidgeCV (coef_ shape)
    if penalty == "l2":
        coefs = pipeline.named_steps["ridgecv"].coef_
    elif penalty == "l1":
        coefs = pipeline.named_steps["lassocv"].coef_
    elif penalty == "elasticnet":
        coefs = pipeline.named_steps["elasticnetcv"].coef_

    coefficients = pd.Series(coefs, index=X.columns)
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)
    stats["n_nonzero_coefficients"] = len(coefficients)
    stats["nonzero_coefficients"] = coefficients

    if verbose:
        display(Markdown("Coefficients of the regression model:"))
        display(stats["nonzero_coefficients"])

        if illustrate_predicted_vs_actual:
            show_predicted_vs_actual_response(y, y_pred)

    return stats
