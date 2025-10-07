"""
Utility functions for feature selection project.
Includes:
- Sampling
- Metrics
- Data handling
"""

from IPython.display import display, Markdown
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
from plotly import graph_objects as go
import torch
import warnings
from sklearn.exceptions import ConvergenceWarning
from feature_selection.visualizations import (
    plot_elbo,
    plot_mu,
    plot_alpha,
    show_confusion_matrix,
    show_predicted_vs_actual_response,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def kldiv(p, q, eps=1e-10):
    """
    Compute KL divergence between two discrete probability distributions.

    Parameters
    ----------
    p : np.ndarray
        Array of probabilities (ground truth).
    q : np.ndarray
        Array of probabilities (approximation).
    eps : float, optional
        Small value to avoid log(0), default is 1e-10.

    Returns
    -------
    float
        KL divergence value.
    """
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)))


def batch_data(X, y, batch_size):
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
    n_components,
    sparsity,
    regularize,
    lambda_jaccard,
    regularization_threshold,
    n_iterations,
) -> None:
    """
    Print the optimization settings for the Bayesian Feature Selector.

    Parameters
    ----------
    n_components : int
        Number of mixture components (desiredsolutions).
    sparsity : int
        Desired sparsity level (number of features to select).
    regularize : bool
        Whether regularization is applied.
    lambda_jaccard : float
        Regularization strength for Jaccard similarity penalty.
    regularization_threshold : float
        Threshold for support computation.
    n_iterations : int
        Number of optimization iterations.
    """
    display(Markdown(f"#### Running Bayesian Feature Selector:"))
    display(Markdown(f"- searching for {n_components} solutions"))
    display(Markdown(f"- of desired sparsity: {sparsity}"))
    display(Markdown(f"- number of iterations: {n_iterations}"))

    if regularize:
        display(Markdown("- regularization parameters:"))
        display(Markdown(f" - Jaccard penalization: {lambda_jaccard}"))
        display(Markdown(f" - threshold for support: {regularization_threshold}"))
    else:
        display(Markdown("- no regularization"))

    return


def save_history(history, fname):
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


def recover_solutions(
    search_history,
    desired_sparsity: int,
    min_mu_threshold: float = 0.25,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """
    Recover solutions from the optimization history by identifying features that
    have significant mean values (mu) in the final iterations.

    Parameters:
    search_history: Dict[str, List]
        The history of the optimization process containing 'mu', 'var', and 'alpha'.
    desired_sparsity: int
        The number of most important features to identify for each component.
    min_mu_threshold: float
        The threshold for considering a feature as important based on its mu value.

    Returns:
    -------
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]
        A tuple containing:
        - A dictionary mapping each component (solution) to its identified features.
        - A dictionary with the final parameters (mu, var, alpha) for each component
        at the iteration where the features were selected.
    """
    n_components = len(search_history["mu"][0])
    n_features = len(search_history["mu"][0][0])

    # from each component, get the two features that zero out the last
    final_mu = np.zeros((n_components, n_features))
    final_var = np.zeros((n_components, n_features))
    final_alpha = np.zeros((n_components))
    final_iteration = np.zeros((n_components), dtype=int)

    # A dictionary to store the solutions found for each component
    solutions = {}

    # 'desired_sparsity' = the number of most important features we are interested in
    # ideally, this should be equal to SPARSITY but, in practice,
    # we might want to look at a few more features

    # If all features converge to zero at the end, we want to look at those that zero out the latest
    # So we set up the threshold and get at least  the minimal number of features that exceed the threshold
    n_last_nonzeros = desired_sparsity  # + 1
    for k in range(n_components):
        # for this component, get the 'n_last_nonzeros' features
        # whose absolute value is greater than 'min_mu_threshold'
        # while the others are less than min_mu_threshold
        features = []
        i = 1
        while len(features) < n_last_nonzeros:
            arr = np.array(
                search_history["mu"]
            )  # shape [n_iter, n_components, n_features]
            mu_traj = arr[-i, k, :]
            features = [
                f"feature_{j}"
                for j in range(len(mu_traj))
                if abs(mu_traj[j]) > min_mu_threshold
            ]
            i += 1

        display(Markdown(f"## Component {k}:"))
        display(
            Markdown(
                f"- Last {n_last_nonzeros}+ features with absolute value greater than {min_mu_threshold}:"
            )
        )
        # Organize these features by their absolute mu values
        top_features = pd.DataFrame(
            {
                "Feature": features,
                "Mu value": [mu_traj[int(f.split("_")[1])] for f in features],
            }
        )
        top_features["absolute_mu"] = np.abs(top_features["Mu value"])
        top_features = top_features.sort_values(by="absolute_mu", ascending=False)
        display(top_features[["Feature", "Mu value"]])

        # Take the top 'desired_sparsity' features according to their absolute mu values
        # and discard the rest
        top_features = top_features.head(desired_sparsity)
        features = top_features["Feature"].tolist()

        # Store the final parameters for this component at the iteration where features were selected
        final_iteration[k] = -(i - 1)
        final_mu[k] = np.array(search_history["mu"])[final_iteration[k], k, :]
        final_var[k] = np.array(search_history["var"])[final_iteration[k], k, :]
        final_alpha[k] = np.array(search_history["alpha"])[final_iteration[k], k]

        # Store the features found for this component
        solutions[f"component_{k}"] = features

    final_parameters = {
        "final iteration": final_iteration,  # the iteration from which the final parameters are taken
        "final mu": final_mu,
        "final var": final_var,
        "final alpha": final_alpha,
    }
    return (solutions, final_parameters)


def solve_with_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    show_cm_figure: bool = True,
):
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
    show_cm_figure : bool, optional
        Whether to display the confusion matrix figure using Plotly (otherwise it is just printed).
        Default is True.

    Returns
    -------
    None
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

    # Print results
    display(
        Markdown(
            f"### Logistic regression with {penalty.upper()} penalty - Performance on training set"
        )
    )
    display(Markdown(f"**Accuracy:** {accuracy_score(y, y_pred)}"))
    display(Markdown(f"**ROC-AUC:** {roc_auc_score(y, y_pred_prob)}"))
    display(Markdown("**Coefficients of the logistic regression model:**"))
    coefficients = (
        pd.Series(clf.named_steps["logisticregressioncv"].coef_[0], index=X.columns)
        .rename("Coefficient")
        .sort_values(ascending=False)
    )
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)
    display(coefficients)

    cm = confusion_matrix(y, y_pred)
    if show_cm_figure:
        # Show confusion matrix using Plotly
        show_confusion_matrix(confusion_matrix=cm)
    else:
        display(Markdown("**Confusion Matrix:**"))
        display(cm)

    return


def solve_with_linear_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    illustrate_predicted_vs_actual: bool = False,
):
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
    illustrate_predicted_vs_actual : bool, optional
        Whether to illustrate predicted vs actual response using Plotly. Default is False.

    Returns
    -------
    None
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

    display(
        Markdown(
            f"### Linear Regression with {model_name} Penalty - Performance on Training Set"
        )
    )
    display(Markdown(f"**R² Score:** {r2_score(y, y_pred)}"))
    display(Markdown(f"**Mean Squared Error:** {mean_squared_error(y, y_pred)}"))
    display(Markdown("Coefficients of the regression model:"))

    # Get coefficients, handle difference for RidgeCV (coef_ shape)
    if penalty == "l2":
        coefs = pipeline.named_steps["ridgecv"].coef_
    elif penalty == "l1":
        coefs = pipeline.named_steps["lassocv"].coef_
    elif penalty == "elasticnet":
        coefs = pipeline.named_steps["elasticnetcv"].coef_

    coefficients = pd.Series(coefs, index=X.columns)
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)
    display(coefficients)

    # Illustrate predicted vs actual
    if illustrate_predicted_vs_actual:
        show_predicted_vs_actual_response(y, y_pred)
    return


def show_regression_results_for_solutions(
    solutions: Dict[str, List[str]],
    df: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
):
    """
    Show regression results for each solution using the identified features.

    Parameters
    ----------
    solutions : Dict[str, List[str]]
        Dictionary mapping each component (solution) to its identified features.
    df : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Response vector (binary for classification, continuous for regression).
    penalty : str, optional
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l1".
    """
    is_binary = set(np.unique(y)) <= {0, 1}

    for component, features in solutions.items():
        display(Markdown(f"## Features of **{component}**"))

        if is_binary:
            solve_with_logistic_regression(X=df[features], y=y, penalty=penalty)
        else:
            solve_with_linear_regression(X=df[features], y=y, penalty=penalty)
        display(Markdown("------------------"))

    return


def display_features_overview(
    features_found: List[str],
    true_support_features: List[str],
    n_total_features: int,
) -> None:
    """
    Display an overview of features found vs true support features.

    Parameters
    ----------
    features_found : List[str]
        List of features found by the model.
    true_support_features : List[str]
        List of true support features.
    n_total_features : int
        Total number of features in the dataset.
    """
    missing_features = set(true_support_features) - features_found
    extra_features = features_found - set(true_support_features)

    display(Markdown(f"### All features: {n_total_features}"))
    display(
        Markdown(
            f"### True support features: {len(true_support_features)} ({len(true_support_features)/n_total_features:.1%})"
        )
    )
    display(Markdown(f"{sorted(true_support_features)}"))
    display(
        Markdown(
            f"### All features found: {len(features_found)} ({len(features_found)/n_total_features:.1%})"
        )
    )
    display(Markdown(f"{sorted(features_found)}"))
    display(Markdown(f"### Missing true support features: {len(missing_features)}"))
    display(Markdown(f"{sorted(missing_features)}"))
    display(
        Markdown(
            f"### Extra features found: {len(extra_features)} ({len(extra_features)/n_total_features:.1%})"
        )
    )
    display(Markdown(f"{sorted(extra_features)}"))
    return


def show_algorithm_progress(history: Dict[str, np.ndarray]) -> None:
    """
    Show the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights.

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing optimization history with keys 'elbo', 'mu', and 'alpha'.
        It is assumed that 'mu' has shape [n_iter, n_components, n_features].
        It is the output of the `optimize` method of `BayesianFeatureSelector`.
    """
    display(Markdown(f"## Algorithm progress: {len(history['elbo'])} iterations"))
    n_components = len(history["mu"][0])

    # Plot ELBO progress
    plot_elbo(history)

    # Plot mixture means trajectory for each component
    for k in range(n_components):
        plot_mu(history, component=k)

    # Plot mixture weights (alpha)
    plot_alpha(history)
    return
