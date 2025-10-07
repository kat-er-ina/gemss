"""
This module provides functions to recover, display and analyze solutions
from the optimization history of the Bayesian feature selection algorithm.
"""

from typing import Dict, List, Tuple, Literal, Any
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from feature_selection.visualizations import plot_elbo, plot_mu, plot_alpha
from feature_selection.utils import (
    solve_with_logistic_regression,
    solve_with_linear_regression,
)


def recover_solutions(
    search_history: Dict[str, List],
    desired_sparsity: int,
    min_mu_threshold: float = 0.25,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
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
    verbose: bool, optional
        Whether to print detailed information about the recovered solutions. Default is True.

    Returns:
    -------
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, pd.DataFrame]]
        A tuple containing:
        - A dictionary mapping each component (solution) to its identified features.
        - A dictionary with the final parameters (mu, var, alpha) for each component
        at the iteration where the features were selected.
        - A dictionary mapping each component to a DataFrame of all features that
        exceeded the min_mu_threshold in the last iterations, along with their mu values.
    """
    n_components = len(search_history["mu"][0])
    n_features = len(search_history["mu"][0][0])

    # from each component, get the two features that zero out the last
    final_mu = np.zeros((n_components, n_features))
    final_var = np.zeros((n_components, n_features))
    final_alpha = np.zeros((n_components))
    final_iteration = np.zeros((n_components), dtype=int)

    # A dictionary to store the solutions found for each component
    # all features with mu > min_mu_threshold in last iterations
    full_nonzero_solutions = {}
    # top 'desired_sparsity' features for each component
    solutions = {}

    # In cases when all features converge to zero at the end, we want to look at those that
    # zero out the latest. So, we set up the threshold and get at least
    # the minimal number of features that exceed the threshold
    for k in range(n_components):
        # for this component, get the 'desired_sparsity' features
        # whose absolute value is greater than 'min_mu_threshold'
        # while the others are less than min_mu_threshold
        features = []
        i = 1
        while len(features) < desired_sparsity:
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
        if verbose:
            display(Markdown(f"## Component {k}:"))
            display(
                Markdown(
                    f"- Last {desired_sparsity}+ features with absolute value greater than {min_mu_threshold}:"
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
        top_features = top_features.sort_values(by="absolute_mu", ascending=False).drop(
            columns=["absolute_mu"]
        )
        full_nonzero_solutions[f"component_{k}"] = top_features
        if verbose:
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
    return (solutions, final_parameters, full_nonzero_solutions)


def show_algorithm_progress(
    history: Dict[str, np.ndarray],
    plot_elbo_progress: bool = True,
    plot_mu_progress: bool = True,
    plot_alpha_progress: bool = True,
) -> None:
    """
    Show the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights.

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing optimization history with keys 'elbo', 'mu', and 'alpha'
        (fewer if they are not required to be plotted).
        It is assumed that 'mu' has shape [n_iter, n_components, n_features].
        It is the output of the `optimize` method of `BayesianFeatureSelector`.
    plot_elbo_progress : bool, optional
        Whether to plot the ELBO progress. Default is True.
    plot_mu_progress : bool, optional
        Whether to plot the mixture means (mu) trajectory. Default is True.
    plot_alpha_progress : bool, optional
        Whether to plot the mixture weights (alpha) progress. Default is True. Default is True.
    """
    display(Markdown(f"### Algorithm progress:"))
    n_components = len(history["mu"][0])

    # Plot ELBO progress
    if plot_elbo_progress:
        plot_elbo(history)

    # Plot mixture means trajectory for each component
    if plot_mu_progress:
        for k in range(n_components):
            plot_mu(history, component=k)

    # Plot mixture weights (alpha)
    if plot_alpha_progress:
        plot_alpha(history)
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
