"""
This module provides functions to recover, display and analyze solutions
from the optimization history of the Bayesian feature selection algorithm.
"""

from typing import Dict, List, Tuple, Literal, Any, Optional, Union
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from gemss.diagnostics.visualizations import plot_elbo, plot_mu, plot_alpha
from gemss.diagnostics.simple_regressions import (
    solve_with_logistic_regression,
    solve_with_linear_regression,
)


def get_long_solutions_df(
    full_nonzero_solutions: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert the full nonzero solutions dictionary into a long-format DataFrame.

    Parameters
    ----------
    full_nonzero_solutions : Dict[str, pd.DataFrame]
        Dictionary mapping each component (solution) to a DataFrame containing
        features that exceeded the min_mu_threshold in the last iterations,
        with columns ['Feature', 'Mu value'].

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame where each column corresponds to a component and contains
        all the features that were considered nonzero for that component, ordered by the absolute
        value of their mu values. Missing values are filled with NaN.

    Notes
    -----
    This function displays a markdown header "## Full long solutions" as a side effect.
    """
    display(Markdown("## Full long solutions"))
    max_len = max(
        [
            len(full_solution["Feature"])
            for _, full_solution in full_nonzero_solutions.items()
        ]
    )
    df_full_solutions = pd.DataFrame(index=range(max_len))

    for component, full_solution in full_nonzero_solutions.items():
        df_full_solutions[component] = pd.Series(full_solution["Feature"]).reset_index(
            drop=True
        )
    return df_full_solutions


def recover_solutions(
    search_history: Dict[str, List[Any]],
    desired_sparsity: int,
    min_mu_threshold: float = 0.25,
    verbose: bool = True,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, Union[np.ndarray, int]],
    Dict[str, pd.DataFrame],
]:
    """
    Recover solutions from the optimization history by identifying features that
    have significant mean values (mu) in the final iterations.

    Parameters
    ----------
    search_history : Dict[str, List[Any]]
        The history of the optimization process containing 'mu', 'var', and 'alpha'.
        Expected keys: 'mu', 'var', 'alpha' with values as lists of arrays.
        'mu' should have shape [n_iterations, n_components, n_features].
    desired_sparsity : int
        The number of most important features to identify for each component.
        Must be positive.
    min_mu_threshold : float, optional
        The threshold for considering a feature as important based on its absolute mu value.
        Default is 0.25.
    verbose : bool, optional
        Whether to print detailed information about the recovered solutions. Default is True.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the recovered features will be displayed using the original names.
        Default is None.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, Union[np.ndarray, int]], Dict[str, pd.DataFrame]]
        A tuple containing:
        - solutions: Dictionary mapping each component (solution) to its identified features.
        - final_parameters: Dictionary with keys 'final iteration', 'final mu', 'final var', 'final alpha'
          containing the final parameters for each component at the iteration where features were selected.
        - full_nonzero_solutions: Dictionary mapping each component to a DataFrame of all features
          that exceeded the min_mu_threshold, with columns ['Feature', 'Mu value'].

    Raises
    ------
    ValueError
        If desired_sparsity is not positive.
    KeyError
        If search_history is missing required keys ('mu', 'var', 'alpha').

    Notes
    -----
    This function displays markdown output as a side effect when verbose=True.
    """
    # Input validation
    if desired_sparsity <= 0:
        raise ValueError("desired_sparsity must be positive")

    required_keys = {"mu", "var", "alpha"}
    if not required_keys.issubset(search_history.keys()):
        missing_keys = required_keys - set(search_history.keys())
        raise KeyError(f"search_history missing required keys: {missing_keys}")

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
        if original_feature_names_mapping is not None:
            feature_names = [original_feature_names_mapping[f] for f in features]
        else:
            feature_names = features
        top_features = pd.DataFrame(
            {
                "Feature": feature_names,
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
    return (
        solutions,
        final_parameters,
        full_nonzero_solutions,
    )


def show_algorithm_progress(
    history: Dict[str, List[Any]],
    plot_elbo_progress: bool = True,
    plot_mu_progress: bool = True,
    plot_alpha_progress: bool = True,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """
    Show the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Dictionary containing optimization history with keys 'elbo', 'mu', and 'alpha'
        (fewer keys allowed if corresponding plots are disabled).
        'mu' should have shape [n_iterations, n_components, n_features].
        This is the output of the `optimize` method of `BayesianFeatureSelector`.
    plot_elbo_progress : bool, optional
        Whether to plot the ELBO progress. Default is True.
    plot_mu_progress : bool, optional
        Whether to plot the mixture means (mu) trajectory. Default is True.
    plot_alpha_progress : bool, optional
        Whether to plot the mixture weights (alpha) progress. Default is True.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the plots will use the original feature names where applicable.
        Default is None.

    Returns
    -------
    None

    Notes
    -----
    This function displays markdown output and plots as side effects.
    The function requires the corresponding keys in history for each plot type requested.
    """
    display(Markdown(f"### Algorithm progress:"))
    n_components = len(history["mu"][0])

    # Plot ELBO progress
    if plot_elbo_progress:
        plot_elbo(history)

    # Plot mixture means trajectory for each component
    if plot_mu_progress:
        for k in range(n_components):
            plot_mu(
                history,
                component=k,
                original_feature_names_mapping=original_feature_names_mapping,
            )

    # Plot mixture weights (alpha)
    if plot_alpha_progress:
        plot_alpha(history)
    return


def show_regression_results_for_solutions(
    solutions: Dict[str, List[str]],
    df: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    verbose: bool = True,
) -> None:
    """
    Show regression results for each solution using the identified features.

    Parameters
    ----------
    solutions : Dict[str, List[str]]
        Dictionary mapping each component (solution) to its identified features.
        Feature names should correspond to column names in df.
    df : pd.DataFrame
        Feature matrix with features as columns.
    y : Union[pd.Series, np.ndarray]
        Response vector. Binary values {0, 1} for classification,
        continuous values for regression.
    penalty : Literal["l1", "l2", "elasticnet"], optional
        Type of regularization to use. Default is "l1".
    verbose : bool, optional
        Whether to print detailed regression metrics and coefficients
        for each component. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function displays markdown output and regression metrics as side effects.
    Automatically detects binary classification vs regression based on unique values in y.
    """
    is_binary = set(np.unique(y)) <= {0, 1}

    stats = {}
    for component, features in solutions.items():
        if verbose:
            display(Markdown(f"## Features of **{component}**"))

        if is_binary:
            stats[component] = solve_with_logistic_regression(
                X=df[features],
                y=y,
                penalty=penalty,
                verbose=verbose,
            )
        else:
            stats[component] = solve_with_linear_regression(
                X=df[features],
                y=y,
                penalty=penalty,
                verbose=verbose,
            )
        if verbose:
            display(Markdown("------------------"))

    display(Markdown(f"## Regression metrics overview"))
    # print the stats as data frame
    # each entry is a column in the data frame
    metrics_df = pd.DataFrame.from_dict(stats, orient="index")
    display(metrics_df)

    return


def display_features_overview(
    features_found: Union[List[str], set],
    true_support_features: List[str],
    n_total_features: int,
) -> None:
    """
    Display an overview of features found vs true support features.

    Parameters
    ----------
    features_found : Union[List[str], set]
        Collection of features found by the model.
    true_support_features : List[str]
        List of true support features (ground truth).
    n_total_features : int
        Total number of features in the dataset. Must be positive.

    Returns
    -------
    None

    Notes
    -----
    This function displays markdown output as a side effect.
    Shows statistics about missing features, extra features, and coverage percentages.
    """
    # Convert to set for set operations
    features_found = set(features_found)

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
