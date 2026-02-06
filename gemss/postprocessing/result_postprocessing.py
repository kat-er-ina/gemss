"""
This module provides functions to recover, display and analyze solutions
from the optimization history of the Bayesian feature selection algorithm.
"""

from typing import Any

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from gemss.postprocessing.outliers import (
    detect_outlier_features,
    get_outlier_solutions,
    show_outlier_info,
)
from gemss.utils.utils import myprint
from gemss.utils.visualizations import compare_parameters, get_algorithm_progress_plots


def get_full_solutions(
    search_history: dict[str, list[Any]],
    desired_sparsity: int,
    min_mu_threshold: float | None = 1e-6,
    original_feature_names_mapping: dict[str, str] | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, np.ndarray | int],
]:
    """
    Recover full non-zero solutions and the corresponding final parameters from the optimization history.

    Parameters
    ----------
    search_history : Dict[str, List[Any]]
        The history of the optimization process containing 'mu', 'var', and 'alpha'.
        Expected keys: 'mu', 'var', 'alpha' with values as lists of arrays.
        'mu' should have shape [n_iterations, n_components, n_features].
    min_mu_threshold : float, optional
        The threshold for considering a feature as important based on its absolute mu value.
        Default is 1e-6.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the recovered features will be displayed using the original names.
        Default is None.

    Returns
    -------
    Tuple[
        Dict[str, pd.DataFrame],
        Dict[str, Union[np.ndarray, int]],
    ]
        A tuple containing:
        - full_solutions: Dictionary mapping each component to a DataFrame of all features
          that exceeded the min_mu_threshold, with columns ['Feature', 'Mu value'].
        - final_parameters: Dictionary with keys 'final iteration', 'final mu', 'final var', 'final alpha'
          containing the final parameters for each component at the iteration where features were selected.
    """
    n_components = len(search_history['mu'][0])
    n_features = len(search_history['mu'][0][0])

    # from each component, get the two features that zero out the last
    final_mu = np.zeros((n_components, n_features))
    final_var = np.zeros((n_components, n_features))
    final_alpha = np.zeros(n_components)
    final_iteration = np.zeros((n_components), dtype=int)

    # A dictionary to store the solutions found for each component
    # all features with mu > min_mu_threshold in last iterations
    full_solutions = {}

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
            arr = np.array(search_history['mu'])  # shape [n_iter, n_components, n_features]
            mu_traj = arr[-i, k, :]
            features = [
                f'feature_{j}' for j in range(len(mu_traj)) if abs(mu_traj[j]) > min_mu_threshold
            ]
            i += 1

        # Organize these features by their absolute mu values
        if original_feature_names_mapping is not None:
            feature_names = [original_feature_names_mapping[f] for f in features]
        else:
            feature_names = features
        df_features = pd.DataFrame(
            {
                'Feature': feature_names,
                'Mu value': [mu_traj[int(f.split('_')[1])] for f in features],
            }
        )
        df_features['absolute_mu'] = np.abs(df_features['Mu value'])
        df_features = (
            df_features.sort_values(by='absolute_mu', ascending=False)
            .drop(columns=['absolute_mu'])
            .reset_index(drop=True)
        )
        full_solutions[f'component_{k}'] = df_features

        # Store the final parameters for this component at the iteration where features were selected
        final_iteration[k] = -(i - 1)
        final_mu[k] = np.array(search_history['mu'])[final_iteration[k], k, :]
        final_var[k] = np.array(search_history['var'])[final_iteration[k], k, :]
        final_alpha[k] = np.array(search_history['alpha'])[final_iteration[k], k]

    # Get the final parameters
    final_parameters = {
        'final iteration': final_iteration,  # the iteration from which the final parameters are taken
        'final mu': final_mu,
        'final var': final_var,
        'final alpha': final_alpha,
    }
    return (
        full_solutions,
        final_parameters,
    )


def get_top_solutions(
    full_solutions: dict[str, pd.DataFrame],
    top_n: int,
) -> dict[str, pd.DataFrame]:
    """
    Extract the top few features from full non-zero solutions based on their mu values.

    Parameters:
    -----------
    full_solutions: Dict[str, pd.DataFrame]
        A dictionary where each key is a component identifier and each value is a DataFrame
        of features and their mu values in a column named 'Mu value'.
    top_n: int
        The number of top features to extract based on their absolute mu values.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary mapping each component to its DataFrame of top features.
    """
    top_solutions = {}
    for component, df in full_solutions.items():
        df_copy = df.copy()
        df_copy['abs_mu'] = df_copy['Mu value'].abs()
        df_copy = (
            df_copy.sort_values(by='abs_mu', ascending=False)
            .drop(columns=['abs_mu'])
            .reset_index(drop=True)
        )
        top_solutions[component] = df_copy.head(top_n)

    return top_solutions


def get_features_from_solutions(
    solutions: dict[str, pd.DataFrame],
) -> dict[str, list[float]]:
    """
    Extract the lists of features of a given set of solutions.

    Parameters:
    -----------
    solutions: Dict[str, pd.DataFrame]
        A dictionary where each key is a component identifier and each value is a DataFrame
        of features (in columns named 'Feature') and their mu values.

    Returns:
    --------
    Dict[str, List[float]]
        A dictionary mapping each component to the corresponding list of features.
    """
    feature_lists = {}
    for component, df in solutions.items():
        feature_lists[component] = df['Feature'].tolist()
    return feature_lists


def recover_solutions(
    search_history: dict[str, list[Any]],
    desired_sparsity: int,
    min_mu_threshold: float | None = 1e-6,
    use_median_for_outlier_detection: bool | None = False,
    outlier_deviation_thresholds: list[float] | None = [2.5, 3.5],
    original_feature_names_mapping: dict[str, str] | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, np.ndarray | int],
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
        Default is 1e-6.
    use_median_for_outlier_detection: bool, optional
        If True, use median and MAD for outlier detection; otherwise, use mean and STD.
        Default is False, i.e. use mean and STD.
    outlier_deviation_thresholds : List[float], optional
        A list of deviation thresholds to use for outlier detection. Default is [2.0, 2.5, 3.0].
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the recovered features will be displayed using the original names.
        Default is None.

    Returns
    -------
    Tuple[
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
        Dict[str, Dict[str, pd.DataFrame]],
        Dict[str, Union[np.ndarray, int]],
    ]
        A tuple containing:
        - full_solutions: Dictionary mapping each component to a DataFrame of all features
          that exceeded the min_mu_threshold, with columns ['Feature', 'Mu value'].
        - top_solutions: Dictionary mapping each component (solution) to a DataFrame of the top few features
          based on their |mu| values. The DataFrames contain columns ['Feature', 'Mu value'].
        - outlier_solutions: A dictionary where the keys are MAD/STD values for outlier detection
          and the corresponding values are dictionaries mapping each component to a DataFrame
          of outlier features.
        - final_parameters: Dictionary with keys 'final iteration', 'final mu', 'final var', 'final alpha'
          containing the final parameters for each component at the iteration where features were selected.

    Raises
    ------
    ValueError
        If desired_sparsity is not positive.
    KeyError
        If search_history is missing required keys ('mu', 'var', 'alpha').
    """
    # Input validation
    if (desired_sparsity <= 0) or (not isinstance(desired_sparsity, int)):
        raise ValueError('Desired_sparsity must be a positive integer.')

    required_keys = {'mu', 'var', 'alpha'}
    if not required_keys.issubset(search_history.keys()):
        missing_keys = required_keys - set(search_history.keys())
        raise KeyError(f'Search_history missing required keys: {missing_keys}')

    # Get the full solutions
    full_solutions, final_parameters = get_full_solutions(
        search_history=search_history,
        desired_sparsity=desired_sparsity,
        min_mu_threshold=min_mu_threshold,
        original_feature_names_mapping=original_feature_names_mapping,
    )

    # Get the solutions with top 'desired_sparsity' features for each component
    top_solutions = get_top_solutions(full_solutions, desired_sparsity)

    # Get the outlier solutions
    if use_median_for_outlier_detection:
        std_or_mad = 'MAD'
    else:
        std_or_mad = 'STD'

    outlier_solutions = {}
    for deviation in outlier_deviation_thresholds:
        outlier_solutions[f'{std_or_mad}_{deviation}'] = get_outlier_solutions(
            history=search_history,
            use_medians_for_outliers=use_median_for_outlier_detection,
            outlier_threshold_coeff=deviation,
            original_feature_names_mapping=original_feature_names_mapping,
        )

    return (
        full_solutions,
        top_solutions,
        outlier_solutions,
        final_parameters,
    )


def compare_true_and_found_features(
    features_found: list[str] | set,
    true_support_features: list[str],
    n_total_features: int,
    use_markdown: bool | None = True,
) -> None:
    """
    Print an overview of features found vs true support features.

    Parameters
    ----------
    features_found : Union[List[str], set]
        Collection of features found by the model.
    true_support_features : List[str]
        List of true support features (ground truth).
    n_total_features : int
        Total number of features in the dataset. Must be positive.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns
    -------
    None
    """
    # Convert to set for set operations
    features_found = set(features_found)
    missing_features = set(true_support_features) - features_found
    extra_features = features_found - set(true_support_features)

    myprint(f'All features: {n_total_features}', use_markdown=use_markdown, header=3)
    myprint(
        f'True support features: {len(true_support_features)} ({len(true_support_features) / n_total_features:.1%})',
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f'{sorted(true_support_features)}', use_markdown=use_markdown)
    myprint(
        f'All features found: {len(features_found)} ({len(features_found) / n_total_features:.1%})',
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f'{sorted(features_found)}', use_markdown=use_markdown)
    myprint(
        f'Missing true support features: {len(missing_features)}',
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f'{sorted(missing_features)}', use_markdown=use_markdown)
    myprint(
        f'Extra features found: {len(extra_features)} ({len(extra_features) / n_total_features:.1%})',
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f'{sorted(extra_features)}', use_markdown=use_markdown)
    return


def get_features_from_long_solutions(
    solutions: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """
    Extract features from full non-zero solutions.

    Parameters:
    -----------
    solutions: Dict[str, pd.DataFrame]
        A dictionary where each key is a solution identifier and each value is a DataFrame
        of features and their mu values.

    Returns:
    --------
    Dict[str, List[str]]
        A dictionary mapping each component to its list of features.
    """
    extracted_solutions = {}
    for component, details in solutions.items():
        extracted_solutions[component] = details['Feature'].tolist()
    return extracted_solutions


def get_unique_features(solutions: dict[str, dict[str, list[Any]]]) -> list[str]:
    """
    Returns the list of features contained in all of the solutions. Each feature is listed only once.

    Parameters:
    -----------
    solutions: Dict[str, Dict[str]]
        A dictionary where each key is a solution identifier and each value is a dictionary
        of features.

    Returns:
    --------
    List[str]
        A list of unique features across all solutions.
    """
    return list(set().union(*solutions.values()))


def show_unique_features(
    solutions: dict[str, dict[str, list[Any]]],
    use_markdown: bool | None = True,
) -> None:
    """
    Display the unique features found across all solutions.

    Parameters
    ----------
    solutions : Dict[str, Dict[str, List[Any]]]
        A dictionary where each key is a solution identifier and each value is a dictionary
        of features.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns
    -------
    None
    """
    unique_features = get_unique_features(solutions)

    myprint(
        msg=f'Unique features across all solutions ({len(unique_features)} total):',
        use_markdown=use_markdown,
        header=2,
    )
    myprint(
        msg=f'{sorted(unique_features)}',
        use_markdown=use_markdown,
        code=True,
    )
    return


def show_unique_features_from_full_solutions(
    solutions: dict[str, pd.DataFrame],
    use_markdown: bool | None = True,
) -> None:
    """
    Display the unique features found across all full non-zero solutions.

    Parameters
    ----------
    solutions : Dict[str, pd.DataFrame]
        A dictionary where each key is a solution identifier and each value is a DataFrame
        of features and their mu values.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.
    """
    solutions = get_features_from_solutions(solutions)
    show_unique_features(
        solutions=solutions,
        use_markdown=use_markdown,
    )
    return


def show_features_in_solutions(
    solutions: dict[str, dict[str, list[Any]]],
    history: dict[str, list[np.ndarray]],
    constants: dict[str, float],
    use_markdown: bool = True,
) -> None:
    """
    Display detailed information about each solution including component weights.

    Parameters:
    -----------
    solutions : Dict[str, Dict[str]]
        A dictionary where each key is a solution identifier and each value is a dictionary
        of features.
    history : Dict[str, List[np.ndarray]]
        The optimization history containing 'alpha' values.
    constants : Dict[str, float]
        A dictionary containing the algorithm settings, namely the 'DESIRED_SPARSITY'.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns:
    -------
    None
    """
    myprint(
        msg=f'Required sparsity = {constants["DESIRED_SPARSITY"]}',
        use_markdown=use_markdown,
        bold=True,
    )

    for component, features in solutions.items():
        i = component.split('_')[-1]
        alpha = history['alpha'][-1][int(i)]
        myprint(
            msg=f'Candidate solution no. {i}:',
            use_markdown=use_markdown,
            header=2,
        )
        myprint(
            msg=f'Component weight = {alpha:.3f}',
            use_markdown=use_markdown,
            bold=True,
        )
        for feature in features:
            myprint(
                msg=f'- {feature}',
                use_markdown=use_markdown,
            )
    return


def show_final_parameter_comparison(
    true_parameters: dict[str, Any],
    final_parameters: dict[str, Any],
) -> None:
    """
    Display a comparison between true parameters and final estimated parameters.

    Parameters:
    -----------
    true_parameters : Dict[str, Any]
        A dictionary containing the true parameters for comparison.
    final_parameters : Dict[str, Any]
        A dictionary containing the final estimated parameters.

    Returns:
    --------
    None
    """

    # Show final mixture means and weights
    compare_parameters(true_parameters, final_parameters['final mu'])

    # Show final alpha weights
    display(Markdown('### Final mixture weights (alpha):'))
    for i, alpha in enumerate(final_parameters['final alpha']):
        display(Markdown(f'- **Component {i}:** {alpha:.3f}'))
    return


# This function cannot be in visualizations.py due to circular import issues (outlier functions)
def show_algorithm_progress_with_outliers(
    history: dict[str, list[Any]],
    plot_elbo_progress: bool = True,
    plot_mu_progress: bool = True,
    plot_alpha_progress: bool = True,
    original_feature_names_mapping: dict[str, str] | None = None,
    detect_outliers: bool = False,
    use_medians_for_outliers: bool = False,
    outlier_threshold_coeff: float = 2.5,
    subsample_history_for_plotting: bool = False,
) -> None:
    """
    Show the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights. This function uses markdown formatting
    and Plotly for interactive visualizations.

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
    detect_outliers : bool, optional
        If True, detect and display outlier features in the mu progress plots. Default is False.
    use_medians_for_outliers : bool, optional
        If True, use median and MAD for outlier detection; otherwise, use mean and STD.
        Default is False.
    outlier_threshold_coeff : float, optional
        The coefficient for determining outliers based on deviation from mean/MAD. Default is 2.5.
    subsample_history_for_plotting: bool, optional
        If True, plot only every N-th iteration in order to save resources during plotting.

    Returns
    -------
    None

    Notes
    -----
    This function displays markdown output and plots as side effects.
    The function requires the corresponding keys in history for each plot type requested.
    """
    myprint('Algorithm progress:', header=2, use_markdown=True)

    figures = get_algorithm_progress_plots(
        history,
        plot_elbo_progress=plot_elbo_progress,
        plot_mu_progress=plot_mu_progress,
        plot_alpha_progress=plot_alpha_progress,
        original_feature_names_mapping=original_feature_names_mapping,
        subsample_history_for_plotting=subsample_history_for_plotting,
    )

    if plot_elbo_progress:
        figures['elbo'].show(config={'displayModeBar': False})

    if plot_alpha_progress:
        figures['alpha'].show(config={'displayModeBar': False})

    if plot_mu_progress:
        # Optionally add info about outliers
        final_mus_df = pd.DataFrame(
            index=[
                (
                    original_feature_names_mapping[f'feature_{i}']
                    if original_feature_names_mapping
                    else f'feature_{i}'
                )
                for i in range(len(history['mu'][0][0]))
            ]
        )

        n_components = len(history['mu'][0])
        for k in range(n_components):
            final_mus_df[f'component_{k}_mus'] = history['mu'][-1][k]

            # show mu progress plots
            figures[f'mu_{k}'].show(config={'displayModeBar': False})

            if detect_outliers:
                outlier_info = detect_outlier_features(
                    values=final_mus_df[f'component_{k}_mus'],
                    threshold_coeff=outlier_threshold_coeff,
                    use_median=use_medians_for_outliers,
                    replace_middle_by_zero=True,
                )
                outlier_dict = {f'component_{k}': outlier_info}

                show_outlier_info(
                    outlier_info=outlier_dict,
                    component_numbers=k,
                    use_markdown=True,
                )
    return
