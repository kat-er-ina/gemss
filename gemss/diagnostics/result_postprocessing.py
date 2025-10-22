"""
This module provides functions to recover, display and analyze solutions
from the optimization history of the Bayesian feature selection algorithm.
"""

from typing import Dict, List, Tuple, Literal, Any, Optional, Union, Set
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn.preprocessing import StandardScaler
from gemss.diagnostics.visualizations import (
    plot_elbo,
    plot_mu,
    plot_alpha,
    compare_parameters,
)
from gemss.diagnostics.simple_regressions import (
    solve_with_logistic_regression,
    solve_with_linear_regression,
)


def myprint(
    msg: str,
    use_markdown: Optional[bool] = True,
    bold: Optional[bool] = False,
    header: Optional[int] = 0,
    code: Optional[bool] = False,
    file: Optional[Any] = None,
) -> None:
    """
    Print a message using Markdown formatting if specified, otherwise in plain text.

    Parameters
    ----------
    msg : str
        The message to print.
    use_markdown : bool
        Whether to use Markdown formatting rather than plain text.
    bold : bool, optional
        Whether to make the message bold. Default is False.
    header : int, optional
        The header level (1-6) for the message. Default is 0 (no header).
    file : Any, optional
        The file to print the message to. Default is None (prints to stdout).

    Returns
    -------
    None
    """
    if use_markdown:
        if header > 0:
            msg = f"{'#' * header} {msg}"
        if bold:
            msg = f"**{msg}**"
        if code:
            msg = f"```{msg}```"
        display(Markdown(msg))
    else:
        if header > 0:
            print("\n")
        print(msg, file=file)
        if header > 0:
            print("-" * len(msg), file=file)
    return


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


def show_long_solutions(
    full_nonzero_solutions: Dict[str, pd.DataFrame],
    title: str = "Long solutions",
    use_markdown: Optional[bool] = True,
) -> None:
    """
    Display the solutions in a DataFrame format: each column corresponds to a component
    and contains the identified features.

    Parameters
    ----------
    full_nonzero_solutions : Dict[str, pd.DataFrame]
        Dictionary mapping each component (solution) to a DataFrame containing
        features that exceeded the min_mu_threshold in the last iterations,
        with columns ['Feature', 'Mu value'].
    title : str, optional
        Title for the displayed DataFrame. Default is "Long solutions".
    use_markdown : bool, optional
        Whether to format the title using Markdown. Default is True.

    Returns
    -------
    None
    """
    df_full_solutions = get_long_solutions_df(full_nonzero_solutions)

    myprint(
        msg=title,
        use_markdown=use_markdown,
        header=2,
    )
    if use_markdown:
        display(df_full_solutions)
    else:
        print(df_full_solutions)

    return


def recover_solutions(
    search_history: Dict[str, List[Any]],
    desired_sparsity: int,
    min_mu_threshold: Optional[float] = 1e-6,
    verbose: Optional[bool] = True,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
    use_markdown: Optional[bool] = True,
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
        Default is 1e-6.
    verbose : bool, optional
        Whether to print detailed information about the recovered solutions. Default is True.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the recovered features will be displayed using the original names.
        Default is None.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

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
    if (desired_sparsity <= 0) or (not isinstance(desired_sparsity, int)):
        raise ValueError("Desired_sparsity must be a positive integer.")

    required_keys = {"mu", "var", "alpha"}
    if not required_keys.issubset(search_history.keys()):
        missing_keys = required_keys - set(search_history.keys())
        raise KeyError(f"Search_history missing required keys: {missing_keys}")

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
            myprint(
                msg=f"Component {k}:",
                use_markdown=use_markdown,
                header=2,
            )
            myprint(
                msg=f"- Last {desired_sparsity}+ features with absolute value greater than {min_mu_threshold}:",
                use_markdown=use_markdown,
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
            if use_markdown:
                display(top_features[["Feature", "Mu value"]])
            else:
                print(top_features[["Feature", "Mu value"]])

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
    detect_outliers: bool = True,
    use_medians_for_outliers: bool = False,
    outlier_threshold_coeff: float = 3.0,
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

    Returns
    -------
    None

    Notes
    -----
    This function displays markdown output and plots as side effects.
    The function requires the corresponding keys in history for each plot type requested.
    """
    myprint("Algorithm progress:", header=2, use_markdown=True)
    n_components = len(history["mu"][0])

    if detect_outliers:
        outliers = {}
        final_mus_df = pd.DataFrame(
            index=[
                (
                    original_feature_names_mapping[f"feature_{i}"]
                    if original_feature_names_mapping
                    else f"feature_{i}"
                )
                for i in range(len(history["mu"][0][0]))
            ]
        )

    # Plot ELBO progress
    if plot_elbo_progress:
        plot_elbo(history)

    # Plot mixture weights (alpha)
    if plot_alpha_progress:
        plot_alpha(history)

    # Plot mixture means trajectory for each component
    if plot_mu_progress:
        for k in range(n_components):
            myprint(f"Candidate solution {k}", header=3, use_markdown=True)

            if detect_outliers:
                final_mus_df[f"component_{k}_mus"] = history["mu"][-1][k]
                outliers[f"component_{k}"] = detect_outlier_features(
                    final_mus_df[f"component_{k}_mus"],
                    threshold=outlier_threshold_coeff,
                    use_median=use_medians_for_outliers,
                    replace_middle_by_zero=True,
                )
                print_outlier_info(
                    outlier_info=outliers,
                    component_no=k,
                    use_markdown=True,
                )

            plot_mu(
                history,
                component=k,
                original_feature_names_mapping=original_feature_names_mapping,
            )
    return


def show_regression_results_for_solutions(
    solutions: Dict[str, List[str]],
    df: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    use_standard_scaler: bool = True,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    verbose: bool = True,
    use_markdown: bool = True,
) -> None:
    """
    Show regression results for each solution using the identified features. Based on the type
    of response vector y, it automatically selects logistic regression or linear regression.

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
    use_standard_scaler : bool, optional
        Whether to standardize features before regression. Default is True.
    penalty : Literal["l1", "l2", "elasticnet"], optional
        Type of regularization to use. Default is "l1".
    verbose : bool, optional
        Whether to print detailed regression metrics and coefficients
        for each component. Default is True.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns
    -------
    None
    """
    if set(np.unique(y)).__len__() == 2:
        is_binary = True
    else:
        is_binary = False

    stats = {}
    for component, features in solutions.items():
        if verbose:
            myprint(
                msg=f"Features of **{component}**",
                use_markdown=use_markdown,
                header=2,
            )

        if use_standard_scaler:
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])

            if verbose:
                myprint(
                    msg=f"- Features standardized using StandardScaler.",
                    use_markdown=use_markdown,
                )

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
            myprint(msg="------------------", use_markdown=use_markdown)

    # get the stats as data frame
    # each entry is a column in the data frame
    metrics_df = pd.DataFrame.from_dict(stats, orient="index")

    if is_binary:
        myprint(
            msg=f"Classification metrics overview (penalty: {penalty})",
            use_markdown=use_markdown,
            header=2,
        )
    else:
        myprint(
            msg=f"Regression metrics overview (penalty: {penalty})",
            use_markdown=use_markdown,
            header=2,
        )

    if use_markdown:
        display(metrics_df)
    else:
        print(metrics_df)

    return


def compare_true_and_found_features(
    features_found: Union[List[str], set],
    true_support_features: List[str],
    n_total_features: int,
    use_markdown: Optional[bool] = True,
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

    myprint(f"All features: {n_total_features}", use_markdown=use_markdown, header=3)
    myprint(
        f"True support features: {len(true_support_features)} ({len(true_support_features)/n_total_features:.1%})",
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f"{sorted(true_support_features)}", use_markdown=use_markdown)
    myprint(
        f"All features found: {len(features_found)} ({len(features_found)/n_total_features:.1%})",
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f"{sorted(features_found)}", use_markdown=use_markdown)
    myprint(
        f"Missing true support features: {len(missing_features)}",
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f"{sorted(missing_features)}", use_markdown=use_markdown)
    myprint(
        f"Extra features found: {len(extra_features)} ({len(extra_features)/n_total_features:.1%})",
        use_markdown=use_markdown,
        header=3,
    )
    myprint(f"{sorted(extra_features)}", use_markdown=use_markdown)
    return


def get_unique_features(solutions: Dict[str, Dict[str, List[Any]]]) -> List[str]:
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
    solutions: Dict[str, Dict[str, List[Any]]],
    use_markdown: Optional[bool] = True,
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
        msg=f"Unique features across all solutions ({len(unique_features)} total):",
        use_markdown=use_markdown,
        header=2,
    )
    myprint(
        msg=f"{sorted(unique_features)}",
        use_markdown=use_markdown,
        code=True,
    )
    return


def show_solutions_details(
    solutions: Dict[str, Dict[str, List[Any]]],
    history: Dict[str, List[np.ndarray]],
    constants: Dict[str, float],
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
        msg=f"Required sparsity = {constants['DESIRED_SPARSITY']}",
        use_markdown=use_markdown,
        bold=True,
    )

    for component, features in solutions.items():
        i = component.split("_")[-1]
        alpha = history["alpha"][-1][int(i)]
        myprint(
            msg=f"Candidate solution no. {i}:",
            use_markdown=use_markdown,
            header=2,
        )
        myprint(
            msg=f"Component weight = {alpha:.3f}",
            use_markdown=use_markdown,
            bold=True,
        )
        for feature in features:
            myprint(
                msg=f"- {feature}",
                use_markdown=use_markdown,
            )
    return


def show_final_parameter_comparison(
    true_parameters: Dict[str, Any],
    final_parameters: Dict[str, Any],
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
    compare_parameters(true_parameters, final_parameters["final mu"])

    # Show final alpha weights
    display(Markdown("### Final mixture weights (alpha):"))
    for i, alpha in enumerate(final_parameters["final alpha"]):
        display(Markdown(f"- **Component {i}:** {alpha:.3f}"))
    return


def detect_outlier_features(
    values: pd.Series,
    threshold=3,
    use_median: Optional[bool] = False,
    replace_middle_by_zero: Optional[bool] = False,
) -> Dict[str, List[float]]:
    """
    Detects outlier features based on their deviation from the mean or median. Return the names of the outliers.

    Parameters:
    -----------
    values: pd.Series
        Series of mu values to analyze. Index of the series represents feature indices.
    threshold: float
        Threshold multiplier for the absolute deviation to identify outliers.
    use_median: bool, optional
        Whether to use median for outlier detection instead of mean. Using median is less sensitive to large
        values but it leads to larger deviation values, i.e. more outliers detected. Default is False.
    replace_middle_by_zero: bool, optional
        If True, uses zero instead of median or mean for outlier detection. Default is False.

    Returns:
    -----------
    Dict[str, List[float]]
        A dictionary with two keys:
        - "features": List of feature indices identified as outliers.
        - "values": Corresponding mu values of the outlier features.
    """

    if values.empty:
        return {"features": [], "values": []}

    if replace_middle_by_zero:
        middle = 0
    elif use_median:
        middle = values.median()
    else:  # use mean
        middle = values.mean()

    if use_median:
        deviation = (values - middle).abs().median()
    else:  # use mean
        deviation = (values - middle).abs().mean()

    outlier_condition = (values - middle).abs() > threshold * deviation

    return {
        "features": values.index[outlier_condition].tolist(),
        "values": values[outlier_condition].tolist(),
    }


def print_outlier_info(outlier_info, component_no, use_markdown) -> None:
    """
    Helper function to print outlier information.
    Parameters:
    -----------
    outlier_info: Dict[str, List[float]]
        A dictionary with outlier features and their values.
    component_no: int
        The component number to print outlier information for.
    use_markdown: bool
        Whether to format the output using Markdown.

    Returns:
    --------
    None
    """
    myprint(
        msg=f"{len(outlier_info[f'component_{component_no}']['features'])} outliers in candidate solution {component_no}:",
        use_markdown=use_markdown,
        bold=True,
    )
    for feat, val in zip(
        outlier_info[f"component_{component_no}"]["features"],
        outlier_info[f"component_{component_no}"]["values"],
    ):
        myprint(
            msg=f"- {feat}: {val:.4f}",
            use_markdown=use_markdown,
        )
    return


def show_outlier_features_by_component(
    history: Dict[str, List],
    use_median: Optional[bool] = False,
    outlier_threshold_coeff: Optional[float] = 3,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
    use_markdown: bool = True,
) -> None:
    """
    Displays outlier features detected in each candidate solution's mu values.

    Parameters:
    -----------
    history: Dict[str, List]
        The search history containing mu values for each iteration.
    use_median: bool, optional
        Whether to use median for outlier detection instead of mean. Using median is less sensitive to large
        values but it leads to larger deviation values, i.e. more outliers detected. Default is False.
    outlier_threshold_coeff: float, optional
        The multiplier of absolute deviation for outlier detection. Default is 3.
        Larger values lead to fewer outliers being detected.
    original_feature_names_mapping: Optional[Dict[str, str]], optional
        A mapping from feature indices to original feature names. If provided, will use original names in the output.
    use_markdown: bool, optional
        Whether to format the output using Markdown for better readability instead of plain text. Default is True.

    Returns:
    --------
    None
    """
    n_features = history["mu"][-1][0].shape[0]
    n_components = len(history["mu"][-1])

    if original_feature_names_mapping is not None:
        feature_names = [
            original_feature_names_mapping.get(f"feature_{i}", f"feature_{i}")
            for i in range(n_features)
        ]
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    myprint(
        msg=f"Detection of outlier features based on their last mu values (no |mu| thresholding)",
        use_markdown=use_markdown,
        header=2,
    )

    final_mus_df = pd.DataFrame(index=feature_names)
    outliers = {}
    for component in range(n_components):
        final_mus_df[f"component_{component}_mus"] = history["mu"][-1][component]
        outliers[f"component_{component}"] = detect_outlier_features(
            final_mus_df[f"component_{component}_mus"],
            threshold=outlier_threshold_coeff,
            use_median=use_median,
            replace_middle_by_zero=True,
        )
        print_outlier_info(
            outlier_info=outliers,
            component_no=component,
            use_markdown=use_markdown,
        )

    return None
