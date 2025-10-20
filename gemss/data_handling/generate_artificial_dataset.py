from IPython.display import Markdown, display
from typing import Dict, List
import numpy as np
import pandas as pd

from gemss.diagnostics.visualizations import (
    show_correlations_with_response,
    show_correlation_matrix,
    show_label_histogram,
)


def generate_multi_solution_data(
    n_samples=100,
    n_features=10,
    n_solutions=3,
    sparsity=2,
    noise_data_std=0.01,
    binarize=True,
    binary_response_ratio=0.5,
    random_seed=42,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    """
    Generate a synthetic dataset with multiple valid sparse solutions. If the parameter
    'binarize' is True, the response variable is binary (0 or 1) and the corresponding problem
    is a binary classification problem. Otherwise, the response variable is continuous and the
    corresponding problem is a regression problem.
    Each solution is a sparse set of features that produces the same response.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples (rows) in the generated dataset.
    n_features : int, default=10
        Number of features (columns) in the generated dataset.
    n_solutions : int, default=3
        Number of sparse solutions ("true" supports), each a sparse linear classifier.
    sparsity : int, default=2
        Number of nonzero entries (support size) per solution.
    noise_data_std : float, default=0.01
        Standard deviation of the Gaussian noise added to the features.
    binarize : bool, default=True
        If True, the response variable is binary (0 or 1).
    binary_response_ratio : float, default=0.5
        Proportion of samples assigned label 1 (controls class balance). Used only if binarize is
        True.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated dataset with columns ['feature_0', ..., 'feature_n-1'].
    pd.Series
        Binary response variable (0 or 1) for each sample.
    Dict[str, List[str]]
        Dictionary of the generating solutions (supports).
    pd.DataFrame
        DataFrame describing the generating parameters (supports and weights).
    """
    rng = np.random.default_rng(random_seed)
    uniform_interval_low = 2.0
    uniform_interval_high = 10.0

    # Choose random indices for supports
    solutions = []
    supports = []
    for k in range(n_solutions):
        support = rng.choice(n_features, sparsity, replace=False)
        supports.append(support)

    # Initialize data matrix
    X = np.zeros((n_samples, n_features))

    # Create first solution and corresponding data
    w0 = np.zeros(n_features)
    w0[supports[0]] = rng.uniform(
        uniform_interval_low, uniform_interval_high, size=sparsity
    )  # * rng.choice([-1, 1], size=sparsity)
    solutions.append(w0)

    # Generate data for first support
    X[:, supports[0]] = rng.normal(0, 1.0, size=(n_samples, sparsity))

    # Compute response from first solution
    y_continuous = X @ w0

    # Create additional solutions that produce the same response
    # as linear combinations of the first solution
    for k in range(1, n_solutions):
        # get coefficients for linear combination
        # allow negative coefficients but not too close to zero
        coeff_min = 2.0
        coeff_max = 10.0
        coeffs = rng.uniform(
            coeff_min, coeff_max, size=[sparsity, sparsity]
        ) * rng.choice(
            [-1, 1],
            size=sparsity,
            replace=True,
        )
        # create a new set of supporting data vectors as linear combinations
        # of the first support's data vectors
        X[:, supports[k]] = X[:, supports[0]] @ coeffs

        # Find the corresponding solution weights
        # Use least squares to solve: wk[supports[k]] = pinv(X[:, supports[k]]) @ y_continuous
        wk = np.zeros(n_features)
        wk[supports[k]] = np.linalg.pinv(X[:, supports[k]]) @ y_continuous
        solutions.append(wk)

    # Fill remaining features with noise
    remaining_features = set(range(n_features)) - set(np.concatenate(supports))
    for feat in remaining_features:
        X[:, feat] = rng.normal(0, noise_data_std, size=n_samples)

    # Add the same white noise to everything to make it non-exact
    X += rng.normal(0, noise_data_std, size=(n_samples, n_features))

    # Binarize. Adjust threshold to match desired binary response ratio
    if binarize:
        y_prob = 1 / (1 + np.exp(-y_continuous))  # sigmoid
        threshold = np.quantile(y_prob, 1 - binary_response_ratio)
        y = (y_prob > threshold).astype(int)
    else:
        y = y_continuous

    # Convert to required format
    solutions = np.stack(solutions)

    # Save generating parameters as a DataFrame
    parameters = []
    for k in range(n_solutions):
        parameters.append(
            {
                "solution_index": k,
                "support_indices": supports[k].tolist(),
                "weights": solutions[k, supports[k]].tolist(),
                "full_weights": solutions[k].tolist(),
                "sparsity": sparsity,
            }
        )
    parameters_df = pd.DataFrame(parameters)

    df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
    response = pd.Series(y, name="binary_response")
    generating_solutions = {
        f"solution_{k}": [f"feature_{i}" for i in supports[k]]
        for k in range(n_solutions)
    }

    return df, response, generating_solutions, parameters_df


def show_overview_of_generated_data(
    df: pd.DataFrame,
    y: pd.Series,
    parameters: pd.DataFrame,
    show_feature_correlations: bool = False,
) -> None:
    """
    Display an overview of the generated dataset, including its dimensions,
    generating solutions, label distribution, and possibly feature correlations.

    Parameters:
    -----------
    df : pd.DataFrame
        Generated dataset with features as columns.
    y : pd.Series
        Generated binary response variable.
    parameters : pd.DataFrame
        DataFrame containing the parameters used for data generation.
    show_feature_correlations : bool, default=False
        If True, displays the correlation matrix of the features. The matrix is shown
        only if the number of features is less than or equal to 100.

    Returns:
    --------
    None
    """
    n_samples, n_features = df.shape
    n_solutions = parameters.shape[0]
    sparsity = parameters["sparsity"].iloc[0]

    display(Markdown("### Artificial dataset"))
    display(Markdown(f"- **Number of samples:** {n_samples}"))
    display(Markdown(f"- **Number of features:** {n_features}"))
    display(Markdown(f"- **Number of generating solutions:** {n_solutions}"))
    display(
        Markdown(
            f"- **Number of nonzero features per solution (sparsity):** {sparsity}"
        )
    )

    support_indices = parameters["support_indices"].sum()
    support_features = [f"feature_{i}" for i in support_indices]
    display(
        Markdown(
            f"- **Nonzero features:** {len(support_features)}<br>{sorted(support_features)}"
        )
    )

    display(Markdown("- **Parameters of the mixture components:**"))
    display(parameters)

    # Plot the distribution of labels y using Plotly
    is_binary = set(np.unique(y)) == {np.int64(0), np.int64(1)}
    if is_binary:
        display(Markdown("- **Distribution of binary labels:**"))
        display(pd.Series(y).value_counts(normalize=True))
    else:
        display(Markdown("- **Distribution of continuous labels:**"))
        show_label_histogram(y, nbins=10)

    # Compute features' correlation with the binary response
    show_correlations_with_response(df, y, support_features)

    # Display the correlation matrix of the features
    if show_feature_correlations and (n_features <= 100):
        display(Markdown("- **Correlation matrix of features:**"))
        show_correlation_matrix(df)
    return


def generate_artificial_dataset(
    n_samples=100,
    n_features=5,
    n_solutions=3,
    sparsity=1,
    noise_data_std=0.01,
    binarize=True,
    binary_response_ratio=0.5,
    random_seed=42,
    save_to_csv=False,
    print_data_overview=True,
    show_feature_correlations=False,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], pd.DataFrame]:
    """
    Generate an artificial binary classification dataset with multiple sparse solutions.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples (rows) in the generated dataset.
    n_features : int, default=5
        Number of features (columns) in the generated dataset.
    n_solutions : int, default=3
        Number of sparse solutions ("true" supports).
    sparsity : int, default=1
        Number of nonzero entries per solution.
    noise_data_std : float, default=0.01
        Standard deviation of the Gaussian noise added to the features.
    binarize : bool, default=True
        If True, the response variable is binary (0 or 1).
    binary_response_ratio : float, default=0.5
        Proportion of samples assigned label 1. Used only if binarize is True.
    random_seed : int, default=42
        Random seed for reproducibility.
    save_to_csv : bool, default=False
        If True, saves the generated dataset and parameters to CSV files.
    print_data_overview : bool, default=True
        If True, prints an overview of the generated data and plots.
    show_feature_correlations : bool, default=False
        If True, displays the correlation matrix of the features
        (only if number of features <= 100).

    Returns
    -------
    pd.DataFrame
        Generated dataset with columns ['feature_1', ...].
    pd.Series
        Binary response variable (0 or 1) for each sample.
    Dict[str, List[str]]
        Dictionary of the generating solutions (supports).
    pd.DataFrame
        DataFrame describing the generating parameters (supports and weights).
    """
    data, response, solutions, parameters = generate_multi_solution_data(
        n_samples=n_samples,
        n_features=n_features,
        n_solutions=n_solutions,
        sparsity=sparsity,
        noise_data_std=noise_data_std,
        binarize=binarize,
        binary_response_ratio=binary_response_ratio,
        random_seed=random_seed,
    )

    if print_data_overview:
        show_overview_of_generated_data(
            df=data,
            y=response,
            parameters=parameters,
            show_feature_correlations=show_feature_correlations,
        )

    if save_to_csv:
        suffix = f"{n_samples}x{n_features}_{n_solutions}sols_{sparsity}sparse_{noise_data_std}noise"
        data.to_csv(f"../data/artificial_dataset_{suffix}.csv")
        solutions.to_csv(f"../data/artificial_solutions_{suffix}.csv")
        parameters.to_csv(f"../data/artificial_parameters_{suffix}.csv")

        if binarize:
            response.to_csv(f"../data/artificial_binary_labels_{suffix}.csv")
        else:
            response.to_csv(f"../data/artificial_continuous_labels_{suffix}.csv")
        print(
            f"Data and generating parameters saved to 'data/' directory with suffix '{suffix}'."
        )

    return data, response, solutions, parameters


if __name__ == "__main__":
    # Example parameters
    N_SAMPLES = 20
    N_FEATURES = 60
    N_GENERATING_SOLUTIONS = 3
    SPARSITY = 1
    NOISE_STD = 0.01
    BINARIZE = True
    BINARY_RESPONSE_RATIO = 0.5
    DATASET_SEED = 42

    # Generate dataset
    data, response, solutions, parameters = generate_artificial_dataset(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_solutions=N_GENERATING_SOLUTIONS,
        sparsity=SPARSITY,
        noise_data_std=NOISE_STD,
        binarize=BINARIZE,
        binary_response_ratio=BINARY_RESPONSE_RATIO,
        random_seed=DATASET_SEED,
        save_to_csv=True,
    )
