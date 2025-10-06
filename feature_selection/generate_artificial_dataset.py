import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


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
    np.ndarray
        Array of solution vectors, shape (n_solutions, n_features).
    pd.DataFrame
        DataFrame describing the generating parameters (supports and weights).
    """
    rng = np.random.default_rng(random_seed)
    uniform_interval_low = 2.0
    uniform_interval_high = 10.0

    # 1. Choose random indices for supports
    solutions = []
    supports = []
    for k in range(n_solutions):
        support = rng.choice(n_features, sparsity, replace=False)
        supports.append(support)

    # 2. Initialize data matrix
    X = np.zeros((n_samples, n_features))

    # 3. Create first solution and corresponding data
    w0 = np.zeros(n_features)
    w0[supports[0]] = rng.uniform(
        uniform_interval_low, uniform_interval_high, size=sparsity
    )  # * rng.choice([-1, 1], size=sparsity)
    solutions.append(w0)

    # Generate data for first support
    X[:, supports[0]] = rng.normal(0, 1.0, size=(n_samples, sparsity))

    # Compute response from first solution
    y_continuous = X @ w0

    # 4. Create additional solutions that produce the same response
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

    # 5. Fill remaining features with noise
    remaining_features = set(range(n_features)) - set(np.concatenate(supports))
    for feat in remaining_features:
        X[:, feat] = rng.normal(0, noise_data_std, size=n_samples)

    # 6. Add the same white noise to everything to make it non-exact
    X += rng.normal(0, noise_data_std, size=(n_samples, n_features))

    # 7. Binarize. Adjust threshold to match desired binary response ratio
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
    return df, response, solutions, parameters_df


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
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
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

    Returns
    -------
    pd.DataFrame
        Generated dataset with columns ['feature_1', ...].
    pd.Series
        Binary response variable (0 or 1) for each sample.
    np.ndarray
        Array of solution vectors, shape (n_solutions, n_features).
    pd.DataFrame
        DataFrame describing the generating parameters (supports and weights).
    """
    data, response, solutions, parameters = generate_multi_solution_data(
        n_samples=n_samples,
        n_features=n_features,
        n_solutions=n_solutions,
        sparsity=sparsity,
        noise_data_std=noise_data_std,
        binary_response_ratio=binary_response_ratio,
        random_seed=random_seed,
    )

    if save_to_csv:
        suffix = f"{n_samples}x{n_features}_{n_solutions}sols_{sparsity}sparse"
        data.to_csv(f"../data/artificial_dataset_{suffix}.csv")
        pd.DataFrame(solutions).to_csv(f"../data/artificial_solutions_{suffix}.csv")
        parameters.to_csv(f"../data/artificial_parameters_{suffix}.csv")

        if binarize:
            response.to_csv(f"../data/artificial_binary_labels_{suffix}.csv")
        else:
            response.to_csv(f"../data/artificial_continuous_labels_{suffix}.csv")
        print("Data and generating parameters saved to 'data/' directory.")

    return data, response, solutions, parameters


if __name__ == "__main__":
    # Example parameters
    NSAMPLES = 20
    NFEATURES = 60
    NSOLUTIONS = 3
    SPARSITY = 1
    NOISE_STD = 0.01
    BINARIZE = True
    BINARY_RESPONSE_RATIO = 0.5
    RANDOM_SEED = 42

    # Generate dataset
    data, response, solutions, parameters = generate_artificial_dataset(
        n_samples=NSAMPLES,
        n_features=NFEATURES,
        n_solutions=NSOLUTIONS,
        sparsity=SPARSITY,
        noise_data_std=NOISE_STD,
        binarize=BINARIZE,
        binary_response_ratio=BINARY_RESPONSE_RATIO,
        random_seed=RANDOM_SEED,
        save_to_csv=True,
    )
