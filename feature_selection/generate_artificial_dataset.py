import numpy as np
import pandas as pd


def generate_sparse_mixture_parameters(
    n_features,
    n_components,
    sparsity=1,
    nonzero_range=(-10.0, 10.0),
    seed=42,
):
    """
    Generate sparse parameters for a Gaussian mixture model.

    Parameters
    ----------
    n_features : int
        Number of features (dimension).
    n_components : int
        Number of mixture components (solutions).
    sparsity : int, optional
        Number of nonzero entries per solution (default: 1).
    nonzero_range : tuple of (float, float), optional
        Range for nonzero values (default: (-10.0, 10.0)).
    seed : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    mu : np.ndarray
        Means array of shape (n_solutions, n_features), sparse.
    sigma : np.ndarray
        Standard deviations array of shape (n_solutions, n_features).
    alpha : np.ndarray
        Mixture weights of shape (n_solutions,).
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.zeros((n_components, n_features))
    for k in range(n_components):
        idx = np.random.choice(n_features, sparsity, replace=False)
        mu[k, idx] = np.random.uniform(*nonzero_range, size=sparsity)
    sigma = np.ones((n_components, n_features))
    alpha = np.ones(n_components) / n_components
    return mu, sigma, alpha


def generate_mixture_data(
    n_samples=100,
    n_features=5,
    n_components=3,
    mu=None,
    sigma=None,
    alpha=None,
    random_seed=42,
):
    """Generate a dataset from a mixture of Gaussians.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples (rows) in the generated dataset.
    n_features : int, default=5
        Number of features (columns) in the generated dataset.
    n_components : int, default=3
        Number of mixture components (solutions) in the Gaussian mixture.
    mu : np.ndarray, optional
        Component means array of shape (n_components, n_features).
        If None, generated using sparse mixture parameters.
    sigma : np.ndarray, optional
        Component standard deviations array of shape (n_components, n_features).
        If None, generated using sparse mixture parameters.
    alpha : np.ndarray, optional
        Mixture weights array of shape (n_components,), must sum to 1.
        If None, uniform weights are used.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated dataset with named rows ('sample_1', ...) and
        columns ('feature_1', ...).
    np.ndarray
        Array of true component labels for each sample, shape (n_samples,).

    Notes
    -----
    If any of mu, sigma, or alpha are None, they will be generated using
    the `generate_sparse_mixture_parameters` function with default settings.
    """
    np.random.seed(random_seed)

    # Default: random means, stddevs, uniform weights
    if (mu is None) or (sigma is None) or (alpha is None):
        mu_computed, sigma_computed, alpha_computed = (
            generate_sparse_mixture_parameters(
                n_features=n_features,
                n_components=n_components,
            )
        )
    if mu is None:
        mu = mu_computed
    if sigma is None:
        sigma = sigma_computed
    if alpha is None:
        alpha = alpha_computed

    # Choose mixture component for each sample according to alpha
    components = np.random.choice(n_components, size=n_samples, p=alpha)
    data = np.zeros((n_samples, n_features))

    for i in range(n_samples):
        k = components[i]
        data[i] = np.random.normal(loc=mu[k], scale=sigma[k], size=n_features)

    # Create DataFrame with appropriate naming
    row_names = [f"sample_{i+1}" for i in range(n_samples)]
    col_names = [f"feature_{j+1}" for j in range(n_features)]
    df = pd.DataFrame(data, index=row_names, columns=col_names)

    return df, components


import numpy as np


def generate_binary_labels(
    X, component_assignment, mu, threshold=0.0, noise_std=0.0, random_seed=42
):
    """
    Generate binary labels for each sample, such that
    each sample's label is determined by the features in its solution (mu).

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    component_assignment : np.ndarray
        Array of assigned mixture components (n_samples,)
    mu : np.ndarray
        Matrix of solution vectors (n_components, n_features)
    threshold : float, optional
        Decision threshold (default: 0.0)
    noise_std : float, optional
        Standard deviation of Gaussian noise added to activation (default: 0.0)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    y : np.ndarray
        Binary labels (n_samples,)
    """
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        k = component_assignment[i]
        support = np.flatnonzero(mu[k])
        activation = np.sum(X[i, support] * mu[k, support])
        if noise_std > 0:
            activation += np.random.normal(0, noise_std)
        y[i] = int(activation > threshold)
    return y


if __name__ == "__main__":
    # Example parameters
    NSAMPLES = 100
    NFEATURES = 5
    NSOLUTIONS = 3

    mu, sigma, alpha = generate_sparse_mixture_parameters(
        n_features=NFEATURES,
        n_components=NSOLUTIONS,
        sparsity=1,
    )
    df, component_assignment = generate_mixture_data(
        NSAMPLES, NFEATURES, NSOLUTIONS, mu, sigma, alpha
    )
    y = generate_binary_labels(
        X=df.values, component_assignment=component_assignment, mu=mu
    )

    # Save to CSV
    df.to_csv("../data/artificial_mixture_dataset.csv")
    pd.Series(y, index=df.index, name="binary label").to_csv(
        "../data/artificial_mixture_binary_labels.csv"
    )
    # Save component assignments for reference
    pd.Series(component_assignment, index=df.index, name="true_component").to_csv(
        "../data/artificial_mixture_component_assignment.csv"
    )
    print("Dataset and component assignments saved to 'data/' directory.")
