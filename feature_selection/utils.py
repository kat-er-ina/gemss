"""
Utility functions for feature selection project.
Includes:
- Sampling
- Metrics
- Data handling
"""

import numpy as np
import torch

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
