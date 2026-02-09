"""Tests for log-likelihood with missing data in BayesianFeatureSelector.

- test_log_likelihood_with_missing_matches_manual: _log_likelihood_with_missing
  matches manual sum of Gaussian log-likelihoods over observed entries only.
- test_log_likelihood_with_missing_all_missing_returns_zero: when all X and
  response are missing, returns zero log-likelihood per sample.
- test_log_likelihood_with_missing_skips_nan_response: rows with NaN in y are
  skipped; result matches likelihood over non-NaN rows only.
- test_log_likelihood_complete_data_matches_manual: when X has no NaNs,
  log_likelihood(z) equals -0.5 * sum over samples of (z @ x_n - y_n)^2.
- test_log_likelihood_with_missing_single_observed_row: when only one sample
  has any observed features, result has correct shape and matches manual for that row.
"""

import numpy as np
import torch
from gemss.feature_selection.inference import BayesianFeatureSelector


def test_log_likelihood_with_missing_matches_manual() -> None:
    X = np.array(
        [
            [1.0, np.nan, 2.0],
            [np.nan, np.nan, np.nan],
            [3.0, 4.0, np.nan],
        ],
        dtype=np.float32,
    )
    y = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=2,
        n_iter=1,
        device='cpu',
    )

    z = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
    actual = selector._log_likelihood_with_missing(z)

    x_obs0 = torch.tensor([1.0, 2.0], dtype=z.dtype)
    pred0 = z[:, [0, 2]] @ x_obs0
    log0 = -0.5 * (pred0 - y[0]) ** 2

    x_obs2 = torch.tensor([3.0, 4.0], dtype=z.dtype)
    pred2 = z[:, [0, 1]] @ x_obs2
    log2 = -0.5 * (pred2 - y[2]) ** 2

    expected = log0 + log2

    assert torch.allclose(actual, expected)


def test_log_likelihood_with_missing_all_missing_returns_zero() -> None:
    X = np.full((2, 3), np.nan, dtype=np.float32)
    y = np.array([0.3, -1.2], dtype=np.float32)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=4,
        n_iter=1,
        device='cpu',
    )

    z = torch.randn(4, X.shape[1])
    actual = selector._log_likelihood_with_missing(z)
    expected = torch.zeros(4, dtype=z.dtype)

    assert actual.shape == (4,)
    assert torch.allclose(actual, expected)


def test_log_likelihood_with_missing_skips_nan_response() -> None:
    X = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=2,
        n_iter=1,
        device='cpu',
    )

    selector.y = selector.y.clone()
    selector.y[0] = float('nan')

    z = torch.tensor([[0.1, 0.2], [-0.3, 0.5]], dtype=torch.float32)
    actual = selector._log_likelihood_with_missing(z)

    pred = z[:, 0] * 3.0
    expected = -0.5 * (pred - y[1]) ** 2

    assert torch.allclose(actual, expected)


def test_log_likelihood_complete_data_matches_manual() -> None:
    X = np.array(
        [[1.0, 0.0, -1.0], [0.5, 2.0, 1.0], [3.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    y = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=2,
        n_iter=1,
        device='cpu',
    )
    z = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
    actual = selector.log_likelihood(z)
    pred = z @ torch.tensor(X.T, dtype=z.dtype)
    manual = -0.5 * ((pred - torch.tensor(y, dtype=z.dtype)) ** 2).sum(dim=1)
    assert torch.allclose(actual, manual)


def test_log_likelihood_with_missing_single_observed_row() -> None:
    X = np.array(
        [[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan]],
        dtype=np.float32,
    )
    y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=2,
        n_iter=1,
        device='cpu',
    )
    z = torch.tensor([[0.1, 0.2], [-0.3, 0.5]], dtype=torch.float32)
    actual = selector._log_likelihood_with_missing(z)
    assert actual.shape == (2,)
    pred = z @ torch.tensor([1.0, 2.0], dtype=z.dtype)
    expected = -0.5 * (pred - 1.0) ** 2
    assert torch.allclose(actual, expected)
