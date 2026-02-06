"""Tests for BayesianFeatureSelector.optimize() and optimization workflow.

- test_optimize_history_shapes: optimize() returns history with keys elbo, mu,
  var, alpha; correct lengths and shapes; finite ELBO; alpha sums to 1.
- test_optimize_regularized_callback: with regularize=True and log_callback,
  callback is invoked at expected iterations and ELBO stays finite.
- test_optimize_with_missing_values: optimize() runs and returns finite ELBO
  when X contains missing values (NaNs).
- test_optimize_raises_when_y_contains_nan: constructor raises ValueError when
  y contains NaN.
- test_optimize_regularize_lambda_zero_uses_elbo: with regularize=True and
  lambda_jaccard=0, optimization runs and returns finite ELBO (unregularized path).
- test_optimize_without_callback: optimize(log_callback=None) runs and returns
  full history.
"""

import numpy as np
import torch

import pytest

from gemss.feature_selection.inference import BayesianFeatureSelector


def _make_dataset(
    n_samples: int = 12,
    n_features: int = 6,
    noise_std: float = 0.1,
    missing_rate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    true_w = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    y = X @ true_w + rng.normal(0.0, noise_std, size=(n_samples,)).astype(np.float32)

    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        X = X.copy()
        X[mask] = np.nan

    return X, y


def test_optimize_history_shapes() -> None:
    torch.manual_seed(0)
    X, y = _make_dataset(seed=0)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=2,
        X=X,
        y=y,
        n_iter=5,
        batch_size=4,
        device='cpu',
    )

    history = selector.optimize(verbose=False)

    assert set(history.keys()) == {'elbo', 'mu', 'var', 'alpha'}
    assert len(history['elbo']) == 5
    assert len(history['mu']) == 5
    assert len(history['var']) == 5
    assert len(history['alpha']) == 5

    mu0 = np.asarray(history['mu'][0])
    var0 = np.asarray(history['var'][0])
    alpha0 = np.asarray(history['alpha'][0])

    assert mu0.shape == (2, X.shape[1])
    assert var0.shape == (2, X.shape[1])
    assert alpha0.shape == (2,)

    assert np.isfinite(history['elbo']).all()
    assert np.isclose(np.sum(history['alpha'][-1]), 1.0, atol=1e-5)


def test_optimize_regularized_callback() -> None:
    torch.manual_seed(1)
    X, y = _make_dataset(seed=1)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=2,
        X=X,
        y=y,
        n_iter=101,
        batch_size=4,
        device='cpu',
    )

    calls = []

    def _callback(it: int, elbo: float, mixture) -> None:
        calls.append(it)

    history = selector.optimize(
        log_callback=_callback,
        regularize=True,
        lambda_jaccard=1.0,
        verbose=False,
    )

    assert calls == [0, 100]
    assert len(history['elbo']) == 101
    assert np.isfinite(history['elbo']).all()


def test_optimize_with_missing_values() -> None:
    torch.manual_seed(2)
    X, y = _make_dataset(seed=2, missing_rate=0.2)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=2,
        X=X,
        y=y,
        n_iter=3,
        batch_size=4,
        device='cpu',
    )

    history = selector.optimize(verbose=False)

    assert len(history['elbo']) == 3
    assert np.isfinite(history['elbo']).all()


def test_optimize_raises_when_y_contains_nan() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([1.0, float('nan')], dtype=np.float32)
    with pytest.raises(ValueError, match='Response variable.*NaN'):
        BayesianFeatureSelector(
            n_features=X.shape[1],
            n_components=1,
            X=X,
            y=y,
            batch_size=2,
            n_iter=1,
            device='cpu',
        )


def test_optimize_regularize_lambda_zero_uses_elbo() -> None:
    torch.manual_seed(42)
    X, y = _make_dataset(seed=42)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=2,
        X=X,
        y=y,
        n_iter=5,
        batch_size=4,
        device='cpu',
    )
    history = selector.optimize(
        regularize=True,
        lambda_jaccard=0.0,
        verbose=False,
    )
    assert len(history['elbo']) == 5
    assert np.isfinite(history['elbo']).all()


def test_optimize_without_callback() -> None:
    torch.manual_seed(0)
    X, y = _make_dataset(seed=0)
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=2,
        X=X,
        y=y,
        n_iter=3,
        batch_size=4,
        device='cpu',
    )
    history = selector.optimize(log_callback=None, verbose=False)
    assert set(history.keys()) == {'elbo', 'mu', 'var', 'alpha'}
    assert len(history['elbo']) == 3
