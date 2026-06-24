"""Tests for ParetoScaler."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from gemss.data_handling.pareto_scaler import ParetoScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_X() -> np.ndarray:
    return np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])


@pytest.fixture()
def X_with_nan() -> np.ndarray:
    return np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])


# ---------------------------------------------------------------------------
# Core mathematical correctness
# ---------------------------------------------------------------------------


def test_transform_formula(simple_X: np.ndarray) -> None:
    """Output equals Pareto scaling followed by [0, 1] rescaling."""
    scaler = ParetoScaler()
    result = scaler.fit_transform(simple_X)

    mean = np.mean(simple_X, axis=0)
    std = np.std(simple_X, axis=0)
    pareto = (simple_X - mean) / np.sqrt(std)
    p_min = pareto.min(axis=0)
    p_max = pareto.max(axis=0)
    expected = (pareto - p_min) / (p_max - p_min)

    np.testing.assert_allclose(result, expected)


def test_fit_stores_mean_and_std(simple_X: np.ndarray) -> None:
    """fit() sets mean_ and std_ to the column-wise mean and std."""
    scaler = ParetoScaler().fit(simple_X)

    np.testing.assert_allclose(scaler.mean_, np.mean(simple_X, axis=0))
    np.testing.assert_allclose(scaler.std_, np.std(simple_X, axis=0))


def test_fit_transform_equals_fit_then_transform(simple_X: np.ndarray) -> None:
    result_combined = ParetoScaler().fit_transform(simple_X)
    scaler = ParetoScaler().fit(simple_X)
    result_split = scaler.transform(simple_X)

    np.testing.assert_array_equal(result_combined, result_split)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_fit_ignores_nan_in_statistics(X_with_nan: np.ndarray) -> None:
    """Mean and std are computed ignoring NaN values."""
    scaler = ParetoScaler().fit(X_with_nan)

    np.testing.assert_allclose(scaler.mean_, np.nanmean(X_with_nan, axis=0))
    np.testing.assert_allclose(scaler.std_, np.nanstd(X_with_nan, axis=0))


def test_transform_preserves_nan(X_with_nan: np.ndarray) -> None:
    """NaN values in X are passed through unchanged after scaling."""
    scaler = ParetoScaler().fit(X_with_nan)
    result = scaler.transform(X_with_nan)

    assert np.isnan(result[1, 1]), 'NaN at [1,1] should be preserved'
    assert not np.any(np.isnan(result[~np.isnan(X_with_nan)])), (
        'No new NaN values should be introduced for non-NaN inputs'
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_feature_no_division_by_zero() -> None:
    """A constant feature (std == 0) must not raise and must yield ~0."""
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = ParetoScaler().fit(X)
    result = scaler.transform(X)

    # Constant feature (col 0) after centering is all zeros; dividing by
    # sqrt(eps) instead of 0 keeps it near zero — not nan or inf.
    assert np.all(np.isfinite(result)), 'Result must contain no inf or nan'
    np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-6)


def test_single_feature() -> None:
    """Works correctly with a single-column array."""
    X = np.array([[2.0], [4.0], [6.0]])
    result = ParetoScaler().fit_transform(X)
    assert result.shape == (3, 1)
    assert np.all(np.isfinite(result))


def test_output_shape_preserved(simple_X: np.ndarray) -> None:
    result = ParetoScaler().fit_transform(simple_X)
    assert result.shape == simple_X.shape


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises() -> None:
    """Calling transform before fit raises NotFittedError."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(NotFittedError):
        ParetoScaler().transform(X)


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params() -> None:
    assert ParetoScaler().get_params() == {}


def test_clone_produces_unfitted_copy(simple_X: np.ndarray) -> None:
    from sklearn.base import clone

    fitted = ParetoScaler().fit(simple_X)
    cloned = clone(fitted)
    assert not hasattr(cloned, 'mean_'), 'Cloned scaler must not carry fitted attributes'


@parametrize_with_checks([ParetoScaler()])
def test_sklearn_estimator_checks(estimator: ParetoScaler, check: object) -> None:
    """Run sklearn's built-in estimator compatibility checks."""
    check(estimator)
