"""Custom sklearn-compatible Pareto scaler."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class ParetoScaler(TransformerMixin, BaseEstimator):
    """
    Scales features using Pareto scaling followed by [0, 1] rescaling.

    Step 1: Centers data to the mean and divides by the square root of the
    standard deviation (classical Pareto scaling).
    Step 2: Rescales the result to [0, 1] using the training-set min/max of
    the Pareto-scaled values.

    This produces bounded output on the same scale as MinMaxScaler while
    preserving Pareto's property of downweighting high-variance features
    less aggressively than standard scaling.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    std_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
    pareto_min_ : ndarray of shape (n_features,)
        Per-feature minimum of the Pareto-scaled training data.
    pareto_max_ : ndarray of shape (n_features,)
        Per-feature maximum of the Pareto-scaled training data.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self) -> None:
        pass

    def __sklearn_tags__(self) -> object:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'ParetoScaler':
        """
        Compute statistics for Pareto scaling and the subsequent [0, 1] rescaling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. May contain NaN values; they are ignored during
            statistics computation.
        y : None
            Ignored.

        Returns
        -------
        self : ParetoScaler
            Fitted scaler.
        """
        X = validate_data(self, X, ensure_all_finite='allow-nan')
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)

        # Compute Pareto-scaled training values to derive min/max for rescaling
        safe_std = np.where(self.std_ == 0, np.finfo(float).eps, self.std_)
        X_pareto = (X - self.mean_) / np.sqrt(safe_std)
        self.pareto_min_ = np.nanmin(X_pareto, axis=0)
        self.pareto_max_ = np.nanmax(X_pareto, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Pareto scaling then rescale to [0, 1].

        NaN values are preserved throughout.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform. May contain NaN values.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed data in [0, 1] range (on training data).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite='allow-nan')

        # Step 1: Pareto scaling
        safe_std = np.where(self.std_ == 0, np.finfo(float).eps, self.std_)
        X_pareto = (X - self.mean_) / np.sqrt(safe_std)

        # Step 2: rescale to [0, 1]
        pareto_range = self.pareto_max_ - self.pareto_min_
        safe_range = np.where(pareto_range == 0, np.finfo(float).eps, pareto_range)
        X_tr = (X_pareto - self.pareto_min_) / safe_range
        return X_tr
