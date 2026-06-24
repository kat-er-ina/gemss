"""Custom sklearn-compatible Pareto scaler."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class ParetoScaler(TransformerMixin, BaseEstimator):
    """
    Scales features using Pareto scaling.

    Centers data to the mean and scales it by dividing by the square root
    of the standard deviation.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    std_ : ndarray of shape (n_features,)
        The standard deviation for each feature in the training set.
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
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the per-feature mean and standard deviation.
            May contain NaN values; they are ignored during statistics computation.
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
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Pareto scaling: center by mean and divide by sqrt(std).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to scale. May contain NaN values; they are preserved as-is.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite='allow-nan')

        # Prevent division by zero for constant features
        safe_std = np.where(self.std_ == 0, np.finfo(float).eps, self.std_)

        X_tr = (X - self.mean_) / np.sqrt(safe_std)
        return X_tr
