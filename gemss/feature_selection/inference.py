"""
Variational inference and optimization for Bayesian sparse feature selection.

Implements:

- ELBO objective
- Gradient-based optimization of variational parameters (Adam)
- Optional support-overlap regularization (Jaccard penalty)

Classes
-------
BayesianFeatureSelector : Main class for variational inference-based feature selection.

Functions
---------
log_likelihood(z)      : Computes the regression log-likelihood for a batch of parameter samples.
h(z)                   : Computes the variational objective h(z) for a batch of samples.
elbo(z)                : Computes the standard evidence lower bound (ELBO).
elbo_regularized(z)    : Computes the ELBO with a penalty for support similarity (Jaccard).
optimize(...)          : Runs the main optimization loop.
"""

from collections.abc import Callable
from time import time
from typing import Literal

import numpy as np
import torch
from torch.optim import Adam

from gemss.utils.utils import myprint

from .models import (
    GaussianMixture,
    SpikeAndSlabPrior,
    StructuredSpikeAndSlabPrior,
    StudentTPrior,
)


class BayesianFeatureSelector:
    """
    Bayesian feature selection using variational inference and mixture of Gaussians.

    Attributes
    ----------
    n_features : int
        Number of features (dimension of beta/z).
    n_components : int
        Number of mixture components.
    X : torch.Tensor
        Data matrix of shape (n_samples, n_features).
    y : torch.Tensor
        Target vector of shape (n_samples,).
    batch_size : int
        Batch size for optimization.
    n_iter : int
        Number of optimization iterations.
    prior : object
        Prior (SpikeAndSlabPrior, StructuredSpikeAndSlabPrior, or StudentTPrior).
    mixture : GaussianMixture
        Learnable mixture of diagonal Gaussians (variational posterior).
    opt : torch.optim.Optimizer
        Optimizer for variational parameters.
    device : str
        Device to run computation on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        n_features: int,
        n_components: int,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        prior: Literal['ss', 'sss', 'student'] = 'sss',
        sss_sparsity: int = 3,
        sample_more_priors_coeff: float = 1.0,
        var_slab: float = 100.0,
        var_spike: float = 0.1,
        weight_slab: float = 0.9,
        weight_spike: float = 0.1,
        student_df: int = 1,
        student_scale: float = 1.0,
        lr: float = 2e-3,
        batch_size: int = 16,
        n_iter: int = 5000,
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        """
        Initialize the BayesianFeatureSelector.

        Parameters
        ----------
        n_features : int
            Number of features (dimension of beta/z).
        n_components : int
            Number of mixture components.
        X : array-like
            Data matrix of shape (n_samples, n_features).
        y : array-like
            Target vector of shape (n_samples,).
        prior : str, optional
            Type of prior to use ('ss', 'sss', 'student'):
            'ss' = Spike-and-Slab,
            'sss' (default) = Structured Spike-and-Slab,
            'student' = Student-t prior.
        sss_sparsity : int, optional
            Number of nonzero entries per solution (for structured spike-and-slab prior).
            Default is 3.
        sample_more_priors_coeff : float, optional
            Coefficient to scale the number of supports sampled in the structured spike-and-slab prior.
            A higher value increases the number of supports sampled, potentially improving
            approximation at the cost of computation. Default is 1.0.
        var_slab : float, optional
            Variance of the slab prior (default: 100.0). Used only in 'ss' and 'sss' priors.
        var_spike : float, optional
            Variance of the spike prior (default: 0.1). Used only in 'ss' and 'sss' priors.
        weight_slab : float, optional
            Weight of slab prior (default: 0.9). Used only in 'ss' prior.
        weight_spike : float, optional
            Weight of spike prior (default: 0.1). Used only in 'ss' prior.
        student_df : float, optional
            Degrees of freedom for Student-t prior (default: 1). Used only in 'student' prior.
        student_scale : float, optional
            Scale parameter for Student-t prior (default: 1.0). Used only in 'student' prior.
        lr : float, optional
            Learning rate for Adam optimizer (default: 2e-3).
        batch_size : int, optional
            Batch size for optimization (default: 16).
        n_iter : int, optional
            Number of optimization iterations (default: 5000).
        device : Literal["cpu", "cuda"], optional
            Device to run computation on ('cpu' or 'cuda', default: 'cpu').
        """  # noqa: E501
        self.n_features = n_features
        self.n_components = n_components

        # Convert data to tensors, preserving NaN values
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

        # Validate that response values don't contain NaN
        if torch.isnan(self.y).any():
            raise ValueError('Response (y) contains NaN. Remove samples with missing responses.')

        # Precompute NaN-related information for X to avoid repeated computation
        self._has_missing_X = torch.isnan(self.X).any().item()
        if self._has_missing_X:
            self._X_observed_mask = ~torch.isnan(self.X)
            self._X_filled = torch.nan_to_num(self.X, nan=0.0)
            # Samples that have at least one observed feature (y is guaranteed non-NaN)
            self._valid_sample_mask = self._X_observed_mask.any(dim=1)
        else:
            self._X_observed_mask = None
            self._X_filled = None
            self._valid_sample_mask = None

        self.batch_size = batch_size
        self.n_iter = n_iter

        if prior == 'ss':
            self.prior = SpikeAndSlabPrior(var_slab, var_spike, weight_slab, weight_spike)
        elif prior == 'sss':
            self.prior = StructuredSpikeAndSlabPrior(
                n_features,
                sparsity=sss_sparsity,
                sample_more_priors_coeff=sample_more_priors_coeff,
                var_slab=var_slab,
                var_spike=var_spike,
            )
        elif prior == 'student':
            self.prior = StudentTPrior(df=student_df, scale=student_scale)

        self.mixture = GaussianMixture(n_components, n_features).to(device)
        self.opt = Adam(self.mixture.parameters(), lr=lr)
        self.device = device

    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood for regression: log p(y | z, X).

        Handles missing values in X by using only observed features for each sample.
        This allows the algorithm to work with missing data without imputation or dropping samples.

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-likelihood values for each sample, shape (batch_size,).
        """
        # Use precomputed flag to check if X has missing values
        if self._has_missing_X:
            return self._log_likelihood_with_missing(z)
        else:
            # Standard computation for complete data
            pred = torch.matmul(z, self.X.T)  # [batch_size, n_samples]
            mse = ((pred - self.y.unsqueeze(0)) ** 2).sum(dim=-1)  # sum over samples
            return -0.5 * mse

    def _log_likelihood_with_missing(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood when X contains missing values.

        For each data sample, use only the observed features to compute the prediction.
        This maintains gradient flow while handling missing data naturally.

        Parameters
        ----------
        z : torch.Tensor
            Parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-likelihood values, shape (batch_size,).
        """
        batch_size = z.shape[0]

        # These are guaranteed non-None since this method is only called when _has_missing_X is True
        assert self._valid_sample_mask is not None
        assert self._X_filled is not None

        # Combine precomputed X mask with dynamic y NaN check (y could be modified after init)
        valid_mask = self._valid_sample_mask & ~torch.isnan(self.y)
        if not valid_mask.any():
            return torch.zeros(batch_size, device=z.device)

        # Use precomputed X with NaNs filled with zeros
        pred = torch.matmul(z, self._X_filled.T)  # [batch_size, n_samples]
        residual = pred - torch.nan_to_num(self.y, nan=0.0).unsqueeze(0)
        log_likes = -0.5 * (residual**2)

        valid_mask = valid_mask.to(dtype=log_likes.dtype, device=log_likes.device)
        total_log_like = (log_likes * valid_mask.unsqueeze(0)).sum(dim=1)

        return total_log_like

    def h(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the variational objective h(z).

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            h(z) values, shape (batch_size,).
            Each entry is log p(z) + log p(y|z,X) - log q(z).
        """
        logp = self.prior.log_prob(z) + self.log_likelihood(z)
        logq = self.mixture.log_prob(z)
        return logp - logq

    def elbo(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the standard evidence lower bound (ELBO).

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Scalar ELBO (mean over batch).
        """
        return self.h(z).mean()

    def elbo_regularized(
        self,
        z: torch.Tensor,
        lambda_jaccard: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute ELBO with regularization using average Jaccard similarity between supports.

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).
        lambda_jaccard : float, optional
            Strength of regularization penalty (penalizes overlap between supports).
            Default is 10.0.

        Returns
        -------
        torch.Tensor
            Regularized ELBO (mean over batch minus penalty).
        """
        sigmoid_coeff = 100.0  # Steepness parameter for sigmoid
        batch_size = z.shape[0]

        # Compute soft supports using sigmoid for differentiability
        support_mask = torch.sigmoid(sigmoid_coeff * torch.abs(z))  # [batch_size, n_features]

        # Compute pairwise Jaccard similarities (vectorized)
        if batch_size < 2:
            avg_jaccard = torch.tensor(0.0, device=z.device)
        else:
            intersection = support_mask @ support_mask.T  # [batch_size, batch_size]
            # Use |A ∪ B| = |A| + |B| - |A ∩ B| to avoid creating 3D tensor
            support_sizes = support_mask.sum(dim=1)  # [batch_size]
            union = support_sizes[:, None] + support_sizes[None, :] - intersection
            jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
            pair_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=z.device)
            avg_jaccard = jaccard[pair_idx[0], pair_idx[1]].mean()

        # Regularized ELBO
        elbo_reg = self.elbo(z) - lambda_jaccard * avg_jaccard
        return elbo_reg

    def optimize(
        self,
        log_callback: Callable[[int, float, GaussianMixture], None] | None = None,
        regularize: bool = False,
        lambda_jaccard: float = 10.0,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Main optimization loop for the feature selector.

        Each iteration:
        - Samples `z ~ q(z)` from the mixture
        - Computes the ELBO (optionally regularized)
        - Backpropagates and takes an Adam step on variational parameters

        Missing values in `X` are handled inside `log_likelihood` via masking.

         Parameters
        ----------
        log_callback : callable, optional
            Function to call for logging every 100 iterations.
        regularize : bool, optional
            If True, use elbo_regularized during optimization (default: False) by penalizing overlap
            between solutions.
        lambda_jaccard : float, optional
            Regularization strength for penalty in the ELBO computation (default: 10.0). Higher
            values encourage more diverse solutions (lesser overlap). Used only if regularize=True.
        verbose : bool, optional
            If True, print optimization progress (default: True).

        Returns
        -------
        history : dict[str, list[float]]
            Dictionary containing optimization history:
            - 'elbo': list of ELBO values
            - 'mu': list of mixture means per iteration
            - 'var': list of mixture variances per iteration
            - 'alpha': list of mixture weights per iteration
        """
        history = {'elbo': [], 'mu': [], 'var': [], 'alpha': []}
        start_time = time()
        for it in range(self.n_iter):
            # Sample z ~ q(z) - this should always produce complete samples (no NaNs)
            z, comp_idx = self.mixture.sample(self.batch_size)
            # Note: z automatically has requires_grad=True through reparameterization

            if regularize and (lambda_jaccard != 0):
                elbo = self.elbo_regularized(
                    z,
                    lambda_jaccard=lambda_jaccard,
                )
            else:
                elbo = self.elbo(z)

            self.opt.zero_grad()
            (-elbo).backward()
            self.opt.step()

            # Record statistics
            history['elbo'].append(elbo.item())
            history['mu'].append(self.mixture.mu.detach().cpu().clone().numpy())
            history['var'].append(self.mixture.get_variance().detach().cpu().clone().numpy())

            # Store alphas safely
            alpha_val = self.mixture.get_alpha()
            history['alpha'].append(alpha_val.detach().cpu().clone().numpy())

            nth_iteration = it % 100 == 0
            if log_callback and nth_iteration:
                log_callback(it, elbo.item(), self.mixture)
            if verbose and nth_iteration and it > 0:
                elapsed_time = time() - start_time
                myprint(
                    msg=(
                        f'**Iteration {it}:** elapsed {elapsed_time:.0f}s, '
                        f'remaining {elapsed_time / it * (self.n_iter - it):.0f}s, '
                        f'ELBO = {elbo.item()}'
                    )
                )
        if verbose:
            myprint('Optimization complete.', use_markdown=True)
        return history
