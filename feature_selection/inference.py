"""
Variational inference and optimization for Bayesian sparse feature selection.
Implements:
- ELBO objective
- Gradient and Hessian computation
- SGD/Adam parameter updates
"""

from typing import Literal
import torch
from torch.optim import Adam
from .models import (
    StudentTPrior,
    SpikeAndSlabPrior,
    StructuredSpikeAndSlabPrior,
    GaussianMixture,
)


class BayesianFeatureSelector:
    """
    Bayesian feature selection using variational inference and mixture of Gaussians.
    """

    def __init__(
        self,
        n_features,
        n_components,
        X,
        y,
        sparsity,
        prior: Literal["ss", "sss", "student"] = "sss",
        var_slab=100.0,
        var_spike=0.1,
        w_slab=0.9,
        w_spike=0.1,
        student_df=3,
        student_scale=1.0,
        lr=2e-3,
        batch_size=16,
        n_iter=10000,
        device="cpu",
    ):
        """
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
        sparsity : int
            Number of nonzero entries per solution (for structured spike-and-slab prior).
        prior : str, optional
            Type of prior to use ('ss', 'sss', 'student'). The shortcuts stand for:
            'ss' = Spike-and-Slab,
            'sss' (default) = Structured Spike-and-Slab,
            'student' = Student-t prior.
        var_slab : float, optional
            Variance of the slab prior (default: 100.0). Used only if prior is 'ss' or 'sss'.
        var_spike : float, optional
            Variance of the spike prior (default: 0.1). Used only if prior is 'ss' or 'sss'.
        w_slab : float, optional
            Weight of slab prior (default: 0.9). Used only if prior is 'ss' or 'sss'.
        w_spike : float, optional
            Weight of spike prior (default: 0.1). Used only if prior is 'ss' or 'sss'.
        student_df : float, optional
            Degrees of freedom for Student-t prior (default: 3). Used only if prior is 'student'.
        student_scale : float, optional
            Scale parameter for Student-t prior (default: 1.0). Used only if prior is 'student'.
        lr : float, optional
            Learning rate for Adam optimizer (default: 2e-3).
        batch_size : int, optional
            Batch size for optimization (default: 16).
        n_iter : int, optional
            Number of optimization iterations (default: 10000).
        device : str, optional
            Device to run computation on ('cpu' or 'cuda', default: 'cpu').
        """
        self.n_features = n_features
        self.n_components = n_components
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.n_iter = n_iter

        if prior == "ss":
            self.prior = SpikeAndSlabPrior(var_slab, var_spike, w_slab, w_spike)
        elif prior == "sss":
            self.prior = StructuredSpikeAndSlabPrior(
                n_features,
                sparsity=sparsity,
                var_slab=var_slab,
                var_spike=var_spike,
            )
        elif prior == "student":
            self.prior = StudentTPrior(df=student_df, scale=student_scale)

        self.mixture = GaussianMixture(n_components, n_features).to(device)
        self.opt = Adam(self.mixture.parameters(), lr=lr)
        self.device = device

    def log_likelihood(self, z):
        """
        Compute log-likelihood for regression: log p(y | z, X).

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-likelihood values for each sample, shape (batch_size,).
        """
        pred = torch.matmul(z, self.X.T)  # [batch_size, n_samples]
        mse = ((pred - self.y.unsqueeze(0)) ** 2).sum(dim=-1)  # sum over samples
        return -0.5 * mse

    def h(self, z):
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
        """
        logp = self.prior.log_prob(z) + self.log_likelihood(z)
        logq = self.mixture.log_prob(z)
        return logp - logq

    def elbo(self, z):
        """
        Compute the evidence lower bound (ELBO).

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

    def optimize(self, log_callback=None):
        """
        Main optimization loop.

        Parameters
        ----------
        log_callback : callable, optional
            Function to call for logging every 100 iterations.

        Returns
        -------
        history : dict
            Dictionary containing optimization history:
            - 'elbo': list of ELBO values
            - 'mu': list of mixture means per iteration
            - 'var': list of mixture variances per iteration
            - 'alpha': list of mixture weights per iteration
        """
        history = {"elbo": [], "mu": [], "var": [], "alpha": []}
        for it in range(self.n_iter):
            # Sample z ~ q(z)
            z, comp_idx = self.mixture.sample(self.batch_size)
            z = z.requires_grad_()  # Ensure gradients for z

            # h(z)
            h_val = self.h(z)
            elbo = h_val.mean()
            self.opt.zero_grad()
            (-elbo).backward()
            self.opt.step()

            # Record statistics
            history["elbo"].append(elbo.item())
            history["mu"].append(self.mixture.mu.detach().cpu().clone().numpy())
            history["var"].append(
                self.mixture.get_variance().detach().cpu().clone().numpy()
            )
            history["alpha"].append(
                self.mixture.get_alpha().detach().cpu().clone().numpy()
            )

            if log_callback and it % 100 == 0:
                log_callback(it, elbo.item(), self.mixture)
        return history
