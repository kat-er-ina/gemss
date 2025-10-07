"""
Variational inference and optimization for Bayesian sparse feature selection.

Implements:

- ELBO objective
- Gradient and Hessian computation
- SGD/Adam parameter updates
- Regularized ELBO with Jaccard similarity penalty

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

from typing import Literal, Dict, List
from IPython.display import display, Markdown
import torch
from torch.optim import Adam
from feature_selection.models import (
    StudentTPrior,
    SpikeAndSlabPrior,
    StructuredSpikeAndSlabPrior,
    GaussianMixture,
)
from feature_selection.utils import print_optimization_setting


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
        Prior distribution object (SpikeAndSlabPrior, StructuredSpikeAndSlabPrior, or StudentTPrior).
    mixture : GaussianMixture
        Learnable mixture of diagonal Gaussians (variational posterior).
    opt : torch.optim.Optimizer
        Optimizer for variational parameters.
    device : str
        Device to run computation on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        n_features,
        n_components,
        X,
        y,
        prior: Literal["ss", "sss", "student"] = "sss",
        sss_sparsity=3,
        var_slab=100.0,
        var_spike=0.1,
        weight_slab=0.9,
        weight_spike=0.1,
        student_df=1,
        student_scale=1.0,
        lr=2e-3,
        batch_size=16,
        n_iter=10000,
        device="cpu",
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
            self.prior = SpikeAndSlabPrior(
                var_slab, var_spike, weight_slab, weight_spike
            )
        elif prior == "sss":
            self.prior = StructuredSpikeAndSlabPrior(
                n_features,
                sparsity=sss_sparsity,
                var_slab=var_slab,
                var_spike=var_spike,
            )
        elif prior == "student":
            self.prior = StudentTPrior(df=student_df, scale=student_scale)

        self.mixture = GaussianMixture(n_components, n_features).to(device)
        self.opt = Adam(self.mixture.parameters(), lr=lr)
        self.device = device

    def log_likelihood(self, z) -> torch.Tensor:
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

    def h(self, z) -> torch.Tensor:
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

    def elbo(self, z) -> torch.Tensor:
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
        z,
        lambda_jaccard=0.5,
        threshold=1e-3,
    ) -> torch.Tensor:
        """
        Compute ELBO with regularization using average Jaccard similarity between supports.

        Parameters
        ----------
        z : torch.Tensor
            Batch of parameter samples, shape (batch_size, n_features).
        lambda_jaccard : float
            Strength of regularization penalty (penalizes overlap between supports).
        threshold : float
            Threshold for considering a feature active (nonzero).

        Returns
        -------
        torch.Tensor
            Regularized ELBO (mean over batch minus penalty).
        """
        batch_size = z.shape[0]
        h_val = self.h(z)
        elbo_val = h_val.mean()

        # Compute supports: binary mask of nonzero features
        support_mask = (torch.abs(z) > threshold).float()  # [batch_size, n_features]

        # Compute pairwise Jaccard similarities
        jaccard_vals = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                intersection = (support_mask[i] * support_mask[j]).sum()
                union = ((support_mask[i] + support_mask[j]) > 0).sum()
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = torch.tensor(0.0, device=z.device)
                jaccard_vals.append(jaccard)
        # Average Jaccard similarity
        if len(jaccard_vals) > 0:
            avg_jaccard = torch.stack(jaccard_vals).mean()
        else:
            avg_jaccard = torch.tensor(0.0, device=z.device)

        # Regularized ELBO
        elbo_reg = elbo_val - lambda_jaccard * avg_jaccard
        return elbo_reg

    def optimize(
        self,
        log_callback=None,
        regularize=False,
        lambda_jaccard=1.0,
        regularization_threshold=1e-3,
    ) -> Dict[str, List[float]]:
        """
        Main optimization loop for variational inference.

        Parameters
        ----------
        log_callback : callable, optional
            Function to call for logging every 100 iterations.
        regularize : bool, optional
            If True, use elbo_regularized during optimization (default: False) by penalizing overlap
            between solutions.
        lambda_jaccard : float, optional
            Regularization strength for penalty in the ELBO computation (default: 1.0). Higher values
            encourage more diverse solutions (lesser overlap). Used only if regularize=True.
        regularization_threshold : float, optional
            Nonzero threshold for support computation (default: 1e-3).

        Returns
        -------
        history : Dict[str, List[float]]
            Dictionary containing optimization history:
            - 'elbo': list of ELBO values
            - 'mu': list of mixture means per iteration
            - 'var': list of mixture variances per iteration
            - 'alpha': list of mixture weights per iteration
        """
        print_optimization_setting(
            n_components=self.n_components,
            sparsity=self.prior.sparsity,
            regularize=regularize,
            lambda_jaccard=lambda_jaccard,
            regularization_threshold=regularization_threshold,
            n_iterations=self.n_iter,
        )

        history = {"elbo": [], "mu": [], "var": [], "alpha": []}
        for it in range(self.n_iter):
            # Sample z ~ q(z)
            z, comp_idx = self.mixture.sample(self.batch_size)
            z = z.requires_grad_()  # Ensure gradients for z

            if regularize:
                elbo = self.elbo_regularized(
                    z,
                    lambda_jaccard=lambda_jaccard,
                    threshold=regularization_threshold,
                )
            else:
                elbo = self.elbo(z)

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
        display(Markdown("Optimization complete."))
        return history
