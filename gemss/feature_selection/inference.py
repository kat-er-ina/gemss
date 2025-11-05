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
from .models import (
    StudentTPrior,
    SpikeAndSlabPrior,
    StructuredSpikeAndSlabPrior,
    GaussianMixture,
)
from gemss.utils import print_nice_optimization_settings, myprint


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
        n_features: int,
        n_components: int,
        X,
        y,
        prior: Literal["ss", "sss", "student"] = "sss",
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
        device: Literal["cpu", "cuda"] = "cpu",
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
        """
        self.n_features = n_features
        self.n_components = n_components

        # Convert data to tensors, preserving NaN values
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

        # Validate that response values don't contain NaN
        if torch.isnan(self.y).any():
            raise ValueError(
                "Response variable (y) contains NaN values. Please remove samples with missing responses before feature selection."
            )

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
                sample_more_priors_coeff=sample_more_priors_coeff,
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
        # Check if X has missing values
        if torch.isnan(self.X).any():
            return self._log_likelihood_with_missing(z)
        else:
            # Standard computation for complete data
            pred = torch.matmul(z, self.X.T)  # [batch_size, n_samples]
            mse = ((pred - self.y.unsqueeze(0)) ** 2).sum(dim=-1)  # sum over samples
            return -0.5 * mse

    def _log_likelihood_with_missing(self, z) -> torch.Tensor:
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
        n_samples = self.X.shape[0]

        # Compute log-likelihood for each data sample separately
        log_likes = []

        for i in range(n_samples):
            x_i = self.X[i]  # [n_features]
            y_i = self.y[i]  # scalar

            # Skip if response is missing (shouldn't happen if properly preprocessed)
            if torch.isnan(y_i):
                continue

            # Find observed features for this sample
            observed_mask = ~torch.isnan(x_i)  # [n_features]

            if observed_mask.sum() == 0:
                # If no features observed, skip this sample
                continue

            # Use only observed features for prediction
            x_obs = x_i[observed_mask]  # [n_observed]
            z_obs = z[:, observed_mask]  # [batch_size, n_observed]

            # Compute prediction using observed features only
            pred_i = torch.matmul(z_obs, x_obs)  # [batch_size]

            # Compute log-likelihood for this sample
            # Scale by number of observed features to maintain comparable magnitudes
            mse_i = (pred_i - y_i) ** 2
            log_like_i = -0.5 * mse_i

            log_likes.append(log_like_i)

        if len(log_likes) == 0:
            # If no valid samples, return zero log-likelihood
            return torch.zeros(batch_size, device=z.device)

        # Sum log-likelihoods across all data samples
        total_log_like = torch.stack(log_likes, dim=0).sum(dim=0)  # [batch_size]

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
        support_mask = torch.sigmoid(
            sigmoid_coeff * torch.abs(z)
        )  # [batch_size, n_features]

        # Compute pairwise Jaccard similarities
        jaccard_vals = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                intersection = (support_mask[i] * support_mask[j]).sum()  #
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
        elbo_reg = self.elbo(z) - lambda_jaccard * avg_jaccard
        return elbo_reg

    def optimize(
        self,
        log_callback: callable = None,
        regularize: bool = False,
        lambda_jaccard: float = 10.0,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Main optimization loop for the feature selector.

        Handles missing values internally using masking, ensures valid alphas,
        and prevents propagation of NaNs in mixture weights.

         Parameters
        ----------
        log_callback : callable, optional
            Function to call for logging every 100 iterations.
        regularize : bool, optional
            If True, use elbo_regularized during optimization (default: False) by penalizing overlap
            between solutions.
        lambda_jaccard : float, optional
            Regularization strength for penalty in the ELBO computation (default: 10.0). Higher values
            encourage more diverse solutions (lesser overlap). Used only if regularize=True.
        verbose : bool, optional
            If True, print optimization settings and progress (default: True).

        Returns
        -------
        history : Dict[str, List[float]]
            Dictionary containing optimization history:
            - 'elbo': list of ELBO values
            - 'mu': list of mixture means per iteration
            - 'var': list of mixture variances per iteration
            - 'alpha': list of mixture weights per iteration
        """

        if verbose:
            print_nice_optimization_settings(
                n_components=self.n_components,
                regularize=regularize,
                lambda_jaccard=lambda_jaccard,
                n_iterations=self.n_iter,
                prior_settings={
                    "prior_name": type(self.prior).__name__,
                    "prior_sparsity": getattr(self.prior, "sparsity", None),
                    "var_slab": getattr(self.prior, "var_slab", None),
                    "var_spike": getattr(self.prior, "var_spike", None),
                    "weight_slab": getattr(self.prior, "weight_slab", None),
                    "weight_spike": getattr(self.prior, "weight_spike", None),
                    "student_df": getattr(self.prior, "df", None),
                    "student_scale": getattr(self.prior, "scale", None),
                },
            )

        history = {"elbo": [], "mu": [], "var": [], "alpha": []}

        for it in range(self.n_iter):
            # Sample z ~ q(z) - this should always produce complete samples (no NaNs)
            z, comp_idx = self.mixture.sample(self.batch_size)
            # Note: z automatically has requires_grad=True through reparameterization

            if regularize:
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
            history["elbo"].append(elbo.item())
            history["mu"].append(self.mixture.mu.detach().cpu().clone().numpy())
            history["var"].append(
                self.mixture.get_variance().detach().cpu().clone().numpy()
            )

            # Store alphas safely
            alpha_val = self.mixture.get_alpha()
            history["alpha"].append(alpha_val.detach().cpu().clone().numpy())

            if log_callback and it % 100 == 0:
                log_callback(it, elbo.item(), self.mixture)
        if verbose:
            myprint("Optimization complete.", use_markdown=True)
        return history
