"""
Models for Bayesian sparse feature selection via mixture of Gaussians.
Implements:
- Spike-and-slab prior
- Mixture of Gaussian posterior approximation
- Core parameter structures
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTPrior:
    """
    Student-t prior for Bayesian sparse modeling.
    """

    def __init__(self, df: float = 3.0, scale: float = 1.0):
        """
        Parameters
        ----------
        df : float
            Degrees of freedom (controls heaviness of tails; lower = heavier).
        scale : float
            Scale (analogous to standard deviation).
        """
        self.df = df
        self.scale = scale

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of the Student-t prior for input z.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (..., D), where D is the number of features.

        Returns
        -------
        torch.Tensor
            Log-probability summed over features, shape (...,).
        """
        # Student-t log-density (up to constant)
        logp = -0.5 * (self.df + 1) * torch.log(1 + (z / self.scale) ** 2 / self.df)
        return logp.sum(dim=-1)


class SpikeAndSlabPrior:
    """
    Implements the log-probability of the spike-and-slab prior.
    p(z) = prod_i [w_slab * N(z_i | 0, var_slab) + w_spike * N(z_i | 0, var_spike)]
    """

    def __init__(
        self,
        var_slab: float = 100.0,
        var_spike: float = 0.1,
        w_slab: float = 0.9,
        w_spike: float = 0.1,
    ):
        """
        Parameters
        ----------
        var_slab : float, optional
            Variance of the slab component (default: 100.0).
        var_spike : float, optional
            Variance of the spike component (default: 0.1).
        w_slab : float, optional
            Weight of the slab component (default: 0.9).
        w_spike : float, optional
            Weight of the spike component (default: 0.1).
        """
        self.var_slab = var_slab
        self.var_spike = var_spike
        self.w_slab = w_slab
        self.w_spike = w_spike

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of the spike-and-slab prior for input z.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (..., D), where D is the number of features.

        Returns
        -------
        torch.Tensor
            Log-probability summed over features, shape (...,).
        """
        slab = self.w_slab * torch.exp(-0.5 * (z / self.var_slab) ** 2) / self.var_slab
        spike = self.w_spike * torch.exp(-0.5 * (z / self.var_spike) ** 2) / self.var_spike
        logp = torch.log(slab + spike)
        return logp.sum(dim=-1)  # sum over features


class StructuredSpikeAndSlabPrior:
    """
    Structured spike-and-slab prior enforcing exactly k nonzero entries.
    For each solution, only supports of size k are allowed.
    """

    def __init__(
        self,
        n_features: int,
        sparsity: int,
        sample_more_priors_coeff: float = 1.0,
        var_slab: float = 100.0,
        var_spike: float = 0.1,
    ):
        """
        Parameters
        ----------
        n_features : int
            Number of features.
        sparsity : int
            Number of nonzero entries (support size).
        sample_more_priors_coeff : float, optional
            Coefficient to scale number of sampled supports if enumeration is infeasible.
            Default is 1.0 (no scaling).
        var_slab : float
            Variance of the slab component.
        var_spike : float
            Variance of the spike component.
        """
        self.n_features = n_features
        self.sparsity = sparsity
        self.var_slab = var_slab
        self.var_spike = var_spike
        self.sample_more_priors_coeff = sample_more_priors_coeff

        # For small n_features, enumerate all supports. For large, sample.
        self._all_supports = None
        if n_features <= 10 and sparsity <= 3:
            self._all_supports = list(itertools.combinations(range(n_features), sparsity))

    def log_prob(self, z: torch.Tensor, n_support_samples: int = 100) -> torch.Tensor:
        """
        Compute log-probability of z under the structured prior.
        For each support S of size k, compute p(z | S), then average.
        The default number of supports to sample is 100 if enumeration is infeasible.
        The chosen number of supports is scaled by sample_more_priors_coeff.

        Parameters
        ----------
        z : torch.Tensor
            Shape (..., n_features)
        n_support_samples : int, optional
            Number of supports to sample if enumeration is infeasible. Default is 100.

        Returns
        -------
        torch.Tensor
            Log-probability under the prior, shape (...,)
        """
        batch_shape = z.shape[:-1]
        z_flat = z.view(-1, self.n_features)
        if self._all_supports is not None:
            supports_tensor = torch.tensor(self._all_supports, dtype=torch.long, device=z.device)
        else:
            if self.sample_more_priors_coeff:
                n_support_samples = np.round(
                    n_support_samples * self.sample_more_priors_coeff
                ).astype(int)
            # Sample supports (vectorized) on the same device as z
            n_support_samples = int(n_support_samples)
            if self.sparsity == 0:
                supports_tensor = torch.empty(
                    n_support_samples, 0, dtype=torch.long, device=z.device
                )
            else:
                scores = torch.rand(n_support_samples, self.n_features, device=z.device)
                supports_tensor = torch.topk(scores, k=self.sparsity, dim=1).indices

        n_supports = supports_tensor.shape[0]
        if n_supports == 0:
            raise ValueError('StructuredSpikeAndSlabPrior requires at least one support.')

        if self.sparsity == 0:
            supports_tensor = supports_tensor[:, :0]

        var_slab = torch.as_tensor(self.var_slab, device=z.device, dtype=z.dtype)
        var_spike = torch.as_tensor(self.var_spike, device=z.device, dtype=z.dtype)
        slab_const = torch.log(var_slab)
        spike_const = torch.log(var_spike)

        slab_term = -0.5 * ((z_flat / var_slab) ** 2 + slab_const)
        spike_term = -0.5 * ((z_flat / var_spike) ** 2 + spike_const)

        total_spike = spike_term.sum(dim=1)
        if self.sparsity == 0:
            logps = total_spike[:, None].expand(-1, n_supports)
        else:
            delta = slab_term - spike_term
            logps = total_spike[:, None] + delta[:, supports_tensor].sum(dim=-1)

        # Average over supports
        logp_avg = torch.logsumexp(logps, dim=1) - torch.log(
            torch.as_tensor(n_supports, dtype=z.dtype, device=z.device)
        )
        return logp_avg.view(*batch_shape)


class GaussianMixture(nn.Module):
    """
    Implements a learnable mixture of diagonal Gaussians.
    Mixture parameters: means, diagonal variances, mixing weights.
    Includes robust support for missing features (NaNs) during log probability evaluation.
    """

    def __init__(self, n_components: int, n_features: int):
        """
        Parameters
        ----------
        n_components : int
            Number of mixture components.
        n_features : int
            Number of features (dimension of each component).
        """
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

        # Means [n_components, n_features]
        self.mu = nn.Parameter(torch.randn(n_components, n_features))
        # Variances [n_components, n_features], positive via softplus
        self._log_var = nn.Parameter(torch.zeros(n_components, n_features))
        # Mixing log-weights [n_components-1] (last alpha is 1-sum(...))
        self._log_alpha = nn.Parameter(torch.zeros(n_components - 1))

    def get_variance(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components, n_features) with positive variances.
        """
        return F.softplus(self._log_var)

    def get_alpha(self) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components,) with mixing weights summing to 1.
            Handles NaN robustness by replacing any NaN with 0 and re-normalizing.
        """
        log_alpha = torch.cat([self._log_alpha, torch.zeros(1, device=self._log_alpha.device)])
        alpha = torch.exp(log_alpha)
        # NaN-safe: Replace NaNs with zeros, renormalize. If degenerate, fallback to uniform.
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
        if alpha.sum() == 0 or not torch.isfinite(alpha).all():
            alpha = torch.ones_like(alpha) / len(alpha)
        else:
            alpha = alpha / alpha.sum()
        return alpha

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points from the mixture using the reparameterization trick.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.

        Returns
        -------
        z : torch.Tensor
            Sampled points of shape (batch_size, n_features).
        comp_idx : torch.Tensor
            Indices of chosen components for each sample, shape (batch_size,).
        """
        alpha = self.get_alpha()

        # Use multinomial sampling for component selection
        comp_idx = torch.multinomial(alpha, batch_size, replacement=True)

        # Reparameterized sampling from selected components
        mu = self.mu[comp_idx]  # [batch_size, n_features]
        var = self.get_variance()[comp_idx]  # [batch_size, n_features]

        # Standard reparameterization trick
        eps = torch.randn(batch_size, self.n_features, device=mu.device)
        z = mu + eps * torch.sqrt(var)

        return z, comp_idx

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of samples under the mixture model.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-probabilities (in mixture) of shape (batch_size,).
        """
        # Ensure z has no NaNs for variational inference
        if torch.isnan(z).any():
            raise ValueError(
                'z contains NaN values - variational inference requires complete latent samples'
            )

        mu = self.mu.unsqueeze(0)  # [1, K, D]
        var = self.get_variance().unsqueeze(0)  # [1, K, D]
        alpha = self.get_alpha()  # [K]

        # Standard mixture log-probability computation
        # log N(z | mu_k, var_k), shape [batch, K]
        log_prob_k = -0.5 * torch.sum(
            torch.log(2 * torch.pi * var) + ((z.unsqueeze(1) - mu) ** 2) / var, dim=-1
        )

        # log mixture: logsumexp over components
        log_prob_weighted = log_prob_k + torch.log(alpha.unsqueeze(0))
        return torch.logsumexp(log_prob_weighted, dim=-1)  # [batch_size]

    def log_prob_masked(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-probability of each sample under each mixture component,
        masking missing values (NaNs) locally.

        Note: This method is for data analysis only, not for variational inference.
        Returns only the per-component likelihoods (not marginalized), averaged over observed
        features.

        Parameters
        ----------
        z : torch.Tensor
            Data tensor of shape (batch_size, n_features) with possible NaNs.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size, n_components),
            averaged over observed features for each sample.
        """
        batch, n_features = z.shape
        mu = self.mu.unsqueeze(0)  # [1, K, D]
        var = self.get_variance().unsqueeze(0)  # [1, K, D]
        mask = ~torch.isnan(z)  # [batch, features]
        z_filled = torch.where(mask, z, torch.zeros_like(z))
        z_b = z_filled.unsqueeze(1)  # [batch, 1, features]
        mask_b = mask.unsqueeze(1)  # [batch, 1, features]

        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((z_b - mu) ** 2) / var)
        log_prob = log_prob * mask_b
        f_sum = log_prob.sum(-1)
        n_obs = mask_b.sum(-1).clamp(min=1)
        log_prob_mean = f_sum / n_obs
        return log_prob_mean  # [batch, n_components]

    def log_marginal(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the marginalized log-probability (logsumexp over components),
        using NaN-masked per-component log-likelihood.

        Note: This method is for data analysis only, not for variational inference.

        Parameters
        ----------
        z : torch.Tensor
            Input with shape (batch_size, n_features).
        Returns
        -------
        torch.Tensor
            Marginal log-likelihood, (batch_size,).
        """
        component_logp = self.log_prob_masked(z)  # (batch, n_components)
        alpha = self.get_alpha()  # (n_components,)

        log_weighted = component_logp + torch.log(alpha)
        return torch.logsumexp(log_weighted, dim=-1)
