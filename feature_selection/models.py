"""
Models for Bayesian sparse feature selection via mixture of Gaussians.
Implements:
- Spike-and-slab prior
- Mixture of Gaussian posterior approximation
- Core parameter structures
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTPrior:
    """
    Student-t prior for Bayesian sparse modeling.
    """

    def __init__(self, df=3.0, scale=1.0):
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

    def log_prob(self, z):
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

    def __init__(self, var_slab=100.0, var_spike=0.1, w_slab=0.9, w_spike=0.1):
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

    def log_prob(self, z):
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
        spike = (
            self.w_spike * torch.exp(-0.5 * (z / self.var_spike) ** 2) / self.var_spike
        )
        logp = torch.log(slab + spike)
        return logp.sum(dim=-1)  # sum over features


class StructuredSpikeAndSlabPrior:
    """
    Structured spike-and-slab prior enforcing exactly k nonzero entries.
    For each solution, only supports of size k are allowed.
    """

    def __init__(self, n_features, sparsity, var_slab=100.0, var_spike=0.1):
        """
        Parameters
        ----------
        n_features : int
            Number of features.
        sparsity : int
            Number of nonzero entries (support size).
        var_slab : float
            Variance of the slab component.
        var_spike : float
            Variance of the spike component.
        """
        self.n_features = n_features
        self.sparsity = sparsity
        self.var_slab = var_slab
        self.var_spike = var_spike

        # For small n_features, enumerate all supports. For large, sample.
        self._all_supports = None
        if n_features <= 10 and sparsity <= 3:
            self._all_supports = list(
                itertools.combinations(range(n_features), sparsity)
            )

    def log_prob(self, z, n_support_samples=100):
        """
        Compute log-probability of z under the structured prior.
        For each support S of size k, compute p(z | S), then average.

        Parameters
        ----------
        z : torch.Tensor
            Shape (..., n_features)
        n_support_samples : int, optional
            Number of supports to sample if enumeration is infeasible.

        Returns
        -------
        torch.Tensor
            Log-probability under the prior, shape (...,)
        """
        batch_shape = z.shape[:-1]
        z_flat = z.view(-1, self.n_features)
        logps = []
        if self._all_supports is not None:
            supports = self._all_supports
        else:
            # Sample supports
            supports = [
                torch.randperm(self.n_features)[: self.sparsity].tolist()
                for _ in range(n_support_samples)
            ]
        for support in supports:
            mask = torch.zeros(self.n_features, dtype=torch.bool)
            mask[support] = True
            # Slab for mask, spike for others
            logp = torch.zeros(z_flat.shape[0], device=z.device)
            # Slab part
            logp += -0.5 * torch.sum(
                (z_flat[:, mask] / self.var_slab) ** 2
                + torch.log(torch.tensor(self.var_slab)),
                dim=1,
            )
            # Spike part
            logp += -0.5 * torch.sum(
                (z_flat[:, ~mask] / self.var_spike) ** 2
                + torch.log(torch.tensor(self.var_spike)),
                dim=1,
            )
            logps.append(logp)
        # Average over supports
        logps = torch.stack(logps, dim=1)  # [batch, n_supports]
        logp_avg = torch.logsumexp(logps, dim=1) - torch.log(
            torch.tensor(len(supports), dtype=torch.float)
        )
        return logp_avg.view(*batch_shape)


class GaussianMixture(nn.Module):
    """
    Implements a learnable mixture of diagonal Gaussians.
    Mixture parameters: means, diagonal variances, mixing weights.
    """

    def __init__(self, n_components, n_features):
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

    def get_variance(self):
        """
        Get positive variances for each component via softplus transformation.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components, n_features) with positive variances.
        """
        return F.softplus(self._log_var)

    def get_alpha(self):
        """
        Get normalized mixing weights for the mixture.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components,) with mixing weights summing to 1.
        """
        log_alpha = torch.cat(
            [self._log_alpha, torch.zeros(1, device=self._log_alpha.device)]
        )
        alpha = torch.exp(log_alpha)
        alpha = alpha / alpha.sum()
        return alpha

    def sample(self, batch_size):
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
        comp_idx = torch.multinomial(alpha, batch_size, replacement=True)
        mu = self.mu[comp_idx]
        var = self.get_variance()[comp_idx]
        eps = torch.randn(batch_size, self.n_features, device=mu.device)
        z = mu + eps * torch.sqrt(var)
        return z, comp_idx

    def log_prob(self, z):
        """
        Compute log-probability of samples under the mixture model.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape (batch_size,).
        """
        mu = self.mu.unsqueeze(0)  # [1, K, D]
        var = self.get_variance().unsqueeze(0)  # [1, K, D]
        log_alpha = torch.cat(
            [self._log_alpha, torch.zeros(1, device=self._log_alpha.device)]
        )
        alpha = torch.exp(log_alpha)
        alpha = alpha / alpha.sum()
        # log N(z | mu_k, var_k), shape [batch_size, K]
        log_prob_k = -0.5 * torch.sum(
            torch.log(var) + ((z.unsqueeze(1) - mu) ** 2) / var, dim=-1
        )
        # log mixture: logsumexp over components
        log_mix = torch.logsumexp(torch.log(alpha) + log_prob_k, dim=-1)
        return log_mix

    def component_log_prob(self, z):
        """
        Compute log-probabilities for each mixture component separately.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Log-probabilities for each component, shape (batch_size, n_components).
        """
        mu = self.mu.unsqueeze(0)
        var = self.get_variance().unsqueeze(0)
        log_prob_k = -0.5 * torch.sum(
            torch.log(var) + ((z.unsqueeze(1) - mu) ** 2) / var, dim=-1
        )
        return log_prob_k
