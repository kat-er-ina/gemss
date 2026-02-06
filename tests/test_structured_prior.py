"""Tests for StructuredSpikeAndSlabPrior (log_prob and sampling).

- test_structured_log_prob_matches_enumeration: log_prob(z) matches manual
  average log-prob over all fixed-sparsity support sets.
- test_structured_log_prob_preserves_batch_shape: log_prob returns correct
  batch shape (e.g. (2, 3)) for batched z.
- test_structured_log_prob_batch_size_one: log_prob works for single-sample
  input and returns shape (1,).
- test_structured_log_prob_sampling_uses_rng_and_coeff: with
  n_support_samples and sample_more_priors_coeff, log_prob uses RNG and
  matches manual log_prob over the sampled supports.
- test_structured_log_prob_sparsity_zero: sparsity=0 (all spike, single empty
  support); log_prob matches manual.
- test_structured_prior_raises_when_zero_support_samples: when sampling
  supports (large n_features), log_prob(..., n_support_samples=0) raises
  ValueError (zero supports).
"""

import itertools

import numpy as np
import torch

import pytest

from gemss.feature_selection.models import StructuredSpikeAndSlabPrior


def _manual_log_prob(
    z: torch.Tensor,
    n_features: int,
    var_slab: float,
    var_spike: float,
    supports: list[list[int]],
) -> torch.Tensor:
    batch_shape = z.shape[:-1]
    z_flat = z.view(-1, n_features)
    logps = []

    slab_const = torch.log(torch.tensor(var_slab, dtype=z.dtype, device=z.device))
    spike_const = torch.log(torch.tensor(var_spike, dtype=z.dtype, device=z.device))

    for support in supports:
        mask = torch.zeros(n_features, dtype=torch.bool, device=z.device)
        if len(support) > 0:
            mask[list(support)] = True

        logp = torch.zeros(z_flat.shape[0], device=z.device, dtype=z.dtype)

        if mask.any():
            slab_term = (z_flat[:, mask] / var_slab) ** 2
            logp += -0.5 * torch.sum(slab_term + slab_const, dim=1)

        if (~mask).any():
            spike_term = (z_flat[:, ~mask] / var_spike) ** 2
            logp += -0.5 * torch.sum(spike_term + spike_const, dim=1)

        logps.append(logp)

    logps = torch.stack(logps, dim=1)
    logp_avg = torch.logsumexp(logps, dim=1) - torch.log(
        torch.tensor(len(supports), dtype=z.dtype, device=z.device)
    )
    return logp_avg.view(*batch_shape)


def test_structured_log_prob_matches_enumeration() -> None:
    torch.manual_seed(0)
    n_features = 4
    sparsity = 2
    var_slab = 2.0
    var_spike = 0.5

    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        var_slab=var_slab,
        var_spike=var_spike,
    )

    z = torch.tensor(
        [[0.1, -0.3, 0.5, 1.2], [1.0, -0.4, 0.2, -0.8]],
        dtype=torch.float32,
    )
    supports = list(itertools.combinations(range(n_features), sparsity))
    expected = _manual_log_prob(z, n_features, var_slab, var_spike, supports)

    actual = prior.log_prob(z)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_structured_log_prob_preserves_batch_shape() -> None:
    torch.manual_seed(1)
    n_features = 5
    sparsity = 3
    var_slab = 1.5
    var_spike = 0.2

    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        var_slab=var_slab,
        var_spike=var_spike,
    )

    z = torch.randn(2, 3, n_features)
    supports = list(itertools.combinations(range(n_features), sparsity))
    expected = _manual_log_prob(z, n_features, var_slab, var_spike, supports)

    actual = prior.log_prob(z)

    assert actual.shape == (2, 3)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_structured_log_prob_batch_size_one() -> None:
    torch.manual_seed(3)
    n_features = 5
    sparsity = 3
    var_slab = 2.5
    var_spike = 0.4

    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        var_slab=var_slab,
        var_spike=var_spike,
    )

    z = torch.randn(1, n_features)
    supports = list(itertools.combinations(range(n_features), sparsity))
    expected = _manual_log_prob(z, n_features, var_slab, var_spike, supports)

    actual = prior.log_prob(z)

    assert actual.shape == (1,)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_structured_log_prob_sampling_uses_rng_and_coeff() -> None:
    torch.manual_seed(123)
    n_features = 11
    sparsity = 4
    var_slab = 3.0
    var_spike = 0.7
    sample_more_priors_coeff = 2.0
    n_support_samples = 4

    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        sample_more_priors_coeff=sample_more_priors_coeff,
        var_slab=var_slab,
        var_spike=var_spike,
    )

    z = torch.randn(3, n_features)

    rng_state = torch.random.get_rng_state()
    actual = prior.log_prob(z, n_support_samples=n_support_samples)

    torch.random.set_rng_state(rng_state)
    expected_samples = int(np.round(n_support_samples * sample_more_priors_coeff).astype(int))
    scores = torch.rand(expected_samples, n_features, device=z.device)
    supports = torch.topk(scores, k=sparsity, dim=1).indices.tolist()
    expected = _manual_log_prob(z, n_features, var_slab, var_spike, supports)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_structured_log_prob_sparsity_zero() -> None:
    torch.manual_seed(0)
    n_features = 3
    sparsity = 0
    var_slab = 2.0
    var_spike = 0.5
    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        var_slab=var_slab,
        var_spike=var_spike,
    )
    z = torch.tensor([[0.1, -0.3, 0.5], [1.0, -0.4, 0.2]], dtype=torch.float32)
    supports = list(itertools.combinations(range(n_features), sparsity))
    expected = _manual_log_prob(z, n_features, var_slab, var_spike, supports)
    actual = prior.log_prob(z)
    assert actual.shape == (2,)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_structured_prior_raises_when_zero_support_samples() -> None:
    n_features = 11
    sparsity = 4
    prior = StructuredSpikeAndSlabPrior(
        n_features=n_features,
        sparsity=sparsity,
        var_slab=1.0,
        var_spike=0.1,
    )
    z = torch.randn(2, n_features)
    with pytest.raises(ValueError, match='at least one support'):
        prior.log_prob(z, n_support_samples=0)
