"""Tests for the regularized ELBO (Evidence Lower Bound) in BayesianFeatureSelector.

- test_elbo_regularized_matches_manual_penalty: elbo_regularized equals ELBO minus
  lambda_jaccard * manual average Jaccard over latent supports.
- test_elbo_regularized_batch_size_one_no_penalty: with batch size 1, regularized
  ELBO equals unregularized ELBO (no Jaccard penalty).
- test_elbo_regularized_lambda_zero_matches_elbo: lambda_jaccard=0 gives the same
  value as plain elbo().
"""

import numpy as np
import torch

from gemss.feature_selection.inference import BayesianFeatureSelector


def _make_selector() -> BayesianFeatureSelector:
    X = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 2.0, 1.0],
            [3.0, -1.0, 0.0],
            [-2.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, -0.5, 2.0, 0.0], dtype=np.float32)
    return BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=1,
        X=X,
        y=y,
        batch_size=4,
        n_iter=1,
        device="cpu",
    )


def _manual_avg_jaccard(z: torch.Tensor) -> torch.Tensor:
    sigmoid_coeff = 100.0
    support_mask = torch.sigmoid(sigmoid_coeff * torch.abs(z))

    jaccard_vals = []
    batch_size = z.shape[0]
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            intersection = (support_mask[i] * support_mask[j]).sum()
            union = ((support_mask[i] + support_mask[j]) > 0).sum()
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = torch.tensor(0.0, device=z.device)
            jaccard_vals.append(jaccard)

    if len(jaccard_vals) == 0:
        return torch.tensor(0.0, device=z.device)

    return torch.stack(jaccard_vals).mean()


def test_elbo_regularized_matches_manual_penalty() -> None:
    selector = _make_selector()
    z = torch.tensor(
        [[0.0, 0.5, -1.0], [2.0, -0.5, 0.0], [-1.5, 1.0, 0.25]],
        dtype=torch.float32,
    )
    lambda_jaccard = 2.5

    expected = selector.elbo(z) - lambda_jaccard * _manual_avg_jaccard(z)
    actual = selector.elbo_regularized(z, lambda_jaccard=lambda_jaccard)

    assert torch.allclose(actual, expected)


def test_elbo_regularized_batch_size_one_no_penalty() -> None:
    selector = _make_selector()
    z = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)

    expected = selector.elbo(z)
    actual = selector.elbo_regularized(z, lambda_jaccard=5.0)

    assert torch.allclose(actual, expected)


def test_elbo_regularized_lambda_zero_matches_elbo() -> None:
    selector = _make_selector()
    z = torch.tensor([[0.1, -0.2, 0.3], [1.0, -1.0, 0.5]], dtype=torch.float32)

    expected = selector.elbo(z)
    actual = selector.elbo_regularized(z, lambda_jaccard=0.0)

    assert torch.allclose(actual, expected)
