"""
Configuration loader for Bayesian Sparse Feature Selection

This module loads all dataset generation and algorithm settings from two JSON files:
- generated_dataset_parameters.json (dataset generation parameters)
- algorithm_settings.json (algorithm/hyperparameters)

All loaded parameters are available as module-level variables, with detailed comments.

----------------------------------------------------------------------------------------
DATASET GENERATION PARAMETERS (from generated_dataset_parameters.json)
----------------------------------------------------------------------------------------
- NSAMPLES: Number of samples (rows) in the synthetic dataset.
- NFEATURES: Number of features (columns) in the dataset.
- NSOLUTIONS: Number of distinct sparse solutions ("true" supports).
- SPARSITY: Number of nonzero features per solution (support size).
- NOISE_STD: Standard deviation of noise added to the data.
- BINARIZE: Whether to binarize the response variable
            (True => classification problem, False => regression problem).
- BINARY_RESPONSE_RATIO: Proportion of samples assigned label 1
                         (controls class balance for classification).
- RANDOM_SEED: Random seed for reproducibility.

----------------------------------------------------------------------------------------
ALGORITHM SETTINGS (from algorithm_settings.json)
----------------------------------------------------------------------------------------
- N_COMPONENTS:Number of mixture components in the variational posterior (typically >= 2 * NSOLUTIONS).
- N_ITER: Number of optimization iterations.
- PRIOR_TYPE: Prior type;
                'ss' = spike-and-slab,
                'sss' = structured spike-and-slab,
                'student' = Student-t.
- PRIOR_SPARSITY: Prior expected number of nonzero features per component
                  Should be ideally equaly to true sparsity.
                  Used only if PRIOR_TYPE='sss'.
- STUDENT_DF: Degrees of freedom for Student-t prior. Used only if PRIOR_TYPE='student'.
- STUDENT_SCALE: Scale for Student-t prior. Used only if PRIOR_TYPE='student'.
- VAR_SLAB: Variance of the 'slab' in spike-and-slab/structured spike-and-slab prior.
            Increasing VAR_SLAB makes the slab more diffuse, allowing larger coefficients.
- VAR_SPIKE: Variance of the 'spike' in spike-and-slab/structured spike-and-slab prior.
             Decreasing VAR_SPIKE makes the spike more concentrated around zero,
             promoting stronger sparsity.
- WEIGHT_SLAB: Weight of the 'slab' in spike-and-slab prior.
              Increasing WEIGHT_SLAB makes the slab more influential in the Spike-and-Slab
              mixture compared to the spike.
- WEIGHT_SPIKE: Weight of the 'spike' in spike-and-slab prior.
                Increasing WEIGHT_SPIKE makes the spike more influential in the Spike-and-Slab
                mixture compared to the slab.
- IS_REGULARIZED: Whether to use Jaccard similarity penalty for component diversity.
- LAMBDA_JACCARD: Regularization strength for the Jaccard similarity penalty.
                  Increasing LAMBDA_JACCARD encourages more diverse (less overlapping) supports.
- BATCH_SIZE: Mini-batch size for stochastic optimization.
              Larger batches give more stable gradients.
- LEARNING_RATE: Learning rate for the Adam optimizer.
                 Smaller values lead to more stable but slower convergence.

----------------------------------------------------------------------------------------
ALGORITHM SETTINGS (from solution_postprocessing_settings.json)
----------------------------------------------------------------------------------------
- DESIRED_SPARSITY: Desired number of nonzero features in the recovered solutions.
                    Only this number of features with highest |mu| are selected per solution.
- MIN_MU_THRESHOLD: Minimum absolute mu value for feature selection in recovered solutions.
                    Only features with |mu| above this threshold are considered nonzero.

Usage:
    import gemss.config as config
    # Then access, e.g., config.NSAMPLES, config.N_COMPONENTS, etc.

JSON files must be in the parent directory of the repo root (../).
"""

import json
import os
from pathlib import Path

# Locate config JSONs in the parent directory (../)
parent_dir = Path(__file__).parent.parent
dataset_json = parent_dir / "generated_dataset_parameters.json"
algo_json = parent_dir / "algorithm_settings.json"
postprocessing_json = parent_dir / "solution_postprocessing_settings.json"

# Load dataset parameters
try:
    with open(dataset_json, "r") as f:
        _dataset_params = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset parameters file not found: {dataset_json}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in dataset parameters file {dataset_json}: {e}")

# Load algorithm settings
try:
    with open(algo_json, "r") as f:
        _algo_settings = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Algorithm settings file not found: {algo_json}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in algorithm settings file {algo_json}: {e}")

# Load postprocessing settings
try:
    with open(postprocessing_json, "r") as f:
        _postproc_settings = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Postprocessing settings file not found: {postprocessing_json}"
    )
except json.JSONDecodeError as e:
    raise ValueError(
        f"Invalid JSON in postprocessing settings file {postprocessing_json}: {e}"
    )


# ------------------ DATASET GENERATION PARAMETERS ------------------
NSAMPLES = _dataset_params["NSAMPLES"]
"""Number of samples (rows) in the synthetic dataset."""

NFEATURES = _dataset_params["NFEATURES"]
"""Number of features (columns) in the dataset."""

NSOLUTIONS = _dataset_params["NSOLUTIONS"]
"""Number of distinct sparse solutions ("true" supports)."""

SPARSITY = _dataset_params["SPARSITY"]
"""Number of nonzero features per solution (support size)."""

NOISE_STD = _dataset_params["NOISE_STD"]
"""Standard deviation of noise added to the data."""

BINARIZE = _dataset_params["BINARIZE"]
"""Whether to binarize the response variable (True = classification, False = regression)."""

BINARY_RESPONSE_RATIO = _dataset_params["BINARY_RESPONSE_RATIO"]
"""Proportion of samples assigned label 1 (controls class balance for classification)."""

RANDOM_SEED = _dataset_params["RANDOM_SEED"]
"""Random seed for reproducibility."""


# ------------------ ALGORITHM SETTINGS ------------------
N_COMPONENTS = _algo_settings["N_COMPONENTS"]
"""Number of mixture components in the variational posterior (typically >= 2 * NSOLUTIONS)."""

N_ITER = _algo_settings["N_ITER"]
"""Number of optimization iterations."""

PRIOR_TYPE = _algo_settings["PRIOR_TYPE"]
"""Prior type; 'ss' = spike-and-slab, 'sss' = structured spike-and-slab, 'student' = Student-t."""

PRIOR_SPARSITY = _algo_settings.get("PRIOR_SPARSITY", None)
"""Prior expected number of nonzero features per component (used if PRIOR_TYPE='sss').
Should be ideally equal to true sparsity."""

STUDENT_DF = _algo_settings["STUDENT_DF"]
"""Degrees of freedom for Student-t prior (used if PRIOR_TYPE='student')."""

STUDENT_SCALE = _algo_settings["STUDENT_SCALE"]
"""Scale for Student-t prior (used if PRIOR_TYPE='student')."""

VAR_SLAB = _algo_settings["VAR_SLAB"]
"""Variance of the 'slab' in spike-and-slab/structured spike-and-slab prior."""

VAR_SPIKE = _algo_settings["VAR_SPIKE"]
"""Variance of the 'spike' in spike-and-slab/structured spike-and-slab prior."""

WEIGHT_SLAB = _algo_settings["WEIGHT_SLAB"]
"""Weight of the 'slab' in spike-and-slab prior."""

WEIGHT_SPIKE = _algo_settings["WEIGHT_SPIKE"]
"""Weight of the 'spike' in spike-and-slab prior."""

IS_REGULARIZED = _algo_settings["IS_REGULARIZED"]
"""Whether to use Jaccard similarity penalty for component diversity."""

LAMBDA_JACCARD = _algo_settings["LAMBDA_JACCARD"]
"""Regularization strength for the Jaccard similarity penalty."""

BATCH_SIZE = _algo_settings["BATCH_SIZE"]
"""Mini-batch size for stochastic optimization."""

LEARNING_RATE = _algo_settings["LEARNING_RATE"]
"""Learning rate for the Adam optimizer."""

# ------------------ POSTPROCESSING SETTINGS ------------------
DESIRED_SPARSITY = _postproc_settings["DESIRED_SPARSITY"]
"""Desired number of nonzero features in the recovered solutions.
Only this number of features with highest |mu| are selected per final solution.
Should be ideally equal to true sparsity."""

MIN_MU_THRESHOLD = _postproc_settings["MIN_MU_THRESHOLD"]
"""Only features with |mu| above this minimal threshold are considered nonzero
and are included in the 'full' solutions."""


def check_sparsities():
    """
    Print the sparsity settings for verification. Ideally, all three sparsities are equal.
    In practice, the desired sparsity might be set slightly higher to cover all true features.
    The prior sparsity is only used if PRIOR_TYPE='sss' but it is not a hard constraint.
    """
    print("Sparsity settings:")
    print(f" - True sparsity: {SPARSITY}")
    print(f" - Prior sparsity: {PRIOR_SPARSITY}")
    print(f" - Desired sparsity: {DESIRED_SPARSITY}")
    return


def as_dict():
    """
    Return all config variables as a dictionary (for logging or debugging).
    """
    out = {}
    for k in (
        list(_dataset_params.keys())
        + list(_algo_settings.keys())
        + list(_postproc_settings.keys())
    ):
        out[k] = globals()[k]
    return out
