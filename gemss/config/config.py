"""
Configuration loader for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This module provides efficient loading and management of experiment parameters
from JSON configuration files co-located with this module.

Features:
- Lazy loading with caching
- Comprehensive parameter validation
- Structured parameter access by category
- Rich display functionality for notebooks
- Efficient dictionary conversion for logging

Usage:
    import gemss.config as config
    # Access parameters: config.NSAMPLES, config.N_COMPONENTS, etc.
    # Display configuration: config.display_current_config(config.as_dict())
"""

import json
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from functools import lru_cache

from .constants import CONFIG_FILES, PROJECT_NAME


class ConfigurationManager:
    """
    Efficient configuration manager with lazy loading and parameter categorization.
    """

    # Parameter category definitions
    DATASET_PARAMS = {
        "NSAMPLES",
        "NFEATURES",
        "NSOLUTIONS",
        "SPARSITY",
        "NOISE_STD",
        "BINARIZE",
        "BINARY_RESPONSE_RATIO",
        "RANDOM_SEED",
    }

    ALGORITHM_PARAMS = {
        "N_COMPONENTS",
        "N_ITER",
        "PRIOR_TYPE",
        "PRIOR_SPARSITY",
        "SAMPLE_MORE_PRIORS_COEFF",
        "STUDENT_DF",
        "STUDENT_SCALE",
        "VAR_SLAB",
        "VAR_SPIKE",
        "WEIGHT_SLAB",
        "WEIGHT_SPIKE",
        "IS_REGULARIZED",
        "LAMBDA_JACCARD",
        "BATCH_SIZE",
        "LEARNING_RATE",
    }

    POSTPROCESSING_PARAMS = {"DESIRED_SPARSITY", "MIN_MU_THRESHOLD"}

    # Parameter descriptions for display
    PARAM_DESCRIPTIONS = {
        # Dataset generation
        "NSAMPLES": "Number of samples (rows) in the synthetic dataset",
        "NFEATURES": "Number of features (columns) in the dataset",
        "NSOLUTIONS": "Number of distinct sparse solutions ('true' supports)",
        "SPARSITY": "Number of nonzero features per solution (support size)",
        "NOISE_STD": "Standard deviation of noise added to the data",
        "BINARIZE": "Whether to binarize the response variable",
        "BINARY_RESPONSE_RATIO": "Proportion of samples assigned label 1",
        "RANDOM_SEED": "Random seed for reproducibility",
        # Algorithm settings
        "N_COMPONENTS": "Number of mixture components in variational posterior",
        "N_ITER": "Number of optimization iterations",
        "PRIOR_TYPE": "Prior type ('ss', 'sss', or 'student')",
        "PRIOR_SPARSITY": "Prior expected number of nonzero features per component",
        "SAMPLE_MORE_PRIORS_COEFF": "Coefficient for increased support sampling",
        "STUDENT_DF": "Degrees of freedom for Student-t prior",
        "STUDENT_SCALE": "Scale parameter for Student-t prior",
        "VAR_SLAB": "Variance of the 'slab' component",
        "VAR_SPIKE": "Variance of the 'spike' component",
        "WEIGHT_SLAB": "Weight of the 'slab' component",
        "WEIGHT_SPIKE": "Weight of the 'spike' component",
        "IS_REGULARIZED": "Whether to use Jaccard similarity penalty",
        "LAMBDA_JACCARD": "Regularization strength for Jaccard penalty",
        "BATCH_SIZE": "Mini-batch size for optimization",
        "LEARNING_RATE": "Learning rate for Adam optimizer",
        # Postprocessing
        "DESIRED_SPARSITY": "Desired number of features in final solution",
        "MIN_MU_THRESHOLD": "Minimum mu threshold for feature selection",
    }

    def __init__(self):
        self._config_dir = Path(__file__).parent
        self._cache = {}

    @lru_cache(maxsize=None)
    def _load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load and cache JSON file contents."""
        file_path = self._config_dir / filename
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def get_dataset_params(self) -> Dict[str, Any]:
        """Get dataset generation parameters."""
        return self._load_json_file(CONFIG_FILES["DATASET"])

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self._load_json_file(CONFIG_FILES["ALGORITHM"])

    def get_postprocessing_params(self) -> Dict[str, Any]:
        """Get postprocessing parameters."""
        return self._load_json_file(CONFIG_FILES["POSTPROCESSING"])

    def get_all_params(self) -> Dict[str, Any]:
        """Get all parameters in a single dictionary."""
        if "all_params" not in self._cache:
            all_params = {}
            all_params.update(self.get_dataset_params())
            all_params.update(self.get_algorithm_params())
            all_params.update(self.get_postprocessing_params())
            self._cache["all_params"] = all_params
        return self._cache["all_params"]

    def get_params_by_category(self, category: str) -> Dict[str, Any]:
        """Get parameters filtered by category."""
        all_params = self.get_all_params()

        if category == "dataset":
            return {k: v for k, v in all_params.items() if k in self.DATASET_PARAMS}
        elif category == "algorithm":
            return {k: v for k, v in all_params.items() if k in self.ALGORITHM_PARAMS}
        elif category == "postprocessing":
            return {
                k: v for k, v in all_params.items() if k in self.POSTPROCESSING_PARAMS
            }
        elif category == "all":
            return all_params
        else:
            raise ValueError(f"Unknown category: {category}")


# Global configuration manager instance
_config_manager = ConfigurationManager()

# Load all parameters at module level for backward compatibility
_all_params = _config_manager.get_all_params()

# Dataset parameters
NSAMPLES = _all_params["NSAMPLES"]
NFEATURES = _all_params["NFEATURES"]
NSOLUTIONS = _all_params["NSOLUTIONS"]
SPARSITY = _all_params["SPARSITY"]
NOISE_STD = _all_params["NOISE_STD"]
BINARIZE = _all_params["BINARIZE"]
BINARY_RESPONSE_RATIO = _all_params["BINARY_RESPONSE_RATIO"]
RANDOM_SEED = _all_params["RANDOM_SEED"]

# Algorithm parameters
N_COMPONENTS = _all_params["N_COMPONENTS"]
N_ITER = _all_params["N_ITER"]
PRIOR_TYPE = _all_params["PRIOR_TYPE"]
PRIOR_SPARSITY = _all_params.get("PRIOR_SPARSITY")
SAMPLE_MORE_PRIORS_COEFF = _all_params.get("SAMPLE_MORE_PRIORS_COEFF", 1.0)
STUDENT_DF = _all_params["STUDENT_DF"]
STUDENT_SCALE = _all_params["STUDENT_SCALE"]
VAR_SLAB = _all_params["VAR_SLAB"]
VAR_SPIKE = _all_params["VAR_SPIKE"]
WEIGHT_SLAB = _all_params["WEIGHT_SLAB"]
WEIGHT_SPIKE = _all_params["WEIGHT_SPIKE"]
IS_REGULARIZED = _all_params["IS_REGULARIZED"]
LAMBDA_JACCARD = _all_params["LAMBDA_JACCARD"]
BATCH_SIZE = _all_params["BATCH_SIZE"]
LEARNING_RATE = _all_params["LEARNING_RATE"]

# Postprocessing parameters
DESIRED_SPARSITY = _all_params["DESIRED_SPARSITY"]
MIN_MU_THRESHOLD = _all_params["MIN_MU_THRESHOLD"]


def check_sparsities() -> None:
    """Print sparsity settings for verification."""
    print("Sparsity settings:")
    print(f" - True sparsity: {SPARSITY}")
    print(f" - Prior sparsity: {PRIOR_SPARSITY}")
    print(f" - Desired sparsity: {DESIRED_SPARSITY}")


def as_dict() -> Dict[str, Any]:
    """Return all configuration parameters as a dictionary."""
    return _config_manager.get_all_params().copy()


def get_params_by_category(category: str) -> Dict[str, Any]:
    """
    Get parameters filtered by category.

    Parameters
    ----------
    category : str
        Category name: 'dataset', 'algorithm', 'postprocessing', or 'all'

    Returns
    -------
    Dict[str, Any]
        Filtered parameters dictionary
    """
    return _config_manager.get_params_by_category(category)


def display_current_config(
    constants: Optional[Dict[str, Any]] = None,
    constant_type: Literal["algorithm", "postprocessing", "dataset", "all"] = "all",
) -> None:
    """
    Display configuration parameters in a formatted table.

    Parameters
    ----------
    constants : Dict[str, Any], optional
        Configuration parameters to display. If None, uses current config.
    constant_type : str
        Parameter category to display: 'algorithm', 'postprocessing', 'dataset', 'all'
    """
    try:
        from IPython.display import display, Markdown
    except ImportError:
        print("IPython not available. Cannot display formatted configuration.")
        return

    if constants is None:
        constants = as_dict()

    # Map legacy category names
    category_map = {"artificial_data": "dataset", "algorithm_and_postprocessing": "all"}
    category = category_map.get(constant_type, constant_type)

    if category != "all":
        filtered_constants = get_params_by_category(category)
        # Filter constants to only include requested parameters
        constants = {k: v for k, v in constants.items() if k in filtered_constants}

    section_title = f"{category} parameters" if category != "all" else "all parameters"

    display(Markdown(f"## Current configuration: {section_title}"))

    if not constants:
        display(Markdown("No parameters to display."))
        return

    # Create formatted table
    table_lines = [
        "| Parameter | Current Value | Description |",
        "|-----------|---------------|-------------|",
    ]

    for param_name in sorted(constants.keys()):
        param_value = constants[param_name]
        description = _config_manager.PARAM_DESCRIPTIONS.get(
            param_name, "Configuration parameter"
        )

        # Format value based on type
        if isinstance(param_value, float):
            formatted_value = f"{param_value:.6g}"
        elif isinstance(param_value, str):
            formatted_value = f'"{param_value}"'
        else:
            formatted_value = str(param_value)

        table_lines.append(f"| `{param_name}` | {formatted_value} | {description} |")

    display(Markdown("\n".join(table_lines)))
