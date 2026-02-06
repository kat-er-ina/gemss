"""
Configuration loader for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This module provides efficient loading and management of experiment parameters
from JSON configuration files co-located with this module.

Parameter Categories:
- Artificial Dataset: Parameters for synthetic data generation (development/demo only)
- Algorithm: Core algorithm parameters (used for both synthetic and real data)
- Postprocessing: Solution recovery and analysis parameters

Features:
- Lazy loading with caching
- Comprehensive parameter validation
- Structured parameter access by category
- Rich display functionality for notebooks
- Efficient dictionary conversion for logging

Usage:
    import gemss.config as config
    # Core algorithm: config.N_CANDIDATE_SOLUTIONS, config.PRIOR_TYPE, etc.
        # Access parameters: config.N_SAMPLES, config.N_CANDIDATE_SOLUTIONS, etc.
    # Artificial data: config.N_SAMPLES, config.N_FEATURES, etc. (demo only)
    # Display configuration: config.display_current_config(config.as_dict())
"""

import json
from collections import OrderedDict
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Literal

from .constants import CONFIG_FILES


class ConfigurationManager:
    """
    Efficient configuration manager with lazy loading and parameter categorization.
    """

    # Parameter category definitions
    ARTIFICIAL_DATASET_PARAMS = [
        'N_SAMPLES',
        'N_FEATURES',
        'N_GENERATING_SOLUTIONS',
        'SPARSITY',
        'NOISE_STD',
        'NAN_RATIO',
        'BINARIZE',
        'BINARY_RESPONSE_RATIO',
        'DATASET_SEED',
    ]

    ALGORITHM_PARAMS = [
        'N_CANDIDATE_SOLUTIONS',
        'N_ITER',
        'PRIOR_TYPE',
        'PRIOR_SPARSITY',
        'SAMPLE_MORE_PRIORS_COEFF',
        'STUDENT_DF',
        'STUDENT_SCALE',
        'VAR_SLAB',
        'VAR_SPIKE',
        'WEIGHT_SLAB',
        'WEIGHT_SPIKE',
        'IS_REGULARIZED',
        'LAMBDA_JACCARD',
        'BATCH_SIZE',
        'LEARNING_RATE',
    ]

    POSTPROCESSING_PARAMS = [
        'DESIRED_SPARSITY',
        'MIN_MU_THRESHOLD',
        'USE_MEDIAN_FOR_OUTLIER_DETECTION',
        'OUTLIER_DEVIATION_THRESHOLDS',
    ]

    # Parameter descriptions for display
    PARAM_DESCRIPTIONS = {
        # Artificial dataset generation (development/demo only)
        'N_SAMPLES': 'Number of samples (rows) in the synthetic dataset.',
        'N_FEATURES': 'Number of features (columns) in the synthetic dataset.',
        'N_GENERATING_SOLUTIONS': "Number of distinct sparse solutions ('true' supports).",
        'SPARSITY': 'Number of nonzero features per solution (support size).',
        'NOISE_STD': 'Standard deviation of noise added to synthetic data.',
        'NAN_RATIO': 'Proportion of missing values (NaNs) in the synthetic dataset.',
        'BINARIZE': 'Whether to binarize the synthetic response variable.',
        'BINARY_RESPONSE_RATIO': 'Proportion of synthetic samples assigned label 1.',
        'DATASET_SEED': 'Random seed for synthetic data reproducibility.',
        # Algorithm settings
        'N_CANDIDATE_SOLUTIONS': 'Desired number of candidate solutions (components of the Gaussian mixture approximating the variational posterior). Set to 2-3x the value of expected true solutions.',
        'N_ITER': 'Number of optimization iterations.',
        'PRIOR_TYPE': "Prior type ('ss', 'sss', or 'student')",
        'PRIOR_SPARSITY': "Expected number of nonzero features per component. Used only in 'sss' prior",
        'SAMPLE_MORE_PRIORS_COEFF': 'Coefficient for increased support sampling. Experimental use only.',
        'STUDENT_DF': "Degrees of freedom for the Student-t prior. Used only if PRIOR_TYPE is 'student'.",
        'STUDENT_SCALE': "Scale parameter for the Student-t prior. Used only if PRIOR_TYPE is 'student'.",
        'VAR_SLAB': "Variance of the 'slab' component in the 'ss' or 'sss' prior. Ignored for 'student' prior.",
        'VAR_SPIKE': "Variance of the 'spike' component in the 'ss' or 'sss' prior. Ignored for 'student' prior.",
        'WEIGHT_SLAB': "Weight of the 'slab' component in the 'ss' prior. Ignored for other priors.",
        'WEIGHT_SPIKE': "Weight of the 'spike' component in the 'ss' prior. Ignored for other priors.",
        'IS_REGULARIZED': 'Whether to use Jaccard similarity penalty.',
        'LAMBDA_JACCARD': 'Regularization strength for Jaccard penalty. Applies only if IS_REGULARIZED is True.',
        'BATCH_SIZE': 'Minibatch size for stochastic updates in the SGD optimization.',
        'LEARNING_RATE': 'Learning rate for the Adam optimizer.',
        # Postprocessing
        'DESIRED_SPARSITY': 'Desired number of features in final solution.',
        'MIN_MU_THRESHOLD': 'Minimum mu threshold for feature selection. Specific for each dataset.',
        'USE_MEDIAN_FOR_OUTLIER_DETECTION': 'Whether to use median and MAD or mean and STD when selecting features by outlier detection.',
        'OUTLIER_DEVIATION_THRESHOLDS': 'A list of thresholding values of either MAD or STD to be used to define outliers.',
    }

    def __init__(self):
        self._config_dir = Path(__file__).parent
        self._cache = {}

    @cache
    def _load_json_file(self, filename: str) -> dict[str, Any]:
        """Load and cache JSON file contents."""
        file_path = self._config_dir / filename
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Configuration file not found: {file_path}')
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in {file_path}: {e}')

    @cache
    def get_artificial_dataset_params(self) -> dict[str, Any]:
        """Get artificial dataset generation parameters (for development/demo only)."""
        params = self._load_json_file(CONFIG_FILES['ARTIFICIAL_DATASET'])
        # Use explicit order defined by ARTIFICIAL_DATASET_PARAMS list
        ordered_params = OrderedDict()
        for k in self.ARTIFICIAL_DATASET_PARAMS:
            if k in params:
                ordered_params[k] = params[k]
        return dict(ordered_params)

    @cache
    def get_algorithm_params(self) -> dict[str, Any]:
        """Get algorithm parameters."""
        params = self._load_json_file(CONFIG_FILES['ALGORITHM'])
        # Use explicit order defined by ALGORITHM_PARAMS list
        ordered_params = OrderedDict()
        for k in self.ALGORITHM_PARAMS:
            if k in params:
                ordered_params[k] = params[k]
        return dict(ordered_params)

    @cache
    def get_postprocessing_params(self) -> dict[str, Any]:
        """Get postprocessing parameters."""
        params = self._load_json_file(CONFIG_FILES['POSTPROCESSING'])
        # Use explicit order defined by POSTPROCESSING_PARAMS list
        ordered_params = OrderedDict()
        for k in self.POSTPROCESSING_PARAMS:
            if k in params:
                ordered_params[k] = params[k]
        return dict(ordered_params)

    @lru_cache(maxsize=1)
    def get_all_params(self) -> dict[str, Any]:
        """Get all parameters in a single dictionary, preserving fixed order."""
        all_params = OrderedDict()

        # 1. Add Artificial Dataset Params
        dataset_params = self.get_artificial_dataset_params()
        for k in self.ARTIFICIAL_DATASET_PARAMS:
            if k in dataset_params:
                all_params[k] = dataset_params[k]

        # 2. Add Algorithm Params
        algorithm_params = self.get_algorithm_params()
        for k in self.ALGORITHM_PARAMS:
            if k in algorithm_params:
                all_params[k] = algorithm_params[k]

        # 3. Add Postprocessing Params
        postprocessing_params = self.get_postprocessing_params()
        for k in self.POSTPROCESSING_PARAMS:
            if k in postprocessing_params:
                all_params[k] = postprocessing_params[k]

        # Return as a regular dict (which retains order in Python 3.7+),
        # but the order is explicitly set by the OrderedDict logic above.
        return dict(all_params)

    def get_params_by_category(self, category: str) -> dict[str, Any]:
        """Get parameters filtered by category (efficient, uses cached dicts)."""
        if category in ('artificial_dataset', 'dataset'):
            return self.get_artificial_dataset_params()
        elif category == 'algorithm':
            return self.get_algorithm_params()
        elif category == 'postprocessing':
            return self.get_postprocessing_params()
        elif category == 'all':
            return self.get_all_params()
        else:
            raise ValueError(
                f"Unknown category: {category}. Valid categories: 'artificial_dataset', 'algorithm', 'postprocessing', 'all'"
            )


# Global configuration manager instance
_config_manager = ConfigurationManager()

# Load all parameters at module level for backward compatibility
_all_params = _config_manager.get_all_params()

# Artificial dataset parameters (for synthetic data generation - development/demo only)
N_SAMPLES = _all_params['N_SAMPLES']
N_FEATURES = _all_params['N_FEATURES']
N_GENERATING_SOLUTIONS = _all_params['N_GENERATING_SOLUTIONS']
SPARSITY = _all_params['SPARSITY']
NOISE_STD = _all_params['NOISE_STD']
NAN_RATIO = _all_params['NAN_RATIO']
BINARIZE = _all_params['BINARIZE']
BINARY_RESPONSE_RATIO = _all_params['BINARY_RESPONSE_RATIO']
DATASET_SEED = _all_params['DATASET_SEED']

# Algorithm parameters
N_CANDIDATE_SOLUTIONS = _all_params['N_CANDIDATE_SOLUTIONS']
N_ITER = _all_params['N_ITER']
PRIOR_TYPE = _all_params['PRIOR_TYPE']
PRIOR_SPARSITY = _all_params.get('PRIOR_SPARSITY')
SAMPLE_MORE_PRIORS_COEFF = _all_params.get('SAMPLE_MORE_PRIORS_COEFF', 1.0)
STUDENT_DF = _all_params['STUDENT_DF']
STUDENT_SCALE = _all_params['STUDENT_SCALE']
VAR_SLAB = _all_params['VAR_SLAB']
VAR_SPIKE = _all_params['VAR_SPIKE']
WEIGHT_SLAB = _all_params['WEIGHT_SLAB']
WEIGHT_SPIKE = _all_params['WEIGHT_SPIKE']
IS_REGULARIZED = _all_params['IS_REGULARIZED']
LAMBDA_JACCARD = _all_params['LAMBDA_JACCARD']
BATCH_SIZE = _all_params['BATCH_SIZE']
LEARNING_RATE = _all_params['LEARNING_RATE']

# Postprocessing parameters
DESIRED_SPARSITY = _all_params['DESIRED_SPARSITY']
MIN_MU_THRESHOLD = _all_params['MIN_MU_THRESHOLD']
USE_MEDIAN_FOR_OUTLIER_DETECTION = _all_params['USE_MEDIAN_FOR_OUTLIER_DETECTION']
OUTLIER_DEVIATION_THRESHOLDS = _all_params['OUTLIER_DEVIATION_THRESHOLDS']


def check_sparsities(artificial_dataset: bool = True) -> None:
    """
    Print sparsity settings for verification.
    Parameters
    ----------
    artificial_dataset : bool
        Whether to include artificial dataset sparsity settings.
        Applicable only if synthetic data is used.
    """
    print('Sparsity settings:')
    if artificial_dataset:
        print(f' - True sparsity of artificial dataset: {SPARSITY}')
    print(f' - Prior sparsity: {PRIOR_SPARSITY}')
    print(f' - Desired sparsity: {DESIRED_SPARSITY}')


def as_dict() -> dict[str, Any]:
    """Return all configuration parameters as a dictionary."""
    return _config_manager.get_all_params().copy()


def get_core_algorithm_params() -> dict[str, Any]:
    """
    Get core algorithm parameters only (excludes artificial dataset parameters).

    This function returns parameters needed for the algorithm to work with
    real user datasets. Use this when you don't need synthetic data generation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing algorithm and postprocessing parameters only
    """
    algorithm_params = _config_manager.get_params_by_category('algorithm')
    postprocessing_params = _config_manager.get_params_by_category('postprocessing')

    # Merge, maintaining the order defined in the class lists
    core_params = OrderedDict()
    for k in ConfigurationManager.ALGORITHM_PARAMS:
        if k in algorithm_params:
            core_params[k] = algorithm_params[k]
    for k in ConfigurationManager.POSTPROCESSING_PARAMS:
        if k in postprocessing_params:
            core_params[k] = postprocessing_params[k]

    return dict(core_params)


def get_params_by_category(category: str) -> dict[str, Any]:
    """
    Get parameters filtered by category.

    Parameters
    ----------
    category : str
        Category name: 'artificial_dataset' (or 'dataset' for compatibility),
        'algorithm', 'postprocessing', or 'all'

    Returns
    -------
    Dict[str, Any]
        Filtered parameters dictionary
    """
    return _config_manager.get_params_by_category(category)


def get_current_config(
    constants: dict[str, Any] | None = None,
    constant_type: Literal[
        'algorithm',
        'postprocessing',
        'algorithm_and_postprocessing',
        'dataset',
        'all',
    ] = 'all',
) -> str:
    """
    Get configuration parameters in a formatted table.

    Parameters
    ----------
    constants : Dict[str, Any], optional
        Configuration parameters to display. If None, uses current config.
    constant_type : str
        Parameter category to display:
        'algorithm', 'postprocessing', 'algorithm_and_postprocessing', 'dataset', 'all'
    Returns
    -------
    str
        Formatted table of configuration parameters.
    """
    if constants is None:
        constants = as_dict()

    # Map legacy category names and handle special cases
    if constant_type in ('artificial_data', 'dataset'):
        category = 'artificial_dataset'
    elif constant_type == 'algorithm_and_postprocessing':
        # Special case: combine algorithm and postprocessing parameters
        algo_params = get_params_by_category('algorithm')
        post_params = get_params_by_category('postprocessing')
        filtered_constants = {**algo_params, **post_params}
        constants = {k: v for k, v in constants.items() if k in filtered_constants}
    else:
        category = constant_type

    if constant_type != 'algorithm_and_postprocessing':
        if category != 'all':
            filtered_constants = get_params_by_category(category)
            constants = {k: v for k, v in constants.items() if k in filtered_constants}

    if not constants:
        return 'No parameters to display.'

    # Create formatted table
    table_lines = [
        '| Parameter | Current Value | Description |',
        '|-----------|---------------|-------------|',
    ]

    for param_name in constants.keys():
        param_value = constants[param_name]
        description = ConfigurationManager.PARAM_DESCRIPTIONS.get(
            param_name, 'Configuration parameter'
        )

        # Format value based on type
        if isinstance(param_value, float):
            formatted_value = f'{param_value:.6g}'
        elif isinstance(param_value, str):
            formatted_value = f'"{param_value}"'
        else:
            formatted_value = str(param_value)

        table_lines.append(f'| `{param_name}` | {formatted_value} | {description} |')

    return '\n'.join(table_lines)


def display_current_config(
    constants: dict[str, Any] | None = None,
    constant_type: Literal[
        'algorithm',
        'postprocessing',
        'algorithm_and_postprocessing',
        'dataset',
        'all',
    ] = 'all',
) -> None:
    """
    Display configuration parameters in a formatted table.

    Parameters
    ----------
    constants : Dict[str, Any], optional
        Configuration parameters to display. If None, uses current config.
    constant_type : str
        Parameter category to display:
        'algorithm', 'postprocessing', 'algorithm_and_postprocessing', 'dataset', 'all'
    """
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print('IPython not available. Cannot display formatted configuration.')
        return

    table_lines = get_current_config(
        constants=constants,
        constant_type=constant_type,
    )

    # Map legacy category names and handle special cases
    if constant_type in ('artificial_data', 'dataset'):
        section_title = 'artificial dataset parameters'
    elif constant_type == 'algorithm_and_postprocessing':
        section_title = 'algorithm and postprocessing parameters'
    else:
        section_title = f'{constant_type} parameters'

    display(Markdown(f'## Configuration: {section_title}'))
    display(Markdown(table_lines))
    return
