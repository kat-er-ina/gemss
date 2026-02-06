"""
Configuration package for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This package contains configuration loading functionality and JSON parameter files.
"""

from gemss.config.config import (
    BATCH_SIZE,
    BINARIZE,
    BINARY_RESPONSE_RATIO,
    DATASET_SEED,
    DESIRED_SPARSITY,
    IS_REGULARIZED,
    LAMBDA_JACCARD,
    LEARNING_RATE,
    MIN_MU_THRESHOLD,
    N_CANDIDATE_SOLUTIONS,
    N_FEATURES,
    N_GENERATING_SOLUTIONS,
    N_ITER,
    N_SAMPLES,
    NAN_RATIO,
    NOISE_STD,
    OUTLIER_DEVIATION_THRESHOLDS,
    PRIOR_SPARSITY,
    PRIOR_TYPE,
    SAMPLE_MORE_PRIORS_COEFF,
    SPARSITY,
    STUDENT_DF,
    STUDENT_SCALE,
    USE_MEDIAN_FOR_OUTLIER_DETECTION,
    VAR_SLAB,
    VAR_SPIKE,
    WEIGHT_SLAB,
    WEIGHT_SPIKE,
    ConfigurationManager,
    as_dict,
    check_sparsities,
    display_current_config,
    get_core_algorithm_params,
    get_current_config,
    get_params_by_category,
)
from gemss.config.constants import (
    CONFIG_FILES,
    CONFIG_PACKAGE_NAME,
    DATA_DIR,
    EXPERIMENT_RESULTS_DIR,
    PROJECT_ABBREV,
    PROJECT_NAME,
    ROOT_DIR,
)
