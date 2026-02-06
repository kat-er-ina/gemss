"""
Feature selection package for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This package contains the core feature selection algorithms, model definitions,
and utility functions for Bayesian sparse feature selection.

Modules:
- inference: Main variational inference logic (BayesianFeatureSelector)
- models: Prior distributions and model components
- utils: Utility functions for optimization settings display
"""

from ..utils.utils import (
    SelectorHistory,
    dataframe_to_ascii_table,
    display_feature_lists,
    format_summary_row_feature_with_mu,
    generate_feature_names,
    get_solution_summary_df,
    load_constants_json,
    load_feature_lists_json,
    load_selector_history_json,
    myprint,
    save_constants_json,
    save_feature_lists_json,
    save_feature_lists_txt,
    save_selector_history_json,
    show_solution_summary,
)
from .inference import BayesianFeatureSelector
from .models import (
    GaussianMixture,
    SpikeAndSlabPrior,
    StructuredSpikeAndSlabPrior,
    StudentTPrior,
)
