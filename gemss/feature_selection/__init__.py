"""
Feature selection package for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This package contains the core feature selection algorithms, model definitions,
and utility functions for Bayesian sparse feature selection.

Modules:
- inference: Main variational inference logic (BayesianFeatureSelector)
- models: Prior distributions and model components
- utils: Utility functions for optimization settings display
"""

from .inference import *
from .models import *
from .utils import *