"""
GEMSS Package

GEMSS (Gaussian Ensemble for Multiple Sparse Solutions) is an algorithm for
feature selection in high-dimensional data. It is intended to be used during
dataset analysis to identify relevant features for predictive modeling.

GEMSS is a Bayesian variational method that approximates multimodal
posteriors by Gaussian mixtures to recover diversified sparse feature sets.
"""

import importlib.metadata

__version__ = importlib.metadata.version('gemss')
