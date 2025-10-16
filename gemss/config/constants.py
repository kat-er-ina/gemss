"""
Project constants for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This module contains essential project-related constants including file names,
paths, and project metadata used throughout the configuration system.
"""

from typing import Final

# Configuration file names
CONFIG_FILES: Final = {
    "DATASET": "generated_dataset_parameters.json",
    "ALGORITHM": "algorithm_settings.json",
    "POSTPROCESSING": "solution_postprocessing_settings.json",
}

# Project metadata
PROJECT_NAME: Final = "Gaussian Ensemble for Multiple Sparse Solutions"
PROJECT_ABBREV: Final = "GEMSS"
CONFIG_PACKAGE_NAME: Final = "gemss.config"
