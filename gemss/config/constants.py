"""
Project constants for GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

This module contains essential project-related constants including file names,
paths, and project metadata used throughout the configuration system.
"""

from typing import Final
from pathlib import Path

# Configuration file names
CONFIG_FILES: Final = {
    "ARTIFICIAL_DATASET": "generated_dataset_parameters.json",
    "ALGORITHM": "algorithm_settings.json",
    "POSTPROCESSING": "solution_postprocessing_settings.json",
}
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Project metadata
PROJECT_NAME: Final = "Gaussian Ensemble for Multiple Sparse Solutions"
PROJECT_ABBREV: Final = "GEMSS"
CONFIG_PACKAGE_NAME: Final = "gemss.config"

# Experiment results directory
EXPERIMENT_RESULTS_DIR: Final = ROOT_DIR / "scripts" / "results"
