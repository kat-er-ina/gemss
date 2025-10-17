# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

---

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Features

- **Multiple sparse solutions:** Identifies all distinct sparse supports that explain the data
- **Flexible priors:** Spike-and-slab, structured spike-and-slab (SSS), and Student-t priors
- **Variational inference:** Efficient optimization with PyTorch backend
- **Diversity regularization:** Jaccard penalty to encourage component diversity
- **Dual-purpose design:** Works with both synthetic data (development) and real datasets (production)
- **Performance diagnostics:** Automated optimization analysis and parameter recommendations
- **Recommendation system:** Intelligent suggestions for parameter adjustment
- **Rich visualization:** Interactive plots for optimization history and results
- **Modular configuration:** Clean separation of artificial dataset, algorithm, and postprocessing parameters
- **Batch processing:** Parameter sweep scripts for systematic experimentation
- **Comprehensive output:** Detailed summaries, diagnostics, and solution tables

---

## Repository Structure

```
gemss/                      # Core package
  config/                   # Configuration package
    config.py               # Configuration manager and parameter loading
    constants.py            # Project constants and file paths
    algorithm_settings.json # Algorithm configuration
    generated_dataset_parameters.json  # Synthetic dataset configuration
    solution_postprocessing_settings.json  # Postprocessing configuration
  diagnostics/              # Diagnostics and testing package
    performance_tests.py    # Performance diagnostics and testing
    recommendations.py      # Parameter recommendation system
    recommendation_messages.py # Recommendation message templates
    result_postprocessing.py              # Solution extraction and diagnostics
  generate_artificial_dataset.py        # Synthetic dataset generator
  inference.py                          # Main variational inference logic (BayesianFeatureSelector)
  models.py                             # Prior distributions and model components
  utils.py                              # Utility functions
  visualizations.py                     # Plotting and visualization functions

notebooks/
  demo.ipynb                            # Interactive demo with synthetic data
  explore_unknown_dataset.ipynb        # Example with real user dataset

scripts/
  run_experiment.py                     # Run single experiment (headless)
  run_sweep.ps1                         # PowerShell sweep script for batch experiments

results/                                # Experiment output summaries (created automatically)

run_sweep.ps1                           # PowerShell sweep script (from root directory)
requirements.txt
setup.py
README.md
```

---

## Configuration Files

The project uses a modular configuration system with 3 JSON files located in `gemss/config/`:

1. **generated_dataset_parameters.json**  
   Artificial dataset generation parameters (for development/demo only):
   - `N_SAMPLES`: Number of samples (rows)
   - `N_FEATURES`: Number of features (columns) 
   - `N_GENERATING_SOLUTIONS`: Number of distinct sparse solutions that were explicitely constructed during data generation.
   - `SPARSITY`: Support size (nonzero features per solution)
   - `NOISE_STD`: Noise level
   - `BINARIZE`: Whether the response vector should be continuous or binary.
   - `BINARY_RESPONSE_RATIO`: The required ratio of binary classes, if a binary classification problem is required by `BINARIZE`.
   - `DATASET_SEED`: The random seed used to generate the artificial data.

2. **algorithm_settings.json**  
   Core algorithm parameters (used for both synthetic and real data):
   - `N_CANDIDATE_SOLUTIONS`: Number of candidate solutions (Gaussian mixture components) to search for.
   - `PRIOR_TYPE`: Choice of the sparsifying prior distribution ('ss', 'sss', 'student').
   - `N_ITER`: Numbe of optimization iterations, `LEARNING_RATE`, regularization, etc.

3. **solution_postprocessing_settings.json**  
   Solution extraction and analysis parameters:
   - `DESIRED_SPARSITY`: Target number of features in final solution
   - `MIN_MU_THRESHOLD`: Feature importance threshold, etc.

The configuration system (`gemss.config`) provides:
- **Lazy loading** with caching for efficiency
- **Parameter categorization** (artificial dataset, algorithm, postprocessing)
- **Rich display functions** for notebooks
- **Validation and error handling**

---

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. *(Optional)* Install the package in editable/development mode:
   ```bash
   pip install -e .
   ```

---

## Usage

### A Single Demo Experiment

1. **Configure parameters:**
   - Edit `generated_dataset_parameters.json`, `algorithm_settings.json`, and `solution_postprocessing_settings.json` manually,
     **or** let the sweep script generate them
     **or** define them manually in your own notebook.

2. **Run the experiment:**
   ```bash
   python scripts/run_experiment.py
   ```
   - Optionally, specify the output file name:
     ```bash
     python scripts/run_experiment.py --output my_results.txt
     ```
   - Results are saved in the `results/` directory.

---

### Batch Experiments (Parameter Sweep)

- **On Windows:** Use the PowerShell script:
   ```powershell
   .\run_sweep.ps1
   ```
   - The script will:
     - Iterate over each parameter combination (see `$combinations` in the script).
     - Overwrite the JSON config files for each run.
     - Call `run_experiment.py` and save output in `results/` with filenames including all parameter values.

- **On Linux/macOS:** Adapt the logic from `run_sweep.ps1` to a Bash script as needed.

---

### Interactive Exploration

- **Demo Notebook (`demo.ipynb`):**
  ```bash
  jupyter notebook notebooks/demo.ipynb
  ```
  - Complete walkthrough with synthetic data
  - Data generation, model fitting, and diagnostics
  - Performance testing and recommendations

- **Real Data Notebook (`explore_unknown_dataset.ipynb`):**
  - Example workflow for user datasets
  - Parameter tuning for real-world problems
  - Best practices for unknown data exploration

---

## Performance Diagnostics & Recommendations

The system provides comprehensive tools for analyzing optimization results and extracting meaningful solutions from the GEMSS feature selection algorithm. The basic functionalities are available through the result postprocessing module:

```python
from gemss.result_postprocessing import (
    recover_solutions,
    show_algorithm_progress, 
    show_regression_results_for_solutions,
    display_features_overview,
    get_long_solutions_df
)
```

### Core Analysis Functions

**Solution Recovery:**
- `recover_solutions()` - Extracts sparse feature sets from optimization history based on significance thresholds
- Identifies features with high mean values (mu) in final iterations
- Returns both compact solutions (top features) and comprehensive feature rankings
- Supports custom sparsity targets and importance thresholds

**Algorithm Progress Visualization:**
- `show_algorithm_progress()` - Comprehensive optimization monitoring with interactive plots
- ELBO convergence tracking to assess optimization quality
- Mixture component means (mu) evolution over iterations  
- Mixture weights (alpha) progression showing component importance
- Support for original feature name mapping for interpretability

**Predictive Performance Assessment:**
- `show_regression_results_for_solutions()` - Validates discovered solutions using supervised learning
- Automatic detection of binary classification vs regression tasks
- Support for L1 (Lasso), L2 (Ridge), and ElasticNet penalties
- Component-wise performance metrics and coefficient analysis
- Comprehensive results overview across all discovered solutions

**Solution Comparison and Evaluation:**
- `display_features_overview()` - Ground truth comparison (for synthetic data)
- Missing vs extra features analysis with coverage statistics
- `get_long_solutions_df()` - Structured solution comparison in tabular format

### Advanced Diagnostics (Work in Progress)

The system also includes automated performance analysis to assess the algorithmic sensibility of discovered solutions and to aid with hyperparameter tuning:

```python
from gemss.diagnostics.performance_tests import run_performance_diagnostics
from gemss.diagnostics.recommendations import display_recommendations

# Run diagnostics on optimization history
diagnostics = run_performance_diagnostics(history, desired_sparsity=C.DESIRED_SPARSITY)

# Get intelligent parameter recommendations based on the diagnostics  
display_recommendations(diagnostics=diagnostics, constants=C.as_dict())
```

**Available tests:**
- Top feature ordering consistency to assess convergence
- Sparsity gap analysis (work in progress)

---

## Working with Configuration

All configuration is centralized in the `gemss/config/` package:

```
gemss/config/
├── config.py                              # Main configuration manager
├── constants.py                           # Project constants and file paths
├── generated_dataset_parameters.json     # Artificial dataset parameters
├── algorithm_settings.json               # Core algorithm parameters  
└── solution_postprocessing_settings.json # Postprocessing parameters
```

The configuration package (`gemss.config`) loads all three files and exposes parameters as Python variables:

```python
import gemss.config as C
print(C.N_SAMPLES, C.N_CANDIDATE_SOLUTIONS, C.PRIOR_TYPE)
print(C.MIN_MU_THRESHOLD, C.DESIRED_SPARSITY)  # Postprocessing params

# Access by category
dataset_params = C.get_params_by_category('artificial_dataset')
algorithm_params = C.get_params_by_category('algorithm') 
core_params = C.get_core_algorithm_params()  # Excludes synthetic data params

# Display in notebooks
C.display_current_config(constants=C.as_dict(), constant_type='algorithm')
```

**Parameter sweeps:**  
The sweep scripts automatically update the JSON configuration files:
- Use `run_sweep.ps1` (root) or `scripts/run_sweep.ps1`
- Scripts dynamically resolve file paths using `constants.py`
- Each run overwrites config files with new parameter combinations

---

## Output

- Results of the script-based runs are saved in the `scipts/results/` directory as text files.
- Filenames include timestamps and all key parameter values.
- Each file contains:
  - All run parameters
  - True and discovered supports
  - Solution tables for each mixture component
  - Diagnostic information

---

## Customization & Extending

- **Add new priors:** Implement in `gemss/models.py` and update `BayesianFeatureSelector`
- **Custom diagnostics:** Extend `gemss/diagnostics/performance_tests.py` with new test methods
- **New recommendations:** Add message templates to `gemss/diagnostics/recommendation_messages.py`
- **Visualization:** Create new plots in `gemss/visualizations.py`
- **Configuration:** Modify JSON files or add new parameter categories
- **Sweep parameters:** Edit `run_sweep.ps1` for custom batch experiments
- **Real data workflows:** Follow `explore_unknown_dataset.ipynb` as template

---

## Requirements

- Python 3.8+
- numpy
- pandas
- torch
- scikit-learn
- plotly
- seaborn
- jupyter

(See `requirements.txt` for full details.)
