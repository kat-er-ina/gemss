# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

---

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Features

- **Multiple sparse solutions:** Identifies all distinct sparse supports that explain the data
- **Native missing data handling:** Works directly with datasets containing missing values without imputation or sample dropping
- **Flexible priors:** Spike-and-slab, structured spike-and-slab (SSS), and Student-t priors
- **Variational inference:** Efficient optimization with PyTorch backend
- **Diversity regularization:** Jaccard penalty to encourage component diversity
- **Dual-purpose design:** Works with both synthetic data (development) and real datasets (production)
- **Basic data preprocessing:** Automatic handling of categorical variables, scaling, and data quality checks
- **Performance diagnostics:** Automated optimization analysis and parameter recommendations
- **Recommendation system:** Intelligent suggestions for parameter adjustment
- **Rich visualization:** Interactive plots for optimization history and results
- **Modular configuration:** Clean separation of artificial dataset, algorithm, and postprocessing parameters
- **Batch processing:** Parameter sweep scripts for systematic experimentation
- **Automated testing:** Limited automated test suite for functionality validation
- **Comprehensive output:** Detailed summaries, diagnostics, and solution tables

---

## Repository Structure

```
gemss/                      # Core package
  config/                   # Configuration package
    config.py               # Configuration manager and parameter loading
    constants.py            # Project constants and file paths
    algorithm_settings.json                # Algorithm configuration
    generated_dataset_parameters.json      # Synthetic dataset configuration
    solution_postprocessing_settings.json  # Postprocessing configuration
  data_handling/                     # Utilities to handle data
    generate_artificial_dataset.py   # Synthetic dataset generator
    data_processing.py               # Utilities to preprocess user-provided datasets
  diagnostics/              # Diagnostics and testing package
    performance_tests.py    # Performance diagnostics and testing
    recommendations.py      # Parameter recommendation system
    recommendation_messages.py  # Recommendation message templates
    result_postprocessing.py    # Solution extraction and diagnostics
    simple_regressions.py       # Simple regression solvers (logistic, linear)
    visualizations.py           # Plotting and visualization functions
  feature_selection/  # Core feature selection package
    inference.py      # Main variational inference logic (BayesianFeatureSelector)
    models.py         # Prior distributions and model components  
  utils.py          # Utility functions for optimization settings

notebooks/
  demo.ipynb                     # Interactive demo with synthetic data
  explore_custom_dataset.ipynb  # Example with real user dataset

scripts/
  run_experiment.py  # Run single experiment (headless)
  run_sweep.ps1      # PowerShell sweep script for batch experiments

tests/
  test_missing_data_native.py  # Comprehensive test suite for missing data handling
```

---

## Configuration Files

The project uses a modular configuration system with 3 JSON files located in `gemss/config/`:

1. **generated_dataset_parameters.json**  
   Artificial dataset generation parameters (for development/demo only):
   - `N_SAMPLES`: Number of samples (rows)
   - `N_FEATURES`: Number of features (columns) 
   - `N_GENERATING_SOLUTIONS`: Number of distinct sparse solutions that were explicitly constructed during data generation.
   - `SPARSITY`: Support size (nonzero features per solution)
   - `NOISE_STD`: Noise level
   - `NAN_RATIO`: Proportion of missing values (NaNs) to introduce randomly in the dataset (0.0 to 1.0)
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

### Custom dataset

GEMSS provides a notebook to explore your own datasets. While basic preprocessing utilities are provided, it is advisable to provide cleaned data with only numerical values. Missing values are handled natively. Standard scaling is available.

**Steps:**
1. Copy your dataset in a .csv format in the `data` folder.
2. Open the notebook `explore_custom_dataset.ipynb` and follow its structure.

**Workflow in `explore_custom_dataset`:**
1. Modify the data file name, choose the index and target value columns.
2. Supervise data preprocessing: check out the cells output and possibly adjust parameters as desired.
3. Adjust the algorithm hyperparameters in the notebook.
4. Run all remaining cells.
5. Review the results. Check out the comprehensive diagnostics and visualizations.
6. Iterate: adjust the hyperparameters based on convergence properties and desired outcome.

**Advanced Data Processing Functions:**
```python
from gemss.data_handling.data_processing import (
    load_data, 
    preprocess_features, 
    preprocess_non_numeric_features,
    get_feature_name_mapping
)

# Load and preprocess your dataset
df, response = load_data(
    csv_dataset_name="your_dataset.csv",
    index_column_name="sample_id", 
    label_column_name="target"
)

# Get feature name mapping for interpretability
feature_to_name = get_feature_name_mapping(df)

# Handle categorical variables and scaling
X_processed = preprocess_features(
    df, 
    drop_non_numeric_features=False,  # Encode instead of dropping
    apply_standard_scaling=True
)

# Missing values are preserved and handled natively by the algorithm
selector = BayesianFeatureSelector(
    n_features=X_processed.shape[1], 
    n_components=3, 
    X=X_processed, 
    y=response.values
)
```

---

### Interactive Exploration

- **Demo Notebook (`demo.ipynb`):**
  ```bash
  jupyter notebook notebooks/demo.ipynb
  ```
  - Complete walkthrough with synthetic data
  - Data generation, model fitting, and diagnostics
  - Performance testing and recommendations
  - Handling missing data when setting `NAN_RATIO > 0` in configuration

- **Real Data Notebook (`explore_custom_dataset.ipynb`):**
  - Example workflow for user datasets
  - Parameter tuning for real-world problems
  - Best practices for unknown data exploration

---

## Missing Data Handling

GEMSS natively supports datasets with missing feature values **without requiring imputation or sample removal**. The algorithm automatically detects missing data and handles them during likelihood computation. Only samples without a valid target value are dropped.

### Key Features:
- **Automatic Detection:** Missing values (NaN) in feature matrix are automatically detected
- **Per-Sample Masking:** Each sample uses only its observed features for likelihood computation
- **Gradient Preservation:** Maintains proper gradient flow for optimization
- **No Data Loss:** All samples are retained regardless of missing data patterns
- **Statistical Rigor:** Uses only observed features per sample while preserving Bayesian inference properties

### Usage Example:
```python
import numpy as np
from gemss.feature_selection.inference import BayesianFeatureSelector

# Create data with missing values
X = np.random.randn(100, 20)
X[np.random.rand(*X.shape) < 0.3] = np.nan  # 30% missing values
y = np.random.randn(100)

# Algorithm automatically handles missing data
selector = BayesianFeatureSelector(
    n_features=20, 
    n_components=3, 
    X=X, 
    y=y
)
history = selector.optimize()
```

### Generating Artificial Datasets with Missing Data:
GEMSS can generate synthetic datasets with controlled missing data patterns for testing and development:

```python
from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset

# Generate dataset with 20% missing values
data, response, solutions, parameters = generate_artificial_dataset(
    n_samples=100,
    n_features=20,
    n_solutions=3,
    sparsity=2,
    nan_ratio=0.2,  # 20% missing values randomly distributed
    random_seed=42
)

# Missing values are introduced after structured generation
# preserving the multiple sparse solutions structure
```

The `NAN_RATIO` parameter in `generated_dataset_parameters.json` controls missing data generation:
- `NAN_RATIO = 0.0` → No missing data (default for clean testing)
- `NAN_RATIO = 0.1` → 10% missing values randomly distributed
---

## Performance Diagnostics & Recommendations

The system provides comprehensive tools for analyzing optimization results and extracting meaningful solutions from the GEMSS feature selection algorithm. The basic functionalities are available through the result postprocessing module:

```python
from gemss.result_postprocessing import (
    recover_solutions,
    show_algorithm_progress, 
    solve_any_regression,
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
- `solve_any_regression()` - Validates discovered solutions using supervised learning
- Automatic detection of binary classification vs regression tasks
- Support for L1 (Lasso), L2 (Ridge), and ElasticNet penalties
- Component-wise performance metrics and coefficient analysis
- Comprehensive results overview across all discovered solutions
- Powered by `gemss.diagnostics.simple_regressions` module for logistic and linear regression utilities

**Solution Comparison and Evaluation:**
- `display_features_overview()` - Ground truth comparison (for synthetic data)
- Missing vs extra features analysis with coverage statistics
- `get_long_solutions_df()` - Structured solution comparison in tabular format

All plotting functions are available in `gemss.diagnostics.visualization.py` module.

### Advanced Diagnostics (Work in Progress)

The system includes automated performance analysis to assess the algorithmic sensibility of discovered solutions and to aid with hyperparameter tuning:

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

- Results of the script-based runs are saved in the `scripts/results/` directory as text files.
- Filenames include timestamps and all key parameter values.
- Each file contains:
  - All run parameters
  - True and discovered supports
  - Solution tables for each mixture component
  - Diagnostic information

---

## Testing

The repository includes basic tests to validate functionalities, in particular the handling of missing data.

### Running Tests:
```bash
# Test missing data handling
python tests/test_missing_data_native.py

# Run with Python from project root
cd gemss/
python ../tests/test_missing_data_native.py
```

### Test Coverage:
- **Native missing data handling:** Validates algorithm can process datasets with arbitrary missing patterns
- **Gradient flow preservation:** Ensures optimization works correctly with missing data
- **Statistical correctness:** Verifies likelihood computation with masked features
- **Edge cases:** Tests various missing data patterns and sparsity levels

The test suite automatically generates synthetic datasets with controlled missing data patterns and validates that:
1. Algorithm initialization succeeds with missing data
2. Optimization converges properly
3. Gradient computation is stable
4. Results are statistically meaningful

---

## Customization & Extending

- **Add new priors:** Implement in `gemss/feature_selection/models.py` and update `BayesianFeatureSelector`
- **Custom diagnostics:** Extend `gemss/diagnostics/performance_tests.py` with new test methods
- **New recommendations:** Add message templates to `gemss/diagnostics/recommendation_messages.py`
- **Visualization:** Create new plots in `gemss/diagnostics/visualizations.py`
- **Configuration:** Modify JSON files or add new parameter categories
- **Sweep parameters:** Edit `run_sweep.ps1` for custom batch experiments
- **Real data workflows:** Follow `explore_custom_dataset.ipynb` as template
- **Testing:** Add new test cases to `tests/` directory following `test_missing_data_native.py` pattern
- **Data preprocessing:** Extend `gemss/data_handling/data_processing.py` for custom preprocessing pipelines

---

## Requirements

- Python 3.11+
- numpy
- pandas
- torch
- scikit-learn
- plotly
- seaborn
- jupyter

(See `requirements.txt` for full details.)
