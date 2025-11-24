# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

---

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Features

- **Multiple sparse solutions:** Recovers diverse sparse feature 
- **Missing data:** Native handling without imputation
- **Flexible priors:** Structured spike-and-slab by default, with Student-t and vanilla spike-and-slab alternatives
- **Variational inference:** PyTorch-based optimization
- **Diversity regularization:** Jaccard-based penalty to promote greater diversity, if needed
- **Diagnostics & recommendations [work in progress]:** Convergence checks, tuning hints
- **Visualization:** Interactive plots of history and solutions
- **Configuration:** Separate dataset/algorithm/postprocessing JSONs
- **Batch processing & tests:** Sweeps, tiered suites, functionality tests

---

## Repository Structure

High-level layout of the project with key responsibilities:

```
gemss/                       # Project root (editable install target)
  README.md                  # Project overview & usage guide
  requirements.txt           # Python dependencies
  setup.py                   # Package metadata for installation
  TODO.md                    # Internal task notes
  data/                      # (Optional) user-provided raw datasets
  results/                   # Top-level results (if any script writes here)
  tests/                     # Automated tests
  scripts/                   # Experiment & sweep utilities
    run_experiment.py        # Single headless experiment driver
    run_sweep.ps1            # Parameter sweep across JSON configs
    run_tiers.ps1            # Tiered artificial data benchmark launcher
    run_sweep_with_tiers.ps1 # Combined sweep + tier logic
    experiment_parameters.json        # Full 7-tier experimental design (110 experiments)
    experiment_parameters_short.json  # Reduced tier set (quick checks)
    results/                 # Structured logs + tier outputs
      logs/                  # Execution summaries & error logs
      tier1/ tier2/ ...      # Output text files per experiment combination
      tierX/tier_summary_metrics.csv # Aggregated metrics for all runs in a tier

  notebooks/                 # Interactive exploration & evaluation
    demo.ipynb               # Synthetic end-to-end demo
    analyze_tier_results.ipynb # Visualization & analysis of CSV logs from tier runs
    explore_custom_dataset.ipynb      # Workflow for user data
    tabpfn_evaluation_example.ipynb   # TabPFN evaluation showcase
    tabpfn_evaluate_custom_dataset_results.ipynb # Evaluation of saved solutions
    results/                 # Notebook-specific artifacts

  gemss/                     # Core Python package
    __init__.py
    utils.py                 # Persistence & display helpers (save/load history, solutions)
    config/                  # Modular configuration system
      config.py              # Loader, caching, display utilities
      constants.py           # Paths & global names
      algorithm_settings.json
      generated_dataset_parameters.json
      solution_postprocessing_settings.json
    data_handling/
      data_processing.py     # Preprocessing (scaling, categorical handling)
      generate_artificial_dataset.py  # Synthetic data with controlled sparsity & missingness
    feature_selection/
      inference.py           # Variational optimization
      models.py              # Prior & model component definitions
    diagnostics/
      visualizations.py      # Plotting (ELBO, mu trajectories, alpha distributions)
      result_postprocessing.py  # Solution recovery, metrics (SI/ASI) & summarization
      simple_regressions.py  # Lightweight regression/classification evaluation
      outliers.py            # Outlier-based feature set extraction
      performance_tests.py   # Convergence & stability diagnostics
      recommendations.py     # Parameter tuning heuristics
      recommendation_messages.py # Message templates for recommendations
      tabpfn_evaluation.py   # Nested CV + metrics + optional SHAP (TabPFN)
```


### Artifacts

- Feature selection runs (notebooks or `run_experiment.py`) typically save:
  - `search_setup*.json` (constants/config used)
  - `search_history_results*.json` (ELBO, mu, var, alpha trajectories)
  - `all_candidate_solutions*.json` and `.txt` (components → feature lists)
- Evaluation notebooks may additionally emit:
  - `tabpfn_evaluation_average_scores.csv` (nested CV aggregate metrics)
  - `tabpfn_feature_importances.csv` (SHAP fold-wise summaries)
- Script tier runs create:
  - Timestamped text reports under `scripts/results/tier*/`
  - `tier_summary_metrics.csv` aggregating Recall, Precision, Success Index (SI), and Adjusted Success Index (ASI) for all runs in that tier


### Key Utility Functions (from `utils.py`)
- `save_feature_lists_json` / `load_feature_lists_json` — structured solution persistence, title keyed.
- `save_selector_history_json` / `load_selector_history_json` — optimization trajectory round-trip with automatic array reconstruction.
- `save_constants_json` / `load_constants_json` — exact configuration provenance.

Use these together for full reproducibility: constants (inputs) + history (process) + solutions (outputs).

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
   - `N_ITER`: Number of optimization iterations, `LEARNING_RATE`, regularization, etc.

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

## Quick Start

```powershell
pip install -r requirements.txt
pip install -e .  # optional, for development
```

---

## Usage

- **Demo notebook:** `notebooks\demo.ipynb` contains the complete walkthrough with synthetic data.
- **Custom dataset notebook:** `notebooks\explore_custom_dataset.ipynb` guides you when using GEMSS on your own data.
- **Scripted experiments, single or in batches:** all run the script `scripts\run_experiment.py` with parameters configured in corresponding JSON files. Batches of experiments can be run using PowerShell scripts.

---

### Custom dataset

GEMSS provides a notebook to explore your own datasets. While basic preprocessing utilities are provided, it is advisable to provide cleaned data with only numerical values. Missing values are handled natively. Standard and minmax scaling is available.

**Steps:**
1. Copy your dataset in a .csv format in the `data` folder.
2. Open the notebook `explore_custom_dataset.ipynb` and follow the instructions.

**Workflow in `explore_custom_dataset`:**
1. Modify the data file name, choose the index and target value columns.
2. Supervise basic data preprocessing: check out the cells output and possibly adjust parameters as desired.
3. Adjust the algorithm hyperparameters in the notebook.
4. Run all remaining cells.
5. Review the results. Check out the comprehensive diagnostics and visualizations.
6. Iterate: adjust the hyperparameters based on convergence properties and desired outcome.

**Advanced Postprocessing Functions:**


---

## Missing Data Handling

GEMSS natively supports datasets with missing feature values **without requiring imputation or sample removal**. The algorithm automatically detects missing data and handles them during likelihood computation. Only samples without a valid target value are dropped.
In case of significant amount of missing data, it is advisable to increase the batch size.


## Persistence & Reproducibility

### Feature Lists

The JSON format stores sections (solution types) each containing component → features mapping. Use a single dictionary keyed by solution titles.

```python
from gemss.utils import save_feature_lists_json, load_feature_lists_json

all_features_lists = {
    "Top features": {"component_0": ["feat_a", "feat_b"], "component_1": ["feat_c"]},
    "Full features": {"component_0": ["feat_a", "feat_b", "feat_d"], "component_1": ["feat_c", "feat_e"]},
    "Outlier features (STD_2.0)": {"component_0": ["feat_a", "feat_b"], "component_1": ["feat_c", "feat_e"]},
    "Outlier features (STD_3.0)": {"component_0": ["feat_a"], "component_1": ["feat_e"]},
}

msg = save_feature_lists_json(all_features_lists, "all_candidate_solutions.json")
print(msg)

loaded_feature_lists, msg = load_feature_lists_json("all_candidate_solutions.json")
print(msg)
print(list(loaded_feature_lists.keys()))  # {'Top features', 'Full features', ...}```
```

### Optimization History

History contains per-iteration arrays (ELBO, mu, var, alpha). Arrays are stored as nested lists and automatically converted back to NumPy arrays when loading.

```python
from gemss.utils import save_selector_history_json, load_selector_history_json

msg = save_selector_history_json(history, "search_history_results.json")
print(msg)

history_loaded, msg = load_selector_history_json("search_history_results.json")
print(msg)  # iterations count and keys
```

### Configuration Constants

Persist the exact hyperparameter setup used for a run.

```python
from gemss.utils import save_constants_json, load_constants_json

msg = save_constants_json(constants, "search_setup.json")
print(msg)

constants_loaded, msg = load_constants_json("search_setup.json")
print(msg, len(constants_loaded))
```

For reproducibility: pair `search_setup.json` (inputs), `search_history_results.json` (trajectory), and `all_candidate_solutions.json` (outputs).

---

## Integrated Evaluation of Results

### Logistic and Linear Regression

Use lightweight baselines to quickly validate discovered feature sets. This utility automatically detects regression vs binary classification and reports metrics on the training data.

```python
from gemss.result_postprocessing import solve_any_regression

# X_selected: pd.DataFrame or np.ndarray of selected features
# y: target vector (continuous for regression, binary/0-1 for classification)
results = solve_any_regression(
  X_selected,
  y,
  scaling="standard",    # or "minmax" or None
)
```

- Regression metrics: `r2_score`, `adjusted_r2`, `MSE`, `RMSE`, `MAE`, `MAPE` (if safe).
- Classification metrics: `accuracy`, `balanced_accuracy`, `roc_auc` (binary), `f1_score`, per-class precision/recall.



### TabPFN Evaluation

The `tabpfn_evaluate` helper (in `gemss.diagnostics.tabpfn_evaluation`) offers quick performance estimation of discovered feature sets via nested (outer) cross-validation and optional SHAP explanations.

```python
from gemss.diagnostics.tabpfn_evaluation import tabpfn_evaluate

results = tabpfn_evaluate(
    X_selected,            # pd.DataFrame or np.ndarray
    y,                     # target vector
    apply_scaling="standard",  # or "minmax" or None
    outer_cv_folds=3,
    random_state=42,
    explain=True,          # compute SHAP values (can be costly)
    shap_sample_size=100,  # cap SHAP sample size
)

print(results["average_scores"])       # Dict of aggregated metrics
print(results["fold_scores"][0])        # Per-fold metrics
```

For regression tasks metrics include: `r2_score`, `adjusted_r2`, `MSE`, `RMSE`, `MAE`, `MAPE` (if safe). For classification: `accuracy`, `balanced_accuracy`, `roc_auc` (binary), `f1_score`, per-class precision/recall, and confusion matrix.

SHAP output (if `explain=True`) is available under `results['shap_explanations_per_fold']` as a list of dictionaries (mean absolute SHAP per feature per fold).

Use cases:
- Sanity check of explanatory power of aggregated discovered features.
- Baseline comparison versus randomly selected feature sets of equal cardinality.
- Rapid evaluation using a strong, nonlinear model.

---

## Diagnostics & Recommendations

```python
from gemss.result_postprocessing import (
    recover_solutions,
    show_algorithm_progress, 
    solve_any_regression,
    display_features_overview,
    get_long_solutions_df
)
```

### Core Functions
- `recover_solutions`: extract sparse sets; compact vs full rankings; custom sparsity/thresholds
- `show_algorithm_progress`: ELBO, mu, alpha trajectories; name mapping support
- `solve_any_regression`: regression/classification validation; L1/L2/ElasticNet; per-component metrics
- `display_features_overview`, `get_long_solutions_df`: ground truth comparison, tabular summaries
- Plotting lives in `gemss.diagnostics.visualizations.py`

### Advanced (WIP)

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

## Configuration

All configuration is centralized in `gemss/config/`:

```
gemss/config/
├── config.py                              # Main configuration manager
├── constants.py                           # Project constants and file paths
├── generated_dataset_parameters.json     # Artificial dataset parameters
├── algorithm_settings.json               # Core algorithm parameters  
└── solution_postprocessing_settings.json # Postprocessing parameters
```

The configuration package exposes parameters as Python variables:

```python
import gemss.config as C

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

- Text reports under `scripts/results/` with timestamps and parameters
- Includes run parameters, true/discovered supports, solution tables, diagnostics


## Tiered Artificial Data Experiments

The artificial data experiment framework provides a reproducible, 7-tier benchmark (110 total experiments) to stress, validate, and characterize GEMSS across regimes:

- **Tier 1**: Basic validation (18 experiments) - baseline performance on clean data, N < P
- **Tier 2**: High-dimensional stress test (9 experiments) - P ≥ 1000, N << P
- **Tier 3**: Sample-rich scenarios (14 experiments) - N ≥ P control group
- **Tier 4**: Robustness under adversity (22 experiments) - severe noise and missing data
- **Tier 5**: Effect of Jaccard penalty (12 experiments) - diversity enforcement testing
- **Tier 6**: Regression validation (29 experiments) - continuous response, mirrors Tiers 1, 2 and 4
- **Tier 7**: Class imbalance (6 experiments) - unbalanced binary responses

It is driven by JSON parameter specifications and PowerShell orchestration scripts under `scripts/`.

### Modes of Execution

- Single run (quick check) using manually defined parameters:
  ```powershell
  python scripts\run_experiment.py
  ```
- Simple sweep (legacy) using parameters manually defined inside this script:
  ```powershell
  .\scripts\run_sweep.ps1
  ```
- Tiered suite (recommended):
  ```powershell
  cd scripts
  .\run_tiers.ps1                    # Full suite (tiers 1–7)
  .\run_tiers.ps1 -tiers @("1","4")  # Selected tiers only
  .\run_tiers.ps1 -parametersFile experiment_parameters_short.json  # Using a custom file with parameter combinations
  ```

During execution you are prompted for confirmation (experiment count summary). Logs, errors, and a final execution summary are written to `scripts/results/logs/`.

### Parameter Files

Two JSON specifications exist:
- `experiment_parameters.json` (full design, 110 experiments across 7 tiers)
- `experiment_parameters_short.json` (condensed sanity grid)

Both define:
```text
parameter_format = "N_SAMPLES,N_FEATURES,N_GENERATING_SOLUTIONS,SPARSITY,NOISE_STD,NAN_RATIO,N_CANDIDATE_SOLUTIONS,LAMBDA_JACCARD,BATCH_SIZE,BINARY_RESPONSE_RATIO"
```

Each comma-separated combination string populates the artificial dataset + core algorithm context for a run. Static `algorithm_parameters` inside every tier block set global optimizer / prior hyperparameters reused across that tier’s combinations.

#### Combination Field Meanings
- `N_SAMPLES`: Rows in synthetic dataset.
- `N_FEATURES`: Total feature dimensionality (p).
- `N_GENERATING_SOLUTIONS`: True sparse supports used to generate the response.
- `SPARSITY`: Nonzero features per generating solution (k).
- `NOISE_STD`: Gaussian noise standard deviation on generated response.
- `NAN_RATIO`: Fraction of feature entries randomly set to NaN post-generation.
- `N_CANDIDATE_SOLUTIONS`: Mixture components (candidate sparse solutions) optimized by GEMSS.
- `LAMBDA_JACCARD`: Diversity penalty weight encouraging support dissimilarity.
- `BATCH_SIZE`: Minibatch size for stochastic updates (dynamic with missingness—see below).
- `BINARY_RESPONSE_RATIO`: Class proportion for binary tasks (ignored when regression).

#### Algorithm Parameter Block (`algorithm_parameters` per tier)
- `N_ITER`: Optimization iterations.
- `PRIOR_TYPE`: Prior family (`sss` structured spike-and-slab, `student`, etc.).
- `STUDENT_DF`, `STUDENT_SCALE`: Heavy-tailed prior shape (if Student-t active).
- `VAR_SLAB`, `VAR_SPIKE`: Spike-and-slab variances.
- `WEIGHT_SLAB`, `WEIGHT_SPIKE`: Prior mixture weights.
- `IS_REGULARIZED`: Toggles diversity regularization pathway.
- `LEARNING_RATE`: Optimizer step size.
- `MIN_MU_THRESHOLD`: Postprocessing importance threshold baseline.
- `BINARIZE`: Selects classification (True) vs regression (False) synthetic response.

#### Dynamic Batch Size Logic
For robustness under missing data, larger batches are used as `NAN_RATIO` increases. In the full design:
```
if NAN_RATIO == 0.0: BATCH_SIZE = 16
else: BATCH_SIZE = int(16 * 1.5 / (1 - NAN_RATIO))
```

### Analysis of experiment results

Use notebook `analyze_tier_results.ipynb` to explore results of the tiered experiments.
