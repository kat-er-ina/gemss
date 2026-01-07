# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Features

* **Multiple sparse solutions:** Recovers diverse sparse feature sets (supports)  
* **Missing data:** Native handling without imputation  
* **Flexible priors:** Structured spike-and-slab by default, with Student-t and vanilla spike-and-slab alternatives  
* **Variational inference:** PyTorch-based optimization  
* **Diversity regularization:** Jaccard-based penalty to promote greater diversity, if needed  
* Diagnostics & recommendations $$work in progress$$  
  : Convergence checks, tuning hints  
* **Visualization:** Interactive plots of history and solutions  
* **Configuration:** Separate dataset/algorithm/postprocessing JSONs  
* **Batch processing & tests:** Sweeps, tiered suites, functionality tests

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
    experiment_parameters.json        # Full 7-tier experimental design (128 experiments)  
    experiment_parameters_short.json  # Reduced tier set (quick checks)  
    results/                 # Structured logs + tier outputs  
      logs/                  # Execution summaries & error logs  
      tier1/ tier2/ ...      # Output text files per experiment combination  
      tierX/tier_summary_metrics.csv # Aggregated metrics for all runs in a tier

  notebooks/                 # Interactive exploration & evaluation  
    demo.ipynb               # Synthetic end-to-end demo  
    explore_custom_dataset.ipynb      # Workflow for user data  
    tabpfn_evaluation_example.ipynb   # TabPFN evaluation showcase  
    tabpfn_evaluate_custom_dataset_results.ipynb # Evaluation of saved solutions  
    analyze_experiment_results/       # Advanced analysis notebooks  
      analysis_per_testcase.ipynb     # Per-testcase analysis (cross-tier)  
      analyze_hyperparameters.ipynb   # Hyperparameter effect analysis  
      tier_level_analysis.ipynb       # Tier-level summary and visualization  
    results/                 # Notebook-specific artifacts and experiment outputs, if saved

  gemss/                     # Core Python package
    utils/
      utils.py               # Persistence & display helpers (save/load history, solutions)  
      visualizations.py      # Plotting utilities for result postprocessing (ELBO, mu trajectories, alpha distributions)  
    config/                  # Modular configuration system  
      config.py              # Loader, caching, display utilities  
      constants.py           # Paths & global names  
      algorithm_settings.json  
      generated_dataset_parameters.json  
      solution_postprocessing_settings.json  
    data_handling/  
      data_processing.py              # Preprocessing (scaling, categorical handling)  
      generate_artificial_dataset.py  # Synthetic data with controlled sparsity & missingness  
    feature_selection/  
      inference.py           # Variational optimization  
      models.py              # Prior & model component definitions  
    postprocessing/ # Solution extraction from the optimization run, evaluation and downstream modeling
      outliers.py               # Outlier-based feature set extraction
      result_postprocessing.py  # Solution recovery, metrics (SI/ASI) & summarization  
      simple_regressions.py     # Lightweight regression/classification evaluation
      tabpfn_evaluation.py      # Nested CV \+ metrics \+ optional SHAP (TabPFN)
    diagnostics/  # WORK IN PROGRESS: diagnostics of performance and results, hyperparameter tuning recommendations
      performance_tests.py       # Convergence & stability diagnostics  
      recommendations.py         # Parameter tuning heuristics
      recommendation_messages.py # Message templates for recommendations
    experiment_assessment/   # Experiment results analysis & visualization (development only)
      case_analysis.py                   # Utilities for analyzing test cases (cross-tier logic)  
      experiment_results_analysis.py     # Core analysis functions and metrics  
      experiment_results_interactive.py  # Interactive widgets for result exploration  
      experiment_results_visualizations.py # Plotting functions for experiment results  
```

### Artifacts

* Feature selection runs (notebooks or run_experiment.py) typically save:  
  * `search_setup*.json` (constants/config used)  
  * `search_history_results*.json` (ELBO, mu, var, alpha trajectories)  
  * `all_candidate_solutions*.json` and `.txt` (components → feature lists)  
* Evaluation notebooks may additionally emit:  
  * tabpfn_evaluation_average_scores.csv (nested CV aggregate metrics)  
  * tabpfn_feature_importances.csv (SHAP fold-wise summaries)  
* Script tier runs create:  
  * Timestamped text reports under `scripts/results/tier*/  `
  * tier_summary_metrics.csv aggregating Recall, Precision, Success Index (SI), and Adjusted Success Index (ASI) for all runs in that tier

### Key Utility Functions (from utils.py)

* `save_feature_lists_json` / `load_feature_lists_json` — structured solution persistence, title keyed.  
* `save_selector_history_json` / `load_selector_history_json` — optimization trajectory round-trip with automatic array reconstruction.  
* `save_constants_json` / `load_constants_json` — exact configuration provenance.

Use these together for full reproducibility: constants (inputs) \+ history (process) \+ solutions (outputs).

## Configuration Files

The project uses a modular configuration system with 3 JSON files located in `gemss/config/`:

1. **generated_dataset_parameters.json** Artificial dataset generation parameters (for development/demo only).
2. **algorithm_settings.json** Core algorithm parameters (used for both synthetic and real data).
3. **solution_postprocessing_settings.json** Solution extraction parameters.

The configuration system (`gemss.config`) provides:

* **Lazy loading** with caching for efficiency  
* **Parameter categorization** (artificial dataset, algorithm, postprocessing)  
* **Rich display functions** for notebooks  
* **Validation and error handling**

## Quick Start

Core dependencies:

```
pip install -r requirements.txt  
pip install -e .  # optional, for development
```

## Usage

* **Demo notebook:** `notebooks\demo.ipynb` contains the complete walkthrough with synthetic data.  
* **Custom dataset notebook:** `notebooks\explore_custom_dataset.ipynb` guides you when using GEMSS on your own data.  
* **Scripted experiments, single or in batches:** all run the script `scripts\run_experiment.py` with parameters configured in corresponding JSON files. Batches of experiments can be run using PowerShell scripts.

### Custom dataset

GEMSS provides a notebook to explore your own datasets. While basic preprocessing utilities are provided, it is advisable to provide cleaned data with only numerical values. Missing values are handled natively. Standard and minmax scaling is available.

**Steps:**

1. Copy your dataset in a .csv format in the data folder.  
2. Open the notebook explore_custom_dataset.ipynb and follow the instructions.

**Workflow in `explore_custom_dataset`:**

1. Modify the data file name, choose the index and target value columns.  
2. Supervise basic data preprocessing: check out the cells output and possibly adjust parameters as desired.  
3. Adjust the algorithm hyperparameters in the notebook.  
4. Run all remaining cells.  
5. Review the results. Check out the comprehensive diagnostics and visualizations.  
6. Iterate: adjust the hyperparameters based on convergence properties and desired outcome.

## Missing Data Handling

GEMSS **natively supports datasets with missing feature values** without requiring imputation or sample removal. The algorithm automatically detects missing data and handles them during likelihood computation. Only samples without a valid target value are dropped.  
In case of significant amount of missing data, it is advisable to increase the batch size.

## Persistence & Reproducibility

### Feature Lists

The JSON format stores sections (solution types) each containing component → features mapping. Use a single dictionary keyed by solution titles.

```python
from gemss.utils.utils import save_feature_lists_json, load_feature_lists_json

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
print(list(loaded_feature_lists.keys()))  # {'Top features', 'Full features', ...}
```

### Optimization History

History contains per-iteration arrays (ELBO, mu, var, alpha). Arrays are stored as nested lists and automatically converted back to NumPy arrays when loading.

```python
from gemss.utils.utils import save_selector_history_json, load_selector_history_json

msg = save_selector_history_json(history, "search_history_results.json")  
print(msg)

history_loaded, msg = load_selector_history_json("search_history_results.json")  
print(msg)  # iterations count and keys
```

### Configuration Constants

Persist the exact hyperparameter setup used for a run.

```python
from gemss.utils.utils import save_constants_json, load_constants_json

msg = save_constants_json(constants, "search_setup.json")  
print(msg)

constants_loaded, msg = load_constants_json("search_setup.json")  
print(msg, len(constants_loaded))
```

For reproducibility: pair search_setup.json (inputs), search_history_results.json (trajectory), and all_candidate_solutions.json (outputs).

## Integrated Evaluation of Results

### Logistic and Linear Regression

Use lightweight baselines to quickly validate discovered feature sets. This utility automatically detects regression vs binary classification and reports metrics on the training data.

```python
from gemss.postprocessing.simple_regressions import solve_any_regression

# X_selected: pd.DataFrame or np.ndarray of selected features  
# y: target vector (continuous for regression, binary/0-1 for classification)  
results = solve_any_regression(  
  solutions=solutions_dict,  
  df=df,  
  response=y,  
  apply_scaling="standard",    # or "minmax" or None  
)
```

* Regression metrics: r2_score, adjusted_r2, MSE, RMSE, MAE, MAPE (if safe).  
* Classification metrics: accuracy, balanced_accuracy, roc_auc (binary), f1_score, per-class precision/recall.

### TabPFN Evaluation

The tabpfn_evaluate helper (in gemss.diagnostics.tabpfn_evaluation) offers quick performance estimation of discovered feature sets via nested (outer) cross-validation and optional SHAP explanations.

```python
from gemss.postprocessing.tabpfn_evaluation import tabpfn_evaluate

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

For regression tasks metrics include: r2_score, adjusted_r2, MSE, RMSE, MAE, MAPE (if safe). For classification: accuracy, balanced_accuracy, roc_auc (binary), f1_score, per-class precision/recall, and confusion matrix.

SHAP output (if explain=True) is available under results['shap_explanations_per_fold'] as a list of dictionaries (mean absolute SHAP per feature per fold).

Use cases:

* Sanity check of explanatory power of aggregated discovered features.  
* Baseline comparison versus randomly selected feature sets of equal cardinality.  
* Rapid evaluation using a strong, nonlinear model.

## Diagnostics & Recommendations

```python
from gemss.postprocessing.result_postprocessing import recover_solutions, show_algorithm_progress  
from gemss.postprocessing.simple_regressions import solve_any_regression  
from gemss.utils.utils import show_solution_summary
```
### Core Functions

* `show_algorithm_progress`: show the progress of the feature selector over iterations. Useful for diagnostics and assessment of output reliability.
* `recover_solutions`: extract solution from the feature selector's history.
* `solve_any_regression`: simple validation using L1- or L2-regularized logistic or linear regression.
* `show_solution_summary`: tabular summary of discovered features.
* Plotting lives in `gemss.diagnostics.visualizations.py`.

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

* Top feature ordering consistency to assess convergence  
* Sparsity gap analysis (work in progress)

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

* Use `scripts/run_sweep.ps1`
* Scripts dynamically resolve file paths using constants.py
* Each run overwrites config files with new parameter combinations

---

## Tiered Artificial Data Experiments

The artificial data experiment framework provides a reproducible, 7-tier benchmark to systematically stress, validate, and characterize GEMSS across different data regimes. The experimental design follows a this structure:

### Total scope

128 individual test cases across 7 tiers.

### Tiers represent different experimental scenarios:

* **Tier 1**: Basic validation (18 experiments) - baseline performance on clean data, n < p
* **Tier 2**: High-dimensional stress test (9 experiments) - p ≥ 1000, n << p  
* **Tier 3**: Sample-rich scenarios (14 experiments) - n ≥ p control group  
* **Tier 4**: Robustness under adversity (22 experiments) - severe noise and missing data  
* **Tier 5**: Effect of Jaccard penalty (28 experiments) - diversity enforcement testing  
* **Tier 6**: Regression validation (29 experiments) - continuous response, mirrors Tiers 1, 2 and 4  
* **Tier 7**: Class imbalance (8 experiments) - unbalanced binary responses

### Test cases

Each test case covers a specific research question (e.g. how does the performance differ for baseline problems for binary classification and regression?). Each test case takes a set of experiments from one or across multiple tiers. The data needed for the individual test cases overlap.

The definitions for the 47 specific test cases are located in `gemss/experiment_assessment/case_analysis.py`.

This design enables us to efficiently cover many questions with a concise and efficiently organized set of experiments.

The individual experiments are defined by JSON parameter specifications and run by PowerShell orchestration scripts under `scripts/`.

### Modes of Execution

* Single run (quick check) using manually defined parameters:  
  ```python scripts\run_experiment.py```

* Simple sweep (legacy) using parameters manually defined inside this script:  
  ```.\scripts\run_sweep.ps1```

* Tiered suite (recommended):

  `.\run_tiers.ps1`                    # Full suite (tiers 1 - 7)  
  `.\run_tiers.ps1 -tiers @("1","4")`  # Selected tiers only  
  `.\run_tiers.ps1 -parametersFile experiment_parameters_short.json`  # Using a custom file with parameter combinations

During execution you are prompted for confirmation (experiment count summary). Logs, errors, and a final execution summary are written to `scripts/results/logs/`.

### Parameter Files

Two JSON specifications exist:

* `experiment_parameters.json` (full design, 128 experiments across 7 tiers)  
* `experiment_parameters_short.json` (small set of quick experiments for testing)

Both define:

```python
parameter_format = "N_SAMPLES,N_FEATURES,N_GENERATING_SOLUTIONS,SPARSITY,NOISE_STD,NAN_RATIO,N_CANDIDATE_SOLUTIONS,LAMBDA_JACCARD,BATCH_SIZE,BINARY_RESPONSE_RATIO"
```

Each comma-separated combination string populates the artificial dataset \+ core algorithm context for a run. Static algorithm_parameters inside every tier block set global optimizer / prior hyperparameters reused across that tier’s combinations.

#### Combination Field Meanings

* `N_SAMPLES`: Rows in synthetic dataset.  
* `N_FEATURES`: Total feature dimensionality (p).  
* `N_GENERATING_SOLUTIONS`: True sparse supports used to generate the response.  
* `SPARSITY`: Nonzero features per generating solution (k).  
* `NOISE_STD`: Gaussian noise standard deviation on generated response.  
* `NAN_RATIO`: Fraction of feature entries randomly set to NaN post-generation.  
* `N_CANDIDATE_SOLUTIONS`: Mixture components (candidate sparse solutions) optimized by GEMSS.  
* `LAMBDA_JACCARD`: Diversity penalty weight encouraging support dissimilarity.  
* `BATCH_SIZE`: Minibatch size for stochastic updates (dynamic with missingness—see below).  
* `BINARY_RESPONSE_RATIO`: Class proportion for binary tasks (ignored when regression).

#### Algorithm Parameter Block (algorithm_parameters per tier)

* `N_ITER`: Optimization iterations.  
* `PRIOR_TYPE`: Prior family (sss structured spike-and-slab, student, etc.).  
* `STUDENT_DF`, `STUDENT_SCALE`: Heavy-tailed prior shape (if Student-t active).  
* `VAR_SLAB`, `VAR_SPIKE`: Spike-and-slab variances.  
* `WEIGHT_SLAB`, `WEIGHT_SPIKE`: Prior mixture weights.  
* `IS_REGULARIZED`: Toggles diversity regularization pathway.  
* `LEARNING_RATE`: Optimizer step size.  
* `MIN_MU_THRESHOLD`: Postprocessing importance threshold baseline.  
* `BINARIZE`: Selects classification (True) vs regression (False) synthetic response.

### Analysis of Experiment Results

The analysis framework leverages the tier-case structure to enable comprehensive exploration of the 128 experimental results across multiple analytical dimensions.

#### Understanding the Analysis Dimensions

The experiments are organized in 7 tiers. However, the tiers often need to be combined across tiers (most notably with the baseline experiments in Tier 1) to answer research questions.

1. **Tier-Level Analysis** (group_identifier="TIER_ID"):  
   * Use notebooks/analyze_experiment_results/tier_level_analysis.ipynb.  
   * Examines performance across different experimental scenarios (high-dimensional, noisy data, etc.)  
   * Identifies which data regimes pose systemic challenges to GEMSS  
   * Useful for understanding algorithm strengths and limitations across problem domains  
   * Does not include baseline performances, unless Tier 1 is manually included.

2. **Case-Level Analysis** (group_identifier="CASE_ID"):  
   * Use notebooks/analyze_experiment_results/analysis_per_testcase.ipynb.  
   * Focuses on specific research questions that may span multiple tiers (e.g. Robustness to noise).  
   * Analyzes individual parameter combinations and their effects on performance  
   * Enables detailed investigation of cross-tier phenomena and parameter interactions  
   * Useful for understanding why certain configurations succeed or fail across different contexts

Since test cases represent research questions that can draw data from multiple tiers, case-level analysis is particularly powerful for understanding cross-scenario performance patterns.

This dual-perspective approach supports both systematic algorithmic validation (tier-level) and focused research question investigation (case-level), enabling comprehensive understanding of GEMSS performance across the full experimental landscape.

#### Interactive Visualization Suite

The analysis notebooks (including `analyze_hyperparameters.ipynb`) provide five complementary visualization functions that work with both analytical dimensions:

* **Performance Overview**: Metrics summary with configurable thresholds for categorizing algorithm performance  
* **Solution Comparison**: Multi-solution analysis across different algorithm configurations and scenarios  
* **Parameter Grouping**: Results visualization grouped by key algorithm parameters (dimensionality, sparsity, noise, etc.)  
* **Parameter Heatmaps**: 2D visualization of parameter interaction effects on performance metrics  
* **Success Analysis**: Success Index (SI) vs Adjusted Success Index (ASI) scatter plots for advanced solution quality assessment that includes problem difficulty in the metrics.

#### Recommended Analysis Workflow

1. **Begin with tier-level overview** to identify problematic or exceptionally performing scenarios.  
2. **Switch to case-level analysis** to investigate specific research questions that span multiple experimental contexts.  
3. **Use parameter grouping** to understand which experimental factors drive performance differences.  
4. **Apply heatmaps** to visualize complex parameter interactions across the experimental space.  
5. **Examine success metrics** to assess both individual solution quality and ensemble diversity.
