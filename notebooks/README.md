# Notebooks guide

This directory contains interactive Jupyter notebooks for exploring GEMSS capabilities, from quick-start demos to in-depth analysis of the results and their use in modeling. The notebooks support both learning the algorithm and applying it to real-world data.


## Overview of the notebooks

### Quick start

* **[demo.ipynb](demo.ipynb)** — Complete end-to-end demonstration of GEMSS on synthetic data with known ground truth. Covers data generation, algorithm execution, convergence diagnostics, solution extraction, and validation. **Start here** if you're new to GEMSS.

* **[explore_custom_dataset.ipynb](explore_custom_dataset.ipynb)** — Apply GEMSS to your own dataset with unknown ground truth. Mirrors the `demo.ipynb` workflow adapted for custom data. Supports experiment persistence and reloading.

### Downstream modeling notebooks

* **[tabpfn_evaluation_example.ipynb](tabpfn_evaluation_example.ipynb)** — Demonstrates advanced evaluation of discovered feature sets using TabPFN (a Transformer-based prior-data fitted network for tabular data).

* **[tabpfn_evaluate_custom_dataset_results.ipynb](tabpfn_evaluate_custom_dataset_results.ipynb)** — Load saved GEMSS solutions and evaluate them with TabPFN on your custom dataset.

### Notebooks for experiment analysis

The `analyze_experiment_results/` directory is dedicated to evaluation of extensive experiments that provide validation for the newly published GEMSS algorithm. The notebooks analyze the results provided in the `scripts/results`:
* Aggregating metrics across parameter sweeps
* Comparing algorithm configurations
* Generating performance reports


## Workflow Overview

A typical GEMSS workflow consists of the following steps:

1. **Data preparation:** Load your dataset or generate synthetic data.
2. **Configuration:** Set algorithm parameters via JSON files or programmatically.
3. **Execution:** Run the Bayesian feature selector.
4. **Diagnostics:** Assess convergence and algorithm performance.
5. **Solution extraction:** Recover sparse feature sets ("candidate solutions") from the mixture model.
6. **Persistence:** Save results for reproducibility and further analysis.
7. **Validation & downstream modeling:** Evaluate solutions' predictive potential using simplified runs of logistic/linear regression or advanced models (e.g. TabPFN).

The notebooks guide you through each of these steps with detailed examples and explanations.


## Getting Started

First, [install the package](../README.md#package-installation) and activate the environment as described.

### First-time users

1. Open [demo.ipynb](demo.ipynb).
2. Run all cells to see GEMSS in action on synthetic data. Depending on the setting (mostly number of iterations and batch size), it will take a few minutes to run.
3. Experiment with different parameter values in the configuration cells.
4. Review the [Parameter configuration](#parameter-configuration) section below.

### Using your own data

1. Prepare your dataset as a CSV file. It will be loaded as a Pandas DataFrame with:
   - an index column,
   - a target column (numeric for regression, binary for classification),
   - feature columns (numeric).
2. Open [explore_custom_dataset.ipynb](explore_custom_dataset.ipynb).
3. Follow the data loading instructions.
4. Configure parameters appropriate to your dataset size and problem type.
5. Run the analysis and save results for later evaluation.


## Parameter configuration

Three JSON files in `gemss/config/` control all parameters:

1. **algorithm_settings.json** — Core algorithm parameters
2. **generated_dataset_parameters.json** — Artificial data generation (dev/demo)
3. **solution_postprocessing_settings.json** — Solution extraction settings

You can print the configurations using the following function:

```python
import gemss.config as C
C.display_current_config(constants=C.as_dict(), constant_type='all')
```

### Available algorithm parameters

**Optimization:**
* `N_ITER`: Number of optimization iterations.
* `LEARNING_RATE`: Learning rate for the Adam optimizer.
* `BATCH_SIZE`: Minibatch size for stochastic updates in the SGD optimization. Depends on available number of samples and adverse effects (noise, NaNs, class imbalance etc.)

**Prior configuration:**
* `PRIOR_TYPE`: Prior type ('ss', 'sss', or 'student')
* `PRIOR_SPARSITY`: Expected number of nonzero features per component. Used only in 'sss' prior
* `VAR_SLAB`: Variance of the 'slab' component in the 'ss' or 'sss' prior. Ignored for 'student' prior.
* `VAR_SPIKE`: Variance of the 'spike' component in the 'ss' or 'sss' prior. Ignored for 'student' prior.
* `WEIGHT_SLAB`: Weight of the 'slab' component in the 'ss' prior. Ignored for other priors.
* `WEIGHT_SPIKE`: Weight of the 'spike' component in the 'ss' prior. Ignored for other priors.
* `STUDENT_DF`: Degrees of freedom for the Student-t prior. Used only if PRIOR_TYPE is 'student'.
* `STUDENT_SCALE`: Scale parameter for the Student-t prior. Used only if PRIOR_TYPE is 'student'.
* `SAMPLE_MORE_PRIORS_COEFF`: Coefficient for increased support sampling. Experimental use only.

**Solution recovery:**
* `N_CANDIDATE_SOLUTIONS`: Desired number of candidate solutions (components of the Gaussian mixture approximating the variational posterior). Set to 2-3x the value of expected true solutions.
* `DESIRED_SPARSITY`: Desired number of features in final solution.
* `MIN_MU_THRESHOLD`: Minimum mu threshold for feature selection. Specific for each dataset.
* `USE_MEDIAN_FOR_OUTLIER_DETECTION`: Whether to use median and MAD or mean and STD when selecting features by outlier detection.
* `OUTLIER_DEVIATION_THRESHOLDS`: A list of thresholding values of either MAD or STD to be used to define outliers.

**Diversity control:**
* `IS_REGULARIZED`: Whether to use Jaccard similarity penalty.
* `LAMBDA_JACCARD`: Regularization strength for Jaccard penalty. Applies only if IS_REGULARIZED is True.

### Artificial dataset parameters

For synthetic data generation (experiments and demos):

* `N_SAMPLES`: Number of samples (rows) in the synthetic dataset.
* `N_FEATURES`: Number of features (columns) in the synthetic dataset.
* `N_GENERATING_SOLUTIONS`: Number of distinct sparse solutions ('true' supports).
* `SPARSITY`: Number of nonzero features per solution (support size).
* `NOISE_STD`: Standard deviation of noise added to synthetic data.
* `NAN_RATIO`: Proportion of missing values (NaNs) in the synthetic dataset.
* `BINARIZE`: Whether to binarize the synthetic response variable.
* `BINARY_RESPONSE_RATIO`: Proportion of synthetic samples assigned label 1.
* `DATASET_SEED`: Random seed for synthetic data reproducibility.


## Missing data handling

GEMSS natively supports missing feature values without imputation or sample removal. The algorithm automatically handles missing data during likelihood computation. Only samples without a valid target are dropped. For datasets with substantial missing data, increase the batch size.

## Persistence & reproducibility

GEMSS provides utilities to save and load complete analysis runs, ensuring full reproducibility:

```python
from gemss.utils.utils import (
    save_feature_lists_json, load_feature_lists_json,
    save_selector_history_json, load_selector_history_json,
    save_constants_json, load_constants_json
)

# Save complete run
save_constants_json(constants, "search_setup.json")  # inputs
save_selector_history_json(history, "search_history_results.json")  # trajectory
save_feature_lists_json(solutions, "all_candidate_solutions.json")  # outputs

# Load for analysis
constants, _ = load_constants_json("search_setup.json")
history, _ = load_selector_history_json("search_history_results.json")
solutions, _ = load_feature_lists_json("all_candidate_solutions.json")
```

## Algorithm performance monitoring

Monitor convergence, extract solutions, and assess algorithm performance with built-in diagnostic tools. They provide essential information for performance assessment and hyperparameter tuning.

```python
from gemss.postprocessing.result_postprocessing import (
    show_algorithm_progress, recover_solutions
)
from gemss.utils.utils import show_solution_summary

# Visual convergence diagnostics
show_algorithm_progress(history)

# Extract solutions
solutions = recover_solutions(history, desired_sparsity=5)

# Tabular summary
show_solution_summary(solutions)
```

**Advanced diagnostics (work in progress):**
```python
from gemss.diagnostics.performance_tests import run_performance_diagnostics
from gemss.diagnostics.recommendations import display_recommendations

diagnostics = run_performance_diagnostics(history, desired_sparsity=5)
display_recommendations(diagnostics=diagnostics, constants=C.as_dict())
```

## Tips & Best Practices

### Parameter tuning

* **Start small:** Begin with fewer iterations (~2000) to test configuration, then scale up
* **Monitor convergence:** Use `show_algorithm_progress(history)` to verify the algorithm has converged
* **Batch size:** Increase for datasets with many missing values (30-50% missing → batch_size ≥ 32)
* **Diversity:** Enable regularization (`IS_REGULARIZED=True`) and tune `LAMBDA_JACCARD` if solutions are too similar

### Performance optimization

* **Sparsity guidance:** Set `DESIRED_SPARSITY` close to your expected number of relevant features
* **Number of solutions:** Start with `N_CANDIDATE_SOLUTIONS=3-5`, increase if you need more diversity
* **Iterations:** Typical ranges: 3500-5000 for well-behaved problems, 7000+ for challenging datasets

### Common issues

* **Non-convergence:** Increase `N_ITER`, decrease `LEARNING_RATE`, or adjust prior variances
* **Similar solutions:** Increase `LAMBDA_JACCARD` or reduce prior variance ratio
* **Empty solutions:** Decrease `MIN_MU_THRESHOLD` or adjust prior parameters
* **Slow execution:** Reduce `N_ITER`, `BATCH_SIZE`, or `N_CANDIDATE_SOLUTIONS`


## Solution evaluation

Validate discovered feature sets using lightweight regression baselines or advanced TabPFN evaluation.

### Quick validation (linear/logistic regression)

```python
from gemss.postprocessing.simple_regressions import solve_any_regression

results = solve_any_regression(
    solutions=solutions_dict,
    df=df,
    response=y,
    apply_scaling="standard"  # or "minmax" or None
)
```

Returns task-appropriate metrics (R², MSE, accuracy, F1, etc.). Cross-validation is performed to optimize the parameters but the validation is only run on the full training dataset.

### Advanced evaluation with TabPFN

The following notebooks provide tools for downstream evaluation of feature sets discovered by GEMSS:

* [notebooks/tabpfn_evaluation_example.ipynb](notebooks/tabpfn_evaluation_example.ipynb) — Example TabPFN evaluation workflow.
* [notebooks/tabpfn_evaluate_custom_dataset_results.ipynb](notebooks/tabpfn_evaluate_custom_dataset_results.ipynb) — Evaluate feature sets discovered by GEMSS on your custom data.

The core functionality:

```python
from gemss.postprocessing.tabpfn_evaluation import tabpfn_evaluate

results = tabpfn_evaluate(
    X_selected, y,
    apply_scaling="standard",
    outer_cv_folds=3,
    explain=True,  # optional SHAP values computation (costly)
    shap_sample_size=100  # cap SHAP sample size
    random_state=42
)
```

Provides nested CV metrics and optional SHAP feature importance.

**Warning:** TabPFN is only usable on problems with small to medium number of features. Also, SHAP can be very slow if the dimension exceeds but a few features.
