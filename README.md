# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Citation

If you use GEMSS in your research, please cite:

```bibtex
@article{GEMSS2026,
  author = {Katerina Henclova, Vaclav Smidl},
  title = {GEMSS: A Variational Bayesian Method for Discovering Multiple
Sparse Solutions in Classification and Regression Problems},
  journal = {Journal Name},
  year = {2026},
  volume = {XX},
  pages = {XX--XX},
  doi = {XX.XXXX/XXXXX}
}
```

---

## Features

GEMSS provides a comprehensive framework for Bayesian feature selection with the following capabilities:

* **Multiple sparse solutions:** Recovers diverse sparse feature sets rather than a single solution
* **Missing data:** Native handling without imputation
* **Flexible priors:** Structured spike-and-slab (default), Student-t, vanilla spike-and-slab
* **Variational inference:** PyTorch-based optimization
* **Diversity regularization:** Optional Jaccard penalty for enforcing solution diversity
* **Visualization:** Interactive plots and comprehensive diagnostics
* **Modular configuration:** JSON-based dataset/algorithm/postprocessing settings
* **Batch experiments:** Parameter sweeps and tiered validation suites

---

## Repository structure

The repository is organized into core packages, interactive notebooks, batch experiment scripts, and configuration files:

```
gemss/
  data/                      # User datasets
  notebooks/                 # Interactive demos and analysis
    demo.ipynb               # End-to-end synthetic demo
    explore_custom_dataset.ipynb      # Custom data workflow
    tabpfn_evaluation_example.ipynb   # TabPFN evaluation demo
    tabpfn_evaluate_custom_dataset_results.ipynb # Evaluate saved solutions with TabPFN
    analyze_experiment_results/       # Experiment analysis (development)
    results/                          # Artifacts from the notebook runs
  scripts/                   # Batch experiments
    run_experiment.py        # Single experiment
    run_sweep.ps1            # Parameter sweeps
    run_tiers.ps1            # Tiered benchmark suite
    experiment_parameters.json    # 128-experiment design
    results/                 # Outputs and logs from the scripted experiments
  gemss/                     # Core package
    config/                  # JSON configuration files
    data_handling/           # Data generation and preprocessing
    feature_selection/       # Variational inference core
    postprocessing/          # Solution extraction and evaluation
    diagnostics/             # Performance diagnostics (WIP)
    experiment_assessment/   # Result analysis utilities
    utils/                   # Persistence and visualization
```

**Output artifacts:**
* `search_setup*.json` — configuration used
* `search_history_results*.json` — optimization trajectories
* `all_candidate_solutions*.json` — discovered feature sets
* `tier_summary_metrics.csv` — aggregated experiment metrics

## Configuration

Three JSON files in `gemss/config/` control all parameters:

1. **algorithm_settings.json** — Core algorithm parameters
2. **generated_dataset_parameters.json** — Artificial data generation (dev/demo)
3. **solution_postprocessing_settings.json** — Solution extraction settings

```python
import gemss.config as C
C.display_current_config(constants=C.as_dict(), constant_type='algorithm')
```

### Key algorithm parameters

**Optimization:**
* `N_ITER`: Number of optimization iterations (3500-5000 typical)
* `LEARNING_RATE`: Optimizer step size (default: 0.002)
* `BATCH_SIZE`: Minibatch size for stochastic updates (increase with missing data)

**Prior configuration:**
* `PRIOR_TYPE`: Prior family: `sss` (structured spike-and-slab), `student`, or `vanilla`
* `VAR_SLAB`, `VAR_SPIKE`: Spike-and-slab variances
* `WEIGHT_SLAB`, `WEIGHT_SPIKE`: Prior mixture weights

**Solution recovery:**
* `N_CANDIDATE_SOLUTIONS`: Number of mixture components to optimize
* `DESIRED_SPARSITY`: Target sparsity level for solution extraction
* `MIN_MU_THRESHOLD`: Importance threshold for feature inclusion

**Diversity control:**
* `IS_REGULARIZED`: Enable diversity regularization
* `LAMBDA_JACCARD`: Jaccard penalty weight (0 = no penalty, higher = more diversity)

### Artificial dataset parameters

For synthetic data generation (experiments and demos):

* `N_SAMPLES`, `N_FEATURES`: Dataset dimensions
* `N_GENERATING_SOLUTIONS`: Number of true sparse supports
* `SPARSITY`: Features per generating solution
* `NOISE_STD`: Gaussian noise level
* `NAN_RATIO`: Fraction of missing values
* `BINARIZE`: Binary classification (true) vs regression (false)
* `BINARY_RESPONSE_RATIO`: Class balance for classification

## Quick start

```bash
pip install -r requirements.txt
pip install -e .  # optional, for development
```

**Demo:** [notebooks/demo.ipynb](notebooks/demo.ipynb) — complete synthetic data walkthrough

**Custom data:** [notebooks/explore_custom_dataset.ipynb](notebooks/explore_custom_dataset.ipynb) — guided workflow for your datasets

## Usage

GEMSS can be applied to both custom datasets and synthetic data for validation and benchmarking.

### Custom datasets

1. Place your CSV file in `data/`
2. Open [notebooks/explore_custom_dataset.ipynb](notebooks/explore_custom_dataset.ipynb)
3. Configure data loading (file name, index, target column)
4. Review preprocessing outputs and adjust as needed
5. Adjust algorithm hyperparameters
6. Run and review diagnostics/visualizations
7. Iterate based on convergence and results

**Data requirements:** Numerical features preferred. Missing values handled natively. Preprocessing utilities include standard/minmax scaling.

### Experiments on artificial data

```bash
# Single experiment
python scripts/run_experiment.py

# Parameter sweeps
.\scripts\run_sweep.ps1

# Full tiered benchmark (128 experiments)
.\scripts\run_tiers.ps1
```

See [scripts/README.md](scripts/README.md) for detailed experimental design documentation.

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

### Advanced evaluation (TabPFN)

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

## Diagnostics

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

---

## Benchmark experiments

A comprehensive experimental framework validates GEMSS across diverse data scenarios, from clean baseline conditions to challenging high-dimensional and noisy settings.

There are 128 experiments organized in 7 tiers:

* **Tier 1:** Baseline (18): clean data, n < p
* **Tier 2:** High-dimensional (9): p ≥ 1000, n << p
* **Tier 3:** Sample-rich (14): n ≥ p
* **Tier 4:** Robustness (22): noise and missing data
* **Tier 5:** Jaccard penalty (28): diversity effects
* **Tier 6:** Regression (29): continuous response
* **Tier 7:** Class imbalance (8): unbalanced labels

Experiments are grouped into **47 test cases** addressing specific research questions.

**Run experiments:**
```bash
.\scripts\run_tiers.ps1                          # Full suite
.\scripts\run_tiers.ps1 -tiers @("1","4")        # Selected tiers
```

**Analyze results:**
* [notebooks/analyze_experiment_results/tier_level_analysis.ipynb](notebooks/analyze_experiment_results/tier_level_analysis.ipynb) — tier-level performance
* [notebooks/analyze_experiment_results/analysis_per_testcase.ipynb](notebooks/analyze_experiment_results/analysis_per_testcase.ipynb) — cross-tier research questions

**Detailed documentation:** [scripts/README.md](scripts/README.md)

---

## License

This project is licensed under the MIT License.
