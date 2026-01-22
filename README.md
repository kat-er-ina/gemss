# GEMSS: Gaussian Ensemble for Multiple Sparse Solutions

This repository implements Bayesian sparse feature selection using variational inference with Gaussian mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.


## Citation

If you use GEMSS in your research, please cite:

```bibtex
@misc{GEMSS2026,
  author = {Henclova, Katerina and Smidl, Vaclav},
  title = {GEMSS: A Variational Bayesian Method for Discovering Multiple Sparse Solutions in Classification and Regression Problems},
  year = {2026},
  note = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

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

## Package installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. `uv` replaces tools like `pip`.

### 1. Install uv
If you do not have `uv` installed, run one of the following commands:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the environment

Navigate to the repository root and sync the environment. This command will create a virtual environment and install all dependencies (including the `gemss` package itself) defined in `pyproject.toml`.

```bash
uv sync
```

## Quick start

GEMSS can be applied to both custom datasets and synthetic data for validation and benchmarking.

### Interactive demos

**Demo:** [notebooks/demo.ipynb](notebooks/demo.ipynb) — complete synthetic data walkthrough

**Custom data:** [notebooks/explore_custom_dataset.ipynb](notebooks/explore_custom_dataset.ipynb) — guided workflow for your datasets

The notebooks can be opened either in your favorite editor or by using `uv run`, e.g.:

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

**Detailed documentation:** [notebooks/README.md](notebooks/README.md)

### Custom datasets

1. Place your CSV file in `data/`
2. Open [notebooks/explore_custom_dataset.ipynb](notebooks/explore_custom_dataset.ipynb)
3. Configure data loading (file name, index, target column)
4. Review preprocessing outputs and adjust as needed
5. Adjust algorithm hyperparameters
6. Run and review diagnostics/visualizations
7. Iterate based on convergence and results

**Data requirements:** Numerical features preferred. Missing values handled natively. Preprocessing utilities include standard/minmax scaling.


## Proof-of-concept experiments

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

### Running experiments

In order to use correct Python dependencies, it is recommended that scripts are run using `uv run python` instead of the `python` command.

```bash
# Single experiment
uv run python scripts/run_experiment.py
```

**Batch experiments (PowerShell):** for the PowerShell scripts, it is often easier to activate the environment first:

```bash
# Activate environment (Windows)
.venv\Scripts\activate.ps1
```

Then run:

```bash
# Parameter sweeps (custom parameter setting)
.\scripts\run_sweep.ps1

# The benchmark (128 experiments)
.\scripts\run_tiers.ps1                          # Full suite
.\scripts\run_tiers.ps1 -tiers @("1","4")        # Selected tiers
```

### Result analysis

* [notebooks/analyze_experiment_results/tier_level_analysis.ipynb](notebooks/analyze_experiment_results/tier_level_analysis.ipynb) — tier-level performance
* [notebooks/analyze_experiment_results/analysis_per_testcase.ipynb](notebooks/analyze_experiment_results/analysis_per_testcase.ipynb) — cross-tier research questions


**For more details, see the dedicated documentation:** [scripts/README.md](scripts/README.md)


## License

This project is licensed under the MIT License.
