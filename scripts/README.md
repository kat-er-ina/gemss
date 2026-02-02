# GEMSS Experiment Scripts

This directory contains scripts and configuration files for running systematic algorithm validation experiments with GEMSS on artificial datasets. These experiments are designed for benchmarking and validating the GEMSS algorithm across different data scenarios and parameter configurations.

Even though these experiments are **not** intended for hyperparameter optimization, *run_sweep.ps1* can be easily adapted for that purpose.

## Directory structure

```
experiment_parameters.json          # Full 7-tier experimental design (128 total experiments)
run_experiment.py                   # Python script to run a single experiment
run_sweep.ps1                       # PowerShell script for batch parameter sweeps (all combinations)
run_tiers.ps1                       # PowerShell script for tiered experiment runs (multi-condition)
run_sweep_with_tiers.ps1            # Internal script called by run_tiers.ps1 for each tier
results/                            # Output directory for experiment results and logs
  logs/                             # Log files for each tier and run
  tier1/                            # Output files for tier 1 experiments
  tier2/                            # Output files for tier 2 experiments
  tier3/                            # Output files for tier 3 experiments
  tier4/                            # Output files for tier 4 experiments
  tier5/                            # Output files for tier 5 experiments
  tier6/                            # Output files for tier 6 experiments
  tier7/                            # Output files for tier 7 experiments
```

## Experimental design overview

The experiment suite uses a **7-tier design** (128 experiments total). Each tier targets a specific scenario:

### Tier 1: Basic validation (18 experiments)

* **Purpose:** Baseline performance on almost clean data (NOISE_STD=0.1, NAN_RATIO=0.0)  
* **Description:** Systematically varies the number of samples and features to cover the *n < p* and *n << p* zones with small and medium number of features. Tests two low sparsity levels. Verifies core algorithm correctness and stability in ideal conditions.
* **Parameter ranges:** N_SAMPLES: 25 – 200, N_FEATURES: 100 – 500, SPARSITY: 3 or 5, N_CANDIDATE_SOLUTIONS: 6 or 9.
* **Response:** Binary classification.

### Tier 2: High-dimensional stress test (9 experiments)

* **Purpose:** Scalability and performance when *p ≥ 1000* and *n << p*
* **Description:** Pushes the algorithm into high-dimensional regimes. Uses realistic small/medium sample sizes with fixed ultra sparsity. Assesses support recovery and convergence in ultra-sparse, high-p settings.  
* **Parameter ranges:** N_SAMPLES: 50 – 200, N_FEATURES: 1000 – 5000, SPARSITY: 5, N_CANDIDATE_SOLUTIONS: 12.
* **Response:** Binary classification.

### Tier 3: Sample-rich scenarios (14 experiments)

* **Purpose:** Control group for convergence when information is abundant (*n ≥ p*).
* **Description:** Traditional ML regime with abundant samples relative to features. Tests two sparsity levels. Ensures the algorithm converges and recovers supports when data is plentiful, serving as a baseline for comparison with harder tiers.
* **Parameter ranges:** N_SAMPLES: 100 – 1000, N_FEATURES: 100 – 500, SPARSITY: 3 or 5, N_CANDIDATE_SOLUTIONS: 6 or 9.
* **Response:** Binary classification.

### Tier 4: Robustness under adversity (22 experiments)

* **Purpose:** Test stability under severe noise and missing data  
* **Description:** Uses fixed configurations \[N=100, P=200\] and \[N=200, P=500\] with systematic variation of data quality. Tests high noise and high missing data ratios. Batch size dynamically increases (16 → 24 → 48\) with missing data ratio. Evaluates robustness to real-world data quality issues.  
* **Parameter ranges:** N_SAMPLES: 100 or 200, N_FEATURES: 200 or 500, NOISE_STD: 0.1–1.0, NAN_RATIO: 0.0 – 0.5, BATCH_SIZE: 16 – 48.
* **Response:** Binary classification.

### Tier 5: Effect of Jaccard penalty (28 experiments)

* **Purpose:** Investigate the effect of diversity regularization (LAMBDA_JACCARD)  
* **Description:** Uses clean data on representative problem sizes with both sparsity levels. This tier tests multiple penalty regimes to analyze how Jaccard regularization influences solution diversity and support overlap. To make the difference more prominent, the number of candidate solutions is intentionally too low: it equals the number of generating solutions. I.e. it would require perfect diversification in order to achieve full recall in "top" solutions.
* **Parameter ranges:** N_SAMPLES: 100, N_FEATURES: 200 or 500, SPARSITY: 3 or 5, LAMBDA_JACCARD: 0 to 10000, N_GENERATING_SOLUTIONS = N_CANDIDATE_SOLUTIONS = 3.
* **Response:** Binary classification.

### Tier 6: Regression validation (29 experiments)

* **Purpose:** Validate performance on continuous response (regression).
* **Description:** Comprehensive testing of regression settings. It mirrors the experimental conditions (n, p, noise, NaNs, sparsity) of Tier 1 (basic configuration), Tier 2 (high dimensions), and Tier 4 (robustness tests) but uses a continuous response variable instead of binary labels.
* **Parameter ranges:** N_SAMPLES: 25 – 200, N_FEATURES: 100 – 5000, SPARSITY: 5, NOISE_STD: 0.1 – 1.0, NAN_RATIO: 0.0 – 0.5.
* **Response:** Continuous regression.

### Tier 7: Class imbalance (8 experiments)

* **Purpose:** Test stability with unbalanced binary responses.
* **Description:** Tests severe class imbalance with minority class ratios of 10%, 20%, and 30%. Assesses algorithm robustness to label imbalance across different sample sizes.
* **Parameter ranges:** N_SAMPLES: 100 or 200, N_FEATURES: 200 or 500, SPARSITY: 5, BINARY_RESPONSE_RATIO: 0.1 – 0.3.
* **Response:** Binary classification.

### Definition of test cases

The 128 experiments are organized into **47 test cases** for analysis purposes. These test cases group experiments across tiers to answer specific research questions about algorithm performance.

#### Classification baseline and scalability (Cases 1-16)

**Baseline performance (Tier 1):**
- **Case 1:** Baseline performance for SPARSITY = 3
- **Case 2:** Baseline performance for SPARSITY = 5
- **Case 3:** Overall baseline performance comparing sparsity levels

**High-dimensional scalability (Tiers 1 + 2 + 3):**
- **Case 4:** Scalability analysis: p = 1000
- **Case 5:** Scalability analysis: p = 2000
- **Case 6:** Scalability analysis: p = 5000
- **Case 7:** Scalability analysis: n = 50 (Tiers 1 + 2 + 3)
- **Case 8:** Scalability analysis: n = 100 (Tiers 1 + 2 + 3)
- **Case 9:** Scalability analysis: n = 200 (Tiers 1 + 2 + 3)
- **Case 10:** Overall high-dimensional performance (Tier 2)

**Sample-rich scenarios (Tiers 1+3):**
- **Case 11:** Sample-rich performance for SPARSITY = 3
- **Case 12:** Sample-rich performance for SPARSITY = 5
- **Case 13:** Overall sample-rich performance
- **Case 14:** Overall analysis of n/p ratio, SPARSITY = 3 (Tiers 1 + 3)
- **Case 15:** Overall analysis of n/p ratio, SPARSITY = 5 (Tiers 1 + 3)

**Summary of basic experiments (Tiers 1+2+3):**
- **Case 16:** Overall basic performance for SPARSITY = 5 (Tiers 1 + 2 + 3)

#### Classification robustness and diversity (Cases 17-31)

**Adversity: noise and missing data (Tiers 1+4):**
- **Case 17:** [n=100, p=200], varying noise only
- **Case 18:** [n=100, p=200], varying NaNs only
- **Case 19:** [n=100, p=200], varying both noise and NaNs
- **Case 20:** [n=200, p=500], varying noise only
- **Case 21:** [n=200, p=500], varying NaNs only
- **Case 22:** [n=200, p=500], varying both noise and NaNs
- **Case 23:** Comparing [n=100, p=200] and [n=200, p=500] under adversity

**Jaccard penalty effects (Tiers 1+5):**
- **Case 24:** Effect of Jaccard penalty for [n=100, p=200]
- **Case 25:** Effect of Jaccard penalty for [n=100, p=500]
- **Case 26:** Effect of Jaccard penalty for SPARSITY = 3
- **Case 27:** Effect of Jaccard penalty for SPARSITY = 5
- **Case 28:** Overall effect of Jaccard penalty

**Class imbalance (Tiers 1+7):**
- **Case 29:** Effect of class imbalance for [n=100, p=200]
- **Case 30:** Effect of class imbalance for [n=200, p=500]
- **Case 31:** Overall effect of class imbalance

#### Regression validation (Cases 32-42, Tier 6)

**Baseline and scalability:**
- **Case 32:** Regression baseline performance (p ≤ 500)
- **Case 33:** Overall baseline + high-dimensional performance
- **Case 34:** Regression scalability: p = 1000
- **Case 35:** Regression scalability: p = 2000
- **Case 36:** Regression scalability: p = 5000
- **Case 37:** Regression scalability: baseline + high-dim for n = 50
- **Case 38:** Regression scalability: baseline + high-dim for n = 100
- **Case 39:** Regression scalability: baseline + high-dim for n = 200

**Adversity:**
- **Case 40:** Effect of varying noise
- **Case 41:** Effect of varying NaNs
- **Case 42:** Overall effect of varying both noise and NaNs

#### Regression vs. Classification comparison (Cases 43-47, Tiers 1+2+4+6)

- **Case 43:** Basic scenarios (Tiers 1 + 6)
- **Case 44:** High-dimensional scenarios (Tiers 2 + 6)
- **Case 45:** Effect of noise (Tiers 4 + 6)
- **Case 46:** Effect of NaNs (Tiers 4 + 6)
- **Case 47:** Effect of both noise and NaNs (Tiers 4 + 6)

Each test case combines relevant experiments to isolate specific factors (dimensionality, noise, sparsity, response type, etc.) and provide statistical power for meaningful conclusions. The complete case definitions and analysis parameters are implemented in `gemss/experiment_assessment/case_analysis.py` and analyzed in `notebooks/analyze_experiment_results/`.

## Parameters

### Parameter configuration files

Experiments use parameters defined in `experiment_parameters.json`, which specifies parameter combinations for each tier. The format is:

```
N_SAMPLES,N_FEATURES,N_GENERATING_SOLUTIONS,SPARSITY,NOISE_STD,NAN_RATIO,N_CANDIDATE_SOLUTIONS,LAMBDA_JACCARD,BATCH_SIZE,BINARY_RESPONSE_RATIO
```

### Fixed parameters (all tiers)

These remain constant across all experiments:

* `N_ITER`: 3500–5000 (varies by tier complexity)
* `PRIOR_TYPE`: `sss` (structured spike-and-slab)
* `VAR_SLAB`: 100.0
* `VAR_SPIKE`: 0.1
* `IS_REGULARIZED`: true
* `LEARNING_RATE`: 0.002
* `MIN_MU_THRESHOLD`: 0.2
* `DATASET_SEED`: 42
* `BINARIZE`: true (except Tier 6: regression)

### Variable parameters (per experiment)

These vary across experiments to test different scenarios:

**Dataset dimensions:**
* `N_SAMPLES`: Number of samples (25 - 1000 across tiers)
* `N_FEATURES`: Feature dimensionality (100 - 5000 across tiers)
* `N_GENERATING_SOLUTIONS`: True sparse supports generating the response
* `SPARSITY`: Features per generating solution (k)

**Data quality:**
* `NOISE_STD`: Gaussian noise standard deviation (0.1 - 1.0)
* `NAN_RATIO`: Fraction of missing values (0.0 - 0.5)

**Algorithm configuration:**
* `N_CANDIDATE_SOLUTIONS`: Mixture components optimized by GEMSS
* `LAMBDA_JACCARD`: Diversity penalty weight (0 - 10000 in Tier 5)
* `BATCH_SIZE`: Minibatch size (16–64, scales with problem complexity)

**Classification settings:**
* `BINARY_RESPONSE_RATIO`: Class proportion (0.1 - 0.5 for imbalanced scenarios)

## How to reproduce experiments on artificial data

In order to use correct Python dependencies, it is recommended that scripts are run using `uv run python` instead of the `python` command.

```bash
# Single experiment  
uv run python run_experiment.py
```

**Batch experiments (PowerShell):** for the PowerShell scripts, it is often easier to activate the environment first:

```bash
# Activate environment (Windows)
.venv\Scripts\activate.ps1
```

Then run:

```bash
# Full suite (all tiers, 128 experiments)  
.\run_tiers.ps1 -parametersFile "experiment_parameters.json"

# Selected tiers  
.\run_tiers.ps1 -tiers @("1","4") -parametersFile "experiment_parameters.json"
```

## Experiment results

Saved in `results/` with tier-specific subfolders and logs.  
The `tier_summary_metrics.csv` files contain aggregated performance metrics (Recall, Precision, F1, Success Index, etc.) calculated for three different solution extraction methods:

1. **Full:** All features with non-zero importance.  
2. **Top:** Top features matching the DESIRED_SPARSITY.  
3. **Outlier:** Features detected as outliers using deviations of 2.0, 2.5, and 3.0.


### Assessment of results

Results can be viewed and analyzed by notebooks provided in directory `notebooks/analyze_experiment_results/`:

- Analysis per tier: `tier_level_analysis.ipynb` - used mainly for designing experimental tiers.
- Analysis per test case: `analysis_per_testcase.ipynb` - used for the actual assessment of results.

### Evaluation metrics

While the experiments can be and are evaluated in many ways, the main focus is on how the joint set of all discovered features (i.e. all features across all candidate solutions) corresponds with the "true" joint set of all the generating features. The provided metrics (recall, precision, F1 score) thus relate to the relationship between these two sets, *not* to any predictive potential.

Details are described in the [technical report](../technical_report.pdf).

## **Limitations**

* Artificial dataset generation is properly seeded for reproducibility. However, individual runs of the feature selector may differ since seeding is not fully supported for every included statistical component.
* Experiment design does not account for stochastic effects throughout the algorithm and regression evaluation. The experiments were not replicated. The important trends are apparent due to the overall number of experiments.
