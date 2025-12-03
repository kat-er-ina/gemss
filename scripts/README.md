# GEMSS Experiment Scripts

This directory contains scripts and configuration files for running systematic algorithm validation experiments with GEMSS on artificial datasets. These experiments are designed for benchmarking and validating the GEMSS algorithm across different data scenarios and parameter configurations.

Even though these experiments are **not** intended for hyperparameter optimization, *run_sweep.ps1* can be easily adapted for that purpose.

## Directory Structure

```
experiment_parameters.json          # Full 6-tier experimental design (476 total experiments)
experiment_parameters_short.json    # Shortened version for quick testing (30 experiments)
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

## Experimental Design Overview

The experiment suite uses a **7-tier design** (110 experiments in full, 28 in short version). Each tier targets a specific scenario:

### Tier 1: Basic Validation (18 experiments)
- **Purpose:** Baseline performance on almost clean data (NOISE_STD=0.1, NAN_RATIO=0.0)
- **Description:** Systematically varies the number of samples and features to cover the n << p transition zone. Tests two low sparsity levels. Verifies core algorithm correctness and stability in ideal conditions.
- **Parameter ranges:** N_SAMPLES: 25 – 200, N_FEATURES: 100 – 500, SPARSITY: 3 or 5, N_CANDIDATE_SOLUTIONS: 6 or 9.
- **Response:** Binary classification

### Tier 2: High-Dimensional Stress Test (9 experiments)
- **Purpose:** Scalability and performance when P ≥ 1000 and N << P
- **Description:** Pushes the algorithm into high-dimensional regimes typical of omics/biomedical data. Uses moderate sample sizes with fixed ultra sparsity. Assesses support recovery and convergence in ultra-sparse, high-p settings.
- **Parameter ranges:** N_SAMPLES: 50 – 200, N_FEATURES: 1000 – 5000, SPARSITY: 5, N_CANDIDATE_SOLUTIONS: 12.
- **Response:** Binary classification

### Tier 3: Sample-Rich Scenarios (14 experiments)
- **Purpose:** Control group for convergence when information is abundant (N ≥ P)
- **Description:** Traditional ML regime with abundant samples relative to features. Tests two sparsity levels. Ensures the algorithm converges and recovers supports when data is plentiful, serving as a baseline for comparison with harder tiers.
- **Parameter ranges:** N_SAMPLES: 100 – 1000, N_FEATURES: 100 – 500, SPARSITY: 3 or 5, N_CANDIDATE_SOLUTIONS: 6 or 9.
- **Response:** Binary classification

### Tier 4: Robustness Under Adversity (22 experiments)
- **Purpose:** Test stability under severe noise and missing data
- **Description:** Uses fixed configurations [N=100, P=200] and [N=200, P=500] with systematic variation of data quality. Tests high noise and high missing data ratios. Batch size dynamically increases (16 → 24 → 48) with missing data ratio. Evaluates robustness to real-world data quality issues.
- **Parameter ranges:** N_SAMPLES: 100 or 200, N_FEATURES: 200 or 500, NOISE_STD: 0.1–1.0, NAN_RATIO: 0.0 – 0.5, BATCH_SIZE: 16 – 48
- **Response:** Binary classification

### Tier 5: Effect of Jaccard Penalty (12 experiments)
- **Purpose:** Investigate the effect of diversity regularization (LAMBDA_JACCARD)
- **Description:** Uses clean data on representative problem sizes with both sparsity levels. In addition to standard penalty setting, this tier tests three more penalty regimes to analyze how Jaccard regularization influences solution diversity and support overlap.
- **Parameter ranges:** N_SAMPLES: 100, N_FEATURES: 200 or 500, SPARSITY: 3 or 5, LAMBDA_JACCARD: 0 or 1000 or 5000
- **Response:** Binary classification

### Tier 6: Regression Validation (29 experiments)
- **Purpose:** Validate performance on continuous response (regression)
- **Description:** Comprehensive testing of regression settings. It mirrors (with a single sparsity setup) Tier 1 (basic configuration), Tier 2 (high dimensions), and Tier 4 (robustness tests) but it uses a continuous response.
- **Parameter ranges:** N_SAMPLES: 25 – 200, N_FEATURES: 100 – 5000, SPARSITY: 5, NOISE_STD: 0.1 – 1.0, NAN_RATIO: 0.0 – 0.5
- **Response:** Continuous regression

### Tier 7: Class Imbalance (6 experiments)
- **Purpose:** Test stability with unbalanced binary responses
- **Description:** Tests severe class imbalance with minority class ratios of 10%, 20%, and 30%. Assesses algorithm robustness to label imbalance across different sample sizes.
- **Parameter ranges:** N_SAMPLES: 100 or 200, N_FEATURES: 200 or 500, SPARSITY: 5, BINARY_RESPONSE_RATIO: 0.1 – 0.3
- **Response:** Binary classification

**Parameter format:**
`N_SAMPLES,N_FEATURES,N_GENERATING_SOLUTIONS,SPARSITY,NOISE_STD,NAN_RATIO,N_CANDIDATE_SOLUTIONS,LAMBDA_JACCARD,BATCH_SIZE,BINARY_RESPONSE_RATIO`

**Common algorithm parameters:**
- N_GENERATING_SOLUTIONS: 3
- N_ITER: 3500–4500 (see tier)
- PRIOR_TYPE: sss, VAR_SLAB: 100.0, VAR_SPIKE: 0.1
- IS_REGULARIZED: true
- LEARNING_RATE: 0.002
- MIN_MU_THRESHOLD: 0.2
- DATASET_SEED: 42

**How to run:**
```powershell
# Single experiment
python run_experiment.py

# Full suite (all tiers, 110 experiments)
.\run_tiers.ps1 -parametersFile "experiment_parameters.json"

# Short suite (28 experiments)
.\run_tiers.ps1 -parametersFile "experiment_parameters_short.json"

# Selected tiers
.\run_tiers.ps1 -tiers @("1","4") -parametersFile "experiment_parameters.json"
```

**Results:**
Saved in `results/` with tier-specific subfolders and logs.

## Design Notes

- **Total experiments**: 476 (full) / 30 (short)
- **Tier breakdown**: 
  - Tier 1: 64 experiments
  - Tier 2: 36 experiments  
  - Tier 3: 144 experiments
  - Tier 4: 48 experiments
  - Tier 5: 48 experiments
  - Tier 6: 32 experiments
  - Tier 7: 6 experiments
- **Focus**: Ultra-sparse signals (3-5 features across all scenarios)
- **Response types**: Binary classification (Tiers 1-5, 7) + Continuous regression (Tier 6)
- **Data generation**: Seeded (DATASET_SEED=42) for reproducibility
- **Evaluation**: Fixed outlier detection thresholds [2.0, 2.5, 3.0]

## Limitations

- Experimental design does not account for stochastic effects throughout the algorithm and regression evaluation, as seeding is not fully supported for every included statistical component.
- Artificial dataset generation is properly seeded for reproducibility.
- Tier 6 introduces continuous regression scenarios while Tiers 1-5 and 7 focus on binary classification.