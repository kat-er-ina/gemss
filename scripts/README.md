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
```

## Experimental Design Overview

The experiment suite follows a **6-tier design** addressing different research questions and use cases:

### Tier 1: Comprehensive Small-Scale Validation (64 experiments)
- **Purpose**: Extended proof of concept and algorithm verification on primary use case
- **Focus**: Progressive scale increase from n≈p to n<<p scenarios
- **Parameter ranges**: 20-150 samples, 100-1000 features, 3-20 sparsity levels
- **Response type**: Binary classification
- **Key scenarios**: Small to medium-scale problems with varying dimensionality

### Tier 2: Extreme High-Dimensional Stress Test (36 experiments)
- **Purpose**: Ultra-high-dimensional scenarios with ultra-sparse signal detection
- **Focus**: Realistic biomedical/omics scenarios
- **Parameter ranges**: 50-200 samples, 1000-10000 features, 3-20 sparsity levels
- **Response type**: Binary classification
- **Key scenarios**: Typical genomics/proteomics data characteristics

### Tier 3: Sample-Rich Scenarios (144 experiments)
- **Purpose**: Traditional ML problems where n ≥ p or n ~ p
- **Focus**: Well-powered statistical scenarios
- **Parameter ranges**: 60-2000 samples, 40-2000 features, 2-10 sparsity levels
- **Response type**: Binary classification
- **Key scenarios**: Classical machine learning regime with adequate sample sizes

### Tier 4: Robustness Under Adversity (48 experiments)
- **Purpose**: Algorithm stability with data quality challenges
- **Focus**: Noise robustness and missing data handling
- **Parameter variations**: Noise levels (0.1-1.0), missing data ratios (0.0-0.9)
- **Response type**: Binary classification
- **Key scenarios**: Real-world data quality issues

### Tier 5: Effect of Jaccard Penalty (48 experiments)
- **Purpose**: Investigate regularization effects on feature set diversity
- **Focus**: Jaccard penalty parameter exploration
- **Parameter variations**: Lambda values from 0 (no penalty) to 5000 (extreme penalty)
- **Response type**: Binary classification
- **Key scenarios**: Solution diversity and coverage analysis

### Tier 6: Regression Validation (32 experiments)
- **Purpose**: Regression scenarios covering key aspects with reduced scope
- **Focus**: Continuous response prediction under challenging conditions
- **Parameter ranges**: 50-100 samples, 1000-10000 features, 5-10 sparsity levels
- **Parameter variations**: Moderate to high noise (0.5-1.0), missing data (0.1-0.3)
- **Response type**: **Continuous regression** (BINARIZE = false)
- **Key scenarios**: High-dimensional regression with data quality challenges

## Parameter Format

All experiment combinations follow the format:
```
N_SAMPLES,N_FEATURES,N_GENERATING_SOLUTIONS,SPARSITY,NOISE_STD,NAN_RATIO,N_CANDIDATE_SOLUTIONS,LAMBDA_JACCARD
```

## Algorithm Configuration

**Common parameters across all tiers:**
- **Iterations**: 4000
- **Prior type**: Spike-and-Slab (sss)
- **Student-t parameters**: DF=1, Scale=1.0
- **Variance parameters**: Slab=100.0, Spike=0.1
- **Prior weights**: Slab=0.9, Spike=0.1
- **Regularization**: Enabled
- **Training**: Batch size=16, Learning rate=0.002

**Response type differences:**
- **Tiers 1-5**: Binary classification (BINARIZE=true, ratio=0.5)
- **Tier 6**: Continuous regression (BINARIZE=false)

## How to Run Experiments

### Single experiment:
```powershell
python run_experiment.py
```

### Batch sweep (all combinations):
```powershell
.un_sweep.ps1
```

### Tiered experiments:
- **Run all tiers** (476 experiments):
```powershell
.un_tiers.ps1
```

- **Run specific tiers** with custom parameter file:
```powershell
.un_tiers.ps1 -tiers @("1", "2", "6") -parametersFile experiment_parameters.json
```

- **Run shortened validation** (30 experiments):
```powershell
.un_tiers.ps1 -parametersFile experiment_parameters_short.json
```

- **Run only regression experiments** (Tier 6):
```powershell
.un_tiers.ps1 -tiers @("6") -parametersFile experiment_parameters.json
```

All results and logs are automatically saved in the `results/` directory with tier-specific organization for later analysis.

## Design Notes

- **Total experiments**: 476 (full) / 30 (short)
- **Tier breakdown**: 
  - Tier 1: 64 experiments
  - Tier 2: 36 experiments  
  - Tier 3: 144 experiments
  - Tier 4: 48 experiments
  - Tier 5: 48 experiments
  - Tier 6: 32 experiments (NEW - regression focus)
- **Focus**: Ultra-sparse signals (2-20 features across all scenarios)
- **Response types**: Binary classification (Tiers 1-5) + Continuous regression (Tier 6)
- **Data generation**: Seeded (DATASET_SEED=42) for reproducibility
- **Evaluation**: Fixed outlier detection thresholds [2.0, 2.5, 3.0]

## Limitations

- Experimental design does not account for stochastic effects throughout the algorithm and regression evaluation, as seeding is not fully supported for every included statistical component.
- Artificial dataset generation is properly seeded for reproducibility.
- Tier 6 introduces continuous regression scenarios while Tiers 1-5 focus on binary classification.