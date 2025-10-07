# Bayesian Sparse Feature Selection

This repository implements Bayesian sparse feature selection using variational inference with mixture models. The main objective is to recover all sparse feature subsets (supports) that explain the response in high-dimensional regression or classification tasks.

---

## Motivation

In many real-world problems, e.g. in life sciences, datasets with far more features than samples are common because collecting new data points is costly or impractical. In these situations, there are often several distinct, sparse combinations of features that can explain the observed outcomes, each corresponding to a different underlying mechanism or hypothesis. Moreover, in many cases, the quality of a combination of predictors can be assessed only ex-post by utilizing advanced domain knowledge.

Traditional feature selection methods typically identify only a single solution to a classification or regression problem, overlooking the ambiguity and the potential for multiple valid interpretations. This project addresses that gap by providing a Bayesian framework that systematically recovers all plausible sparse solutions, enabling a more complete understanding of the data and supporting the exploration and comparison of alternative explanatory hypotheses.

---

## Features

- **Multiple sparse solutions:** Identifies all distinct sparse supports that explain the data.
- **Flexible priors:** Includes spike-and-slab, structured spike-and-slab (SSS), and Student-t priors.
- **Variational inference:** Approximates the posterior with a mixture of diagonal Gaussians.
- **Diversity regularization:** Jaccard penalty to encourage component diversity.
- **Synthetic data generation:** Easily generate datasets with known supports.
- **Batch/script support:** Run headless experiments or parameter sweeps.
- **Rich output:** Summaries, solution tables, and diagnostics are saved as `.txt` files.

---

## Repository Structure

```
feature_selection/                      # Core package: models, inference, config, data generation, result postprocessing
  config.py                             # Loads config from 3 JSONs and exposes as Python variables
  generate_artificial_dataset.py        # Synthetic dataset generator
  inference.py                          # Main variational inference logic
  result_postprocessing.py              # Solution extraction and diagnostics

notebooks/
  demo.ipynb                            # Interactive demo notebook

scripts/
  run_experiment.py                     # Run a single experiment (headless, summary output)
  generate_artificial_dataset.py        # Generate new datasets

results/                                # Experiment output summaries (created automatically)

generated_dataset_parameters.json       # Dataset config (overwritten by sweep scripts)
algorithm_settings.json                 # Algorithm config (overwritten by sweep scripts)
solution_postprocessing_settings.json   # Solution extraction/postprocessing config (overwritten by sweep scripts or edited manually)

run_sweep.ps1                           # PowerShell sweep script for batch experiments (Windows)

requirements.txt
setup.py
README.md
```

---

## Configuration Files

The workflow uses **three** JSON configuration files (always expected in the parent directory of the repo root):

1. **generated_dataset_parameters.json**  
   Dataset generation parameters (number of samples, features, supports, sparsity, noise, class balance, random seed, etc.)

2. **algorithm_settings.json**  
   Algorithm and inference parameters (number of mixture components, prior type, variances, regularization, optimization settings, etc.)

3. **solution_postprocessing_settings.json**  
   Parameters for solution extraction/postprocessing (e.g., thresholds, filtering, merging, and other post-hoc analysis options).

The config module (`feature_selection/config.py`) loads all three JSON files and exposes every parameter as a Python variable for use throughout the codebase.

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

### Single Experiment

1. **Configure parameters:**
   - Edit `generated_dataset_parameters.json`, `algorithm_settings.json`, and `solution_postprocessing_settings.json` manually,
     **or** let the sweep script generate them.

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

- **Notebook Demo:**
   ```bash
   jupyter notebook notebooks/demo.ipynb
   ```
   - Walks through data generation, fitting, and diagnostics interactively.

---

## Working with Configuration in Code

All configuration is handled via the three JSON files in the parent directory:

- `generated_dataset_parameters.json`
- `algorithm_settings.json`
- `solution_postprocessing_settings.json`

The module `feature_selection/config.py` loads all three and exposes every parameter as a Python variable:

```python
import feature_selection.config as C
print(C.NSAMPLES, C.N_COMPONENTS, C.PRIOR_TYPE)
print(C.MIN_MU_THRESHOLD, C.POSTPROCESSING_THRESHOLD)  # Example: postprocessing params
```

**Parameter sweeps:**  
The sweep script (`run_sweep.ps1`) will automatically generate and overwrite these files for each run.

---

## Output

- Results are saved in the `results/` directory as text files.
- Filenames include timestamps and all key parameter values.
- Each file contains:
  - All run parameters
  - True and discovered supports
  - Solution tables for each mixture component
  - Diagnostic information

---

## Customization & Extending

- **Add new priors:** Implement your prior in `feature_selection/models.py` and update inference logic.
- **Add diagnostics:** Extend `feature_selection/result_postprocessing.py` or the notebook.
- **Change or add sweeps:** Edit `run_sweep.ps1` for custom batch runs.
- **Add solution postprocessing logic:** Edit `feature_selection/result_postprocessing.py` and/or update the postprocessing JSON.

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
