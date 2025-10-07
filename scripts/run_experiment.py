"""
Run Bayesian Sparse Feature Selection as a script, closely matching the demo notebook.

Assumes:
- generated_dataset_parameters.json and algorithm_settings.json exist in the parent directory
- feature_selection package is installed and available

Saves:
- A text file summarizing the experiment parameters, discovered features, solutions, and final parameters
  in a "results" folder in the parent directory.

"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from feature_selection.generate_artificial_dataset import generate_artificial_dataset
from feature_selection.inference import BayesianFeatureSelector
from feature_selection.utils import recover_solutions

# ---- Load parameters and settings ----
parent_dir = Path(os.path.dirname(os.getcwd()))

with open(parent_dir / "generated_dataset_parameters.json", "r") as f:
    dataset_params = json.load(f)

with open(parent_dir / "algorithm_settings.json", "r") as f:
    algo_settings = json.load(f)

# Dataset parameters
NSAMPLES = dataset_params["NSAMPLES"]
NFEATURES = dataset_params["NFEATURES"]
NSOLUTIONS = dataset_params["NSOLUTIONS"]
SPARSITY = dataset_params["SPARSITY"]
NOISE_STD = dataset_params["NOISE_STD"]
BINARIZE = dataset_params["BINARIZE"]
BINARY_RESPONSE_RATIO = dataset_params["BINARY_RESPONSE_RATIO"]
RANDOM_SEED = dataset_params["RANDOM_SEED"]

# Algorithm settings
# Heuristic: double the number of generating solutions for number of components
# N_COMPONENTS = 2 * NSOLUTIONS
N_COMPONENTS = algo_settings["N_COMPONENTS"]
N_ITER = algo_settings["N_ITER"]
PRIOR_TYPE = algo_settings["PRIOR_TYPE"]
STUDENT_DF = algo_settings["STUDENT_DF"]
STUDENT_SCALE = algo_settings["STUDENT_SCALE"]
VAR_SLAB = algo_settings["VAR_SLAB"]
VAR_SPIKE = algo_settings["VAR_SPIKE"]
WEIGHT_SLAB = algo_settings["WEIGHT_SLAB"]
WEIGHT_SPIKE = algo_settings["WEIGHT_SPIKE"]
IS_REGULARIZED = algo_settings["IS_REGULARIZED"]
LAMBDA_JACCARD = algo_settings["LAMBDA_JACCARD"]
BATCH_SIZE = algo_settings["BATCH_SIZE"]
LEARNING_RATE = algo_settings["LEARNING_RATE"]
# Minimum mu threshold for feature selection in recover_solutions
MIN_MU_THRESHOLD = algo_settings["MIN_MU_THRESHOLD"]

# ---- Generate Artificial Dataset ----
print("Generating dataset with:")
print(f" - {NSAMPLES} samples,")
print(f" - {NFEATURES} features,")
print(f" - {NSOLUTIONS} original solutions, each with {SPARSITY} supporting vectors.")

df, y, generating_solutions, parameters = generate_artificial_dataset(
    n_samples=NSAMPLES,
    n_features=NFEATURES,
    n_solutions=NSOLUTIONS,
    sparsity=SPARSITY,
    noise_data_std=NOISE_STD,
    binarize=BINARIZE,
    binary_response_ratio=BINARY_RESPONSE_RATIO,
    random_seed=RANDOM_SEED,
    save_to_csv=False,
    print_data_overview=False,
)

support_indices = parameters["support_indices"].sum()
true_support_features = [f"feature_{i}" for i in set(support_indices)]

# ---- Run Bayesian Feature Selector ----
print("\nRunning Bayesian Feature Selector...")

selector = BayesianFeatureSelector(
    n_features=NFEATURES,
    n_components=N_COMPONENTS,
    X=df.values,
    y=y,
    prior=PRIOR_TYPE,
    sss_sparsity=SPARSITY,
    var_slab=VAR_SLAB,
    var_spike=VAR_SPIKE,
    weight_slab=WEIGHT_SLAB,
    weight_spike=WEIGHT_SPIKE,
    student_df=STUDENT_DF,
    student_scale=STUDENT_SCALE,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    n_iter=N_ITER,
)

history = selector.optimize(
    regularize=IS_REGULARIZED,
    lambda_jaccard=LAMBDA_JACCARD,
    verbose=False,
)
print("Optimization finished.")

solutions, final_parameters, full_nonzero_solutions = recover_solutions(
    search_history=history,
    desired_sparsity=SPARSITY,
    min_mu_threshold=MIN_MU_THRESHOLD,
    verbose=False,
)

features_found = set().union(*solutions.values())
missing_features = set(true_support_features) - features_found
extra_features = features_found - set(true_support_features)

print("Writing results...")
# ---- Prepare output ----
lines = []
lines.append("# Bayesian Sparse Feature Selection Experiment\n")
lines.append("## Parameters and Settings\n")
lines.append("### Dataset Parameters:")
for k, v in dataset_params.items():
    lines.append(f"- {k}: {v}")
lines.append("\n### Algorithm Settings:")
for k, v in algo_settings.items():
    lines.append(f"- {k}: {v}")

lines.append("\n## Summary of discovered features:\n")
lines.append(f" - {len(true_support_features)} unique true support features:")
lines.append(f"{sorted(true_support_features)}\n")
lines.append(f" - {len(features_found)} discovered features:")
lines.append(f"{sorted(features_found)}\n")
lines.append(f" - {len(missing_features)} missed true support features:")
lines.append(f"{sorted(missing_features)}\n")
lines.append(f" - {len(extra_features)} extra features found (not in true support):")
lines.append(f"{sorted(extra_features)}")

lines.append("\n## Solutions (top features for each component)\n")
solutions_df = pd.DataFrame.from_dict(solutions, orient="index").T
lines.append(solutions_df.to_string())

lines.append("\n## Full solutions\n")
lines.append(
    " - all features with mu greater than the minimal threshold in last iterations"
)
lines.append(f" - minimal mu threshold: {MIN_MU_THRESHOLD}")
for component, df in full_nonzero_solutions.items():
    lines.append(f"\n### {component.upper()} ({df.shape[0]} features):\n")
    for i, row in df.iterrows():
        lines.append(f" - {row['Feature']}: mu = {row['Mu value']:.4f}")
    lines.append("")

# ---- Write output ----
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_path = parent_dir / "results" / f"experiment_output_{timestamp}.txt"
# create results directory if it doesn't exist
os.makedirs(parent_dir / "results", exist_ok=True)
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Experiment summary written to {output_path}")
