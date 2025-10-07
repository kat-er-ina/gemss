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

import feature_selection.config as C
from feature_selection.generate_artificial_dataset import generate_artificial_dataset
from feature_selection.inference import BayesianFeatureSelector
from feature_selection.utils import recover_solutions

# ---- Generate Artificial Dataset ----
print("Generating dataset with:")
print(f" - {C.NSAMPLES} samples,")
print(f" - {C.NFEATURES} features,")
print(
    f" - {C.NSOLUTIONS} original solutions, each with {C.SPARSITY} supporting vectors."
)

df, y, generating_solutions, parameters = generate_artificial_dataset(
    n_samples=C.NSAMPLES,
    n_features=C.NFEATURES,
    n_solutions=C.NSOLUTIONS,
    sparsity=C.SPARSITY,
    noise_data_std=C.NOISE_STD,
    binarize=C.BINARIZE,
    binary_response_ratio=C.BINARY_RESPONSE_RATIO,
    random_seed=C.RANDOM_SEED,
    save_to_csv=False,
    print_data_overview=False,
)

support_indices = parameters["support_indices"].sum()
true_support_features = [f"feature_{i}" for i in set(support_indices)]

# ---- Run Bayesian Feature Selector ----
print("\nRunning Bayesian Feature Selector...")

selector = BayesianFeatureSelector(
    n_features=C.NFEATURES,
    n_components=C.N_COMPONENTS,
    X=df.values,
    y=y,
    prior=C.PRIOR_TYPE,
    sss_sparsity=C.SPARSITY,
    var_slab=C.VAR_SLAB,
    var_spike=C.VAR_SPIKE,
    weight_slab=C.WEIGHT_SLAB,
    weight_spike=C.WEIGHT_SPIKE,
    student_df=C.STUDENT_DF,
    student_scale=C.STUDENT_SCALE,
    lr=C.LEARNING_RATE,
    batch_size=C.BATCH_SIZE,
    n_iter=C.N_ITER,
)

history = selector.optimize(
    regularize=C.IS_REGULARIZED,
    lambda_jaccard=C.LAMBDA_JACCARD,
    verbose=False,
)
print("Optimization finished.")

solutions, final_parameters, full_nonzero_solutions = recover_solutions(
    search_history=history,
    desired_sparsity=C.SPARSITY,
    min_mu_threshold=C.MIN_MU_THRESHOLD,
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
params = C.as_dict()
for k, v in params.items():
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

lines.append(f"\n## Solutions found (top {C.SPARSITY} features for each component)\n")
solutions_df = pd.DataFrame.from_dict(solutions, orient="index").T
lines.append(solutions_df.to_string())

lines.append("\n## Full solutions\n")
lines.append(
    " - all features with mu greater than the minimal threshold in last iterations"
)
lines.append(f" - minimal mu threshold: {C.MIN_MU_THRESHOLD}")
for component, df in full_nonzero_solutions.items():
    lines.append(f"\n### {component.upper()} ({df.shape[0]} features):\n")
    for i, row in df.iterrows():
        lines.append(f" - {row['Feature']}: mu = {row['Mu value']:.4f}")
    lines.append("")

# ---- Write output ----
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
parent_dir = Path(os.path.dirname(os.getcwd()))
output_path = parent_dir / "results" / f"experiment_output_{timestamp}.txt"
os.makedirs(parent_dir / "results", exist_ok=True)
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Experiment summary written to {output_path}")
