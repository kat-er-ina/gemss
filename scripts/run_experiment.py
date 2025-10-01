"""
Run Bayesian Gaussian Mixture Feature Selection algorithm on an artificial dataset.

Usage:
    python scripts/run_experiment.py

This script loads a synthetic dataset, runs the variational mixture model optimizer, and saves or plots diagnostics.
"""

import os
import pandas as pd
import numpy as np
import torch

from feature_selection.inference import BayesianFeatureSelector
from feature_selection.diagnostics import plot_elbo, plot_mu, plot_alpha

# ---- Config ----
DATA_PATH = "data/artificial_mixture_dataset.csv"
N_COMPONENTS = 3  # Number of mixture components (solutions)
N_ITER = 10000  # Number of optimization iterations
BATCH_SIZE = 16  # Batch size for SGD
LEARNING_RATE = 2e-3  # Adam learning rate
SPARSITY = 2  # Number of nonzero entries per solution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load Data ----
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\nPlease generate the dataset first."
    )

df = pd.read_csv(DATA_PATH, index_col=0)
X = df.values  # shape: [n_samples, n_features]
y_path = DATA_PATH.replace("dataset", "labels")
if os.path.exists(y_path):
    y = pd.read_csv(y_path, index_col=0).values.squeeze()
else:
    # If no labels, use zeros as placeholder targets (for regression)
    y = np.zeros(X.shape[0])

n_samples, n_features = X.shape

# ---- Instantiate and Run Algorithm ----
selector = BayesianFeatureSelector(
    n_features=n_features,
    n_components=N_COMPONENTS,
    X=X,
    y=y,
    sparsity=SPARSITY,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    n_iter=N_ITER,
    device=DEVICE,
)

print(
    f"Running Bayesian Feature Selector on {DATA_PATH} with {N_COMPONENTS} mixture components..."
)
history = selector.optimize()

# ---- Diagnostics & Plots ----
print("Plotting ELBO progress...")
plot_elbo(history)

print("Plotting mixture means trajectory for each component...")
for k in range(N_COMPONENTS):
    plot_mu(history, component=k)

print("Plotting mixture weights (alpha)...")
plot_alpha(history)

# ---- Optionally save history ----
import pickle

with open("data/optimization_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Experiment complete. History saved to data/optimization_history.pkl")
