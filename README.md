# Feature Selection via Bayesian Gaussian Mixture Approximation

This project implements an algorithm for feature selection in regression/classification with many features (often \( p \gg n \)).  
The algorithm finds all sparse solutions that fit the data well, using a Bayesian approach and approximating the posterior distribution with a mixture of Gaussians.

---

## Key Features

- **Multisolution regression:** Finds all sparse solutions (\( ||\beta||_{l_0} \leq \delta \)) with residuals below a threshold.
- **Easily pluggable prior:** Uses a prior class with a simple `.log_prob(z)` interface; default is spike-and-slab, but any prior class can be substituted.
- **Bayesian inference:** Computes the evidence lower bound (ELBO) and fits a flexible posterior.
- **Gaussian mixture modeling:** Approximates the posterior with a learnable mixture of diagonal Gaussians.
- **Stochastic optimization:** PyTorch-based automatic differentiation and Adam optimizer for all mixture parameters.
- **Efficient reparametrization:** Implements the correct gradient flow for mixture models with reparameterization.
- **Visualization:** Interactive plots with Plotly and statistical plots with Seaborn.
- **Modular structure:** All modeling, inference, and utilities are separated for clarity and extensibility.

---

## Project Structure

```
feature_selection/
├── feature_selection/    # Core algorithm, modeling, inference, utils, diagnostics
├── data/                # Example/synthetic datasets
├── scripts/             # Entry points for generating data, running experiments
├── tests/               # Unit tests
├── notebooks/           # Interactive demos and visualizations
├── README.md
├── requirements.txt
└── setup.py
```

---

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Generate synthetic dataset:**
    ```bash
    python scripts/generate_artificial_dataset.py
    ```

3. **Run an experiment (fit the mixture):**
    ```bash
    python scripts/run_experiment.py
    ```

4. **Plot results:**
    ```bash
    python scripts/plot_results.py
    ```

5. **Explore interactively:**
    Open and run `notebooks/exploratory.ipynb` in Jupyter.

---

## Usage

- **Configuration:** Edit `feature_selection/config.py` for hyperparameters.
- **Modeling:** All priors must implement a `.log_prob(z)` method for easy swapping (see `feature_selection/models.py`).
- **Algorithm:** Core logic in `feature_selection/models.py`, `inference.py`, and `optimizer.py`.
- **Diagnostics:** Use `feature_selection/diagnostics.py` and plotting scripts for visual analysis.

---

## Requirements

See [`requirements.txt`](requirements.txt) for details.  
Main dependencies: numpy, scipy, pandas, torch, plotly, seaborn, scikit-learn, jupyter.

---

## References

- [Spike-and-slab priors](https://en.wikipedia.org/wiki/Spike-and-slab_variable_selection)
- [Variational inference and ELBO](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
- [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html)
- [PyTorch](https://pytorch.org/)
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org/)

---
