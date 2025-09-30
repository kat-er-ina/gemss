# Feature Selection via Bayesian Gaussian Mixture Approximation

This project implements a feature selection algorithm for classification and regression problems with many features (often p ≫ n).  
The algorithm finds all sparse solutions that fit the data well, using a Bayesian approach and approximating the posterior distribution with a mixture of Gaussians.

## Key Features

- **Multisolution regression:** Finds all feature subsets (sparse solutions) with residuals below a threshold.
- **Bayesian inference:** Uses spike-and-slab prior and computes the evidence lower bound (ELBO).
- **Gaussian mixture modeling:** Approximates the posterior with a learnable mixture model.
- **Stochastic optimization:** Optimizes mixture parameters using PyTorch's automatic differentiation and Adam optimizer.
- **Visualization:** Interactive plots with Plotly and statistical plots with Seaborn.

## Project Structure

```
feature_selection/
├── feature_selection/    # Core algorithm implementation
├── data/                # Example/synthetic datasets
├── scripts/             # Entry points for running experiments
├── tests/               # Unit tests
├── notebooks/           # Interactive demos
├── README.md
├── requirements.txt
└── setup.py
```

## Getting Started

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run an experiment:**
    ```bash
    python scripts/run_experiment.py
    ```

3. **Plot results:**
    ```bash
    python scripts/plot_results.py
    ```

4. **Explore interactively:**
    Open and run `notebooks/exploratory.ipynb` in Jupyter.

## Requirements

See [`requirements.txt`](requirements.txt) for details.

## Usage

- **Configuration:** Edit `feature_selection/config.py` for hyperparameters.
- **Algorithm:** Core logic in `feature_selection/models.py`, `optimizer.py`, and `inference.py`.
- **Diagnostics:** Use `feature_selection/diagnostics.py` and plotting scripts for visual analysis.

## References

- [Spike-and-slab priors](https://en.wikipedia.org/wiki/Spike-and-slab_variable_selection)
- [Variational inference and ELBO](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
- [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/mixture.html)
- [PyTorch](https://pytorch.org/)
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org/)
