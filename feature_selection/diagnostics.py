"""
Diagnostics and plotting for Bayesian feature selection.
Supports Plotly and Seaborn visualizations.
"""

import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt


def plot_elbo(history):
    """
    Plot ELBO progress over optimization iterations (Plotly).

    Parameters
    ----------
    history : dict
        Dictionary containing 'elbo' key with ELBO values per iteration.

    Returns
    -------
    None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["elbo"], mode="lines", name="ELBO"))
    fig.update_layout(
        title="ELBO Progress", xaxis_title="Iteration", yaxis_title="ELBO"
    )
    fig.show()


def plot_mu(history, component=0):
    """
    Plot trajectory of mixture means for a given component (Plotly).

    Parameters
    ----------
    history : dict
        Dictionary containing 'mu' key with mean values per iteration.
    component : int, optional
        Index of mixture component to plot.

    Returns
    -------
    None
    """
    arr = np.array(history["mu"])  # shape [n_iter, n_components, n_features]
    mu_traj = arr[:, component, :]
    fig = go.Figure()
    for f in range(mu_traj.shape[1]):
        fig.add_trace(go.Scatter(y=mu_traj[:, f], mode="lines", name=f"feature_{f}"))
    fig.update_layout(
        title=f"Mu Trajectory, Component {component+1}",
        xaxis_title="Iteration",
        yaxis_title="Mu value",
    )
    fig.show()


def plot_alpha(history):
    """
    Plot mixture weights (alpha) progress over iterations (Plotly).

    Parameters
    ----------
    history : dict
        Dictionary containing 'alpha' key with weights per iteration.

    Returns
    -------
    None
    """
    arr = np.array(history["alpha"])  # shape [n_iter, n_components]
    fig = go.Figure()
    for k in range(arr.shape[1]):
        fig.add_trace(go.Scatter(y=arr[:, k], mode="lines", name=f"alpha_{k}"))
    fig.update_layout(
        title="Alpha Progress", xaxis_title="Iteration", yaxis_title="Mixture Weight"
    )
    fig.show()
