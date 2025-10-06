"""
Diagnostics and plotting for Bayesian feature selection.
"""

from IPython.display import display, Markdown
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


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
        title=f"Mu trajectory, Component {component}",
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


def show_correlations_with_response(df, y, support_features):
    correlation_with_response = df.corrwith(y, method="kendall").sort_values(
        ascending=False
    )

    display(Markdown("### Features' Correlation with Binary Response"))
    fig = px.bar(
        correlation_with_response,
        x=correlation_with_response.index,
        y=correlation_with_response.values,
        color=[
            "blue" if f in support_features else "red"
            for f in correlation_with_response.index
        ],
        title="Features' Correlation with Binary Response",
        labels={
            "index": "Feature",
            "y": "Correlation",
        },
    )
    fig.update_layout(width=200 + df.shape[1] * 15, height=500, showlegend=False)
    fig.show(config={"displayModeBar": False})


def show_correlation_matrix(
    df: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
):
    fig = px.imshow(
        df.corr(),
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(
        width=200 + df.shape[1] * 15 if width is None else width,
        height=200 + df.shape[1] * 15 if height is None else height,
    )
    fig.show(config={"displayModeBar": False})


def show_label_histogram(y, nbins=10):
    """
    Show histogram of continuous labels y using Plotly.
    Parameters
    ----------
    y : array-like
        Continuous labels.
    nbins : int, optional
        Number of bins for the histogram.
    """
    hist_data = np.histogram(y, bins=nbins)
    fig = go.Figure(
        data=[
            go.Bar(
                x=hist_data[1][:-1],
                y=hist_data[0],
                width=np.diff(hist_data[1]),
                marker_color="purple",
            )
        ]
    )
    fig.update_layout(
        title="Distribution of continuous labels",
        xaxis_title="Value",
        yaxis_title="Count",
        width=450,
        height=300,
    )
    fig.show()
    return


def show_features_in_components(
    df_solutions: pd.DataFrame,
    features_to_show: List[str] = None,
):
    """
    Show a heatmap of which features are found in each component.

    Parameters
    ----------
    df_solutions : pd.DataFrame
        DataFrame where each column corresponds to a component and contains the features found.
    features_to_show : List[str], optional
        List of features to highlight in the heatmap. If None, only features in the provided
        DataFrame are shown.
    """
    if features_to_show is None:
        features_to_show = df_solutions.columns.tolist()

    heatmap_data = pd.DataFrame(
        0, index=df_solutions.columns, columns=sorted(features_to_show)
    )
    for col in df_solutions.columns:
        for feature in df_solutions[col].dropna():
            heatmap_data.at[col, feature] = 1
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=["white", "blue"],
        labels={"color": "Feature presence"},
        title="Features found in each component",
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        width=200 + 30 * len(heatmap_data.columns),
        showlegend=False,
    )
    fig.show()
    return
