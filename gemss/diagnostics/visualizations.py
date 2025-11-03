"""
Diagnostics and plotting for Bayesian feature selection.
"""

from IPython.display import display, Markdown
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def plot_elbo(
    history: Dict[str, List[float]],
) -> None:
    """
    Plot ELBO progress over optimization iterations (Plotly).

    Parameters
    ----------
    history : Dict[str, List[float]]
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
    fig.show(config={"displayModeBar": False})
    return


def plot_mu(
    history: Dict[str, np.ndarray],
    component: int = 0,
    original_feature_names_mapping: Dict[str, str] = None,
) -> None:
    """
    Plot trajectory of mixture means for a given component (Plotly).

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing 'mu' key with mean values per iteration.
    component : int, optional
        Index of mixture component to plot.
    original_feature_names_mapping : Dict[str, str], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the plots will use the original feature names.

    Returns
    -------
    None
    """
    arr = np.array(history["mu"])  # shape [n_iter, n_components, n_features]
    mu_traj = arr[:, component, :]
    fig = go.Figure()
    for f in range(mu_traj.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=mu_traj[:, f],
                mode="lines",
                name=(
                    original_feature_names_mapping.get(f"feature_{f}", f"feature_{f}")
                    if original_feature_names_mapping is not None
                    else f"feature_{f}"
                ),
            )
        )
    fig.update_layout(
        title=f"Mu trajectory, Component {component}",
        xaxis_title="Iteration",
        yaxis_title="Mu value",
    )
    fig.show(config={"displayModeBar": False})
    return


def plot_alpha(
    history: Dict[str, np.ndarray],
) -> None:
    """
    Plot mixture weights (alpha) progress over iterations (Plotly).

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing 'alpha' key with weights per iteration.

    Returns
    -------
    None
    """
    arr = np.array(history["alpha"])  # shape [n_iter, n_components]
    fig = go.Figure()
    for k in range(arr.shape[1]):
        fig.add_trace(
            go.Scatter(
                y=arr[:, k],
                mode="lines",
                name=f"alpha_{k}",
            ),
        )
    fig.update_layout(
        title="Alpha Progress",
        xaxis_title="Iteration",
        yaxis_title="Mixture Weight",
    )
    fig.show(config={"displayModeBar": False})
    return


def show_correlations_with_response(
    df: pd.DataFrame,
    y: pd.Series,
    support_features: List[str],
) -> None:
    """
    Show bar plot of features' correlation with binary response.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature data.
    y : pd.Series
        Binary response variable.
    support_features : List[str]
        List of features to highlight in the plot.

    Returns
    -------
    None
    """
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
    return


def show_correlation_matrix(
    df: pd.DataFrame,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """
    Show correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature data.
    width : int, optional
        Width of the plot.
    height : int, optional
        Height of the plot.

    Returns
    -------
    None
    """
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
    return


def show_label_histogram(
    y: np.ndarray,
    nbins: Optional[int] = 10,
) -> None:
    """
    Show histogram of continuous labels y using Plotly.

    Parameters
    ----------
    y : array-like
        Continuous labels.
    nbins : int, optional
        Number of bins for the histogram. If y has at most 10 unique values, the number of unique
        values is used regardless of this parameter. Otherwise, default is 10.

    Returns
    -------
    None
    """
    # if the label is binary or discrete with at most 10 unique values, take only those few bins
    y_unique = np.unique(y).tolist()
    y_unique.sort()
    nunique = len(y_unique)
    if nunique <= 10:
        nbins = nunique

    hist_data = np.histogram(y, bins=nbins)
    fig = go.Figure(
        data=[
            go.Bar(
                x=y_unique,
                y=hist_data[0],
                width=np.diff(hist_data[1]),
                marker_color="blue",
            )
        ],
    )
    fig.update_layout(
        title="Distribution of labels",
        xaxis_title="Value",
        yaxis_title="Count",
        width=450,
        height=300,
    )
    fig.show(config={"displayModeBar": False})
    return


def show_features_in_components(
    solutions: Dict[str, List[str]],
    features_to_show: List[str] = None,
) -> None:
    """
    Show a heatmap of which features are found in each component.

    Parameters
    ----------
    df_solutions : Dict[str, List[str]]
        Dictionary where each key is a component name and each value is a list of features
        in that component.
    features_to_show : List[str], optional
        List of features to highlight in the heatmap. If None, only features in the provided
        DataFrame are shown.

    Returns
    -------
    None
    """
    df_solutions = pd.DataFrame.from_dict(solutions, orient="index").T

    if features_to_show is None:
        features_to_show = df_solutions.columns.tolist()

    heatmap_data = pd.DataFrame(
        0,
        index=df_solutions.columns,
        columns=sorted(features_to_show),
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
        width=200 + 40 * len(heatmap_data.columns),
        showlegend=False,
    )
    fig.show(config={"displayModeBar": False})
    return


def compare_parameters(
    parameters: Dict[str, Any],
    final_mu: np.ndarray,
) -> None:
    """
    Compare learned mixture means and weights to true generating parameters.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing true generating parameters with the key
        'full_weights' (list of lists of full weight vectors for each true solution).
    final_mu : np.ndarray
        Learned mixture means, shape (n_components, n_features).

    Returns
    -------
    None
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    # Add traces for learned mixture means for each component
    n_features = len(final_mu[0])
    n_components = len(final_mu)
    for k in range(n_components):
        # Learned means
        color = colors[k % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_features),
                y=final_mu[k],
                mode="markers",
                name=f"Learned μ {k}",
                marker=dict(size=8, symbol="circle", color=color),
                showlegend=True,
            )
        )

    # Add traces for true (generating) solutions
    n_solutions = len(parameters["full_weights"])
    for k in range(n_solutions):
        # True means
        color = colors[k % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_features),
                y=parameters["full_weights"][k],
                mode="markers",
                name=f"True μ {k}",
                marker=dict(size=8, symbol="x", color=color),
                showlegend=True,
            )
        )

    # Update layout
    fig.update_layout(
        title="Comparison: Learned vs True Mixture Means",
        xaxis_title="Feature Index",
        yaxis_title="Mean Value",
        width=800,
        height=500,
        template="plotly_white",
    )

    # Update x-axis to show integer ticks
    fig.update_xaxes(dtick=1)

    fig.show(config={"displayModeBar": False})
    return


def show_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> None:
    """
    Show confusion matrix using Plotly.

    Parameters
    ----------
    confusion_matrix : array-like, shape (2, 2)
        Confusion matrix to display.

    Returns
    -------
    None
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix", width=350, height=350, showlegend=False)
    fig.show(config={"displayModeBar": False})
    return


def show_predicted_vs_actual_response(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Show scatter plot of predicted vs actual response using Plotly.

    Parameters
    ----------
    y : np.ndarray
        Actual response values.
    y_pred : np.ndarray
        Predicted response values.

    Returns
    -------
    None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y, y=y_pred, mode="markers", marker_color="blue"))
    fig.add_trace(
        go.Scatter.line(
            x=[y.min(), y.max()],
            y=[y.min(), y.max()],
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title="Predicted vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        width=400,
        height=300,
        showlegend=False,
    )
    fig.show(config={"displayModeBar": False})
    return


def show_final_alphas(
    history: Dict[str, List[Any]],
    show_bar_plot: bool = True,
    show_pie_chart: bool = True,
) -> None:
    """
    Show final mixture weights as bar plot and/or pie chart using Plotly.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Dictionary containing 'alpha' key with weights per iteration.
    show_bar_plot : bool, optional
        Whether to show the bar plot. Default is True.
    show_pie_chart : bool, optional
        Whether to show the pie chart. Default is True.

    Returns
    -------
    None
    """
    alphas = history["alpha"]
    final_alphas = alphas[:][-1]
    display(Markdown("## Final mixture weights"))

    if show_bar_plot:
        px.bar(
            x=[f"Component {i}" for i in range(len(final_alphas))],
            y=final_alphas,
            labels={"x": "Component", "y": "Weight"},
            title="Absolute weights of candidate solutions <br>in the final mixture",
            width=500,
            height=400,
        ).show(config={"displayModeBar": False})

    # create a pie chart
    if show_pie_chart:
        px.pie(
            names=[f"Component {i}" for i in range(len(final_alphas))],
            values=final_alphas,
            title="Relative weights of candidate solutions <br>in the final mixture",
            width=500,
            height=400,
        ).show(config={"displayModeBar": False})
    return
