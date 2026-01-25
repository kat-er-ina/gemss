"""
Diagnostics and plotting for Bayesian feature selection.
"""

from IPython.display import display, Markdown
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def get_elbo_plot(
    history: Dict[str, List[float]],
) -> go.Figure:
    """
    Get ELBO progress over optimization iterations (Plotly Figure).

    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing 'elbo' key with ELBO values per iteration.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the ELBO progress.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["elbo"], mode="lines", name="ELBO"))
    fig.update_layout(
        title="ELBO Progress", xaxis_title="Iteration", yaxis_title="ELBO"
    )
    return fig


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
    fig = get_elbo_plot(history)
    fig.show(config={"displayModeBar": False})
    return


def get_mu_plot(
    history: Dict[str, np.ndarray],
    component: int = 0,
    original_feature_names_mapping: Dict[str, str] = None,
) -> go.Figure:
    """
    Get trajectory of mixture means for a given component (Plotly Figure).

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
    go.Figure
        Plotly Figure object representing the mu trajectory.
    """
    arr = np.array(history["mu"])  # shape [n_iter, n_components, n_features]
    mu_traj = arr[:, component, :]
    fig = go.Figure()
    longest_name_len = 0
    for f in range(mu_traj.shape[1]):
        name = (
            original_feature_names_mapping.get(f"feature_{f}", f"feature_{f}")
            if original_feature_names_mapping is not None
            else f"feature_{f}"
        )
        longest_name_len = max(longest_name_len, len(name))
        fig.add_trace(
            go.Scatter(
                y=mu_traj[:, f],
                mode="lines",
                name=name,
                hovertemplate=f"<b>{name}</b><br>Iteration=%{{x}}<br>μ=%{{y}}<extra></extra>",
            )
        )
    # Dynamic width scales with longest label and number of features
    base_width = 650
    width = base_width + longest_name_len * 8 + mu_traj.shape[1] * 6
    width = int(min(width, 1800))
    fig.update_layout(
        title=f"Mu trajectory, Component {component}",
        xaxis_title="Iteration",
        yaxis_title="Mu value",
        width=width,
        height=600,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


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
    fig = get_mu_plot(
        history,
        component=component,
        original_feature_names_mapping=original_feature_names_mapping,
    )
    fig.show(config={"displayModeBar": False})
    return


def get_alpha_plot(
    history: Dict[str, np.ndarray],
) -> go.Figure:
    """
    Get mixture weights (alpha) progress over iterations (Plotly Figure).

    Parameters
    ----------
    history : Dict[str, np.ndarray]
        Dictionary containing 'alpha' key with weights per iteration.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the alpha progress.
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
        title="Alpha progress",
        xaxis_title="Iteration",
        yaxis_title="Component weight",
    )
    return fig


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
    fig = get_alpha_plot(history)
    fig.show(config={"displayModeBar": False})
    return


def get_correlation_with_response_plot(
    df: pd.DataFrame,
    y: pd.Series,
    support_features: List[str],
) -> go.Figure:
    """
    Get bar plot of features' correlation with binary response (Plotly Figure).

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
    go.Figure
        Plotly Figure object representing the correlation with response.
    """
    correlation_with_response = df.corrwith(y, method="kendall").sort_values(
        ascending=False
    )

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
    fig.update_layout(
        width=200 + df.shape[1] * 15,
        height=500,
        showlegend=False,
    )
    return fig


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
    fig = get_correlation_with_response_plot(df, y, support_features)
    fig.show(config={"displayModeBar": False})
    return


def get_correlation_matrix_plot(
    df: pd.DataFrame,
    width: Optional[int],
    height: Optional[int],
) -> go.Figure:
    """
    Get correlation matrix heatmap (Plotly Figure).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature data.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the correlation matrix heatmap.
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
    return fig


def show_correlation_matrix(
    df: pd.DataFrame,
    width: Optional[int] = None,
    height: Optional[int] = None,
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
    fig = get_correlation_matrix_plot(df, width, height)
    fig.show(config={"displayModeBar": False})
    return


def get_label_histogram_plot(
    y: np.ndarray,
    nbins: Optional[int] = 10,
) -> go.Figure:
    """
    Get histogram of continuous labels y using Plotly (Plotly Figure).

    Parameters
    ----------
    y : array-like
        Continuous labels.
    nbins : int, optional
        Number of bins for the histogram. If y has at most 10 unique values, the number of unique
        values is used regardless of this parameter. Otherwise, default is 10.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the histogram of labels.
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
    return fig


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
    fig = get_label_histogram_plot(y, nbins)
    fig.show(config={"displayModeBar": False})
    return


def get_label_piechart(
    y: np.ndarray,
) -> go.Figure:
    """
    Get pie chart of label distribution using Plotly (Plotly Figure).

    Parameters
    ----------
    y : array-like
        Labels.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the pie chart of labels.
    """
    unique, counts = np.unique(y, return_counts=True)
    fig = px.pie(
        names=unique,
        values=counts,
        title="Label distribution",
        width=450,
        height=300,
    )
    return fig


def get_features_in_components_plot(
    solutions: Dict[str, List[str]],
    features_to_show: List[str] = None,
) -> go.Figure:
    """
    Get heatmap of which features are found in each component (Plotly Figure).

    Parameters
    ----------
    solutions : Dict[str, List[str]]
        Dictionary where each key is a component name and each value is a list of features
        in that component.
    features_to_show : List[str], optional
        List of features to highlight in the heatmap. If None, only features in the provided
        DataFrame are shown.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the heatmap of features in components.
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
    return fig


def show_features_in_components(
    solutions: Dict[str, List[str]],
    features_to_show: List[str] = None,
) -> None:
    """
    Show a heatmap of which features are found in each component.

    Parameters
    ----------
    solutions : Dict[str, List[str]]
        Dictionary where each key is a component name and each value is a list of features
        in that component.
    features_to_show : List[str], optional
        List of features to highlight in the heatmap. If None, only features in the provided
        DataFrame are shown.

    Returns
    -------
    None
    """
    fig = get_features_in_components_plot(solutions, features_to_show)
    fig.show(config={"displayModeBar": False})
    return


def get_compare_parameters_plot(
    parameters: Dict[str, Any],
    final_mu: np.ndarray,
) -> go.Figure:
    """
    Get comparison plot of learned mixture means and weights to true generating parameters
    (Plotly Figure).

    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary containing true generating parameters with the key
        'full_weights' (list of lists of full weight vectors for each true solution).
    final_mu : np.ndarray
        Learned mixture means, shape (n_components, n_features).

    Returns
    -------
    go.Figure
        Plotly Figure object representing the comparison of learned and true parameters.
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
    return fig


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
    fig = get_compare_parameters_plot(parameters, final_mu)
    fig.show(config={"displayModeBar": False})
    return


def get_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
) -> go.Figure:
    """
    Get confusion matrix heatmap using Plotly (Plotly Figure).

    Parameters
    ----------
    confusion_matrix : array-like, shape (2, 2)
        Confusion matrix to display.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the confusion matrix.
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
    return fig


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
    fig = get_confusion_matrix_plot(confusion_matrix)
    fig.show(config={"displayModeBar": False})
    return


def get_predicted_vs_actual_response_plot(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> go.Figure:
    """
    Get scatter plot of predicted vs actual response using Plotly (Plotly Figure).

    Parameters
    ----------
    y : np.ndarray
        Actual response values.
    y_pred : np.ndarray
        Predicted response values.

    Returns
    -------
    go.Figure
        Plotly Figure object representing the predicted vs actual response.
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
    return fig


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
    fig = get_predicted_vs_actual_response_plot(y, y_pred)
    fig.show(config={"displayModeBar": False})
    return


def get_final_alphas_plot(
    history: Dict[str, List[Any]],
    show_bar_plot: bool = True,
    show_pie_chart: bool = True,
) -> List[go.Figure]:
    """
    Get final mixture weights as bar plot and/or pie chart using Plotly (Plotly Figures).

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Dictionary containing 'alpha' key with weights per iteration.
    show_bar_plot : bool, optional
        Whether to create the bar plot. Default is True.
    show_pie_chart : bool, optional
        Whether to create the pie chart. Default is True.

    Returns
    -------
    List[go.Figure]
        List of Plotly Figure objects representing the final mixture weights.
    """
    alphas = history["alpha"]
    final_alphas = alphas[:][-1]
    figures = []

    if show_bar_plot:
        bar_fig = px.bar(
            x=[f"Component {i}" for i in range(len(final_alphas))],
            y=final_alphas,
            labels={"x": "Component", "y": "Weight"},
            title="Absolute weights of components <br>in the final mixture",
            width=500,
            height=400,
        )
        figures.append(bar_fig)

    if show_pie_chart:
        pie_fig = px.pie(
            names=[f"Component {i}" for i in range(len(final_alphas))],
            values=final_alphas,
            title="Relative weights of candidate solutions <br>in the final mixture",
            width=500,
            height=400,
        )
        figures.append(pie_fig)
    return figures


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
    figures = get_final_alphas_plot(history, show_bar_plot, show_pie_chart)
    for fig in figures:
        fig.show(config={"displayModeBar": False})
    return


def get_subsampled_history(
    history: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    """
    Subsample the history dictionary for plotting purposes.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Original history dictionary.

    Returns
    -------
    Dict[str, List[Any]]
        Subsampled history dictionary.
    """
    every_nth_iteration = 20
    if (len(history) >= 2000) and (len(history) * len(history["mu"][0])) > 1e6:
        every_nth_iteration = 50
    if (len(history) >= 4000) and (len(history) * len(history["mu"][0])) > 1e6:
        every_nth_iteration = 100
    if len(history) >= 8000:
        every_nth_iteration = 200
    if len(history) >= 15000:
        every_nth_iteration = 500

    subsampled_history = {
        key: values[::every_nth_iteration] if isinstance(values, list) else values
        for key, values in history.items()
    }
    return subsampled_history


def get_algorithm_progress_plots(
    history: Dict[str, List[Any]],
    elbo: bool = True,
    mu: bool = True,
    alpha: bool = True,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
    subsample_history_for_plotting: bool = False,
) -> Dict[str, go.Figure]:
    """
    Get the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights. This function uses Plotly for interactive visualizations.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Dictionary containing optimization history with keys 'elbo', 'mu', and 'alpha'
        (fewer keys allowed if corresponding plots are disabled).
        'mu' should have shape [n_iterations, n_components, n_features].
        This is the output of the `optimize` method of `BayesianFeatureSelector`.
    elbo : bool, optional
        Whether to plot the ELBO progress. Default is True.
    mu : bool, optional
        Whether to plot the mixture means (mu) trajectory. Default is True.
    alpha : bool, optional
        Whether to plot the mixture weights (alpha) progress. Default is True.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the plots will use the original feature names where applicable.
        Default is None.
    subsample_history_for_plotting: bool, optional
        If True, plot only every N-th iteration in order to save resources during plotting.

    Returns
    -------
    Dict[str, go.Figure]
        Dictionary of Plotly Figure objects for each requested plot type.

    Notes
    -----
    This function displays markdown output and plots as side effects.
    The function requires the corresponding keys in history for each plot type requested.
    """
    if subsample_history_for_plotting:
        history_to_plot = get_subsampled_history(history)
    else:
        history_to_plot = history

    figures = {}

    if alpha:
        figures["alpha"] = get_alpha_plot(history_to_plot)

    if elbo:
        figures["elbo"] = get_elbo_plot(history_to_plot)

    if mu:
        n_components = len(history["mu"][0])
        for k in range(n_components):
            fig_mu = get_mu_plot(
                history_to_plot,
                component=k,
                original_feature_names_mapping=original_feature_names_mapping,
            )
            figures[f"mu_{k}"] = fig_mu
    return figures


def show_algorithm_progress(
    history: Dict[str, List[Any]],
    plot_elbo_progress: bool = True,
    plot_mu_progress: bool = True,
    plot_alpha_progress: bool = True,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
    subsample_history_for_plotting: bool = False,
) -> None:
    """
    Show the progress of the algorithm by plotting the evolution of
    ELBO, mixture means, and weights. This function uses Plotly for interactive visualizations.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        Dictionary containing optimization history with keys 'elbo', 'mu', and 'alpha'
        (fewer keys allowed if corresponding plots are disabled).
        'mu' should have shape [n_iterations, n_components, n_features].
        This is the output of the `optimize` method of `BayesianFeatureSelector`.
    plot_elbo_progress : bool, optional
        Whether to plot the ELBO progress. Default is True.
    plot_mu_progress : bool, optional
        Whether to plot the mixture means (mu) trajectory. Default is True.
    plot_alpha_progress : bool, optional
        Whether to plot the mixture weights (alpha) progress. Default is True.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        A mapping from internal feature names (e.g., 'feature_0') to original feature names.
        If provided, the plots will use the original feature names where applicable.
        Default is None.
    subsample_history_for_plotting: bool, optional
        If True, plot only every N-th iteration in order to save resources during plotting.

    Returns
    -------
    None
    """
    figures = get_algorithm_progress_plots(
        history,
        elbo=plot_elbo_progress,
        mu=plot_mu_progress,
        alpha=plot_alpha_progress,
        original_feature_names_mapping=original_feature_names_mapping,
        subsample_history_for_plotting=subsample_history_for_plotting,
    )

    # Display plots in specific order: elbo first, then mu_1, mu_2, ..., mu_k, then alpha
    if "elbo" in figures:
        figures["elbo"].show(config={"displayModeBar": False})

    mu_keys = sorted(
        [k for k in figures.keys() if k.startswith("mu_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for mu_key in mu_keys:
        figures[mu_key].show(config={"displayModeBar": False})

    if "alpha" in figures:
        figures["alpha"].show(config={"displayModeBar": False})
    return
