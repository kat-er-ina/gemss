""" """

from typing import List, Dict, Literal, Optional, Tuple
from IPython.display import display, Markdown
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from gemss.experiment_assessment.experiment_results_analysis import (
    COVERAGE_METRICS,
    DEFAULT_ASI_SI_COMPARISON_THRESHOLDS,
    DEFAULT_F1SCORE_THRESHOLDS,
    DEFAULT_PRECISION_THRESHOLDS,
    DEFAULT_RECALL_THRESHOLDS,
    THRESHOLDS_FOR_METRIC,
    DEFAULT_AGGREGATION_FUNC,
    DEFAULT_METRIC,
)

# Define order and colors for any threshold-based categories
CATEGORY_ORDER = ["Excellent", "Good", "Moderate", "Poor", "Unknown"]
CATEGORY_COLORS = {
    "Excellent": "green",
    "Good": "lightgreen",
    "Moderate": "lightyellow",
    "Poor": "lightcoral",
    "Unknown": "gray",
}


def plot_solution_grouped(
    df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    color_by: str,
    hover_params: List[str],
    group_identifier: Literal["TIER_ID", "CASE_ID", None],
    identifiers_list: Optional[List[str]] = None,
    solution_type: Optional[str] = "all types",
) -> None:
    """
    Plot a metric for a given solution type, grouped by a specified parameter.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    color_by : str
        The parameter to group the lines by (or "None" for no grouping).
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : Optional[str] = "all types"
        The type of solution to plot (e.g., "full", "top", "outlier_STD_2.5").
        Input "all types" to include all solution types.

    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df_plot = df[df[group_identifier].isin(identifiers_list)]
    else:
        df_plot = df

    if solution_type != "all types":
        df_plot = df_plot[df_plot["solution_type"] == solution_type]
    else:
        solution_type = "all"  # for display purposes
    df_plot = df_plot.copy()

    if metric_name not in df.columns:
        print(f"Column '{metric_name}' not found in dataframe.")
        return

    # Create title
    title = f"{metric_name} vs. {x_axis} (Solution: {solution_type})"
    if color_by != "None":
        title += f", grouped by {color_by}"

    # Convert group_by to string for discrete coloring if needed
    if color_by != "None":
        df_plot[color_by] = df_plot[color_by].astype(str)

    # Sort for cleaner line plots
    sort_cols = [x_axis]
    if color_by != "None":
        sort_cols.insert(0, color_by)
    df_plot = df_plot.sort_values(by=sort_cols)

    # Create figure
    fig = go.Figure()

    if color_by != "None":
        # Group by the specified parameter and create separate traces
        unique_groups = df_plot[color_by].unique()
        colors = px.colors.qualitative.Plotly

        for i, group_val in enumerate(unique_groups):
            group_data = df_plot[df_plot[color_by] == group_val]

            # Create hover template
            hovertemplate = (
                f"{x_axis}: %{{x}}<br>"
                f"{metric_name}: %{{y}}<br>"
                f"{color_by}: {group_val}<br><br>"
                + "<br>".join(
                    [
                        f"{param}: %{{customdata[{j}]}}"
                        for j, param in enumerate(hover_params)
                    ]
                )
                + "<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=group_data[x_axis],
                    y=group_data[metric_name].round(3),
                    mode="markers",
                    marker=dict(size=10, opacity=0.7, color=colors[i % len(colors)]),
                    name=str(group_val),
                    hovertemplate=hovertemplate,
                    customdata=group_data[hover_params].round(3).values,
                )
            )
    else:
        # No grouping - single trace
        hovertemplate = (
            f"{x_axis}: %{{x}}<br>"
            f"{metric_name}: %{{y}}<br><br>"
            + "<br>".join(
                [
                    f"{param}: %{{customdata[{j}]}}"
                    for j, param in enumerate(hover_params)
                ]
            )
            + "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=df_plot[x_axis],
                y=df_plot[metric_name],
                mode="markers",
                marker=dict(size=10, opacity=0.7),
                name=solution_type,
                hovertemplate=hovertemplate,
                customdata=df_plot[hover_params].round(3).values,
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        xaxis_title=x_axis,
        yaxis_title=metric_name,
        yaxis_type="log" if "Success_Index" in metric_name else "linear",
        legend_title=color_by if color_by != "None" else "",
    )

    fig.show(config={"displayModeBar": False})
    return


def plot_solution_comparison(
    df: pd.DataFrame,
    solution_types: List[str],
    metric_name: str,
    x_axis: str,
    hover_params: List[str],
    group_identifier: Literal["TIER_ID", "CASE_ID", None],
    identifiers_list: Optional[List[str]] = None,
) -> None:
    """
    Plot a comparison of a metric across different solution types.
    Each solution type is a separate line.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    solution_types : List[str]
        List of solution types to compare (e.g., ["full", "top", "outlier_STD_2.5"]).
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df = df[df[group_identifier].isin(identifiers_list)]
    else:
        df = df

    fig = go.Figure()
    for solution_type in solution_types:
        df_solution = df[df["solution_type"] == solution_type]
        if metric_name not in df.columns:
            print(f"Column '{metric_name}' not found in dataframe.")
            continue

        other_solution_metrics = [
            col for col in df.columns if col in COVERAGE_METRICS and col != metric_name
        ]
        fig.add_trace(
            go.Scatter(
                x=df_solution[x_axis],
                y=df_solution[metric_name].round(3),
                mode="markers",
                marker={"size": 10, "opacity": 0.7},
                name=solution_type,
                hovertemplate=(
                    f"{x_axis}: %{{x}}<br>"
                    f"{metric_name}: %{{y}}<br><br>"
                    + "<br>".join(
                        [
                            f"{param}: %{{customdata[{i}]}}"
                            for i, param in enumerate(hover_params)
                        ]
                    )
                    + "<br>".join(
                        [
                            f"{metric}: %{{customdata[{len(hover_params) + j}]}}"
                            for j, metric in enumerate(other_solution_metrics)
                        ]
                    )
                    + "<extra></extra>"
                ),
                customdata=df[hover_params + other_solution_metrics].round(3).values,
            )
        )

    fig.update_layout(
        title=f"{metric_name} vs. {x_axis}",
        height=600,
        xaxis_title=x_axis,
        yaxis_title=metric_name,
        yaxis_type="log" if "Success_Index" in metric_name else "linear",
        legend_title="Solution Type",
    ).show(config={"displayModeBar": False})
    return


def plot_si_asi_scatter(
    df: pd.DataFrame,
    color_by: str,
    hover_params: List[str],
    group_identifier: Literal["TIER_ID", "CASE_ID", None],
    identifiers_list: Optional[List[str]] = None,
    solution_type: Optional[str] = "all types",
) -> None:
    """
    Plot a scatter plot of Adjusted Success Index (ASI) vs Success Index (SI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results data.
    color_by : str
        The parameter to color the points by (or "None" for no coloring).
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : Optional[str] = "all types"
        The type of solution to plot (e.g., "full", "top", "outlier_STD_2.5").
        Input "all types" to include all solution types.

    Returns
    -------
    None
    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df_plot = df[df[group_identifier].isin(identifiers_list)]
    else:
        df_plot = df
    if solution_type != "all types":
        df_plot = df_plot[df_plot["solution_type"] == solution_type]
    else:
        solution_type = "all"  # for display purposes
    df_plot = df_plot.copy()

    si_col = f"Success_Index"
    asi_col = f"Adjusted_Success_Index"

    if si_col not in df.columns:
        print(f"Metrics not found for {solution_type}")
        return

    # Convert color column to string for categorical coloring
    if color_by != "None":
        df_plot[color_by] = df_plot[color_by].astype(str)

    # Add diagonal line for reference (Ideal: ASI = SI)
    max_val = max(df_plot[si_col].max(), df_plot[asi_col].max()) * 1.1
    fig = px.scatter(
        df_plot,
        x=si_col,
        y=asi_col,
        color=color_by if color_by != "None" else None,
        hover_data=hover_params,
        title=f"Quality vs Quantity: ASI vs SI for {solution_type} solution",
        height=600,
    )

    # Add reference line y=x
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color="Gray", dash="dash"),
    )

    # Add lines indicating performance thresholds
    for level, threshold in DEFAULT_ASI_SI_COMPARISON_THRESHOLDS.items():
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=threshold * max_val,
            line=dict(color=CATEGORY_COLORS.get(level, "LightGray"), dash="dot"),
            name=f"{level}",
            legendgroup=level,
            showlegend=True,
        )

    fig.update_layout(
        xaxis_title="Success Index",
        yaxis_title="Adjusted Success Index",
        xaxis_range=[0, max_val],
        yaxis_range=[0, max_val],
    ).show(config={"displayModeBar": False})
    return


def analyze_metric_results(
    df: pd.DataFrame,
    group_identifier: Literal["TIER_ID", "CASE_ID", None],
    identifiers_list: Optional[List[str]] = None,
    solution_type: Optional[str] = "all types",
    metric_name: Literal[
        "Recall",
        "Precision",
        "F1_Score",
        "Success_Index",
        "Adjusted_Success_Index",
        "Jaccard",
        "Miss_Rate",
        "FDR",
        "Global_Miss_Rate",
        "Global_FDR",
    ] = DEFAULT_METRIC,
    thresholds: Dict[str, float] = None,
    custom_title: Optional[str] = None,
) -> None:
    """
    Analyze and visualize the distribution of a specified metric for a given solution type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The unique identifier values to filter the data (e.g., tier IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : Optional[str]
        The solution type to analyze (e.g., "full", "top", "outlier_STD_2.5").
        Input "all types" (default) to include all solution types.
    metric_name : Optional[Literal[
        "Recall",
        "Precision",
        "F1_Score",
        "Success_Index",
        "Adjusted_Success_Index",
        "Jaccard",
        "Miss_Rate",
        "FDR",
        "Global_Miss_Rate",
        "Global_FDR",
    ]] = DEFAULT_METRIC,
        The metric to analyze (e.g., "Recall", "Precision"). DEFAULT_METRIC by default.
    thresholds : Optional[Dict[str, float]]
        Dictionary defining the lower bounds for performance categories.
        Defaults to THRESHOLDS_FOR_METRIC for the given metric.
    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df_plot = df[df[group_identifier].isin(identifiers_list)]
    else:
        df_plot = df

    if solution_type != "all types":
        df_plot = df_plot[df_plot["solution_type"] == solution_type]
    else:
        solution_type = "all"  # for display purposes

    # Categorize performance
    def categorize_metric_higher_is_better(val):
        if pd.isna(val):
            return "Unknown"
        if val >= thresholds["Excellent"]:
            return "Excellent"
        if val >= thresholds["Good"]:
            return "Good"
        if val >= thresholds["Moderate"]:
            return "Moderate"
        return "Poor"

    def categorize_metric_lower_is_better(val):
        if pd.isna(val):
            return "Unknown"
        if val <= thresholds["Excellent"]:
            return "Excellent"
        if val <= thresholds["Good"]:
            return "Good"
        if val <= thresholds["Moderate"]:
            return "Moderate"
        return "Poor"

    # Get default thresholds, if defined
    if thresholds is None:
        thresholds = THRESHOLDS_FOR_METRIC.get(metric_name, None)
    if thresholds is None:
        print(f"No thresholds defined for {metric_name}. Skipping analysis.")
        return

    # Categorize based on whether higher or lower values are better
    if metric_name in [
        "Recall",
        "Precision",
        "F1_Score",
        "Success_Index",
        "Adjusted_Success_Index",
    ]:
        categories = df_plot[metric_name].apply(categorize_metric_higher_is_better)
    else:
        categories = df_plot[metric_name].apply(categorize_metric_lower_is_better)

    # Count occurrences in each category
    counts = categories.value_counts()

    # Ensure all categories are present for consistent plotting
    plot_data = pd.DataFrame(
        {
            "Category": CATEGORY_ORDER,
            "Count": [counts.get(cat, 0) for cat in CATEGORY_ORDER],
        }
    )

    # Print summary stats
    display(Markdown(f"#### Statistics:"))
    display(Markdown(f"- **Mean {metric_name}:** {df[metric_name].mean():.3f}"))
    display(Markdown(f"- **Median {metric_name}:** {df[metric_name].median():.3f}\n"))

    if custom_title is not None:
        title = custom_title
    else:
        title = f"{metric_name} performance for {solution_type} solutions,<br>{group_identifier} = {identifiers_list}"

    fig = px.bar(
        plot_data,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        title=title,
        text="Count",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Number of Experiments",
        showlegend=False,
        width=600,
    )
    fig.show(config={"displayModeBar": False})
    return


def plot_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    y_axis: str,
    group_identifier: Literal["TIER_ID", "CASE_ID", None],
    identifiers_list: Optional[List[str]] = None,
    solution_type: Optional[str] = "all types",
    aggregation_func: Literal["mean", "median"] = DEFAULT_AGGREGATION_FUNC,
) -> None:
    """
    Plot a heatmap showing the interaction of two parameters on a metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results data.
    x_axis : str
        Parameter for x-axis.
    y_axis : str
        Parameter for y-axis.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : Optional[str] = "all types"
        The solution type to analyze. Input "all types" to include all solution types.
    metric_name : str
        The metric to plot as color.
    aggregation_func : Literal["mean", "median"], optional
        The aggregation function to use when multiple entries exist
        for the same (x_axis, y_axis) pair. DEFAULT_AGGREGATION_FUNC by default.

    Returns
    -------
    None
    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df_plot = df[df[group_identifier].isin(identifiers_list)]
    else:
        df_plot = df

    if solution_type != "all types":
        df_plot = df_plot[df_plot["solution_type"] == solution_type]
    else:
        solution_type = "all"  # for display purposes
    df_plot = df_plot.copy()

    if metric_name not in df_plot.columns:
        print(f"Column '{metric_name}' not found in dataframe.")
        return

    if aggregation_func == "mean":
        agg_func = pd.Series.mean
    elif aggregation_func == "median":
        agg_func = pd.Series.median
    else:
        print(f"Unsupported aggregation function: {aggregation_func}")
        return

    # Aggregate: there might be multiple runs for the same grid point
    heatmap_data = (
        df_plot.groupby([y_axis, x_axis])[metric_name].agg(agg_func).reset_index()
    )

    # Pivot for heatmap format
    pivot_table = heatmap_data.pivot(index=y_axis, columns=x_axis, values=metric_name)

    fig = px.imshow(
        pivot_table,
        labels=dict(x=x_axis, y=y_axis, color=metric_name),
        title=f"{metric_name} for {solution_type} solution",
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=(
            "RdBu_r" if "Miss" in metric_name or "FDR" in metric_name else "Viridis"
        ),
    )
    fig.update_layout(height=500)
    fig.show(config={"displayModeBar": False})
    return


def plot_metric_vs_hyperparam(
    df_grouped: pd.DataFrame,
    hyperparam: str,
    solution_options: List[str],
) -> None:
    """
    Add a line for each solution and each metric that ranges [0,1], colored by
        1. solution type
        2. metric type

    Parameters
    ----------
    df_grouped : pd.DataFrame
        DataFrame containing grouped results data.
    hyperparam : str
        The hyperparameter to plot on the x-axis.
    solution_options : List[str]
        List of solution types to include in the plot.

    Returns
    -------
    None
    """
    select_metrics = [
        "Recall",
        "Precision",
        "F1_Score",
        "Jaccard",
    ]  # only metrics with range [0,1]
    fig = go.Figure()
    for sol in solution_options:
        for metric in select_metrics:
            col_name = f"{sol}_{metric}"
            if col_name in df_grouped.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_grouped.index,
                        y=df_grouped[col_name],
                        mode="lines+markers",
                        name=f"{sol} - {metric}",
                        line=dict(
                            color=px.colors.qualitative.Plotly[
                                solution_options.index(sol)
                            ],
                            dash=(
                                "solid"
                                if metric == select_metrics[0]
                                else (
                                    "dash"
                                    if metric == select_metrics[1]
                                    else (
                                        "dot"
                                        if metric == select_metrics[2]
                                        else "longdash"
                                    )
                                )
                            ),
                        ),
                    )
                )
    fig.update_layout(
        title=f"Effect of {hyperparam} on metrics",
        xaxis_title=hyperparam,
        yaxis_title="Metric value",
        yaxis=dict(range=[-0.1, 1.1]),
        height=600,
    ).show(config={"displayModeBar": False})
    return
