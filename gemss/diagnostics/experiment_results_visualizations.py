""" """

from typing import List, Dict, Literal, Tuple
from IPython.display import display, Markdown
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Define order and colors for any threshold-based categories
CATEGORY_ORDER = ["Excellent", "Good", "Moderate", "Poor", "Unknown"]
CATEGORY_COLORS = {
    "Excellent": "green",
    "Good": "lightgreen",
    "Moderate": "lightyellow",
    "Poor": "lightcoral",
    "Unknown": "gray",
}
# ASI vs SI comparison thresholds
DEFAULT_ASI_SI_COMPARISON_THRESHOLDS = {
    "Excellent": 0.85,
    "Good": 0.5,
    "Moderate": 0.1,
}
# anything below 'Moderate' is 'Poor'
DEFAULT_RECALL_THRESHOLDS = {
    "Excellent": 0.9,
    "Good": 0.8,
    "Moderate": 0.65,
}
DEFAULT_PRECISION_THRESHOLDS = {
    "Excellent": 0.9,
    "Good": 0.8,
    "Moderate": 0.65,
}
DEFAULT_F1SCORE_THRESHOLDS = {
    "Excellent": 0.9,
    "Good": 0.8,
    "Moderate": 0.65,
}
THRESHOLDS_FOR_METRIC = {
    "Recall": DEFAULT_RECALL_THRESHOLDS,
    "Precision": DEFAULT_PRECISION_THRESHOLDS,
    "F1_Score": DEFAULT_F1SCORE_THRESHOLDS,
    "Success_Index": None,
    "Adjusted_Success_Index": None,
}


def plot_solution_grouped(
    df: pd.DataFrame,
    tier: Tuple[str],
    solution_type: str,
    metric_name: str,
    x_axis: str,
    color_by: str,
    hover_params: List[str],
) -> None:
    """
    Plot a metric for a given solution type, grouped by a specified parameter.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    tier : Tuple[str]
        The tier identifiers to filter the data.
    solution_type : str
        The type of solution to plot (e.g., "full", "top", "outlier_STD_2.5").
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    color_by : str
        The parameter to group the lines by (or "None" for no grouping).
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    """
    df = df[df["TIER_ID"].isin(tier)]

    # Construct the full column name
    # The solution types in CSV usually look like: "full", "top", "outlier_STD_2.5"
    full_metric_col = f"{solution_type}_{metric_name}"

    if full_metric_col not in df.columns:
        print(
            f"Column '{full_metric_col}' not found in dataframe. Available columns similar to '{metric_name}':"
        )
        print([c for c in df.columns if metric_name in c])
        return

    # Create title
    title = f"{metric_name} vs. {x_axis} (Solution: {solution_type})"
    if color_by != "None":
        title += f", grouped by {color_by}"

    # Convert group_by to string for discrete coloring if needed
    df_plot = df.copy()
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
                f"{full_metric_col}: %{{y}}<br>"
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
                    y=group_data[full_metric_col].round(3),
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
            f"{full_metric_col}: %{{y}}<br><br>"
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
                y=df_plot[full_metric_col],
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
    tier: Tuple[str],
    solution_types: List[str],
    metric_name: str,
    x_axis: str,
    hover_params: List[str],
) -> None:
    """
    Plot a comparison of a metric across different solution types.
    Each solution type is a separate line.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    tier : Tuple[str]
        The tier identifiers to filter the data.
    solution_types : List[str]
        List of solution types to compare (e.g., ["full", "top", "outlier_STD_2.5"]).
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    """
    df = df[df["TIER_ID"].isin(tier)]

    fig = go.Figure()
    for solution_type in solution_types:
        full_metric_col = f"{solution_type}_{metric_name}"
        if full_metric_col not in df.columns:
            print(
                f"Column '{full_metric_col}' not found in dataframe. Available columns similar to '{metric_name}':"
            )
            print([c for c in df.columns if metric_name in c])
            continue

        other_solution_metrics = [
            col for col in df.columns if solution_type in col and col != full_metric_col
        ]
        fig.add_trace(
            go.Scatter(
                x=df[x_axis],
                y=df[full_metric_col].round(3),
                mode="markers",
                marker={"size": 10, "opacity": 0.7},
                name=solution_type,
                hovertemplate=(
                    f"{x_axis}: %{{x}}<br>"
                    f"{full_metric_col}: %{{y}}<br><br>"
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
    tier: Tuple[str],
    solution_type: str,
    color_by: str,
    hover_params: List[str],
) -> None:
    """
    Plot a scatter plot of Adjusted Success Index (ASI) vs Success Index (SI).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    tier : Tuple[str]
        The tier identifiers to filter the data.
    solution_type : str
        The type of solution to plot (e.g., "full", "top", "outlier_STD_2.5").
    color_by : str
        The parameter to color the points by (or "None" for no coloring).
    hover_params : List[str]
        List of parameters to show on hover in the plot.
    """
    df = df[df["TIER_ID"].isin(tier)]

    si_col = f"{solution_type}_Success_Index"
    asi_col = f"{solution_type}_Adjusted_Success_Index"

    if si_col not in df.columns:
        print(f"Metrics not found for {solution_type}")
        return

    # Convert color column to string for categorical coloring
    df_plot = df.copy()
    if color_by != "None":
        df_plot[color_by] = df_plot[color_by].astype(str)

    # Add diagonal line for reference (Ideal: ASI = SI)
    max_val = max(df[si_col].max(), df[asi_col].max()) * 1.1

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
    tier: Tuple[str],
    solution_type: str,
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
    ],
    thresholds: Dict[str, float] = None,
) -> None:
    """
    Analyze and visualize the distribution of a specified metric for a given solution type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    tier : Tuple[str]
        The tier identifiers to filter the data.
    metric_name : str
        The metric to analyze (e.g., "Recall", "Precision").
    solution_type : str
        The solution type to analyze (e.g., "full", "top", "outlier_STD_2.5").
    thresholds : Dict[str, float], optional
        Dictionary defining the lower bounds for performance categories.
        Defaults to THRESHOLDS_FOR_METRIC for the given metric.
    """
    df = df[df["TIER_ID"].isin(tier)]

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

    metric_col = f"{solution_type}_{metric_name}"

    if metric_col not in df.columns:
        print(f"{metric_name} metric not found for {solution_type}")
        return

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
        categories = df[metric_col].apply(categorize_metric_higher_is_better)
    else:
        categories = df[metric_col].apply(categorize_metric_lower_is_better)

    # Count occurrences in each category
    counts = categories.value_counts()

    # Ensure all categories are present for consistent plotting
    plot_data = pd.DataFrame(
        {
            "Category": CATEGORY_ORDER,
            "Count": [counts.get(cat, 0) for cat in CATEGORY_ORDER],
        }
    )

    # Create visualization
    fig = px.bar(
        plot_data,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        title=f"{metric_name} Performance Distribution for {solution_type} solutions",
        text="Count",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Number of Experiments",
        showlegend=False,
        width=600,
    )
    fig.show(config={"displayModeBar": False})

    # Print summary stats
    display(Markdown(f"### Analysis for {solution_type} solutions:"))
    display(Markdown(f"**Mean {metric_name}:** {df[metric_col].mean():.3f}"))
    display(Markdown(f"**Median {metric_name}:** {df[metric_col].median():.3f}\n"))
    return


def plot_heatmap(
    df: pd.DataFrame,
    tier: Tuple[str],
    solution_type: str,
    metric_name: str,
    x_axis: str,
    y_axis: str,
) -> None:
    """
    Plot a heatmap showing the interaction of two parameters on a metric.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    tier : Tuple[str]
        The tier identifiers to filter the data.
    solution_type : str
        The solution type to analyze.
    metric_name : str
        The metric to plot as color.
    x_axis : str
        Parameter for x-axis.
    y_axis : str
        Parameter for y-axis.
    """
    df = df[df["TIER_ID"].isin(tier)]

    full_metric_col = f"{solution_type}_{metric_name}"

    if full_metric_col not in df.columns:
        return

    # Aggregate: there might be multiple runs for the same grid point
    heatmap_data = df.groupby([y_axis, x_axis])[full_metric_col].mean().reset_index()

    # Pivot for heatmap format
    pivot_table = heatmap_data.pivot(
        index=y_axis, columns=x_axis, values=full_metric_col
    )

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
    fig.update_layout(height=500, yaxis_autorange="reversed")
    fig.show(config={"displayModeBar": False})
    return
