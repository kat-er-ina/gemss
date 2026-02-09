""" """

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from gemss.experiment_assessment.experiment_results_analysis import (
    COVERAGE_METRICS,
    DEFAULT_AGGREGATION_FUNC,
    DEFAULT_ASI_SI_COMPARISON_THRESHOLDS,
    DEFAULT_METRIC,
    analyze_metric_results,
)

# Define colors for any threshold-based categories
CATEGORY_COLORS = {
    'Excellent': 'green',
    'Good': 'lightgreen',
    'Moderate': 'lightyellow',
    'Poor': 'lightcoral',
    'Unknown': 'gray',
}
AVAILABLE_SYMBOLS = [
    'circle',
    'square',
    'diamond',
    'cross',
    'x',
    'triangle-up',
    'triangle-down',
    'triangle-left',
    'triangle-right',
    'pentagon',
    'hexagon',
    'star',
    'hourglass',
    'bowtie',
    'asterisk',
    'hash',
    'y-up',
    'y-down',
    'y-left',
    'y-right',
    'line-ew',
    'line-ns',
    'line-ne',
    'line-nw',
    'arrow-up',
    'arrow-down',
    'arrow-left',
    'arrow-right',
    'arrow-bar-up',
    'arrow-bar-down',
    'arrow-bar-left',
    'arrow-bar-right',
]


def plot_solution_grouped(
    df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    color_by: str | None,
    symbol_by: str | None,
    group_identifier: Literal['TIER_ID', 'CASE_ID', None],
    identifiers_list: list[str] | None = None,
    solution_type: str | None = 'all types',
    hover_params: list[str] | None = None,
) -> None:
    """
    Plot a metric for a given solution type, grouped by a specified parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results data.
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    color_by : str | None
        The parameter to group the lines by (or None for no grouping).
    symbol_by : str | None
        The parameter to differentiate point symbols by (or None for no differentiation).
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : list[str] | None = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : str | None = "all types"
        The type of solution to plot (e.g., "full", "top", "outlier_STD_2.5").
        Input "all types" to include all solution types.
    hover_params : list[str] | None
        List of parameters to show on hover in the plot.
        If None, defaults to ["EXPERIMENT_ID"]. "EXPERIMENT_ID" is always included.

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

    if solution_type != 'all types':
        df_plot = df_plot[df_plot['solution_type'] == solution_type]
    else:
        solution_type = 'all'  # for display purposes
    df_plot = df_plot.copy()

    if metric_name not in df.columns:
        print(f"Column '{metric_name}' not found in dataframe.")
        return

    # Create title
    title = f'{metric_name} for {x_axis}'
    if color_by is not None:
        title += f', grouped by {color_by}'
    if symbol_by is not None:
        title += f', symbol by {symbol_by}'

    # Sort for cleaner line plots
    sort_cols = [x_axis]
    if color_by is not None:
        sort_cols.append(color_by)
    if symbol_by is not None:
        sort_cols.append(symbol_by)
    df_plot = df_plot.sort_values(
        by=sort_cols,
        ascending=True,
    )

    # Convert group_by to string for discrete coloring if needed
    if color_by is not None:
        df_plot[color_by] = df_plot[color_by].astype(str)
    if symbol_by is not None:
        df_plot[symbol_by] = df_plot[symbol_by].astype(str)

    # Ensure EXPERIMENT_ID is always shown upon hover
    if (color_by != 'EXPERIMENT_ID') & (x_axis != 'EXPERIMENT_ID'):
        if hover_params is None:
            hover_params = ['EXPERIMENT_ID']
        elif 'EXPERIMENT_ID' not in hover_params:
            hover_params = ['EXPERIMENT_ID'] + hover_params

    # Create figure
    fig = go.Figure()

    if color_by is not None:
        # Group by the specified parameter and create separate traces
        unique_groups = df_plot[color_by].unique()
        colors = px.colors.qualitative.Plotly

        # If symbol_by is specified, create traces for each combination of color_by and symbol_by
        if symbol_by is not None:
            unique_symbol_values = df_plot[symbol_by].unique()
            symbol_map = {
                val: AVAILABLE_SYMBOLS[j % len(AVAILABLE_SYMBOLS)]
                for j, val in enumerate(unique_symbol_values)
            }

            for i, group_val in enumerate(unique_groups):
                group_data = df_plot[df_plot[color_by] == group_val]

                for k, symbol_val in enumerate(unique_symbol_values):
                    symbol_data = group_data[group_data[symbol_by] == symbol_val]

                    if len(symbol_data) == 0:
                        continue

                    hovertemplate = (
                        f'{x_axis}: %{{x}}<br>'
                        f'{metric_name}: %{{y}}<br>'
                        f'{color_by}: {group_val}<br>'
                        f'{symbol_by}: {symbol_val}<br><br>'
                        + '<br>'.join(
                            [
                                f'{param}: %{{customdata[{j}]}}'
                                for j, param in enumerate(hover_params)
                            ]
                        )
                        + '<extra></extra>'
                    )

                    # Use legendgroup to group traces by color, but show individual symbols
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data[x_axis],
                            y=symbol_data[metric_name].round(3),
                            mode='markers',
                            marker=dict(
                                size=10,
                                opacity=0.7,
                                color=colors[i % len(colors)],
                                symbol=symbol_map[symbol_val],
                            ),
                            name=f'{symbol_val}',  # Simple name showing only symbol value
                            legendgroup=f'{group_val}',  # Group by color parameter
                            legendgrouptitle_text=f'{color_by}: {group_val}',  # Group title
                            hovertemplate=hovertemplate,
                            customdata=symbol_data[hover_params].round(3).values,
                        )
                    )
        else:
            # No symbol differentiation, group only by color_by
            for i, group_val in enumerate(unique_groups):
                group_data = df_plot[df_plot[color_by] == group_val]

                hovertemplate = (
                    f'{x_axis}: %{{x}}<br>'
                    f'{metric_name}: %{{y}}<br>'
                    f'{color_by}: {group_val}<br><br>'
                    + '<br>'.join(
                        [f'{param}: %{{customdata[{j}]}}' for j, param in enumerate(hover_params)]
                    )
                    + '<extra></extra>'
                )

                fig.add_trace(
                    go.Scatter(
                        x=group_data[x_axis],
                        y=group_data[metric_name].round(3),
                        mode='markers',
                        marker=dict(
                            size=10,
                            opacity=0.7,
                            color=colors[i % len(colors)],
                            symbol='circle',
                        ),
                        name=str(group_val),
                        hovertemplate=hovertemplate,
                        customdata=group_data[hover_params].round(3).values,
                    )
                )
    else:
        # No grouping - single trace
        hovertemplate = (
            f'{x_axis}: %{{x}}<br>'
            f'{metric_name}: %{{y}}<br><br>'
            + '<br>'.join(
                [f'{param}: %{{customdata[{j}]}}' for j, param in enumerate(hover_params)]
            )
            + '<extra></extra>'
        )

        fig.add_trace(
            go.Scatter(
                x=df_plot[x_axis],
                y=df_plot[metric_name],
                mode='markers',
                marker=dict(size=10, opacity=0.7),
                name=solution_type,
                hovertemplate=hovertemplate,
                customdata=df_plot[hover_params].round(3).values,
            )
        )

    # Create legend title
    legend = f'{color_by}' if color_by is not None else ''
    legend += f'<br>symbol: {symbol_by}' if symbol_by is not None else ''

    # Update layout
    fig.update_layout(
        title=title,
        height=450,
        xaxis_title=x_axis,
        yaxis_title=metric_name,
        yaxis_type='log' if 'Success_Index' in metric_name else 'linear',
        legend_title=legend,
    )

    fig.show(config={'displayModeBar': False})
    return


def plot_solution_comparison(
    df: pd.DataFrame,
    solution_types: list[str],
    metric_name: str,
    x_axis: str,
    group_identifier: Literal['TIER_ID', 'CASE_ID', None],
    identifiers_list: list[str] | None = None,
    hover_params: list[str] | None = None,
) -> None:
    """
    Plot a comparison of a metric across different solution types.
    Each solution type is a separate line.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    solution_types : list[str]
        List of solution types to compare (e.g., ["full", "top", "outlier_STD_2.5"]).
    metric_name : str
        The base name of the metric to plot (e.g., "Recall", "Precision").
    x_axis : str
        The parameter to plot on the x-axis.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : list[str] | None = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    hover_params : list[str] | None
        List of parameters to show on hover in the plot.
        If None, defaults to ["EXPERIMENT_ID"]. "EXPERIMENT_ID" is always included.
    """
    if group_identifier is not None:
        # if no identifiers provided, use all unique identifiers
        if identifiers_list is None:
            identifiers_list = df[group_identifier].unique().tolist()
        # filter dataframe
        df = df[df[group_identifier].isin(identifiers_list)]
    else:
        df = df

    # Ensure EXPERIMENT_ID is always shown upon hover
    if x_axis != 'EXPERIMENT_ID':
        if hover_params is None:
            hover_params = ['EXPERIMENT_ID']
        elif 'EXPERIMENT_ID' not in hover_params:
            hover_params = ['EXPERIMENT_ID'] + hover_params

    fig = go.Figure()
    for solution_type in solution_types:
        df_solution = df[df['solution_type'] == solution_type]
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
                mode='markers',
                marker={'size': 10, 'opacity': 0.7},
                name=solution_type,
                hovertemplate=(
                    f'{x_axis}: %{{x}}<br>'
                    f'{metric_name}: %{{y}}<br><br>'
                    + '<br>'.join(
                        [f'{param}: %{{customdata[{i}]}}' for i, param in enumerate(hover_params)]
                    )
                    + '<br>'
                    + '<br>'.join(
                        [
                            f'{metric}: %{{customdata[{len(hover_params) + j}]}}'
                            for j, metric in enumerate(other_solution_metrics)
                        ]
                    )
                    + '<extra></extra>'
                ),
                customdata=df[hover_params + other_solution_metrics].round(3).values,
            )
        )

    fig.update_layout(
        title=f'{metric_name} for {x_axis}',
        height=600,
        xaxis_title=x_axis,
        yaxis_title=metric_name,
        yaxis_type='log' if 'Success_Index' in metric_name else 'linear',
        legend_title='Solution Type',
    ).show(config={'displayModeBar': False})
    return


def plot_si_asi_scatter(
    df: pd.DataFrame,
    color_by: str,
    hover_params: list[str],
    group_identifier: Literal['TIER_ID', 'CASE_ID', None],
    identifiers_list: list[str] | None = None,
    solution_type: str | None = 'all types',
) -> None:
    """
    Plot a scatter plot of Adjusted Success Index (ASI) vs Success Index (SI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results data.
    color_by : str
        The parameter to color the points by (or "None" for no coloring).
    hover_params : list[str]
        List of parameters to show on hover in the plot.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : list[str] | None = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : str | None = "all types"
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
    if solution_type != 'all types':
        df_plot = df_plot[df_plot['solution_type'] == solution_type]
    else:
        solution_type = 'all'  # for display purposes
    df_plot = df_plot.copy()

    si_col = 'Success_Index'
    asi_col = 'Adjusted_Success_Index'

    if si_col not in df.columns:
        print(f'Metrics not found for {solution_type}')
        return

    # Convert color column to string for categorical coloring
    if color_by != 'None':
        df_plot[color_by] = df_plot[color_by].astype(str)

    # Add diagonal line for reference (Ideal: ASI = SI)
    max_val = max(df_plot[si_col].max(), df_plot[asi_col].max()) * 1.1
    fig = px.scatter(
        df_plot,
        x=si_col,
        y=asi_col,
        color=color_by if color_by != 'None' else None,
        hover_data=hover_params,
        title=f'Quality vs. quantity: ASI vs SI for {solution_type} solution',
        height=600,
    )

    # Add reference line y=x
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color='Gray', dash='dash'),
    )

    # Add lines indicating performance thresholds
    for level, threshold in DEFAULT_ASI_SI_COMPARISON_THRESHOLDS.items():
        fig.add_shape(
            type='line',
            x0=0,
            y0=0,
            x1=max_val,
            y1=threshold * max_val,
            line=dict(color=CATEGORY_COLORS.get(level, 'LightGray'), dash='dot'),
            name=f'{level}',
            legendgroup=level,
            showlegend=True,
        )

    fig.update_layout(
        xaxis_title='Success Index',
        yaxis_title='Adjusted Success Index',
        xaxis_range=[0, max_val],
        yaxis_range=[0, max_val],
    ).show(config={'displayModeBar': False})
    return


def plot_category_counts(
    df_category_counts: pd.DataFrame,
    title: str | None = None,
) -> None:
    """
    Visualize the distribution of performance categories as a bar chart.

    Parameters
    ----------
    df_category_counts : pd.DataFrame
        DataFrame containing counts of experiments in each performance category.
    title : str | None = None
        Title for the plot, optional.

    Returns
    -------
    None
    """
    fig = px.bar(
        df_category_counts,
        x='Category',
        y='Count',
        color='Category',
        color_discrete_map=CATEGORY_COLORS,
        title=title,
        text='Count',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis_title='Number of experiments',
        showlegend=False,
        width=600,
        height=400,
    )
    fig.show(config={'displayModeBar': False})
    return


def plot_metric_analysis_overview(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID', None],
    identifiers_list: list[str] | None = None,
    solution_type: str | None = 'all types',
    metric_name: Literal[
        'Recall',
        'Precision',
        'F1_Score',
        'Success_Index',
        'Adjusted_Success_Index',
        'Jaccard',
        'Miss_Rate',
        'FDR',
        'Global_Miss_Rate',
        'Global_FDR',
    ] = DEFAULT_METRIC,
    thresholds: dict[str, float] = None,
    custom_title: str | None = None,
    verbose: bool | None = True,
) -> None:
    """
    Analyze and visualize the distribution of a specified metric for a given solution type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : list[str] | None = None
        The unique identifier values to filter the data (e.g., tier IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : str | None
        The solution type to analyze (e.g., "full", "top", "outlier_STD_2.5").
        Input "all types" (default) to include all solution types.
    metric_name : Literal[
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
    ] | None = DEFAULT_METRIC,
        The metric to analyze (e.g., "Recall", "Precision"). DEFAULT_METRIC by default.
    custom_title : str | None = None
        Custom title for the plot. If None, a default title is generated.
    thresholds : dict[str, float] | None
        Dictionary defining the lower bounds for performance categories.
        Defaults to THRESHOLDS_FOR_METRIC for the given metric.
    verbose : bool | None = True
        Whether to print summary statistics.
    """
    df_category_counts = analyze_metric_results(
        df=df,
        group_identifier=group_identifier,
        identifiers_list=identifiers_list,
        solution_type=solution_type,
        metric_name=metric_name,
        thresholds=thresholds,
        verbose=verbose,
    )

    if custom_title is not None:
        title = custom_title
    else:
        title = (
            f'{metric_name} for {solution_type} solutions,<br>'
            f'{group_identifier} = {identifiers_list}'
        )

    plot_category_counts(
        df_category_counts=df_category_counts,
        title=title,
    )
    return


def plot_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    x_axis: str,
    y_axis: str,
    group_identifier: Literal['TIER_ID', 'CASE_ID', None],
    identifiers_list: list[str] | None = None,
    solution_type: str | None = 'all types',
    aggregation_func: Literal['mean', 'median'] = DEFAULT_AGGREGATION_FUNC,
    title: str | None = None,
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
    identifiers_list : list[str] | None = None
        The identifier values to filter the data (e.g., tier IDs or case IDs).
        If None, all unique identifiers are used.
        Ignored if group_identifier is None.
    solution_type : str | None = "all types"
        The solution type to analyze. Input "all types" to include all solution types.
    metric_name : str
        The metric to plot as color.
    aggregation_func : Literal["mean", "median"], optional
        The aggregation function to use when multiple entries exist
        for the same (x_axis, y_axis) pair. DEFAULT_AGGREGATION_FUNC by default.
    title : str | None = None
        Title for the plot. If None, a default title is generated.

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

    if solution_type != 'all types':
        df_plot = df_plot[df_plot['solution_type'] == solution_type]
    else:
        solution_type = 'all'  # for display purposes
    df_plot = df_plot.copy()

    if metric_name not in df_plot.columns:
        print(f"Column '{metric_name}' not found in dataframe.")
        return

    if aggregation_func == 'mean':
        agg_func = pd.Series.mean
    elif aggregation_func == 'median':
        agg_func = pd.Series.median
    else:
        print(f'Unsupported aggregation function: {aggregation_func}')
        return

    # Aggregate: there might be multiple runs for the same grid point
    heatmap_data = df_plot.groupby([y_axis, x_axis])[metric_name].agg(agg_func).reset_index()

    # Pivot for heatmap format
    pivot_table = heatmap_data.pivot(index=y_axis, columns=x_axis, values=metric_name)

    # Convert axes to categorical (strings) for equal-sized cells
    pivot_table.index = pivot_table.index.astype(str)
    pivot_table.columns = pivot_table.columns.astype(str)

    is_01_range = False if 'ASI' in metric_name or 'SI' in metric_name else True
    is_reversed = True if 'Miss' in metric_name or 'FDR' in metric_name else False
    fig = px.imshow(
        pivot_table,
        labels=dict(x=x_axis, y=y_axis, color=metric_name),
        title=f'{metric_name} for {solution_type} solution' if title is None else title,
        text_auto='.2f',
        aspect='equal',
        origin='lower',
        zmin=0.0 if is_01_range else None,
        zmax=1.0 if is_01_range else None,
        color_continuous_scale='PuBuGn_r' if is_reversed else 'PuBuGn',
        # width=900,
        # height=900,
    )
    fig.update_traces(textfont_size=12)
    fig.show(config={'displayModeBar': False})
    return


def plot_metric_vs_hyperparam(
    df_grouped: pd.DataFrame,
    hyperparam: str,
    solution_options: list[str],
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
    solution_options : list[str]
        List of solution types to include in the plot.

    Returns
    -------
    None
    """
    select_metrics = [
        'Recall',
        'Precision',
        'F1_Score',
    ]  # only metrics with range [0,1]
    fig = go.Figure()
    for sol in solution_options:
        for metric in select_metrics:
            col_name = f'{sol}_{metric}'
            if col_name in df_grouped.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_grouped.index,
                        y=df_grouped[col_name],
                        mode='lines+markers',
                        name=f'{sol} - {metric}',
                        line=dict(
                            color=px.colors.qualitative.Plotly[solution_options.index(sol)],
                            dash=(
                                'solid'
                                if metric == select_metrics[0]
                                else (
                                    'dash'
                                    if metric == select_metrics[1]
                                    else ('dot' if metric == select_metrics[2] else 'longdash')
                                )
                            ),
                        ),
                    )
                )
    fig.update_layout(
        title=f'Effect of {hyperparam} on metrics',
        xaxis_title=hyperparam,
        yaxis_title='Metric value',
        yaxis=dict(range=[-0.1, 1.1]),
        height=600,
    ).show(config={'displayModeBar': False})
    return
