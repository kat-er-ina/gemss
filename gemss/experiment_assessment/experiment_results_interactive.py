""" """

from typing import List, Dict, Literal, Tuple
from IPython.display import display, Markdown
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

from gemss.experiment_assessment.experiment_results_analysis import (
    ALL_PARAMETERS,
    COVERAGE_METRICS,
    DEFAULT_AGGREGATION_FUNC,
    DEFAULT_SOLUTION,
    THRESHOLDS_FOR_METRIC,
    DEFAULT_METRIC,
)
from gemss.experiment_assessment.experiment_results_visualizations import (
    plot_metric_analysis_overview,
    plot_solution_comparison,
    plot_solution_grouped,
    plot_heatmap,
    plot_si_asi_scatter,
)


def show_interactive_performance_overview(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] = 'TIER_ID',
    metrics_list: List[str] = ['Recall', 'Precision', 'F1_Score'],
    show_metric_thresholds: bool = True,
) -> None:
    """
    Display an interactive performance overview of experiment results.
    Optionally show also metric thresholds for performance categories.

    Parameters:
    -----------
    df : pd.DataFrame
        The pivoted DataFrame containing experiment results.
        Must contain 'solution_type' and group_identifier columns.
    group_identifier : Literal["TIER_ID", "CASE_ID"], optional
        The column name used to group the data. Default is "TIER_ID".
    metrics_list : List[str], optional
        List of metrics available for analysis. Default includes
        "Recall", "Precision", and "F1_Score".
    show_metric_thresholds : bool, optional
        Whether to show a table of performance thresholds
        for the selected metrics. Default is True.

    Returns:
    --------
    None
    """
    group_ids = df[group_identifier].unique().tolist()
    solution_options = sorted(df['solution_type'].unique().tolist()) + ['all types']

    if show_metric_thresholds:
        df_thresholds = pd.DataFrame()
        for metric, thresholds in THRESHOLDS_FOR_METRIC.items():
            if (metric in metrics_list) and (thresholds is not None):
                df_thresholds[metric] = pd.Series(thresholds)
        display(Markdown(f'### Performance thresholds for selected metrics'))
        display(df_thresholds)

    if group_identifier == 'TIER_ID':
        group_identifier_description = 'Tier:'
    elif group_identifier == 'CASE_ID':
        group_identifier_description = 'Case ID:'
    else:
        group_identifier_description = group_identifier + ':'

    display(Markdown(f'### Quick performance overview'))
    interact(
        plot_metric_analysis_overview,
        df=fixed(df),
        identifiers_list=widgets.SelectMultiple(
            options=group_ids,
            value=group_ids,
            description=group_identifier_description,
        ),
        group_identifier=fixed(group_identifier),
        solution_type=widgets.Dropdown(
            options=solution_options,
            value=('all types' if group_identifier == 'CASE_ID' else DEFAULT_SOLUTION),
            description='Solution:',
        ),
        metric_name=widgets.Dropdown(
            options=sorted(metrics_list),
            value=DEFAULT_METRIC,
            description='Metric:',
        ),
        custom_title=fixed(None),
        thresholds=fixed(None),
        verbose=fixed(True),
    )
    return


def show_interactive_solution_comparison(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] = 'TIER_ID',
    show_average_metrics: bool = False,
) -> None:
    """
    Display an interactive solution comparison of experiment results.

    Parameters:
    -----------
    df : pd.DataFrame
        The pivoted DataFrame containing experiment results.
        Must contain 'solution_type' and group_identifier columns.
    group_identifier : Literal["TIER_ID", "CASE_ID"], optional
        The column name used to group the data. Default is "TIER_ID".
    show_average_metrics : bool, optional
        Whether to show a table of average metric values
        for each solution type and tier. Default is False.

    Returns:
    --------
    None
    """
    # interactive solution comparison
    display(Markdown('### Solution comparison'))

    group_ids = df[group_identifier].unique().tolist()
    solution_options = sorted(df['solution_type'].unique().tolist())
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]
    unvaried_params = [p for p in ALL_PARAMETERS if p in df.columns and p not in varied_params]

    if group_identifier == 'TIER_ID':
        group_identifier_description = 'Tier:'
    elif group_identifier == 'CASE_ID':
        group_identifier_description = 'Case ID:'
    else:
        group_identifier_description = group_identifier + ':'

    interact(
        plot_solution_comparison,
        df=fixed(df),
        identifiers_list=widgets.SelectMultiple(
            options=group_ids,
            value=group_ids,
            description=group_identifier_description,
        ),
        group_identifier=fixed(group_identifier),
        solution_types=fixed(solution_options),
        metric_name=widgets.Dropdown(
            options=COVERAGE_METRICS,
            value=DEFAULT_METRIC,
            description='Metric:',
        ),
        x_axis=widgets.Dropdown(
            options=varied_params,
            value='N_FEATURES' if 'N_FEATURES' in varied_params else varied_params[0],
            description='X-Axis:',
        ),
        hover_params=fixed(varied_params + unvaried_params),
    )

    # show average metrics table
    if show_average_metrics:
        display(Markdown('### Average values for selected metrics'))
        display(
            df[[group_identifier, 'solution_type'] + COVERAGE_METRICS]
            .groupby([group_identifier, 'solution_type'])
            .mean()
        )
    return


def show_interactive_comparison_with_grouping(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] = 'TIER_ID',
) -> None:
    """
    Display an interactive comparison of experiment results
    with grouping by a selected parameter.

    Parameters:
    -----------
    df : pd.DataFrame
        The pivoted DataFrame containing experiment results.
        Must contain 'solution_type' and group_identifier columns.
    group_identifier : Literal["TIER_ID", "CASE_ID"], optional
        The column name used to group the data. Default is "TIER_ID".

    Returns:
    --------
    None
    """

    group_ids = df[group_identifier].unique().tolist()
    solution_options = sorted(df['solution_type'].unique().tolist())
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]
    unvaried_params = [p for p in ALL_PARAMETERS if p in df.columns and p not in varied_params]

    if group_identifier == 'TIER_ID':
        group_identifier_description = 'Tier:'
    elif group_identifier == 'CASE_ID':
        group_identifier_description = 'Case ID:'
    else:
        group_identifier_description = group_identifier + ':'

    interact(
        plot_solution_grouped,
        df=fixed(df),
        identifiers_list=widgets.SelectMultiple(
            options=group_ids,
            value=group_ids,
            description=group_identifier_description,
        ),
        group_identifier=fixed(group_identifier),
        solution_type=widgets.Dropdown(
            options=solution_options,
            value=DEFAULT_SOLUTION,
            description='Solution:',
        ),
        metric_name=widgets.Dropdown(
            options=COVERAGE_METRICS,
            value=DEFAULT_METRIC,
            description='Metric:',
        ),
        x_axis=widgets.Dropdown(
            options=varied_params,
            value='N_FEATURES' if 'N_FEATURES' in varied_params else varied_params[0],
            description='X-Axis:',
        ),
        color_by=widgets.Dropdown(
            options=[None] + varied_params,
            value=(
                'SAMPLE_VS_FEATURE_RATIO' if 'SAMPLE_VS_FEATURE_RATIO' in varied_params else None
            ),
            description='Color by:',
        ),
        symbol_by=widgets.Dropdown(
            options=[None] + varied_params,
            value=None,
            description='Symbol by:',
        ),
        hover_params=fixed(varied_params + unvaried_params),
    )
    return


def show_interactive_heatmap(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] = 'TIER_ID',
) -> None:
    """
    Display an interactive heatmap of 2 parameters and 1 metric.

    Parameters:
    -----------
    df : pd.DataFrame
        The pivoted DataFrame containing experiment results.
    group_identifier : Literal["TIER_ID", "CASE_ID"], optional
        The column name used to group the data. Default is "TIER_ID".

    Returns:
    --------
    None
    """
    group_ids = df[group_identifier].unique().tolist()
    solution_options = sorted(df['solution_type'].unique().tolist())
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]
    unvaried_params = [p for p in ALL_PARAMETERS if p in df.columns and p not in varied_params]

    if group_identifier == 'TIER_ID':
        group_identifier_description = 'Tier:'
    elif group_identifier == 'CASE_ID':
        group_identifier_description = 'Case ID:'
    else:
        group_identifier_description = group_identifier + ':'

    interact(
        plot_heatmap,
        df=fixed(df),
        identifiers_list=widgets.SelectMultiple(
            options=group_ids,
            value=group_ids,
            description=group_identifier_description,
        ),
        group_identifier=fixed(group_identifier),
        solution_type=widgets.Dropdown(
            options=solution_options,
            value=DEFAULT_SOLUTION,
            description='Solution:',
        ),
        metric_name=widgets.Dropdown(
            options=sorted(COVERAGE_METRICS),
            value=DEFAULT_METRIC,
            description='Metric:',
        ),
        x_axis=widgets.Dropdown(
            options=varied_params + unvaried_params,
            value='N_FEATURES' if 'N_FEATURES' in varied_params else varied_params[0],
            description='X-Axis:',
        ),
        y_axis=widgets.Dropdown(
            options=varied_params,
            value='SPARSITY' if 'SPARSITY' in varied_params else varied_params[0],
            description='Y-Axis:',
        ),
        aggregation_func=widgets.Dropdown(
            options=['mean', 'median'],
            value=DEFAULT_AGGREGATION_FUNC,
            description='Aggregation:',
        ),
        title=fixed(None),
    )
    return


def show_interactive_si_asi_comparison(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] = 'TIER_ID',
) -> None:
    """
    Display an interactive SI vs ASI scatter plot of experiment results.

    Parameters:
    -----------
    df : pd.DataFrame
        The pivoted DataFrame containing experiment results.
    group_identifier : Literal["TIER_ID", "CASE_ID"], optional
        The column name used to group the data. Default is "TIER_ID".

    Returns:
    --------
    None
    """
    group_ids = df[group_identifier].unique().tolist()
    solution_options = sorted(df['solution_type'].unique().tolist())
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]

    if group_identifier == 'TIER_ID':
        group_identifier_description = 'Tier:'
    elif group_identifier == 'CASE_ID':
        group_identifier_description = 'Case ID:'
    else:
        group_identifier_description = group_identifier + ':'

    interact(
        plot_si_asi_scatter,
        df=fixed(df),
        identifiers_list=widgets.SelectMultiple(
            options=group_ids,
            value=group_ids,
            description=group_identifier_description,
        ),
        group_identifier=fixed(group_identifier),
        solution_type=widgets.Dropdown(
            options=solution_options,
            value=DEFAULT_SOLUTION,
            description='Solution:',
        ),
        color_by=widgets.Dropdown(
            options=[None] + varied_params,
            value='NOISE_STD' if 'NOISE_STD' in varied_params else None,
            description='Color By:',
        ),
        hover_params=fixed(varied_params),
    )
    return
