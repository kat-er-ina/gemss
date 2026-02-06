import os
from pathlib import Path
from typing import Literal

import pandas as pd
from IPython.display import Markdown, display

from gemss.config.constants import EXPERIMENT_RESULTS_DIR
from gemss.experiment_assessment.case_analysis import (
    CASE_DESCRIPTION,
    case2set,
)

# Define order for any threshold-based categories
CATEGORY_ORDER = ['Excellent', 'Good', 'Moderate', 'Poor', 'Unknown']

# Identify metric columns (those containing the base name of coverage metrics)
# List synchronized with keys returned by calculate_coverage_metrics in run_experiment.py
# All coverage metrics are numeric (possibly None)
COVERAGE_METRICS = [
    'F1_Score',
    'Recall',
    'Precision',
    'Jaccard',
    'Miss_Rate',
    'FDR',
    'Global_Miss_Rate',
    'Global_FDR',
    'Success_Index',
    'Adjusted_Success_Index',
]

CORE_METRICS = [
    'F1_Score',
    'Adjusted_Success_Index',
    'Recall',
    'Jaccard',
]

IMPORTANT_METRICS = CORE_METRICS + [
    'Precision',
    'Success_Index',
]

DEFAULT_METRIC = 'F1_Score'

SOLUTION_OPTIONS = [
    'full',
    'top',
    'outlier_STD_2.0',
    'outlier_STD_2.5',
    'outlier_STD_3.0',
]

DEFAULT_SOLUTION = 'top'

DEFAULT_AGGREGATION_FUNC = 'mean'

ALL_PARAMETERS = [
    'N_SAMPLES',
    'N_FEATURES',
    '[N_SAMPLES, N_FEATURES] COMBINATION',
    'SAMPLE_VS_FEATURE_RATIO',
    'SPARSITY',
    'N_GENERATING_SOLUTIONS',
    'N_CANDIDATE_SOLUTIONS',
    'NOISE_STD',
    'NAN_RATIO',
    '[NOISE_STD, NAN_RATIO] COMBINATION',
    'LAMBDA_JACCARD',
    'BINARY_RESPONSE_RATIO',
    'BINARIZE',
]

# ASI vs SI comparison thresholds
DEFAULT_ASI_SI_COMPARISON_THRESHOLDS = {
    'Excellent': 0.85,
    'Good': 0.5,
    'Moderate': 0.2,
}
# anything below 'Moderate' is 'Poor'
DEFAULT_RECALL_THRESHOLDS = {
    'Excellent': 0.9,
    'Good': 0.8,
    'Moderate': 0.65,
}
DEFAULT_PRECISION_THRESHOLDS = {
    'Excellent': 0.9,
    'Good': 0.7,
    'Moderate': 0.5,
}
DEFAULT_F1SCORE_THRESHOLDS = {
    'Excellent': 0.85,  # corresponds to Precision = 0.8, Recall = 0.9
    'Good': 0.71,  # average between moderate and excellent
    'Moderate': 0.565,  # corresponds to Precision = 0.5, Recall = 0.65
}

THRESHOLDS_FOR_METRIC = {
    'Recall': DEFAULT_RECALL_THRESHOLDS,
    'Precision': DEFAULT_PRECISION_THRESHOLDS,
    'F1_Score': DEFAULT_F1SCORE_THRESHOLDS,
    'Success_Index': None,
    'Adjusted_Success_Index': None,
}

THRESHOLDED_METRICS = [m for m in CORE_METRICS if THRESHOLDS_FOR_METRIC.get(m) is not None]
################################################################################################


def load_experiment_results(
    tier_id_list: list[int] = [1, 2, 3, 4, 5, 6, 7],
    results_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load experiment results from specified tiers into a single DataFrame.

    Parameters:
    -----------
    tier_id_list: List[int], optional
        List of tier IDs to load experiment results from. Each tier corresponds to a subdirectory
        named "tier{tier_id}" under the results_path (if provided) or the default experiment
        results directory.
        If not provided, defaults to tiers 1 through 7.
    results_dir: str, optional
        Base path to the experiment results directory. If None, defaults to EXPERIMENT_RESULTS_DIR.
    verbose: bool, optional
        If True, prints status messages during loading.

    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]
        A tuple containing:
        - A DataFrame with combined experiment results from the specified tiers.
        - A list of metric column names (containing both metric names and solution types).
    """
    df = pd.DataFrame()
    results_root = Path(results_dir) if results_dir is not None else EXPERIMENT_RESULTS_DIR
    for tier_id in tier_id_list:
        results_path = results_root / f'tier{tier_id}' / 'tier_summary_metrics.csv'

        if os.path.exists(results_path):
            df_tier = pd.read_csv(results_path)
            if verbose:
                display(
                    Markdown(
                        f'Successfully loaded {len(df_tier)} experiment records from **Tier {tier_id}**.'
                    )
                )
            # Ensure numeric columns are actually numeric
            metric_cols = [c for c in df_tier.columns if any(x in c for x in COVERAGE_METRICS)]
            for col in metric_cols:
                if col in df_tier.columns:
                    df_tier[col] = pd.to_numeric(df_tier[col], errors='coerce')

            # Add TIER_ID column
            df_tier['TIER_ID'] = int(tier_id)

            # Add EXPERIMENT_ID column: {tier_id}.{experiment_number_in_tier}
            df_tier['EXPERIMENT_ID'] = str(tier_id) + '.' + (df_tier.index + 1).astype(str)

        else:
            if verbose:
                display(Markdown(f'**ERROR:** File not found at {results_path}'))
                display(
                    Markdown('Please run the experiments for this tier first, or check the path.')
                )
            df_tier = pd.DataFrame()

        # Append to main DataFrame
        df = pd.concat([df, df_tier], ignore_index=True)

    # Add the combination of samples and features
    df['[N_SAMPLES, N_FEATURES] COMBINATION'] = (
        '[' + df['N_SAMPLES'].astype(str) + ', ' + df['N_FEATURES'].astype(str) + ']'
    )
    # Add the "SAMPLE_VS_FEATURE_RATIO" column
    df['SAMPLE_VS_FEATURE_RATIO'] = df['N_SAMPLES'] / df['N_FEATURES']
    # Add a feature that combines information about noise and missingness
    df['[NOISE_STD, NAN_RATIO] COMBINATION'] = (
        '[' + df['NOISE_STD'].astype(str) + ', ' + df['NAN_RATIO'].astype(str) + ']'
    )
    return (df, metric_cols)


def print_dataframe_overview(
    df: pd.DataFrame,
) -> None:
    """
    Print an overview of the experiment results DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing experiment results.
    """
    tier_id_list = df['TIER_ID'].unique().tolist()
    display(Markdown(f'### All results for tiers: {tier_id_list}'))

    if 'solution_type' in df.columns:
        total_exp_count = len(df) // df['solution_type'].nunique()
    else:
        total_exp_count = len(df)
    display(Markdown(f'- **Total experiments:** {total_exp_count}'))

    display(
        Markdown(f'- **{len(SOLUTION_OPTIONS)} solution types:** {", ".join(SOLUTION_OPTIONS)}')
    )
    display(
        Markdown(f'- **{len(COVERAGE_METRICS)} available metrics:** {", ".join(COVERAGE_METRICS)}')
    )
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]
    display(Markdown(f'- **Varied parameters in this analysis:** {", ".join(varied_params)}'))
    return


def pivot_df_by_solution_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the DataFrame by solution type.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame containing all experiment results.

    Returns
    -------
    pd.DataFrame
        The pivoted DataFrame with a 'solution_type' column and unified metric columns
        instead of separate metric columns for each solution type.
    """
    varied_params = [p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1]
    df_pivot = pd.DataFrame()
    for solution in SOLUTION_OPTIONS:
        solution_cols = [col for col in df.columns if solution in col]
        df_solution = df[['TIER_ID', 'EXPERIMENT_ID'] + varied_params + solution_cols].copy()
        df_solution['TIER_ID'] = df_solution['TIER_ID'].astype(str)
        df_solution.rename(
            columns={col: col.replace(f'{solution}_', '') for col in solution_cols},
            inplace=True,
        )
        df_solution['solution_type'] = solution
        df_pivot = pd.concat([df_pivot, df_solution], ignore_index=True)
    return df_pivot


def get_all_experiment_results(
    tier_id_list: list[int] = [1, 2, 3, 4, 5, 6, 7],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and return all experiment results from the specified tier IDs,
    pivoted by solution type.

    Parameters
    ----------
    tier_id_list : List[int], optional
        List of tier IDs to load results from. Default is [1, 2, 3, 4, 5, 6, 7].
    verbose : bool, optional
        If True, print loading status for each tier and an overview
        of the loaded DataFrame. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all experiment results, pivoted by solution type.
    """
    df, _ = load_experiment_results(tier_id_list, verbose=verbose)
    if verbose:
        print_dataframe_overview(df)
    df = pivot_df_by_solution_type(df)
    return df


def get_average_metrics_per_group(
    df: pd.DataFrame,
    group_identifier: str | None = 'TIER_ID',
    aggregation_func: Literal['mean', 'median'] = 'mean',
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Calculate mean or median performance metrics per group (e.g., per tier or case).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results with performance metrics.
        Must include 'solution_type' and performance metric columns and
        the group identifier column.
    group_identifier : str, optional
        Column name to group by (e.g., 'TIER_ID' or 'CASE_ID'), by default 'TIER_ID'.
    aggregation_func : Literal["mean", "median"], optional
        Aggregation function to use for averaging metrics, by default 'mean'.
    verbose : bool, optional
        If True, display detailed information, by default False.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are group identifiers and values are DataFrames
        containing mean or median performance metrics per solution type for that group.
    """
    if aggregation_func == 'mean':
        agg_func = pd.Series.mean
    elif aggregation_func == 'median':
        agg_func = pd.Series.median
    average_metrics = {}
    all_groups = df[group_identifier].unique().tolist()
    second_metric = 'Recall' if DEFAULT_METRIC == 'F1_Score' else 'F1_Score'
    for group in all_groups:
        df_group = df[df[group_identifier] == group]
        df_group_avg = (
            df_group.groupby('solution_type')[IMPORTANT_METRICS]
            .agg(agg_func)
            .sort_values(by=[DEFAULT_METRIC, second_metric], ascending=False)
            .round(3)
        )
        average_metrics[f'{group_identifier} = {group}'] = df_group_avg

        if verbose:
            display(
                Markdown(
                    f'### {aggregation_func.capitalize()} performance metrics for {group_identifier} = {group}'
                )
            )
            display(df_group_avg)
    return average_metrics


def get_average_metrics_per_case(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes mean metrics for all cases provided in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results with performance metrics. It is intended to be
        used on the output of function filter_df_best_solutions()

    Returns
    -------
    pd.DataFrame
        A DataFrame containing mean performance metrics per case, with case descriptions.
    """

    df_mean_metrics_per_case = get_average_metrics_per_group(
        df,
        group_identifier='CASE_ID',
        aggregation_func='mean',
    )

    # Concatenate the dictionary of DataFrames into a single DataFrame
    df_mean_metrics_per_case_concat = pd.concat(
        df_mean_metrics_per_case.values(), keys=df_mean_metrics_per_case.keys()
    )
    # Add "Mean " prefix to all metric columns
    df_mean_metrics_per_case_concat = df_mean_metrics_per_case_concat.rename(
        columns={col: f'Mean {col}' for col in df_mean_metrics_per_case_concat.columns}
    )

    # Currently, the index has the format [f"CASE_ID = {CASE_ID}, {SOLUTION_TYPE}"]
    # Reset the index so that CASE_ID is a column and drop SOLUTION_TYPE
    df_mean_metrics_per_case_concat = df_mean_metrics_per_case_concat.reset_index(col_level=[0, 1])
    df_mean_metrics_per_case_concat = df_mean_metrics_per_case_concat.rename(
        columns={'level_0': 'CASE_ID'}
    ).drop(columns=['solution_type'])

    # Reformat the values in CASE_ID column to include just numbers
    df_mean_metrics_per_case_concat['CASE_ID'] = df_mean_metrics_per_case_concat['CASE_ID'].apply(
        lambda x: int(x.split(' = ')[1])
    )

    # Add case descriptions
    df_mean_metrics_per_case_concat['Case description'] = df_mean_metrics_per_case_concat[
        'CASE_ID'
    ].map(CASE_DESCRIPTION)

    # Reorder columns to have CASE_ID first, Case description second, and metrics after
    df_mean_metrics_per_case_concat = df_mean_metrics_per_case_concat[
        ['CASE_ID', 'Case description']
        + [
            col
            for col in df_mean_metrics_per_case_concat.columns
            if col not in ['CASE_ID', 'Case description']
        ]
    ]

    return df_mean_metrics_per_case_concat


def get_best_solution_type_per_group(
    average_metrics: dict,
    group_identifier: str = 'TIER_ID',
    metric: str | None = DEFAULT_METRIC,
    aggregation_func: Literal['mean', 'median', None] = None,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Identify the best solution type per group based on the chosen performance metric.

    Parameters
    ----------
    average_metrics : dict
        A dictionary where keys are group identifiers and values are DataFrames
        containing average performance metrics per solution type for that group.
    group_identifier : str, optional
        Column name to group by (e.g., 'TIER_ID' or 'CASE_ID'), by default 'TIER_ID'.
    metric : str, optional
        Performance metric to use for determining the best solution type,
        DEFAULT_METRIC by default.
    aggregation_func : Literal["mean", "median", None], optional
        Aggregation function used for averaging metrics, by default None.
        Has no effect on the function's operation, used only when verbose is True.
    verbose : bool, optional
        If True, display detailed information, by default False.

    Returns
    -------
    Dict[str, str]
        A dictionary where keys are group identifiers and values are the best
        solution type based on the chosen performance metric.
    """
    sol_type_per_group = {}
    for group, metrics in average_metrics.items():
        sol_type_per_group[group] = metrics[metric].idxmax()

    if verbose:
        aggregation_func_display = aggregation_func if aggregation_func else ''
        display(
            Markdown(
                f'## Best solution types per {group_identifier} based on {aggregation_func_display} {metric}'
            )
        )
        for group, sol_type in sol_type_per_group.items():
            if group_identifier == 'CASE_ID':
                i = int(group.split(' = ')[1])
                display(
                    Markdown(
                        f'- **{group}:** {sol_type} ({case2set(i).upper()}: {CASE_DESCRIPTION[i]})'
                    )
                )
            else:
                display(Markdown(f'- **{group}:** {sol_type}'))

    return sol_type_per_group


def choose_best_solution_per_group(
    df: pd.DataFrame,
    group_identifier: str = 'TIER_ID',
    metric: str | None = DEFAULT_METRIC,
    aggregation_func: Literal['mean', 'median'] = 'mean',
    verbose: bool = False,
) -> dict[str, str]:
    """
    Wrapper function to get the best solution types per group (e.g., 'TIER_ID' or 'CASE_ID')
    based on a chosen performance metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results with performance metrics.
    group_identifier : str, optional
        Column name to group by (e.g., 'TIER_ID' or 'CASE_ID'), by default 'TIER_ID'.
    metric : str, optional
        Performance metric to use for determining the best solution type,
        DEFAULT_METRIC by default.
    aggregation_func : Literal["mean", "median"], optional
        Aggregation function to use for averaging metrics, by default 'mean'.
    verbose : bool, optional
        If True, display detailed information, by default False.

    Returns
    -------
    Dict[str, str]
        A dictionary where keys are group identifiers and values are the best
        solution type based on the chosen performance metric.
    """
    average_metrics = get_average_metrics_per_group(
        df,
        group_identifier=group_identifier,
        aggregation_func=aggregation_func,
        verbose=verbose,
    )

    best_solution_per_group = get_best_solution_type_per_group(
        average_metrics,
        group_identifier=group_identifier,
        metric=metric,
        verbose=verbose,
    )
    return best_solution_per_group


def filter_df_best_solutions(
    df: pd.DataFrame,
    best_solutions: dict[str, str],
    group_identifier: str = 'TIER_ID',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter the DataFrame to include only the best solution types per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results with performance metrics.
    best_solutions : Dict[str, str]
        A dictionary where keys are group identifiers and values are the best
        solution type based on the chosen performance metric.
    group_identifier : str, optional
        Column name to group by (e.g., 'TIER_ID' or 'CASE_ID'), by default 'TIER_ID'.
    verbose : bool, optional
        If True, display information of the filtered DataFrame, by default False.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only the best solution types per group.
    """
    df_filtered = pd.DataFrame()
    for group, sol_type in best_solutions.items():
        group_value = group.split(' = ')[1]
        df_group = df[
            (df[group_identifier] == int(group_value)) & (df['solution_type'] == sol_type)
        ]
        df_filtered = pd.concat([df_filtered, df_group], ignore_index=True)

    if verbose:
        display(Markdown(f'### Filtered data with best solutions per **{group_identifier}**'))
        display(Markdown(f'- Total number of records: {len(df_filtered)}'))
        display(Markdown('- Solution types included:'))
        display(pd.Series(best_solutions.values()).value_counts().to_frame())

        display(
            Markdown(
                f'- Number of **{group_identifier}**s: {df_filtered[group_identifier].nunique()}'
            )
        )

        display(Markdown(f'- Number of **experiments per {group_identifier}**:'))
        exp_count = (
            df_filtered[group_identifier]
            .value_counts()
            .to_frame()
            .rename(columns={'count': 'number of experiments'})
        )
        # add case descriptions if grouping by CASE_ID
        if group_identifier == 'CASE_ID':
            exp_count['CASE_DESCRIPTION'] = exp_count.index.to_list()
            exp_count['CASE_DESCRIPTION'] = exp_count['CASE_DESCRIPTION'].apply(
                lambda x: CASE_DESCRIPTION.get(x, 'N/A')
            )
            # order columns
            exp_count = exp_count[['CASE_DESCRIPTION', 'number of experiments']]

        display(exp_count.sort_index(ascending=True))
    return df_filtered


def analyze_metric_results(
    df: pd.DataFrame,
    group_identifier: Literal['TIER_ID', 'CASE_ID'] | None,
    identifiers_list: list[int] | list[str] | None = None,
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
    thresholds: dict[str, float] | None = None,
    verbose: bool | None = True,
) -> pd.DataFrame:
    """
    Analyze the distribution of a specified metric for a given solution type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the results data.
    group_identifier : Literal["TIER_ID", "CASE_ID", None]
        The column name used to group the data, or None to use all data.
    identifiers_list : Optional[List[str]] = None
        The unique identifier values to filter the data (e.g., tier IDs).
        If None, all unique identifiers are used. Ignored if group_identifier is None.
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
        One of the coverage metrics to be analyze (e.g., "Recall", "Precision").
        DEFAULT_METRIC by default.
    thresholds : Optional[Dict[str, float]]
        Dictionary defining the lower bounds for performance categories.
        Defaults to THRESHOLDS_FOR_METRIC for the given metric.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with counts of experiments in each performance category
        ("Excellent", "Good", "Moderate", "Poor", "Unknown").
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

    # Categorize performance
    def categorize_metric_higher_is_better(val: float | int | None) -> str:
        if pd.isna(val):
            return 'Unknown'
        if val >= thresholds['Excellent']:
            return 'Excellent'
        if val >= thresholds['Good']:
            return 'Good'
        if val >= thresholds['Moderate']:
            return 'Moderate'
        return 'Poor'

    def categorize_metric_lower_is_better(val: float | int | None) -> str:
        if pd.isna(val):
            return 'Unknown'
        if val <= thresholds['Excellent']:
            return 'Excellent'
        if val <= thresholds['Good']:
            return 'Good'
        if val <= thresholds['Moderate']:
            return 'Moderate'
        return 'Poor'

    # Get default thresholds, if defined
    if thresholds is None:
        thresholds = THRESHOLDS_FOR_METRIC.get(metric_name, None)
    if thresholds is None:
        print(f'No thresholds defined for {metric_name}. Skipping analysis.')
        return

    # Categorize based on whether higher or lower values are better
    if metric_name in [
        'Recall',
        'Precision',
        'F1_Score',
        'Success_Index',
        'Adjusted_Success_Index',
    ]:
        categories = df_plot[metric_name].apply(categorize_metric_higher_is_better)
    else:
        categories = df_plot[metric_name].apply(categorize_metric_lower_is_better)

    # Count occurrences in each category
    counts = categories.value_counts()

    # Ensure all categories are present for consistent plotting
    df_category_counts = pd.DataFrame(
        {
            'Category': CATEGORY_ORDER,
            'Count': [counts.get(cat, 0) for cat in CATEGORY_ORDER],
        }
    )

    # Print summary stats
    if verbose:
        display(Markdown('#### Statistics:'))
        display(Markdown(f'- **Total experiments:** {len(df_plot)}'))
        display(Markdown(f'- **Mean {metric_name}:** {df_plot[metric_name].mean():.3f}'))
        display(Markdown(f'- **Median {metric_name}:** {df_plot[metric_name].median():.3f}\n'))
    return df_category_counts


def compute_performance_overview(
    df_cases: pd.DataFrame,
    select_metrics: list[str],
) -> pd.DataFrame:
    df_performance_overview = pd.DataFrame()

    for select_metric in select_metrics:
        df_performance_overview_metric = {}

        for c in CASE_DESCRIPTION.keys():
            result_df = analyze_metric_results(
                df_cases,
                group_identifier='CASE_ID',
                identifiers_list=[c],
                metric_name=select_metric,
                verbose=False,
            )

            # Extract the count column (or whichever column you want)
            # Assuming the DataFrame has a 'Count' column - adjust column name as needed
            df_performance_overview_metric[f'CASE_ID = {c}'] = result_df.set_index('Category')[
                'Count'
            ]

        # Convert to DataFrame where rows are cases and columns are categories
        df_performance_overview_metric = pd.DataFrame(df_performance_overview_metric).T
        df_performance_overview_metric = df_performance_overview_metric.rename(
            columns={
                col: col + f' {select_metric}' for col in df_performance_overview_metric.columns
            }
        )

        # Concatenate to the main performance overview dataframe
        if df_performance_overview.empty:
            df_performance_overview = df_performance_overview_metric
        else:
            df_performance_overview = pd.concat(
                [df_performance_overview, df_performance_overview_metric], axis=1
            )

    # Drop all the "Unknown ..." columns
    df_performance_overview = df_performance_overview.drop(
        columns=[col for col in df_performance_overview.columns if 'Unknown' in col]
    )

    # add case descriptions as the first column
    df_performance_overview.insert(
        0,
        'Case description',
        [CASE_DESCRIPTION[c] for c in range(1, len(df_performance_overview.index) + 1)],
    )
    return df_performance_overview


def show_performance_overview(
    df_performance_overview: pd.DataFrame,
    select_metrics: list[str],
) -> None:
    # Show each metric block separately for better readability
    for select_metric in select_metrics:
        metric_cols = [
            col
            for col in df_performance_overview.columns
            if ((select_metric in col) & ('Mean' not in col) & ('Median' not in col))
        ]
        display(Markdown(f'### **Performance overview:** {select_metric}'))
        display(df_performance_overview[['Case description'] + metric_cols])
    return
