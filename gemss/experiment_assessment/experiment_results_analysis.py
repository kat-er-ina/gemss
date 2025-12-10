import os
from typing import Dict, List, Literal, Tuple, Optional
from IPython.display import display, Markdown
import pandas as pd

from gemss.config.constants import EXPERIMENT_RESULTS_DIR

# Identify metric columns (those containing the base name of coverage metrics)
# List synchronized with keys returned by calculate_coverage_metrics in run_experiment.py
# All coverage metrics are numeric (possibly None)
COVERAGE_METRICS = [
    "Recall",
    "Precision",
    "F1_Score",
    "Jaccard",
    "Miss_Rate",
    "FDR",
    "Global_Miss_Rate",
    "Global_FDR",
    "Success_Index",
    "Adjusted_Success_Index",
]

IMPORTANT_METRICS = [
    "F1_Score",
    "Recall",
    "Precision",
    "Adjusted_Success_Index",
    "Success_Index",
]

DEFAULT_METRIC = "F1_Score"

SOLUTION_OPTIONS = [
    "full",
    "top",
    "outlier_STD_2.0",
    "outlier_STD_2.5",
    "outlier_STD_3.0",
]

DEFAULT_SOLUTION = "top"

DEFAULT_AGGREGATION_FUNC = "median"

ALL_PARAMETERS = [
    "N_SAMPLES",
    "N_FEATURES",
    "SAMPLE_VS_FEATURE_RATIO",
    "SPARSITY",
    "N_GENERATING_SOLUTIONS",
    "N_CANDIDATE_SOLUTIONS",
    "NOISE_STD",
    "NAN_RATIO",
    "[NOISE_STD, NAN_RATIO] COMBINATION",
    "LAMBDA_JACCARD",
    "BINARY_RESPONSE_RATIO",
    "BINARIZE",
]

# ASI vs SI comparison thresholds
DEFAULT_ASI_SI_COMPARISON_THRESHOLDS = {
    "Excellent": 0.85,
    "Good": 0.5,
    "Moderate": 0.2,
}
# anything below 'Moderate' is 'Poor'
DEFAULT_RECALL_THRESHOLDS = {
    "Excellent": 0.9,
    "Good": 0.8,
    "Moderate": 0.65,
}
DEFAULT_PRECISION_THRESHOLDS = {
    "Excellent": 0.9,
    "Good": 0.7,
    "Moderate": 0.5,
}
DEFAULT_F1SCORE_THRESHOLDS = {
    "Excellent": 0.85,  # corresponds to Precision = 0.8, Recall = 0.9
    "Good": 0.71,  # average between moderate and excellent
    "Moderate": 0.565,  # corresponds to Precision = 0.5, Recall = 0.65
}
THRESHOLDS_FOR_METRIC = {
    "Recall": DEFAULT_RECALL_THRESHOLDS,
    "Precision": DEFAULT_PRECISION_THRESHOLDS,
    "F1_Score": DEFAULT_F1SCORE_THRESHOLDS,
    "Success_Index": None,
    "Adjusted_Success_Index": None,
}
################################################################################################


def load_experiment_results(
    tier_id_list: List[int] = [1, 2, 3, 4, 5, 6, 7],
    results_dir: str = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
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
    for tier_id in tier_id_list:
        if results_dir is None:
            results_dir = EXPERIMENT_RESULTS_DIR

        results_path = results_dir / f"tier{tier_id}" / "tier_summary_metrics.csv"

        if os.path.exists(results_path):
            df_tier = pd.read_csv(results_path)
            if verbose:
                display(
                    Markdown(
                        f"Successfully loaded {len(df_tier)} experiment records from **Tier {tier_id}**."
                    )
                )
            # Ensure numeric columns are actually numeric
            metric_cols = [
                c for c in df_tier.columns if any(x in c for x in COVERAGE_METRICS)
            ]
            for col in metric_cols:
                if col in df_tier.columns:
                    df_tier[col] = pd.to_numeric(df_tier[col], errors="coerce")

            # Add TIER_ID column
            df_tier["TIER_ID"] = int(tier_id)

            # Add EXPERIMENT_ID column: {tier_id}.{experiment_number_in_tier}
            df_tier["EXPERIMENT_ID"] = (
                str(tier_id) + "." + (df_tier.index + 1).astype(str)
            )

        else:
            if verbose:
                display(Markdown(f"**ERROR:** File not found at {results_path}"))
                display(
                    Markdown(
                        "Please run the experiments for this tier first, or check the path."
                    )
                )
            df_tier = pd.DataFrame()

        # Append to main DataFrame
        df = pd.concat([df, df_tier], ignore_index=True)

    # Add the "SAMPLE_VS_FEATURE_RATIO" column
    df["SAMPLE_VS_FEATURE_RATIO"] = df["N_SAMPLES"] / df["N_FEATURES"]
    # Add a feature that combines information about noise and missingness
    df["[NOISE_STD, NAN_RATIO] COMBINATION"] = (
        "[" + df["NOISE_STD"].astype(str) + ", " + df["NAN_RATIO"].astype(str) + "]"
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
    tier_id_list = df["TIER_ID"].unique().tolist()
    display(Markdown(f"### All results for tiers: {tier_id_list}"))

    if "solution_type" in df.columns:
        total_exp_count = len(df) // df["solution_type"].nunique()
    else:
        total_exp_count = len(df)
    display(Markdown(f"- **Total experiments:** {total_exp_count}"))

    display(
        Markdown(
            f"- **{len(SOLUTION_OPTIONS)} solution types:** {', '.join(SOLUTION_OPTIONS)}"
        )
    )
    display(
        Markdown(
            f"- **{len(COVERAGE_METRICS)} available metrics:** {', '.join(COVERAGE_METRICS)}"
        )
    )
    varied_params = [
        p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1
    ]
    display(
        Markdown(
            f"- **Varied parameters in this analysis:** {', '.join(varied_params)}"
        )
    )
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
    varied_params = [
        p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1
    ]
    df_pivot = pd.DataFrame()
    for solution in SOLUTION_OPTIONS:
        solution_cols = [col for col in df.columns if solution in col]
        df_solution = df[["TIER_ID"] + varied_params + solution_cols].copy()
        df_solution["TIER_ID"] = df_solution["TIER_ID"].astype(str)
        df_solution.rename(
            columns={col: col.replace(f"{solution}_", "") for col in solution_cols},
            inplace=True,
        )
        df_solution["solution_type"] = solution
        df_pivot = pd.concat([df_pivot, df_solution], ignore_index=True)
    return df_pivot


def get_all_experiment_results(
    tier_id_list: List[int] = [1, 2, 3, 4, 5, 6, 7],
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
    group_identifier: Optional[str] = "TIER_ID",
    aggregation_func: Literal["mean", "median"] = "mean",
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate mean or median performance metrics per group (e.g., per tier or test case).

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
    if aggregation_func == "mean":
        agg_func = pd.Series.mean
    elif aggregation_func == "median":
        agg_func = pd.Series.median
    average_metrics = {}
    all_groups = df[group_identifier].unique().tolist()
    second_metric = "Recall" if DEFAULT_METRIC == "F1_Score" else "F1_Score"
    for group in all_groups:
        df_group = df[df[group_identifier] == group]
        df_group_avg = (
            df_group.groupby("solution_type")[IMPORTANT_METRICS]
            .agg(agg_func)
            .sort_values(by=[DEFAULT_METRIC, second_metric], ascending=False)
            .round(3)
        )
        average_metrics[f"{group_identifier} = {group}"] = df_group_avg

        if verbose:
            display(
                Markdown(
                    f"### {aggregation_func.capitalize()} performance metrics for {group_identifier} = {group}"
                )
            )
            display(df_group_avg)
    return average_metrics


def get_best_solution_type_per_group(
    average_metrics: dict,
    group_identifier: str = "TIER_ID",
    metric: Optional[str] = DEFAULT_METRIC,
    aggregation_func: Literal["mean", "median", None] = None,
    verbose: bool = False,
) -> Dict[str, str]:
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
        aggregation_func_display = aggregation_func if aggregation_func else ""
        display(
            Markdown(
                f"## Best solution types per {group_identifier} based on {aggregation_func_display} {metric}"
            )
        )
        for group, sol_type in sol_type_per_group.items():
            display(Markdown(f"- **{group}:** {sol_type}"))

    return sol_type_per_group


def choose_best_solution_per_group(
    df: pd.DataFrame,
    group_identifier: str = "TIER_ID",
    metric: Optional[str] = DEFAULT_METRIC,
    aggregation_func: Literal["mean", "median"] = "mean",
    verbose: bool = False,
) -> dict:
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
    dict
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
