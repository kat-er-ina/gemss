import os
from typing import List, Tuple
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

SOLUTION_OPTIONS = [
    "full",
    "top",
    "outlier_STD_2.0",
    "outlier_STD_2.5",
    "outlier_STD_3.0",
]

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
]

HYPERPARAMETERS = [
    "BATCH_SIZE",
    "BINARIZE",
]


def load_experiment_results(
    tier_id_list: List[int] = [1, 2, 3, 4, 5, 6, 7],
    results_dir: str = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
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
        - A list of metric column names.
        - A list of parameters that vary among experiments in the dataset.
        - A list of parameters that do not vary among experiments in the dataset.
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

    # Identify which parameters actually vary in this dataset
    varied_params = [
        p for p in ALL_PARAMETERS if p in df.columns and df[p].nunique() > 1
    ]
    unvaried_params = [
        p for p in ALL_PARAMETERS if p in df.columns and p not in varied_params
    ]
    return (df, metric_cols, varied_params, unvaried_params)


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
    display(Markdown(f"- **Total experiments:** {len(df)}"))
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
    metric_cols = [c for c in df.columns if any(x in c for x in COVERAGE_METRICS)]
    display(Markdown(f"- **Total metrics columns:** {len(metric_cols)}"))

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
