"""
Contains functions to perform  analysis of experiments per each test case.
"""

from typing import Dict, Tuple
import pandas as pd

BASELINE_NOISE_LEVEL = 0.1
BASELINE_NAN_RATIO = 0.0
DEFAULT_SPARSITY = 5

CASE_DESCRIPTION = {}
# Descriptions for classification experiment cases
CASE_DESCRIPTION[1] = "Basic performance in small dimensional scenario: SPARSITY = 3"
CASE_DESCRIPTION[2] = "Basic performance in small dimensional scenario: SPARSITY = 5"
CASE_DESCRIPTION[3] = (
    "Comparison of performance for different sparsity levels in Tier 1"
)
CASE_DESCRIPTION[4] = "Scalability analysis: p = 1000"
CASE_DESCRIPTION[5] = "Scalability analysis: p = 2000"
CASE_DESCRIPTION[6] = "Scalability analysis: p = 5000"
CASE_DESCRIPTION[7] = "Scalability analysis: n = 50"
CASE_DESCRIPTION[8] = "Scalability analysis: n = 100"
CASE_DESCRIPTION[9] = "Scalability analysis: n = 200"
CASE_DESCRIPTION[10] = "Performance in Tier 3 only (SPARSITY = 3)"
CASE_DESCRIPTION[11] = "Performance in Tier 3 only (SPARSITY = 5)"
CASE_DESCRIPTION[12] = "Overall performance in Tier 3"
CASE_DESCRIPTION[13] = "Comparison of Tier 1 vs Tier 3 (SPARSITY = 3)"
CASE_DESCRIPTION[14] = "Comparison of Tier 1 vs Tier 3 (SPARSITY = 5)"
CASE_DESCRIPTION[15] = "Overall performance in Tiers 1 + 2 + 3"
CASE_DESCRIPTION[16] = "Robustness under adversity: [n=100, p=200], only varying noise"
CASE_DESCRIPTION[17] = "Robustness under adversity: [n=100, p=200], only varying NaNs"
CASE_DESCRIPTION[18] = (
    "Robustness under adversity: [n=100, p=200], varying both noise and NaNs"
)
CASE_DESCRIPTION[19] = "Robustness under adversity: [n=200, p=500], only varying noise"
CASE_DESCRIPTION[20] = "Robustness under adversity: [n=200, p=500], only varying NaNs"
CASE_DESCRIPTION[21] = (
    "Robustness under adversity: [n=200, p=500], varying both noise and NaNs"
)
CASE_DESCRIPTION[22] = (
    "Robustness under adversity: compare [n=100, p=200] and [n=200, p=500]"
)
CASE_DESCRIPTION[23] = (
    "Effect of Jaccard penalty for [n=100, p=200], SPARSITY=3 (Tiers 1 + 5)"
)
CASE_DESCRIPTION[24] = (
    "Effect of Jaccard penalty for [n=100, p=200], SPARSITY=5 (Tiers 1 + 5)"
)
CASE_DESCRIPTION[25] = (
    "Effect of Jaccard penalty for [n=100, p=500], SPARSITY=3 (Tiers 1 + 5)"
)
CASE_DESCRIPTION[26] = (
    "Effect of Jaccard penalty for [n=100, p=500], SPARSITY=5 (Tiers 1 + 5)"
)
CASE_DESCRIPTION[27] = "Overall effect of Jaccard penalty (Tiers 1 + 5)"
CASE_DESCRIPTION[28] = "Effect of class imbalance for [n=100, p=200] (Tiers 1 + 7)"
CASE_DESCRIPTION[29] = "Effect of class imbalance for [n=200, p=500] (Tiers 1 + 7)"
CASE_DESCRIPTION[30] = "Overall effect of class imbalance (Tiers 1 + 7)"
# Descriptions for regression experiment cases
CASE_DESCRIPTION[31] = "Regression: basic performance (Tier 6 only)"
CASE_DESCRIPTION[32] = "Same as CASE 31 but for all [n, p] combinations"
CASE_DESCRIPTION[33] = "Regression: high-dim performance: p = 1000 (Tier 6 only)"
CASE_DESCRIPTION[34] = "Regression: high-dim performance: p = 2000 (Tier 6 only)"
CASE_DESCRIPTION[35] = "Regression: high-dim performance: p = 5000 (Tier 6 only)"
CASE_DESCRIPTION[36] = "Regression: high-dim performance: n = 50 (Tier 6 only)"
CASE_DESCRIPTION[37] = "Regression: high-dim performance: n = 100 (Tier 6 only)"
CASE_DESCRIPTION[38] = "Regression: high-dim performance: n = 200 (Tier 6 only)"
CASE_DESCRIPTION[39] = "Regression: effect of varying noise (Tier 6 only)"
CASE_DESCRIPTION[40] = "Regression: effect of varying Nans (Tier 6 only)"
CASE_DESCRIPTION[41] = "Regression: effect of varying both noise and Nans (Tier 6 only)"
# Descriptions for regression vs. classification experiment cases
CASE_DESCRIPTION[42] = "Regression vs. classification: basic scenarios (Tiers 1 and 6)"
CASE_DESCRIPTION[43] = (
    "Regression vs. classification: high-dimensional scenarios (Tiers 2 and 6)"
)
CASE_DESCRIPTION[44] = "Regression vs. classification: effect of noise (Tiers 3 + 6)"
CASE_DESCRIPTION[45] = "Regression vs. classification: effect of NaNs (Tiers 3 + 6)"
CASE_DESCRIPTION[46] = (
    "Regression vs. classification: effect of both noise and NaNs (Tiers 3 + 6)"
)


def get_df_cases(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Splits the main dataframe into multiple dataframes corresponding to different test cases.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe containing all experiment results.

    Returns
    -------
    Dict[int, pd.DataFrame]
        A dictionary mapping case numbers to their corresponding filtered dataframes.
    """
    df_cases = {}

    # CASE 01: Basic performance in small dimensional scenario: SPARSITY = 3
    # CASE 02: Basic performance in small dimensional scenario: SPARSITY = 5
    for c, sp in zip([1, 2], [3, 5]):
        df_cases[c] = df[(df["TIER_ID"] == 1) & (df["SPARSITY"] == sp)]

    # CASE 03: Comparison of performance for different sparsity levels in Tier 1
    df_cases[3] = df[(df["TIER_ID"] == 1)]

    # CASE 04: Scalability analysis: p = 1000
    # CASE 05: Scalability analysis: p = 2000
    # CASE 06: Scalability analysis: p = 5000
    for c, p in zip([4, 5, 6], [1000, 2000, 5000]):
        df_cases[c] = df[
            (df["TIER_ID"] == 2)
            & (df["SPARSITY"] == DEFAULT_SPARSITY)
            & (df["N_FEATURES"] == p)
        ]

    # CASE 07: Scalability analysis: n = 50
    # CASE 08: Scalability analysis: n = 100
    # CASE 09: Scalability analysis: n = 200
    for c, n in zip([7, 8, 9], [50, 100, 200]):
        df_cases[c] = df[
            (df["TIER_ID"].isin([1, 2, 3]))
            & (df["SPARSITY"] == DEFAULT_SPARSITY)
            & (df["N_SAMPLES"] == n)
        ]

    # CASE 10: Performance in Tier 3 only (SPARSITY = 3)
    # CASE 11: Performance in Tier 3 only (SPARSITY = 5)
    for c, sp in zip([10, 11], [3, 5]):
        df_cases[c] = df[(df["TIER_ID"] == 3) & (df["SPARSITY"] == sp)]

    # CASE 12: Overall performance in Tier 3
    df_cases[12] = df[(df["TIER_ID"] == 3)]

    # CASE 13: Comparison of Tier 1 vs Tier 3 (SPARSITY = 3)
    # CASE 14: Comparison of Tier 1 vs Tier 3 (SPARSITY = 5)
    for c, sp in zip([13, 14], [3, 5]):
        df_cases[c] = df[(df["TIER_ID"].isin([1, 3])) & (df["SPARSITY"] == sp)]

    # CASE 15: Overall performance in Tiers 1 + 2 + 3
    df_cases[15] = df[(df["TIER_ID"].isin([1, 2, 3]))]

    # CASE 16: Robustness under adversity: [n=100, p=200], only varying noise
    # CASE 17: Robustness under adversity: [n=100, p=200], only varying NaNs
    # CASE 18: Robustness under adversity: [n=100, p=200], varying both noise and NaNs
    # CASE 19: Robustness under adversity: [n=200, p=500], only varying noise
    # CASE 20: Robustness under adversity: [n=200, p=500], only varying NaNs
    # CASE 21: Robustness under adversity: [n=200, p=500], varying both noise and NaNs
    for c, n, p in zip(
        [16, 19],
        [100, 200],
        [200, 500],
    ):
        dff = df[
            (df["TIER_ID"].isin([1, 4]))
            & (df["SPARSITY"] == DEFAULT_SPARSITY)
            & (df["N_SAMPLES"] == n)
            & (df["N_FEATURES"] == p)
        ]
        df_cases[c] = dff[dff["NAN_RATIO"] == BASELINE_NAN_RATIO]
        df_cases[c + 1] = dff[dff["NOISE_STD"] == BASELINE_NOISE_LEVEL]
        df_cases[c + 2] = dff

    # CASE 22: Robustness under adversity: compare [n=100, p=200] and [n=200, p=500]
    df_cases[22] = dff = df[
        (df["TIER_ID"].isin([1, 4]))
        & (df["SPARSITY"] == DEFAULT_SPARSITY)
        & (
            (df["N_SAMPLES"] == 100) & (df["N_FEATURES"] == 200)
            | (df["N_SAMPLES"] == 200) & (df["N_FEATURES"] == 500)
        )
    ]

    # CASE 23: Effect of Jaccard penalty for [n=100, p=200], SPARSITY=3 (Tiers 1 + 5)
    # CASE 24: Effect of Jaccard penalty for [n=100, p=200], SPARSITY=5 (Tiers 1 + 5)
    # CASE 25: Effect of Jaccard penalty for [n=100, p=500], SPARSITY=3 (Tiers 1 + 5)
    # CASE 26: Effect of Jaccard penalty for [n=100, p=500], SPARSITY=5 (Tiers 1 + 5)
    for c, sp, p in zip(
        [23, 24, 25, 26],
        [3, 5, 3, 5],
        [200, 200, 500, 500],
    ):
        df_cases[c] = df[
            (df["TIER_ID"].isin([1, 5]))
            & (df["SPARSITY"] == sp)
            & (df["N_SAMPLES"] == 100)
            & (df["N_FEATURES"] == p)
        ]

    # CASE 27: Overall effect of Jaccard penalty (Tiers 1 + 5)
    df_cases[27] = df[
        (df["TIER_ID"].isin([1, 5]))
        & (df["N_SAMPLES"] == 100)
        & (df["N_FEATURES"].isin([200, 500]))
    ]

    # CASE 28: Effect of class imbalance for [n=100, p=200] (Tiers 1 + 7)
    # CASE 29: Effect of class imbalance for [n=200, p=500] (Tiers 1 + 7)
    for c, n, p in zip(
        [28, 29],
        [100, 200],
        [200, 500],
    ):
        df_cases[c] = df[
            (df["TIER_ID"].isin([1, 7]))
            & (df["SPARSITY"] == DEFAULT_SPARSITY)
            & (df["N_SAMPLES"] == n)
            & (df["N_FEATURES"] == p)
        ]
    # CASE 30: Overall effect of class imbalance (Tiers 1 + 7)
    df_cases[30] = df[
        (df["TIER_ID"].isin([1, 7]))
        & (df["SPARSITY"] == DEFAULT_SPARSITY)
        & (
            (df["N_SAMPLES"] == 100) & (df["N_FEATURES"] == 200)
            | (df["N_SAMPLES"] == 200) & (df["N_FEATURES"] == 500)
        )
    ]

    ###############################################################################################
    #    REGRESSION EXPERIMENTS
    ###############################################################################################

    df_six = df[df["TIER_ID"] == 6]

    # CASE 31: Regression: basic performance (Tier 6 only)
    # - baseline noise level, no NaNs, p <= 500
    # - all regression experiments have default SPARSITY = 5
    df_cases[31] = df_six[
        (df_six["NOISE_STD"] == BASELINE_NOISE_LEVEL)
        & (df_six["NAN_RATIO"] == BASELINE_NAN_RATIO)
        & (df_six["N_FEATURES"] < 501)  # only p <= 500
    ]

    # CASE 32: Same as CASE 31 but for all [n, p] combinations
    df_cases[32] = df_six[
        (df_six["NOISE_STD"] == BASELINE_NOISE_LEVEL)
        & (df_six["NAN_RATIO"] == BASELINE_NAN_RATIO)
    ]

    # CASE 33: Regression: high-dim performance: p = 1000 (Tier 6 only)
    # CASE 34: Regression: high-dim performance: p = 2000 (Tier 6 only)
    # CASE 35: Regression: high-dim performance: p = 5000 (Tier 6 only)
    for c, p in zip([33, 34, 35], [1000, 2000, 5000]):
        df_cases[c] = df_six[
            (df_six["N_FEATURES"] == p)
            & (df_six["NOISE_STD"] == BASELINE_NOISE_LEVEL)
            & (df_six["NAN_RATIO"] == BASELINE_NAN_RATIO)
        ]

    # CASE 36: Regression: high-dim performance: n = 50 (Tier 6 only)
    # CASE 37: Regression: high-dim performance: n = 100 (Tier 6 only)
    # CASE 38: Regression: high-dim performance: n = 200 (Tier 6 only)
    for c, n in zip([36, 37, 38], [50, 100, 200]):
        df_cases[c] = df_six[
            (df_six["N_SAMPLES"] == n)
            & (df_six["NOISE_STD"] == BASELINE_NOISE_LEVEL)
            & (df_six["NAN_RATIO"] == BASELINE_NAN_RATIO)
        ]

    # CASE 39: Regression: effect of varying noise (Tier 6 only)
    # CASE 40: Regression: effect of varying Nans (Tier 6 only)
    # CASE 41: Regression: effect of varying both noise and Nans (Tier 6 only)
    dff = df_six[(df_six["N_SAMPLES"] == 100) & (df_six["N_FEATURES"] == 200)]
    df_cases[39] = dff[dff["NAN_RATIO"] == BASELINE_NAN_RATIO]
    df_cases[40] = dff[dff["NOISE_STD"] == BASELINE_NOISE_LEVEL]
    df_cases[41] = dff

    ##############################################################################
    #    REGRESSION VS. CLASSIFICATION EXPERIMENTS
    ##############################################################################

    df_comp = df[
        (df["TIER_ID"].isin([1, 2, 4, 6])) & (df["SPARSITY"] == DEFAULT_SPARSITY)
    ]

    # CASE 42: Regression vs. classification: basic scenarios (Tiers 1 and 6)
    df_cases[42] = df_comp[
        (  # baseline noise level, no NaNs
            (df_comp["NOISE_STD"] == BASELINE_NOISE_LEVEL)
            & (df_comp["NAN_RATIO"] == BASELINE_NAN_RATIO)
            & (df_comp["N_FEATURES"] < 501)  # only p <= 500
        )
    ]
    # CASE 43: Regression vs. classification: high-dimensional scenarios (Tiers 2 and 6)
    df_cases[43] = df_comp[
        (  # baseline noise level, no NaNs
            (df_comp["NOISE_STD"] == BASELINE_NOISE_LEVEL)
            & (df_comp["NAN_RATIO"] == BASELINE_NAN_RATIO)
            & (df_comp["N_FEATURES"] >= 1000)  # only p >= 1000
        )
    ]
    # CASE 44: Regression vs. classification: effect of noise (Tiers 3 + 6)
    # CASE 45: Regression vs. classification: effect of NaNs (Tiers 3 + 6)
    # CASE 46: Regression vs. classification: effect of both noise and NaNs (Tiers 3 + 6)
    dff = df_comp[(df_comp["N_SAMPLES"] == 100) & (df_comp["N_FEATURES"] == 200)]
    df_cases[44] = dff[dff["NAN_RATIO"] == BASELINE_NAN_RATIO]
    df_cases[45] = dff[dff["NOISE_STD"] == BASELINE_NOISE_LEVEL]
    df_cases[46] = dff

    return df_cases


def concatenate_cases(df_cases: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all case dataframes into a single dataframe with an additional column 'CASE_ID'.

    Parameters
    ----------
    df_cases : Dict[int, pd.DataFrame]
        A dictionary where keys are case IDs and values are dataframes for each case.

    Returns
    -------
    pd.DataFrame
        A single dataframe containing all cases with an additional 'CASE_ID' column.
    """

    df_all_cases = pd.DataFrame()
    for case_id, df_case in df_cases.items():
        df_case = df_case.copy()
        df_case["CASE_ID"] = case_id
        df_all_cases = pd.concat([df_all_cases, df_case], ignore_index=True)
    return df_all_cases
