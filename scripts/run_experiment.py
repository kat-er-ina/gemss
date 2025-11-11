"""
Run GEMSS (Gaussian Ensemble for Multiple Sparse Solutions) as a script.

This script provides a complete experiment pipeline including:
- Dataset generation with configurable parameters
- GEMSS optimization for sparse feature selection
- Solution recovery and analysis
- Performance diagnostics
- Comprehensive result reporting

Assumes:
- gemss package is installed and available
- gemss.config is properly set up with experiment parameters

Outputs:
- A text file with comprehensive experiment results in the "results" directory
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional

import gemss.config as C
from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset
from gemss.feature_selection.inference import BayesianFeatureSelector
from gemss.diagnostics.result_postprocessing import (
    recover_solutions,
    get_features_from_solutions,
    get_unique_features,
)
from gemss.diagnostics.performance_tests import run_performance_diagnostics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the GEMSS experiment script.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - output: Optional custom output filename
        - diagnostics: Boolean flag to run performance diagnostics
    """
    parser = argparse.ArgumentParser(
        description="Run GEMSS experiment for sparse feature selection."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom filename for output summary (should end with .txt)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run performance diagnostics (default: False)",
    )
    return parser.parse_args()


def generate_dataset() -> Tuple[pd.DataFrame, Any, Dict, Dict, List[str]]:
    """
    Generate artificial dataset for the experiment using configuration parameters.

    Returns
    -------
    tuple
        A 5-tuple containing:
        - pd.DataFrame: Feature matrix with shape (n_samples, n_features)
        - array-like: Target variable (binary or continuous)
        - dict: Generating solutions used to create the data
        - dict: Dataset parameters including support indices and coefficients
        - list[str]: True support feature names in format ['feature_0', 'feature_1', ...]
    """
    print("Generating dataset with:")
    print(f" - {C.N_SAMPLES} samples")
    print(f" - {C.N_FEATURES} features")
    print(f" - noise std: {C.NOISE_STD}")
    print(f" - ratio of missing values: {C.NAN_RATIO}")
    if C.BINARIZE:
        print(
            f" - binary classification problem with {C.BINARY_RESPONSE_RATIO} ratio of positive labels"
        )
    else:
        print(" - regression problem")
    print(
        f" - {C.N_GENERATING_SOLUTIONS} original solutions, each with {C.SPARSITY} supporting vectors"
    )

    df, y, generating_solutions, parameters = generate_artificial_dataset(
        n_samples=C.N_SAMPLES,
        n_features=C.N_FEATURES,
        n_solutions=C.N_GENERATING_SOLUTIONS,
        sparsity=C.SPARSITY,
        noise_data_std=C.NOISE_STD,
        nan_ratio=C.NAN_RATIO,
        binarize=C.BINARIZE,
        binary_response_ratio=C.BINARY_RESPONSE_RATIO,
        random_seed=C.DATASET_SEED,
        save_to_csv=False,
        print_data_overview=False,
    )

    support_indices = parameters["support_indices"].sum()
    true_support_features = [f"feature_{i}" for i in set(support_indices)]

    return df, y, generating_solutions, parameters, true_support_features


def run_feature_selection(df: pd.DataFrame, y: Any) -> Tuple[Any, List[Dict], Tuple]:
    """
    Run GEMSS optimization for sparse feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with shape (n_samples, n_features)
    y : array-like
        Target variable (binary classification or regression)

    Returns
    -------
    tuple
        A 3-tuple containing:
        - BayesianFeatureSelector: Trained GEMSS selector instance
        - list[dict]: Optimization history with loss and parameter traces
        - tuple: Solution data (full_solutions, top_solutions, outlier_solutions, final_parameters)
    """
    print("\nRunning GEMSS...")

    selector = BayesianFeatureSelector(
        n_features=C.N_FEATURES,
        n_components=C.N_CANDIDATE_SOLUTIONS,
        X=df.values,
        y=y,
        prior=C.PRIOR_TYPE,
        sss_sparsity=C.PRIOR_SPARSITY,
        sample_more_priors_coeff=C.SAMPLE_MORE_PRIORS_COEFF,
        var_slab=C.VAR_SLAB,
        var_spike=C.VAR_SPIKE,
        weight_slab=C.WEIGHT_SLAB,
        weight_spike=C.WEIGHT_SPIKE,
        student_df=C.STUDENT_DF,
        student_scale=C.STUDENT_SCALE,
        lr=C.LEARNING_RATE,
        batch_size=C.BATCH_SIZE,
        n_iter=C.N_ITER,
    )

    history = selector.optimize(
        regularize=C.IS_REGULARIZED,
        lambda_jaccard=C.LAMBDA_JACCARD,
        verbose=False,
    )
    print("Optimization finished.")

    full_solutions, top_solutions, outlier_solutions, final_parameters = (
        recover_solutions(
            search_history=history,
            desired_sparsity=C.DESIRED_SPARSITY,
            min_mu_threshold=C.MIN_MU_THRESHOLD,
            use_median_for_outlier_detection=C.USE_MEDIAN_FOR_OUTLIER_DETECTION,
            outlier_deviation_thresholds=C.OUTLIER_DEVIATION_THRESHOLDS,
        )
    )

    return (
        selector,
        history,
        (full_solutions, top_solutions, outlier_solutions, final_parameters),
    )


def analyze_feature_discovery(
    solutions: Dict,
    true_support_features: List[str],
) -> Tuple[Set, Set, Set]:
    """
    Analyze feature discovery performance for any set of candidate solutions.

    Parameters
    ----------
    solutions : dict
        Solutions by component (can be full, top, or outlier solutions).
        Each key is a component name, each value is a DataFrame with 'Feature' column.
    true_support_features : list[str]
        True support feature names from the generating process

    Returns
    -------
    tuple
        A 3-tuple of sets containing:
        - set: Features found by the algorithm
        - set: True support features that were missed
        - set: Extra features found that are not in true support
    """
    true_support_features = set(true_support_features)

    # Analyze the provided solutions
    features = get_features_from_solutions(solutions)
    features_found = set(get_unique_features(features))
    missing_features = true_support_features - features_found
    extra_features = features_found - true_support_features

    return (
        features_found,
        missing_features,
        extra_features,
    )


def run_diagnostics(history: List[Dict]) -> Optional[List[Dict]]:
    """
    Run performance diagnostics on optimization history.

    Parameters
    ----------
    history : list[dict]
        Optimization history containing loss values, parameter traces,
        and convergence information from the GEMSS optimization

    Returns
    -------
    list[dict] or None
        Performance test results with status, messages, and component details,
        or None if diagnostics failed to run
    """
    print("Running performance diagnostics...")
    try:
        diagnostics = run_performance_diagnostics(
            history=history, desired_sparsity=C.DESIRED_SPARSITY, verbose=False
        )
        performance_results = diagnostics.test_results
        print(f"Performance tests completed: {len(performance_results)} tests run")
        return performance_results
    except Exception as e:
        print(f"Performance diagnostics failed: {e}")
        return None


def format_parameters_section() -> List[str]:
    """
    Format the parameters section for experiment output.

    Returns
    -------
    list[str]
        Lines of text containing the experiment title and all configuration
        parameters from gemss.config formatted as markdown
    """
    lines = ["# GEMSS Experiment Results\n"]
    lines.append("## Parameters and Settings\n")
    params = C.as_dict()
    for k, v in params.items():
        lines.append(f"- {k}: {v}")
    return lines


def format_feature_summary(
    solution_type: str,
    feature_analysis: Tuple[Set, Set, Set],
    true_support_features: List[str],
) -> List[str]:
    """
    Format the feature discovery summary for any solution type.

    Parameters
    ----------
    solution_type : str
        Type of solutions being analyzed ('full', 'top', or an outlier variant)
    feature_analysis : tuple[set, set, set]
        Results from analyze_feature_discovery: (found, missing, extra) features
    true_support_features : list[str]
        True support feature names from the generating process

    Returns
    -------
    list[str]
        Formatted text lines summarizing feature discovery performance
        including counts and lists of found, missing, and extra features
    """
    features_found, missing_features, extra_features = feature_analysis

    lines = [
        f"\n## Summary of discovered features for {solution_type.upper()} solutions:\n"
    ]
    lines.append(f" - {len(true_support_features)} unique true support features:")
    lines.append(f"   {sorted(true_support_features)}\n")

    # Solution type summary
    lines.append(f"\n - {len(features_found)} discovered features:")
    lines.append(f"   {sorted(features_found)}")

    lines.append(f"\n - {len(missing_features)} missed true support features:")
    lines.append(f"   {sorted(missing_features)}")

    lines.append(
        f"\n - {len(extra_features)} extra features found (not in true support):"
    )
    lines.append(f"   {sorted(extra_features)}")

    return lines


def format_solution_type_header(
    solution_type: str,
) -> List[str]:
    """
    Format the header section for a specific solution type.

    Parameters
    ----------
    solution_type : str
        Type of solutions ('full', 'top', or outlier variant)

    Returns
    -------
    list[str]
        Formatted header lines explaining the solution type and its parameters
    """
    solution_label = solution_type.upper()
    lines = []
    lines.append(
        f"   ------   ANALYSIS - Solution type: {solution_type.upper()}   ------   "
    )
    lines.append(f"\n## {solution_label} Solutions\n")

    if solution_type == "full":
        lines.append(f"All features with |mu| > {C.MIN_MU_THRESHOLD}\n")
    elif solution_type == "top":
        lines.append(f"Required sparsity = {C.DESIRED_SPARSITY}\n")
    elif "outlier" in solution_type:
        lines.append(f"\n## {solution_label} Solutions\n")
        lines.append(
            " Features identified as outliers based on statistical deviation.\n"
        )
    return lines


def format_features_in_components(
    solutions: Dict,
) -> List[str]:
    """
    Format the detailed breakdown of features within each component.

    Parameters
    ----------
    solutions : dict
        Solutions by component where each key is component name and
        each value is DataFrame with 'Feature' and 'Mu value' columns

    Returns
    -------
    list[str]
        Formatted text lines showing features and their mu values
        organized by component
    """
    lines = []
    lines.append(f"\n\n## Features in components")
    for component, df in solutions.items():
        lines.append(f"\n### {component.upper()} ({df.shape[0]} features):")
        for i, row in df.iterrows():
            lines.append(f" - {row['Feature']}: mu = {row['Mu value']:.4f}")
        lines.append("")

    return lines


def format_performance_diagnostics(
    performance_results: Optional[List[Dict]],
) -> List[str]:
    """
    Format the performance diagnostics section for experiment output.

    Parameters
    ----------
    performance_results : list[dict] or None
        Results from run_diagnostics containing test statuses and messages,
        or None if diagnostics failed

    Returns
    -------
    list[str]
        Formatted text lines showing diagnostic test results with
        summary counts and detailed failure/warning information
    """
    lines = ["\n## Performance Diagnostics\n"]

    if not performance_results:
        lines.append("Performance diagnostics could not be completed.")
        lines.append("")
        return lines

    # Summary counts
    status_counts = {"FAILED": 0, "WARNING": 0, "PASSED": 0}
    for result in performance_results:
        status_counts[result["status"]] += 1

    lines.append(
        f"Total tests: {len(performance_results)} | "
        f"Failed: {status_counts['FAILED']} | "
        f"Warnings: {status_counts['WARNING']} | "
        f"Passed: {status_counts['PASSED']}\n"
    )

    # Results by status (failed first, then warnings, then passed)
    status_symbols = {"FAILED": "[FAIL]", "WARNING": "[WARN]", "PASSED": "[PASS]"}

    for status in ["FAILED", "WARNING", "PASSED"]:
        status_tests = [r for r in performance_results if r["status"] == status]
        if not status_tests:
            continue

        lines.append(f"\n{status} TESTS:")
        lines.append("-" * 40)

        for result in status_tests:
            lines.append(f"{status_symbols[status]} {result['test_name']}")
            lines.append(f"Message: {result['message']}")

            # Add component details for failed/warning tests
            if (
                status in ["FAILED", "WARNING"]
                and "component_details" in result["details"]
            ):
                problem_components = [
                    f"Component {comp['component']} ({comp['component_status']})"
                    for comp in result["details"]["component_details"]
                    if comp["component_status"] != "PASSED"
                ]

                if problem_components:
                    lines.append(f"Problem components: {', '.join(problem_components)}")

            lines.append("")  # Empty line between tests

    lines.append("")
    return lines


def determine_output_path(custom_output: Optional[str]) -> Path:
    """
    Determine the output file path for experiment results.

    Parameters
    ----------
    custom_output : str or None
        Custom output filename provided by user, or None for automatic naming

    Returns
    -------
    pathlib.Path
        Full path where experiment results will be written.
        Uses timestamp-based naming if custom_output is None.
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if custom_output:
        output_path = Path(custom_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        parent_dir = Path(os.path.dirname(os.getcwd()))
        results_dir = parent_dir / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"experiment_output_{timestamp}.txt"

    return output_path


def write_results(output_lines: List[str], output_path: Path) -> None:
    """
    Write experiment results to file.

    Parameters
    ----------
    output_lines : list[str]
        All formatted text lines to write to the output file
    output_path : pathlib.Path
        Full path where the results file will be created

    Returns
    -------
    None
    """
    print("Writing results...")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Experiment summary written to {output_path}")


def main() -> None:
    """
    Main experiment pipeline for GEMSS sparse feature selection.

    Executes the complete workflow:
    1. Parse command line arguments
    2. Generate artificial dataset with known ground truth
    3. Run GEMSS optimization to find sparse solutions
    4. Analyze feature discovery performance for all solution types
    5. Generate comprehensive results report
    6. Optionally run performance diagnostics
    7. Write results to timestamped output file

    The output file contains parameter settings, feature discovery analysis,
    detailed solution breakdowns, and diagnostic information.
    """
    # Parse arguments
    args = parse_arguments()
    output_path = determine_output_path(args.output)

    # Generate dataset
    df, y, generating_solutions, parameters, true_support_features = generate_dataset()

    # Run feature selection
    selector, history, solution_data = run_feature_selection(df, y)
    full_solutions, top_solutions, outlier_solutions, final_parameters = solution_data

    # Write experiment parameters
    output_lines = []
    output_lines.extend(format_parameters_section())
    output_lines.append(
        "\n----------------------------------------------------------------------\n"
    )

    # Analyze results
    feature_discovery_analysis = {}
    all_solutions = {
        "full": full_solutions,
        "top": top_solutions,
    }
    for deviation, solutions in outlier_solutions.items():
        all_solutions[f"outlier ({deviation})"] = solutions

    for solution_type, solutions in all_solutions.items():
        feature_discovery_analysis[solution_type] = analyze_feature_discovery(
            solutions, true_support_features
        )  # [found, missing, extra]
        output_lines.extend(format_solution_type_header(solution_type))
        output_lines.extend(
            format_feature_summary(
                solution_type,
                feature_discovery_analysis[solution_type],
                true_support_features,
            )
        )
        output_lines.extend(
            format_features_in_components(
                solutions,
            )
        )

        output_lines.append(
            "\n----------------------------------------------------------------------\n"
        )

    # Run diagnostics once for the entire optimization, if requested
    if args.diagnostics:
        performance_results = run_diagnostics(history)
        output_lines.extend(format_performance_diagnostics(performance_results))
        output_lines.append(
            "\n----------------------------------------------------------------------\n"
        )

    # Write results
    write_results(output_lines, output_path)


if __name__ == "__main__":
    main()
