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
from gemss.generate_artificial_dataset import generate_artificial_dataset
from gemss.inference import BayesianFeatureSelector
from gemss.result_postprocessing import recover_solutions
from gemss.diagnostics.performance_tests import run_performance_diagnostics


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GEMSS experiment for sparse feature selection."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom filename for output summary (should end with .txt)",
    )
    return parser.parse_args()


def generate_dataset() -> Tuple[pd.DataFrame, Any, Dict, Dict, List[str]]:
    """
    Generate artificial dataset for the experiment.

    Returns
    -------
    tuple
        DataFrame with features, target variable, generating solutions,
        parameters dict, and true support feature names
    """
    print("Generating dataset with:")
    print(f" - {C.N_SAMPLES} samples")
    print(f" - {C.N_FEATURES} features")
    print(
        f" - {C.N_GENERATING_SOLUTIONS} original solutions, each with {C.SPARSITY} supporting vectors"
    )

    df, y, generating_solutions, parameters = generate_artificial_dataset(
        n_samples=C.N_SAMPLES,
        n_features=C.N_FEATURES,
        n_solutions=C.N_GENERATING_SOLUTIONS,
        sparsity=C.SPARSITY,
        noise_data_std=C.NOISE_STD,
        binarize=C.BINARIZE,
        binary_response_ratio=C.BINARY_RESPONSE_RATIO,
        random_seed=C.DATASET_SEED,
        save_to_csv=False,
        print_data_overview=False,
    )

    support_indices = parameters["support_indices"].sum()
    true_support_features = [f"feature_{i}" for i in set(support_indices)]

    return df, y, generating_solutions, parameters, true_support_features


def run_feature_selection(df: pd.DataFrame, y: Any) -> Tuple[Any, List[Dict], Any]:
    """
    Run GEMSS optimization for sparse feature selection.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    y : array-like
        Target variable

    Returns
    -------
    tuple
        Trained selector, optimization history, solutions
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

    solutions, final_parameters, full_nonzero_solutions = recover_solutions(
        search_history=history,
        desired_sparsity=C.DESIRED_SPARSITY,
        min_mu_threshold=C.MIN_MU_THRESHOLD,
        verbose=False,
    )

    return selector, history, (solutions, final_parameters, full_nonzero_solutions)


def analyze_feature_discovery(
    solutions: Dict, true_support_features: List[str]
) -> Tuple[Set, Set, Set]:
    """
    Analyze feature discovery performance.

    Parameters
    ----------
    solutions : dict
        Discovered solutions by component
    true_support_features : list
        True support feature names

    Returns
    -------
    tuple
        Sets of found features, missing features, extra features
    """
    features_found = set().union(*solutions.values())
    missing_features = set(true_support_features) - features_found
    extra_features = features_found - set(true_support_features)

    return features_found, missing_features, extra_features


def run_diagnostics(history: List[Dict]) -> Optional[List[Dict]]:
    """
    Run performance diagnostics on optimization history.

    Parameters
    ----------
    history : list
        Optimization history

    Returns
    -------
    list or None
        Performance test results or None if failed
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
    """Format the parameters section for output."""
    lines = ["# GEMSS Experiment Results\n"]
    lines.append("## Parameters and Settings\n")
    params = C.as_dict()
    for k, v in params.items():
        lines.append(f"- {k}: {v}")
    return lines


def format_feature_summary(
    true_support_features: List[str],
    features_found: Set,
    missing_features: Set,
    extra_features: Set,
) -> List[str]:
    """Format the feature discovery summary."""
    lines = ["\n## Summary of discovered features:\n"]
    lines.append(f" - {len(true_support_features)} unique true support features:")
    lines.append(f"{sorted(true_support_features)}\n")
    lines.append(f" - {len(features_found)} discovered features:")
    lines.append(f"{sorted(features_found)}\n")
    lines.append(f" - {len(missing_features)} missed true support features:")
    lines.append(f"{sorted(missing_features)}\n")
    lines.append(
        f" - {len(extra_features)} extra features found (not in true support):"
    )
    lines.append(f"{sorted(extra_features)}")
    return lines


def format_solutions_section(
    solutions: Dict, full_nonzero_solutions: Dict
) -> List[str]:
    """Format the solutions section for output."""
    lines = [
        f"\n## Solutions found (top {C.DESIRED_SPARSITY} features for each component)\n"
    ]
    solutions_df = pd.DataFrame.from_dict(solutions, orient="index").T
    lines.append(solutions_df.to_string())

    lines.append("\n## Full solutions\n")
    lines.append(
        " - all features with mu greater than the minimal threshold in last iterations"
    )
    lines.append(f" - minimal mu threshold: {C.MIN_MU_THRESHOLD}")

    for component, df in full_nonzero_solutions.items():
        lines.append(f"\n### {component.upper()} ({df.shape[0]} features):\n")
        for i, row in df.iterrows():
            lines.append(f" - {row['Feature']}: mu = {row['Mu value']:.4f}")
        lines.append("")

    return lines


def format_performance_diagnostics(
    performance_results: Optional[List[Dict]],
) -> List[str]:
    """Format the performance diagnostics section."""
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
    """Determine the output file path."""
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
    """Write results to file."""
    print("Writing results...")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Experiment summary written to {output_path}")


def main() -> None:
    """Main experiment pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Generate dataset
    df, y, generating_solutions, parameters, true_support_features = generate_dataset()

    # Run feature selection
    selector, history, solution_data = run_feature_selection(df, y)
    solutions, final_parameters, full_nonzero_solutions = solution_data

    # Analyze results
    features_found, missing_features, extra_features = analyze_feature_discovery(
        solutions, true_support_features
    )

    # Run diagnostics
    performance_results = run_diagnostics(history)

    # Format output
    output_lines = []
    output_lines.extend(format_parameters_section())
    output_lines.extend(
        format_feature_summary(
            true_support_features, features_found, missing_features, extra_features
        )
    )
    output_lines.extend(format_solutions_section(solutions, full_nonzero_solutions))
    output_lines.extend(format_performance_diagnostics(performance_results))

    # Write results
    output_path = determine_output_path(args.output)
    write_results(output_lines, output_path)


if __name__ == "__main__":
    main()
