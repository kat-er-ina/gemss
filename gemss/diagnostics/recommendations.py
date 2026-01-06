"""
Parameter recommendations for performance test diagnostics.

This module provides functionality for analyzing performance test results
and generating specific parameter adjustment recommendations based on the
diagnostic outcomes.
"""

from typing import Dict, List, Any, Optional
from IPython.display import display, Markdown
from .recommendation_messages import get_recommendation_message
from gemss.config import display_current_config


class RecommendationEngine:
    """
    A framework for analyzing performance test results and generating
    parameter adjustment recommendations for the user of the GEMSS algorithm.

    This class takes performance test diagnostics and provides structured
    recommendations for improving algorithm performance through parameter
    adjustments.

    Attributes
    ----------
    diagnostics : PerformanceTests
        The performance tests instance with completed test results.
    constants : Optional[Dict[str, Any]]
        Configuration constants dictionary for displaying current settings.
    recommendation_keys : List[str]
        List of applicable recommendation keys based on test results.
    severity_level : str
        Overall severity assessment: "CRITICAL", "WARNING", or "OPTIMIZATION".
    test_status : Dict[str, str]
        Categorized test results by type and status.
    """

    # Success combination constant
    SUCCESS_COMBINATION = "feature_ordering_passed_sparsity_gap_passed"

    # Parameter ranges for reference
    # May need to be adjusted
    PARAMETER_REFERENCE_RANGES = {
        "N_ITER": (1000, 10000),
        "LEARNING_RATE": (0.0005, 0.005),
        "VAR_SLAB": (10, 500),
        "VAR_SPIKE": (0.0001, 1.0),
        "LAMBDA_JACCARD": (0, 2000),
        "BATCH_SIZE": (16, 64),
    }

    def __init__(
        self,
        diagnostics,
        constants: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the recommendation engine with diagnostic results.

        Parameters
        ----------
        diagnostics : PerformanceTests
            The performance tests instance with completed test results.
        constants : Optional[Dict[str, Any]], optional
            Configuration constants dictionary for displaying current settings.
            Default is None.
        """
        self.diagnostics = diagnostics
        self.constants = constants
        self.recommendation_keys: List[str] = []
        self.severity_level: str = ""
        self.test_status: Dict[str, str] = {}

        # Analyze test results upon initialization
        self.analyze_test_results()

    def analyze_test_results(self) -> None:
        """
        Analyze test results and determine recommendation keys and severity level.

        This method categorizes test results, determines appropriate recommendation
        keys, and assesses the overall severity level.
        """
        self.test_status = self._categorize_test_results()
        self.recommendation_keys = self._determine_recommendation_keys()
        self.severity_level = self._assess_severity_level()

    def display_all_recommendations(self) -> None:
        """
        Display all recommendations based on the analyzed test results.
        """
        if not self.diagnostics.test_results:
            display(Markdown("## No test results available for recommendations"))
            return

        # Display current configuration if constants are provided
        if self.constants is not None:
            self._display_configuration_summary()

        # Display recommendations header
        self._display_recommendations_header()

        # Display individual recommendations for each key
        for key in self.recommendation_keys:
            self._display_single_recommendation(key)

        # Display parameter adjustment summary (except when all tests pass)
        if not self._has_only_success():
            display_parameter_adjustment_summary()

    def get_recommendation_summary(self) -> Dict[str, Any]:
        """
        Get a structured summary of recommendations.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing recommendation summary with keys:
            - severity_level: Overall severity assessment
            - recommendation_keys: List of recommendation keys
            - test_status: Categorized test results
            - has_issues: Boolean indicating if there are issues to address
        """
        return {
            "severity_level": self.severity_level,
            "recommendation_keys": self.recommendation_keys,
            "test_status": self.test_status,
            "has_issues": not self._has_only_success(),
        }

    def export_recommendations_to_dict(self) -> Dict[str, Any]:
        """
        Export recommendations to a dictionary for logging or saving.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing full recommendation data including
            messages and structured analysis results.
        """
        recommendations_data = []
        for key in self.recommendation_keys:
            recommendation = get_recommendation_message(key)
            recommendations_data.append(
                {
                    "key": key,
                    "section_header": recommendation["section_header"],
                    "title": recommendation["title"],
                    "description": recommendation["description"],
                    "recommendations": recommendation.get("recommendations", ""),
                }
            )

        return {
            "summary": self.get_recommendation_summary(),
            "recommendations": recommendations_data,
            "constants": self.constants,
        }

    def _categorize_test_results(self) -> Dict[str, str]:
        """
        Categorize test results by type and status.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping test types to their status.
        """
        test_status = {}
        for result in self.diagnostics.test_results:
            test_name = result["test_name"]
            status = result["status"]

            if "Ordering of top" in test_name:
                test_status["feature_ordering"] = status
            elif "Sparsity gap" in test_name:
                test_status["sparsity_gap"] = status
            # Add more test types here as they are implemented

        return test_status

    def _determine_recommendation_keys(self) -> List[str]:
        """
        Determine recommendation keys based on test result combinations.

        Returns
        -------
        List[str]
            List of recommendation keys to display.
        """
        recommendation_keys = []

        # Get status values with defaults
        fo_status = self.test_status.get("feature_ordering", "PASSED")
        sg_status = self.test_status.get("sparsity_gap", "PASSED")

        # Logic for determining recommendation keys
        if fo_status == "FAILED" and sg_status == "FAILED":
            recommendation_keys.append("feature_ordering_failed_sparsity_gap_failed")
        elif fo_status == "FAILED" and sg_status == "WARNING":
            recommendation_keys.append("feature_ordering_failed_sparsity_gap_warning")
        elif fo_status == "FAILED" and sg_status == "PASSED":
            recommendation_keys.append("feature_ordering_failed_sparsity_gap_passed")
        elif fo_status == "WARNING" and sg_status == "FAILED":
            recommendation_keys.append("feature_ordering_warning_sparsity_gap_failed")
        elif fo_status == "PASSED" and sg_status == "FAILED":
            recommendation_keys.append("feature_ordering_passed_sparsity_gap_failed")
        elif fo_status == "WARNING" and sg_status == "WARNING":
            recommendation_keys.append("feature_ordering_warning_sparsity_gap_warning")
        elif fo_status == "WARNING" and sg_status == "PASSED":
            recommendation_keys.append("feature_ordering_warning_sparsity_gap_passed")
        elif fo_status == "PASSED" and sg_status == "WARNING":
            recommendation_keys.append("feature_ordering_passed_sparsity_gap_warning")
        elif fo_status == "PASSED" and sg_status == "PASSED":
            recommendation_keys.append("feature_ordering_passed_sparsity_gap_passed")
        else:
            # Fallback for unexpected combinations
            recommendation_keys.append("unknown_combination")

        # Future extension point: Add logic for additional tests here
        # Example:
        # if "memory_usage" in self.test_status and self.test_status["memory_usage"] == "FAILED":
        #     recommendation_keys.append("memory_optimization_needed")

        return recommendation_keys

    def _assess_severity_level(self) -> str:
        """
        Assess the overall severity level based on test results.

        Returns
        -------
        str
            Severity level: "CRITICAL", "WARNING", or "OPTIMIZATION".
        """
        # Check for any FAILED tests
        if any(status == "FAILED" for status in self.test_status.values()):
            return "CRITICAL"

        # Check for any WARNING tests
        if any(status == "WARNING" for status in self.test_status.values()):
            return "WARNING"

        # All tests passed
        return "OPTIMIZATION"

    def _display_configuration_summary(self) -> None:
        """Display current configuration summary if constants are provided."""
        display_current_config(
            self.constants,
            constant_type="algorithm_and_postprocessing",
        )

    def _display_recommendations_header(self) -> None:
        """Display the main recommendations header with severity indication."""
        severity_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "OPTIMIZATION": "💡"}

        emoji = severity_emoji.get(self.severity_level, "📊")
        display(Markdown(f"# {emoji} Parameter recommendations"))

    def _display_single_recommendation(self, key: str) -> None:
        """
        Display a single recommendation based on its key.

        Parameters
        ----------
        key : str
            The recommendation key to display.
        """
        recommendation = get_recommendation_message(key)

        # Display section header
        display(Markdown(f"## {recommendation['section_header']}"))

        # Display specific recommendation
        display(Markdown(f"### {recommendation['title']}"))
        display(Markdown(recommendation["description"]))

        # Display recommendations if present
        if recommendation.get("recommendations"):
            display(Markdown(recommendation["recommendations"]))

    def _format_parameter_range(self, param_name: str) -> str:
        """Format parameter range as string.

        Parameters
        ----------
        param_name : str
            Name of the parameter to format range for.

        Returns
        -------
        str
            Formatted range string like "min-max".
        """
        range_tuple = self.PARAMETER_REFERENCE_RANGES[param_name]
        return f"{range_tuple[0]}-{range_tuple[1]}"

    def _has_only_success(self) -> bool:
        """
        Check if all recommendations indicate successful performance.

        Returns
        -------
        bool
            True if only success recommendations are present.
        """
        return (
            len(self.recommendation_keys) == 1
            and self.recommendation_keys[0] == self.SUCCESS_COMBINATION
        )


# Convenience functions for backward compatibility
def get_recommendation_keys(diagnostics) -> List[str]:
    """
    Get recommendation keys for diagnostics (backward compatibility).

    Parameters
    ----------
    diagnostics : PerformanceTests
        The performance tests instance with completed test results.

    Returns
    -------
    List[str]
        List of recommendation keys to display.
    """
    engine = RecommendationEngine(diagnostics)
    return engine.recommendation_keys


def display_recommendation(key: str) -> None:
    """
    Display a single recommendation based on its key (backward compatibility).

    Parameters
    ----------
    key : str
        The recommendation key to display.

    Returns
    -------
    None
        Displays recommendation as markdown output.
    """
    recommendation = get_recommendation_message(key)

    # Display section header
    display(Markdown(f"## {recommendation['section_header']}"))

    # Display specific recommendation
    display(Markdown(f"### {recommendation['title']}"))
    display(Markdown(recommendation["description"]))

    # Display recommendations if present
    if recommendation.get("recommendations"):
        display(Markdown(recommendation["recommendations"]))


def display_parameter_adjustment_summary() -> None:
    """Display a summary table of key parameters.

    Returns
    -------
    None
        Displays parameter adjustment table as markdown output.
    """
    display(Markdown("### Parameter adjustment quick reference"))

    # Use the same ranges as the class
    def format_range(param_name: str) -> str:
        range_tuple = RecommendationEngine.PARAMETER_REFERENCE_RANGES[param_name]
        return f"{range_tuple[0]}-{range_tuple[1]}"

    n_iter_range = format_range("N_ITER")
    lr_range = format_range("LEARNING_RATE")
    var_slab_range = format_range("VAR_SLAB")
    var_spike_range = format_range("VAR_SPIKE")
    lambda_jaccard_range = format_range("LAMBDA_JACCARD")
    batch_range = format_range("BATCH_SIZE")

    display(
        Markdown(
            "| Parameter | Increase when | Decrease when | Typical range |\n"
            "|-----------|---------------|---------------|---------------|\n"
            "| `DESIRED_SPARSITY` | Too restrictive. | The progress of mu values indicates that fewer features remain non-negligible. | Dataset-dependent. |"
            "|`N_CANDIDATE_SOLUTIONS`| Current set of solutions does not contain enough combinations despite proper Jaccard regularization and desired sparsity settings. | Solutions overlap siginificantly and uselessly. Some component weights (alphas) might be negligible too. | Dataset-dependent. Advisable to be at least double the expected `true` number of solutions. |\n"
            f"| `VAR_SPIKE` | All features converge (uniformly) to 0, i.e. over-regularization leads to no optimization and feature selection. | Too many features are selected in each component (false positives). | {var_spike_range} |\n"
            f"| `VAR_SLAB` | Poor feature separation. | Over-regularization. | {var_slab_range} |\n"
            f"| `IS_REGULARIZED` | True = penalize similarity among solutions. | False = do not influence diversity of features among solutions. | False (0) or True (1) |\n"
            f"| `LAMBDA_JACCARD` | Greater diversity wanted: the individual solutions contain too many similar features. | Interested in solutions with overlapping feature sets. | {lambda_jaccard_range} |\n"
            f"| `N_ITER` | The ELBO has not converged, the mu values are still changing significantly. | The ELBO convergence plateaus early, time constraints. | {n_iter_range} |\n"
            f"| `LEARNING_RATE` | Too slow progress. | Unstable training. | {lr_range} |\n"
            f"|`BATCH_SIZE`| Dataset contains noise, missing data, or imbalanced classes. When the sample size is large. | Iterations are too slow. Sample size is too small. | Dataset-dependent, typically {batch_range}. |"
        )
    )


def display_recommendations(
    diagnostics,
    constants: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display parameter recommendations based on performance test results.
    Use the RecommendationEngine class internally.

    Parameters
    ----------
    diagnostics : PerformanceTests
        The performance tests instance with completed test results.
    constants : Optional[Dict[str, Any]]
        A dictionary of algorithm-related constants

    Returns
    -------
    None
        Displays recommendations as markdown output.
    """
    engine = RecommendationEngine(diagnostics, constants)
    engine.display_all_recommendations()
