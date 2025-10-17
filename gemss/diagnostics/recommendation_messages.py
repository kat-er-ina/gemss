"""
Recommendation messages for performance test diagnostics.

This module contains all the message content for different recommendation scenarios.
Each message dictionary contains standardized keys for consistent display formatting.
"""

from typing import Dict, Any


# Message templates for different recommendation scenarios
RECOMMENDATION_MESSAGES: Dict[str, Dict[str, str]] = {
    "feature_ordering_failed_sparsity_gap_failed": {
        "section_header": "Critical issues detected",
        "title": "Feature ordering and sparsity gap tests failed",
        "description": "Both feature ordering and sparsity separation are problematic.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_failed_sparsity_gap_warning": {
        "section_header": "Critical issues detected",
        "title": "Feature ordering test failed, sparsity gap test warning",
        "description": "Primary issue: poor feature ordering convergence. Secondary: suboptimal separation.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_failed_sparsity_gap_passed": {
        "section_header": "Critical issues detected",
        "title": "Feature ordering test failed",
        "description": "Good feature discrimination but poor convergence behavior.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_warning_sparsity_gap_failed": {
        "section_header": "Critical issues detected",
        "title": "Sparsity gap test failed, feature ordering test warning",
        "description": "Primary issue: poor feature separation. Secondary: suboptimal convergence.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_passed_sparsity_gap_failed": {
        "section_header": "Critical issues detected",
        "title": "Sparsity gap test failed",
        "description": "Good convergence behavior but poor feature discrimination.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_warning_sparsity_gap_warning": {
        "section_header": "Optimization opportunities",
        "title": "Feature ordering and sparsity gap tests warning",
        "description": "Both tests show room for improvement with balanced adjustments.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_warning_sparsity_gap_passed": {
        "section_header": "Optimization opportunities",
        "title": "Feature ordering test warning",
        "description": "Good feature separation with room for convergence improvement.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_passed_sparsity_gap_warning": {
        "section_header": "Optimization opportunities",
        "title": "Sparsity gap test warning",
        "description": "Good convergence with room for separation improvement.",
        "recommendations": (
            "**Lorem ipsum:**\n"
            "- Example recommendation 1\n"
            "- Example recommendation 2\n"
        ),
    },
    "feature_ordering_passed_sparsity_gap_passed": {
        "section_header": "Excellent performance!",
        "title": "Feature ordering and sparsity gap tests passed",
        "description": "Your current parameter configuration is working well for this dataset.",
    },
    "unknown_combination": {
        "section_header": "Unknown status combination",
        "title": "Unexpected test results",
        "description": "An unexpected combination of test results was encountered.",
        "recommendations": "There are no recommendations for this situation.",
    },
}


def get_recommendation_message(key: str) -> Dict[str, str]:
    """
    Get recommendation message dictionary for a given key.

    Parameters
    ----------
    key : str
        The recommendation key to look up.

    Returns
    -------
    Dict[str, str]
        Dictionary containing message data with keys:
        - section_header: Main section title with emoji
        - title: Specific scenario title with emoji
        - description: Problem description
        - recommendations: Formatted recommendations
    """
    return RECOMMENDATION_MESSAGES.get(
        key, RECOMMENDATION_MESSAGES["unknown_combination"]
    )


def get_available_recommendation_keys() -> list[str]:
    """
    Get list of all available recommendation keys.

    Returns
    -------
    list[str]
        List of all recommendation keys available in the message mapping.
    """
    return list(RECOMMENDATION_MESSAGES.keys())
