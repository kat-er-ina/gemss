"""
Recommendation messages for performance test diagnostics.

This module contains all the message content for different recommendation scenarios.
Each message dictionary contains standardized keys for consistent display formatting.
"""

from typing import Dict


# Message templates for different recommendation scenarios
RECOMMENDATION_MESSAGES: Dict[str, Dict[str, str]] = {
    'feature_ordering_failed_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering and sparsity gap test failed',
        'description': 'The optimization process is not working, likely due to too tight spike in the prior.',
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
        ),
    },
    'feature_ordering_failed_sparsity_gap_warning': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering test failed, sparsity gap test warning',
        'description': 'The optimization process is not working, likely due to too tight spike in the prior.',
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
        ),
    },
    'feature_ordering_failed_sparsity_gap_passed': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering test failed',
        'description': 'The optimization process is not working, likely due to too tight spike in the prior.',
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
            '- Very carefully decrease VAR_SPIKE\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Sparsity gap test failed, feature ordering test warning',
        'description': 'The quality of the feature selection process is questionable for at least some components. Poor discrimination among features.',
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '**Other possible actions:**\n'
            '- Decrease DESIRED_SPARSITY to have more sparse priors.\n'
            '- Consider whether to decrease the number of components N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_passed_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Sparsity gap test failed and the feature ordering test passed.',
        'description': 'Two scenarios have been observed that lead to this outcome.',
        'recommendations': (
            "**If there is a large variance among the features' mu and their ordering changes over the iterations, the algorithm probably needs just fine tuning to improve feature discrimination:**\n"
            '- Increase the number of iterations N_ITER.\n'
            '- Very carefully decrease VAR_SPIKE.\n'
            '- Possibly play with Jaccard regularization to obtain optimally diverse solutions.\n'
            '- Decrease DESIRED_SPARSITY to have more sparse priors.\n'
            '**If the same features have the largest mu(s) at the beginning and at the end while the rest converges to 0, the algorithm is NOT working:**\n'
            '- Increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '- Consider decreasing DESIRED_SPARSITY.\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_warning': {
        'section_header': 'Optimization opportunities',
        'title': 'Feature ordering and sparsity gap tests warning',
        'description': 'The quality of the feature selection process is questionable for at least some components.',
        'recommendations': (
            '**Possible courses of action:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Carefully balance VAR_SPIKE and VAR_SLAB.\n'
            '- Possibly play with Jaccard regularization to obtain optimally diverse solutions.\n'
            '- Consider whether to decrease the number of components N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_passed': {
        'section_header': 'Optimization opportunities',
        'title': 'Feature ordering test warning',
        'description': 'The quality of the feature selection process is questionable for at least some components.',
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '- Consider whether to decrease the number of components N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_passed_sparsity_gap_warning': {
        'section_header': 'Optimization opportunities',
        'title': 'Sparsity gap test warning',
        'description': 'Good convergence with room for improvement regarding separation of selected and not-selected features.',
        'recommendations': (
            '**Possible courses of action:**\n'
            '- Consider increasing the number of iterations N_ITER.\n'
            '- Very carefully decrease VAR_SPIKE\n'
            '- Possibly play with Jaccard regularization to obtain optimally diverse solutions.\n'
        ),
    },
    'feature_ordering_passed_sparsity_gap_passed': {
        'section_header': 'Excellent performance!',
        'title': 'Feature ordering and sparsity gap tests passed',
        'description': 'Your current parameter configuration is working well for this dataset.',
    },
    'unknown_combination': {
        'section_header': 'Unknown status combination',
        'title': 'Unexpected test results',
        'description': 'An unexpected combination of test results was encountered.',
        'recommendations': 'There are no recommendations for this situation.',
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
    return RECOMMENDATION_MESSAGES.get(key, RECOMMENDATION_MESSAGES['unknown_combination'])


def get_available_recommendation_keys() -> list[str]:
    """
    Get list of all available recommendation keys.

    Returns
    -------
    list[str]
        List of all recommendation keys available in the message mapping.
    """
    return list(RECOMMENDATION_MESSAGES.keys())
