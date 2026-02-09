"""
Recommendation messages for performance test diagnostics.

This module contains all the message content for different recommendation scenarios.
Each message dictionary contains standardized keys for consistent display formatting.
"""

# Message templates for different recommendation scenarios
RECOMMENDATION_MESSAGES: dict[str, dict[str, str]] = {
    'feature_ordering_failed_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering and sparsity gap test failed',
        'description': ('The optimization is not working, likely due to too tight spike in prior.'),
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
        ),
    },
    'feature_ordering_failed_sparsity_gap_warning': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering failed, sparsity gap warning',
        'description': ('The optimization is not working, likely due to too tight spike in prior.'),
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
        ),
    },
    'feature_ordering_failed_sparsity_gap_passed': {
        'section_header': 'Critical issues detected',
        'title': 'Feature ordering test failed',
        'description': ('The optimization is not working, likely due to too tight spike in prior.'),
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Increase VAR_SPIKE by at least one order of magnitude\n'
            '- Very carefully decrease VAR_SPIKE\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Sparsity gap test failed, feature ordering test warning',
        'description': (
            'Feature selection quality questionable for some components. '
            'Poor discrimination among features.'
        ),
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '**Other possible actions:**\n'
            '- Decrease DESIRED_SPARSITY to have more sparse priors.\n'
            '- Consider decreasing N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_passed_sparsity_gap_failed': {
        'section_header': 'Critical issues detected',
        'title': 'Sparsity gap test failed and the feature ordering test passed.',
        'description': 'Two scenarios have been observed that lead to this outcome.',
        'recommendations': (
            '**If feature mu variance is large and ordering changes over iters, '
            'fine tuning may help:**\n'
            '- Increase N_ITER.\n'
            '- Very carefully decrease VAR_SPIKE.\n'
            '- Try Jaccard regularization for diverse solutions.\n'
            '- Decrease DESIRED_SPARSITY.\n'
            '**If same features have largest mu at start and end (rest → 0), '
            'algorithm is NOT working:**\n'
            '- Increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '- Consider decreasing DESIRED_SPARSITY.\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_warning': {
        'section_header': 'Optimization opportunities',
        'title': 'Feature ordering and sparsity gap tests warning',
        'description': ('Feature selection quality questionable for some components.'),
        'recommendations': (
            '**Possible courses of action:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Carefully balance VAR_SPIKE and VAR_SLAB.\n'
            '- Try Jaccard regularization for diverse solutions.\n'
            '- Consider decreasing N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_warning_sparsity_gap_passed': {
        'section_header': 'Optimization opportunities',
        'title': 'Feature ordering test warning',
        'description': ('Feature selection quality questionable for some components.'),
        'recommendations': (
            '**Focus on getting the features not to converge uniformly:**\n'
            '- Carefully increase VAR_SPIKE.\n'
            '- Possibly counter-balance with VAR_SLAB.\n'
            '- Consider decreasing N_CANDIDATE_SOLUTIONS.\n'
        ),
    },
    'feature_ordering_passed_sparsity_gap_warning': {
        'section_header': 'Optimization opportunities',
        'title': 'Sparsity gap test warning',
        'description': (
            'Good convergence; room for improvement in separation of '
            'selected vs not-selected features.'
        ),
        'recommendations': (
            '**Possible courses of action:**\n'
            '- Consider increasing the number of iterations N_ITER.\n'
            '- Very carefully decrease VAR_SPIKE\n'
            '- Try Jaccard regularization for diverse solutions.\n'
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


def get_recommendation_message(key: str) -> dict[str, str]:
    """
    Get recommendation message dictionary for a given key.

    Parameters
    ----------
    key : str
        The recommendation key to look up.

    Returns
    -------
    dict[str, str]
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
