"""
Performance tests for the feature selector algorithm.

This module provides a framework for running various diagnostic tests on the
optimization history to detect potential issues or warnings about the optimizer's
performance and convergence behavior.
"""

from typing import Any

import numpy as np
from IPython.display import Markdown, display


class PerformanceTests:
    """
    A framework for running performance diagnostic tests on optimization history.

    This class analyzes the optimization history from BayesianFeatureSelector
    and runs various tests to detect potential issues with convergence,
    feature selection stability, and other optimization characteristics.

    Attributes
    ----------
    history : Dict[str, List[Any]]
        The optimization history containing 'mu', 'var', 'alpha', and 'elbo' keys.
    test_results : List[Dict[str, Any]]
        List of test results, each containing test name, status, and details.
    mu_array : np.ndarray
        Numpy array of mu values with shape [n_iterations, n_components, n_features].
    n_iterations : int
        Number of optimization iterations.
    n_components : int
        Number of mixture components.
    n_features : int
        Number of features.
    desired_sparsity : int, optional
        Desired number of non-zero features (only present if specified in __init__).
    """

    # Icons for quick orientation
    TEST_RESULT_EMOJIS = {'FAILED': '❌', 'WARNING': '⚠️', 'PASSED': '✅'}
    COMPONENT_STATUS_ICONS = {'FAILED': '🔴', 'WARNING': '🟠', 'PASSED': '🟢'}

    # Scoring weights for test results
    # PASS_WEIGHT = 0.0  # weight for passed components in scoring
    FAIL_WEIGHT = 1.0  # weight for failed components in scoring
    WARNING_WEIGHT = 0.5  # weight for warning components in scoring

    # Component status calculation constants
    COMPONENT_FAIL_THRESHOLD_RATIO = 0.5  # no more than 50% of components should fail
    COMPONENT_WARNING_THRESHOLD_RATIO = 0.5  # ratio between fail and warning thresholds

    # Threshold constants for feature ordering test
    # to pass, at least half of the features should be different
    FAIL_PROPORTION_THRESHOLD = 0.5
    # to pass without warning, at least 75% of the features should be different
    WARNING_PROPORTION_THRESHOLD = 0.25  # 1.00 - 0.75

    # Default feature selection constants
    DEFAULT_TOP_FEATURES_MIN = 10  # minimum number of top features to compare
    # coefficient for default top features (5% of n_features)
    DEFAULT_TOP_FEATURES_COEFFICIENT = 0.05
    # fraction for default sparsity (25% of n_features)
    DEFAULT_SPARSITY_FRACTION = 0.25

    # Feature limit constants
    MAX_TOP_FEATURES_FRACTION = 0.5  # top_n_features should not exceed 50% of n_features
    EXTENDED_SPARSITY_MULTIPLIER = 2  # extended sparsity = 2 * desired_sparsity

    def __init__(
        self,
        history: dict[str, list[Any]],
        desired_sparsity: int | float | None = None,
    ) -> None:
        """
        Initialize the performance tests system with optimization history.

        Parameters
        ----------
        history : Dict[str, List[Any]]
            The optimization history from BayesianFeatureSelector.optimize().
            Expected to contain keys: 'mu', 'var', 'alpha', 'elbo'.
            The 'mu' key should contain a list of arrays with shape [n_components, n_features]
            for each iteration.
        desired_sparsity : Optional[Union[int, float]], optional
            Desired number of non-zero features (int) or fraction of features (float).
            If float, should be between 0 and 1. If int, should be positive.
            Default is None.

        Raises
        ------
        ValueError
            If desired_sparsity is not positive or if history is missing required keys.
        """
        self.history = history
        self.test_results = []

        # Convert mu to numpy array for easier processing
        self.mu_array = np.array(history['mu'])  # shape: [n_iterations, n_components, n_features]
        self.n_iterations, self.n_components, self.n_features = self.mu_array.shape

        if desired_sparsity is not None:
            if desired_sparsity <= 0:
                raise ValueError('desired_sparsity must be positive')
            elif desired_sparsity < 1:
                self.desired_sparsity = int(desired_sparsity * self.n_features)
            else:
                self.desired_sparsity = int(desired_sparsity)

    def run_all_tests(
        self,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run all available performance tests.

        Parameters
        ----------
        verbose : bool, optional
            Whether to display test results. Default is True.

        Returns
        -------
        List[Dict[str, Any]]
            List of test results.
        """
        self.test_results = []

        # Run individual tests
        self.test_top_feature_ordering()
        self.test_sparsity_gap()

        if verbose:
            self._display_results()

        return self.test_results

    def test_top_feature_ordering(
        self,
        top_n_features: int | None = None,
    ) -> dict[str, Any]:
        """
        Test whether the top N features with highest absolute mu values change
        between the first and last iteration.

        This test checks if the optimization is making progress by comparing
        the sets of most important features at the beginning and end of optimization.
        If the sets are (nearly) identical, it indicates lack of progress.

        Parameters
        ----------
        top_n_features : int, optional
            Number of top features to compare. Default is None.
            If None, uses desired_sparsity + 2 if available, otherwise defaults to
            max(10, 5% of total features). Ensures it does not exceed half
            of total features.

        Returns
        -------
        Dict[str, Any]
            Test result containing status, message, and details.
        """
        if top_n_features is None:
            top_n_features = self._get_default_top_n_features()
        # Ensure n_features doesn't exceed half of total features
        top_n_features = min(top_n_features, int(self.MAX_TOP_FEATURES_FRACTION * self.n_features))
        test_name = f'Ordering of top {top_n_features} features'

        # Get mu values for first and last iteration
        first_mu = self.mu_array[0]  # shape: [n_components, n_features]
        last_mu = self.mu_array[-1]  # shape: [n_components, n_features]

        n_failed_components = 0
        n_warning_components = 0
        component_details = []

        for component in range(self.n_components):
            # Get top n_features indices based on absolute mu values
            first_top_indices = np.argsort(np.abs(first_mu[component]))[-top_n_features:]
            last_top_indices = np.argsort(np.abs(last_mu[component]))[-top_n_features:]

            # Convert to sets for comparison
            first_set = set(first_top_indices)
            last_set = set(last_top_indices)

            intersection = first_set.intersection(last_set)
            intersection_count = len(intersection)
            intersection_proportion = intersection_count / top_n_features

            # Determine component status
            if intersection_proportion > self.FAIL_PROPORTION_THRESHOLD:
                component_status = 'FAILED'
            elif intersection_proportion > self.WARNING_PROPORTION_THRESHOLD:
                component_status = 'WARNING'
            else:
                component_status = 'PASSED'

            component_details.append(
                {
                    'component': component,
                    'first_top_features': first_top_indices.tolist(),
                    'last_top_features': last_top_indices.tolist(),
                    'common_features': [int(x) for x in intersection],
                    'intersection_count': intersection_count,
                    'intersection_proportion': intersection_proportion,
                    'component_status': component_status,
                    'jaccard_similarity': intersection_count / len(first_set.union(last_set)),
                }
            )

            if component_status == 'FAILED':
                n_failed_components += 1
            elif component_status == 'WARNING':
                n_warning_components += 1

        # Determine the test status
        status = self._determine_test_status(n_failed_components, n_warning_components)

        if status == 'FAILED':
            message = (
                f'Significant number of components have identical top {top_n_features} features between first and last iteration:'
                + self._format_component_summary(n_failed_components, n_warning_components)
            )
        elif status == 'WARNING':
            message = (
                f'Some components have identical top {top_n_features} features between first and last iteration:'
                + self._format_component_summary(n_failed_components, n_warning_components)
            )
        else:
            message = f'The components have sufficiently different top {top_n_features} features between first and last iteration'

        test_result = {
            'test_name': test_name,
            'status': status,
            'message': message,
            'details': {
                'n_features_tested': top_n_features,
                'identical_components': n_failed_components,
                'total_components': self.n_components,
                'component_details': component_details,
            },
        }

        self.test_results.append(test_result)
        return test_result

    def test_sparsity_gap(
        self,
        desired_sparsity: int | None = None,
        difference_coefficient_boundary: float | None = None,
        difference_coefficient_extreme: float = 2.0,
    ) -> dict[str, Any]:
        """
        Test whether there is sufficient separation between selected and non-selected features
        based on their absolute mu values in the last iteration.

        This test examines whether the features selected based on desired sparsity are
        clearly distinguishable from the non-selected features. Poor separation may
        indicate insufficient regularization in the prior's setting or too few iterations.

        Parameters
        ----------
        desired_sparsity : Optional[int], optional
            Number of features to select. If None, uses self.desired_sparsity.
            If that's also not set, defaults to min(10, 25% of n_features).
            Default is None.
        difference_coefficient_boundary : Optional[float], optional
            Coefficient to determine the threshold for required difference
            at the feature selection boundary. Default is 1.0 / desired_sparsity.
            I.e. the difference at the boundary should be at least
            1/desired_sparsity * range of selected features.
        difference_coefficient_extreme : float, optional
            Coefficient to determine the threshold for minimal difference between
            the most and least important selected features. Default is 2.0, i.e.
            the difference of the extremes should be at least 2 * the range of selected features.

        Returns
        -------
        Dict[str, Any]
            Test result containing status, message, and details.
        """
        if desired_sparsity is None:
            desired_sparsity = self._get_default_sparsity()

        # Ensure desired_sparsity doesn't exceed total features
        desired_sparsity = min(desired_sparsity, self.n_features - 1)
        test_name = f'Sparsity gap (desired number of solutions = {desired_sparsity}) - WARNING: WORK IN PROGRESS!'

        if difference_coefficient_boundary is None:
            difference_coefficient_boundary = 1.0 / desired_sparsity

        # Get mu values for last iteration
        last_mu = self.mu_array[-1]  # shape: [n_components, n_features]

        n_failed_components = 0
        n_warning_components = 0
        component_details = []

        for component in range(self.n_components):
            # Get absolute mu values and sort them in descending order
            abs_mu = np.abs(last_mu[component])
            sorted_indices = np.argsort(abs_mu)[::-1]  # descending order
            sorted_values = abs_mu[sorted_indices]

            # Extract twice the 'desired sparsity' number of features
            extended_sparsity = min(
                self.EXTENDED_SPARSITY_MULTIPLIER * desired_sparsity, self.n_features
            )
            top_features_values = sorted_values[:extended_sparsity]
            selected_values = top_features_values[:desired_sparsity]
            not_selected_values = top_features_values[desired_sparsity:]

            # Difference between selected and not selected features
            last_selected_value = selected_values[-1]
            first_not_selected_value = not_selected_values[0]
            boundary_diff = last_selected_value - first_not_selected_value

            # Difference between the first selected and last not selected features
            first_selected_value = selected_values[0]
            last_not_selected_value = not_selected_values[-1]
            extreme_diff = first_selected_value - last_not_selected_value

            # Difference between first and last selected features
            selected_diff = first_selected_value - last_selected_value

            # Calculate the thresholds relative to the range of selected features
            relative_threshold_boundary = difference_coefficient_boundary * selected_diff
            relative_threshold_extreme = difference_coefficient_extreme * selected_diff

            # Determine component status
            if extreme_diff <= relative_threshold_extreme:
                component_status = 'FAILED'
            elif boundary_diff <= relative_threshold_boundary:
                component_status = 'WARNING'
            else:
                component_status = 'PASSED'

            # Count component failures/warnings
            if component_status == 'FAILED':
                n_failed_components += 1
            elif component_status == 'WARNING':
                n_warning_components += 1

            component_details.append(
                {
                    'component': component,
                    'last_selected_value': last_selected_value,
                    'first_not_selected_value': first_not_selected_value,
                    'boundary_diff': boundary_diff,
                    'extreme_diff': extreme_diff,
                    'coefficient_boundary': difference_coefficient_boundary,
                    'coefficient_extreme': difference_coefficient_extreme,
                    'relative_threshold_boundary': relative_threshold_boundary,
                    'relative_threshold_extreme': relative_threshold_extreme,
                    'component_status': component_status,
                    'top_features_values': selected_values.tolist(),
                    'other_features_values': not_selected_values.tolist(),
                }
            )

        # Determine overall test status
        status = self._determine_test_status(n_failed_components, n_warning_components)

        if status == 'FAILED':
            message = (
                'Significant number of components have poor separation between selected and not selected features:\n'
                + self._format_component_summary(n_failed_components, n_warning_components)
            )
        elif status == 'WARNING':
            message = (
                'Some components have poor separation between selected and not selected features:\n'
                + self._format_component_summary(n_failed_components, n_warning_components)
            )
        else:
            message = (
                'All components have good separation between selected and not selected features '
            )

        test_result = {
            'test_name': test_name,
            'status': status,
            'message': message + ' WARNING: WORK IN PROGRESS! RESULTS MAY BE TOO PESIMISTIC!',
            'details': {
                'desired_sparsity': desired_sparsity,
                'failed_components': n_failed_components,
                'warning_components': n_warning_components,
                'total_components': self.n_components,
                'component_details': component_details,
            },
        }

        self.test_results.append(test_result)
        return test_result

    def _determine_test_status(
        self,
        n_failed_components: int,
        n_warning_components: int,
    ) -> str:
        """
        Determine overall test status based on component failure/warning counts.

        Parameters
        ----------
        n_failed_components : int
            Number of components that failed the test.
        n_warning_components : int
            Number of components that triggered warnings.

        Returns
        -------
        str
            Overall test status: "FAILED", "WARNING", or "PASSED".
        """
        fail_score = (
            self.FAIL_WEIGHT * n_failed_components + self.WARNING_WEIGHT * n_warning_components
        )
        test_fail_threshold = self.n_components * self.COMPONENT_FAIL_THRESHOLD_RATIO
        test_warn_threshold = test_fail_threshold * self.COMPONENT_WARNING_THRESHOLD_RATIO

        if fail_score >= test_fail_threshold:
            return 'FAILED'
        elif (fail_score >= test_warn_threshold) or (n_failed_components > 0):
            return 'WARNING'
        else:
            return 'PASSED'

    def _format_component_summary(
        self,
        n_failed: int,
        n_warning: int,
    ) -> str:
        """
        Format component summary for test messages.

        Parameters
        ----------
        n_failed : int
            Number of components that failed.
        n_warning : int
            Number of components with warnings.

        Returns
        -------
        str
            Formatted summary string with component counts.
        """
        n_passed = self.n_components - n_failed - n_warning
        return (
            f'\n - {n_failed} components failed,'
            f'\n - {n_warning} components with warnings,'
            f'\n - {n_passed} components passed,'
            f'\n - out of {self.n_components} total components.'
        )

    def _get_default_top_n_features(self) -> int:
        """
        Get default number of top features to compare.

        Returns
        -------
        int
            Default number of top features. Uses desired_sparsity + 2 if available,
            otherwise max(DEFAULT_TOP_FEATURES_MIN, 5% of total features).
        """
        if hasattr(self, 'desired_sparsity'):
            return self.desired_sparsity + 2
        return max(
            self.DEFAULT_TOP_FEATURES_MIN,
            int(self.DEFAULT_TOP_FEATURES_COEFFICIENT * self.n_features),
        )

    def _get_default_sparsity(self) -> int:
        """
        Get default sparsity value.

        Returns
        -------
        int
            Default sparsity value. Uses self.desired_sparsity if available,
            otherwise min(DEFAULT_TOP_FEATURES_MIN, 25% of total features).
        """
        if hasattr(self, 'desired_sparsity'):
            return self.desired_sparsity
        return min(
            self.DEFAULT_TOP_FEATURES_MIN,
            int(self.DEFAULT_SPARSITY_FRACTION * self.n_features),
        )

    def _display_top_feature_ordering(
        self,
        details: dict[str, Any],
    ) -> None:
        """
        Display detailed results for the top feature ordering test.

        Parameters
        ----------
        details : Dict[str, Any]
            Test details to display.

        Returns
        -------
        None
            Displays detailed test results as markdown output.
        """
        display(Markdown('#### Component Analysis:'))

        for comp_detail in details['component_details']:
            component = comp_detail['component']
            component_status = comp_detail['component_status']
            jaccard = comp_detail['jaccard_similarity']

            status_icon = self.COMPONENT_STATUS_ICONS.get(component_status, '❓')
            display(Markdown(f'**Component {component}** {status_icon}'))
            display(
                Markdown(
                    f' - {comp_detail["intersection_count"]}/{details["n_features_tested"]} common features'
                )
            )
            if comp_detail['intersection_count'] == 0:
                display(Markdown('- **Component passed.**'))
            elif comp_detail['intersection_count'] == details['n_features_tested']:
                display(Markdown(f' - Common features: {comp_detail["common_features"]}'))
                display(Markdown(' - **Component failed.**'))
            else:
                display(Markdown(f' - Common features: {comp_detail["common_features"]}'))
                display(Markdown(f' - Jaccard similarity = {jaccard:.3f}'))

    def _display_sparsity_gap(self, details: dict[str, Any]) -> None:
        """
        Display detailed results for the sparsity gap test.

        Parameters
        ----------
        details : Dict[str, Any]
            Test details to display.

        Returns
        -------
        None
            Displays detailed test results as markdown output.
        """
        display(Markdown('#### Component Analysis:'))

        for comp_detail in details['component_details']:
            component = comp_detail['component']
            component_status = comp_detail['component_status']
            status_icon = self.COMPONENT_STATUS_ICONS.get(component_status, '❓')

            display(Markdown(f'**Component {component}** {status_icon}'))

            # Status-specific messages
            if component_status == 'FAILED':
                threshold = comp_detail['relative_threshold_extreme']
                current_diff = comp_detail['extreme_diff']
                display(
                    Markdown(
                        f'  - Small difference between best and worst feature: {current_diff:.4f} ≤ {threshold:.4f}'
                    )
                )
            elif component_status == 'WARNING':
                threshold = comp_detail['relative_threshold_boundary']
                current_diff = comp_detail['boundary_diff']
                display(
                    Markdown(
                        f'  - Small gap at feature selection boundary: {current_diff:.4f} ≤ {threshold:.4f}'
                    )
                )
            else:
                display(Markdown('  - Good separation of selected and not selected features.'))

            # Show feature values for failed/warning components
            if component_status in ['FAILED', 'WARNING']:
                selected_values = [np.round(v, 4) for v in comp_detail['top_features_values']]
                other_values = [np.round(v, 4) for v in comp_detail['other_features_values']]
                display(
                    Markdown(
                        f' - Top {details["desired_sparsity"]} feature |μ| values: {selected_values}\n'
                        f' - {len(other_values)} other compared values: {other_values}'
                    )
                )

    def _display_results(self) -> None:
        """
        Display all test results in a formatted way.

        Returns
        -------
        None
            Displays test results as markdown output.
        """
        display(Markdown('# Testing algorithm performance (work in progress)'))
        display(Markdown(f'**Total tests run:** {len(self.test_results)}'))

        failed_tests = [r for r in self.test_results if r['status'] == 'FAILED']
        warning_tests = [r for r in self.test_results if r['status'] == 'WARNING']
        passed_tests = [r for r in self.test_results if r['status'] == 'PASSED']

        display(
            Markdown(
                f'**Failed:** {len(failed_tests)} | **Warnings:** {len(warning_tests)} | **Passed:** {len(passed_tests)}'
            )
        )

        # Display failed tests first
        if failed_tests:
            display(Markdown('## ❌ FAILED TESTS'))
            for result in failed_tests:
                self._display_single_result(result)

        # Display warning tests
        if warning_tests:
            display(Markdown('## ⚠️ WARNINGS'))
            for result in warning_tests:
                self._display_single_result(result)

        # Display passed tests
        if passed_tests:
            display(Markdown('## ✅ PASSED TESTS'))
            for result in passed_tests:
                self._display_single_result(result)

    def _display_single_result(self, result: dict[str, Any]) -> None:
        """
        Display a single test result.

        Parameters
        ----------
        result : Dict[str, Any]
            Test result to display.

        Returns
        -------
        None
            Displays single test result as markdown output.
        """
        emoji = self.TEST_RESULT_EMOJIS.get(result['status'], '❓')

        display(Markdown(f'### {emoji} {result["test_name"]}'))
        display(Markdown(f'**Status:** {result["status"]}'))
        display(Markdown(f'**Message:** {result["message"]}'))

        # Display specific details based on test type
        if (
            result['test_name'].startswith('Ordering of top')
            and 'component_details' in result['details']
        ):
            self._display_top_feature_ordering(result['details'])
        elif (
            result['test_name'].startswith('Sparsity gap')
            and 'component_details' in result['details']
        ):
            self._display_sparsity_gap(result['details'])


def run_performance_diagnostics(
    history: dict[str, list[Any]],
    desired_sparsity: int | float | None = None,
    verbose: bool | None = True,
) -> PerformanceTests:
    """
    Convenience function to run performance diagnostics on optimization history.

    Parameters
    ----------
    history : Dict[str, List[Any]]
        The optimization history from BayesianFeatureSelector.optimize().
        Expected to contain keys: 'mu', 'var', 'alpha', 'elbo'.
    desired_sparsity : Optional[Union[int, float]], optional
        The desired sparsity level for feature selection. If float, should be
        between 0 and 1 representing fraction of features. If int, should be
        positive representing absolute number of features. Default is None.
    verbose : Optional[bool]
        Whether to display test results. Default is True.

    Returns
    -------
    PerformanceTests
        The performance tests instance with test results.
    """
    diagnostics = PerformanceTests(
        history=history,
        desired_sparsity=desired_sparsity,
    )
    diagnostics.run_all_tests(
        verbose=verbose,
    )
    return diagnostics
