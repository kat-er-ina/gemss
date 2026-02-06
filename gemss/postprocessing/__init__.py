"""
This module aggregates all postprocessing and downstream modeling functionalities, including:
- Outlier detection and handling
- Extraction of solutions from the optimization run
- Simple regression analyses
"""

from .outliers import (
    detect_outlier_features,
    get_outlier_info_df,
    get_outlier_solutions,
    get_outlier_summary_from_history,
    show_outlier_features_by_component,
    show_outlier_info,
    show_outlier_summary,
)
from .result_postprocessing import (
    compare_true_and_found_features,
    get_features_from_long_solutions,
    get_features_from_solutions,
    get_full_solutions,
    get_top_solutions,
    get_unique_features,
    recover_solutions,
    show_algorithm_progress_with_outliers,
    show_features_in_solutions,
    show_final_parameter_comparison,
    show_unique_features,
    show_unique_features_from_full_solutions,
)
from .simple_regressions import (
    detect_task,
    print_verbose_linear_regression_results,
    print_verbose_logistic_regression_results,
    show_regression_metrics,
    solve_any_regression,
    solve_with_linear_regression,
    solve_with_logistic_regression,
)
from .tabpfn_evaluation import (
    Predictable,
    classification_metrics,
    regression_metrics,
    tabpfn_evaluate,
)
