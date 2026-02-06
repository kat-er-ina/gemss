"""
Modules focusing on assessment of experiments on artificial data.

Used only for development of GEMSS and the corresponding publication of results.
"""

from .case_analysis import case2set, concatenate_cases, get_df_cases
from .experiment_results_analysis import (
    analyze_metric_results,
    choose_best_solution_per_group,
    compute_performance_overview,
    filter_df_best_solutions,
    get_all_experiment_results,
    get_average_metrics_per_case,
    get_average_metrics_per_group,
    get_best_solution_type_per_group,
    load_experiment_results,
    pivot_df_by_solution_type,
    print_dataframe_overview,
    show_performance_overview,
)
from .experiment_results_interactive import (
    show_interactive_comparison_with_grouping,
    show_interactive_heatmap,
    show_interactive_performance_overview,
    show_interactive_si_asi_comparison,
    show_interactive_solution_comparison,
)
from .experiment_results_visualizations import (
    plot_category_counts,
    plot_heatmap,
    plot_metric_analysis_overview,
    plot_metric_vs_hyperparam,
    plot_si_asi_scatter,
    plot_solution_comparison,
    plot_solution_grouped,
)
