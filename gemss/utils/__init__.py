"""
Utility functions for the GEMSS package:

- Data manipulation utilities.
- File input/output operations.
- Visualization tools for results interpretation.
"""

from gemss.utils.utils import (
    SelectorHistory,
    dataframe_to_ascii_table,
    display_feature_lists,
    format_summary_row_feature_with_mu,
    generate_feature_names,
    get_solution_summary_df,
    load_constants_json,
    load_feature_lists_json,
    load_selector_history_json,
    myprint,
    save_constants_json,
    save_feature_lists_json,
    save_feature_lists_txt,
    save_selector_history_json,
    show_solution_summary,
)
from gemss.utils.visualizations import (
    compare_parameters,
    get_algorithm_progress_plots,
    get_alpha_plot,
    get_compare_parameters_plot,
    get_confusion_matrix_plot,
    get_correlation_matrix_plot,
    get_correlation_with_response_plot,
    get_elbo_plot,
    get_features_in_components_plot,
    get_final_alphas_plot,
    get_label_histogram_plot,
    get_label_piechart,
    get_mu_plot,
    get_predicted_vs_actual_response_plot,
    get_subsampled_history,
    plot_alpha,
    plot_elbo,
    plot_mu,
    show_algorithm_progress,
    show_confusion_matrix,
    show_correlation_matrix,
    show_correlations_with_response,
    show_features_in_components,
    show_final_alphas,
    show_label_histogram,
    show_predicted_vs_actual_response,
)
