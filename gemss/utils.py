"""
Utility functions for feature selection project.
Includes:
- Sampling
- Metrics
- Data handling
- Display utilities
"""

from IPython.display import display, Markdown
from typing import Any, Dict


def print_nice_optimization_settings(
    n_components: int,
    regularize: bool,
    lambda_jaccard: float,
    regularization_threshold: float,
    n_iterations: int,
    prior_settings: Dict[str, Any],
) -> None:
    """
    Print the optimization settings for the Bayesian Feature Selector.

    Parameters
    ----------
    n_components : int
        Number of mixture components (desired solutions).
    regularize : bool
        Whether regularization is applied.
    lambda_jaccard : float
        Regularization strength for Jaccard similarity penalty.
    regularization_threshold : float
        Threshold for support computation.
    n_iterations : int
        Number of optimization iterations.
    prior_settings : Dict[str, Any]
        Dictionary containing prior settings such as prior name and parameters.

    Returns
    -------
    None
    """
    display(Markdown(f"#### Running Bayesian Feature Selector:"))
    display(Markdown(f"- desired number of solutions: {n_components}"))
    display(Markdown(f"- number of iterations: {n_iterations}"))

    if regularize:
        display(Markdown("- regularization parameters:"))
        display(Markdown(f"  - Jaccard penalization: {lambda_jaccard}"))
        display(Markdown(f"  - threshold for support: {regularization_threshold}"))
    else:
        display(Markdown("- no regularization"))

    display(Markdown("##### Algorithm settings:"))

    prior_name = prior_settings.get("prior_name", "N/A")
    prior_settings_to_display = {"prior name": prior_name}
    if prior_name == "StructuredSpikeAndSlabPrior":
        prior_settings_to_display.update(
            {
                "prior_sparsity": prior_settings.get("prior_sparsity", "N/A"),
                "var_slab": prior_settings.get("var_slab", "N/A"),
                "var_spike": prior_settings.get("var_spike", "N/A"),
            }
        )
    if prior_name == "SpikeAndSlabPrior":
        prior_settings_to_display.update(
            {
                "var_slab": prior_settings.get("var_slab", "N/A"),
                "var_spike": prior_settings.get("var_spike", "N/A"),
                "weight_slab": prior_settings.get("weight_slab", "N/A"),
                "weight_spike": prior_settings.get("weight_spike", "N/A"),
            }
        )
    elif prior_name == "StudentTPrior":
        prior_settings_to_display.update(
            {
                "student_df": prior_settings.get("student_df", "N/A"),
                "student_scale": prior_settings.get("student_scale", "N/A"),
            }
        )

    for key, value in prior_settings_to_display.items():
        display(Markdown(f" - {key.lower()}: {value}"))

    return
