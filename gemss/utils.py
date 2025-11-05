"""
Utility functions for feature selection project.
"""

from IPython.display import display, Markdown
from typing import Any, Dict, Optional, List
import pandas as pd


# TODO: move to a more general utils module along with other printing functions
# TODO: add use_markdown parameter and use myprint function in print_nice_optimization_settings
# TODO: set up a global parameter defining whether to use markdown or not (logging in plain text files during batch runs etc.)
# TODO: edit README accordingly


def myprint(
    msg: str,
    use_markdown: Optional[bool] = True,
    bold: Optional[bool] = False,
    header: Optional[int] = 0,
    code: Optional[bool] = False,
    file: Optional[Any] = None,
) -> None:
    """
    Print a message using Markdown formatting if specified, otherwise in plain text.

    Parameters
    ----------
    msg : str
        The message to print.
    use_markdown : bool
        Whether to use Markdown formatting rather than plain text.
    bold : bool, optional
        Whether to make the message bold. Default is False.
    header : int, optional
        The header level (1-6) for the message. Default is 0 (no header).
    file : Any, optional
        The file to print the message to. Default is None (prints to stdout).

    Returns
    -------
    None
    """
    if use_markdown:
        if header > 0:
            msg = f"{'#' * header} {msg}"
        if bold:
            msg = f"**{msg}**"
        if code:
            msg = f"```{msg}```"
        display(Markdown(msg))
    else:
        if header > 0:
            print("\n", file=file)
        print(msg, file=file)
        if header > 0:
            print("-" * len(msg), file=file)
    return


def format_summary_row_feature_with_mu(row):
    """Formatting function for outlier information."""
    return f"{row['Feature']} (mu = {row['Mu value']:.4f})"


def get_solution_summary_df(
    data_dict: Dict[str, pd.DataFrame],
    value_column: str = "Feature",
    format_function: Optional[callable] = format_summary_row_feature_with_mu,
) -> pd.DataFrame:
    """
    Convert a dictionary of DataFrames into a summary DataFrame where each column
    corresponds to a key and contains values from the specified column.

    This is a generalized function that can be used for both long solutions
    and outlier summaries.

    Parameters
    ----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping component/solution names to DataFrames.
        Each DataFrame should contain the specified value_column.
    value_column : str, optional
        Name of the column to extract values from. Default is "Feature".
    format_function : callable, optional
        Optional function to format each value. Should take a row and return a string.
        If None, uses the raw values from value_column. Default is None.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame where each column corresponds to a component and contains
        formatted values. Missing values are filled with NaN.

    Examples
    --------
    For long solutions:
    >>> solutions = {
    ...     "component_0": pd.DataFrame({"Feature": ["f1", "f2"], "Mu value": [0.5, 0.3]}),
    ...     "component_1": pd.DataFrame({"Feature": ["f3"], "Mu value": [0.7]})
    ... }
    >>> create_summary_dataframe(solutions)

    For outliers with formatting:
    >>> format_func = lambda row: f"{row['Feature']} ({row['Mu value']:.3f})"
    >>> create_summary_dataframe(solutions, format_function=format_func)
    """
    if not data_dict:
        return pd.DataFrame()

    # Find maximum length across all DataFrames
    max_length = max(len(df) for df in data_dict.values())

    # Early return if all DataFrames are empty
    if max_length == 0:
        return pd.DataFrame()

    summary_df = pd.DataFrame()

    for component_name, df in data_dict.items():
        if df.empty:
            # Handle empty DataFrames
            summary_df[component_name] = pd.Series([None] * max_length)
            continue

        # Extract and format values
        if format_function is not None:
            # Apply formatting function to each row
            formatted_values = df.apply(format_function, axis=1)
        else:
            # Use raw values from the specified column
            if value_column not in df.columns:
                raise ValueError(
                    f"Column '{value_column}' not found in DataFrame for {component_name}"
                )
            formatted_values = df[value_column]

        # Create padded series to ensure all columns have the same length
        padded_series = pd.Series([None] * max_length)
        padded_series.iloc[: len(formatted_values)] = formatted_values.values

        summary_df[component_name] = padded_series

    return summary_df


def show_solution_summary(
    solution_data: Dict[str, pd.DataFrame],
    title: str = "Solution summary",
    value_column: str = "Feature",
    format_function: Optional[callable] = None,
    use_markdown: Optional[bool] = True,
) -> None:
    """
    Display solutions or data in a DataFrame format where each column corresponds to a component
    and contains the identified features or formatted values.

    This is a generalized function that can display both long solutions and outlier summaries.

    Parameters
    ----------
    solution_data : Dict[str, pd.DataFrame]
        Dictionary mapping each component/solution to a DataFrame containing
        the data to display (e.g., features, outliers).
    title : str, optional
        Title for the displayed DataFrame. Default is "Solution summary".
    value_column : str, optional
        Name of the column to extract values from. Default is "Feature".
    format_function : callable, optional
        Optional function to format each value. Should take a row and return a string.
        If None, uses the raw values from value_column. Default is None.
    use_markdown : bool, optional
        Whether to format the title using Markdown. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If solution_data is empty or contains invalid DataFrames.

    Examples
    --------
    For long solutions:
    >>> solutions = {
    ...     "component_0": pd.DataFrame({"Feature": ["f1", "f2"], "Mu value": [0.5, 0.3]}),
    ...     "component_1": pd.DataFrame({"Feature": ["f3"], "Mu value": [0.7]})
    ... }
    >>> show_solution_summary(solutions, title="Long solutions")

    For formatted outliers:
    >>> format_func = lambda row: f"{row['Feature']} ({row['Mu value']:.4f})"
    >>> show_solution_summary(solutions, title="Outlier summary", format_function=format_func)
    """
    # Input validation
    if not solution_data:
        myprint(
            msg="No data available to display.",
            use_markdown=use_markdown,
            header=2,
        )
        return

    # Validate that all values are DataFrames with required columns
    for component_name, df in solution_data.items():
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame for {component_name}, got {type(df)}")
        if not df.empty and value_column not in df.columns and format_function is None:
            raise ValueError(
                f"DataFrame for {component_name} missing required '{value_column}' column"
            )

    try:
        summary_df = get_solution_summary_df(
            data_dict=solution_data,
            value_column=value_column,
            format_function=format_function,
        )

        # Check if any data was found
        if summary_df.empty:
            myprint(
                msg=f"{title}: No data found in any component.",
                use_markdown=use_markdown,
                header=2,
            )
            return

        myprint(
            msg=title,
            use_markdown=use_markdown,
            header=2,
        )

        if use_markdown:
            display(summary_df)
        else:
            print(summary_df)

    except Exception as e:
        myprint(
            msg=f"Error displaying data: {str(e)}",
            use_markdown=use_markdown,
        )
        raise

    return


def generate_feature_names(
    n_features: int,
    original_feature_names_mapping: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Helper function to generate feature names consistently.

    Parameters:
    -----------
    n_features: int
        The number of features to generate names for.
    original_feature_names_mapping: Optional[Dict[str, str]], optional
        A mapping from feature indices to original feature names. If provided, will use original names.

    Returns:
    --------
    List[str]
        A list of feature names.
    """
    if original_feature_names_mapping is not None:
        return [
            original_feature_names_mapping.get(f"feature_{i}", f"feature_{i}")
            for i in range(n_features)
        ]
    return [f"feature_{i}" for i in range(n_features)]


def print_nice_optimization_settings(
    n_components: int,
    regularize: bool,
    lambda_jaccard: float,
    n_iterations: int,
    prior_settings: Dict[str, Any],
) -> None:
    """
    Print the optimization settings for GEMSS feature selector.

    Parameters
    ----------
    n_components : int
        Number of mixture components (desired solutions).
    regularize : bool
        Whether regularization is applied.
    lambda_jaccard : float
        Regularization strength for Jaccard similarity penalty.
    n_iterations : int
        Number of optimization iterations.
    prior_settings : Dict[str, Any]
        Dictionary containing prior settings such as prior name and parameters.

    Returns
    -------
    None
    """
    display(Markdown(f"#### Running GEMSS feature selector:"))
    display(Markdown(f"- desired number of solutions: {n_components}"))
    display(Markdown(f"- number of iterations: {n_iterations}"))

    if regularize:
        display(Markdown("- regularization parameters:"))
        display(Markdown(f"  - Jaccard penalization: {lambda_jaccard}"))
    else:
        display(Markdown("- no regularization"))

    display(Markdown("##### GEMSS algorithm settings:"))

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
