"""
Utility functions for feature selection project.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, TypedDict

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

# TODO: move to a more general utils module along with other printing functions
# TODO: add use_markdown parameter and use myprint function in print_nice_optimization_settings
# TODO: set up a global parameter defining whether to use markdown or not (logging in plain text files during batch runs etc.) # noqa: E501
# TODO: edit README accordingly


def myprint(
    msg: str,
    use_markdown: bool = True,
    bold: bool = False,
    header: int = 0,
    code: bool = False,
    file: TextIO | None = None,
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
            msg = f'{"#" * header} {msg}'
        if bold:
            msg = f'**{msg}**'
        if code:
            msg = f'```{msg}```'
        display(Markdown(msg))
    else:
        if header > 0:
            print('\n', file=file)
        print(msg, file=file)
        if header > 0:
            print('-' * len(msg), file=file)
    return


def format_summary_row_feature_with_mu(row: pd.Series) -> str:
    """Formatting function for outlier information."""
    return f'{row["Feature"]} (mu = {row["Mu value"]:.4f})'


def get_solution_summary_df(
    data_dict: dict[str, pd.DataFrame],
    value_column: str = 'Feature',
    format_function: Callable[[pd.Series], str] | None = format_summary_row_feature_with_mu,
) -> pd.DataFrame:
    """
    Convert a dictionary of DataFrames into a summary DataFrame where each column
    corresponds to a key and contains values from the specified column.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]
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
    solution_data: dict[str, pd.DataFrame],
    title: str = 'Solution summary',
    value_column: str = 'Feature',
    format_function: Callable[[pd.Series], str] | None = format_summary_row_feature_with_mu,
    use_markdown: bool = True,
) -> None:
    """
    Display solutions or data in a DataFrame format where each column corresponds to a component
    and contains the identified features or formatted values.

    This is a generalized function that can display both long solutions and outlier summaries.

    Parameters
    ----------
    solution_data : dict[str, pd.DataFrame]
        Dictionary mapping each component/solution to a DataFrame containing
        the data to display (e.g., features, outliers).
    title : str, optional
        Title for the displayed DataFrame. Default is "Solution summary".
    value_column : str, optional
        Name of the column to extract values from. Default is "Feature".
    format_function : callable, optional
        Optional function to format each value. Should take a row and return a string.
        If None, uses the raw values from value_column. Default is the callable
        format_summary_row_feature_with_mu, which formats the feature with its Mu value.
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
            msg='No data available to display.',
            use_markdown=use_markdown,
            header=2,
        )
        return

    # Validate that all values are DataFrames with required columns
    for component_name, df in solution_data.items():
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f'Expected DataFrame for {component_name}, got {type(df)}')
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
                msg=f'{title}: No data found in any component.',
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
            msg=f'Error displaying data: {str(e)}',
            use_markdown=use_markdown,
        )
        raise

    return


def generate_feature_names(
    n_features: int,
    original_feature_names_mapping: dict[str, str] | None = None,
) -> list[str]:
    """
    Helper function to generate feature names consistently.

    Parameters:
    -----------
    n_features: int
        The number of features to generate names for.
    original_feature_names_mapping: dict[str, str] | None, optional
        A mapping from feature indices to original feature names.
        If provided, will use original names.

    Returns:
    --------
    list[str]
        A list of feature names.
    """
    if original_feature_names_mapping is not None:
        return [
            original_feature_names_mapping.get(f'feature_{i}', f'feature_{i}')
            for i in range(n_features)
        ]
    return [f'feature_{i}' for i in range(n_features)]


def dataframe_to_ascii_table(
    df: pd.DataFrame,
    title: str | None = None,
    max_col_width: int | None = None,
    precision: int = 3,
) -> list[str]:
    """
    Convert a pandas DataFrame to a nicely formatted ASCII table using | and _.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert to ASCII table format.
    title : str, optional
        Optional title to display above the table. Default is None.
    max_col_width : int, optional
        Maximum width for any column. Long content will be truncated if specified.
        If None, columns will be sized to fit their content without truncation. Default is None.
    precision : int, optional
        Number of decimal places for floating point numbers. Default is 3.

    Returns
    -------
    list[str]
        List of strings representing each line of the ASCII table.
    """

    if df.empty:
        lines = []
        if title:
            lines.append(title)
            lines.append('')
        lines.append('No data to display.')
        return lines

    # Format the data with proper precision for floats
    formatted_df = df.copy()
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            formatted_df[col] = df[col].apply(
                lambda x: f'{x:.{precision}f}' if pd.notnull(x) else ''
            )
        else:
            formatted_df[col] = df[col].astype(str)

    # Prepare column data
    columns = ['Index'] + list(formatted_df.columns)
    rows_data = []

    # Add index and data rows
    for idx, row in formatted_df.iterrows():
        row_data = [str(idx)] + [str(val) for val in row.values]
        rows_data.append(row_data)

    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        # Start with column header width
        max_width = len(col)
        # Check all data in this column
        for row_data in rows_data:
            if i < len(row_data):
                max_width = max(max_width, len(str(row_data[i])))
        # Apply maximum width constraint only if specified
        if max_col_width is not None:
            col_widths.append(min(max_width, max_col_width))
        else:
            col_widths.append(max_width)

    # Truncate content that's too long (only if max_col_width is specified)
    def truncate_content(content: str, width: int) -> str:
        if max_col_width is None or len(content) <= width:
            return content
        return content[: width - 3] + '...'

    # Build the table
    lines = []

    # Add title if provided
    if title:
        lines.append(title)
        lines.append('=' * len(title))
        lines.append('')

    # Create header row
    header_parts = []
    for i, col in enumerate(columns):
        truncated_col = truncate_content(col, col_widths[i])
        header_parts.append(truncated_col.ljust(col_widths[i]))
    lines.append('| ' + ' | '.join(header_parts) + ' |')

    # Create separator row
    separator_parts = []
    for width in col_widths:
        separator_parts.append('_' * width)
    lines.append('| ' + ' | '.join(separator_parts) + ' |')

    # Create data rows
    for row_data in rows_data:
        row_parts = []
        for i, width in enumerate(col_widths):
            if i < len(row_data):
                content = str(row_data[i])
                truncated_content = truncate_content(content, width)
                row_parts.append(truncated_content.ljust(width))
            else:
                row_parts.append(' ' * width)
        lines.append('| ' + ' | '.join(row_parts) + ' |')

    return lines


def display_feature_lists(
    features_dict: dict[str, list[str]],
    title: str = 'Feature lists for candidate solutions',
    use_markdown: bool = True,
) -> None:
    myprint(msg=title, header=3, use_markdown=use_markdown)
    for component, selected_features in features_dict.items():
        myprint(msg=component, header=4, use_markdown=use_markdown)
        myprint(msg=f'{selected_features}', code=True, use_markdown=use_markdown)


def save_feature_lists_txt(
    all_features_lists: dict[str, dict[str, list[str]]],
    filename: str,
) -> str:
    """Save multiple dictionaries of candidate feature lists to a plain text file.

    Each dictionary in ``all_feature_dicts`` is written under a heading defined by the
    corresponding entry in ``feature_dict_titles``. Within each dictionary, every key
    (component identifier) is written in uppercase as a sub-heading, followed by the
    Python list of its features on the next line. A separator line is appended at the end.

    The file is (re)created; existing contents are overwritten.

    Parameters
    ----------
    all_features_lists : dict[str, dict[str, list[str]]]
        A dictionary mapping feature list titles to their corresponding feature lists. Structure:
        { solution_type: {component1: [feat1, feat2], component2: [feat3]}, title2: {...}, ... }
    filename : str
        Path (or filename) of the text file to create.

    Returns
    -------
    str
        A status message.

    Raises
    ------
    ValueError
        If ``all_feature_dicts`` and ``feature_dict_titles`` have different lengths
        or if ``all_feature_dicts`` is empty.

    Examples
    --------
    >>> dicts = [
    ...     {"component_0": ["feat_a", "feat_b"], "component_1": ["feat_c"]},
    ...     {"component_0": ["feat_x"]}
    ... ]
    >>> titles = ["Full solutions", "Top solutions"]
    >>> save_feature_lists_txt(dicts, titles, "solutions.txt")
    'Candidate solutions saved to file solutions.txt.'
    """
    if not all_features_lists:
        raise ValueError("'all_features_lists' is empty; nothing to save.")

    try:
        with Path(filename).open('w', encoding='utf-8') as f:
            for title, feature_dict in all_features_lists.items():
                f.write(f'### {title}\n\n')
                for component, features in feature_dict.items():
                    f.write(f'{component.upper()}\n')
                    f.write(f'{features}\n\n')
            f.write('------------------------------------------------------------------------\n')
        return f'Candidate solutions saved to TXT file for viewing: {filename}.'
    except Exception as e:
        return f'Problem saving to file {filename}: {e}'


def save_feature_lists_json(
    all_features_lists: dict[str, dict[str, list[str]]],
    filename: str,
) -> str:
    """Save multiple dictionaries of candidate feature lists to a structured JSON file.

    Produces a JSON document with a top-level key ``sections`` which is a list of
    objects. Each object contains the keys ``title`` (taken from ``feature_dict_titles``)
    and ``components`` (a mapping from component identifiers to their list of feature
    names). Existing file contents (if any) are overwritten.

    Parameters
    ----------
    all_features_lists : dict[str, dict[str, list[str]]]
        A dictionary mapping feature list titles to their corresponding feature lists. Structure:
        { solution_type: {component1: [feat1, feat2], component2: [feat3]}, title2: {...}, ... }
    filename : str
        Path (or filename) of the JSON file to create.

    Returns
    -------
    str
        A status message.

    Raises
    ------
    ValueError
        If ``all_features_lists`` is empty.
    """
    if not all_features_lists:
        raise ValueError("'all_features_lists' is empty; nothing to save.")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('Filename must be a non-empty string.')

    sections: list[dict[str, object]] = []
    for title, feature_dict in all_features_lists.items():
        components_out: dict[str, list[str]] = {}
        for component, features in feature_dict.items():
            # Ensure list of strings
            if not isinstance(features, list):
                raise ValueError(
                    f"Features for component '{component}' are not a list: {type(features)}"
                )
            components_out[component] = [str(f) for f in features]
        sections.append({'title': title, 'components': components_out})

    data = {'sections': sections}
    try:
        with Path(filename).open('w', encoding='utf-8') as f:
            json.dump(data, f)
        return f'Candidate solutions saved to JSON file for further processing: {filename}.'
    except Exception as e:
        return f'Problem saving JSON file {filename}: {e}'


def load_feature_lists_json(
    filename: str,
) -> tuple[dict[str, dict[str, list[str]]], str]:
    """Load the dictionary of all candidate solutions saved by ``save_feature_lists_json``.

    Reads the JSON file produced by ``save_feature_lists_json`` and reconstructs
    the dictionary mapping solution type titles to their components and feature lists.

    Parameters
    ----------
    filename : str
        Path to the JSON file previously written by ``save_feature_lists_json``.

    Returns
    -------
    (all_features_lists, message) : tuple[dict[str, dict[str, list[str]]], str]
        ``all_features_lists`` with structure {
            solution_type_A: {
                component1: [feat1, feat2],
                component2: [feat3],
                ...
            },
            solution_type_B: {...},
            ...
        }
        ``message`` summarizes load status (file name, number of sections).

    Raises
    ------
    ValueError
        If filename is invalid or JSON structure malformed.
    FileNotFoundError
        If the file does not exist.
    JSONDecodeError
        If the file content is not valid JSON.

    Examples
    --------
    >>> feature_lists, msg = load_feature_lists_json("solutions.json")
    >>> isinstance(feature_lists, dict)
    True
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('Filename must be a non-empty string.')

    if not Path(filename).exists():
        raise FileNotFoundError(f"File '{filename}' not found.")
    with Path(filename).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Top-level JSON must be an object.')
    sections = data.get('sections')
    if not isinstance(sections, list):
        raise ValueError("JSON missing 'sections' list.")

    all_features_lists: dict[str, dict[str, list[str]]] = {}
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError(f'Section index {i} is not an object.')
        title = section.get('title')
        components = section.get('components')
        if title is None or not isinstance(title, str) or not title.strip():
            raise ValueError(f"Section index {i} missing valid 'title'.")
        if not isinstance(components, dict):
            raise ValueError(f"Section '{title}' missing 'components' dict.")

        component_dict: dict[str, list[str]] = {}
        for comp, feats in components.items():
            if not isinstance(comp, str):
                raise ValueError(f"Component key '{comp}' in section '{title}' not a string.")
            if not isinstance(feats, list):
                raise ValueError(
                    f"Features for component '{comp}' in section '{title}' not a list."
                )
            component_dict[comp] = [str(f) for f in feats]
        all_features_lists[title] = component_dict

    message = f"Feature lists loaded from '{filename}' | sections: {len(all_features_lists)}"
    return all_features_lists, message


class SelectorHistory(TypedDict):
    elbo: list[float]
    mu: list[np.ndarray]
    var: list[np.ndarray]
    alpha: list[np.ndarray]


def save_selector_history_json(history: SelectorHistory, filename: str) -> str:
    """Persist optimization history to a JSON file with basic validation.

    Parameters
    ----------
    history : dict[str, Any]
        Dictionary expected to contain lists/iterables under keys 'elbo', 'mu', 'var', 'alpha'.
    filename : str
        Path to JSON file to create/overwrite.

    Returns
    -------
    str
        Status message.
    """
    # Basic input validation
    if not isinstance(history, dict):
        return 'History must be a dict; nothing saved.'
    required_keys = {'elbo', 'mu', 'var', 'alpha'}
    missing = required_keys - set(history.keys())
    if missing:
        return f'History missing required keys: {sorted(missing)}; nothing saved.'
    if not isinstance(filename, str) or not filename.strip():
        return 'Filename must be a non-empty string; nothing saved.'

    try:
        n_iters = len(history.get('elbo', []))
    except Exception:
        return 'Could not determine number of iterations; nothing saved.'
    if n_iters == 0:
        return 'History is empty; nothing saved.'

    # Ensure length consistency among iterable keys
    try:
        lengths = {
            'elbo': len(history['elbo']),
            'mu': len(history['mu']),
            'var': len(history['var']),
            'alpha': len(history['alpha']),
        }
    except Exception:
        return 'History contains non-iterable values; nothing saved.'
    if len(set(lengths.values())) != 1:
        return f'Inconsistent lengths in history: {lengths}; nothing saved.'

    # Create parent directory if needed
    parent = Path(filename).resolve().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Could not create directory '{parent}': {e}"

    # Build serializable structure
    try:
        serializable_history = {
            'elbo': [float(x) for x in history['elbo']],
            'mu': [np.asarray(a).tolist() for a in history['mu']],
            'var': [np.asarray(a).tolist() for a in history['var']],
            'alpha': [np.asarray(a).tolist() for a in history['alpha']],
        }
    except Exception as e:
        return f'Failed to serialize history arrays: {e}'

    try:
        with Path(filename).open('w', encoding='utf-8') as jf:
            json.dump(serializable_history, jf)
        return f"Saved history to '{filename}'"
    except Exception as e:
        return f"Failed writing history to '{filename}': {e}"


def load_selector_history_json(
    filename: str,
) -> tuple[SelectorHistory, str]:
    """Load optimization history saved by ``save_selector_history_json``.

    Opens the JSON file, validates required keys (``elbo``, ``mu``, ``var``, ``alpha``),
    checks length consistency across iterations, and converts nested lists
    back to NumPy arrays for each iteration.

    Parameters
    ----------
    filename : str
        Path to the JSON file produced by ``save_selector_history_json``.

    Returns
    -------
    (history, message) : tuple[dict[str, Any], str]
        ``history`` is the loaded dictionary with NumPy arrays for ``mu``, ``var`` and ``alpha``.
        ``message`` is a human-readable status summary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON is missing required keys or lengths are inconsistent.
    JSONDecodeError
        If the file contents are not valid JSON.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('Filename must be a non-empty string.')

    if not Path(filename).exists():
        raise FileNotFoundError(f"History file '{filename}' not found.")
    with Path(filename).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('History JSON must be an object (dict) at top level.')

    required = {'elbo', 'mu', 'var', 'alpha'}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f'History JSON missing keys: {sorted(missing)}')

    # Length consistency check
    try:
        lengths = {k: len(data[k]) for k in required}
    except Exception as e:
        raise ValueError(f'Unable to determine lengths of history entries: {e}')
    if len(set(lengths.values())) != 1:
        raise ValueError(f'Inconsistent iteration lengths in history: {lengths}')
    n_iters = lengths['elbo']
    if n_iters == 0:
        return data, 'History file loaded but contains zero iterations.'

    # Always convert list entries back to NumPy arrays
    try:
        data['mu'] = [np.asarray(iter_mu) for iter_mu in data['mu']]
        data['var'] = [np.asarray(iter_var) for iter_var in data['var']]
        data['alpha'] = [np.asarray(iter_alpha) for iter_alpha in data['alpha']]
    except Exception as e:
        raise ValueError(f'Failed converting lists to arrays: {e}')

    message = (
        f"**History loaded from** '{filename}' | **iterations:** {n_iters} | "
        f'**data:** {sorted(data.keys())}'
    )
    return data, message


def save_constants_json(constants: dict[str, object], filename: str) -> str:
    """Persist constants dict to JSON after validation.

    Parameters
    ----------
    constants : dict[str, Any]
        Dictionary of configuration constants.
    filename : str
        Path to JSON file to create/overwrite.

    Returns
    -------
    str
        Status message.
    """
    if not isinstance(constants, dict):
        return 'Constants must be a dict; nothing saved.'
    if not constants:
        return 'Constants dict is empty; nothing saved.'
    if not isinstance(filename, str) or not filename.strip():
        return 'Filename must be a non-empty string; nothing saved.'
    # Test JSON serializability
    try:
        json.dumps(constants)
    except Exception as e:
        return f'Constants not JSON serializable: {e}'
    parent = Path(filename).resolve().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"Could not create directory '{parent}': {e}"
    try:
        with Path(filename).open('w', encoding='utf-8') as f:
            json.dump(constants, f)
        return f"Saved constants to '{filename}'"
    except Exception as e:
        return f"Failed writing constants to '{filename}': {e}"


def load_constants_json(filename: str) -> tuple[dict[str, object], str]:
    """Load configuration constants saved by ``save_constants_json``.

    Opens the JSON file, validates it exists and that the top-level object is a
    dictionary. Returns the loaded constants along with a status message.

    Parameters
    ----------
    filename : str
        Path to the JSON file produced by ``save_constants_json``.

    Returns
    -------
    (constants, message) : tuple[dict[str, Any], str]
        ``constants`` is the loaded dictionary of configuration values.
        ``message`` summarizes the load status (file, number of keys).

    Raises
    ------
    ValueError
        If ``filename`` is not a non-empty string or if the JSON top level is not a dict.
    FileNotFoundError
        If the file does not exist.
    JSONDecodeError
        If the file contents are not valid JSON.

    Examples
    --------
    >>> constants, msg = load_constants_json("search_setup.json")
    >>> isinstance(constants, dict)
    True
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('Filename must be a non-empty string.')

    if not Path(filename).exists():
        raise FileNotFoundError(f"Constants file '{filename}' not found.")
    with Path(filename).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('Constants JSON must have a dict as the top-level object.')
    n_keys = len(data)
    message = f"Constants loaded from '{filename}' | keys: {n_keys}"
    return data, message
