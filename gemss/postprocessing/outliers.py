"""
Module for detecting and displaying outlier features in candidate solutions
based on the z-score of their mu values.
"""

from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display

from gemss.utils.utils import (
    generate_feature_names,
    get_solution_summary_df,
    myprint,
    show_solution_summary,
)


def _create_outlier_info_for_component(
    history: dict[str, list[Any]],
    component: int,
    feature_names: list[str],
    outlier_threshold_coeff: float,
    use_medians_for_outliers: bool,
) -> dict[str, list[float]]:
    """
    Helper function to create outlier info for a single component.

    Parameters:
    -----------
    history : Dict[str, List[Any]]
        Optimization history containing 'mu' values.
    component : int
        Component index to analyze.
    feature_names : List[str]
        List of feature names.
    outlier_threshold_coeff : float
        Threshold coefficient for outlier detection.
    use_medians_for_outliers : bool
        Whether to use medians instead of means.

    Returns:
    --------
    Dict[str, List[float]]
        Dictionary containing outlier features and values.
    """
    if outlier_threshold_coeff <= 0:
        raise ValueError('Threshold coefficient must be positive')
    # Create series with feature names as index
    component_values = pd.Series(
        data=history['mu'][-1][component],
        index=feature_names,
    )

    return detect_outlier_features(
        component_values,
        threshold_coeff=outlier_threshold_coeff,
        use_median=use_medians_for_outliers,
        replace_middle_by_zero=True,
    )


def detect_outlier_features(
    values: pd.Series,
    threshold_coeff=3,
    use_median: bool | None = False,
    replace_middle_by_zero: bool | None = False,
) -> dict[str, list[float]]:
    """
    Detects outlier features based on their deviation from the mean (i.e. z-score) or median. Return the names of the outliers.

    Parameters:
    -----------
    values: pd.Series
        Series of mu values to analyze. Index of the series represents feature indices.
    threshold_coeff: float
        Threshold multiplier for the absolute deviation to identify outliers.
    use_median: bool, optional
        Whether to use median for outlier detection instead of mean. Using median is less sensitive to large
        values but it leads to larger deviation values, i.e. more outliers detected. Default is False.
    replace_middle_by_zero: bool, optional
        If True, uses zero instead of median or mean for outlier detection. Default is False.

    Returns:
    -----------
    Dict[str, List[float]]
        A dictionary with two keys:
        - "features": List of feature indices identified as outliers.
        - "values": Corresponding mu values of the outlier features.
    """
    # Input validation
    if threshold_coeff <= 0:
        raise ValueError('Threshold coefficient must be positive.')

    # Early return for empty data
    if values.empty:
        return {'features': [], 'values': []}

    # Calculate center point (middle value)
    if replace_middle_by_zero:
        center_value = 0
    elif use_median:
        center_value = values.median()
    else:  # use mean
        center_value = values.mean()

    # Calculate absolute deviations from center
    absolute_deviations = (values - center_value).abs()

    # Calculate threshold for outlier detection
    if use_median:
        deviation_threshold = absolute_deviations.median()
    else:  # use mean
        deviation_threshold = absolute_deviations.mean()

    # Identify outliers
    outlier_mask = absolute_deviations > threshold_coeff * deviation_threshold

    # Early return if no outliers found
    if not outlier_mask.any():
        return {'features': [], 'values': []}

    return {
        'features': values.index[outlier_mask].tolist(),
        'values': values[outlier_mask].tolist(),
    }


def get_outlier_info_df(
    outlier_info: dict[str, list[float]],
    component_no: int,
) -> pd.DataFrame:
    """
    Helper function to get outlier information as a DataFrame.

    Parameters:
    -----------
    outlier_info: Dict[str, List[float]]
        A dictionary with outlier features and their values.
    component_no: int
        The component number to get outlier information for.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the outlier features and their values for the specified component.
    """
    component_key = f'component_{component_no}'
    features = outlier_info[component_key]['features']
    values = outlier_info[component_key]['values']

    df = pd.DataFrame(
        {
            'Feature': features,
            'Mu value': np.round(values, 4),
        }
    )

    # Sort by absolute value (descending) without keeping the column
    df['_sort_key'] = np.abs(values)
    df = (
        df.sort_values(by='_sort_key', ascending=False)
        .drop(columns=['_sort_key'])
        .reset_index(drop=True)
    )

    return df


def show_outlier_info(
    outlier_info: dict[str, list[float]],
    component_numbers: int | list[int] | None = None,
    use_markdown: bool | None = True,
) -> None:
    """
    Display outlier information for specified components.

    Parameters:
    -----------
    outlier_info: Dict[str, List[float]]
        A dictionary with outlier features and their values.
    component_numbers: int or List[int], optional
        The component number(s) to print outlier information for.
        If None, shows all available components.
    use_markdown: bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns:
    --------
    None
    """
    if not outlier_info:
        myprint('No outlier information available.', use_markdown=use_markdown)
        return

    if component_numbers is None:
        # Extract all available component numbers
        component_numbers = [
            int(key.split('_')[1]) for key in outlier_info.keys() if key.startswith('component_')
        ]

    # Ensure component_numbers is always a list for consistent processing
    if isinstance(component_numbers, int):
        component_numbers = [component_numbers]

    # Convert outlier info to DataFrame format for display
    formatted_outlier_info = {}
    for comp_num in component_numbers:
        component_key = f'component_{comp_num}'
        if component_key in outlier_info:
            outlier_count = len(outlier_info[component_key]['features'])
            if outlier_count > 0:
                formatted_outlier_info[component_key] = pd.DataFrame(
                    {
                        'Feature': outlier_info[component_key]['features'],
                        'Mu value': outlier_info[component_key]['values'],
                    }
                )

    if not formatted_outlier_info:
        myprint('No outliers found in specified components.', use_markdown=use_markdown)
        return

    # Determine appropriate title
    if len(component_numbers) == 1:
        comp_num = component_numbers[0]
        outlier_count = len(outlier_info[f'component_{comp_num}']['features'])
        title = f'{outlier_count} outlier features in component {comp_num}'
    else:
        title = f'Outlier features in {len(component_numbers)} components'

    show_solution_summary(
        solution_data=formatted_outlier_info,
        title=title,
        value_column='Feature',
        use_markdown=use_markdown,
    )

    return None


def get_outlier_solutions(
    history: dict[str, list[Any]],
    use_medians_for_outliers: bool | None = False,
    outlier_threshold_coeff: float | None = 3.0,
    original_feature_names_mapping: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute the outliers features from the optimization history and summarize the results
    across all components.

    Parameters:
    -----------
    history : Dict[str, List[Any]]
        The optimization history containing 'mu' values.
    use_medians_for_outliers : bool, optional
        Whether to use medians for outlier detection. Default is False.
    outlier_threshold_coeff : float, optional
        The threshold coefficient for outlier detection. Default is 3.0.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        Mapping from feature indices to original names. Default is None.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary where each key is a component identifier (e.g., "component_0")
        and each value is a DataFrame containing outlier features for that component.
    """

    if 'mu' not in history or not history['mu']:
        raise KeyError("'mu' key not found in history or history['mu'] is empty")

    n_components = len(history['mu'][0])
    n_features = len(history['mu'][0][0])

    # Generate feature names using helper function
    feature_names = generate_feature_names(n_features, original_feature_names_mapping)

    # Process each component
    outlier_dataframes = {}
    for component in range(n_components):
        # Get outlier info for this component using helper function
        outlier_info = _create_outlier_info_for_component(
            history,
            component,
            feature_names,
            outlier_threshold_coeff,
            use_medians_for_outliers,
        )

        # Create outlier info dict in expected format
        outlier_info_dict = {f'component_{component}': outlier_info}
        component_df = get_outlier_info_df(outlier_info_dict, component)

        if not component_df.empty:
            outlier_dataframes[f'component_{component}'] = component_df

    return outlier_dataframes


def get_outlier_summary_from_history(
    history: dict[str, list[Any]],
    use_medians_for_outliers: bool | None = False,
    outlier_threshold_coeff: float | None = 3.0,
    original_feature_names_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compute the outliers features from the optimization history and return a summary DataFrame.

    Parameters:
    -----------
    history : Dict[str, List[Any]]
        The optimization history containing 'mu' values.
    use_medians_for_outliers : bool, optional
        Whether to use medians for outlier detection. Default is False.
    outlier_threshold_coeff : float, optional
        The threshold coefficient for outlier detection. Default is 3.0.
    original_feature_names_mapping : Optional[Dict[str, str]], optional
        Mapping from feature indices to original names. Default is None.

    Returns:
    --------
    pd.DataFrame
        A DataFrame summarizing outlier features across all components.
        Each column corresponds to a component and contains formatted outlier info.
    """
    outlier_dataframes = get_outlier_solutions(
        history=history,
        use_medians_for_outliers=use_medians_for_outliers,
        outlier_threshold_coeff=outlier_threshold_coeff,
        original_feature_names_mapping=original_feature_names_mapping,
    )

    # Create summary using the utility function
    if outlier_dataframes:
        return get_solution_summary_df(
            data_dict=outlier_dataframes,
            value_column='Feature',
        )
    else:
        return pd.DataFrame()


def show_outlier_summary(
    outlier_summary_df: pd.DataFrame,
    title: str = 'Outlier Summary',
    use_markdown: bool = True,
) -> None:
    """
    Display the summary of outlier features as a DataFrame.

    Parameters:
    -----------
    outlier_summary_df : pd.DataFrame
        The outlier summary DataFrame to display.
    title : str, optional
        Title for the display. Default is "Outlier Summary".
    use_markdown : bool, optional
        Whether to use markdown formatting. Default is True.

    Returns:
    --------
    None
    """
    if outlier_summary_df.empty:
        myprint('No outliers detected in any component.', use_markdown=use_markdown)
        return

    myprint(
        msg=title,
        use_markdown=use_markdown,
        header=2,
    )

    if use_markdown:
        display(outlier_summary_df)
    else:
        print(outlier_summary_df)

    return None


def show_outlier_features_by_component(
    history: dict[str, list],
    use_median: bool | None = False,
    outlier_threshold_coeff: float | None = 3,
    original_feature_names_mapping: dict[str, str] | None = None,
    use_markdown: bool = True,
) -> None:
    """
    Show the outlier features in each component.

    Parameters:
    -----------
    history: Dict[str, List]
        The search history containing mu values for each iteration.
    use_median: bool, optional
        Whether to use median for outlier detection instead of mean. Default is False.
    outlier_threshold_coeff: float, optional
        The multiplier of absolute deviation for outlier detection. Default is 3.
    original_feature_names_mapping: Optional[Dict[str, str]], optional
        A mapping from feature indices to original feature names.
    use_markdown: bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns:
    --------
    None
    """
    # Get the outlier summary data
    outlier_summary_df = get_outlier_summary_from_history(
        history=history,
        use_medians_for_outliers=use_median,
        outlier_threshold_coeff=outlier_threshold_coeff,
        original_feature_names_mapping=original_feature_names_mapping,
    )

    # Display the results
    n_components = len(history['mu'][0])
    title = f'Summary of outlier features across all {n_components} components (threshold = {outlier_threshold_coeff} absolute deviations)'

    show_outlier_summary(
        outlier_summary_df=outlier_summary_df,
        title=title,
        use_markdown=use_markdown,
    )

    return None
