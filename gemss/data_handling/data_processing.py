"""

Data preprocessing utilities for user-provided datasets.

"""

from typing import Literal

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gemss.config.constants import DATA_DIR
from gemss.utils.utils import myprint


def load_data(
    csv_dataset_name: str,
    index_column_name: str,
    label_column_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from a CSV file.

    Parameters:
    -----------
    csv_dataset_name: str
        Name of the CSV file containing the dataset.
    index_column_name: str
        Name of the column to be used as the index.
    label_column_name: str
        Name of the column containing the target response values.

    Returns:
    --------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the response Series.
    """
    df = pd.read_csv(DATA_DIR / csv_dataset_name, index_col=index_column_name)
    response = df.pop(label_column_name)
    return df, response


def get_feature_name_mapping(df: pd.DataFrame) -> dict:
    """
    Generate a mapping from generic feature names (used inside the model) to actual column names.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the features.

    Returns:
    --------
    dict
        A dictionary mapping generic feature names (e.g., 'feature_0') to actual column names.
    """
    feature_to_name = {f'feature_{i}': col_name for i, col_name in enumerate(df.columns)}
    return feature_to_name


def preprocess_non_numeric_features(
    df: pd.DataFrame,
    how: Literal['drop', 'onehot'] = 'drop',
    verbose: bool | None = True,
) -> pd.DataFrame:
    """
    Process non-numeric features in the DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the features.
    how: Literal["drop", "onehot"], optional, default="drop"
        How to handle non-numeric features. Options are:
        - "drop": drop non-numeric features.
        - "onehot": apply one-hot encoding to non-numeric features.
    verbose: bool, optional, default=True
        Whether to display informative messages during processing.

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame with non-numeric features handled according to the specified method.
    """
    df_copy = df.copy()

    if how == 'drop':
        non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()
        df_copy = df_copy.drop(columns=non_numeric_cols)
        if verbose:
            if len(non_numeric_cols) > 0:
                display(
                    Markdown(
                        f'Dropped {len(non_numeric_cols)} non-numeric features: {non_numeric_cols}.'
                    )
                )
            else:
                display(Markdown('No non-numeric features found. No columns were dropped.'))

    elif how == 'onehot':
        df_copy = pd.get_dummies(df_copy, drop_first=True)
        if verbose:
            display(Markdown('Applied one-hot encoding to non-numeric features.'))

    return df_copy


def preprocess_features(
    df: pd.DataFrame,
    response: pd.Series,
    dropna: Literal['response', 'all', 'none'] = 'response',
    allowed_missing_percentage: float | None = None,
    drop_non_numeric_features: bool = True,
    apply_scaling: Literal['standard', 'minmax', None] = None,
    verbose: bool | None = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Preprocess the features by cleaning na values and applying scaling, if specified.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the features.
    response: pd.Series
        The Series containing the target response values.
    dropna: Literal["response", "all", "none"], optional, default="response"
        Whether to drop rows with NA values. Options are:
        - "response": drop rows with NA in the response column only.
        - "all": drop rows with NA in any column.
        - "none": do not drop any rows.
    allowed_missing_percentage: float, optional, default=90
        The maximum allowed percentage of missing values in the dataset.
        Features with a higher percentage of missing values will be dropped.
    drop_non_numeric_features: bool, optional, default=True
        Whether to drop non-numeric features from the DataFrame.
        Default is True because the feature selector works only with numerical values.
    apply_scaling: Literal["standard", "minmax", None] = None,
        Whether to apply scaling to the features. Options are:
        - "standard": apply standard scaling.
        - "minmax": apply Min-Max scaling.
        - None: do not apply any scaling.
    verbose: bool, optional, default=True
        Whether to display informative messages during preprocessing.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Dict[str, str]]
        A tuple containing:
        - the preprocessed features as a NumPy array
        - the response values as a NumPy array
        - a mapping from feature names to their original column names
    """
    df_copy = df.copy()

    if drop_non_numeric_features:
        df_copy = preprocess_non_numeric_features(df_copy, how='drop', verbose=verbose)

    # handle NA values
    if dropna != 'none':
        initial_shape = df_copy.shape
        df_copy[response.name] = response
        if dropna == 'response':
            cols_to_check = [response.name]
        elif dropna == 'all':
            cols_to_check = df_copy.columns.tolist()
        else:
            raise ValueError("Invalid value for dropna. Choose from 'response', 'all', or 'none'.")
        df_copy = df_copy.dropna(subset=cols_to_check)
        response = df_copy.pop(response.name)
        if verbose:
            if df_copy.shape[0] == initial_shape[0]:
                myprint(
                    'No NA values found. No rows were dropped.',
                    use_markdown=True,
                )
            else:
                myprint(
                    f'Dropped rows with NA values. Shape changed from {initial_shape} to {df_copy.shape}.',
                    use_markdown=True,
                )

    # drop features with too many missing values
    if allowed_missing_percentage is not None:
        if allowed_missing_percentage < 0 or allowed_missing_percentage > 100:
            raise ValueError('allowed_missing_percentage must be either None or between 0 and 100.')
        # Treat values between 0 and 1 as fractions
        if allowed_missing_percentage <= 1:
            allowed_missing_percentage *= 100

        initial_shape = df_copy.shape
        missing_pct_per_feature = df_copy.isna().mean() * 100
        features_to_drop = missing_pct_per_feature[
            missing_pct_per_feature > allowed_missing_percentage
        ].index.tolist()
        df_copy = df_copy.drop(columns=features_to_drop)
        if verbose:
            if len(features_to_drop) > 0:
                myprint(
                    f'Dropped {len(features_to_drop)} features with more than {allowed_missing_percentage}% missing values. Shape changed from {initial_shape} to {df_copy.shape}.',
                    use_markdown=True,
                )
            else:
                myprint(
                    f'No features exceeded the allowed missing value percentage {allowed_missing_percentage}%. No columns were dropped.',
                    use_markdown=True,
                )

    # scaling
    if apply_scaling == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(df_copy.values)
        if verbose:
            myprint(
                'Features have been standardized using StandardScaler.',
                use_markdown=True,
            )
    elif apply_scaling == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(df_copy.values)
        if verbose:
            myprint(
                'Features have been scaled using MinMaxScaler.',
                use_markdown=True,
            )
    else:
        X = df_copy.values
        if verbose:
            myprint(
                'No scaling applied to features.',
                use_markdown=True,
            )

    # report % of missing values
    n_missing = np.sum(np.isnan(X).astype(int))
    n_total = X.size
    missing_pct = (n_missing / n_total) * 100
    myprint(
        f'Dataset contains {n_missing}/{n_total} ({missing_pct:.1f}%) missing feature values.',
        use_markdown=True,
    )

    return X, response.values, get_feature_name_mapping(df_copy)


def get_df_from_X(
    X: np.ndarray,
    feature_to_name: dict,
) -> pd.DataFrame:
    """
    Convert a NumPy array of features back to a DataFrame using the feature name mapping.

    Parameters:
    -----------
    X: np.ndarray
        The NumPy array of features.
    feature_to_name: dict
        A dictionary mapping generic feature names to actual column names.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the original feature names as columns.
    """
    df = pd.DataFrame(X, columns=[feature_to_name[f'feature_{i}'] for i in range(X.shape[1])])
    return df.astype(float)
