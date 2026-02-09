"""
Simple regression utilities for feature selection project.
Includes:
- Logistic regression with various penalties
- Linear regression with various penalties
- Function to show the results
"""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gemss.utils.utils import myprint
from gemss.utils.visualizations import (
    show_confusion_matrix,
    show_predicted_vs_actual_response,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

MAX_ALLOWED_NAN_RATIO = 0.9  # maximum proportion of missing values to run regression
MIN_ALLOWED_SAMPLES = 15  # minimum number of samples to run regression


def detect_task(y: pd.Series | np.ndarray, n_class_threshold: int = 10) -> str:
    """
    Detect if the task should be treated as classification or regression.
    Rules:
    - If the target has 2 or fewer unique values, it's classification.
    - If target is integer/boolean and has fewer unique values than threshold,
      it's classification.
    - Otherwise, it's regression.

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Target vector.
    n_class_threshold : int, optional
        Maximum number of unique integer values to consider as classification. 10 by default.

    Returns
    -------
    str
        "classification" or "regression"
    """
    y = np.asarray(y)
    unique = np.unique(y)
    if len(unique) <= 2:
        return 'classification'
    elif (pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y)) and len(
        unique
    ) <= n_class_threshold:
        return 'classification'
    return 'regression'


def print_verbose_logistic_regression_results(
    stats: dict[str, Any],
    penalty: str,
    confusion_matrix: np.ndarray,
    show_cm_figure: bool = True,
) -> None:
    """
    Print detailed logistic regression results in a formatted display.

    Parameters
    ----------
    stats : dict
        Dictionary containing regression statistics with keys:
        - 'n_samples': Number of samples
        - 'class_distribution': Proportion of each class
        - 'accuracy': Classification accuracy
        - 'balanced_accuracy': Balanced classification accuracy
        - 'roc_auc': ROC-AUC score
        - 'f1_score': F1 score
        - 'precision_class_0': Precision for class 0
        - 'precision_class_1': Precision for class 1
        - 'recall_class_0': Recall for class 0
        - 'recall_class_1': Recall for class 1
        - 'nonzero_coefficients': Series of non-zero model coefficients
    penalty : str
        Type of regularization penalty used ('l1', 'l2', or 'elasticnet')
    confusion_matrix : np.ndarray
        2x2 confusion matrix from sklearn.metrics.confusion_matrix
    show_cm_figure : bool, optional
        Whether to display confusion matrix as interactive figure using Plotly.
        If False, displays as plain text. Default is True.

    Returns
    -------
    None
    """

    display(
        Markdown(
            f'### Logistic regression with {penalty.upper()} penalty - performance on training set'
        )
    )
    display(Markdown(f'**Number of samples:** {stats["n_samples"]}'))
    display(
        Markdown(
            f'**Class distribution (0/1):** '
            f'{stats["class_distribution"]["class_0"]:.1%} / '
            f'{stats["class_distribution"]["class_1"]:.1%}'
        )
    )
    display(Markdown(f'**Accuracy:** {stats["accuracy"]}'))
    display(Markdown(f'**Balanced Accuracy:** {stats["balanced_accuracy"]}'))
    display(Markdown(f'**ROC-AUC:** {stats["roc_auc"]}'))
    display(Markdown(f'**Balanced F1 Score:** {stats["f1_score"]}'))
    prec0, prec1 = stats['precision_class_0'], stats['precision_class_1']
    display(Markdown(f'**Precision (class 0/1):** {prec0} / {prec1}'))
    display(
        Markdown(f'**Recall (class 0/1):** {stats["recall_class_0"]} / {stats["recall_class_1"]}')
    )

    n_nz = stats['n_nonzero_coefficients']
    display(Markdown(f'**{n_nz} non-zero features with coefficients:**'))
    display(
        [
            f'{stats["nonzero_feature_names"][i]}: {stats["nonzero_coefficients"][i]}'
            for i in range(len(stats['nonzero_coefficients']))
        ]
    )

    if show_cm_figure:
        # Show confusion matrix using Plotly
        show_confusion_matrix(confusion_matrix=confusion_matrix)
    else:
        display(Markdown('**Confusion Matrix:**'))
        display(confusion_matrix)
    return


def solve_with_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    apply_scaling: Literal['standard', 'minmax', None] = None,
    penalty: Literal['l1', 'l2', 'elasticnet'] = 'l2',
    verbose: bool | None = False,
    show_cm_figure: bool | None = True,
) -> dict[str, Any]:
    """
    Solve a logistic regression problem with the specified penalty.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Binary response vector.
    apply_scaling: Literal["standard", "minmax", None] = None,
        Whether to apply scaling inside the regression pipeline. Options are:
        - "standard": apply standard scaling.
        - "minmax": apply Min-Max scaling.
        - None: do not apply any scaling.
    penalty : str
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l2".
    verbose : bool, optional
        Whether to print detailed metrics, coefficients and confusion matrix.
        Default is False.
    show_cm_figure : bool, optional
        Whether to display the confusion matrix figure using Plotly
        (otherwise it is just printed). Default is True.
        Only applicable if verbose is True.

    Returns
    -------
    dict[str, Any]
        Dictionary containing performance metrics and model coefficients.
    """
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty=penalty,
        solver='saga',
        scoring='roc_auc',
        max_iter=2000,
        random_state=42,
        refit=True,
        class_weight='balanced',
    )

    if apply_scaling == 'standard':
        clf = make_pipeline(StandardScaler(), model)
    elif apply_scaling == 'minmax':
        clf = make_pipeline(MinMaxScaler(), model)
    else:
        clf = make_pipeline(model)

    # Solve the full problem
    # Ignore warnings about convergence for this demo
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        clf.fit(X, y)

    # Predictions and evaluation
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)

    # identify nonzero coefficients
    coefficients = (
        pd.Series(clf.named_steps['logisticregressioncv'].coef_[0], index=X.columns)
        .rename('Coefficient')
        .sort_values(ascending=False)
    )
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)

    stats = {
        'n_samples': len(y),
        'class_distribution': {
            'class_0': np.round(np.sum(y == 0) / len(y), 3),
            'class_1': np.round(np.sum(y == 1) / len(y), 3),
        },
        'accuracy': np.round(accuracy_score(y, y_pred), 3),
        'balanced_accuracy': np.round(balanced_accuracy_score(y, y_pred), 3),
        'roc_auc': np.round(roc_auc_score(y, y_pred_prob), 3),
        'f1_score': np.round(f1_score(y, y_pred, average='weighted'), 3),
        'precision_class_0': np.round(precision_score(y, y_pred, pos_label=0, zero_division=0), 3),
        'precision_class_1': np.round(precision_score(y, y_pred, pos_label=1, zero_division=0), 3),
        'recall_class_0': np.round(np.sum((y_pred == 0) & (y == 0)) / np.sum(y == 0), 3),
        'recall_class_1': np.round(np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1), 3),
        'confusion_matrix [TN, FP, FN, TP]': cm.ravel(),
        'n_nonzero_coefficients': len(coefficients),
        'nonzero_coefficients': coefficients,
        'nonzero_feature_names': coefficients.index.tolist(),
    }

    # Print results
    if verbose:
        print_verbose_logistic_regression_results(
            stats=stats,
            penalty=penalty,
            confusion_matrix=cm,
            show_cm_figure=show_cm_figure,
        )

    return stats


def print_verbose_linear_regression_results(
    stats: dict[str, Any],
    model_name: str,
    illustrate_predicted_vs_actual: bool = False,
    y_actual: pd.Series | np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> None:
    """
    Print detailed linear regression results in a formatted display.

    Parameters
    ----------
    stats : dict
        Dictionary containing regression statistics with keys:
        - 'n_samples': Number of samples
        - 'r2_score': R² coefficient of determination
        - 'adjusted_r2': Adjusted R² score
        - 'MSE': Mean squared error
        - 'RMSE': Root mean squared error
        - 'MAE': Mean absolute error
        - 'MAPE': Mean absolute percentage error
        - 'n_nonzero_coefficients': Number of non-zero coefficients
        - 'nonzero_coefficients': Series of non-zero model coefficients
    model_name : str
        Name of the regression model (e.g., "Ridge (L2)", "Lasso (L1)", "ElasticNet")
    illustrate_predicted_vs_actual : bool, optional
        Whether to show predicted vs actual response plot using Plotly. Default is False.
    y_actual : pd.Series or np.ndarray, optional
        Actual response values. Required if illustrate_predicted_vs_actual is True.
    y_pred : np.ndarray, optional
        Predicted response values. Required if illustrate_predicted_vs_actual is True.

    Returns
    -------
    None
    """
    display(
        Markdown(f'### Linear regression with {model_name} penalty - performance on training set')
    )
    display(Markdown(f'**Number of samples:** {stats["n_samples"]}'))
    display(Markdown(f'**R² Score:** {stats["r2_score"]}'))
    display(Markdown(f'**Adjusted R² Score:** {stats["adjusted_r2"]}'))
    display(Markdown(f'**MSE:** {stats["MSE"]}'))
    display(Markdown(f'**RMSE:** {stats["RMSE"]}'))
    display(Markdown(f'**MAE:** {stats["MAE"]}'))
    if not np.isnan(stats['MAPE']):
        display(Markdown(f'**MAPE:** {stats["MAPE"]}%'))
    n_nz = stats['n_nonzero_coefficients']
    display(Markdown(f'**{n_nz} non-zero features with coefficients:**'))
    display(
        [
            f'{stats["nonzero_feature_names"][i]}: {stats["nonzero_coefficients"][i]}'
            for i in range(len(stats['nonzero_coefficients']))
        ]
    )

    if illustrate_predicted_vs_actual:
        if y_actual is not None and y_pred is not None:
            show_predicted_vs_actual_response(y_actual, y_pred)
        else:
            display(Markdown('*Cannot display predicted vs actual plot: missing data*'))
    return


def solve_with_linear_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    apply_scaling: Literal['standard', 'minmax', None] = None,
    penalty: Literal['l1', 'l2', 'elasticnet'] = 'l2',
    verbose: bool | None = True,
    illustrate_predicted_vs_actual: bool | None = False,
) -> dict[str, Any]:
    """
    Solve a linear regression problem with the specified penalty.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Continuous response vector.
    apply_scaling: Literal["standard", "minmax", None] = None,
        Whether to apply scaling inside the regression pipeline. Options are:
        - "standard": apply standard scaling.
        - "minmax": apply Min-Max scaling.
        - None: do not apply any scaling.
    penalty : str
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l2".
    verbose : bool, optional
        Whether to print detailed metrics and coefficients. Default is True.
    illustrate_predicted_vs_actual : bool, optional
        Whether to illustrate predicted vs actual response using Plotly. Default is False.
        Only applicable if verbose is True.

    Returns
    -------
    dict[str, Any]
        Dictionary containing performance metrics and model coefficients.
    """
    # Choose model based on penalty
    if penalty == 'l1':
        model = LassoCV(cv=5, random_state=42, max_iter=2000)
        model_name = 'Lasso (L1)'
    elif penalty == 'l2':
        model = RidgeCV(cv=5)
        model_name = 'Ridge (L2)'
    elif penalty == 'elasticnet':
        model = ElasticNetCV(cv=5, random_state=42, max_iter=2000)
        model_name = 'ElasticNet'
    else:
        raise ValueError("Penalty must be 'l1', 'l2', or 'elasticnet'.")

    if apply_scaling == 'standard':
        pipeline = make_pipeline(StandardScaler(), model)
    elif apply_scaling == 'minmax':
        pipeline = make_pipeline(MinMaxScaler(), model)
    else:
        pipeline = make_pipeline(model)

    # Fit the model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        pipeline.fit(X, y)

    # Predictions and evaluation
    y_pred = pipeline.predict(X)

    # Get coefficients, handle difference for RidgeCV (coef_ shape)
    if penalty == 'l2':
        coefs = pipeline.named_steps['ridgecv'].coef_
    elif penalty == 'l1':
        coefs = pipeline.named_steps['lassocv'].coef_
    elif penalty == 'elasticnet':
        coefs = pipeline.named_steps['elasticnetcv'].coef_

    coefficients = pd.Series(coefs, index=X.columns)
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)

    # Calculate residuals for additional statistics
    residuals = y - y_pred

    # Calculate additional regression statistics
    stats = {
        'n_samples': len(y),
        'r2_score': np.round(r2_score(y, y_pred), 3),
        'adjusted_r2': np.round(
            1 - (1 - r2_score(y, y_pred)) * (len(y) - 1) / (len(y) - len(coefficients) - 1),
            3,
        ),
        'MSE': np.round(mean_squared_error(y, y_pred), 3),
        'RMSE': np.round(np.sqrt(mean_squared_error(y, y_pred)), 3),
        'MAE': np.round(np.mean(np.abs(residuals)), 3),
        'MAPE': (
            np.round(np.mean(np.abs(residuals / np.where(y != 0, y, 1e-8))) * 100, 3)
            if not np.any(y == 0)
            else np.nan
        ),
        'n_nonzero_coefficients': len(coefficients),
        'nonzero_coefficients': coefficients,
        'nonzero_feature_names': coefficients.index.tolist(),
    }

    # Print results
    if verbose:
        print_verbose_linear_regression_results(
            stats=stats,
            model_name=model_name,
            illustrate_predicted_vs_actual=illustrate_predicted_vs_actual,
            y_actual=y,
            y_pred=y_pred,
        )
    return stats


def solve_any_regression(
    solutions: dict[str, list[str]],
    df: pd.DataFrame,
    response: pd.Series | np.ndarray,
    apply_scaling: Literal['standard', 'minmax', None] = None,
    penalty: Literal['l1', 'l2', 'elasticnet'] = 'l1',
    verbose: bool | None = False,
    use_markdown: bool | None = True,
) -> pd.DataFrame:
    """
    Compute regression/classification for each candidate solution using the identified features.
    Based on the type of response vector y, it automatically selects logistic regression
    or linear regression.

    Parameters
    ----------
    solutions : dict[str, list[str]]
        Dictionary mapping each component (solution) to its identified features.
        Feature names should correspond to column names in df.
    df : pd.DataFrame
        Feature matrix with features as columns.
    response : pd.Series | np.ndarray
        Response vector. Binary values {0, 1} for classification, continuous values for regression.
    apply_scaling : Literal["standard", "minmax", None], optional
        Type of feature scaling to apply before regression. Options are:
        - "standard": apply standard scaling.
        - "minmax": apply Min-Max scaling.
        - None: do not apply any scaling.
    penalty : Literal["l1", "l2", "elasticnet"], optional
        Type of regularization to use. Default is "l1".
    verbose : bool, optional
        Whether to print detailed regression metrics and coefficients
        for each component and the resulting overview of metrics. Default is False.
    use_markdown : bool, optional
        Whether to format the output using Markdown. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing regression metrics for all candidate solutions.
    """
    # if response is a numpy array, convert to pd.Series
    if isinstance(response, np.ndarray):
        response = pd.Series(response)

    if set(np.unique(response.dropna())).__len__() == 2:
        is_binary = True
    else:
        is_binary = False

    task = 'Classification' if is_binary else 'Regression'
    stats = {}
    # for each candidate solution
    for component, features in solutions.items():
        if verbose:
            myprint(
                msg=f'Features of **{component}**',
                use_markdown=use_markdown,
                header=2,
            )
            myprint(
                f'- {len(features)} features: {features}',
                use_markdown=use_markdown,
            )

        df_filtered = df[features].copy()
        df_filtered['response'] = response
        df_filtered = df_filtered.dropna()
        y_filtered = df_filtered.pop('response')

        # check missing value ratio and sample size
        n_vals = len(df[features]) * len(features)
        nan_ratio = df[features].isna().sum().sum() / n_vals  # noqa: F841

        # FIXME: nan_ratio is in range 0 to 100 for some components
        # if nan_ratio > MAX_ALLOWED_NAN_RATIO:
        #     myprint(
        #         msg=f"**Cannot run classical regression for {component}.** NAN_RATIO is {nan_ratio}, which is greater than the allowed ratio {MAX_ALLOWED_NAN_RATIO}.", # noqa: E501
        #         use_markdown=use_markdown,
        #     )
        #     continue

        if df_filtered.shape[0] < MIN_ALLOWED_SAMPLES:
            n_left = df_filtered.shape[0]
            myprint(
                msg=(
                    f'**Cannot run regression for {component}.** '
                    f'After dropping NaNs: {n_left} samples '
                    f'(need ≥ {MIN_ALLOWED_SAMPLES}).'
                ),
                use_markdown=use_markdown,
            )
            continue

        if is_binary:
            stats[component] = solve_with_logistic_regression(
                X=df_filtered[features],
                y=y_filtered,
                apply_scaling=apply_scaling,
                penalty=penalty,
                verbose=verbose,
            )
        else:
            stats[component] = solve_with_linear_regression(
                X=df_filtered[features],
                y=y_filtered,
                apply_scaling=apply_scaling,
                penalty=penalty,
                verbose=verbose,
            )
        if verbose:
            myprint(msg='------------------', use_markdown=use_markdown)

    # get the stats as data frame
    # each entry is a column in the data frame
    metrics_df = pd.DataFrame.from_dict(stats, orient='index')

    if verbose:
        show_regression_metrics(
            metrics_df=metrics_df,
            title=f'{task} results on training data ({penalty} penalty)',
            use_markdown=use_markdown,
        )

    return metrics_df


def show_regression_metrics(
    metrics_df: pd.DataFrame,
    title: str = 'Results on training data',
    use_markdown: bool | None = True,
) -> None:
    """
    Show regression metrics from a DataFrame.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing regression metrics for each solution.
    title : str
        Optional custom title to display above the metrics.

    Returns
    -------
    None
    """
    if metrics_df.empty:
        myprint(
            msg='No regression results to display.',
            use_markdown=use_markdown,
        )
    else:
        myprint(
            msg=title,
            use_markdown=use_markdown,
            header=2,
        )

    if use_markdown:
        display(metrics_df)
    else:
        print(metrics_df)
    return
