"""
Utility functions for feature selection project.
Includes:
- Sampling
- Metrics
- Data handling
"""

from IPython.display import display, Markdown
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from plotly import graph_objects as go
import torch
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def kldiv(p, q, eps=1e-10):
    """
    Compute KL divergence between two discrete probability distributions.

    Parameters
    ----------
    p : np.ndarray
        Array of probabilities (ground truth).
    q : np.ndarray
        Array of probabilities (approximation).
    eps : float, optional
        Small value to avoid log(0), default is 1e-10.

    Returns
    -------
    float
        KL divergence value.
    """
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * (np.log(p) - np.log(q)))


def batch_data(X, y, batch_size):
    """
    Randomly sample a batch of data.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    batch_size : int
        Number of samples to select.

    Returns
    -------
    X_batch : np.ndarray
        Batch of data features.
    y_batch : np.ndarray
        Batch of data targets.
    """
    n = X.shape[0]
    idx = np.random.choice(n, batch_size, replace=False)
    return X[idx], y[idx]


def save_history(history, fname):
    """
    Save optimization history to a file using pickle.

    Parameters
    ----------
    history : dict
        Dictionary containing optimization history.
    fname : str
        Output file path.

    Returns
    -------
    None
    """
    import pickle

    with open(fname, "wb") as f:
        pickle.dump(history, f)


def solve_with_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
):
    """
    Solve a logistic regression problem with the specified penalty.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Binary response vector.
    penalty : str
        Type of regularization to use ("l1", "l2", or "elasticnet").
        Default is "l2".

    Returns
    -------
    None
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty=penalty,
            solver="saga",
            scoring="roc_auc",
            max_iter=2000,
            random_state=42,
            refit=True,
            class_weight="balanced",
        ),
    )

    # Solve the full problem
    # Ignore warnings about convergence for this demo
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X, y)

    # Predictions and evaluation
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)[:, 1]

    # Print results
    display(
        Markdown(
            f"### Logistic Regression with {penalty.upper()} Penalty - Performance on Training Set"
        )
    )
    display(Markdown(f"**Accuracy:** {accuracy_score(y, y_pred)}"))
    display(Markdown(f"**ROC-AUC:** {roc_auc_score(y, y_pred_prob)}"))
    display(Markdown("Coefficients of the logistic regression model:"))
    coefficients = pd.Series(
        clf.named_steps["logisticregressioncv"].coef_[0], index=X.columns
    )
    coefficients = coefficients[coefficients != 0].sort_values(ascending=False)
    display(coefficients)

    # Show confusion matrix using Plotly
    cm = confusion_matrix(y, y_pred)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix", width=350, height=350, showlegend=False)
    fig.show(config={"displayModeBar": False})
    return
