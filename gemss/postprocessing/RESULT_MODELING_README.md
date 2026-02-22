# Result Modeling with Nested Cross-Validation

This module provides robust evaluation of feature selection solutions using nested cross-validation with scikit-learn models.

## Overview

Unlike `simple_regressions` which evaluates on training data only, `result_modeling` provides proper generalization performance estimates through nested cross-validation.

### Key Features

- **Nested Cross-Validation**: Proper estimation of generalization performance
- **Multiple Models**: Wide range of scikit-learn classifiers and regressors
- **Leave-One-Out Support**: For small datasets
- **Flexible Scaling**: StandardScaler, MinMaxScaler, or no scaling
- **Auto Model Selection**: Automatically chooses appropriate model for task type
- **Comprehensive Metrics**: Same metrics as `tabpfn_evaluation` and `simple_regressions`

## Main Functions

### `evaluate_with_nested_cv`

Evaluate a single solution (feature set) using nested cross-validation.

```python
from gemss.postprocessing.result_modeling import evaluate_with_nested_cv

result = evaluate_with_nested_cv(
    X=X_selected,                    # Features (DataFrame or ndarray)
    y=y,                             # Target values
    model_name='logistic_l2',        # Model to use
    apply_scaling='standard',        # 'standard', 'minmax', or None
    outer_cv_folds=5,                # Number of CV folds or 'loo'
    inner_cv_folds=5,                # Inner CV folds (default)
    random_state=42,                 # For reproducibility
    verbose=True,                    # Print progress
    stratify=None                    # Optional custom stratification vector
)
```

**Returns**: Dictionary with task, model, cv_type, metrics, and n_features

### `evaluate_all_solutions`

Evaluate multiple solutions (analogous to `solve_any_regression`).

```python
from gemss.postprocessing.result_modeling import evaluate_all_solutions

# Define solutions (e.g., from GEMSS recovery)
solutions = {
    'component_1': ['feature_0', 'feature_1', 'feature_2'],
    'component_2': ['feature_3', 'feature_4', 'feature_5'],
}

# Evaluate all solutions
results_df = evaluate_all_solutions(
    solutions=solutions,
    df=df,                           # Full dataframe
    response=y,                      # Target values
    model_name='auto',               # Auto-select based on task
    apply_scaling='standard',
    outer_cv_folds=5,
    random_state=42,
    verbose=True,
    stratify=None                    # Optional custom stratification vector
)
```

**Returns**: DataFrame with solutions as rows and metrics as columns

## Available Models

### Classification Models

- `logistic_l1` - Logistic regression with L1 penalty
- `logistic_l2` - Logistic regression with L2 penalty
- `logistic_elasticnet` - Logistic regression with ElasticNet
- `random_forest` - Random forest classifier
- `xgboost` - XGBoost classifier
- `svm` - Support vector machine
- `knn` - K-nearest neighbors
- `decision_tree` - Decision tree
- `naive_bayes` - Gaussian Naive Bayes
- `lda` - Linear discriminant analysis
- `qda` - Quadratic discriminant analysis

### Regression Models

- `linear_l1` - Lasso regression (L1)
- `linear_l2` - Ridge regression (L2)
- `linear_elasticnet` - ElasticNet regression
- `random_forest` - Random forest regressor
- `xgboost` - XGBoost regressor
- `svm` - Support vector regression
- `knn` - K-nearest neighbors regressor
- `decision_tree` - Decision tree regressor

## Usage Examples

### Example 1: Single Solution Evaluation

```python
from gemss.postprocessing.result_modeling import evaluate_with_nested_cv

# Your data
X_selected = df[['feature_1', 'feature_2', 'feature_5']]
y = df['target']

# Evaluate
result = evaluate_with_nested_cv(
    X=X_selected,
    y=y,
    model_name='random_forest',
    apply_scaling='standard',
    outer_cv_folds=10,
    random_state=42,
    verbose=True
)

print(result['metrics'])
```

### Example 2: Multiple Solutions

```python
from gemss.postprocessing.result_modeling import evaluate_all_solutions
from gemss.postprocessing.result_postprocessing import recover_solutions

# Recover solutions from GEMSS
solutions = recover_solutions(history, min_mu=0.3)

# Evaluate all
results = evaluate_all_solutions(
    solutions=solutions,
    df=df,
    response=y,
    model_name='auto',  # Uses logistic_l2 or linear_l2
    apply_scaling='standard',
    outer_cv_folds=5,
    verbose=True
)

# View results
print(results)
```

### Example 3: Leave-One-Out for Small Datasets

```python
# For datasets with < 50 samples
results = evaluate_all_solutions(
    solutions=solutions,
    df=df,
    response=y,
    model_name='logistic_l2',
    outer_cv_folds='loo',  # Leave-One-Out
    verbose=True
)
```

### Example 4: Compare Different Models

```python
models = ['logistic_l2', 'random_forest', 'gradient_boosting', 'hist_gradient_boosting', 'xgboost']

comparison = []
for model in models:
    result = evaluate_with_nested_cv(
        X=X_selected,
        y=y,
        model_name=model,
        apply_scaling='standard',
        outer_cv_folds=5,
        random_state=42
    )
    comparison.append({
        'model': model,
        'accuracy': result['metrics']['accuracy'],
        'f1_score': result['metrics']['f1_score']
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df)
```

## Metrics

### Classification Metrics

- `n_samples` - Number of samples
- `n_features` - Number of features
- `accuracy` - Classification accuracy
- `balanced_accuracy` - Balanced accuracy (handles imbalanced classes)
- `roc_auc` - ROC-AUC score (binary classification only)
- `f1_score` - Weighted F1 score
- `precision_class_X` - Precision for each class
- `recall_class_X` - Recall for each class
- `class_distribution` - Proportion of each class
- `confusion_matrix` - Confusion matrix

### Regression Metrics

- `n_samples` - Number of samples
- `n_features` - Number of features
- `r2_score` - R² score
- `adjusted_r2` - Adjusted R² score
- `MSE` - Mean squared error
- `RMSE` - Root mean squared error
- `MAE` - Mean absolute error
- `MAPE` - Mean absolute percentage error

## Stratification in Cross-Validation

Both `evaluate_with_nested_cv` and `evaluate_all_solutions` support custom stratification vectors:

**Default behavior:**
- Classification: Stratified by target values (preserves class distribution)
- Regression: No stratification (random splits)

**Custom stratification:**
Use the `stratify` parameter to provide a custom vector for stratification:
- Useful when samples have inherent grouping (experimental batches, time periods, patient cohorts)
- Must have the same length as the target vector
- Values should be categorical or discrete

```python
# Example: stratify by experimental batch
batch_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, ...])

results = evaluate_all_solutions(
    solutions=solutions,
    df=df,
    response=y,
    model_name='logistic_l2',
    outer_cv_folds=5,
    stratify=batch_ids,  # Ensure each fold has samples from all batches
    random_state=42
)
```

## Comparison with Other Modules

| Module | Evaluation Method | Use Case |
|--------|------------------|----------|
| `simple_regressions` | Training data only | Quick exploratory analysis |
| `tabpfn_evaluation` | Outer CV with TabPFN | When TabPFN is suitable |
| `result_modeling` | Nested CV with sklearn | Robust generalization estimates |

## Integration with GEMSS Workflow

```python
# 1. Run GEMSS feature selection
selector = BayesianFeatureSelector(...)
selector.fit(X, y)

# 2. Recover solutions
from gemss.postprocessing.result_postprocessing import recover_solutions
solutions = recover_solutions(selector.history, min_mu=0.3)

# 3. Quick check with simple regressions (training data)
from gemss.postprocessing.simple_regressions import solve_any_regression
quick_results = solve_any_regression(solutions, df, y, verbose=False)

# 4. Proper evaluation with nested CV
from gemss.postprocessing.result_modeling import evaluate_all_solutions
cv_results = evaluate_all_solutions(
    solutions=solutions,
    df=df,
    response=y,
    model_name='auto',
    outer_cv_folds=5,
    apply_scaling='standard',
    random_state=42,
    verbose=True
)

# 5. Compare results
print("Training performance (optimistic):")
print(quick_results[['accuracy', 'f1_score']])
print("\nNested CV performance (realistic):")
print(cv_results[['accuracy', 'f1_score']])
```

## Advanced Usage

### Custom Model Selection

```python
# Get available models
from gemss.postprocessing.result_modeling import _get_available_models

classification_models = _get_available_models('classification')
regression_models = _get_available_models('regression')

print("Available classification models:", classification_models)
print("Available regression models:", regression_models)
```

### Handling Missing Data

The module automatically drops rows with missing values. Solutions with insufficient samples after dropping NaNs (< 15 by default) are skipped.

```python
# Minimum samples required
MIN_ALLOWED_SAMPLES = 15

# Solutions with < MIN_ALLOWED_SAMPLES after dropna() are skipped
```

### Leave-one-out cross-validation

This is suitable only for the smallest datasets (sample size n < 50). No stratification is applied.
