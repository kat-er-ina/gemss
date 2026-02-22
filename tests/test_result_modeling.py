"""Test script for result_modeling module."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from gemss.postprocessing.result_modeling import (
    evaluate_with_nested_cv,
    evaluate_all_solutions,
    _get_available_models,
)


def test_classification_basic():
    """Test basic classification with nested CV."""
    print('\n' + '=' * 60)
    print('TEST 1: Basic Classification (logistic_l2)')
    print('=' * 60)

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

    # Evaluate with nested CV
    result = evaluate_with_nested_cv(
        X=X_df,
        y=y,
        model_name='logistic_l2',
        apply_scaling='standard',
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    print('\nResult keys:', result.keys())
    print('Task:', result['task'])
    print('Model:', result['model'])
    print('CV Type:', result['cv_type'])
    print('\nMetrics:')
    for key, value in result['metrics'].items():
        if key not in ['confusion_matrix', 'class_distribution']:
            print(f'  {key}: {value}')

    assert result['task'] == 'classification'
    assert result['model'] == 'logistic_l2'
    assert 'accuracy' in result['metrics']
    print('\n[PASSED] Test passed!')


def test_regression_basic():
    """Test basic regression with nested CV."""
    print('\n' + '=' * 60)
    print('TEST 2: Basic Regression (linear_l2)')
    print('=' * 60)

    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

    # Evaluate with nested CV
    result = evaluate_with_nested_cv(
        X=X_df,
        y=y,
        model_name='linear_l2',
        apply_scaling='standard',
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
    )

    print('\nResult keys:', result.keys())
    print('Task:', result['task'])
    print('Model:', result['model'])
    print('CV Type:', result['cv_type'])
    print('\nMetrics:')
    for key, value in result['metrics'].items():
        print(f'  {key}: {value}')

    assert result['task'] == 'regression'
    assert result['model'] == 'linear_l2'
    assert 'r2_score' in result['metrics']
    print('\n[PASSED] Test passed!')


def test_leave_one_out():
    """Test leave-one-out cross-validation."""
    print('\n' + '=' * 60)
    print('TEST 3: Leave-One-Out CV')
    print('=' * 60)

    # Generate small dataset suitable for LOO
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

    # Evaluate with LOO
    result = evaluate_with_nested_cv(
        X=X_df,
        y=y,
        model_name='logistic_l2',
        outer_cv_folds='loo',
        random_state=42,
        verbose=True,
    )

    print('\nCV Type:', result['cv_type'])
    assert result['cv_type'] == 'Leave-One-Out'
    print('\n[PASSED] Test passed!')


def test_multiple_solutions():
    """Test evaluating multiple solutions."""
    print('\n' + '=' * 60)
    print('TEST 4: Multiple Solutions')
    print('=' * 60)

    # Generate data with more features
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

    # Create multiple solutions
    solutions = {
        'solution_1': [f'feature_{i}' for i in range(5)],
        'solution_2': [f'feature_{i}' for i in range(5, 10)],
        'solution_3': [f'feature_{i}' for i in range(10, 15)],
    }

    # Evaluate all solutions
    results_df = evaluate_all_solutions(
        solutions=solutions,
        df=df,
        response=y,
        model_name='auto',
        apply_scaling='standard',
        outer_cv_folds=5,
        random_state=42,
        verbose=True,
        use_markdown=False,
    )

    print('\nResults shape:', results_df.shape)
    print('\nSolutions evaluated:', list(results_df.index))
    print('\nMetrics available:', list(results_df.columns))

    assert len(results_df) == 3
    assert 'solution_1' in results_df.index
    assert 'accuracy' in results_df.columns
    print('\n[PASSED] Test passed!')


def test_available_models():
    """Test model registry."""
    print('\n' + '=' * 60)
    print('TEST 5: Available Models')
    print('=' * 60)

    class_models = _get_available_models('classification')
    reg_models = _get_available_models('regression')

    print('\nClassification models:')
    for model in class_models:
        print(f'  - {model}')

    print('\nRegression models:')
    for model in reg_models:
        print(f'  - {model}')

    assert 'logistic_l1' in class_models
    assert 'logistic_l2' in class_models
    assert 'logistic_elasticnet' in class_models
    assert 'random_forest' in class_models
    assert 'linear_l1' in reg_models
    assert 'linear_l2' in reg_models
    assert 'linear_elasticnet' in reg_models
    print('\n[PASSED] Test passed!')


def test_different_models():
    """Test different model types."""
    print('\n' + '=' * 60)
    print('TEST 6: Different Models')
    print('=' * 60)

    # Generate data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

    # Test different classification models
    models_to_test = ['logistic_l1', 'random_forest', 'xgboost', 'knn']

    for model_name in models_to_test:
        print(f'\nTesting model: {model_name}')
        result = evaluate_with_nested_cv(
            X=X_df,
            y=y,
            model_name=model_name,
            apply_scaling='standard',
            outer_cv_folds=3,
            random_state=42,
            verbose=False,
        )
        print(f'  Accuracy: {result["metrics"]["accuracy"]}')
        assert result['model'] == model_name

    print('\n[PASSED] Test passed!')


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('TESTING RESULT_MODELING MODULE')
    print('=' * 60)

    test_classification_basic()
    test_regression_basic()
    test_leave_one_out()
    test_multiple_solutions()
    test_available_models()
    test_different_models()

    print('\n' + '=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)
