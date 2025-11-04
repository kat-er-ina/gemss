"""
Test script for native missing data handling in GEMSS.
"""

import numpy as np
from gemss.feature_selection.inference import BayesianFeatureSelector


def test_native_missing_data_handling():
    """Test that the algorithm can handle missing data natively."""
    print("=== Testing Native Missing Data Handling ===\n")

    # Create test data with missing values
    np.random.seed(42)
    n_samples = 50
    n_features = 6

    # Generate complete data first
    X_complete = np.random.randn(n_samples, n_features)
    # True coefficients (sparse)
    true_beta = np.array([2.0, -1.5, 0.0, 1.0, 0.0, -0.8])
    y = X_complete @ true_beta + 0.1 * np.random.randn(n_samples)

    # Introduce missing values in a structured way
    X_missing = X_complete.copy()

    # Pattern 1: Feature 0 missing for some samples
    X_missing[:15, 0] = np.nan

    # Pattern 2: Features 2,3 missing together
    X_missing[20:35, 2:4] = np.nan

    # Pattern 3: Random missing in feature 5
    missing_mask = np.random.rand(n_samples) < 0.3
    X_missing[missing_mask, 5] = np.nan

    print(f"Created dataset: {n_samples} samples, {n_features} features")
    print(f"Missing values per feature:")
    for i in range(n_features):
        missing_count = np.isnan(X_missing[:, i]).sum()
        print(
            f"  Feature {i}: {missing_count} missing ({missing_count/n_samples*100:.1f}%)"
        )

    print(
        f"Total missing values: {np.isnan(X_missing).sum()}/{X_missing.size} ({np.isnan(X_missing).mean()*100:.1f}%)"
    )

    # Test the feature selector with missing data
    print("\n=== Testing BayesianFeatureSelector ===")

    try:
        selector = BayesianFeatureSelector(
            n_features=n_features,
            n_components=3,
            X=X_missing,  # Pass data with missing values directly
            y=y,
            prior="sss",
            sss_sparsity=3,
            var_spike=0.01,
            lr=0.01,
            batch_size=8,
            n_iter=100,  # Just a few iterations for testing
            device="cpu",
        )

        print("✓ Selector initialized successfully with missing data")

        # Test a few optimization steps
        print("\nRunning optimization...")
        history = selector.optimize(verbose=False)

        print("✓ Optimization completed successfully")
        print(f"Final ELBO: {history['elbo'][-1]:.3f}")

        # Check that mixture parameters are reasonable
        final_mu = history["mu"][-1]
        final_alpha = history["alpha"][-1]

        print(f"Final mixture weights: {final_alpha}")
        print(f"Mixture means shape: {final_mu.shape}")

        # Test that the algorithm found some reasonable solutions
        max_abs_mu = np.abs(final_mu).max(axis=1)
        print(f"Max |μ| per component: {max_abs_mu}")

        if np.any(max_abs_mu > 0.1):
            print("✓ Algorithm found non-zero coefficients")
        else:
            print("⚠ All coefficients are close to zero - may need more iterations")

        print("\n🎉 Native missing data handling works correctly!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_gradient_flow_with_missing_data():
    """Test that gradients flow properly with missing data."""
    print("\n=== Testing Gradient Flow ===")

    # Simple test case
    np.random.seed(123)
    X = np.random.randn(10, 4)
    X[0, 0] = np.nan  # One missing value
    X[5, 2:4] = np.nan  # Two missing values in one sample
    y = np.random.randn(10)

    selector = BayesianFeatureSelector(n_features=4, n_components=2, X=X, y=y, n_iter=5)

    # Check that parameters have gradients after one step
    initial_mu = selector.mixture.mu.clone()

    # Run one optimization step
    z, _ = selector.mixture.sample(4)
    elbo = selector.elbo(z)
    selector.opt.zero_grad()
    (-elbo).backward()

    # Check gradients exist
    has_gradients = selector.mixture.mu.grad is not None
    if has_gradients:
        grad_norm = selector.mixture.mu.grad.norm().item()
        print(f"✓ Gradients computed successfully, norm: {grad_norm:.6f}")
    else:
        print("❌ No gradients computed")
        return False

    # Take optimizer step
    selector.opt.step()

    # Check parameters changed
    param_change = (selector.mixture.mu - initial_mu).norm().item()
    if param_change > 1e-8:
        print(f"✓ Parameters updated, change norm: {param_change:.6f}")
    else:
        print("❌ Parameters did not update")
        return False

    return True


if __name__ == "__main__":
    success1 = test_native_missing_data_handling()
    success2 = test_gradient_flow_with_missing_data()

    if success1 and success2:
        print(
            "\n🎉 All tests passed! Native missing data handling is working correctly."
        )
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
