import numpy as np
import pytest
import mne

from mdu.plotly.mne_plotting_utils.shared import bootstrap, combine_epochs


class TestCombineEpochs:
    """Test suite for combine_epochs function."""

    @pytest.fixture
    def sample_epochs(self):
        """Create sample MNE epochs for testing."""
        # Create info with 3 channels
        ch_names = ["Ch1", "Ch2", "Ch3"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")

        # Create data: 5 epochs, 3 channels, 50 timepoints
        n_epochs = 5
        n_channels = 3
        n_times = 50
        data = np.random.RandomState(42).randn(n_epochs, n_channels, n_times)

        # Create epochs
        epochs = mne.EpochsArray(data, info, verbose=False)
        return epochs, data

    def test_combine_epochs_mean(self, sample_epochs):
        """Test combining epochs using mean method."""
        epochs, data = sample_epochs
        result = combine_epochs(epochs, "mean")

        # Check shape
        assert result.shape == (data.shape[0], data.shape[2])

        # Check values - should be mean across channel dimension
        expected = data.mean(axis=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_combine_epochs_gfp(self, sample_epochs):
        """Test combining epochs using global field power (GFP)."""
        epochs, data = sample_epochs
        result = combine_epochs(epochs, "gfp")

        # Check shape
        assert result.shape == (data.shape[0], data.shape[2])

        # Check values - should be RMS across channel dimension
        expected = np.sqrt((data**2).mean(axis=1))
        np.testing.assert_array_almost_equal(result, expected)

    def test_combine_epochs_gfp_always_positive(self, sample_epochs):
        """Test that GFP values are always non-negative."""
        epochs, _ = sample_epochs
        result = combine_epochs(epochs, "gfp")

        # GFP should always be >= 0
        assert np.all(result >= 0)

    def test_combine_epochs_invalid_method(self, sample_epochs):
        """Test that invalid combine method raises KeyError."""
        epochs, _ = sample_epochs

        with pytest.raises(KeyError):
            combine_epochs(epochs, "invalid_method")

    def test_combine_epochs_single_channel(self):
        """Test combining epochs with a single channel."""
        # Create single channel epochs
        info = mne.create_info(["Ch1"], sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(3, 1, 40)
        epochs = mne.EpochsArray(data, info, verbose=False)

        # For single channel, mean should equal the channel data
        result_mean = combine_epochs(epochs, "mean")
        np.testing.assert_array_almost_equal(result_mean, data[:, 0, :])

        # GFP should be absolute value for single channel
        result_gfp = combine_epochs(epochs, "gfp")
        expected_gfp = np.sqrt(data[:, 0, :] ** 2)
        np.testing.assert_array_almost_equal(result_gfp, expected_gfp)


class TestBootstrap:
    """Test suite for bootstrap function."""

    def test_bootstrap_basic_functionality(self):
        """Test basic bootstrap functionality with default parameters."""
        data = np.random.RandomState(42).randn(100, 5)
        ci_bounds, boot_dist = bootstrap(data, seed=42)

        # Check shapes
        assert ci_bounds.shape == (2, 5)
        assert boot_dist.shape == (2000, 5)

        # Lower bound should be less than upper bound
        assert np.all(ci_bounds[0] < ci_bounds[1])

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with same seed."""
        data = np.random.RandomState(42).randn(50, 3)

        ci1, bd1 = bootstrap(data, seed=123, nboot=500)
        ci2, bd2 = bootstrap(data, seed=123, nboot=500)

        np.testing.assert_array_equal(ci1, ci2)
        np.testing.assert_array_equal(bd1, bd2)

    def test_bootstrap_different_seeds(self):
        """Test that different seeds produce different results."""
        data = np.random.RandomState(42).randn(50, 3)

        ci1, _ = bootstrap(data, seed=123, nboot=500)
        ci2, _ = bootstrap(data, seed=456, nboot=500)

        # Results should be different with different seeds
        assert not np.allclose(ci1, ci2)

    def test_bootstrap_custom_rng(self):
        """Test bootstrap with custom random number generator."""
        data = np.random.RandomState(42).randn(50, 3)
        rng = np.random.Generator(np.random.PCG64(999))

        ci_bounds, boot_dist = bootstrap(data, rng=rng, nboot=500)

        assert ci_bounds.shape == (2, 3)
        assert boot_dist.shape == (500, 3)

    def test_bootstrap_min_max(self):
        """Test bootstrap with min_max option."""
        data = np.random.RandomState(42).randn(100, 4)
        ci_bounds, boot_dist = bootstrap(data, min_max=True, nboot=1000, seed=42)

        # Check that bounds are min and max of bootstrap distribution
        assert ci_bounds.shape == (2, 4)
        np.testing.assert_array_almost_equal(ci_bounds[0], boot_dist.min(axis=0))
        np.testing.assert_array_almost_equal(ci_bounds[1], boot_dist.max(axis=0))

    def test_bootstrap_custom_ci_levels(self):
        """Test bootstrap with custom confidence interval levels."""
        data = np.random.RandomState(42).randn(100, 3)
        ci_bounds, _ = bootstrap(data, ci=[0.05, 0.95], nboot=1000, seed=42)

        # Should still have shape (2, 3)
        assert ci_bounds.shape == (2, 3)
        # Lower bound should be less than upper bound
        assert np.all(ci_bounds[0] < ci_bounds[1])

    def test_bootstrap_nboot_parameter(self):
        """Test that nboot parameter controls number of bootstrap samples."""
        data = np.random.RandomState(42).randn(50, 2)

        for nboot in [100, 500, 1500]:
            _, boot_dist = bootstrap(data, nboot=nboot, seed=42)
            assert boot_dist.shape[0] == nboot

    def test_bootstrap_1d_array(self):
        """Test bootstrap with 1D array."""
        data = np.random.RandomState(42).randn(100)
        ci_bounds, boot_dist = bootstrap(data, nboot=500, seed=42)

        # For 1D input, output should be (2,) for CI and (500,) for distribution
        assert ci_bounds.shape == (2,)
        assert boot_dist.shape == (500,)

    def test_bootstrap_multidimensional_array(self):
        """Test bootstrap with multidimensional array."""
        # 3D array: 50 samples, 4x3 feature matrix
        data = np.random.RandomState(42).randn(50, 4, 3)
        ci_bounds, boot_dist = bootstrap(data, nboot=500, seed=42)

        # Shape should preserve all dimensions except sample dimension
        assert ci_bounds.shape == (2, 4, 3)
        assert boot_dist.shape == (500, 4, 3)

    def test_bootstrap_bounds_contain_mean(self):
        """Test that confidence intervals typically contain the true mean."""
        # Create data with known mean
        true_mean = 5.0
        data = np.random.RandomState(42).randn(100, 1) + true_mean

        ci_bounds, _ = bootstrap(data, ci=[0.025, 0.975], nboot=2000, seed=42)

        # The sample mean should typically be within the 95% CI
        sample_mean = data.mean(axis=0)
        assert np.all(ci_bounds[0] <= sample_mean)
        assert np.all(sample_mean <= ci_bounds[1])

    def test_bootstrap_small_sample(self):
        """Test bootstrap with small sample size."""
        data = np.random.RandomState(42).randn(10, 2)
        ci_bounds, boot_dist = bootstrap(data, nboot=100, seed=42)

        assert ci_bounds.shape == (2, 2)
        assert boot_dist.shape == (100, 2)

    def test_bootstrap_percentile_calculation(self):
        """Test that percentile-based CI is calculated correctly."""
        data = np.random.RandomState(42).randn(100, 1)
        ci_bounds, boot_dist = bootstrap(data, ci=[0.1, 0.9], nboot=1000, seed=42)

        # Manually compute percentiles
        expected_lower = np.percentile(boot_dist, 10, axis=0)
        expected_upper = np.percentile(boot_dist, 90, axis=0)

        np.testing.assert_array_almost_equal(ci_bounds[0], expected_lower)
        np.testing.assert_array_almost_equal(ci_bounds[1], expected_upper)

    def test_bootstrap_with_zero_data(self):
        """Test bootstrap with data that is all zeros."""
        data = np.zeros((50, 3))
        ci_bounds, boot_dist = bootstrap(data, nboot=100, seed=42)

        # All values should be zero
        np.testing.assert_array_almost_equal(ci_bounds, 0.0)
        np.testing.assert_array_almost_equal(boot_dist, 0.0)

    def test_bootstrap_variance_reduction(self):
        """Test that bootstrap distribution has lower variance than original data."""
        data = np.random.RandomState(42).randn(100, 1)
        _, boot_dist = bootstrap(data, nboot=1000, seed=42)

        # Bootstrap distribution of means should have lower variance
        # than original data (due to averaging)
        original_var = data.var()
        bootstrap_var = boot_dist.var()

        # Bootstrap variance should be smaller
        assert bootstrap_var < original_var
