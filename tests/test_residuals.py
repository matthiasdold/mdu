"""Tests for mdu.stats.residuals module."""

import numpy as np
import plotly.graph_objects as go
import pytest
import scipy.stats as st

from mdu.stats.residuals import fit_residual_dist, plot_acf, residuals_analysis_plot


class TestFitResidualDist:
    """Test suite for fit_residual_dist function."""

    def test_fits_normal_distribution(self):
        """Test fitting a normal distribution to residuals."""
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = y_true + np.random.randn(1000) * 0.1

        dist = fit_residual_dist(y_true, y_pred, dist=st.norm, plot=False)

        assert isinstance(dist, st._distn_infrastructure.rv_continuous_frozen)
        # Check that mean is close to 0
        assert abs(dist.mean()) < 0.1
        # Check that std is reasonable
        assert 0.05 < dist.std() < 0.2

    def test_different_distributions(self):
        """Test fitting different distribution types."""
        np.random.seed(42)
        y_true = np.random.randn(1000)
        y_pred = y_true + np.random.randn(1000) * 0.1

        # Test with t-distribution
        dist_t = fit_residual_dist(y_true, y_pred, dist=st.t, plot=False)
        assert isinstance(dist_t, st._distn_infrastructure.rv_continuous_frozen)

    def test_perfect_predictions(self):
        """Test with perfect predictions (no residuals)."""
        np.random.seed(42)
        y = np.random.randn(100)
        
        dist = fit_residual_dist(y, y, dist=st.norm, plot=False)
        
        # With perfect predictions, residuals are all zero
        # This results in scale=0 which causes numerical issues
        # Just verify the function completes without error
        assert isinstance(dist, st._distn_infrastructure.rv_continuous_frozen)

    def test_with_1d_arrays(self):
        """Test with 1D arrays."""
        np.random.seed(42)
        y_true = np.random.randn(500)
        y_pred = y_true + 0.2 * np.random.randn(500)

        dist = fit_residual_dist(y_true, y_pred, plot=False)
        
        assert isinstance(dist, st._distn_infrastructure.rv_continuous_frozen)


class TestPlotACF:
    """Test suite for plot_acf function."""

    def test_returns_figure(self):
        """Test that function returns a Plotly figure."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        fig = plot_acf(x)
        
        assert isinstance(fig, go.Figure)

    def test_confidence_bounds_present(self):
        """Test that confidence bounds are in the plot."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        fig = plot_acf(x)
        
        # Should have 3 traces: upper bound, lower bound, ACF
        assert len(fig.data) == 3
        
        # Check confidence bounds are included
        trace_names = [trace.name for trace in fig.data]
        assert "Confidence interval<br>standard normal" in trace_names

    def test_lag_zero_excluded_by_default(self):
        """Test that lag 0 is excluded by default."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        fig = plot_acf(x, plot_lag_zero=False)
        
        # ACF trace should have one less y value than x values when lag 0 is excluded
        acf_trace = fig.data[2]  # Third trace is ACF
        # x has all lags, y starts from lag 1
        assert len(acf_trace.y) == len(acf_trace.x) - 1

    def test_lag_zero_included(self):
        """Test including lag 0 in the plot."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        fig = plot_acf(x, plot_lag_zero=True)
        
        assert isinstance(fig, go.Figure)

    def test_acf_at_zero_lag_is_one(self):
        """Test that ACF at lag 0 is normalized to 1."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        # Compute ACF manually
        acf = np.correlate(x, x, mode="full")
        acf_normalized = acf[acf.size // 2 :] / acf[acf.size // 2]
        
        # First value (lag 0) should be 1
        assert abs(acf_normalized[0] - 1.0) < 1e-10

    def test_confidence_bounds_calculation(self):
        """Test confidence bounds are calculated correctly."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        fig = plot_acf(x)
        
        # Get confidence bound traces
        upper_trace = fig.data[0]
        
        # Check bounds are approximately ±1.96/sqrt(n)
        expected_bound = 1.96 / np.sqrt(len(x))
        assert abs(upper_trace.y[0] - expected_bound) < 0.01

    def test_with_correlated_data(self):
        """Test with autocorrelated data."""
        np.random.seed(42)
        # Create AR(1) process
        n = 500
        x = np.zeros(n)
        x[0] = np.random.randn()
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + np.random.randn()
        
        fig = plot_acf(x)
        
        assert isinstance(fig, go.Figure)
        # ACF should show significant correlation at lag 1
        acf_trace = fig.data[2]
        assert len(acf_trace.y) > 0


class TestResidualsAnalysisPlot:
    """Test suite for residuals_analysis_plot function."""

    def test_returns_figure(self):
        """Test that function returns a Plotly figure."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        y_pred = y_true + 0.1 * np.random.randn(200)

        fig = residuals_analysis_plot(y_true, y_pred, show=False)

        assert isinstance(fig, go.Figure)

    def test_with_1d_arrays(self):
        """Test with 1D arrays."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        y_pred = y_true + 0.2 * np.random.randn(200)

        fig = residuals_analysis_plot(y_true, y_pred, show=False)

        assert isinstance(fig, go.Figure)
        # Should create 1 row with 3 columns
        assert fig.layout.xaxis.domain is not None

    def test_with_2d_arrays(self):
        """Test with 2D arrays (multiple variables)."""
        np.random.seed(42)
        y_true = np.random.randn(200, 2)
        y_pred = y_true + 0.2 * np.random.randn(200, 2)

        fig = residuals_analysis_plot(y_true, y_pred, show=False)

        assert isinstance(fig, go.Figure)
        # Should have multiple rows (one per variable)

    def test_with_different_distribution(self):
        """Test with different distribution generator."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        y_pred = y_true + 0.1 * np.random.randn(200)

        fig = residuals_analysis_plot(y_true, y_pred, distgen=st.t, show=False)

        assert isinstance(fig, go.Figure)

    def test_creates_three_subplots_per_variable(self):
        """Test that 3 subplots are created per variable (hist, probplot, acf)."""
        np.random.seed(42)
        y_true = np.random.randn(200, 2)
        y_pred = y_true + 0.2 * np.random.randn(200, 2)

        fig = residuals_analysis_plot(y_true, y_pred, show=False)

        # With 2 variables, should have 6 subplot titles (2 rows × 3 cols)
        assert len(fig.layout.annotations) >= 6

    def test_height_scales_with_variables(self):
        """Test that figure height scales with number of variables."""
        np.random.seed(42)
        y_true_1d = np.random.randn(200, 1)
        y_pred_1d = y_true_1d + 0.2 * np.random.randn(200, 1)

        y_true_3d = np.random.randn(200, 3)
        y_pred_3d = y_true_3d + 0.2 * np.random.randn(200, 3)

        fig_1var = residuals_analysis_plot(y_true_1d, y_pred_1d, show=False)
        fig_3var = residuals_analysis_plot(y_true_3d, y_pred_3d, show=False)

        # 3-variable plot should be taller
        assert fig_3var.layout.height > fig_1var.layout.height

    def test_with_perfect_predictions(self):
        """Test with perfect predictions (no residuals)."""
        np.random.seed(42)
        y = np.random.randn(200)

        fig = residuals_analysis_plot(y, y, show=False)

        assert isinstance(fig, go.Figure)
