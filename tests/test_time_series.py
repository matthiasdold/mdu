import numpy as np
import pytest
import plotly.graph_objects as go

from mdu.plotly.time_series import plot_ts, DataShapeError
from mdu.plotly.resampler_compat import HAS_RESAMPLER


class TestPlotTs:
    """Test suite for plot_ts function."""

    def test_plot_ts_1d_array(self):
        """Test plotting a 1D time series."""
        data = np.random.randn(100)
        fig = plot_ts(data, use_resampler=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].name == "y0"

    def test_plot_ts_2d_array(self):
        """Test plotting multiple time series from 2D array."""
        data = np.random.randn(100, 3)
        fig = plot_ts(data, use_resampler=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        assert fig.data[0].name == "y0"
        assert fig.data[1].name == "y1"
        assert fig.data[2].name == "y2"

    def test_plot_ts_custom_x_axis(self):
        """Test plotting with custom x-axis values."""
        data = np.random.randn(100, 2)
        x = np.linspace(0, 10, 100)
        fig = plot_ts(data, x=x, use_resampler=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        # Check that custom x values are used
        np.testing.assert_array_equal(fig.data[0].x, x)
        np.testing.assert_array_equal(fig.data[1].x, x)

    def test_plot_ts_custom_names(self):
        """Test plotting with custom trace names."""
        data = np.random.randn(50, 2)
        names = ["Signal A", "Signal B"]
        fig = plot_ts(data, names=names, use_resampler=False)

        assert fig.data[0].name == "Signal A"
        assert fig.data[1].name == "Signal B"

    def test_plot_ts_3d_array_raises_error(self):
        """Test that 3D arrays raise DataShapeError."""
        data_3d = np.random.randn(10, 10, 10)

        with pytest.raises(DataShapeError, match="should be at most 2D"):
            plot_ts(data_3d)

    def test_plot_ts_default_x_values(self):
        """Test that default x values are np.arange."""
        data = np.random.randn(50, 2)
        fig = plot_ts(data, use_resampler=False)

        expected_x = np.arange(50)
        np.testing.assert_array_equal(fig.data[0].x, expected_x)

    def test_plot_ts_show_parameter(self):
        """Test that show parameter doesn't raise errors."""
        data = np.random.randn(10)
        # Just verify it doesn't raise - actual display is not testable
        fig = plot_ts(data, show=False)
        assert isinstance(fig, go.Figure)

    def test_plot_ts_empty_array(self):
        """Test behavior with empty array."""
        data = np.array([]).reshape(0, 2)
        fig = plot_ts(data, use_resampler=False)

        assert len(fig.data) == 2
        assert len(fig.data[0].x) == 0

    @pytest.mark.skipif(not HAS_RESAMPLER, reason="plotly-resampler not installed")
    def test_plot_ts_with_resampler(self):
        """Test that resampler is used when available and requested."""
        from plotly_resampler import FigureResampler as RealResampler

        data = np.random.randn(1000, 2)
        fig = plot_ts(data, use_resampler=True)

        assert isinstance(fig, RealResampler)
        assert len(fig.data) == 2

    @pytest.mark.skipif(not HAS_RESAMPLER, reason="plotly-resampler not installed")
    def test_plot_ts_without_resampler_when_available(self):
        """Test that resampler can be disabled even when available."""
        from plotly_resampler import FigureResampler as RealResampler

        data = np.random.randn(1000, 2)
        fig = plot_ts(data, use_resampler=False)

        assert not isinstance(fig, RealResampler)
        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not HAS_RESAMPLER, reason="plotly-resampler not installed")
    def test_plot_ts_large_dataset_with_resampler(self):
        """Test that large datasets work with resampler."""
        large_data = np.random.randn(100_000, 1)
        fig = plot_ts(large_data, use_resampler=True)

        # Should not raise and should use resampler
        from plotly_resampler import FigureResampler as RealResampler

        assert isinstance(fig, RealResampler)

    def test_plot_ts_contiguous_arrays(self):
        """Test that non-contiguous arrays work."""
        # Create non-contiguous array via slicing
        data_full = np.random.randn(200, 4)
        data = data_full[::2, ::2]  # Non-contiguous slice

        # Should not raise even with non-contiguous input
        fig = plot_ts(data, use_resampler=False)
        assert len(fig.data) == 2


class TestDataShapeError:
    """Test suite for DataShapeError exception."""

    def test_data_shape_error_is_exception(self):
        """Test that DataShapeError is an Exception."""
        assert issubclass(DataShapeError, Exception)

    def test_data_shape_error_message(self):
        """Test that DataShapeError can be raised with a message."""
        msg = "Invalid shape"
        with pytest.raises(DataShapeError, match=msg):
            raise DataShapeError(msg)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point(self):
        """Test plotting a single data point."""
        data = np.array([[1.0]])
        fig = plot_ts(data, use_resampler=False)

        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 1

    def test_very_short_series(self):
        """Test plotting very short time series."""
        data = np.array([[1.0], [2.0], [3.0]])
        fig = plot_ts(data, use_resampler=False)

        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 3

    def test_many_traces(self):
        """Test plotting many traces."""
        data = np.random.randn(100, 10)
        fig = plot_ts(data, use_resampler=False)

        assert len(fig.data) == 10

    def test_x_and_data_mismatch_length(self):
        """Test that mismatched x and data lengths work (plotly handles it)."""
        data = np.random.randn(100, 1)
        x = np.arange(50)  # Different length

        # Plotly will handle this - just ensure no crash
        fig = plot_ts(data, x=x, use_resampler=False)
        assert isinstance(fig, go.Figure)
