"""Tests for mdu.plotly.resampler_compat module."""

import pytest
import plotly.graph_objects as go
import numpy as np
import warnings

from mdu.plotly.resampler_compat import (
    FigureResampler,
    HAS_RESAMPLER,
    get_figure_resampler,
    warn_if_no_resampler,
)


class TestResamplerCompat:
    """Test suite for resampler_compat module."""

    def test_has_resampler_is_bool(self):
        """Test that HAS_RESAMPLER is a boolean."""
        assert isinstance(HAS_RESAMPLER, bool)

    def test_figure_resampler_is_importable(self):
        """Test that FigureResampler can be imported."""
        assert FigureResampler is not None
        assert callable(FigureResampler)

    def test_get_figure_resampler_returns_figure(self):
        """Test that get_figure_resampler returns a figure-like object."""
        warnings.simplefilter("ignore")
        fig = get_figure_resampler()
        # Should be either real FigureResampler or fallback
        assert hasattr(fig, 'add_trace')
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    def test_get_figure_resampler_basic(self):
        """Test creating basic figure."""
        warnings.simplefilter("ignore")
        fig = get_figure_resampler()
        
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        assert len(fig.data) >= 1

    @pytest.mark.skipif(not HAS_RESAMPLER, reason="Requires plotly-resampler")
    def test_real_resampler_is_used(self):
        """Test that real FigureResampler is used when available."""
        from plotly_resampler import FigureResampler as RealResampler
        
        fig = get_figure_resampler()
        assert isinstance(fig, RealResampler)

    @pytest.mark.skipif(not HAS_RESAMPLER, reason="Requires plotly-resampler")
    def test_real_resampler_add_trace_with_hf(self):
        """Test real resampler with high-frequency data."""
        fig = get_figure_resampler()
        
        hf_x = np.arange(10000)
        hf_y = np.random.randn(10000)
        
        fig.add_trace(
            go.Scatter(name="test"),
            hf_x=hf_x,
            hf_y=hf_y
        )
        
        assert len(fig.data) == 1

    def test_warn_if_no_resampler_function(self):
        """Test warn_if_no_resampler function."""
        if HAS_RESAMPLER:
            # Should not warn
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_if_no_resampler()
                # Filter out other warnings
                resampler_warnings = [x for x in w if "plotly-resampler" in str(x.message)]
                assert len(resampler_warnings) == 0
        else:
            # Should warn
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_if_no_resampler()
                assert len(w) > 0
                assert "plotly-resampler is not installed" in str(w[0].message)

    def test_get_figure_resampler_with_layout(self):
        """Test creating figure and updating layout."""
        warnings.simplefilter("ignore")
        fig = get_figure_resampler()
        fig.update_layout(title="Test Figure")
        
        assert fig.layout.title.text == "Test Figure"

    def test_multiple_traces(self):
        """Test adding multiple traces."""
        warnings.simplefilter("ignore")
        fig = get_figure_resampler()
        
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="trace1"))
        fig.add_trace(go.Scatter(x=[5, 6], y=[7, 8], name="trace2"))
        
        assert len(fig.data) == 2
        assert fig.data[0].name == "trace1"
        assert fig.data[1].name == "trace2"
