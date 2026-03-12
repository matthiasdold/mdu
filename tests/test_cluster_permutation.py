import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from mdu.plotly.stats import add_cluster_permut_sig_to_plotly


@pytest.fixture
def sample_curves():
    """Create sample time series data for testing."""
    np.random.seed(42)
    n_trials, n_time = 20, 100
    time = np.linspace(0, 1, n_time)
    
    # Group A: baseline
    curves_a = np.random.randn(n_trials, n_time)
    
    # Group B: with signal in middle
    curves_b = np.random.randn(n_trials, n_time)
    curves_b[:, 40:60] += 1.5  # Add signal
    
    return curves_a, curves_b, time


def test_sparkline_mode_basic(sample_curves):
    """Test sparkline mode creates secondary y-axis."""
    curves_a, curves_b, time = sample_curves
    
    # Create basic figure
    fig = go.Figure()
    fig.add_scatter(x=time, y=curves_a.mean(axis=0), name='Group A')
    fig.add_scatter(x=time, y=curves_b.mean(axis=0), name='Group B')
    
    # Add sparkline mode
    fig = add_cluster_permut_sig_to_plotly(
        curves_a, curves_b, fig,
        xaxes_vals=time,
        pval=0.05,
        nperm=100,
        mode='spark'
    )
    
    # Check that secondary y-axis was created
    assert hasattr(fig.layout, 'yaxis2'), "Should have secondary y-axis (yaxis2)"
    assert fig.layout.yaxis2 is not None, "yaxis2 should be configured"
    
    # Check yaxis2 configuration
    yaxis2 = fig.layout.yaxis2
    assert yaxis2.overlaying == 'y', "yaxis2 should overlay primary yaxis"
    assert yaxis2.side == 'right', "yaxis2 should be on the right side"
    assert yaxis2.showgrid is False, "yaxis2 should not show grid"
    
    # Check that F-value traces were added
    f_traces = [t for t in fig.data if t.name and 'F-val' in t.name]
    assert len(f_traces) == 2, "Should have 2 F-value traces (F-values and threshold)"
    
    # Check that F-value traces use secondary y-axis
    for trace in f_traces:
        assert trace.yaxis == 'y2', f"F-value trace '{trace.name}' should use y2"


def test_sparkline_mode_with_subplots(sample_curves):
    """Test sparkline mode with subplots creates separate secondary axes."""
    curves_a, curves_b, time = sample_curves
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=2)
    fig.add_scatter(x=time, y=curves_a.mean(axis=0), row=1, col=1)
    fig.add_scatter(x=time, y=curves_b.mean(axis=0), row=1, col=1)
    fig.add_scatter(x=time, y=curves_a.mean(axis=0), row=1, col=2, showlegend=False)
    fig.add_scatter(x=time, y=curves_b.mean(axis=0), row=1, col=2, showlegend=False)
    
    # Add sparkline to first subplot
    fig = add_cluster_permut_sig_to_plotly(
        curves_a, curves_b, fig,
        xaxes_vals=time, row=1, col=1,
        pval=0.05, nperm=100, mode='spark'
    )
    
    # Add sparkline to second subplot
    fig = add_cluster_permut_sig_to_plotly(
        curves_a, curves_b, fig,
        xaxes_vals=time, row=1, col=2,
        pval=0.05, nperm=100, mode='spark'
    )
    
    # Check that both secondary y-axes were created
    assert hasattr(fig.layout, 'yaxis2'), "Should have yaxis2 for first subplot"
    assert hasattr(fig.layout, 'yaxis3'), "Should have yaxis3 for second subplot"
    
    # Check yaxis2 configuration (subplot 1, col 1)
    yaxis2 = fig.layout.yaxis2
    assert yaxis2.overlaying == 'y', "yaxis2 should overlay y (subplot 1)"
    assert yaxis2.anchor == 'x', "yaxis2 should anchor to x"
    
    # Check yaxis3 configuration (subplot 1, col 2)
    yaxis3 = fig.layout.yaxis3
    assert yaxis3.overlaying == 'y2', "yaxis3 should overlay y2 (subplot 2)"
    assert yaxis3.anchor == 'x2', "yaxis3 should anchor to x2"
    
    # Check trace assignments
    f_traces_col1 = [t for t in fig.data if t.name and 'F-val' in t.name and t.yaxis == 'y2']
    f_traces_col2 = [t for t in fig.data if t.name and 'F-val' in t.name and t.yaxis == 'y3']
    
    assert len(f_traces_col1) == 2, "Should have 2 F-value traces on y2 (col 1)"
    assert len(f_traces_col2) == 2, "Should have 2 F-value traces on y3 (col 2)"


def test_other_modes_no_secondary_axis(sample_curves):
    """Test that other modes don't create secondary y-axes."""
    curves_a, curves_b, time = sample_curves
    
    for mode in ['line', 'p_bg', 'p_colorbar']:
        fig = go.Figure()
        fig.add_scatter(x=time, y=curves_a.mean(axis=0), name='Group A')
        fig.add_scatter(x=time, y=curves_b.mean(axis=0), name='Group B')
        
        fig = add_cluster_permut_sig_to_plotly(
            curves_a, curves_b, fig,
            xaxes_vals=time,
            pval=0.05,
            nperm=100,
            mode=mode
        )
        
        # Check that no secondary y-axis was created
        has_yaxis2 = hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2 is not None
        assert not has_yaxis2, f"Mode '{mode}' should not create secondary y-axis"


def test_sparkline_trace_styling(sample_curves):
    """Test that sparkline traces have appropriate styling."""
    curves_a, curves_b, time = sample_curves
    
    fig = go.Figure()
    fig.add_scatter(x=time, y=curves_a.mean(axis=0), name='Group A')
    
    fig = add_cluster_permut_sig_to_plotly(
        curves_a, curves_b, fig,
        xaxes_vals=time,
        pval=0.05,
        nperm=100,
        mode='spark'
    )
    
    # Find F-value traces
    f_value_trace = [t for t in fig.data if t.name == 'F-values'][0]
    f_thresh_trace = [t for t in fig.data if t.name == 'F-val thresh'][0]
    
    # Check styling
    assert f_value_trace.opacity == 0.6, "F-values trace should be semi-transparent"
    assert f_thresh_trace.opacity == 0.6, "F-threshold trace should be semi-transparent"
    assert f_thresh_trace.line.dash == 'dash', "F-threshold should be dashed line"
    assert f_value_trace.mode == 'lines', "F-values should be line plot"
