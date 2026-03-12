import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdu.plotly.styling import apply_default_styles
from mdu.plotly.resampler_compat import FigureResampler, HAS_RESAMPLER


class DataShapeError(Exception):
    """Exception raised when data has incompatible shape for plotting."""

    pass


def plot_ts(
    data: np.ndarray,
    x: np.ndarray | None = None,
    names: list[str] | None = None,
    show: bool = False,
    use_resampler: bool = True,
) -> go.Figure:
    """Plot one or multiple time series with optional resampling.

    Creates a multi-row subplot figure with one trace per row. Optionally uses
    plotly-resampler for efficient rendering of large time series data.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_samples,) for single time series or
        (n_samples, n_features) for multiple time series.
    x : np.ndarray or None, default=None
        Array of x-axis values with shape (n_samples,). If None, uses
        np.arange(data.shape[0]).
    names : list of str or None, default=None
        Labels for each time series trace. If None, generates default
        names like 'y0', 'y1', etc.
    show : bool, default=False
        If True, display the figure immediately.
    use_resampler : bool, default=True
        If True and plotly-resampler is installed, use FigureResampler
        for efficient large dataset rendering. If False or resampler not
        available, uses standard plotly Figure.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure object (FigureResampler if available and use_resampler=True,
        otherwise standard Figure).

    Raises
    ------
    DataShapeError
        If data has more than 2 dimensions.

    Notes
    -----
    If use_resampler=True but plotly-resampler is not installed, a UserWarning
    will be issued and standard plotly Figure will be used instead.

    Examples
    --------
    >>> import numpy as np
    >>> # Single time series
    >>> data = np.random.randn(1000)
    >>> fig = plot_ts(data)

    >>> # Multiple time series with custom x-axis
    >>> data = np.random.randn(1000, 3)
    >>> x = np.linspace(0, 10, 1000)
    >>> fig = plot_ts(data, x=x, names=['Signal A', 'Signal B', 'Signal C'])

    >>> # Large dataset with resampling
    >>> large_data = np.random.randn(1_000_000, 2)
    >>> fig = plot_ts(large_data, use_resampler=True)

    Notes
    -----
    When plotly-resampler is installed and use_resampler=True, the function
    creates a FigureResampler which dynamically downsamples data during
    pan/zoom operations for improved performance with large datasets.

    Without plotly-resampler, all data points are rendered which may cause
    performance issues with datasets larger than ~100k points.
    """
    if len(data.shape) > 2:
        raise DataShapeError(f"{data.shape=}, but should be at most 2D")

    # Ensure 2D shape for consistent processing
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x = np.arange(data.shape[0]) if x is None else x

    # Defaults
    nrows = data.shape[1]
    names = [f"y{i}" for i in range(nrows)] if names is None else names

    # Create figure with or without resampler
    if use_resampler and HAS_RESAMPLER:
        # Create base figure structure
        base_fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True)
        fig = FigureResampler(base_fig)

        # Add traces with high-frequency data
        # Ensure arrays are contiguous for plotly-resampler
        x_contiguous = np.ascontiguousarray(x)
        for iy in range(nrows):
            y_contiguous = np.ascontiguousarray(data[:, iy])
            fig.add_trace(
                go.Scatter(name=names[iy], showlegend=True),
                hf_x=x_contiguous,
                hf_y=y_contiguous,
                row=iy + 1,
                col=1,
            )
    else:
        # Standard plotly without resampling
        fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True)
        for iy in range(nrows):
            fig.add_scatter(x=x, y=data[:, iy], name=names[iy], row=iy + 1, col=1)

    fig = apply_default_styles(fig)

    if show:
        fig.show()

    return fig
