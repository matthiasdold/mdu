# Plotting methods to work on MNE objects

import mne
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from tqdm import tqdm

from mdu.plotly.mne_plotting_utils.epoch_image import plot_epo_image
from mdu.plotly.mne_plotting_utils.topoplot import create_plotly_topoplot
from mdu.plotly.time_series import plot_ts
from mdu.plotly.template import set_template
from mdu.mne.mne2dataframe import mne_epochs_to_polars
from mdu.plotly.multiline import multiline_plot
from mdu.plotly.shared import rgb_to_hex

set_template()


def plot_topo(
    data: np.ndarray,
    inst: mne.io.Raw | mne.Epochs | mne.Evoked,
    contour_kwargs: dict = {"colorscale": "viridis"},
    show: bool = False,
    scale_range: float = 1.2,
    blank_scaling: float = 0.2,
) -> go.Figure:
    """
    Plot a topoplot from data and an mne instance for meta data information.
    This is a wrapper for ./tne_plotting_utils/topoplot.py::create_plotly_topoplot


    Parameters
    ----------
    data : np.ndarray
        the data for the topoplot, one value for each channel in inst.ch_names

    inst : mne.io.Raw | mne.Epochs | mne.Evoked
        the mne instance to get the channel meta information from

    contour_kwargs : dict, optional
        kwargs for the contour plot, by default {"colorscale": "viridis"}

    Returns
    -------
    go.FigureWidget
        topo plot figure in plotly

    """
    fig = create_plotly_topoplot(
        data,
        inst,
        contour_kwargs=contour_kwargs,
        show=show,
        scale_range=scale_range,
        blank_scaling=blank_scaling,
    )

    return fig


def plot_variances(
    epo: mne.BaseEpochs,
    df: pd.DataFrame,
    color_by: str = "",
    row: Optional[list[int]] = None,
    col: Optional[list[int]] = None,
    show: bool = False,
) -> go.Figure:
    """Plot the variance distribution as scatter along time and
    cumulative densities

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    color_by: str
        name of column in df to group the epochs into colors. This will create
        a column left of the y-axis which shows the color
    fig : go.FigureWidget
        figure to add the subplots to
    row : list(int, int)
        row parameter used in fig.add_traces to add the plot, for the scatter
        and the hist plot
    col : list(int, int)
        col parameter used in fig.add_traces to add the plot for the scatter
        and the hist plot
    show : bool, optional
        if True, fig.show() is called

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
        epo.plot_image(evoked=False)

    """
    row = row if row is not None else [1, 1]
    col = col if col is not None else [1, 2]

    data = epo.get_data()

    if data.shape[1] > 1:
        print("Combining the variance of more than one channel!")

    var_data = np.var(data.mean(axis=1), axis=1)

    grps = list(df.groupby(color_by))
    colormap = dict(zip([g[0] for g in grps], px.colors.qualitative.Plotly))

    dw = pd.DataFrame(var_data, columns=["y"])  # type: ignore
    dw[color_by] = df[color_by].to_numpy()
    dw["color"] = df[color_by].map(colormap).to_numpy()  # type: ignore
    dw = dw.reset_index().rename(columns={"index": "x"})

    fig = None
    fig = (
        make_subplots(1, 2, column_widths=(0.8, 0.2), horizontal_spacing=0)
        if fig is None
        else fig
    )
    # Scatter

    # Bars
    for i, (ck, color) in enumerate(colormap.items()):
        # scatter
        fig.add_trace(
            go.Scatter(
                x=dw.loc[dw[color_by] == ck, "x"],
                y=dw.loc[dw[color_by] == ck, "y"],
                marker=dict(color=color, size=12),
                name=ck,
                mode="markers",
                opacity=0.8,
            ),
            row=row[0],
            col=col[0],
        )

        # histogram
        fig.add_trace(
            go.Histogram(
                y=dw.loc[dw[color_by] == ck, "y"],
                histnorm="probability",
                marker_color=color,
                showlegend=False,
                bingroup=i,
                opacity=0.3,
            ),
            row=row[1],
            col=col[1],
        )

    fig.update_layout(barmode="overlay", bargap=0.0)

    fig.update_yaxes(
        title="Variance [AU²]",
        range=[0, var_data.max() * 1.02],
        row=row[0],
        col=col[0],
    )
    fig.update_xaxes(
        title="Epoch Nbr", row=row[0], col=col[0], range=[0, len(epo) * 1.01]
    )
    fig.update_xaxes(showticklabels=False, row=row[1], col=col[1])
    fig.update_yaxes(
        showticklabels=False,
        row=row[1],
        col=col[1],
        range=[0, var_data.max() * 1.02],
    )
    fig.update_layout(legend=dict(x=0.0, y=1), title="Component variance")

    if show:
        fig.show()

    return fig


# TODO [ ]: -- Consider if wrappers are needed at all
def plot_epoch_image(
    epo: mne.BaseEpochs,
    df: pd.DataFrame,
    sort_by: str = "",
    color_by: str = "",
    combine: str = "mean",
    plot_mode: str = "full",
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    vmin_q: float = 0.01,
    vmax_q: float = 0.99,
    log_vals: bool = False,
    showscale: bool = True,
):
    """Plot the epoch image of given epochs, wrapper around ./mne_plotting_utils/epoch_image.py::plot_epo_image

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    sort_by : str
        name of column in df to soort the epochs.
    color_by: str
        name of column in df to group the epochs into colors. This will create
        a column left of the y-axis which shows the color
    combine : str
        how to combine accross the channel axis, default is 'mean',
        other options is 'gfp' (global field potential)
        => np.sqrt((x**2).mean(axis=1))
        Note: mne default is gfp
    plot_mode : str
        either `full` or `base64`, if full, creates a full plotly fig,
        else just create axis and fill the background with a matplotlib plot
        which saves a lot of data to be processed by the browser
    fig : go.FigureWidget
        if a figure is provided, plot to this figure using the row and col
        parameters
    row : int
        row parameter used in fig.add_traces to add the plot
    col : int
        col parameter used in fig.add_traces to add the plot
    vmin_q : float
        quantile to limit lower color bound - only used in plot_mode=full
    vmax_q : float
        quantile to limit upper color bound - only used in plot_mode=full
    log_vals : bool
        if true, work on log transformed data
    showscale : bool
        if true, show the colorbar

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
        epo.plot_image(evoked=False)

    """

    return plot_epo_image(
        epo=epo,
        df=df,
        sort_by=sort_by,
        color_by=color_by,
        combine=combine,
        plot_mode=plot_mode,
        fig=fig,
        row=row,
        col=col,
        vmin_q=vmin_q,
        vmax_q=vmax_q,
        log_vals=log_vals,
        showscale=showscale,
    )


def plot_epo_concat(epo: mne.BaseEpochs) -> go.Figure:
    """Plot epochs concatenated along the time axis.

    Parameters
    ----------
    epo : mne.BaseEpochs
        Epochs to plot.

    Returns
    -------
    go.Figure
        Plotly figure with concatenated epoch data.
    """

    dims = epo.get_data().shape
    time = np.arange(dims[0] * dims[2]) / epo.info["sfreq"]
    fig = plot_ts(np.hstack(epo.get_data()).T, x=time, show=False, names=epo.ch_names)
    for i_ch, ch in enumerate(epo.ch_names):
        fig.update_yaxes(title=f"{ch} [V]", row=i_ch + 1, col=1)

    # Lines for where the epochs start
    epo_starts = time[:: dims[2]]
    min_max_values = epo.get_data().min(axis=(0, 2)), epo.get_data().max(axis=(0, 2))
    for i, estart_time in tqdm(enumerate(epo_starts), desc="Adding epo start vlines"):
        # add single trace for each channel -> traces are far quicker than annotations
        for ich, ch in enumerate(epo.ch_names):
            fig.add_scatter(
                x=[estart_time] * 5,
                y=np.linspace(min_max_values[0][ich], min_max_values[1][ich], 5),
                line_width=1,
                line_color="#222",
                name=f"Epo {i}",
                mode="lines",
                opacity=0.5,
                row=ich + 1,
                col=1,
                showlegend=False,
            )

    fig.update_xaxes(title="Time [s]")
    fig.update_layout(title="Concatenated epochs")

    return fig


def plot_psds(
    epo: mne.BaseEpochs,
    show: bool = False,
    psd_kwargs: dict = {},
    color_by: str = "",
    facet_col: str = "channel",
    facet_col_wrap: int = 4,
    px_kwargs: dict = {},
    average_epochs: bool = False,
) -> go.Figure:
    """Plot power spectral densities for MNE epochs.

    Parameters
    ----------
    epo : mne.BaseEpochs
        The epoched time series.
    show : bool, default=False
        If True, display the figure.
    psd_kwargs : dict, default={}
        Keyword arguments passed to compute_psd.
    color_by : str, default=""
        Name of metadata column to color epochs by.
    facet_col : str, default="channel"
        Column name to use for creating faceted subplots.
    facet_col_wrap : int, default=4
        Number of facet columns before wrapping.
    px_kwargs : dict, default={}
        Additional keyword arguments passed to plotly express line plot.
    average_epochs : bool, default=False
        If True, average PSDs across epochs before plotting.

    Returns
    -------
    go.Figure
        Plotly figure with PSD plots.
    """
    mne_psd = epo.compute_psd(n_jobs=-1, **psd_kwargs)
    data = np.vstack(np.transpose(mne_psd.get_data(), (0, 2, 1)))  # type: ignore

    data = 10 * np.log10(data) + 120  # convert mV -> V and to dB

    # to long dataframe for ease of use with plotly express
    df = pd.DataFrame(
        data,
        columns=mne_psd.ch_names,
    )
    df["epo_nr"] = np.repeat(np.arange(len(epo)), mne_psd.get_data().shape[-1])  # type: ignore
    df["freqs"] = np.tile(mne_psd.freqs, len(epo))

    # add metadata for coloring
    if color_by != "":
        df[color_by] = np.repeat(
            epo.metadata[color_by].to_numpy(),  # type: ignore
            mne_psd.get_data().shape[-1],  # type: ignore
        )
    else:
        color_by = "epo_nr"

    if average_epochs:
        idx_cols = ["freqs"] + [color_by] if color_by != "epo_nr" else ["freqs"]
        df = df.groupby(idx_cols)[mne_psd.ch_names].mean().reset_index()
        df["epo_nr"] = -1

    df = df.melt(
        id_vars=["epo_nr", "freqs"] + [color_by],
        # value_vars=[*mne_psd.ch_names],
        value_vars=[c for c in df.columns if c in mne_psd.ch_names],
        var_name="channel",
        value_name="psd",
    )

    fig = px.line(
        df,
        x="freqs",
        y="psd",
        line_group="epo_nr",
        color=color_by,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        **px_kwargs,
    )

    if show:
        fig = fig.update_layout(
            title="Power Spectral Densities",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Power [dB]",
        )

        fig.show()

    return fig


def plot_evoked(
    epo: mne.BaseEpochs,
    dp: Optional[pl.DataFrame] = None,
    time_topo: Optional[list[float]] = None,
    cmap: Optional[dict[str, str]] = None,
) -> go.Figure:
    """Create interactive evoked response plot with optional topoplots.
    
    Visualizes event-related potentials (ERPs) with mean and confidence intervals
    for all channels. Optionally adds topographic maps at specific time points
    to show spatial distribution of activity.
    
    Parameters
    ----------
    epo : mne.BaseEpochs
        MNE Epochs object containing the data to plot. Each channel will be
        displayed as a separate line with mean ± CI across epochs.
    dp : pl.DataFrame, optional
        Pre-computed Polars DataFrame from mne_epochs_to_polars(). If None,
        will be computed automatically. Must contain 'sample_idx', 'epoch_nr',
        'time' columns plus all channel columns. Default is None.
    time_topo : list of float, optional
        Time points (in seconds) at which to display topographic maps. If provided,
        creates a subplot layout with topoplots in the top row and the time series
        in the bottom row. If None, only the time series is displayed. Default is None.
    cmap : dict of str to str, optional
        Custom color map for channels. Keys are channel names, values are hex colors.
        If None, uses Viridis colorscale sampled across all channels. Default is None.
    
    Returns
    -------
    go.Figure
        Plotly Figure with:
        - If time_topo is None: Single plot showing all channel traces
        - If time_topo is provided: Subplot layout with topoplots (top) and 
          time series (bottom) with connecting lines to mark time points
    
    Raises
    ------
    ValueError
        If dp is provided but doesn't contain required columns ('sample_idx', 
        'epoch_nr', 'time', and all channel names from epo).
    
    Examples
    --------
    >>> import mne
    >>> from mdu.plotly.mne_plotting import plot_evoked
    >>> # Load sample data
    >>> sample_data = mne.datasets.sample.data_path()
    >>> raw_fname = sample_data / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    >>> raw = mne.io.read_raw_fif(raw_fname, preload=True)
    >>> raw.pick_types(meg=False, eeg=True)
    >>> 
    >>> # Create epochs
    >>> events_fname = sample_data / 'MEG' / 'sample' / 'sample_audvis_raw-eve.fif'
    >>> events = mne.read_events(events_fname)
    >>> epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5)
    >>> 
    >>> # Simple evoked plot
    >>> fig = plot_evoked(epochs)
    >>> fig.show()
    >>> 
    >>> # With topoplots at specific times
    >>> fig = plot_evoked(epochs, time_topo=[0.1, 0.2, 0.3])
    >>> fig.show()
    >>> 
    >>> # Custom color scheme
    >>> custom_colors = {ch: '#1f77b4' for ch in epochs.ch_names[:10]}
    >>> fig = plot_evoked(epochs, cmap=custom_colors)
    >>> fig.show()
    
    See Also
    --------
    mdu.plotly.multiline.multiline_plot : Underlying plotting function for time series
    mdu.plotly.mne_plotting_utils.topoplot.create_plotly_topoplot : Topoplot creation
    mdu.mne.mne2dataframe.mne_epochs_to_polars : Convert epochs to DataFrame
    
    Notes
    -----
    - Data is automatically scaled to microvolts (µV)
    - Confidence intervals are computed using bootstrapping across epochs
    - Topoplots use Clough-Tocher interpolation for smooth spatial distribution
    - When time_topo is used, connecting lines link time points to their topoplots
    """

    dp = dp if dp is not None else mne_epochs_to_polars(epo)
    if not all([c in dp.columns for c in ["sample_idx", "epoch_nr", "time"]]):
        raise ValueError(
            "DataFrame must contain 'sample_idx', 'epoch_nr', and 'time' columns in addition to all epo.ch_names."
        )

    dpp = dp.unpivot(
        index=["sample_idx", "epoch_nr", "time"],
        on=epo.ch_names,
        value_name="signal",
        variable_name="channel",
    )

    # sample channels from Viridis
    cmap = cmap or dict(
        zip(
            epo.ch_names,
            [
                rgb_to_hex(c)
                for c in px.colors.sample_colorscale(
                    "Viridis", [i / len(epo.ch_names) for i in range(len(epo.ch_names))]
                )
            ],
        )
    )

    figml = (
        multiline_plot(
            # dpp.filter(pl.col.channel.is_in(["Fp1", "Fp2"])),
            dpp,
            x="time",
            y="signal",
            color="channel",
            line_group="epoch_idx",
            mean=True,
            mean_ci=True,
            color_discrete_map=cmap,
        )
        .update_traces(selector=dict(fill="tonexty"), showlegend=False)
        .for_each_trace(
            lambda t: t.update(fillcolor=t.fillcolor.replace("0.2)", "0.1)")),
            selector=dict(fill="tonexty"),
        )
    )

    if time_topo is not None:
        fig = add_time_locked_topo(dp, epo, time_topo, figml)
    else:
        fig = figml

    return fig


def add_time_locked_topo(
    dp: pl.DataFrame,
    epo: mne.BaseEpochs,
    time_topo: list[float],
    figml: go.Figure,
) -> go.Figure:
    """Add topographic maps at specific time points to an evoked plot.
    
    Creates a subplot layout with topoplots in the top row showing spatial
    distribution at specified time points, and the time series plot in the
    bottom row. Adds visual connections between time points and their
    corresponding topoplots.
    
    Parameters
    ----------
    dp : pl.DataFrame
        Polars DataFrame containing epoch data (from mne_epochs_to_polars).
        Must contain 'time' column and all channel columns.
    epo : mne.BaseEpochs
        MNE Epochs object used for channel information and topographic mapping.
    time_topo : list of float
        Time points (in seconds) at which to create topoplots. Will be sorted
        automatically. The closest matching time point in the data will be used.
    figml : go.Figure
        The multiline plot figure to add to the bottom subplot. This should be
        the output from the main evoked plotting function.
    
    Returns
    -------
    go.Figure
        Plotly Figure with subplot layout:
        - Top row: Topoplots at each specified time point
        - Bottom row: Time series plot spanning full width
        - Vertical dashed lines marking topoplot time points
        - Connecting lines from time points to topoplots
        - Shared colorbar for all topoplots
    
    Notes
    -----
    - Topoplots are computed as mean across all epochs at each time point
    - Color scale is normalized across all topoplots using max absolute value
    - Subplot titles show the actual time used (after closest match)
    - Row heights are set to 30% for topoplots, 70% for time series
    - All topoplots share a common coloraxis for consistent scaling
    
    Examples
    --------
    This function is typically called internally by plot_evoked() when
    time_topo is provided, but can be used directly:
    
    >>> import polars as pl
    >>> from mdu.mne.mne2dataframe import mne_epochs_to_polars
    >>> from mdu.plotly.multiline import multiline_plot
    >>> from mdu.plotly.mne_plotting import add_time_locked_topo
    >>> 
    >>> # Prepare data
    >>> dp = mne_epochs_to_polars(epochs)
    >>> dpp = dp.unpivot(
    ...     index=['sample_idx', 'epoch_nr', 'time'],
    ...     on=epochs.ch_names,
    ...     value_name='signal',
    ...     variable_name='channel'
    ... )
    >>> 
    >>> # Create base multiline plot
    >>> figml = multiline_plot(dpp, x='time', y='signal', color='channel')
    >>> 
    >>> # Add topoplots
    >>> fig = add_time_locked_topo(dp, epochs, [0.1, 0.2, 0.3], figml)
    >>> fig.show()
    
    See Also
    --------
    plot_evoked : Main function that calls this internally
    create_plotly_topoplot : Creates individual topoplots
    """
    time_topo = sorted(time_topo)
    # select the closed matching for every ts in time_topo,
    dm = (
        pl.concat(
            [
                dp.filter(
                    pl.col.time == dp[int(np.argmin(np.abs(dp["time"] - ts))), "time"]
                )
                for ts in time_topo
            ]
        )
        .group_by("time", maintain_order=True)
        .agg([pl.col(c).mean().alias(c) for c in epo.ch_names])
    )

    # we use a subplots figure
    fig = make_subplots(
        rows=2,
        cols=len(dm),
        specs=[[{}] * len(dm), [{"colspan": len(dm)}] + [None] * (len(dm) - 1)],
        row_heights=[0.3, 0.7],
        subplot_titles=[
            f"Time: {row['time']:.2f}s" for row in dm.iter_rows(named=True)
        ],  # type: ignore
        vertical_spacing=0.01,
    )
    fig = fig.add_traces(figml.data, rows=2, cols=1)  # type: ignore
    fig = (
        fig.update_layout(
            legend=dict(y=0.7, yanchor="top"),
        )
        .update_xaxes(title_text="Time [s]", row=2, col=1)
        .update_yaxes(title="Signal [μV]", row=2, col=1)
    )

    cext = max(abs(dm[epo.ch_names].to_numpy().flatten()))  # type: ignore

    for icol, row in enumerate(dm.iter_rows(named=True)):
        topo_fig = (
            create_plotly_topoplot(
                np.array([row[c] for c in epo.ch_names]),
                epo,  # type: ignore
                blank_scaling=1,
                contour_kwargs=dict(contours_coloring="heatmap"),
            )
            .update_xaxes(visible=False)
            .update_yaxes(visible=False)
        )
        topo_ext = max(abs(topo_fig.data[0].x)) * 1.05
        fig.add_traces(
            topo_fig.data,
            rows=1,
            cols=icol + 1,  # type: ignore
        ).update_xaxes(
            range=[-topo_ext, topo_ext],
            scaleanchor=f"y{icol + 1}",
            row=1,
            col=icol + 1,
            visible=False,
        ).update_yaxes(
            range=[
                -topo_ext,
                topo_ext + topo_ext * 0.25,
            ],  # some extra for the nose
            scaleratio=1,
            row=1,
            col=icol + 1,
            visible=False,
        )

    fig = fig.update_layout(
        coloraxis=dict(
            cmin=-cext,
            cmax=cext,
            colorscale="RdBu_r",
            colorbar=dict(
                title="Signal [μV]", len=0.2, y=1, thickness=10, yanchor="top"
            ),
        ),
    )

    topo_x_centers = [(icol + 0.5) / len(dm) for icol in range(len(dm))]
    ts_x_positions = (dm["time"].to_numpy() - dp["time"].min()) / (  # type: ignore
        dp["time"].max() - dp["time"].min()
    )

    # draw connecting lines with annotations from ts on the row=2 xaxis to the center of the topoplots in the top row
    for icol, ts in enumerate(dm["time"]):
        # lines for each ts
        fig = fig.add_vline(
            x=ts,
            line=dict(color="gray", width=1, dash="dash"),
            row=2,  # type: ignore
            col=1,  # type: ignore
        )

        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=ts_x_positions[icol],  # x position in evoked plot (data coordinates)
            y0=0.7,  # depending on the row heights
            x1=topo_x_centers[icol],  # normalized x position for topoplot
            y1=0.73,
            line=dict(color="gray", width=1),
        )

    return fig
