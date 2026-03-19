import re
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import numpy as np

from typing import Any
from mdu.utils.logging import get_logger

logger = get_logger("mdu.plotly.shared")


def extract_subplot_coordinates(fig: go.Figure) -> list[tuple[str, str, str, str]]:
    """Extract subplot coordinates from a Plotly figure with facets.

    Parses the internal grid structure of a Plotly figure to identify subplot
    positions and their corresponding axis references. Only returns coordinates
    for subplots that contain data.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure object with subplots (faceted plots).

    Returns
    -------
    list of tuple of (str, str, str, str)
        List of tuples containing subplot coordinates as
        (row_index, col_index, xaxis_ref, yaxis_ref).
        For example: [('1', '1', 'x', 'y'), ('1', '2', 'x2', 'y2'), ...].

    Examples
    --------
    >>> import plotly.express as px
    >>> df = px.data.tips()
    >>> fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex")
    >>> coords = extract_subplot_coordinates(fig)
    >>> len(coords)
    2
    >>> coords[0]
    ('1', '1', 'x', 'y')

    Notes
    -----
    This function relies on the internal `_grid_str` attribute of Plotly figures,
    which is set when using faceted plots. Regular single-plot figures will not
    have this attribute properly initialized and will raise a TypeError.
    """
    pattern = r"\((\d+),(\d+)\)\s([xy0-9]+),([xy0-9]+)"
    matches = re.findall(pattern, fig._grid_str)  # type: ignore

    # filter only to those pairs that have data (might have empty facets)
    axes_with_data = set(
        [
            tr.xaxis + "|" + tr.yaxis
            for tr in fig.select_traces()
            if tr.xaxis is not None and tr.yaxis is not None
        ]
    )
    matches = [m for m in matches if m[2] + "|" + m[3] in axes_with_data]

    return matches


def rgb_to_hex(rgb_str):
    nums = [int(x) for x in rgb_str.strip("rgb()").split(",")]
    return "#{:02x}{:02x}{:02x}".format(*nums)


def hex_to_rgba(hex_color: str, opacity: float | None = None) -> str:
    """Convert hex color code to RGBA string format.

    Converts a hexadecimal color string (with or without '#' prefix) to an
    RGBA formatted string. Supports both shorthand (#ddd) and full (#888888)
    hex notation.

    Parameters
    ----------
    hex_color : str
        Hexadecimal color string. Accepts formats: '#rrggbb', 'rrggbb',
        '#rgb', or 'rgb' where r, g, b are hex digits.
    opacity : float or None, default=None
        Opacity value between 0.0 (transparent) and 1.0 (opaque).
        If None, defaults to 1.0.

    Returns
    -------
    str
        RGBA formatted string in the form 'rgba(r, g, b, a)' where
        r, g, b are integers 0-255 and a is the opacity.

    Examples
    --------
    >>> hex_to_rgba('#ff0000', opacity=1.0)
    'rgba(255, 0, 0, 1.0)'
    >>> hex_to_rgba('#ddd', opacity=0.7)
    'rgba(221, 221, 221, 0.7)'
    >>> hex_to_rgba('888888', opacity=0.5)
    'rgba(136, 136, 136, 0.5)'
    >>> hex_to_rgba('#00ff00')
    'rgba(0, 255, 0, 1.0)'

    Notes
    -----
    Shorthand hex notation (e.g., '#ddd') is automatically expanded to full
    notation ('#dddddd') before conversion.
    """
    opacity = opacity if opacity is not None else 1.0

    # 1. Remove '#' and handle shorthand hex (e.g., '#ddd' -> 'dddddd')
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)

    # 2. Convert hex R, G, B to decimal and format the rgba string
    r, g, b = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    return f"rgba({r}, {g}, {b}, {opacity})"


def add_meta_info(fig: go.Figure, text: str | list[str]) -> go.Figure:
    """Adds a metadata info symbol 'ⓘ' to a Plotly figure.

    This function adds a 'ⓘ' annotation with hover text to the top-left
    corner of a plot's drawing area. It is subplot-aware and will place an
    icon in each subplot if the figure contains facets. The default styling
    includes a white background and dark grey font color for the icon.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to be modified.
    text : str or list[str]
        The hover text for the info icon. If `fig` is a single plot, this
        should be a single string. If `fig` has subplots (facets), this must
        be a list of strings, where the length of the list matches the number
        of subplots. The texts are applied to subplots by iterating through
        columns first, then rows.

    Returns
    -------
    go.Figure
        The modified Plotly figure object with the added annotation(s).

    Raises
    ------
    ValueError
        If the figure contains subplots and the number of strings in `text`
        does not match the number of subplots.

    Examples
    --------
    >>> import plotly.express as px

    # --- Example 1: Single plot ---
    >>> fig_single = px.scatter(x=[0, 1], y=[0, 1])
    >>> info_text_single = "This is a simple scatter plot."
    >>> fig_single = add_meta_info(fig_single, info_text_single)
    >>> # fig_single.show()

    # --- Example 2: Faceted plot (subplots) ---
    >>> df = px.data.tips()
    >>> fig_facet = px.scatter(df, x="total_bill", y="tip",
    ...                        facet_col="sex", facet_row="smoker")
    >>> # Text order is col-first: (Male, Smoker), (Female, Smoker), (Male, Non-Smoker),
    ... # (Female, Non-Smoker)
    >>> info_texts_facet = [
    ...     "Data for male smokers.",
    ...     "Data for female smokers.",
    ...     "Data for male non-smokers.",
    ...     "Data for female non-smokers."
    ... ]
    >>> fig_facet = add_meta_info(fig_facet, info_texts_facet)
    >>> # fig_facet.show()

    """
    # Standardize input to always be a list
    texts = [text] if isinstance(text, str) else text
    texts = [t.replace("\n", "<br>") if isinstance(t, str) else t for t in texts]

    shared_options = dict(
        text="ⓘ",
        showarrow=False,
        font=dict(color="#333"),
        bgcolor=hex_to_rgba("#fff", 0.8),
        align="left",
        x=0,
        y=1,
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.8)"),
    )

    if fig._has_subplots() and not isinstance(
        text, str
    ):  # when single string also assume no align with subplots
        # get the grid info
        if len(fig.data) == 0:  # type: ignore
            raise ValueError(
                "Figure has no data traces, cannot determine subplot coordinates"
            )

        matches = extract_subplot_coordinates(fig)

        if len(texts) != len(matches):
            raise ValueError(
                f"Number of texts ({len(texts)}) does not match number of subplots ({len(matches)})"
            )

        for i, (row_str, col_str, xref, yref) in enumerate(matches):
            fig.add_annotation(
                hovertext=texts[i],
                xref=xref + " domain",
                yref=yref + " domain",
                **shared_options,
            )
    else:
        fig.add_annotation(
            hovertext=texts[0], xref="paper", yref="paper", **shared_options
        )

    return fig


def format_float_to_text_with_suffix(num: float) -> str:
    """Format a float with metric suffix and up to 2 decimal places.

    Converts a numerical value to a compact string representation using
    metric prefixes (Mio, k, m, µ, n). The function automatically selects
    the appropriate scale and formats the result to a maximum of 2 decimal
    places, removing trailing zeros.

    Parameters
    ----------
    num : float
        The numerical value to format.

    Returns
    -------
    str
        Formatted string with metric suffix.

    Raises
    ------
    TypeError
        If num is not a number (int or float).

    Examples
    --------
    >>> format_float_to_text_with_suffix(12_345_678)
    '12.35Mio'
    >>> format_float_to_text_with_suffix(10_000)
    '10k'
    >>> format_float_to_text_with_suffix(0.01234)
    '12.34m'
    >>> format_float_to_text_with_suffix(1e-6)
    '1µ'
    >>> format_float_to_text_with_suffix(1e-9)
    '1n'
    >>> format_float_to_text_with_suffix(-5000)
    '-5000'

    Notes
    -----
    Scale thresholds and suffixes:
    - >= 1e7: divided by 1e6, suffix 'Mio'
    - >= 1e4: divided by 1e3, suffix 'k'
    - >= 1: no scaling, no suffix
    - >= 1e-4: divided by 1e-3, suffix 'm'
    - >= 1e-7: divided by 1e-6, suffix 'µ'
    - < 1e-7: divided by 1e-9, suffix 'n'

    Special values (inf, -inf, nan, 0) are handled as string conversions
    without scaling.
    """
    if not isinstance(num, (int, float)):
        raise TypeError("Input must be a number.")

    if np.isinf(num) or np.isnan(num):
        return str(num)

    if num == 0:
        return "0"

    sign = "-" if num < 0 else ""
    num = abs(num)

    if num >= 1e7:
        value, suffix = num / 1e6, "Mio"
    elif num >= 1e4:
        value, suffix = num / 1e3, "k"
    elif num >= 1:
        value, suffix = num, ""
    elif num >= 1e-4:
        value, suffix = num / 1e-3, "m"
    elif num >= 1e-7:
        value, suffix = num / 1e-6, "µ"  # Unicode for micro
    else:  # Handles nano and anything smaller
        value, suffix = num / 1e-9, "n"

    rounded_value = round(value, 2)

    if rounded_value == int(rounded_value):
        formatted_num_str = f"{int(rounded_value)}"
    else:
        formatted_num_str = f"{rounded_value:.2f}".rstrip("0").rstrip(".")

    return f"{sign}{formatted_num_str}{suffix}"


def add_jitter(
    df: pl.DataFrame, ycol: str = "mean acc", jitter_max_width: float = 0.1
) -> pl.DataFrame:
    """Add density-weighted horizontal jitter to dataframe for scatter plots.

    Adds a 'jitter' column with horizontal offset values scaled by the density
    of y-values. Points in dense regions get more jitter to reduce overplotting,
    while sparse regions get less jitter for better position accuracy.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe to add jitter column to.
    ycol : str, default="mean acc"
        Column name to use for density calculation.
    jitter_max_width : float, default=0.1
        Maximum jitter width. Actual jitter is scaled by local density
        from -jitter_max_width to +jitter_max_width.

    Returns
    -------
    polars.DataFrame
        Input dataframe with added 'jitter' column containing horizontal
        offset values.

    Raises
    ------
    AssertionError
        If 'bin' or 'scale_factor' columns already exist in the dataframe.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'x': [1, 1, 1, 2, 2, 3],
    ...     'y': [10, 10.1, 10.2, 20, 20.1, 30]
    ... })
    >>> df_jittered = add_jitter(df, ycol='y', jitter_max_width=0.2)
    >>> 'jitter' in df_jittered.columns
    True

    Notes
    -----
    The function creates a 10-bin histogram of the y-column values and scales
    the jitter magnitude by the bin density (count / max_count). This produces
    more jitter where points are densely clustered and less where they are sparse.

    The jitter values are uniformly distributed random values between
    -jitter_max_width and +jitter_max_width, then scaled by the density factor.
    """

    assert "bin" not in df.columns, "bin column already exists"
    assert "scale_factor" not in df.columns, "scale_factor column already exists"

    hist_counts, hist_boundaries = np.histogram(df[ycol], bins=10)

    scale_factor = hist_counts / hist_counts.max()
    bin_labels = [f"bin_{i}" for i in range(len(hist_counts))]
    scale_factor_map = {
        label: factor for label, factor in zip(bin_labels, scale_factor)
    }

    df = (
        df.with_columns(
            pl.col(ycol)
            .cut(breaks=list(hist_boundaries[1:-1]), labels=bin_labels)
            .cast(pl.String)
            .alias("bin")
        )
        .with_columns(
            pl.col.bin.replace(scale_factor_map).cast(pl.Float32).alias("scale_factor"),
        )
        .with_columns(
            pl.Series(
                "jitter",
                np.random.uniform(-jitter_max_width, jitter_max_width, len(df)),
            )
            * pl.col("scale_factor")
        )
    ).drop("bin", "scale_factor")

    return df


def add_median_and_mean_legend_items(fig: go.Figure) -> go.Figure:
    """Add invisible legend entries for median and mean to a Plotly figure.

    Adds two invisible scatter traces to the legend showing the styling
    conventions used for median (black solid line) and mean (red dotted line)
    in box/violin plots. These serve as a legend key without appearing on
    the plot itself.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to add legend items to.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with added median and mean legend entries.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig = add_median_and_mean_legend_items(fig)
    >>> len(fig.data)
    2
    >>> fig.data[0].name
    'median'
    >>> fig.data[1].name
    'mean'

    Notes
    -----
    The traces use x=[None] and y=[None] to make them invisible on the plot
    area while still appearing in the legend. The median is styled with a
    black solid line, and the mean with a red dotted line, matching the
    default styling in box/violin plots.
    """
    fig = fig.add_trace(
        go.Scatter(
            x=[None],  # Makes the trace invisible on the plot area
            y=[None],
            mode="lines+markers",
            name="median",  # This is the text that will appear in the legend
            line=dict(
                color="black",
                width=2,
            ),
            marker=dict(
                size=30,
                symbol="square",
                color="rgba(200, 200, 200, 0.5)",  # Semi-transparent grey
                line=dict(width=1, color="rgba(100, 100, 100, 0.8)"),
            ),
        )
    )

    fig = fig.add_trace(
        go.Scatter(
            x=[None],  # Makes the trace invisible on the plot area
            y=[None],
            mode="lines+markers",
            name="mean",  # This is the text that will appear in the legend
            line=dict(
                color="red",
                width=2,
                dash="dot",
            ),
            marker=dict(
                symbol="square",
                size=30,
                color="rgba(200, 200, 200, 0.5)",  # Semi-transparent grey
                line=dict(width=1, color="rgba(100, 100, 100, 0.8)"),
            ),
        )
    )

    return fig


def violin_with_connected_points(
    dp: pl.DataFrame,
    xcol: str,
    ycol: str,
    line_group: str,
    color_col: str | None = None,
    color_map: dict[Any, str] | None = None,
    box_only: bool = False,
) -> go.Figure:
    """Create violin/box plot with individual data points connected across categories.

    Generates a Plotly figure combining violin or box plots with individual
    sample trajectories shown as connected points. Useful for visualizing
    repeated measures or paired data across conditions.

    Parameters
    ----------
    dp : polars.DataFrame
        Input dataframe containing the data.
    xcol : str
        Column name for x-axis categories.
    ycol : str
        Column name for y-axis values.
    line_group : str
        Column name for connecting individual samples (e.g., subject ID).
        Points with the same value in this column are connected with lines.
    color_col : str or None, default=None
        Column name for color grouping. If None, uses xcol for colors.
    color_map : dict or None, default=None
        Dictionary mapping color_col values to color strings.
        If None, uses Viridis colorscale.
    box_only : bool, default=False
        If True, shows only box plots without violin distributions.
        Boxes are centered at each x-position.

    Returns
    -------
    plotly.graph_objects.Figure
        Combined violin/box plot with connected individual data points.

    Raises
    ------
    AssertionError
        If 'xnum' or 'xaux' columns already exist in the dataframe
        (these are used internally).

    Notes
    -----
    A UserWarning will be issued if more than 10 unique x-values are found
    (plot may be crowded).

    Examples
    --------
    >>> import polars as pl
    >>> # Create paired data (pre/post measurements)
    >>> df = pl.DataFrame({
    ...     'condition': ['pre', 'post'] * 5,
    ...     'score': [10, 12, 9, 11, 11, 13, 8, 10, 12, 14],
    ...     'subject': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5'],
    ...     'group': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    ... })
    >>> fig = violin_with_connected_points(
    ...     df, xcol='condition', ycol='score',
    ...     line_group='subject', color_col='group'
    ... )

    Notes
    -----
    - Individual points are jittered based on local density to reduce overplotting.
    - When box_only=False, violin plots are split by color groups (left/right).
    - The function adds invisible legend entries for median and mean indicators.
    - X-axis positions are internally converted to numeric for proper spacing.
    """
    for xc in ["xnum", "xaux"]:
        assert xc not in dp.columns, (
            f"{xc} column already exists - a `xaux` column is dynamically being used in this function"
        )

    xvals = dp[xcol].unique(maintain_order=True).to_list()
    if len(xvals) > 10:
        logger.warning(
            f"More than 10 xvals found, this will look messy. xvals: {xvals}"
        )

    pos_offsets = dict(zip(xvals, [0.2 if i == 0 else -0.2 for i in range(len(xvals))]))
    if box_only:
        pos_offsets = dict(zip(xvals, [0.0 for i in range(len(xvals))]))

    # add a numeric x columns
    dw = dp.with_columns(
        pl.col(xcol)
        .replace(dict(zip(xvals, range(len(xvals)))))
        .cast(pl.Int8)
        .alias("xnum")
    )

    dw = (
        dw.group_by(xcol, maintain_order=True)
        .map_groups(lambda dg: add_jitter(dg, ycol=ycol, jitter_max_width=0.1))
        .with_columns(
            (
                pl.col("xnum")
                + pl.col(xcol).replace(pos_offsets).cast(pl.Float32)
                + pl.col("jitter")
            ).alias("xaux")
        )
    )

    # generate a color_map with the default colors if none is provided
    color_vals = np.linspace(0, 1, dw[color_col or xcol].n_unique())
    cmap = color_map or dict(
        zip(
            dp[color_col or xcol].unique(maintain_order=True).to_list(),
            px.colors.sample_colorscale("Viridis", list(color_vals)),
        )
    )

    aux_lines = px.line(
        dw,
        x="xaux",
        y=ycol,
        hover_data=[xcol],
        line_group=line_group,
    ).update_traces(
        mode="lines+markers",
        line_color="#aaa",
        line_width=0.5,
        showlegend=False,
        legendgroup="individual",
        hoverinfo=None,
        name="individual<br>sample",
    )

    # Add the violins now so that they overlap
    aux_violin = px.violin(
        dw,
        x="xnum",
        y=ycol,
        color=color_col or xcol,
        box=True,
        color_discrete_map=cmap,
    )
    if box_only:
        aux_violin = aux_violin.update_traces(
            fillcolor="rgba(0,0,0,0)", line_color="rgba(0,0,0,0)", box_width=0.7
        )

    fig = go.Figure()
    fig = fig.add_traces(aux_lines.data)
    fig = fig.add_traces(aux_violin.data)

    if not box_only:
        for i, clr in enumerate(dw[color_col or xcol].unique(maintain_order=True)):
            fig = fig.update_traces(
                selector={
                    "offsetgroup": str(clr)
                },  # offset group will always be string, even if the color_col is float
                side="negative" if i == 0 else "positive",
            )

    # remove the offset group to have the two halfs align
    fig = fig.update_traces(
        selector={"type": "violin"},
        line_width=2,
        meanline_visible=True,
        offsetgroup="single",
        box_fillcolor="rgba(220,220,220, 0.5)",
        box_line_color="#333",
        meanline_color="#f33",
        meanline_width=2,
    )

    fig = fig.update(
        layout=dict(
            xaxis=dict(
                tickvals=np.arange(len(xvals)),
                ticktext=xvals,
                range=[-0.5, len(xvals) - 0.5],
            ),
            yaxis=dict(title=f"{ycol}"),
            font=dict(size=16),
        )
    )

    fig = add_median_and_mean_legend_items(fig)

    if box_only:
        fig = fig.update_traces(
            selector=dict(name="individual<br>sample"),
            line_color="#888",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="#888", width=1)),
        )

    return fig


def make_colorscale_discrete(
    fig: go.Figure,
    zvals: np.ndarray,
    color_scale_name: str = "Viridis",
    zero_color: str | None = None,
) -> go.Figure:
    """Convert continuous colorscale to discrete bins based on unique values.

    Modifies a Plotly figure's coloraxis to use discrete color bins instead of
    a continuous gradient. Each unique value in zvals gets its own color, with
    colorbar ticks centered on each bin.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure with a coloraxis to modify.
    zvals : np.ndarray
        Array of values used to determine discrete bins. Unique integer values
        are extracted and each gets assigned a discrete color.
    color_scale_name : str, default="Viridis"
        Name of Plotly colorscale to sample colors from.
        See plotly.colors for available options.
    zero_color : str or None, default=None
        If provided, overrides the color for the zero bin. Only valid if
        the minimum value in zvals is zero. Useful for highlighting zero
        or background values.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with modified discrete colorscale.

    Raises
    ------
    AssertionError
        If zero_color is provided but the minimum zval is not zero.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> import numpy as np
    >>> # Create heatmap with discrete values
    >>> z = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    >>> fig = go.Figure(data=go.Heatmap(z=z, coloraxis='coloraxis'))
    >>> fig = make_colorscale_discrete(fig, z.flatten(), color_scale_name='Viridis')
    >>> # Highlight zeros in grey
    >>> fig = make_colorscale_discrete(
    ...     fig, z.flatten(), color_scale_name='Viridis', zero_color='#888888'
    ... )

    Notes
    -----
    The function creates discrete bins by mapping each unique integer value to
    a color from the specified colorscale. The colorbar shows tick marks at the
    center of each bin rather than at boundaries, making it clear which color
    corresponds to which value.

    The discrete colorscale is created by sampling colors at evenly-spaced
    intervals from the continuous colorscale and mapping each to a range in
    the normalized [0, 1] colorscale space.
    """

    nzvals = np.unique(zvals)
    nzvals_int = nzvals.astype(int)
    zmin = nzvals_int.min()
    zmax = nzvals_int.max()

    zrange = np.arange(
        zmin, zmax + 2
    )  # +2 to have the first tick in the middle of the first interval and the last tick in the middle of the last interval

    # mapping range to (0, 1) discretization - these are the boundaries, the ticks will be just in the middle
    zrange_norm = (zrange - zmin) / (zmax + 1 - zmin)
    tickvals = (zrange[:-1] + zrange[1:]) / 2

    colorvals = px.colors.sample_colorscale(color_scale_name, len(nzvals))
    discrete_colorscale = []
    for i in range(len(zrange_norm) - 1):
        discrete_colorscale.append([zrange_norm[i], colorvals[i]])
        discrete_colorscale.append([zrange_norm[i + 1], colorvals[i]])
    if zero_color is not None:
        assert discrete_colorscale[0][0] == 0.0, (
            "zero_color can only be set if the minimum zval is zero"
        )
        discrete_colorscale[0][1] = zero_color
        discrete_colorscale[1][1] = zero_color

    fig = fig.update_layout(
        coloraxis=dict(
            colorscale=discrete_colorscale,
            colorbar=dict(
                title="counts",
                tickvals=tickvals,
                ticktext=[str(v) for v in nzvals_int],
                outlinewidth=1,
                tickwidth=2,
            ),
            cmin=zrange[0],
            cmax=zrange[-1],
        ),
    )

    return fig
