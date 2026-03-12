import polars as pl
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from mdu.plotly.shared import hex_to_rgba
from mdu.utils.logging import get_logger
from mdu.plotly.stats import add_cluster_permut_sig_to_plotly

logger = get_logger("mdu.plotly.html_grids")


def multiline_plot(
    dp: pl.DataFrame,
    x: str,
    y: str,
    line_group: str,
    mean: bool = False,
    std: bool = False,
    mean_ci: bool = False,
    single_lines: bool = False,
    fig: go.Figure | None = None,
    add_significance: bool = False,
    significance_line_kwargs: dict = {},
    **kwargs,
) -> go.Figure:
    """Create line plot with multiple groups and optional statistical overlays.

    Generates a Plotly line plot from grouped data with options to display
    individual lines per group, mean lines, standard deviation bands, and
    95% confidence intervals. Supports significance testing between groups.

    Parameters
    ----------
    dp : polars.DataFrame
        Input dataframe containing the data to plot.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    line_group : str
        Column name for grouping individual lines (e.g., subject ID, trial).
    mean : bool, default=False
        If True, plot the mean line for each group.
    std : bool, default=False
        If True, add shaded standard deviation bands around the mean.
    mean_ci : bool, default=False
        If True, add shaded 95% confidence interval bands (SEM * 1.96).
    single_lines : bool, default=False
        If True, plot individual traces for each line_group with low opacity.
    fig : plotly.graph_objects.Figure or None, default=None
        Existing figure to add traces to. If None, creates new figure.
    add_significance : bool, default=False
        If True, add cluster-based permutation test significance indicators.
        Requires exactly two groups.
    significance_line_kwargs : dict, default={}
        Additional keyword arguments passed to significance testing function.
        Can include 'pval' (default=0.05) for significance threshold.
    **kwargs
        Additional keyword arguments passed to plotly.express.line(),
        such as 'color', 'color_discrete_map', etc.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with the multi-line plot and optional statistical overlays.

    Raises
    ------
    ValueError
        If add_significance=True but the data does not contain exactly 2 groups.

    Notes
    -----
    The function may issue UserWarnings in the following cases:
    - If numeric columns are found in grouping columns (may cause aggregation issues).
    - If x-axis column is float type (will be rounded to 10 decimal places).
    - If some groups have null std/CI values (likely due to n=1 samples).

    Examples
    --------
    >>> import polars as pl
    >>> import plotly.express as px
    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     'time': [0, 1, 2] * 4,
    ...     'value': [1, 2, 3, 1.5, 2.5, 3.5, 0.8, 1.8, 2.8, 1.2, 2.2, 3.2],
    ...     'subject': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2',
    ...                 'S3', 'S3', 'S3', 'S4', 'S4', 'S4'],
    ...     'group': ['A', 'A', 'A', 'A', 'A', 'A',
    ...               'B', 'B', 'B', 'B', 'B', 'B']
    ... })
    >>> # Plot with mean and confidence intervals
    >>> fig = multiline_plot(
    ...     df, x='time', y='value', line_group='subject',
    ...     mean=True, mean_ci=True, color='group'
    ... )

    Notes
    -----
    - The 95% CI is calculated as 1.96 * SEM where SEM = std / sqrt(n).
    - When plotting both std and mean_ci, std bands are shown with dotted lines.
    - Significance testing uses cluster-based permutation tests and requires
      exactly two groups in the data.
    - Float x-axis values are rounded to 10 decimal places to ensure proper
      grouping alignment.
    """
    fig = fig or go.Figure()

    if single_lines:
        fig = fig.add_traces(
            px.line(dp, x=x, y=y, line_group=line_group, **kwargs)
            .update_traces(
                opacity=0.2,
                showlegend=False,
            )
            .data
        )

    if mean or std or mean_ci:
        # check if there is a numeric col in grp_cols which is not 'x'
        grp_cols = [c for c in dp.columns if c not in [y, line_group]]
        float_cols = [c for c in grp_cols if c != x and dp.schema[c].is_float()]

        if any(float_cols):
            logger.warning(
                f"Found numeric columns {grp_cols} in grp_cols. "
                "This can cause problems with the groupby.agg."
            )

        if dp.schema[x].is_float():
            float_precission = 10
            logger.info(
                f"x-axis column ('{x}') is float, will round to {float_precission=} decimals"
            )
            dpp = dp.with_columns(pl.col(x).round(float_precission))
        else:
            dpp = dp

        dpg = (
            dpp.group_by(grp_cols, maintain_order=True)
            .agg(
                pl.col(y).mean().alias("mean"),
                pl.col(y).std().alias("std"),
                pl.col(y).count().alias("n"),  # Add count for sample size
            )
            .sort(x)
        )
        if "color" in kwargs:
            dpg = dpg.sort(kwargs["color"], x)  # to keep legends consistent

            # also create a color_discrete_map, to ensure consistent coloring
            uvals = dpg[kwargs["color"]].unique(maintain_order=True).to_list()
            cmap = px.colors.qualitative.Plotly
            kwargs["color_discrete_map"] = kwargs.get(
                "color_discrete_map",
                {val: cmap[i % len(cmap)] for i, val in enumerate(uvals)},
            )

        # 2. Calculate the 95% confidence interval
        # The half-width of the CI is ~1.96 * SEM (Standard Error of the Mean)
        dpg = dpg.with_columns(
            (1.96 * pl.col("std") / pl.col("n").sqrt()).alias("mean_ci_95")
        )
        if dpg.filter(pl.col.mean_ci_95.is_null() | pl.col.std.is_null()).height > 0:
            logger.warning(
                "Some groups have null values for std or mean_ci_95. "
                "This likely means they have only one sample (n=1). The error bands will look odd."
                " A common reason are float values on the x axis which do not align properly."
            )

        if std:
            # group and group align colors if split provided
            grps = dpg.group_by(kwargs.get("color", None), maintain_order=True)
            clr_map = kwargs.get("color_discrete_map", None)
            fill_kwargs = (
                dict(
                    line_dash="dot",
                    opacity=0.3,
                )  # only dashed line if we have the standard error of the mean
                if mean_ci
                else dict(fill="tonexty", opacity=0.1, line_width=0)
            )

            for (clr,), dg in grps:
                fill_kwargs_first = fill_kwargs.copy()
                fill_kwargs_first["fill"] = None  # type: ignore
                clr_kwargs = dict(line_color=clr_map[clr]) if clr_map else dict()

                fig = fig.add_trace(
                    go.Scatter(
                        x=dg[x],
                        y=dg["mean"] + dg["std"],
                        mode="lines",
                        showlegend=False,
                        legendgroup=f"std_{clr}",
                        name="STD upper (do not show)",
                        **fill_kwargs_first,
                        **clr_kwargs,
                    )
                )

                fig = fig.add_trace(
                    go.Scatter(
                        x=dg[x],
                        y=dg["mean"] - dg["std"],
                        mode="lines",
                        name=f"1xSTD ({clr})" if clr else "1xSTD",
                        showlegend=True,
                        legendgroup=f"std_{clr}",
                        **fill_kwargs,
                        **clr_kwargs,
                    )
                )

        if mean_ci:
            # group and group align colors if split provided
            grps = dpg.group_by(kwargs.get("color", None), maintain_order=True)
            clr_map = kwargs.get("color_discrete_map", None)
            fill_kwargs = dict(fill="tonexty", line_width=0)

            for (clr,), dg in grps:
                fill_kwargs_first = fill_kwargs.copy()
                fill_kwargs_first["fill"] = None  # type: ignore
                clr_kwargs = (
                    dict(line_color=clr_map[clr])
                    if clr_map
                    else dict(line_color="rgba(0,0,0,0.1)")
                )

                # opacity for the fill needs to be set via an rgba color
                # do not specify the fill color if we have a `color` that was split
                # but no color map via color_discrete_map
                if "color" in kwargs and not clr_map:
                    pass
                else:
                    rgba_fill_color = hex_to_rgba(
                        clr_kwargs.get("line_color", "#000"), 0.2
                    )
                    fill_kwargs["fillcolor"] = rgba_fill_color

                fig = fig.add_trace(
                    go.Scatter(
                        x=dg[x],
                        y=dg["mean"] + dg["mean_ci_95"],
                        mode="lines",
                        showlegend=False,
                        name="95% SEM upper (do not show)",
                        **fill_kwargs_first,
                        **clr_kwargs,
                    )
                )

                fig = fig.add_trace(
                    go.Scatter(
                        x=dg[x],
                        y=dg["mean"] - dg["mean_ci_95"],
                        mode="lines",
                        name=f"95% SEM ({clr})" if clr else "95% SEM",
                        showlegend=True,
                        **fill_kwargs,
                        **clr_kwargs,
                    )
                )

        # draw mean last -> so it overlaps
        if mean:
            fig = fig.add_traces(
                px.line(dpg, x=x, y="mean", hover_data=["n"], **kwargs)
                .for_each_trace(
                    lambda tr: tr.update(
                        name=f"mean {tr.name} (n={tr.customdata[0][0]})"
                        if tr.name
                        else "mean",  # type: ignore
                    )
                )
                .data
            )

        if add_significance:
            grp_cols = [c for c in dp.columns if c not in [x, y, line_group]]

            if dpp[grp_cols].n_unique() != 2:
                raise ValueError(
                    "Can only draw significance lines for two groups. "
                    f"Got {dpp[grp_cols].n_unique()=} groups. Adjust the group dimensions."
                )

            curves = []
            for _, dg in dpp.group_by(grp_cols, maintain_order=True):
                curves_df = dg.pivot(index=[x], on=line_group, values=y)
                curves.append(
                    curves_df.select([c for c in curves_df.columns if c != x])
                    .to_numpy()
                    .T
                )

            pval = significance_line_kwargs.pop("pval", 0.05)
            fig = add_cluster_permut_sig_to_plotly(
                curves_a=curves[0],
                curves_b=curves[1],
                xaxes_vals=dpp[x].unique(maintain_order=True),  # type: ignore
                pval=pval,
                fig=fig,
                **significance_line_kwargs,
            )

            # shift x position of significance indicator
            ymax = max([max(t.y) for t in fig.data])  # type: ignore
            ymin = min([min(t.y) for t in fig.data])  # type: ignore
            fig = fig.for_each_trace(
                lambda tr: (
                    tr.update(y=(np.zeros_like(tr.y) + (ymin - (ymax - ymin) * 0.05)))
                    if tr.name and "cl_perm" in tr.name
                    else tr
                )
            )

    # sort the items in the legend to start with mean, SEM, STD and then the individual lines
    fig = (
        fig.for_each_trace(
            lambda tr: (
                tr.update(legendrank=1)
                if tr.name and tr.name.startswith("mean")
                else tr
            )  # type: ignore
        )
        .for_each_trace(
            lambda tr: (
                tr.update(legendrank=2)
                if tr.name and tr.name.startswith("95% SEM")
                else tr
            )  # type: ignore
        )
        .for_each_trace(
            lambda tr: (
                tr.update(legendrank=3)
                if tr.name and tr.name.startswith("1xSTD")
                else tr
            )  # type: ignore
        )
    )
    fig = fig.for_each_trace(
        lambda tr: (
            tr.update(showlegend=False)
            if tr.name and tr.name.startswith("cl_perm")
            else tr
        )  # type: ignore
    )

    fig = fig.update_layout(  # type: ignore
        xaxis_title=x,
        yaxis_title=y,
    )

    return fig
