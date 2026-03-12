from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Optional

import json
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.graph_objs import _box, _violin
from scipy import stats

from mdu.utils.converters import ToFloatConverter
from mdu.utils.logging import get_logger

log = get_logger("mdu.plotly.stats")


@dataclass
class Cat2Nums:
    """
    A convenience wrapper to track xaxis transformation required for
    adding significance indicators
    """

    ax_cfg: dict
    x_cat_map: dict
    offset_cat_map: dict


def add_ols_fit(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    row: int | None = None,
    col: int | None = None,
    ci_alpha: float = 0.05,
    show_ci: bool = True,
    show_obs_ci: bool = False,
    line_kwargs: dict = {
        "line": {"color": "#222"},
    },
    ci_kwargs: dict = {
        "fill": "toself",
        "fillcolor": "#222",
        "line_color": "#222",
        "opacity": 0.2,
    },
    obs_ci_kwargs: dict = {
        "line": {"dash": "dash", "color": "#222"},
        "opacity": 0.5,
    },
) -> go.Figure:
    """Add an OLS fit to a plotly figure

    Parameters
    ----------
    fig : go.Figure
        figure to add to

    x : np.ndarray
        array of endogenous variables (currently only 1D supported)

    y : np.ndarray
        array of exogenous variables to fit to

    row : int | None
        used to add to specific subplot

    col : int | None
        used to add to specific subplot

    ci_alpha : float
        alpha value to use for confidence intervals. Default is 0.05 -> CIs are
        2.5% and 97.5% quantiles

    show_ci : bool
        show the confidence interval

    show_obs_ci : bool
        show the observed confidence interval / prediction interval

    line_kwargs : dict
        options passed to the plotting of the fit line

    ci_kwargs : dict
        options passed to the plotting of the confidence interval

    obs_ci_kwargs : dict
        options passed to the plotting of the observed confidence interval


    Returns
    -------
    go.Figure
        modified figure

    """
    return add_statsmodel_fit(
        fig,
        x,
        y,
        row=row,
        col=col,
        fitfunc=sm.OLS,
        ci_alpha=ci_alpha,
        show_ci=show_ci,
        show_obs_ci=show_obs_ci,
        line_kwargs=line_kwargs,
        ci_kwargs=ci_kwargs,
        obs_ci_kwargs=obs_ci_kwargs,
    )


def add_statsmodel_fit(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    fitfunc: Callable = sm.OLS,
    row: int | None = None,
    col: int | None = None,
    ci_alpha: float = 0.05,
    show_ci: bool = True,
    show_obs_ci: bool = False,
    line_kwargs: dict = {
        "line": {"color": "#222"},
    },
    ci_kwargs: dict = {
        "fill": "toself",
        "fillcolor": "#222",
        "line_color": "#222",
        "opacity": 0.2,
    },
    obs_ci_kwargs: dict = {
        "line": {"dash": "dash", "color": "#222"},
        "opacity": 0.5,
    },
) -> go.Figure:
    """Add statistical model fit with confidence intervals to Plotly figure.

    Fits a statsmodels regression model (OLS, GLM, etc.) to the data and adds
    the fitted line, confidence intervals, and optional prediction intervals
    to an existing Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to add the fit to.
    x : np.ndarray
        1D array of independent variable values (predictor).
    y : np.ndarray
        1D array of dependent variable values (response).
    fitfunc : Callable, default=statsmodels.api.OLS
        Statsmodels model class to use for fitting (e.g., sm.OLS, sm.GLM).
    row : int or None, default=None
        Subplot row index (1-based) to add fit to. None adds to main plot.
    col : int or None, default=None
        Subplot column index (1-based) to add fit to. None adds to main plot.
    ci_alpha : float, default=0.05
        Significance level for confidence intervals. Default 0.05 gives
        95% confidence intervals (2.5% and 97.5% quantiles).
    show_ci : bool, default=True
        If True, display shaded confidence interval around the fit line.
    show_obs_ci : bool, default=False
        If True, display dashed lines for prediction interval (observation CI).
    line_kwargs : dict, default={'line': {'color': '#222'}}
        Keyword arguments passed to the fit line scatter trace.
    ci_kwargs : dict
        Keyword arguments passed to the confidence interval filled area.
        Default creates semi-transparent shaded region.
    obs_ci_kwargs : dict
        Keyword arguments passed to the observation CI (prediction interval) lines.
        Default creates dashed lines.

    Returns
    -------
    plotly.graph_objects.Figure
        Modified figure with added fit line and optional confidence intervals.

    Examples
    --------
    >>> import numpy as np
    >>> import plotly.express as px
    >>> import statsmodels.api as sm
    >>> # Create sample data with linear relationship + noise
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 * x + 5 + np.random.normal(0, 2, 50)
    >>> # Create scatter plot
    >>> fig = px.scatter(x=x, y=y)
    >>> # Add OLS fit with 95% CI
    >>> fig = add_statsmodel_fit(fig, x=x, y=y, show_ci=True)
    >>> # fig.show()

    >>> # Add fit to specific subplot with prediction interval
    >>> from plotly.subplots import make_subplots
    >>> fig = make_subplots(rows=1, cols=2)
    >>> fig.add_scatter(x=x, y=y, mode='markers', row=1, col=1)
    >>> fig = add_statsmodel_fit(
    ...     fig, x=x, y=y, row=1, col=1,
    ...     show_ci=True, show_obs_ci=True
    ... )

    Notes
    -----
    - The function automatically adds a constant (intercept) to the model
    - x-values are sorted before fitting to ensure proper line rendering
    - Confidence interval (CI) represents uncertainty in the mean prediction
    - Observation CI (prediction interval) represents uncertainty for new observations
    - Custom statsmodels models can be used via the fitfunc parameter
    """
    tfc = ToFloatConverter()
    xorig = x.copy()
    sort_idx = np.argsort(xorig)
    xsorted = xorig[sort_idx]
    x = tfc.to_float(x[sort_idx])
    y = y[sort_idx]

    assert len(x.shape) == 1, "x must be a 1D array - TODO: inplement more"

    # add constant for intercept
    x = sm.add_constant(x)  # type: ignore
    model = fitfunc(y, x).fit()
    statframe = model.get_prediction(x).summary_frame(alpha=ci_alpha)

    if show_ci:
        fig.add_scatter(
            x=np.hstack([xsorted, xsorted[::-1]]),
            y=np.hstack([statframe["mean_ci_upper"], statframe["mean_ci_lower"][::-1]]),
            name=f"{ci_alpha:.0%} fit CI",
            hoverinfo="skip",  # hover with the filled trace is tricky
            mode="lines",
            **ci_kwargs,
            row=row,
            col=col,
            legendrank=2,
        )

    if show_obs_ci:
        fig.add_scatter(
            x=xsorted,
            y=statframe["obs_ci_upper"],
            name=f"{ci_alpha:.0%} fit obs CI upper",
            mode="lines",
            legendgroup="obs_ci",
            hovertemplate="Obs CI: %{y}<br>x: %{x}",
            legendrank=3,
            row=row,
            col=col,
            **obs_ci_kwargs,
        )
        fig.add_scatter(
            x=xsorted,
            y=statframe["obs_ci_lower"],
            name=f"{ci_alpha:.0%} fit obs CI lower",
            hovertemplate="Obs CI: %{y}<br>x: %{x}",
            row=row,
            col=col,
            mode="lines",
            legendrank=3,
            legendgroup="obs_ci",
            **obs_ci_kwargs,
        )

    stat_text = "<br>".join(f"{model.summary()}".split("\n")[:-5])

    fig.add_scatter(
        x=xsorted,
        y=statframe["mean"],
        mode="lines",
        name="fit line",
        hovertemplate="<b>Pred. mean: %{y}, x: %{x}</b><br><br>%{text}",
        hoverlabel=dict(font=dict(family="monospace"), bgcolor="#ccc"),
        text=[stat_text] * len(xsorted),
        legendrank=1,
        row=row,
        col=col,
        **line_kwargs,
    )

    return fig


class ModeNotImplementedError(ValueError):
    pass


def add_box_significance_indicator(
    fig: go.Figure,
    same_legendgroup_only: bool = False,
    xval_pairs: list[tuple] | None = None,
    color_pairs: list[tuple] | None = None,
    stat_func: Callable = stats.ttest_ind,
    p_quantiles: tuple = (0.05, 0.01),
    rel_y_offset: float = 0.05,
    only_significant: bool = True,
) -> go.Figure:
    """Add statistical significance indicators between box or violin plots.

    Automatically calculates pairwise statistical tests between groups in a
    box/violin plot and adds significance indicators (*, **, ***, ns) with
    connecting lines above the plot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure containing box or violin plot traces.
    same_legendgroup_only : bool, default=False
        If True, only calculate significance between groups with the same
        legend group (typically same color). If False, tests all combinations.
    xval_pairs : list of tuple or None, default=None
        Specific x-axis value pairs to test. If None, tests all combinations.
        Example: [(0, 1), (1, 2)] tests only these pairs.
    color_pairs : list of tuple or None, default=None
        Specific color/legend group pairs to test. Only used when
        same_legendgroup_only=False. If None, tests all combinations.
    stat_func : Callable, default=scipy.stats.ttest_ind
        Statistical test function to use. Must return a test statistic and
        p-value. Common options: stats.ttest_ind, stats.mannwhitneyu.
    p_quantiles : tuple, default=(0.05, 0.01)
        P-value thresholds for significance levels. Default maps to:
        p < 0.01: '**', p < 0.05: '*', p >= 0.05: 'ns'
    rel_y_offset : float, default=0.05
        Relative vertical offset (as fraction of y-range) for significance
        indicators above the maximum data point.
    only_significant : bool, default=True
        If True, only show significant indicators. If False, show all
        comparisons including 'ns' (not significant).

    Returns
    -------
    plotly.graph_objects.Figure
        Modified figure with significance indicators added as shapes and
        annotations.

    Examples
    --------
    >>> import plotly.express as px
    >>> import pandas as pd
    >>> from scipy import stats
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    ...     'value': [1, 2, 3, 4, 5, 6, 2, 3, 4],
    ...     'category': ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    ... })
    >>> # Create box plot
    >>> fig = px.box(df, x='group', y='value', color='category')
    >>> # Add significance indicators between all groups
    >>> fig = add_box_significance_indicator(
    ...     fig,
    ...     stat_func=stats.ttest_ind,
    ...     only_significant=True
    ... )

    >>> # Test only specific pairs
    >>> fig = add_box_significance_indicator(
    ...     fig,
    ...     xval_pairs=[('A', 'B'), ('B', 'C')],
    ...     same_legendgroup_only=True
    ... )

    Notes
    -----
    - Significance levels are indicated by asterisks: * (p<0.05), ** (p<0.01)
    - The function automatically converts categorical x-axes to numeric for positioning
    - Works with both single-color and multi-color (grouped) box/violin plots
    - For paired data, use appropriate stat_func like stats.wilcoxon
    - The function modifies the figure's x-axis to numeric if it's categorical
    """

    # ----------------------------------------------------------------------
    # Hypothesis tests
    # ----------------------------------------------------------------------
    df_data = plot_data_to_dataframe(fig)

    # Do all paired tests for each axis combination (usually subplot) separately
    dsigs = []
    for axes, dg in df_data.groupby(["xaxis", "yaxis"]):
        dsig = group_paired_tests(
            dg,
            group_cols=["offsetgroup", "legendgroup", "name", "x"],
            value_col="y",
            test_func=stat_func,
        ).assign(xaxis=axes[0], yaxis=axes[1])  # type: ignore
        dsigs.append(dsig)

    dsall = pd.concat(dsigs)
    ds = dsall.copy()  # copy to modify later

    # ----------------------------------------------------------------------
    # Limit results according to specified
    # ----------------------------------------------------------------------
    # limit stats according to config
    # >> for xvals
    if xval_pairs is not None:
        dsf = [
            ds[
                ((ds["x_g1"] == xv1) & (ds["x_g2"] == xv2))
                | ((ds["x_g1"] == xv2) & (ds["x_g2"] == xv1))
            ]
            for xv1, xv2 in xval_pairs
        ]

        ds = pd.concat(dsf)

    # >> for colors
    if same_legendgroup_only:
        ds = ds[ds["legendgroup_g1"] == ds["legendgroup_g2"]]
    elif color_pairs is not None:
        dsf = [
            ds[
                ((ds["legendgroup_g1"] == cv1) & (ds["legendgroup_g2"] == cv2))
                | ((ds["legendgroup_g1"] == cv2) & (ds["legendgroup_g2"] == cv1))
            ]
            for cv1, cv2 in color_pairs
        ]

        ds = pd.concat(dsf)  # type: ignore

    # ----------------------------------------------------------------------
    # Prepare axis
    # ----------------------------------------------------------------------

    cat2nums = (
        None  # working with cat2num to make the rest of the processing idem potent
    )
    if not pd.api.types.is_numeric_dtype(
        dsall["x_g1"]
    ) or not pd.api.types.is_numeric_dtype(dsall["x_g2"]):
        fig, cat2nums = make_xaxis_numeric(fig, cat2num=cat2nums)

    ax_tuples = get_subplot_axis(fig)
    # add the layout per xaxis - to restore labels later
    for i in range(len(ax_tuples)):
        xaxis_layout = json.loads(fig.layout[ax_tuples[i]["xaxis_label"]].to_json())  # type: ignore
        ax_tuples[i]["xaxis_layout"] = xaxis_layout

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------
    for gk, dg in ds.groupby(["xaxis", "yaxis"]):  # type: ignore
        cat2num = None
        if cat2nums is not None:
            # select the correct Cat2Nums wrapper according to the anchor string
            cat2num = (
                cat2nums[0]
                if len(cat2nums) == 1
                else [
                    c2n
                    for c2n in cat2nums
                    if c2n.ax_cfg["xaxis_anchor"] == gk[0]  # type: ignore
                    and c2n.ax_cfg["yaxis_anchor"] == gk[1]  # type: ignore
                ][0]
            )
        # just get the relevant map between axis labels and irow, icol
        ax_cfg = [
            cfg
            for cfg in ax_tuples
            if cfg["xaxis_anchor"] == gk[0] and cfg["yaxis_anchor"] == gk[1]  # type: ignore
        ][0]

        ys = df_data[(df_data.xaxis == gk[0]) & (df_data.yaxis == gk[1])]["y"]  # type: ignore
        dy = ys.max() - ys.min()

        # draw the indicator lines
        yline = ys.min() - dy * rel_y_offset

        # sort according to `left` x for cleaner look with multiple bars
        dg = dg.sort_values(["x_g1", "x_g2", "legendgroup_g1"])  # type: ignore

        for _, row in dg.iterrows():
            msk = [row.pval < pq for pq in p_quantiles]
            if not any(msk):
                sig_label = "ns<br>"  # add the <br> to offset position upwards
                if only_significant:
                    continue

            elif all(msk):
                sig_label = "*" * len(msk)
            else:
                # get the first False
                sig_label = "*" * msk.index(False)

            if cat2num:
                x1p = (
                    cat2num.x_cat_map[row.x_g1]
                    + cat2num.offset_cat_map[row.offsetgroup_g1]
                )
                x2p = (
                    cat2num.x_cat_map[row.x_g2]
                    + cat2num.offset_cat_map[row.offsetgroup_g2]
                )
            else:  # already all numeric
                x1p = row.x_g1
                x2p = row.x_g2

            xmid = (x1p + x2p) / 2

            irow = ax_cfg["row"]
            icol = ax_cfg["col"]

            # the line
            fig.add_trace(
                go.Scatter(
                    x=[x1p, x2p],
                    y=[yline, yline],
                    mode="lines+markers",
                    marker={"size": 10, "symbol": "line-ns", "line_width": 1},
                    line_color="#555555",
                    line_dash="dot",
                    showlegend=False,
                    hoverinfo="skip",  # disable hover
                ),
                row=irow,
                col=icol,
            )

            # Marker for hover
            hovertemplate = (
                f"<b>{row.x_g1}</b> vs. <b>{row.x_g2}</b><br>"
                f"<b>test function</b>: {stat_func.__name__}<br>"
                f"<b>test function kwargs: </b>: {row.kwargs}"
                f"<br><b>N-dist1</b>: {row.n1}<br><b>N-dist2</b>: {row.n2}<br>"
                f"<b>statistic</b>: {row.stat}<br><b>pval</b>: {row.pval:.6f}<br>"
                f"<b>Shapiro-Wilk pvals</b>: dist1_p={row.shapiro_1_p:.6f}, dist2_p={row.shapiro_2_p:.6f}<extra></extra>"
            )

            fig.add_trace(
                go.Scatter(
                    x=[xmid],
                    y=[yline],
                    mode="text",
                    text=[sig_label],
                    showlegend=False,
                    name=sig_label,
                    marker_line_width=1,
                    marker_size=10,
                    hovertemplate=hovertemplate,
                ),
                row=irow,
                col=icol,
            )

            # Offset next line
            yline -= dy * rel_y_offset

        # adjust the y range to a reasonable size
        fig = fig.update_yaxes(
            range=[yline - dy * rel_y_offset, max(ys) + dy * rel_y_offset]
        )

    # restore the x-axes layout
    for ax_tuple in ax_tuples:
        fig = fig.update_xaxes(
            ax_tuple["xaxis_layout"],
            row=ax_tuple["row"],
            col=ax_tuple["col"],
        )

    return fig


def plot_data_to_dataframe(
    fig: go.Figure,
) -> pd.DataFrame:
    """Extract the data of the box/violin plots to a single data frame"""

    dists = [
        elm
        for elm in fig.data
        if isinstance(elm, _box.Box) or isinstance(elm, _violin.Violin)
    ]

    dfs = []
    for dist in dists:
        df = pd.DataFrame(
            {
                "x": dist.x,
                "y": dist.y,
                "offsetgroup": dist.offsetgroup,
                "legendgroup": dist.legendgroup,
                "name": dist.name,
                "yaxis": dist.yaxis,
                "xaxis": dist.xaxis,
            }
        )
        dfs.append(df)

    return pd.concat(dfs, axis=0).reset_index(drop=True)


def group_paired_tests(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    test_func: Callable = stats.ttest_ind,
    test_func_kwargs: Optional[dict] = {"equal_var": False},
) -> pd.DataFrame:
    """
    Perform pairwise statistical tests between all combinations of groups.

    This function groups the data by specified columns and performs pairwise
    statistical tests between all combinations of groups. It returns a DataFrame
    containing test statistics, p-values, and group information for each comparison.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to test
    group_cols : list[str]
        Column names to use for grouping the data. All combinations of unique
        values in these columns will be compared pairwise
    value_col : str
        Column name containing the values to compare statistically
    test_func : Callable, default=stats.ttest_ind
        Statistical test function to use for comparisons. Must accept two arrays
        and return an object with 'statistic', 'pvalue', and 'df' attributes
    test_func_kwargs : dict or None, default={"equal_var": False}
        Keyword arguments to pass to the test function

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per pairwise comparison with columns:
        - {group_col}_g1 : values from first group for each grouping column
        - {group_col}_g2 : values from second group for each grouping column
        - stat : test statistic value
        - pval : p-value from the statistical test
        - dof : degrees of freedom
        - n1 : sample size of first group
        - n2 : sample size of second group
    """
    grps = df.groupby(group_cols)

    data = []
    for (gk1, dg1), (gk2, dg2) in combinations(grps, 2):
        # -- Check if the test_func accepts all test_func_kwargs, if not drop the kwargs that cannot be processed and warn via logger
        import inspect

        if test_func_kwargs:
            sig = inspect.signature(test_func)
            accepted_kwargs = set(sig.parameters.keys())
            filtered_kwargs = {
                k: v for k, v in test_func_kwargs.items() if k in accepted_kwargs
            }
            dropped = set(test_func_kwargs.keys()) - set(filtered_kwargs.keys())
            if dropped:
                log.warning(
                    f"Dropped unsupported kwargs for {test_func.__name__}: {dropped}"
                )
        else:
            filtered_kwargs = {}

        test = test_func(dg1[value_col], dg2[value_col], **filtered_kwargs)  # type: ignore

        # add Shapiro-Wilk as test for normality and Levene's for text on equal variance
        shapiro_1_stat, shapiro_1_p = stats.shapiro(dg1[value_col])
        shapiro_2_stat, shapiro_2_p = stats.shapiro(dg2[value_col])

        dr = pd.DataFrame(
            {
                **dict(zip([g + "_g1" for g in group_cols], gk1)),  # type: ignore
                **dict(zip([g + "_g2" for g in group_cols], gk2)),  # type: ignore
                "stat": test.statistic,
                "pval": test.pvalue,
                "dof": test.df if "df" in test.__dir__() else "NA",
                "n1": len(dg1),
                "n2": len(dg2),
                "shapiro_1_p": shapiro_1_p,
                "shapiro_2_p": shapiro_2_p,
                "kwargs": str(filtered_kwargs),
            },
            index=[0],  # type: ignore
        )

        data.append(dr)

    return pd.concat(data)


def make_xaxis_numeric(
    fig: go.Figure, cat2num: Optional[list[Cat2Nums]] = None
) -> tuple[go.Figure, list[Cat2Nums]]:
    """
    Convert categorical x-axes to numeric for all subplots in a figure.

    This function transforms categorical x-axes to numeric linear axes while
    preserving the categorical labels as tick text. This conversion is necessary
    for properly drawing significance indicator lines between categorical groups.
    The function handles multiple subplots separately, allowing for heterogeneous
    x-axis categories that may result from separate creation with make_subplots.

    Parameters
    ----------
    fig : go.Figure
        The plotly figure to modify. Can contain single plot or multiple subplots
    cat2num : list[Cat2Nums] or None, default=None
        Optional existing list of Cat2Nums objects to extend. If None, a new
        list is created. Allows for idempotent transformations

    Returns
    -------
    fig : go.Figure
        The modified figure with numeric x-axes
    cat2num : list[Cat2Nums]
        List of Cat2Nums objects containing the transformation metadata for
        each subplot, including:
        - ax_cfg: axis configuration dict with subplot information
        - x_cat_map: mapping from categorical values to numeric positions
        - offset_cat_map: mapping from offsetgroups to offset values

    Notes
    -----
    The function modifies the figure in-place and also returns it. For each
    categorical x-axis found, it:
    1. Creates a mapping from categorical values to numeric positions
    2. Computes offsets for multiple groups at the same x position
    3. Updates trace data to use numeric x values
    4. Updates axis properties to show categorical labels at numeric positions
    """

    ax_tuples = get_subplot_axis(fig)

    cat2num = [] if cat2num is None else cat2num

    for ax_cfg in ax_tuples:
        # only use axis if there is data on the axis
        traces = [
            tr
            for tr in fig.select_traces(
                row=ax_cfg["row"],
                col=ax_cfg["col"],
            )
            if tr.type in ("box", "violin")
        ]

        if traces != []:
            # create a map for values
            xvals = []
            offset_grs = []
            for tr in traces:
                xvals.append(tr.x)
                offset_grs.append([tr.offsetgroup])

            uxvals = np.unique(np.hstack(xvals))
            uoffsets = np.unique(np.hstack(offset_grs))

            x_cat_map = dict(zip(uxvals, range(len(uxvals))))

            # +2 to exclude left and right boundary
            offsets = np.linspace(-0.5, 0.5, len(uoffsets) + 2)[1:-1]
            offset_cat_map = dict(zip(uoffsets, offsets))

            # Convert box/violin trace x-values to numeric positions
            for tr in traces:
                tr.x = [x_cat_map[x] + offset_cat_map[tr.offsetgroup] for x in tr.x]
                if isinstance(tr, _box.Box):
                    tr.width = 0.8 / (len(uoffsets) + 2)

            # Update x-axis to linear with categorical labels at numeric positions
            fig.update_xaxes(
                type="linear",
                tickmode="array",
                range=[-0.5, len(uxvals) - 0.5],
                tickvals=list(x_cat_map.values()),
                ticktext=list(x_cat_map.keys()),
                row=ax_cfg["row"],
                col=ax_cfg["col"],
            )

            cat2num.append(
                Cat2Nums(
                    ax_cfg=ax_cfg,
                    x_cat_map=x_cat_map,
                    offset_cat_map=offset_cat_map,
                )
            )

    return fig, cat2num


def get_subplot_axis(fig: go.Figure) -> list[dict]:
    """
    Get axis configuration information for each subplot in a figure.

    This function extracts axis naming and position information for all subplots
    in a plotly figure. For figures without subplots, it returns a single
    configuration dict with default axis names.

    Parameters
    ----------
    fig : go.Figure
        The plotly figure to analyze. Can be a single plot or contain multiple
        subplots created with make_subplots

    Returns
    -------
    list[dict]
        List of dictionaries, one per subplot, each containing:
        - xaxis_label : str
            The layout key for the x-axis (e.g., 'xaxis', 'xaxis2')
        - yaxis_label : str
            The layout key for the y-axis (e.g., 'yaxis', 'yaxis2')
        - xaxis_anchor : str
            The trace anchor name for the x-axis (e.g., 'x', 'x2')
        - yaxis_anchor : str
            The trace anchor name for the y-axis (e.g., 'y', 'y2')
        - row : int or None
            The row number (1-indexed) for subplots, None for single plots
        - col : int or None
            The column number (1-indexed) for subplots, None for single plots

    Notes
    -----
    For single plots (no subplots), returns a list with one dict where row and
    col are None. This allows trace selectors with row=None to work correctly.
    """

    # -> not a subplot return simple
    if fig._grid_ref is None:
        return [
            {
                "xaxis_label": "xaxis",
                "yaxis_label": "yaxis",
                "xaxis_anchor": "x",
                "yaxis_anchor": "y",
                "row": None,  # will allow selectors with row=None to work..
                "col": None,
            }
        ]

    ax_tuples = []
    for ir, subplots_row in enumerate(fig._grid_ref):
        for ic, suplot_ref in enumerate(subplots_row):
            ax_tuples.append(
                {
                    "xaxis_label": suplot_ref[0].layout_keys[0],
                    "yaxis_label": suplot_ref[0].layout_keys[1],
                    "xaxis_anchor": suplot_ref[0].trace_kwargs["xaxis"],
                    "yaxis_anchor": suplot_ref[0].trace_kwargs["yaxis"],
                    "row": ir + 1,
                    "col": ic + 1,
                }
            )

    return ax_tuples


# Cluster permutation results to a plotly plot
def add_cluster_permut_sig_to_plotly(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    fig: go.Figure,
    xaxes_vals: None | list | np.ndarray = None,  # noqa
    row: None | int = None,
    col: None | int = None,
    pval: float = 0.05,
    nperm: int = 1024,
    mode: str = "line",
) -> go.Figure:
    """Add cluster-based permutation test significance indicators to time series plot.

    Performs cluster-based permutation testing on two sets of time series curves
    and adds visual indicators of significant time windows to a Plotly figure.
    This is particularly useful for identifying periods of significant difference
    in EEG/MEG data, behavioral timecourses, or any multi-trial time series.

    Parameters
    ----------
    curves_a : np.ndarray
        First set of curves with shape (n_trials, n_timepoints).
        Each row is one trial/observation.
    curves_b : np.ndarray
        Second set of curves with shape (n_trials, n_timepoints).
        Must have same number of timepoints as curves_a.
    fig : plotly.graph_objects.Figure
        Plotly figure to add significance indicators to.
    xaxes_vals : list, np.ndarray, or None, default=None
        Time values for the x-axis. If None, uses indices 0, 1, 2, ...
    row : int or None, default=None
        Subplot row index (1-based) to add indicators to. None for main plot.
    col : int or None, default=None
        Subplot column index (1-based) to add indicators to. None for main plot.
    pval : float, default=0.05
        Significance threshold for both F-statistic and cluster p-values.
        Clusters with p < pval are marked as significant.
    nperm : int, default=1024
        Number of permutations for the cluster test. Higher values give
        more stable results but take longer to compute.
    mode : str, default='line'
        Visualization style for significance indicators:
        - 'line': Black horizontal line at bottom with "p-val" label (simple)
        - 'spark': Sparklines showing F-statistic values on a secondary y-axis (right side)
        - 'p_bg': Colored background for significant regions
        - 'p_colorbar': Vertical colorbar indicating p-values

    Returns
    -------
    plotly.graph_objects.Figure
        Modified figure with cluster-based significance indicators added.

    Examples
    --------
    >>> import numpy as np
    >>> import plotly.graph_objects as go
    >>> # Simulate two groups of time series (e.g., EEG trials)
    >>> n_trials, n_time = 20, 100
    >>> time = np.linspace(0, 1, n_time)
    >>> # Group A: baseline activity
    >>> curves_a = np.random.randn(n_trials, n_time)
    >>> # Group B: enhanced activity in middle period (time 0.4-0.6)
    >>> curves_b = np.random.randn(n_trials, n_time)
    >>> curves_b[:, 40:60] += 1.5  # Add signal in middle
    >>> # Create figure with mean lines
    >>> fig = go.Figure()
    >>> fig.add_scatter(x=time, y=curves_a.mean(axis=0), name='Group A')
    >>> fig.add_scatter(x=time, y=curves_b.mean(axis=0), name='Group B')
    >>> # Add cluster permutation test
    >>> fig = add_cluster_permut_sig_to_plotly(
    ...     curves_a, curves_b, fig,
    ...     xaxes_vals=time,
    ...     pval=0.05,
    ...     nperm=1000,
    ...     mode='line'
    ... )

    >>> # Use with subplots
    >>> from plotly.subplots import make_subplots
    >>> fig = make_subplots(rows=1, cols=2)
    >>> # Add data to subplot 1
    >>> fig.add_scatter(x=time, y=curves_a.mean(axis=0), row=1, col=1)
    >>> fig.add_scatter(x=time, y=curves_b.mean(axis=0), row=1, col=1)
    >>> # Add significance to subplot 1
    >>> fig = add_cluster_permut_sig_to_plotly(
    ...     curves_a, curves_b, fig,
    ...     xaxes_vals=time, row=1, col=1, mode='p_bg'
    ... )

    Notes
    -----
    - Uses MNE-Python's cluster permutation test implementation
    - Tests are based on F-statistics with appropriate degrees of freedom
    - Cluster threshold is determined by F-distribution at specified pval
    - Only temporally adjacent significant points form clusters
    - Cluster p-values are corrected for multiple comparisons
    - If no significant clusters found, a log message is issued but no error raised
    - For EEG/MEG data, consider using MNE-Python's native plotting functions

    References
    ----------
    Maris, E., & Oostenveld, R. (2007). Nonparametric statistical testing of
    EEG-and MEG-data. Journal of neuroscience methods, 164(1), 177-190.
    """
    # --> understand the correct degrees of freedom
    n_conditions = 2
    n_observations = max(curves_a.shape[0], curves_b.shape[0])
    dfn = n_conditions - 1  # degrees of freedom numerator
    dfd = n_observations - n_conditions  # degrees of freedom denominator
    thresh = stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

    log.info(f"Calculating cluster permutation in F-stats with {thresh=} and {nperm=}.")

    # the last parameter should be relevant for the adjecency -> here time
    fobs, clust_idx, pclust, h0 = mne.stats.permutation_cluster_test(
        [
            curves_a.reshape(*curves_a.shape, 1),
            curves_b.reshape(*curves_b.shape, 1),
        ],
        threshold=thresh,
        n_permutations=nperm,
    )

    time = (
        np.asarray(xaxes_vals)
        if xaxes_vals is not None
        else np.arange(curves_a.shape[1])
    )

    # dbfig = debug_plot(curves_a, curves_b, fobs, h0, thresh)
    # dbfig.savefig("dbfig_test.png")
    if not any([p < pval for p in pclust]):
        log.info("No significant clusters found!")

    if mode == "line":
        fig = fig_add_clust_line(
            fig=fig,
            clust_idx=clust_idx,
            pclust=pclust,
            pval=pval,
            time=time,
            row=row,
            col=col,
        )

    elif mode == "spark":
        fig = fig_add_clust_spark(
            fig=fig,
            fobs=fobs,
            thresh=float(thresh),
            time=time,
            row=row,
            col=col,
        )

    elif mode == "p_bg":
        fig = fig_add_clust_colorbg(
            fig=fig,
            clust_idx=clust_idx,
            pclust=pclust,
            pval=pval,
            time=time,
            row=row,
            col=col,
        )

    elif mode == "p_colorbar":
        fig = fig_add_clust_colorbar(
            fig=fig,
            clust_idx=clust_idx,
            pclust=pclust,
            pval=pval,
            time=time,
            row=row,
            col=col,
        )

    else:
        raise ModeNotImplementedError(
            f"Unknown {mode=} for adding significance indicators. Valid are: 'line', 'spark', 'p_bg', 'p_colorbar'"
        )

    return fig


def fig_add_clust_spark(
    fig: go.Figure,
    fobs: np.ndarray,
    thresh: float,
    time: np.ndarray,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Add sparklines of F-values and threshold to the figure on a secondary y-axis

    Parameters
    ----------
    fig : go.Figure
        the figure to add the sparklines to

    fobs : np.ndarray
        the observed F-values

    thresh : float
        the F-value threshold

    time : np.ndarray
        the time points

    row : int | None
        the row to add the sparklines to

    col : int | None
        the column to add the sparklines to

    Returns
    -------
    go.Figure
        the figure with the sparklines added on a secondary y-axis
    """
    # Get the appropriate axis configuration for this subplot
    subplot_axes = get_subplot_axis(fig)

    # Find the matching subplot
    matching_ax = None
    for ax_cfg in subplot_axes:
        if ax_cfg["row"] == row and ax_cfg["col"] == col:
            matching_ax = ax_cfg
            break

    if matching_ax is None:
        raise ValueError(f"Could not find subplot at row={row}, col={col}")

    # Determine secondary y-axis names
    # For primary yaxis (or yaxis), secondary is yaxis2
    # For yaxis2, secondary is yaxis3, etc.
    primary_yaxis_label = matching_ax["yaxis_label"]
    primary_yaxis_anchor = matching_ax["yaxis_anchor"]

    # Extract number from yaxis label (e.g., "yaxis2" -> 2, "yaxis" -> 1)
    if primary_yaxis_label == "yaxis":
        primary_num = 1
    else:
        primary_num = int(primary_yaxis_label.replace("yaxis", ""))

    # Calculate secondary axis number (need to find next available)
    # For simplicity, we'll use the pattern: if primary is N, secondary is the next even number
    # This works for typical subplot layouts
    secondary_num = primary_num + 1
    secondary_yaxis_label = f"yaxis{secondary_num}"
    secondary_yaxis_anchor = f"y{secondary_num}"

    # Add traces on secondary y-axis
    # Note: We need to track the index to update yaxis after adding
    trace_start_idx = len(fig.data)

    fig.add_trace(
        go.Scatter(
            x=time,
            y=fobs[:, 0],
            name="F-values",
            mode="lines",
            line=dict(color="#888888", width=1),
            opacity=0.6,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=[time[0], time[-1]],
            y=[thresh, thresh],
            name="F-val thresh",
            mode="lines",
            line_color="#338833",
            line_dash="dash",
            line_width=1,
            opacity=0.6,
        ),
        row=row,
        col=col,
    )

    # Update the traces to use secondary y-axis
    # When using row/col with add_trace, yaxis parameter is overridden
    # so we need to set it manually after adding
    fig.data[trace_start_idx].update(yaxis=secondary_yaxis_anchor)
    fig.data[trace_start_idx + 1].update(yaxis=secondary_yaxis_anchor)

    # Update layout for secondary y-axis
    fig.update_layout(
        {
            secondary_yaxis_label: dict(
                title="F-statistic",
                overlaying=primary_yaxis_anchor,
                side="right",
                showgrid=False,
                zeroline=False,
                anchor=matching_ax["xaxis_anchor"],
            )
        }
    )

    return fig


def fig_add_clust_colorbg(
    fig: go.Figure,
    clust_idx: np.ndarray,
    pclust: np.ndarray,
    pval: float,
    time: np.ndarray,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Add colored vertical rectangles for significant clusters

    Parameters
    ----------
    fig : go.Figure
        the figure to add the colored rectangles to

    clust_idx : np.ndarray
        the cluster indices

    pclust : np.ndarray
        the p-values for each cluster

    pval : float
        the p-value threshold for significance

    time : np.ndarray
        the time points

    row : int | None
        the row to add the rectangles to

    col : int | None
        the column to add the rectangles to

    Returns
    -------
    go.Figure
        the figure with the colored rectangles added
    """
    # color the background
    for cl, p in zip(clust_idx, pclust):
        x = time[cl[0][:]]
        if p < pval:
            log.debug(f"Adding significant values at {x[0]} to {x[-1]}")
            fig.add_vrect(
                x0=x[0],
                x1=x[-1],
                line_width=1,
                line_color="#338833",
                fillcolor="#338833",
                name=f"cl_perm_{cl[0][0]}_{cl[0][-1]}",
                opacity=0.2,
                row=row,  # type: ignore
                col=col,  # type: ignore
            )
        else:
            log.debug(f"Cluster not significant from {x[0]} to {x[-1]}")

    return fig


def fig_add_clust_line(
    fig: go.Figure,
    clust_idx: np.ndarray,
    pclust: np.ndarray,
    time: np.ndarray,
    pval: float = 0.05,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Add a line to the figure for each cluster

    Parameters
    ----------
    fig : go.Figure
        the figure to add the lines to

    clust_idx : np.ndarray
        the cluster indices

    pclust : np.ndarray
        the p-values for each cluster

    time : np.ndarray
        the time points

    pval : float
        the p-value threshold for significance

    row : int | None
        the row to add the lines to

    col : int | None
        the column to add the lines to

    Returns
    -------
    go.Figure
        the figure with the lines added
    """
    for cl, p in zip(clust_idx, pclust):
        if p < pval:
            x = time[cl[0][:]]

            # if x is only a single sample create a line segment with the same width
            # as the samples in time (xaxes values)
            dt = np.diff(time).mean()
            if len(x) == 1:
                x = np.array([x[0] - dt / 2, x[0] + dt / 2])

            fig.add_scatter(
                x=x,
                y=np.ones_like(x),  # change to the correct value outsides
                mode="lines+text",
                line_color="#333",
                line_width=1,
                name=f"cl_perm_{cl[0][0]}_{cl[0][-1]}",
                row=row,
                col=col,
                text=[f"p<{p:.3f}"] + [""] * (len(x) - 1),
                textposition="top right",
            )

    return fig


def fig_add_clust_colorbar(
    fig: go.Figure,
    clust_idx: np.ndarray,
    pclust: np.ndarray,
    pval: float,
    time: np.ndarray,
    row: int | None = None,
    col: int | None = None,
    y_range: tuple[float, float] = (0.9, 1.1),
) -> go.Figure:
    """Add a heatmap colorbar colored by p-values for clusters

    Creates a heatmap bar (default y-range 0.9-1.1) where segments are colored
    grey if not within a significant cluster, and colored with viridis colors
    according to p-values for significant clusters.

    Parameters
    ----------
    fig : go.Figure
        the figure to add the colorbar to

    clust_idx : np.ndarray
        the cluster indices

    pclust : np.ndarray
        the p-values for each cluster

    pval : float
        the p-value threshold for significance

    time : np.ndarray
        the time points

    row : int | None
        the row to add the colorbar to

    col : int | None
        the column to add the colorbar to

    y_range : tuple[float, float]
        the y-axis range for the heatmap bar, default is (0.9, 1.1)

    Returns
    -------
    go.Figure
        the figure with the colorbar added
    """
    # Create a p-value map for all time points
    n_time = len(time)
    # Use a value > pval for non-significant regions (will map to grey)
    pval_map = np.full(n_time, 1.0)

    # Fill in actual p-values for significant clusters
    for cl, p in zip(clust_idx, pclust):
        if p < pval:
            cluster_indices = cl[0][:]
            pval_map[cluster_indices] = p

    log_pval_map = np.log10(pval_map)
    log_pval_threshold = np.log10(pval)
    min_log_pval = (
        log_pval_map[pval_map < pval].min()
        if np.any(pval_map < pval)
        else log_pval_threshold
    )

    scale_min = min_log_pval
    scale_max = 0.0  # log10(1.0) = 0

    # Position where threshold occurs in normalized [0, 1] scale
    if scale_max - scale_min != 0:
        threshold_pos = (log_pval_threshold - scale_min) / (scale_max - scale_min)
    else:
        threshold_pos = 1.0

    # Build colorscale with inverted viridis below threshold, grey above
    viridis_colors = px.colors.sequential.Viridis
    n_viridis = len(viridis_colors)

    colorscale = []

    # Inverted viridis from 0 to threshold_pos (yellow at lowest p-values, blue at threshold)
    for i in range(n_viridis):
        pos = i / (n_viridis - 1) * threshold_pos
        colorscale.append([pos, viridis_colors[n_viridis - 1 - i]])

    # Grey from threshold to max
    colorscale.append([threshold_pos, "#aaa"])
    colorscale.append([1.0, "#aaa"])

    # Create heatmap with two rows to give it height
    z_data = np.vstack([log_pval_map, log_pval_map])

    fig = fig.add_trace(
        go.Heatmap(
            x=time,
            y=list(y_range),
            z=z_data,
            colorscale=colorscale,
            zmin=scale_min,
            zmax=scale_max,
            colorbar=dict(
                title="p-value",
                tickvals=[
                    min_log_pval,
                    log_pval_threshold,
                    log_pval_threshold / 2,
                ],
                ticktext=[
                    f"{10**min_log_pval:.1e}",
                    f"{pval:.2f}",
                    "n.s.",
                ],
                len=0.5,
                y=0.5,
            ),
        ),
        row=row,
        col=col,
    )

    return fig


def plot_residuals(
    ypred: np.ndarray,
    ytrue: np.ndarray,
    x: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    px_kwargs: dict = {
        "trendline": "lowess",
        "trendline_color_override": "rgba(0,0,0,0.5)",
        "facet_col_wrap": 4,
    },
) -> go.Figure:
    """Plot the residuals of a regression

    Parameters
    ----------
    ypred : np.ndarray
        the predicted values, n_samples x n_features

    y : np.ndarray
        the true values, n_samples x n_features

    x : np.ndarray | None
        if specified use for the x axis, else a range(len(y)) is used

    feature_names : list[str] | None
        if specified, use for naming the features, else x0, x1, ... are used

    Returns
    -------
    go.Figure
        the figure with the residuals plotted
    """

    res = ytrue - ypred
    if len(res.shape) == 1:
        res = res.reshape(-1, 1)

    xvals = x if x is not None else np.arange(len(ytrue))
    feature_names = feature_names or [f"x{i}" for i in range(res.shape[1])]

    df = pd.DataFrame(res, columns=[f"resid_{f}" for f in feature_names])  # type: ignore
    df["x"] = xvals
    dm = pd.melt(df, id_vars=["x"])

    fig = px.scatter(dm, x="x", y="value", facet_col="variable", **px_kwargs)

    return fig
