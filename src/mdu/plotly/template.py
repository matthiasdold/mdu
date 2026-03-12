import plotly.graph_objects as go
import plotly.io as pio


def set_template():
    """Set custom Plotly template with default styling for box and violin plots.

    Configures and registers a custom Plotly template named 'md' with predefined
    styling for box plots, violin plots, and axis appearance. Sets this template
    as the default by combining it with the 'plotly_white' theme.

    Notes
    -----
    This function modifies the global Plotly template configuration. After calling
    this function, all subsequent plots will use the custom styling by default.

    The template includes:
    - Box plots: Semi-transparent with visible points, mean line, 1px borders
    - Violin plots: 2px lines, visible mean line, red mean indicator, grey box
    - Axes: Visible axis lines in grey (#aaa), outside ticks
    - Margins: Minimal margins with 40px top margin for titles

    The template is registered as 'md' and combined with 'plotly_white' to create
    'plotly_white+md' which becomes the new default template.

    Examples
    --------
    >>> import plotly.express as px
    >>> from mdu.plotly.template import set_template
    >>> # Set custom template
    >>> set_template()
    >>> # All subsequent plots use the custom styling
    >>> fig = px.box(x=['A', 'B', 'C'], y=[1, 2, 3])
    >>> # To revert to default Plotly theme:
    >>> import plotly.io as pio
    >>> pio.templates.default = 'plotly'
    """
    tmpl = go.layout.Template()
    tmpl.data.box = [  # type: ignore
        go.Box(
            line=dict(width=1),
            opacity=0.6,
            boxpoints="all",
            boxmean=True,
            pointpos=0,
        ),
    ]
    tmpl.data.violin = [  # type: ignore
        go.Violin(
            line_width=2,
            meanline_visible=True,
            offsetgroup="single",
            box=dict(
                visible=True,
                fillcolor="#ddd",
                line_color="#333",
            ),
            meanline_color="#f33",  # applies to box, but needs to be defined outside
            meanline_width=2,
        ),
    ]
    tmpl.layout = go.Layout(
        xaxis=dict(showline=True, linecolor="#aaa", ticks="outside"),
        yaxis=dict(showline=True, linecolor="#aaa", ticks="outside"),
        margin=dict(l=0, r=0, t=40, b=0),  # keep top 20 for title
    )
    pio.templates["md"] = tmpl
    pio.templates.default = "plotly_white+md"
