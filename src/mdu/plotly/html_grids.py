import webbrowser
from pathlib import Path

import plotly.graph_objects as go

from mdu.utils.logging import get_logger

logger = get_logger("mdu.plotly.html_grids")


def create_plotly_grid_html(
    figures: list[go.Figure],
    grid_shape: tuple[int, int],
    filename: Path = Path("plotly_grid.html"),
    show: bool = False,
    min_height: int = 200,
):
    """Compose multiple Plotly figures into a single HTML file with grid layout.

    Creates a standalone HTML file containing multiple Plotly figures arranged
    in a CSS grid layout. The first figure includes the Plotly.js library via
    CDN, while subsequent figures reuse it for efficiency.

    Parameters
    ----------
    figures : list of plotly.graph_objects.Figure
        List of Plotly figure objects to display in the grid.
    grid_shape : tuple of (int, int)
        Grid dimensions as (rows, cols). The product must be >= len(figures).
    filename : Path, default=Path("plotly_grid.html")
        Output file path for the HTML file.
    show : bool, default=False
        If True, automatically opens the generated HTML file in a new browser tab.
    min_height : int, default=200
        Minimum height in pixels for each chart in the grid.

    Raises
    ------
    ValueError
        If the number of figures exceeds rows * cols.

    Examples
    --------
    >>> import plotly.express as px
    >>> from pathlib import Path
    >>> # Create sample figures
    >>> fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
    >>> fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])
    >>> fig3 = px.bar(x=['A', 'B', 'C'], y=[10, 15, 13])
    >>> # Create 2x2 grid
    >>> create_plotly_grid_html(
    ...     figures=[fig1, fig2, fig3],
    ...     grid_shape=(2, 2),
    ...     filename=Path("my_grid.html"),
    ...     show=True
    ... )

    Notes
    -----
    The HTML file is self-contained except for the Plotly.js library which is
    loaded from CDN. For offline use, consider downloading Plotly.js locally.
    """
    rows, cols = grid_shape
    if len(figures) > rows * cols:
        raise ValueError(
            f"Number of figures ({len(figures)}) does not match grid shape ({rows}x{cols})."
        )

    divs_html = ""
    for i, fig in enumerate(figures):
        include_plotlyjs = "cdn" if i == 0 else False
        divs_html += f'    <div id="fig-{i}" class="plotly-chart">{fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)}</div>\n'

    html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Composed Plotly Grid</title>
            <!-- Use Plotly.js from a CDN -->
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                /* Basic styling for the grid */
                body {{
                    margin: 0;
                    background-color: #f8f8f8;
                    font-family: sans-serif;
                }}
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat({cols}, 1fr);
                    grid-template-rows: repeat({rows}, 1fr);
                    gap: 15px; /* Space between charts */
                    padding: 15px;
                    box-sizing: border-box;
                }}
                .plotly-chart {{
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    background-color: white;
                    min-height: {min_height}; /* Ensure charts have a minimum height */
                }}
            </style>
        </head>
        <body>
            <div class="grid-container">
        {divs_html}
            </div>
        </body>
        </html>
    """

    # 5. Write the HTML to a file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info(f"Successfully generated {filename}")

    if show:
        webbrowser.open_new_tab(f"file://{filename.resolve()}")


def create_tabbed_plotly_grid_html(
    tabs_data: list[dict],
    filename: Path = Path("plotly_tabbed_grid.html"),
    show: bool = False,
    min_height: int = 200,
):
    """Create multi-tab HTML dashboard with custom grid layout per tab.

    Composes multiple Plotly figures into a tabbed HTML file where each tab
    contains its own grid of figures. Supports custom grid dimensions per tab
    and automatic Plotly.js library sharing for efficiency.

    Parameters
    ----------
    tabs_data : list of dict
        List of dictionaries defining each tab. Each dict must contain:
        - 'title' (str): Tab display name
        - 'figs' (list of go.Figure): Plotly figures for this tab
        - 'grid_dims' (tuple of (int, int)): Grid shape as (rows, cols)
    filename : Path, default=Path("plotly_tabbed_grid.html")
        Output file path for the HTML file.
    show : bool, default=False
        If True, automatically opens the generated HTML file in a new browser tab.
    min_height : int, default=200
        Minimum height in pixels for each chart container in the grid.

    Raises
    ------
    ValueError
        If the number of figures in any tab exceeds its grid dimensions (rows * cols).

    Examples
    --------
    >>> import plotly.express as px
    >>> from pathlib import Path
    >>> # Create figures for different tabs
    >>> fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
    >>> fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])
    >>> fig3 = px.bar(x=['A', 'B', 'C'], y=[10, 15, 13])
    >>> fig4 = px.box(x=['X', 'Y'], y=[5, 7])
    >>> # Define tabs with different grid layouts
    >>> tabs = [
    ...     {
    ...         'title': 'Overview',
    ...         'figs': [fig1, fig2],
    ...         'grid_dims': (1, 2)  # 1 row, 2 columns
    ...     },
    ...     {
    ...         'title': 'Details',
    ...         'figs': [fig3, fig4],
    ...         'grid_dims': (2, 1)  # 2 rows, 1 column
    ...     }
    ... ]
    >>> create_tabbed_plotly_grid_html(
    ...     tabs_data=tabs,
    ...     filename=Path("dashboard.html"),
    ...     show=True
    ... )

    Notes
    -----
    - The HTML file is self-contained except for Plotly.js loaded from CDN
    - Plotly.js is only included once (in the first figure) and reused
    - Tab switching triggers Plotly.Plots.resize() to ensure proper rendering
    - For offline use, consider downloading Plotly.js locally
    """
    tab_buttons_html = ""
    tab_content_html = ""

    # We only want to inject the Plotly JS library once for the very first plot
    is_first_plot = True

    for tab_idx, tab in enumerate(tabs_data):
        title = tab["title"]
        figures = tab["figs"]
        rows, cols = tab["grid_dims"]

        if len(figures) > rows * cols:
            raise ValueError(
                f"Number of figures ({len(figures)}) in tab '{title}' "
                f"does not match grid shape ({rows}x{cols})."
            )

        # 1. Generate HTML for the Tab Buttons
        active_id = ' id="defaultOpen"' if tab_idx == 0 else ""
        tab_buttons_html += f'        <button class="tablinks" onclick="openTab(event, \'tab-{tab_idx}\')"{active_id}>{title}</button>\n'

        # 2. Generate HTML for the Grid Content of this Tab
        tab_content_html += f'    <div id="tab-{tab_idx}" class="tabcontent">\n'
        tab_content_html += f'        <div class="grid-container" style="grid-template-columns: repeat({cols}, 1fr); grid-template-rows: repeat({rows}, 1fr);">\n'

        for fig_idx, fig in enumerate(figures):
            # Only inject CDN for the first graph across all tabs
            include_js = "cdn" if is_first_plot else False
            is_first_plot = False

            # config={'responsive': True} ensures charts fluidly fill grid areas
            fig_html = fig.to_html(
                include_plotlyjs=include_js,
                full_html=False,
                config={"responsive": True},
            )
            tab_content_html += (
                f'            <div class="plotly-chart">{fig_html}</div>\n'
            )

        tab_content_html += "        </div>\n"
        tab_content_html += "    </div>\n"

    # 3. Assemble the final HTML
    html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Composed Plotly Tabbed Grid</title>
            <style>
                body {{
                    margin: 0;
                    background-color: #f8f8f8;
                    font-family: sans-serif;
                }}
                /* Tab styling */
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                    font-size: 17px;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #ccc;
                }}
                .tabcontent {{
                    display: none;
                    padding: 15px;
                    border-top: none;
                }}
                /* Grid styling */
                .grid-container {{
                    display: grid;
                    gap: 15px; /* Space between charts */
                    box-sizing: border-box;
                }}
                .plotly-chart {{
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    background-color: white;
                    min-height: {min_height}px; 
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <!-- Tab Links -->
            <div class="tab">
{tab_buttons_html}
            </div>

            <!-- Tab Content -->
{tab_content_html}

            <script>
            function openTab(evt, tabName) {{
                // Hide all tab contents
                var tabcontent = document.getElementsByClassName("tabcontent");
                for (var i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                
                // Remove the "active" class from all buttons
                var tablinks = document.getElementsByClassName("tablinks");
                for (var i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                
                // Show the current tab, and add an "active" class to the button
                var currentTab = document.getElementById(tabName);
                currentTab.style.display = "block";
                if (evt) {{
                    evt.currentTarget.className += " active";
                }}
                
                // CRITICAL PLOTLY FIX:
                // Plotly cannot calculate dimensions when rendering inside display:none
                // We must force Plotly to recalculate and draw data now that it is visible.
                // Wrapped in a 50ms timeout to ensure the DOM has fully rendered the 'block' change.
                setTimeout(function() {{
                    var plots = currentTab.getElementsByClassName("plotly-graph-div");
                    for (var i = 0; i < plots.length; i++) {{
                        if (typeof Plotly !== 'undefined') {{
                            Plotly.Plots.resize(plots[i]);
                        }}
                    }}
                }}, 50);
            }}

            // Automatically open the first tab on load
            document.getElementById("defaultOpen").click();
            </script>
        </body>
        </html>
    """

    # 4. Write the HTML to a file
    filename = Path(filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info(f"Successfully generated {filename}")

    if show:
        webbrowser.open_new_tab(f"file://{filename.resolve()}")
