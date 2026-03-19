"""Tests for mdu.plotly.styling module."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdu.plotly.styling import apply_default_styles, get_dareplane_colors


class TestApplyDefaultStyles:
    """Test suite for apply_default_styles function."""

    def test_basic_styling_applied(self):
        """Test that basic styling is applied to figure."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        styled_fig = apply_default_styles(fig)

        assert styled_fig.layout.plot_bgcolor == "#ffffff"
        assert styled_fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert styled_fig.layout.font.size == 20

    def test_grid_options_default(self):
        """Test default grid options are applied."""
        fig = px.line(x=[1, 2, 3], y=[2, 4, 6])
        styled_fig = apply_default_styles(fig)

        # Check that grid styling was applied (xaxis and yaxis should have gridcolor)
        assert styled_fig.layout.xaxis.gridcolor == "#444444"
        assert styled_fig.layout.yaxis.gridcolor == "#444444"

    def test_disable_grid(self):
        """Test disabling grid lines."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        styled_fig = apply_default_styles(fig, showgrid=False, xgrid=False, ygrid=False)

        # Grid should be disabled
        assert styled_fig.layout.xaxis.showgrid is False
        assert styled_fig.layout.yaxis.showgrid is False

    def test_custom_grid_options(self):
        """Test custom grid options."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        custom_grid = {"gridcolor": "#FF0000", "gridwidth": 2, "griddash": "solid"}
        styled_fig = apply_default_styles(fig, gridoptions=custom_grid)

        assert styled_fig.layout.xaxis.gridcolor == "#FF0000"
        assert styled_fig.layout.xaxis.gridwidth == 2
        assert styled_fig.layout.xaxis.griddash == "solid"

    def test_separate_x_y_grid_options(self):
        """Test separate grid options for x and y axes."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        x_grid = {"gridcolor": "#FF0000", "gridwidth": 1}
        y_grid = {"gridcolor": "#0000FF", "gridwidth": 2}
        styled_fig = apply_default_styles(
            fig, gridoptions_x=x_grid, gridoptions_y=y_grid, gridoptions=None
        )

        assert styled_fig.layout.xaxis.gridcolor == "#FF0000"
        assert styled_fig.layout.yaxis.gridcolor == "#0000FF"
        assert styled_fig.layout.xaxis.gridwidth == 1
        assert styled_fig.layout.yaxis.gridwidth == 2

    def test_zero_lines(self):
        """Test zero line configuration."""
        fig = px.scatter(x=[-1, 0, 1], y=[-2, 0, 2])
        
        # With zero lines
        styled_fig = apply_default_styles(fig, xzero=True, yzero=True)
        assert styled_fig.layout.xaxis.zerolinecolor == "#444444"
        assert styled_fig.layout.yaxis.zerolinecolor == "#444444"

    def test_subplot_specific_styling(self):
        """Test applying styles to specific subplot."""
        fig = make_subplots(rows=2, cols=2)
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6], row=1, col=1)
        fig.add_scatter(x=[1, 2, 3], y=[2, 4, 6], row=2, col=2)

        # Apply styling to specific subplot
        styled_fig = apply_default_styles(fig, row=1, col=1)

        # Check that the styling was applied (general layout should be styled)
        assert styled_fig.layout.plot_bgcolor == "#ffffff"

    def test_margins_set_correctly(self):
        """Test that margins are set correctly."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        styled_fig = apply_default_styles(fig)

        assert styled_fig.layout.margin.l == 40
        assert styled_fig.layout.margin.r == 5
        assert styled_fig.layout.margin.t == 40
        assert styled_fig.layout.margin.b == 40

    def test_title_centered(self):
        """Test that title is centered."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Test Title")
        styled_fig = apply_default_styles(fig)

        assert styled_fig.layout.title.x == 0.5
        assert styled_fig.layout.title.xanchor == "center"

    def test_returns_figure_object(self):
        """Test that function returns a Figure object."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        result = apply_default_styles(fig)

        assert isinstance(result, go.Figure)

    def test_hoverlabel_font_size(self):
        """Test that hoverlabel font size is set."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        styled_fig = apply_default_styles(fig)

        assert styled_fig.layout.hoverlabel.font.size == 16


class TestGetDareplaneColors:
    """Test suite for get_dareplane_colors function."""

    def test_returns_list_of_colors(self):
        """Test that function returns a list of color strings."""
        colors = get_dareplane_colors()
        
        assert isinstance(colors, list)
        assert len(colors) == 5
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_color_values(self):
        """Test that specific color values are returned."""
        colors = get_dareplane_colors()
        
        expected = [
            "#0868acff",
            "#43a2caff",
            "#7bccc4ff",
            "#bae4bcff",
            "#f0f9e8ff",
        ]
        
        assert colors == expected

    def test_color_format(self):
        """Test that colors are in valid hex format."""
        colors = get_dareplane_colors()
        
        for color in colors:
            # Check format: # followed by 8 hex characters (with alpha)
            assert len(color) == 9
            assert color[0] == "#"
            # Check that all characters after # are valid hex
            hex_chars = color[1:]
            assert all(c in "0123456789abcdefABCDEF" for c in hex_chars)
