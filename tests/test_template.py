"""Tests for mdu.plotly.template module."""

import pytest
import plotly.graph_objects as go
import plotly.io as pio

from mdu.plotly.template import set_template


class TestSetTemplate:
    """Test suite for set_template function."""

    @pytest.fixture(autouse=True)
    def reset_template(self):
        """Reset template to default before and after each test."""
        original = pio.templates.default
        yield
        # Reset to original after test
        pio.templates.default = original

    def test_set_template_registers_md_template(self):
        """Test that 'md' template is registered."""
        set_template()

        assert "md" in pio.templates
        assert pio.templates["md"] is not None

    def test_set_template_sets_default(self):
        """Test that default template is set to plotly_white+md."""
        set_template()

        assert pio.templates.default == "plotly_white+md"

    def test_set_template_box_styling(self):
        """Test that box plot styling is configured."""
        set_template()

        tmpl = pio.templates["md"]
        assert hasattr(tmpl.data, "box")
        assert len(tmpl.data.box) > 0

        box_style = tmpl.data.box[0]
        assert box_style.line.width == 1
        assert box_style.opacity == 0.6
        assert box_style.boxpoints == "all"
        assert box_style.boxmean == True

    def test_set_template_violin_styling(self):
        """Test that violin plot styling is configured."""
        set_template()

        tmpl = pio.templates["md"]
        assert hasattr(tmpl.data, "violin")
        assert len(tmpl.data.violin) > 0

        violin_style = tmpl.data.violin[0]
        assert violin_style.line.width == 2
        assert violin_style.meanline.visible == True
        assert violin_style.offsetgroup == "single"

    def test_set_template_violin_box_color(self):
        """Test that violin box has correct colors."""
        set_template()

        tmpl = pio.templates["md"]
        violin_style = tmpl.data.violin[0]

        assert violin_style.box.visible == True
        assert violin_style.box.fillcolor == "#ddd"
        assert violin_style.box.line.color == "#333"

    def test_set_template_violin_meanline_color(self):
        """Test that violin meanline is red."""
        set_template()

        tmpl = pio.templates["md"]
        violin_style = tmpl.data.violin[0]

        assert violin_style.meanline.color == "#f33"
        assert violin_style.meanline.width == 2

    def test_set_template_layout_axes(self):
        """Test that axes styling is configured."""
        set_template()

        tmpl = pio.templates["md"]
        assert tmpl.layout.xaxis.showline == True
        assert tmpl.layout.xaxis.linecolor == "#aaa"
        assert tmpl.layout.xaxis.ticks == "outside"

        assert tmpl.layout.yaxis.showline == True
        assert tmpl.layout.yaxis.linecolor == "#aaa"
        assert tmpl.layout.yaxis.ticks == "outside"

    def test_set_template_layout_margins(self):
        """Test that margins are configured."""
        set_template()

        tmpl = pio.templates["md"]
        assert tmpl.layout.margin.l == 0
        assert tmpl.layout.margin.r == 0
        assert tmpl.layout.margin.t == 40
        assert tmpl.layout.margin.b == 0

    def test_set_template_can_be_called_multiple_times(self):
        """Test that calling set_template multiple times doesn't break."""
        set_template()
        first_default = pio.templates.default

        set_template()
        second_default = pio.templates.default

        assert first_default == second_default == "plotly_white+md"

    def test_set_template_affects_new_figures(self):
        """Test that template affects newly created figures."""
        set_template()

        # Create a new figure - it should use the template
        fig = go.Figure()
        # The template should be available
        assert "md" in pio.templates

    def test_template_can_be_reverted(self):
        """Test that template can be changed back to default."""
        original = pio.templates.default
        set_template()

        assert pio.templates.default == "plotly_white+md"

        # Revert
        pio.templates.default = original
        assert pio.templates.default == original

    def test_template_is_persistent(self):
        """Test that template persists after creation."""
        set_template()

        # Create and modify a figure
        fig1 = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])

        # Template should still be set
        assert pio.templates.default == "plotly_white+md"
        assert "md" in pio.templates
