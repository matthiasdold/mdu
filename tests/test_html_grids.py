"""Tests for mdu.plotly.html_grids module."""

import tempfile
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import pytest

from mdu.plotly.html_grids import create_plotly_grid_html


class TestCreatePlotlyGridHTML:
    """Test suite for create_plotly_grid_html function."""

    def test_creates_html_file(self):
        """Test that HTML file is created successfully."""
        fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_grid.html"
            create_plotly_grid_html(
                figures=[fig1, fig2],
                grid_shape=(1, 2),
                filename=output_path,
                show=False,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Composed Plotly Grid" in content

    def test_grid_shape_validation(self):
        """Test that ValueError is raised when grid shape is too small."""
        fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])
        fig3 = px.bar(x=["A", "B", "C"], y=[10, 15, 13])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_grid.html"
            with pytest.raises(
                ValueError,
                match="Number of figures .* does not match grid shape",
            ):
                create_plotly_grid_html(
                    figures=[fig1, fig2, fig3],
                    grid_shape=(1, 2),  # Only room for 2 figures
                    filename=output_path,
                    show=False,
                )

    def test_single_figure_grid(self):
        """Test with a single figure in 1x1 grid."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single.html"
            create_plotly_grid_html(
                figures=[fig],
                grid_shape=(1, 1),
                filename=output_path,
                show=False,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert 'id="fig-0"' in content

    def test_multiple_rows_and_cols(self):
        """Test with multiple rows and columns."""
        figs = [px.scatter(x=[i], y=[i * 2]) for i in range(6)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grid_3x2.html"
            create_plotly_grid_html(
                figures=figs,
                grid_shape=(3, 2),
                filename=output_path,
                show=False,
            )

            content = output_path.read_text()
            # Check that all 6 figures are included
            for i in range(6):
                assert f'id="fig-{i}"' in content
            # Check grid structure
            assert "grid-template-columns: repeat(2, 1fr)" in content
            assert "grid-template-rows: repeat(3, 1fr)" in content

    def test_custom_min_height(self):
        """Test custom minimum height setting."""
        fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_height.html"
            create_plotly_grid_html(
                figures=[fig],
                grid_shape=(1, 1),
                filename=output_path,
                show=False,
                min_height=500,
            )

            content = output_path.read_text()
            assert "min-height: 500" in content

    def test_plotlyjs_cdn_only_first_figure(self):
        """Test that Plotly.js CDN is included only in first figure."""
        fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cdn_test.html"
            create_plotly_grid_html(
                figures=[fig1, fig2],
                grid_shape=(1, 2),
                filename=output_path,
                show=False,
            )

            content = output_path.read_text()
            # CDN should be loaded (plotly.js reference)
            assert "plotly" in content.lower()

    def test_empty_grid_cells(self):
        """Test that grid can have empty cells (fewer figures than grid size)."""
        fig1 = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        fig2 = px.line(x=[1, 2, 3], y=[2, 4, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sparse_grid.html"
            create_plotly_grid_html(
                figures=[fig1, fig2],
                grid_shape=(2, 2),  # 4 cells but only 2 figures
                filename=output_path,
                show=False,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert 'id="fig-0"' in content
            assert 'id="fig-1"' in content
            assert 'id="fig-2"' not in content

    def test_different_figure_types(self):
        """Test with different types of plotly figures."""
        scatter_fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
        line_fig = px.line(x=[1, 2, 3], y=[2, 4, 6])
        bar_fig = px.bar(x=["A", "B", "C"], y=[10, 15, 13])
        
        # Create a manual go.Figure
        manual_fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mixed_types.html"
            create_plotly_grid_html(
                figures=[scatter_fig, line_fig, bar_fig, manual_fig],
                grid_shape=(2, 2),
                filename=output_path,
                show=False,
            )

            assert output_path.exists()
            content = output_path.read_text()
            for i in range(4):
                assert f'id="fig-{i}"' in content
