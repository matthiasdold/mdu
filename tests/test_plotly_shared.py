import numpy as np
import pytest
import plotly.graph_objects as go
import plotly.express as px

from mdu.plotly.shared import (
    extract_subplot_coordinates,
    hex_to_rgba,
    format_float_to_text_with_suffix,
)


class TestExtractSubplotCoordinates:
    """Test suite for extract_subplot_coordinates function."""

    def test_extract_single_plot(self):
        """Test extraction with a single plot (no subplots)."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        # Single plots don't have _grid_str attribute set to a string
        # This should raise TypeError when regex tries to match on None
        with pytest.raises(TypeError):
            extract_subplot_coordinates(fig)

    def test_extract_faceted_plot(self):
        """Test extraction with faceted plot."""
        df = px.data.tips()
        fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex")

        result = extract_subplot_coordinates(fig)

        # Should return list of tuples
        assert isinstance(result, list)
        assert len(result) > 0
        # Each element should be a tuple with 4 strings
        for item in result:
            assert len(item) == 4
            assert all(isinstance(s, str) for s in item)

    def test_extract_faceted_row_col_plot(self):
        """Test extraction with both row and column facets."""
        df = px.data.tips()
        fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex", facet_row="time")

        result = extract_subplot_coordinates(fig)

        # Should have multiple subplots
        assert isinstance(result, list)
        assert len(result) > 1


class TestHexToRgba:
    """Test suite for hex_to_rgba function."""

    def test_hex_to_rgba_full_hex(self):
        """Test conversion of full hex color."""
        result = hex_to_rgba("#ff0000", opacity=1.0)
        assert result == "rgba(255, 0, 0, 1.0)"

    def test_hex_to_rgba_short_hex(self):
        """Test conversion of shorthand hex color."""
        result = hex_to_rgba("#ddd", opacity=0.7)
        assert result == "rgba(221, 221, 221, 0.7)"

    def test_hex_to_rgba_no_hash(self):
        """Test conversion without hash symbol."""
        result = hex_to_rgba("888888", opacity=0.5)
        assert result == "rgba(136, 136, 136, 0.5)"

    def test_hex_to_rgba_default_opacity(self):
        """Test conversion with default opacity."""
        result = hex_to_rgba("#00ff00")
        assert result == "rgba(0, 255, 0, 1.0)"

    def test_hex_to_rgba_zero_opacity(self):
        """Test conversion with zero opacity."""
        result = hex_to_rgba("#0000ff", opacity=0.0)
        assert result == "rgba(0, 0, 255, 0.0)"

    def test_hex_to_rgba_white(self):
        """Test conversion of white color."""
        result = hex_to_rgba("#ffffff", opacity=0.8)
        assert result == "rgba(255, 255, 255, 0.8)"

    def test_hex_to_rgba_black(self):
        """Test conversion of black color."""
        result = hex_to_rgba("#000000", opacity=0.3)
        assert result == "rgba(0, 0, 0, 0.3)"

    def test_hex_to_rgba_short_white(self):
        """Test conversion of shorthand white."""
        result = hex_to_rgba("#fff", opacity=0.9)
        assert result == "rgba(255, 255, 255, 0.9)"

    def test_hex_to_rgba_short_black(self):
        """Test conversion of shorthand black."""
        result = hex_to_rgba("#000", opacity=0.5)
        assert result == "rgba(0, 0, 0, 0.5)"

    def test_hex_to_rgba_various_colors(self):
        """Test conversion of various colors."""
        test_cases = [
            ("#ff5733", 0.5, "rgba(255, 87, 51, 0.5)"),
            ("#c0c0c0", 0.7, "rgba(192, 192, 192, 0.7)"),
            ("#123456", 1.0, "rgba(18, 52, 86, 1.0)"),
        ]

        for hex_color, opacity, expected in test_cases:
            result = hex_to_rgba(hex_color, opacity)
            assert result == expected


class TestFormatFloatToTextWithSuffix:
    """Test suite for format_float_to_text_with_suffix function."""

    def test_format_millions(self):
        """Test formatting of million-scale numbers (>= 1e7)."""
        assert format_float_to_text_with_suffix(10_000_000) == "10Mio"
        assert format_float_to_text_with_suffix(12_345_678) == "12.35Mio"
        assert format_float_to_text_with_suffix(100_000_000) == "100Mio"

    def test_format_thousands(self):
        """Test formatting of thousand-scale numbers (>= 1e4)."""
        assert format_float_to_text_with_suffix(10_000) == "10k"
        assert format_float_to_text_with_suffix(12_345) == "12.35k"
        assert format_float_to_text_with_suffix(99_999) == "100k"

    def test_format_base_unit(self):
        """Test formatting of base unit numbers (1 to < 1e4)."""
        assert format_float_to_text_with_suffix(1) == "1"
        assert format_float_to_text_with_suffix(123.45) == "123.45"
        assert format_float_to_text_with_suffix(9999) == "9999"
        assert format_float_to_text_with_suffix(5000) == "5000"

    def test_format_milli(self):
        """Test formatting of milli-scale numbers (>= 1e-4)."""
        assert format_float_to_text_with_suffix(0.01234) == "12.34m"
        assert format_float_to_text_with_suffix(0.001) == "1m"
        assert format_float_to_text_with_suffix(0.0999) == "99.9m"
        assert format_float_to_text_with_suffix(0.0001) == "0.1m"

    def test_format_micro(self):
        """Test formatting of micro-scale numbers (1e-7 <= x < 1e-4)."""
        assert format_float_to_text_with_suffix(1e-6) == "1µ"
        assert format_float_to_text_with_suffix(5.67e-6) == "5.67µ"
        assert format_float_to_text_with_suffix(1e-7) == "0.1µ"

    def test_format_nano(self):
        """Test formatting of nano-scale numbers (< 1e-7)."""
        assert format_float_to_text_with_suffix(1e-9) == "1n"
        assert format_float_to_text_with_suffix(5.5e-8) == "55n"
        assert format_float_to_text_with_suffix(1e-10) == "0.1n"

    def test_format_zero(self):
        """Test formatting of zero."""
        assert format_float_to_text_with_suffix(0) == "0"
        assert format_float_to_text_with_suffix(0.0) == "0"

    def test_format_negative(self):
        """Test formatting of negative numbers."""
        assert format_float_to_text_with_suffix(-10_000) == "-10k"
        assert format_float_to_text_with_suffix(-12.3e6) == "-12.3Mio"
        assert format_float_to_text_with_suffix(-0.001) == "-1m"

    def test_format_infinity(self):
        """Test formatting of infinity."""
        result = format_float_to_text_with_suffix(float("inf"))
        assert result == "inf"

        result = format_float_to_text_with_suffix(float("-inf"))
        assert result == "-inf"

    def test_format_nan(self):
        """Test formatting of NaN."""
        result = format_float_to_text_with_suffix(float("nan"))
        assert result == "nan"

    def test_format_rounding(self):
        """Test that rounding works correctly."""
        # Should round to 2 decimal places
        assert format_float_to_text_with_suffix(1234.5678) == "1234.57"
        assert format_float_to_text_with_suffix(12_345_678) == "12.35Mio"

    def test_format_trailing_zeros_removed(self):
        """Test that trailing zeros are removed."""
        assert format_float_to_text_with_suffix(10_000) == "10k"
        assert format_float_to_text_with_suffix(10_000_000) == "10Mio"

    def test_format_type_error(self):
        """Test that non-numeric input raises TypeError."""
        with pytest.raises(TypeError):
            format_float_to_text_with_suffix("not a number")

        with pytest.raises(TypeError):
            format_float_to_text_with_suffix([1, 2, 3])

    def test_format_boundary_values(self):
        """Test boundary values between different scales."""
        # Boundary between base and k (1e4)
        assert format_float_to_text_with_suffix(9_999) == "9999"
        assert format_float_to_text_with_suffix(10_000) == "10k"

        # Boundary between k and Mio (1e7)
        assert format_float_to_text_with_suffix(9_999_999) == "10000k"
        assert format_float_to_text_with_suffix(10_000_000) == "10Mio"

        # Boundary between base and m (1)
        assert format_float_to_text_with_suffix(1) == "1"
        assert format_float_to_text_with_suffix(0.999) == "999m"

        # Boundary between m and µ (1e-4)
        assert format_float_to_text_with_suffix(0.0001) == "0.1m"
        # 0.00009 is 9e-5 which is >= 1e-4, so uses m suffix: 9e-5 / 1e-3 = 0.09m
        # But it's actually < 1e-4, so it should use µ suffix: 9e-5 / 1e-6 = 90µ
        assert format_float_to_text_with_suffix(0.00009) == "90µ"

    def test_format_scientific_notation_input(self):
        """Test with scientific notation input."""
        assert format_float_to_text_with_suffix(1.5e7) == "15Mio"
        assert format_float_to_text_with_suffix(3.2e-6) == "3.2µ"
        assert format_float_to_text_with_suffix(7.8e4) == "78k"

    def test_format_very_small_numbers(self):
        """Test very small numbers (smaller than nano scale threshold)."""
        # Numbers smaller than 1e-7 use nano
        result = format_float_to_text_with_suffix(1e-12)
        assert "n" in result

    def test_format_integer_like_floats(self):
        """Test that integer-like floats are formatted as integers."""
        assert format_float_to_text_with_suffix(1000.0) == "1000"
        assert format_float_to_text_with_suffix(10_000.0) == "10k"
        assert format_float_to_text_with_suffix(1e7) == "10Mio"


class TestCreatePlotlyGridHtml:
    """Test suite for create_plotly_grid_html function."""
    
    def test_create_grid_html_basic(self, tmp_path):
        """Test creating a basic HTML grid."""
        from mdu.plotly.html_grids import create_plotly_grid_html
        from pathlib import Path
        
        fig1 = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        fig2 = go.Figure(data=go.Scatter(x=[5, 6], y=[7, 8]))
        
        output_file = tmp_path / "test_grid.html"
        create_plotly_grid_html([fig1, fig2], (1, 2), filename=output_file)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "plotly" in content.lower()
        assert "grid" in content.lower()
    
    def test_create_grid_html_too_many_figures(self):
        """Test that too many figures raises ValueError."""
        from mdu.plotly.html_grids import create_plotly_grid_html
        from pathlib import Path
        
        figs = [go.Figure() for _ in range(5)]
        
        with pytest.raises(ValueError, match="does not match grid shape"):
            create_plotly_grid_html(figs, (1, 2), filename=Path("test.html"))
    
    def test_create_grid_html_contains_all_figures(self, tmp_path):
        """Test that all figures are included in output."""
        from mdu.plotly.html_grids import create_plotly_grid_html
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[1], y=[1], name="fig1"))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[2], y=[2], name="fig2"))
        
        output_file = tmp_path / "test.html"
        create_plotly_grid_html([fig1, fig2], (2, 1), filename=output_file)
        
        content = output_file.read_text()
        assert "fig1" in content
        assert "fig2" in content


class TestAddJitter:
    """Test suite for add_jitter function."""
    
    def test_add_jitter_adds_column(self):
        """Test that jitter column is added."""
        import polars as pl
        from mdu.plotly.shared import add_jitter
        
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = add_jitter(df, ycol="value")
        
        assert "jitter" in result.columns
    
    def test_add_jitter_preserves_rows(self):
        """Test that number of rows is preserved."""
        import polars as pl
        from mdu.plotly.shared import add_jitter
        
        df = pl.DataFrame({"score": [10.0, 20.0, 30.0]})
        result = add_jitter(df, ycol="score")
        
        assert len(result) == len(df)
    
    def test_add_jitter_range(self):
        """Test that jitter values are within expected range."""
        import polars as pl
        from mdu.plotly.shared import add_jitter
        
        df = pl.DataFrame({"val": [1.0] * 100})
        result = add_jitter(df, ycol="val", jitter_max_width=0.1)
        
        jitter_vals = result["jitter"].to_numpy()
        assert np.all(jitter_vals >= -0.1)
        assert np.all(jitter_vals <= 0.1)
    
    def test_add_jitter_temp_columns_removed(self):
        """Test that temporary columns are removed."""
        import polars as pl
        from mdu.plotly.shared import add_jitter
        
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = add_jitter(df, ycol="x")
        
        assert "bin" not in result.columns
        assert "scale_factor" not in result.columns
    
    def test_add_jitter_raises_on_existing_bin(self):
        """Test that existing 'bin' column raises assertion."""
        import polars as pl
        from mdu.plotly.shared import add_jitter
        
        df = pl.DataFrame({"val": [1.0, 2.0], "bin": ["a", "b"]})
        
        with pytest.raises(AssertionError, match="bin column already exists"):
            add_jitter(df, ycol="val")
