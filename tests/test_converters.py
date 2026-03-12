"""Tests for mdu.utils.converters module."""

from datetime import datetime
import numpy as np
import pytest

from mdu.utils.converters import ToFloatConverter


class TestToFloatConverter:
    """Test suite for ToFloatConverter class."""

    def test_to_float_with_floats(self):
        """Test that float arrays pass through unchanged."""
        converter = ToFloatConverter()
        x = np.array([1.0, 2.0, 3.0])
        result = converter.to_float(x)

        np.testing.assert_array_equal(result, x)
        assert converter.back_conversion is None

    def test_to_float_with_ints(self):
        """Test that integer arrays pass through unchanged."""
        converter = ToFloatConverter()
        x = np.array([1, 2, 3])
        result = converter.to_float(x)

        np.testing.assert_array_equal(result, x)
        assert converter.back_conversion is None

    def test_to_float_with_datetime(self):
        """Test conversion of datetime array to float."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 12, 1, 0)
        dt3 = datetime(2024, 1, 1, 12, 2, 0)
        x = np.array([dt1, dt2, dt3])

        result = converter.to_float(x)

        # Should start at zero
        assert result[0] == 0.0
        # Should be 60 seconds apart
        assert result[1] == 60.0
        assert result[2] == 120.0
        # Should have back conversion set
        assert converter.back_conversion is not None

    def test_to_float_datetime_offset(self):
        """Test that datetime conversion offsets from first value."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 6, 15, 10, 30, 45)
        dt2 = datetime(2024, 6, 15, 10, 31, 45)
        x = np.array([dt1, dt2])

        result = converter.to_float(x)

        assert result[0] == 0.0
        assert result[1] == 60.0  # 1 minute difference

    def test_to_orig_with_datetime(self):
        """Test conversion back to datetime from float."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 12, 1, 0)
        x_orig = np.array([dt1, dt2])

        # Convert to float
        x_float = converter.to_float(x_orig)

        # Convert back
        x_back = converter.to_orig(x_float)

        assert len(x_back) == len(x_orig)
        assert x_back[0] == dt1
        assert x_back[1] == dt2

    def test_to_orig_roundtrip(self):
        """Test full roundtrip conversion preserves datetime values."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 3, 11, 14, 30, 0)
        dt2 = datetime(2024, 3, 11, 14, 35, 30)
        dt3 = datetime(2024, 3, 11, 14, 40, 15)
        x_orig = np.array([dt1, dt2, dt3])

        # Roundtrip
        x_float = converter.to_float(x_orig)
        x_back = converter.to_orig(x_float)

        for i in range(len(x_orig)):
            assert x_back[i] == x_orig[i]

    def test_to_float_raises_on_mixed_types(self):
        """Test that mixed datetime/non-datetime types are handled."""
        converter = ToFloatConverter()
        # Create array with datetime and float (can't be done directly in numpy)
        # Instead test with integers and datetime which would fail
        dt = datetime(2024, 1, 1)
        x = np.array([dt, dt], dtype=object)
        x[1] = 42  # Mix types
        
        # This should raise when trying to convert
        with pytest.raises((AssertionError, AttributeError, TypeError)):
            converter.to_float(x)

    def test_to_float_single_element(self):
        """Test with single element array."""
        converter = ToFloatConverter()
        x = np.array([42.0])
        result = converter.to_float(x)

        assert result[0] == 42.0

    def test_to_float_single_datetime(self):
        """Test with single datetime element."""
        converter = ToFloatConverter()
        dt = datetime(2024, 1, 1)
        x = np.array([dt])

        result = converter.to_float(x)

        assert result[0] == 0.0  # Offset to zero

    def test_from_timestamp_with_offset(self):
        """Test from_timestamp_with_offset method directly."""
        converter = ToFloatConverter()
        
        # Create a reference datetime
        ref_dt = datetime(2024, 1, 1, 12, 0, 0)
        offset = ref_dt.timestamp()
        
        # 60 seconds after reference
        result = converter.from_timestamp_with_offset(60.0, offset)
        
        expected = datetime(2024, 1, 1, 12, 1, 0)
        assert result == expected

    def test_to_float_preserves_float_precision(self):
        """Test that float values maintain precision."""
        converter = ToFloatConverter()
        x = np.array([1.123456789, 2.987654321])
        result = converter.to_float(x)

        np.testing.assert_array_almost_equal(result, x)

    def test_to_float_large_datetime_range(self):
        """Test with large datetime range."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 12, 31)
        x = np.array([dt1, dt2])

        result = converter.to_float(x)

        assert result[0] == 0.0
        # Should be many days worth of seconds
        assert result[1] > 30_000_000  # ~364 days in seconds

    def test_to_float_negative_numbers(self):
        """Test with negative numbers."""
        converter = ToFloatConverter()
        x = np.array([-1.0, -2.0, -3.0])
        result = converter.to_float(x)

        np.testing.assert_array_equal(result, x)

    def test_to_float_zero_values(self):
        """Test with zero values."""
        converter = ToFloatConverter()
        x = np.array([0.0, 0.0, 0.0])
        result = converter.to_float(x)

        np.testing.assert_array_equal(result, x)

    def test_back_conversion_initially_none(self):
        """Test that back_conversion is None before datetime conversion."""
        converter = ToFloatConverter()
        assert converter.back_conversion is None

    def test_back_conversion_set_after_datetime(self):
        """Test that back_conversion is set after datetime conversion."""
        converter = ToFloatConverter()
        x = np.array([datetime(2024, 1, 1)])
        converter.to_float(x)

        assert converter.back_conversion is not None
        assert callable(converter.back_conversion)

    def test_converter_reusability(self):
        """Test that converter can be reused for multiple conversions."""
        converter = ToFloatConverter()
        
        # First conversion
        x1 = np.array([datetime(2024, 1, 1), datetime(2024, 1, 2)])
        result1 = converter.to_float(x1)
        
        # Second conversion (will overwrite back_conversion)
        x2 = np.array([datetime(2024, 6, 1), datetime(2024, 6, 2)])
        result2 = converter.to_float(x2)
        
        # Both should work
        assert result1[0] == 0.0
        assert result2[0] == 0.0

    def test_to_orig_with_modified_floats(self):
        """Test converting back with modified float values."""
        converter = ToFloatConverter()
        dt1 = datetime(2024, 1, 1, 12, 0, 0)
        dt2 = datetime(2024, 1, 1, 12, 1, 0)
        x_orig = np.array([dt1, dt2])

        # Convert to float
        x_float = converter.to_float(x_orig)
        
        # Modify the float values (add 30 seconds)
        x_modified = x_float + 30.0
        
        # Convert back
        x_back = converter.to_orig(x_modified)
        
        # Should be 30 seconds later than original
        expected1 = datetime(2024, 1, 1, 12, 0, 30)
        expected2 = datetime(2024, 1, 1, 12, 1, 30)
        assert x_back[0] == expected1
        assert x_back[1] == expected2
