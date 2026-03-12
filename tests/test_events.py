"""Tests for mdu.mne.events module."""

import numpy as np
import pytest

from mdu.mne.events import inverse_map_events


class TestInverseMapEvents:
    """Test suite for inverse_map_events function."""

    def test_inverse_map_events_simple_numeric(self):
        """Test inverse mapping with simple numeric event IDs."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
            [300, 0, 1],
        ])
        evid = {"10": 1, "20": 2}

        result = inverse_map_events(events, evid)

        expected = np.array([
            [100, 0, 10],
            [200, 0, 20],
            [300, 0, 10],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_with_string_values(self):
        """Test inverse mapping when evid has non-numeric values."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
        ])
        evid = {"stimulus_1": 1, "stimulus_2": 2}

        result = inverse_map_events(events, evid)

        # Should extract number from key
        expected = np.array([
            [100, 0, 1],
            [200, 0, 2],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_extracts_numbers_from_keys(self):
        """Test that numbers are extracted from string keys."""
        events = np.array([
            [100, 0, 10],
            [200, 0, 20],
        ])
        evid = {"event_42": 10, "event_99": 20}

        result = inverse_map_events(events, evid)

        expected = np.array([
            [100, 0, 42],
            [200, 0, 99],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_preserves_unmapped(self):
        """Test that unmapped events are preserved."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
            [300, 0, 999],  # Not in mapping
        ])
        evid = {"10": 1, "20": 2}

        result = inverse_map_events(events, evid)

        expected = np.array([
            [100, 0, 10],
            [200, 0, 20],
            [300, 0, 999],  # Should remain unchanged
        ])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_does_not_modify_original(self):
        """Test that original events array is not modified."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
        ])
        events_copy = events.copy()
        evid = {"10": 1, "20": 2}

        result = inverse_map_events(events, evid)

        # Original should be unchanged
        np.testing.assert_array_equal(events, events_copy)
        # Result should be different
        assert not np.array_equal(result, events)

    def test_inverse_map_events_empty_mapping(self):
        """Test with empty event mapping."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
        ])
        evid = {}

        result = inverse_map_events(events, evid)

        # Should return copy with unchanged events
        np.testing.assert_array_equal(result, events)

    def test_inverse_map_events_single_event(self):
        """Test with single event."""
        events = np.array([[100, 0, 1]])
        evid = {"5": 1}

        result = inverse_map_events(events, evid)

        expected = np.array([[100, 0, 5]])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_multiple_numbers_in_key(self):
        """Test when key contains multiple numbers (uses first)."""
        events = np.array([[100, 0, 1]])
        evid = {"event_42_trial_99": 1}

        result = inverse_map_events(events, evid)

        # Should extract first number (42)
        expected = np.array([[100, 0, 42]])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_no_numbers_in_key(self):
        """Test when string key has no numbers."""
        events = np.array([[100, 0, 1]])
        evid = {"stimulus": 1}

        result = inverse_map_events(events, evid)

        # Should keep original value if no number found
        # This happens because the key "stimulus" can't be converted to int
        # and has no digits, so it's skipped in mapping
        np.testing.assert_array_equal(result, events)

    def test_inverse_map_events_preserves_first_two_columns(self):
        """Test that first two columns are preserved."""
        events = np.array([
            [100, 5, 1],
            [200, 10, 2],
            [300, 15, 1],
        ])
        evid = {"10": 1, "20": 2}

        result = inverse_map_events(events, evid)

        # First two columns should be identical
        np.testing.assert_array_equal(result[:, :2], events[:, :2])

    def test_inverse_map_events_large_event_ids(self):
        """Test with large event ID numbers."""
        events = np.array([
            [1000, 0, 100],
            [2000, 0, 200],
        ])
        evid = {"1000": 100, "2000": 200}

        result = inverse_map_events(events, evid)

        expected = np.array([
            [1000, 0, 1000],
            [2000, 0, 2000],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_map_events_mixed_mapping(self):
        """Test with mix of numeric strings and text keys."""
        events = np.array([
            [100, 0, 1],
            [200, 0, 2],
            [300, 0, 3],
        ])
        evid = {
            "5": 1,  # Pure numeric string
            "stim_10": 2,  # Text with number
            "response": 3,  # Text only (no mapping will occur)
        }

        result = inverse_map_events(events, evid)

        expected = np.array([
            [100, 0, 5],
            [200, 0, 10],
            [300, 0, 3],  # Unchanged because "response" has no digits
        ])
        np.testing.assert_array_equal(result, expected)
