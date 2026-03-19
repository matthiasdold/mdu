"""Tests for mdu.mne.mne2dataframe module."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import mne

from mdu.mne.mne2dataframe import mne_epochs_to_polars, mne_raw_to_polars


class TestMneEpochsToPolars:
    """Test suite for mne_epochs_to_polars function."""

    @pytest.fixture
    def sample_epochs(self):
        """Create sample MNE epochs for testing."""
        # Create info with 3 channels
        ch_names = ["Ch1", "Ch2", "Ch3"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")

        # Create data: 5 epochs, 3 channels, 50 timepoints
        n_epochs = 5
        n_channels = 3
        n_times = 50
        rng = np.random.RandomState(42)
        data = rng.randn(n_epochs, n_channels, n_times) * 1e-6  # Typical EEG scale

        # Create epochs with metadata
        metadata = pd.DataFrame({
            "condition": ["A", "B", "A", "B", "A"],
            "rt": [0.5, 0.6, 0.55, 0.58, 0.52],
            "correct": [True, True, False, True, True]
        })

        epochs = mne.EpochsArray(data, info, metadata=metadata, verbose=False)
        return epochs, data

    @pytest.fixture
    def sample_epochs_no_metadata(self):
        """Create sample MNE epochs without metadata."""
        ch_names = ["Ch1", "Ch2"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        n_epochs = 3
        data = np.random.RandomState(42).randn(n_epochs, 2, 40) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        return epochs, data

    def test_basic_conversion(self, sample_epochs):
        """Test basic conversion of epochs to polars DataFrame."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Check that result is a polars DataFrame
        assert isinstance(df, pl.DataFrame)

        # Check that all expected columns are present
        assert "time" in df.columns
        assert "epoch_nr" in df.columns
        assert "sample_idx" in df.columns
        assert "Ch1" in df.columns
        assert "Ch2" in df.columns
        assert "Ch3" in df.columns

    def test_data_shape(self, sample_epochs):
        """Test that the output DataFrame has correct shape."""
        epochs, data = sample_epochs
        df = mne_epochs_to_polars(epochs)

        n_epochs, n_channels, n_times = data.shape
        expected_rows = n_epochs * n_times

        assert df.height == expected_rows

    def test_scaling_to_uv(self, sample_epochs):
        """Test that data is scaled from V to µV."""
        epochs, data = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Get first epoch, first channel
        first_epoch_ch1 = df.filter(pl.col("epoch_nr") == 0)["Ch1"].to_numpy()

        # Compare with original data (scaled to µV)
        expected = data[0, 0, :] * 1e6
        np.testing.assert_array_almost_equal(first_epoch_ch1, expected, decimal=5)

    def test_time_column(self, sample_epochs):
        """Test that time column is correctly populated."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Get times for first epoch
        times_epoch_0 = df.filter(pl.col("epoch_nr") == 0)["time"].to_numpy()

        # Compare with MNE times
        np.testing.assert_array_almost_equal(times_epoch_0, epochs.times)

    def test_time_column_repeated_per_epoch(self, sample_epochs):
        """Test that time values are repeated for each epoch."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Get times for different epochs
        times_epoch_0 = df.filter(pl.col("epoch_nr") == 0)["time"].to_numpy()
        times_epoch_1 = df.filter(pl.col("epoch_nr") == 1)["time"].to_numpy()
        times_epoch_2 = df.filter(pl.col("epoch_nr") == 2)["time"].to_numpy()

        # All epochs should have same time values
        np.testing.assert_array_equal(times_epoch_0, times_epoch_1)
        np.testing.assert_array_equal(times_epoch_0, times_epoch_2)

    def test_epoch_nr_column(self, sample_epochs):
        """Test that epoch_nr column is correctly assigned."""
        epochs, data = sample_epochs
        df = mne_epochs_to_polars(epochs)

        n_epochs = data.shape[0]

        for i in range(n_epochs):
            epoch_data = df.filter(pl.col("epoch_nr") == i)
            # Each epoch should have n_times rows
            assert epoch_data.height == data.shape[2]

    def test_sample_idx_column(self, sample_epochs):
        """Test that sample_idx is a continuous index."""
        epochs, data = sample_epochs
        df = mne_epochs_to_polars(epochs)

        sample_indices = df["sample_idx"].to_numpy()

        # Should be continuous from 0 to total number of samples
        expected_indices = np.arange(data.shape[0] * data.shape[2])
        np.testing.assert_array_equal(sample_indices, expected_indices)

    def test_metadata_join(self, sample_epochs):
        """Test that metadata is correctly joined."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Check metadata columns are present
        assert "condition" in df.columns
        assert "rt" in df.columns
        assert "correct" in df.columns

        # Check first epoch has correct metadata
        first_epoch = df.filter(pl.col("epoch_nr") == 0)
        assert first_epoch["condition"][0] == "A"
        assert first_epoch["rt"][0] == 0.5
        assert first_epoch["correct"][0] is True

        # Check metadata is repeated for all samples in epoch
        assert first_epoch["condition"].unique().to_list() == ["A"]

    def test_metadata_consistency_across_epochs(self, sample_epochs):
        """Test that each epoch has its corresponding metadata."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Check each epoch
        expected_conditions = ["A", "B", "A", "B", "A"]
        expected_rts = [0.5, 0.6, 0.55, 0.58, 0.52]

        for i, (cond, rt) in enumerate(zip(expected_conditions, expected_rts)):
            epoch_data = df.filter(pl.col("epoch_nr") == i)
            assert epoch_data["condition"][0] == cond
            assert epoch_data["rt"][0] == rt

    def test_no_metadata(self, sample_epochs_no_metadata):
        """Test conversion when epochs have no metadata."""
        epochs, _ = sample_epochs_no_metadata
        df = mne_epochs_to_polars(epochs)

        # Should still have basic columns
        assert "time" in df.columns
        assert "epoch_nr" in df.columns
        assert "sample_idx" in df.columns
        assert "Ch1" in df.columns
        assert "Ch2" in df.columns

    def test_single_epoch(self):
        """Test with single epoch."""
        ch_names = ["Ch1"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(1, 1, 30) * 1e-6

        epochs = mne.EpochsArray(data, info, verbose=False)
        df = mne_epochs_to_polars(epochs)

        assert df.height == 30
        assert df["epoch_nr"].unique().to_list() == [0]

    def test_channel_names_preserved(self, sample_epochs):
        """Test that channel names are preserved as column names."""
        epochs, _ = sample_epochs
        df = mne_epochs_to_polars(epochs)

        for ch_name in epochs.ch_names:
            assert ch_name in df.columns

    def test_data_integrity(self, sample_epochs):
        """Test that data values are preserved (with scaling)."""
        epochs, data = sample_epochs
        df = mne_epochs_to_polars(epochs)

        # Test random epoch and channel
        epoch_nr = 2
        ch_idx = 1
        ch_name = epochs.ch_names[ch_idx]

        epoch_data = df.filter(pl.col("epoch_nr") == epoch_nr)[ch_name].to_numpy()
        expected = data[epoch_nr, ch_idx, :] * 1e6

        np.testing.assert_array_almost_equal(epoch_data, expected, decimal=5)

    def test_multiple_channel_types(self):
        """Test with mixed channel types."""
        ch_names = ["EEG1", "EEG2", "EOG1"]
        ch_types = ["eeg", "eeg", "eog"]
        info = mne.create_info(ch_names, sfreq=100, ch_types=ch_types)

        data = np.random.RandomState(42).randn(2, 3, 25) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)

        df = mne_epochs_to_polars(epochs)

        # All channels should be present
        assert "EEG1" in df.columns
        assert "EEG2" in df.columns
        assert "EOG1" in df.columns


class TestMneRawToPolars:
    """Test suite for mne_raw_to_polars function."""

    @pytest.fixture
    def sample_raw(self):
        """Create sample MNE Raw object for testing."""
        ch_names = ["Ch1", "Ch2", "Ch3"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")

        n_channels = 3
        n_times = 200
        rng = np.random.RandomState(42)
        data = rng.randn(n_channels, n_times) * 1e-6  # Typical EEG scale

        raw = mne.io.RawArray(data, info, verbose=False)
        return raw, data

    def test_basic_conversion(self, sample_raw):
        """Test basic conversion of Raw to polars DataFrame."""
        raw, _ = sample_raw
        df = mne_raw_to_polars(raw)

        # Check that result is a polars DataFrame
        assert isinstance(df, pl.DataFrame)

        # Check that all expected columns are present
        assert "time" in df.columns
        assert "sample_idx" in df.columns
        assert "Ch1" in df.columns
        assert "Ch2" in df.columns
        assert "Ch3" in df.columns

    def test_data_shape(self, sample_raw):
        """Test that the output DataFrame has correct shape."""
        raw, data = sample_raw
        df = mne_raw_to_polars(raw)

        _, n_times = data.shape
        assert df.height == n_times

    def test_scaling_to_uv(self, sample_raw):
        """Test that data is scaled from V to µV."""
        raw, data = sample_raw
        df = mne_raw_to_polars(raw)

        # Get first channel data
        ch1_data = df["Ch1"].to_numpy()

        # Compare with original data (scaled to µV)
        expected = data[0, :] * 1e6
        np.testing.assert_array_almost_equal(ch1_data, expected, decimal=5)

    def test_time_column(self, sample_raw):
        """Test that time column is correctly populated."""
        raw, _ = sample_raw
        df = mne_raw_to_polars(raw)

        times = df["time"].to_numpy()

        # Compare with MNE times
        np.testing.assert_array_almost_equal(times, raw.times)

    def test_sample_idx_column(self, sample_raw):
        """Test that sample_idx is a continuous index."""
        raw, data = sample_raw
        df = mne_raw_to_polars(raw)

        sample_indices = df["sample_idx"].to_numpy()

        # Should be continuous from 0 to n_times-1
        expected_indices = np.arange(data.shape[1])
        np.testing.assert_array_equal(sample_indices, expected_indices)

    def test_channel_names_preserved(self, sample_raw):
        """Test that channel names are preserved as column names."""
        raw, _ = sample_raw
        df = mne_raw_to_polars(raw)

        for ch_name in raw.ch_names:
            assert ch_name in df.columns

    def test_data_integrity(self, sample_raw):
        """Test that data values are preserved (with scaling)."""
        raw, data = sample_raw
        df = mne_raw_to_polars(raw)

        # Test all channels
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ch_data = df[ch_name].to_numpy()
            expected = data[ch_idx, :] * 1e6
            np.testing.assert_array_almost_equal(ch_data, expected, decimal=5)

    def test_single_channel(self):
        """Test with single channel Raw."""
        info = mne.create_info(["Ch1"], sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(1, 100) * 1e-6

        raw = mne.io.RawArray(data, info, verbose=False)
        df = mne_raw_to_polars(raw)

        assert df.height == 100
        assert "Ch1" in df.columns
        assert df.shape[1] == 3  # Ch1, time, sample_idx

    def test_many_channels(self):
        """Test with many channels."""
        n_channels = 64
        ch_names = [f"Ch{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")

        data = np.random.RandomState(42).randn(n_channels, 50) * 1e-6

        raw = mne.io.RawArray(data, info, verbose=False)
        df = mne_raw_to_polars(raw)

        assert df.height == 50
        # n_channels + time + sample_idx
        assert df.shape[1] == n_channels + 2

    def test_long_recording(self):
        """Test with long recording."""
        info = mne.create_info(["Ch1", "Ch2"], sfreq=1000, ch_types="eeg")
        n_times = 10000
        data = np.random.RandomState(42).randn(2, n_times) * 1e-6

        raw = mne.io.RawArray(data, info, verbose=False)
        df = mne_raw_to_polars(raw)

        assert df.height == n_times

    def test_multiple_channel_types(self):
        """Test with mixed channel types."""
        ch_names = ["EEG1", "EEG2", "EOG1", "ECG1"]
        ch_types = ["eeg", "eeg", "eog", "ecg"]
        info = mne.create_info(ch_names, sfreq=100, ch_types=ch_types)

        data = np.random.RandomState(42).randn(4, 100) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)

        df = mne_raw_to_polars(raw)

        # All channels should be present
        for ch_name in ch_names:
            assert ch_name in df.columns

    def test_time_continuity(self, sample_raw):
        """Test that time values are continuous and evenly spaced."""
        raw, _ = sample_raw
        df = mne_raw_to_polars(raw)

        times = df["time"].to_numpy()
        time_diffs = np.diff(times)

        # All time differences should be approximately equal
        expected_diff = 1.0 / raw.info["sfreq"]
        np.testing.assert_array_almost_equal(
            time_diffs,
            np.full_like(time_diffs, expected_diff),
            decimal=10
        )

    def test_different_sampling_rates(self):
        """Test with different sampling rates."""
        for sfreq in [100, 250, 500, 1000]:
            info = mne.create_info(["Ch1"], sfreq=sfreq, ch_types="eeg")
            data = np.random.RandomState(42).randn(1, 100) * 1e-6

            raw = mne.io.RawArray(data, info, verbose=False)
            df = mne_raw_to_polars(raw)

            # Check time spacing
            times = df["time"].to_numpy()
            time_diff = times[1] - times[0]
            expected_diff = 1.0 / sfreq

            assert abs(time_diff - expected_diff) < 1e-10
