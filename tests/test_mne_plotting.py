"""Tests for mdu.plotly.mne_plotting module."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import mne
import plotly.graph_objects as go

from mdu.plotly.mne_plotting import (
    plot_topo,
    plot_evoked,
    add_time_locked_topo,
    plot_variances,
    plot_epoch_image,
)
from mdu.mne.mne2dataframe import mne_epochs_to_polars


class TestPlotTopo:
    """Test suite for plot_topo function."""

    @pytest.fixture
    def sample_raw(self):
        """Create sample MNE Raw object with standard montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(3, 100) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)
        # Set standard montage for proper channel locations
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw

    @pytest.fixture
    def sample_epochs(self):
        """Create sample MNE Epochs object with standard montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(5, 3, 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    def test_plot_topo_basic(self, sample_raw):
        """Test basic topoplot creation."""
        data = np.array([1.0, 2.0, 3.0])
        fig = plot_topo(data, sample_raw, show=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_topo_with_epochs(self, sample_epochs):
        """Test topoplot with epochs object."""
        data = np.array([1.0, 2.0, 3.0])
        fig = plot_topo(data, sample_epochs, show=False)

        assert isinstance(fig, go.Figure)

    def test_plot_topo_custom_colorscale(self, sample_raw):
        """Test topoplot with custom colorscale."""
        data = np.array([1.0, 2.0, 3.0])
        fig = plot_topo(
            data, sample_raw, contour_kwargs={"colorscale": "RdBu_r"}, show=False
        )

        assert isinstance(fig, go.Figure)

    def test_plot_topo_custom_scaling(self, sample_raw):
        """Test topoplot with custom scaling parameters."""
        data = np.array([1.0, 2.0, 3.0])
        fig = plot_topo(
            data, sample_raw, scale_range=1.5, blank_scaling=0.3, show=False
        )

        assert isinstance(fig, go.Figure)

    def test_plot_topo_negative_values(self, sample_raw):
        """Test topoplot with negative values."""
        data = np.array([-1.0, 0.0, 1.0])
        fig = plot_topo(data, sample_raw, show=False)

        assert isinstance(fig, go.Figure)

    def test_plot_topo_all_zeros(self, sample_raw):
        """Test topoplot with all zero values."""
        data = np.array([0.0, 0.0, 0.0])
        fig = plot_topo(data, sample_raw, show=False)

        assert isinstance(fig, go.Figure)

    def test_plot_topo_large_values(self, sample_raw):
        """Test topoplot with large values."""
        data = np.array([100.0, 200.0, 300.0])
        fig = plot_topo(data, sample_raw, show=False)

        assert isinstance(fig, go.Figure)


class TestPlotEvoked:
    """Test suite for plot_evoked function."""

    @pytest.fixture
    def sample_epochs_with_metadata(self):
        """Create sample epochs with metadata and standard montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        n_epochs = 10
        data = np.random.RandomState(42).randn(n_epochs, 3, 50) * 1e-6

        metadata = pd.DataFrame(
            {
                "condition": ["A"] * 5 + ["B"] * 5,
                "rt": np.random.uniform(0.3, 0.8, n_epochs),
            }
        )

        epochs = mne.EpochsArray(data, info, metadata=metadata, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    @pytest.fixture
    def sample_epochs_simple(self):
        """Create simple sample epochs without metadata but with montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(5, 3, 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    def test_plot_evoked_basic(self, sample_epochs_simple):
        """Test basic evoked plot creation."""
        fig = plot_evoked(sample_epochs_simple)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_evoked_with_metadata(self, sample_epochs_with_metadata):
        """Test evoked plot with metadata."""
        fig = plot_evoked(sample_epochs_with_metadata)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_evoked_with_precomputed_dataframe(self, sample_epochs_simple):
        """Test evoked plot with pre-computed DataFrame."""
        dp = mne_epochs_to_polars(sample_epochs_simple)
        fig = plot_evoked(sample_epochs_simple, dp=dp)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_evoked_with_time_topo(self, sample_epochs_simple):
        """Test evoked plot with time-locked topoplots."""
        times = sample_epochs_simple.times
        time_topo = [times[10], times[20], times[30]]

        fig = plot_evoked(sample_epochs_simple, time_topo=time_topo)

        assert isinstance(fig, go.Figure)
        # Should have multiple subplots (topoplots + time series)
        assert len(fig.data) > len(sample_epochs_simple.ch_names)

    def test_plot_evoked_with_single_time_topo(self, sample_epochs_simple):
        """Test evoked plot with single topoplot time point."""
        times = sample_epochs_simple.times
        time_topo = [times[20]]

        fig = plot_evoked(sample_epochs_simple, time_topo=time_topo)

        assert isinstance(fig, go.Figure)

    def test_plot_evoked_custom_colormap(self, sample_epochs_simple):
        """Test evoked plot with custom color mapping."""
        custom_cmap = {
            "Fz": "#FF0000",
            "Cz": "#00FF00",
            "Pz": "#0000FF",
        }

        fig = plot_evoked(sample_epochs_simple, cmap=custom_cmap)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_evoked_invalid_dataframe_raises_error(self, sample_epochs_simple):
        """Test that invalid DataFrame raises ValueError."""
        # Create DataFrame without required columns
        invalid_df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        with pytest.raises(ValueError, match="sample_idx"):
            plot_evoked(sample_epochs_simple, dp=invalid_df)

    def test_plot_evoked_single_epoch(self):
        """Test evoked plot with single epoch."""
        ch_names = ["Fz", "Cz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(1, 2, 30) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)

        fig = plot_evoked(epochs)

        assert isinstance(fig, go.Figure)

    def test_plot_evoked_many_channels(self):
        """Test evoked plot with many channels."""
        # Use real EEG channel names
        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T7",
            "T8",
            "P7",
            "P8",
        ]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(5, len(ch_names), 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)

        fig = plot_evoked(epochs)

        assert isinstance(fig, go.Figure)
        # Should have traces for all channels
        assert len(fig.data) > len(ch_names)  # Mean + CI for each channel

    def test_plot_evoked_long_epochs(self):
        """Test evoked plot with long time series."""
        ch_names = ["Fz", "Cz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(5, 2, 200) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)

        fig = plot_evoked(epochs)

        assert isinstance(fig, go.Figure)

    def test_plot_evoked_with_time_topo_sorted(self, sample_epochs_simple):
        """Test that time_topo list is sorted automatically."""
        times = sample_epochs_simple.times
        # Provide unsorted times
        time_topo = [times[30], times[10], times[20]]

        fig = plot_evoked(sample_epochs_simple, time_topo=time_topo)

        assert isinstance(fig, go.Figure)
        # Should work without error even with unsorted times


class TestAddTimeLockedTopo:
    """Test suite for add_time_locked_topo function."""

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs with standard montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(5, 3, 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    @pytest.fixture
    def sample_multiline_fig(self):
        """Create a simple multiline figure."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 2, 3], mode="lines"))
        return fig

    def test_add_time_locked_topo_basic(self, sample_epochs, sample_multiline_fig):
        """Test basic time-locked topoplot addition."""
        dp = mne_epochs_to_polars(sample_epochs)
        times = sample_epochs.times
        time_topo = [times[10], times[20]]

        fig = add_time_locked_topo(dp, sample_epochs, time_topo, sample_multiline_fig)

        assert isinstance(fig, go.Figure)
        # Should have more traces than original
        assert len(fig.data) > len(sample_multiline_fig.data)

    def test_add_time_locked_topo_single_time(self, sample_epochs, sample_multiline_fig):
        """Test with single time point."""
        dp = mne_epochs_to_polars(sample_epochs)
        times = sample_epochs.times
        time_topo = [times[20]]

        fig = add_time_locked_topo(dp, sample_epochs, time_topo, sample_multiline_fig)

        assert isinstance(fig, go.Figure)

    def test_add_time_locked_topo_multiple_times(
        self, sample_epochs, sample_multiline_fig
    ):
        """Test with multiple time points."""
        dp = mne_epochs_to_polars(sample_epochs)
        times = sample_epochs.times
        time_topo = [times[10], times[20], times[30], times[40]]

        fig = add_time_locked_topo(dp, sample_epochs, time_topo, sample_multiline_fig)

        assert isinstance(fig, go.Figure)

    def test_add_time_locked_topo_unsorted_times(
        self, sample_epochs, sample_multiline_fig
    ):
        """Test that unsorted times are handled correctly."""
        dp = mne_epochs_to_polars(sample_epochs)
        times = sample_epochs.times
        # Provide unsorted times
        time_topo = [times[30], times[10], times[20]]

        fig = add_time_locked_topo(dp, sample_epochs, time_topo, sample_multiline_fig)

        assert isinstance(fig, go.Figure)


class TestPlotVariances:
    """Test suite for plot_variances function."""

    @pytest.fixture
    def sample_epochs_with_df(self):
        """Create sample epochs with matching DataFrame."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        n_epochs = 10
        data = np.random.RandomState(42).randn(n_epochs, 3, 50) * 1e-6

        df = pd.DataFrame(
            {
                "condition": ["A"] * 5 + ["B"] * 5,
                "rt": np.random.uniform(0.3, 0.8, n_epochs),
            }
        )

        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs, df

    def test_plot_variances_basic(self, sample_epochs_with_df):
        """Test basic variance plot with color grouping."""
        epochs, df = sample_epochs_with_df
        # Ensure df has correct index matching epochs
        df.index = range(len(epochs))
        # Must provide color_by since empty string causes KeyError
        fig = plot_variances(epochs, df, color_by="condition", show=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_variances_with_color_by(self, sample_epochs_with_df):
        """Test variance plot with color grouping."""
        epochs, df = sample_epochs_with_df
        df.index = range(len(epochs))
        fig = plot_variances(epochs, df, color_by="condition", show=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPlotEpochImage:
    """Test suite for plot_epoch_image function."""

    @pytest.fixture
    def sample_epochs(self):
        """Create sample epochs with standard montage."""
        ch_names = ["Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names, sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(10, 3, 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    def test_plot_epoch_image_basic(self, sample_epochs):
        """Test basic epoch image plot."""
        # Create DataFrame for epoch metadata
        df = pd.DataFrame({"trial": range(len(sample_epochs))})
        df.index = range(len(sample_epochs))

        fig = plot_epoch_image(sample_epochs, df)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_epoch_image_single_channel(self):
        """Test epoch image with single channel."""
        info = mne.create_info(["Fz"], sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(10, 1, 50) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)

        df = pd.DataFrame({"trial": range(len(epochs))})
        df.index = range(len(epochs))

        fig = plot_epoch_image(epochs, df)

        assert isinstance(fig, go.Figure)

    def test_plot_epoch_image_many_epochs(self):
        """Test epoch image with many epochs."""
        info = mne.create_info(["Fz", "Cz"], sfreq=100, ch_types="eeg")
        data = np.random.RandomState(42).randn(50, 2, 30) * 1e-6
        epochs = mne.EpochsArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)

        df = pd.DataFrame({"trial": range(len(epochs))})
        df.index = range(len(epochs))

        fig = plot_epoch_image(epochs, df)

        assert isinstance(fig, go.Figure)


class TestIntegration:
    """Integration tests for MNE plotting functions."""

    @pytest.fixture
    def realistic_epochs(self):
        """Create more realistic epochs with proper timing and montage."""
        ch_names = [f"EEG{i:03d}" for i in range(1, 11)]  # 10 EEG channels
        # Use actual standard channel names that exist in standard montages
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        info = mne.create_info(ch_names, sfreq=256, ch_types="eeg")

        # Create epochs from -0.2 to 0.5 seconds
        n_epochs = 20
        times = np.arange(-0.2, 0.5, 1 / 256)
        n_times = len(times)

        # Simulate some realistic ERP-like activity
        rng = np.random.RandomState(42)
        data = rng.randn(n_epochs, 10, n_times) * 1e-6

        # Add a simulated P300-like component around 300ms
        p300_time = np.argmin(np.abs(times - 0.3))
        for epoch in range(n_epochs):
            for ch in range(10):
                data[epoch, ch, p300_time - 5 : p300_time + 5] += 5e-6

        metadata = pd.DataFrame(
            {
                "condition": ["target"] * 10 + ["standard"] * 10,
                "rt": rng.uniform(0.3, 0.8, n_epochs),
                "correct": rng.choice([True, False], n_epochs, p=[0.8, 0.2]),
            }
        )

        epochs = mne.EpochsArray(data, info, tmin=-0.2, metadata=metadata, verbose=False)
        # Set standard montage
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        return epochs

    def test_full_pipeline_simple(self, realistic_epochs):
        """Test full pipeline: epochs -> DataFrame -> plot."""
        # Convert to DataFrame
        dp = mne_epochs_to_polars(realistic_epochs)

        # Create plot with pre-computed DataFrame
        fig = plot_evoked(realistic_epochs, dp=dp)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_full_pipeline_with_topoplots(self, realistic_epochs):
        """Test full pipeline with time-locked topoplots."""
        # Create plot with topoplots at multiple time points
        fig = plot_evoked(realistic_epochs, time_topo=[0.0, 0.1, 0.2, 0.3])

        assert isinstance(fig, go.Figure)
        # Should have many traces (channels + topoplots)
        assert len(fig.data) > len(realistic_epochs.ch_names)

    def test_dataframe_reuse(self, realistic_epochs):
        """Test that pre-computed DataFrame can be reused."""
        dp = mne_epochs_to_polars(realistic_epochs)

        # Create multiple plots with same DataFrame
        fig1 = plot_evoked(realistic_epochs, dp=dp)
        fig2 = plot_evoked(realistic_epochs, dp=dp, time_topo=[0.1, 0.2])

        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert len(fig1.data) > 0
        assert len(fig2.data) > len(fig1.data)  # fig2 has topoplots

    def test_custom_colors_with_topoplots(self, realistic_epochs):
        """Test custom colors work with topoplots."""
        custom_cmap = {ch: "#FF0000" for ch in realistic_epochs.ch_names[:5]}
        custom_cmap.update({ch: "#0000FF" for ch in realistic_epochs.ch_names[5:]})

        fig = plot_evoked(realistic_epochs, time_topo=[0.1, 0.2], cmap=custom_cmap)

        assert isinstance(fig, go.Figure)
