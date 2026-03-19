import numpy as np
import mne
from mdu.plotly.template import set_template
from mdu.plotly.mne_plotting import plot_evoked
from mdu.plotly.mne_plotting_utils.topoplot import create_plotly_topoplot

# Set template for consistent styling
set_template()

# Load MNE sample data
sample_data_folder = mne.datasets.sample.data_path()
raw_fname = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
events_fname = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw-eve.fif"

# Load and prepare data
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
raw.pick_types(meg=False, eeg=True)
raw.crop(tmax=60)  # Use first 60 seconds for demo

# Simulate bilateral activity
np.random.seed(42)
n_channels = len(raw.ch_names)
simulated_data = np.random.randn(n_channels) * 0.3

# Add stronger activity to left hemisphere channels
left_channels = [i for i, name in enumerate(raw.ch_names) if name.startswith("EEG 0")]
simulated_data[left_channels] += 2.0
events = mne.read_events(events_fname, verbose=False)

# Create epochs for auditory left condition
epochs = mne.Epochs(
    raw,
    events,
    event_id={"auditory/left": 1},
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    preload=True,
    verbose=False,
)
epochs = epochs.resample(100)  # Resample for faster plotting

# Create ERP plot with all channels
sample_channels = raw.ch_names[:5]  # Use first 10 channels for demonstration
fig = plot_evoked(epochs.copy().pick(sample_channels))

fig.update_layout(
    title="Event-Related Potentials - All Channels", height=600, width=900
)

fig.show()

fig = plot_evoked(epochs, time_topo=[0.1, 0.2, 0.3])

fig.update_layout(
    title="ERPs with Time-Locked Topoplots",
)

fig.show()


custom_colors = {}
for ch in epochs.ch_names:
    if ch in epochs.ch_names[:2]:
        custom_colors[ch] = "#7f7f7f"
    elif ch in epochs.ch_names[2:4]:
        custom_colors[ch] = "#d62728"
    else:
        custom_colors[ch] = "#1f77b4"

# Plot with custom colors
selected_chs = epochs.ch_names[:6]
fig = plot_evoked(epochs.copy().pick(selected_chs), cmap=custom_colors)
fig.update_layout(title="ERPs - Custom Channel Coloring", height=600, width=900)
fig.show()


fig = create_plotly_topoplot(
    data=simulated_data,
    inst=raw,
    contour_kwargs=dict(contours_coloring="heatmap", colorscale="Viridis"),
    blank_scaling=1,
)

fig.update_layout(
    title="Simulated Left Hemisphere Activity", height=500, width=500
).update_xaxes(visible=False).update_yaxes(visible=False)

fig.show()
