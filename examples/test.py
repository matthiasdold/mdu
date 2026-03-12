import plotly.express as px
from mdu.plotly.stats import add_statsmodel_fit

# Create scatter plot
df = px.data.tips()
fig = px.scatter(
    df, x="total_bill", y="tip", title="Tip vs Bill with Statistical Fit", opacity=0.6
)

# Add OLS regression line with confidence interval
fig = add_statsmodel_fit(
    fig,
    x=df["total_bill"].to_numpy(),
    y=df["tip"].to_numpy(),
)

fig.show()

import plotly.express as px
from mdu.plotly.stats import add_box_significance_indicator

# Create box plot
df = px.data.tips()
fig = px.box(df, x="day", y="total_bill", title="Bill Amount by Day")

# Add significance comparisons
# Compare Friday vs Saturday
fig = add_box_significance_indicator(fig, only_significant=False)

fig.show()


import polars as pl
import numpy as np
from mdu.plotly.multiline import multiline_plot

# Generate sample data: 5 subjects, 2 conditions, 100 timepoints
np.random.seed(1)
time = np.linspace(0, 10, 100)

data = []
for subject in ["S1", "S2", "S3", "S4", "S5"]:
    for condition in ["A", "B"]:
        offset = 0.5 if condition == "B" else 0
        noise = np.random.normal(0, 0.15, 100)
        values = np.sin(time + offset) + noise
        for t, v in zip(time, values):
            data.append(
                {"time": t, "value": v, "subject": subject, "condition": condition}
            )

df = pl.DataFrame(data)

fig = multiline_plot(
    df,
    x="time",
    y="value",
    line_group="subject",
    mean=True,
    mean_ci=True,
    std=True,
    single_lines=False,  # Hide individual lines for clarity
    color="condition",
    add_significance=True,  # Automatic significance testing
    significance_line_kwargs={"pval": 0.05, "nperm": 1000, "mode": "line"},
)

fig.update_layout(title="Multi-line Plot with Significance Testing")
fig.show()


from plotly.subplots import make_subplots

import plotly.express as px
from mdu.plotly.shared import add_meta_info

# Create 2x2 subplot layout
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=["Condition A", "Condition B", "Condition C", "Condition D"],
)

# Add traces to each subplot
# ... (your plotting code here)

# Add metadata for each subplot (order: column-first, then row)
metadata_list = [
    "Condition A: Baseline",
    "Condition B: Post-intervention",
    "Condition C: 6-month",
    "Condition D: 12-month",
]

fig = add_meta_info(fig, text=metadata_list)
fig.show()

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create 2x2 subplot layout
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=["Condition A", "Condition B", "Condition C", "Condition D"],
)
fig = fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1
)  # Add traces for Condition A
fig = fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1]), row=2, col=1
)  # Add traces for Condition A
fig = fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=2
)  # Add traces for Condition A
fig = fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1]), row=2, col=2
)  # Add traces for Condition A

# Add traces to each subplot
# ... (your plotting code here)

# Add metadata for each subplot (order: column-first, then row)
metadata_list = [
    "Condition A: Baseline",
    "Condition B: Post-intervention",
    "Condition C: 6-month",
    "Condition D: 12-month",
]

fig = add_meta_info(fig, text=metadata_list)
fig.show()


import numpy as np
import plotly.graph_objects as go
from mdu.plotly.stats import add_cluster_permut_sig_to_plotly

# Simulate EEG-like data: 20 trials, 100 time points
n_trials, n_time = 20, 100
time = np.linspace(0, 1, n_time)

# Group A: baseline activity
curves_a = np.random.randn(n_trials, n_time) * 0.5

# Group B: enhanced activity during 0.4-0.6s (simulated effect)
curves_b = np.random.randn(n_trials, n_time) * 0.5
curves_b[:, 40:60] += 2.0  # Add signal in middle period

# Create plot with mean ± SEM
# Note: this plot could be much more conveniently created with `mdu.plotly.multiline.multiline_plot`, but here we do it manually to demonstrate the cluster permutation test integration.
fig = go.Figure()
mean_a = curves_a.mean(axis=0)
sem_a = curves_a.std(axis=0) / np.sqrt(n_trials)
fig = (
    fig.add_scatter(x=time, y=mean_a, name="Control", line=dict(color="blue"))
    .add_scatter(x=time, y=mean_a + sem_a, line=dict(width=0), showlegend=False)
    .add_scatter(
        x=time,
        y=mean_a - sem_a,
        fill="tonexty",
        line=dict(width=0),
        fillcolor="rgba(0,0,255,0.2)",
        showlegend=False,
    )
)

mean_b = curves_b.mean(axis=0)
sem_b = curves_b.std(axis=0) / np.sqrt(n_trials)
fig = (
    fig.add_scatter(x=time, y=mean_b, name="Treatment", line=dict(color="red"))
    .add_scatter(x=time, y=mean_b + sem_b, line=dict(width=0), showlegend=False)
    .add_scatter(
        x=time,
        y=mean_b - sem_b,
        fill="tonexty",
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.2)",
        showlegend=False,
    )
)

# Add cluster permutation test
fig = add_cluster_permut_sig_to_plotly(
    curves_a=curves_a,
    curves_b=curves_b,
    fig=fig,
    xaxes_vals=time,
    pval=0.05,
    nperm=1000,
    mode="line",  # Options: 'line', 'spark', 'p_bg', 'p_colorbar'
)

fig = fig.update_layout(
    title="Cluster Permutation Test Example",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (μV)",
)
fig.show()


import statsmodels.api as sm
from mdu.plotly.stats import add_statsmodel_fit

# Simulate count data (e.g., Poisson)
x = np.linspace(0, 5, 50)
lambda_true = np.exp(0.5 + 0.3 * x)
y_count = np.random.poisson(lambda_true)

fig = px.scatter(x=x, y=y_count, title="Poisson GLM")

# Fit GLM with Poisson family
fig = add_statsmodel_fit(
    fig,
    x=x,
    y=y_count,
    fitfunc=lambda y, X: sm.GLM(y, X, family=sm.families.Poisson()),
    show_ci=True,
)


# ------------------------------------------
import plotly.express as px
from mdu.plotly.stats import add_box_significance_indicator
from scipy import stats

# Create sample data
df = px.data.tips()

# Create box plot
fig = px.box(
    df, x="day", y="total_bill", color="time", title="Total Bill by Day and Time"
)

# Add significance indicators between all groups
fig = add_box_significance_indicator(
    fig,
    stat_func=stats.ttest_ind,
    same_legendgroup_only=True,  # Only compare same colors
    only_significant=True,
)

fig.show()

# Test only specific day pairs
fig = add_box_significance_indicator(
    fig,
    xval_pairs=[("Thur", "Fri"), ("Fri", "Sat"), ("Sat", "Sun")],
    same_legendgroup_only=True,
)
fig.show()

# Test specific color combinations
fig = add_box_significance_indicator(
    fig, color_pairs=[("Dinner", "Lunch")], same_legendgroup_only=False
)
fig.show()


import plotly.express as px
from mdu.plotly.stats import add_box_significance_indicator
from scipy import stats

# Create sample data
df = px.data.tips()

# Create box plot
fig = px.box(
    df, x="day", y="total_bill", color="time", title="Total Bill by Day and Time"
)

# Add significance indicators between all groups
fig = add_box_significance_indicator(
    fig,
    stat_func=stats.ttest_ind,
    same_legendgroup_only=True,  # Only compare same colors
    only_significant=True,
)

fig.show()


# Test only specific day pairs
fig = add_box_significance_indicator(
    fig,
    xval_pairs=[("Thur", "Fri"), ("Fri", "Sat"), ("Sat", "Sun")],
    same_legendgroup_only=True,
)
fig.show()

# Test specific color combinations
fig = add_box_significance_indicator(
    fig, color_pairs=[("Dinner", "Lunch")], same_legendgroup_only=False
)
fig.show()


# ------------------------------------------------------------------------
import numpy as np
import plotly.graph_objects as go
from mdu.plotly.stats import add_cluster_permut_sig_to_plotly
from mdu.plotly.template import set_template

set_template()

# Simulate EEG-like data: 20 trials, 100 time points
n_trials, n_time = 20, 100
time = np.linspace(0, 1, n_time)

# Group A: baseline activity
curves_a = np.random.randn(n_trials, n_time) * 0.5

# Group B: enhanced activity during 0.4-0.6s (simulated effect)
curves_b = np.random.randn(n_trials, n_time) * 0.5
curves_b[:, 40:60] += 2.0  # Add signal in middle period

# Create plot with mean ± SEM
# Note: this plot could be much more conveniently created with `mdu.plotly.multiline.multiline_plot`, but here we do it manually to demonstrate the cluster permutation test integration.
fig = go.Figure()
mean_a = curves_a.mean(axis=0)
sem_a = curves_a.std(axis=0) / np.sqrt(n_trials)
fig = (
    fig.add_scatter(x=time, y=mean_a, name="Control", line=dict(color="blue"))
    .add_scatter(x=time, y=mean_a + sem_a, line=dict(width=0), showlegend=False)
    .add_scatter(
        x=time,
        y=mean_a - sem_a,
        fill="tonexty",
        line=dict(width=0),
        fillcolor="rgba(0,0,255,0.2)",
        showlegend=False,
    )
)

mean_b = curves_b.mean(axis=0)
sem_b = curves_b.std(axis=0) / np.sqrt(n_trials)
fig = (
    fig.add_scatter(x=time, y=mean_b, name="Treatment", line=dict(color="red"))
    .add_scatter(x=time, y=mean_b + sem_b, line=dict(width=0), showlegend=False)
    .add_scatter(
        x=time,
        y=mean_b - sem_b,
        fill="tonexty",
        line=dict(width=0),
        fillcolor="rgba(255,0,0,0.2)",
        showlegend=False,
    )
)

# Add cluster permutation test
fig = add_cluster_permut_sig_to_plotly(
    curves_a=curves_a,
    curves_b=curves_b,
    fig=fig,
    # xaxes_vals=time,
    pval=0.05,
    nperm=1000,
    mode="line",  # Options: 'line', 'spark', 'p_bg', 'p_colorbar'
)

fig = fig.update_layout(
    title="Cluster Permutation Test Example",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (μV)",
)
fig.show()


fig = add_cluster_permut_sig_to_plotly(
    curves_a, curves_b, fig, xaxes_vals=time, mode="line"
)
fig.show()
