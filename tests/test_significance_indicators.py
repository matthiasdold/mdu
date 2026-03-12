import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from mdu.plotly.stats import add_box_significance_indicator


@pytest.fixture
def get_test_data() -> pd.DataFrame:
    np.random.seed(42)  # Fixed seed for reproducible tests
    df = pd.DataFrame(np.random.randn(200, 2), columns=["a", "b"])  # type: ignore

    df["label"] = np.random.choice(["aa", "bb", "cc"], 200)
    df["color"] = np.random.choice(["xx", "yy", "zz"], 200)
    df["cat"] = np.random.choice(["rr", "ee"], 200)
    df.loc[df.color == "xx", "a"] += 20

    return df


def test_add_box_significance_indicator(get_test_data: pd.DataFrame) -> None:
    df = get_test_data

    fig = px.box(df, x="label", y="a", color="color", facet_col="cat")
    fig = add_box_significance_indicator(fig)

    sig_ind_11 = list(fig.select_traces(selector=dict(name="**"), col=1))
    sig_ind_12 = list(fig.select_traces(selector=dict(name="**"), col=2))

    # With fixed seed (42), we get 20 and 18 significance indicators
    assert len(sig_ind_11) == 20, (
        f"Expected 20 significance indicators in facet 1, got {len(sig_ind_11)}"
    )
    assert len(sig_ind_12) == 18, (
        f"Expected 18 significance indicators in facet 2, got {len(sig_ind_12)}"
    )


def test_add_box_significance_indicator_color_contrained(
    get_test_data: pd.DataFrame,
) -> None:
    df = get_test_data

    fig2 = px.box(df, x="label", y="a", color="color", facet_col="cat")
    fig2 = add_box_significance_indicator(
        fig2,
        color_pairs=[("xx", "yy")],
    )

    sig_ind_21 = list(fig2.select_traces(selector=dict(name="**"), col=1))
    sig_ind_22 = list(fig2.select_traces(selector=dict(name="**"), col=2))

    assert len(sig_ind_21) == len(sig_ind_22) == 9, (
        "There should be 9 significance indicators in each facet when constraining the color_pairs"
    )


def test_add_box_significance_indicator_x_contrained(
    get_test_data: pd.DataFrame,
) -> None:
    df = get_test_data
    fig3 = px.box(df, x="label", y="a", color="color", facet_col="cat")
    fig3 = add_box_significance_indicator(
        fig3,
        xval_pairs=[("aa", "aa")],
    )

    sig_ind_31 = list(fig3.select_traces(selector=dict(name="**"), col=1))
    sig_ind_32 = list(fig3.select_traces(selector=dict(name="**"), col=2))

    assert len(sig_ind_31) == len(sig_ind_32) == 2, (
        "There should be 2 significance indicators in each facet when constraining the xval"
    )


def test_add_box_significance_indicator_x_contrained_and_color_constrained(
    get_test_data: pd.DataFrame,
) -> None:
    df = get_test_data
    fig4 = px.box(df, x="label", y="a", color="color", facet_col="cat")
    fig4 = add_box_significance_indicator(
        fig4,
        xval_pairs=[("aa", "aa")],
        color_pairs=[("xx", "yy")],
    )

    sig_ind_41 = list(fig4.select_traces(selector=dict(name="**"), col=1))
    sig_ind_42 = list(fig4.select_traces(selector=dict(name="**"), col=2))

    assert len(sig_ind_41) == len(sig_ind_42) == 1, (
        "There should be 1 significance indicators in each facet when constraining the xval"
    )


def test_categorical_xaxis_labels_preserved() -> None:
    """Test that categorical x-axis labels are preserved after adding significance indicators.

    This is a regression test for the issue where string tick labels were lost
    when the x-axis was converted to numeric for positioning significance indicators.
    """
    # Use the tips dataset which has categorical x values
    tips = px.data.tips()

    # Create a box plot with categorical x-axis
    fig = px.box(tips, x="time", y="total_bill", color="sex", custom_data=["size"])

    # Store original categorical values
    original_categories = sorted(tips["time"].unique())

    # Add significance indicators
    fig = add_box_significance_indicator(fig, only_significant=False)

    # Check that x-axis is now linear (numeric) but has categorical labels
    xaxis = fig.layout.xaxis
    assert xaxis.type == "linear", (
        "X-axis should be converted to linear for positioning"
    )
    assert xaxis.ticktext is not None, "X-axis should have tick text labels"
    assert xaxis.tickvals is not None, "X-axis should have tick values"

    # Check that categorical labels are preserved
    assert len(xaxis.ticktext) == len(original_categories), (
        f"Expected {len(original_categories)} tick labels, got {len(xaxis.ticktext)}"
    )
    assert set(xaxis.ticktext) == set(original_categories), (
        f"Tick labels {set(xaxis.ticktext)} don't match original categories {set(original_categories)}"
    )

    # Check that box traces have numeric x values
    box_traces = [tr for tr in fig.data if tr.type == "box"]
    assert len(box_traces) > 0, "Should have box traces"

    for tr in box_traces:
        assert all(isinstance(x, (int, float)) for x in tr.x), (
            f"All x values should be numeric for trace {tr.name}"
        )

    # Check that numeric positions are within expected range
    all_x_values = []
    for tr in box_traces:
        all_x_values.extend(tr.x)

    assert min(all_x_values) >= -0.5, "X values should be >= -0.5"
    assert max(all_x_values) < len(original_categories), (
        f"X values should be < {len(original_categories)}"
    )
