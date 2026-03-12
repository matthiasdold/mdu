"""Tests for mdu.plotly.ml module."""

import numpy as np
import pytest
import plotly.graph_objects as go

from mdu.plotly.ml import plot_roc_curve


class TestPlotRocCurve:
    """Test suite for plot_roc_curve function."""

    def test_plot_roc_curve_basic(self):
        """Test basic ROC curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        fig = plot_roc_curve(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Diagonal line + ROC curve

    def test_plot_roc_curve_perfect_classifier(self):
        """Test ROC curve with perfect classifier (AUC=1.0)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        fig = plot_roc_curve(y_true, y_pred)

        # Check annotation contains AUC=100% or similar
        assert len(fig.layout.annotations) > 0
        annotation_text = fig.layout.annotations[0].text
        assert "AUC" in annotation_text
        assert "100" in annotation_text or "1.0" in annotation_text

    def test_plot_roc_curve_random_classifier(self):
        """Test ROC curve with random classifier (AUC≈0.5)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.rand(100)

        fig = plot_roc_curve(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        # Should have annotation with AUC value
        assert len(fig.layout.annotations) > 0

    def test_plot_roc_curve_has_diagonal_reference(self):
        """Test that diagonal reference line is present."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        fig = plot_roc_curve(y_true, y_pred)

        # First trace should be diagonal line from (0,0) to (1,1)
        diagonal = fig.data[0]
        np.testing.assert_array_equal(diagonal.x, [0, 1])
        np.testing.assert_array_equal(diagonal.y, [0, 1])

    def test_plot_roc_curve_has_roc_trace(self):
        """Test that ROC curve trace is present."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        fig = plot_roc_curve(y_true, y_pred)

        # Second trace should be the ROC curve
        roc = fig.data[1]
        assert roc.name == "ROC"
        assert roc.fill == "tonexty"

    def test_plot_roc_curve_axis_labels(self):
        """Test that axes have correct labels."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.3, 0.7])

        fig = plot_roc_curve(y_true, y_pred)

        assert fig.layout.xaxis.title.text == "False Positive Rate"
        assert fig.layout.yaxis.title.text == "True Positive Rate"

    def test_plot_roc_curve_with_lists(self):
        """Test that function works with Python lists."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        fig = plot_roc_curve(y_true, y_pred)

        assert isinstance(fig, go.Figure)

    def test_plot_roc_curve_imbalanced_classes(self):
        """Test ROC curve with imbalanced classes."""
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.random.rand(100)

        fig = plot_roc_curve(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_roc_curve_all_zeros(self):
        """Test with all negative class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # This should work but may produce warnings
        fig = plot_roc_curve(y_true, y_pred)
        assert isinstance(fig, go.Figure)

    def test_plot_roc_curve_all_ones(self):
        """Test with all positive class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # This should work but may produce warnings
        fig = plot_roc_curve(y_true, y_pred)
        assert isinstance(fig, go.Figure)

    def test_plot_roc_curve_colors(self):
        """Test that curves have expected colors."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        fig = plot_roc_curve(y_true, y_pred)

        # Check diagonal is gray
        assert fig.data[0].line.color == "#888888"
        # Check ROC curve is blue
        assert fig.data[1].line.color == "#5555ff"

    def test_plot_roc_curve_annotation_position(self):
        """Test that AUC annotation is at expected position."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])

        fig = plot_roc_curve(y_true, y_pred)

        annotation = fig.layout.annotations[0]
        assert annotation.x == 0.5
        assert annotation.y == 0.8
