"""
Histogram plot implementation.

Provides distribution visualization with various binning options
and overlay capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from typing import Optional

from ..base import BasePlot
from ...core.types import PlotConfig


class HistogramPlot(BasePlot):
    """
    Histogram for distribution visualization.

    Supports multiple signals, PDF overlay, and various
    normalization options.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize HistogramPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.n_bins = config.get("n_bins", 50)
        self.normalize = config.get("normalize", True)
        self.show_kde = config.get("show_kde", True)

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render histogram.

        Args:
            data: DataFrame with signal columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data
        self.figure = go.Figure()

        signals = self.config.get("signals", [])
        timestamp_col = self.config.get("x_axis", "timestamp")

        if not signals:
            signals = [
                col for col in data.columns if col != timestamp_col
            ][:3]

        colors = self._get_color_sequence()

        for i, signal_name in enumerate(signals):
            if signal_name not in data.columns:
                continue

            values = data[signal_name].dropna().values
            color = colors[i % len(colors)]

            # Add histogram
            histnorm = 'probability density' if self.normalize else None

            self.figure.add_trace(go.Histogram(
                x=values,
                name=self._get_signal_name(signal_name),
                nbinsx=self.n_bins,
                histnorm=histnorm,
                marker_color=color,
                opacity=0.7,
            ))

            # Add KDE if requested
            if self.show_kde and len(values) > 10:
                self._add_kde_overlay(values, signal_name, color)

        # Layout
        layout = self._create_base_layout()
        layout.update({
            "xaxis_title": self.config.get("x_label", "Value"),
            "yaxis_title": "Density" if self.normalize else "Count",
            "barmode": "overlay",
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def _add_kde_overlay(
        self,
        values: np.ndarray,
        signal_name: str,
        color: str
    ) -> None:
        """Add KDE curve overlay."""
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        kde_values = kde(x_range)

        self.figure.add_trace(go.Scatter(
            x=x_range,
            y=kde_values,
            mode='lines',
            name=f'{self._get_signal_name(signal_name)} KDE',
            line=dict(color=color, width=2),
            showlegend=False,
        ))

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        self.render(data)

    def add_normal_fit(self, signal_name: str) -> None:
        """
        Add normal distribution fit overlay.

        Args:
            signal_name: Signal to fit.
        """
        if not self.figure or self._data is None:
            return

        if signal_name not in self._data.columns:
            return

        values = self._data[signal_name].dropna().values
        mu, std = values.mean(), values.std()

        x_range = np.linspace(values.min(), values.max(), 200)
        normal_pdf = stats.norm.pdf(x_range, mu, std)

        self.figure.add_trace(go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode='lines',
            name=f'Normal fit (μ={mu:.2f}, σ={std:.2f})',
            line=dict(color='#e74c3c', width=2, dash='dash'),
        ))

    def add_statistics_annotation(self, signal_name: str) -> None:
        """
        Add statistics annotation box.

        Args:
            signal_name: Signal for statistics.
        """
        if not self.figure or self._data is None:
            return

        if signal_name not in self._data.columns:
            return

        values = self._data[signal_name].dropna()

        stats_text = (
            f"<b>Statistics</b><br>"
            f"Mean: {values.mean():.4f}<br>"
            f"Std: {values.std():.4f}<br>"
            f"Min: {values.min():.4f}<br>"
            f"Max: {values.max():.4f}<br>"
            f"Median: {values.median():.4f}"
        )

        self.figure.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            text=stats_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#ccc",
            borderwidth=1,
        )

    def _get_signal_name(self, path: str) -> str:
        """Extract display name from signal path."""
        parts = path.split(".")
        return parts[-1] if parts else path

