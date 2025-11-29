"""
Time series plot implementation.

Provides interactive time series visualization with zoom, pan,
hover tooltips, and synchronized cursor support.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from ..base import BasePlot
from ...core.types import PlotConfig
from ...core.constants import DEFAULT_LINE_WIDTH, DEFAULT_TIMESTAMP_COLUMN
from ...data.downsampling import lttb_downsample


class TimeSeriesPlot(BasePlot):
    """
    Time series plot for signal visualization.

    Supports multiple signals, downsampling for large datasets,
    and interactive features like zoom and hover.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize TimeSeriesPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.timestamp_column = config.get("x_axis", DEFAULT_TIMESTAMP_COLUMN)
        self.max_display_points = config.get("max_points", 5000)
        self._original_data: Optional[pd.DataFrame] = None

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render time series plot.

        Args:
            data: DataFrame with timestamp and signal columns.

        Returns:
            Plotly Figure object.
        """
        self._original_data = data
        self._data = data

        # Create figure
        self.figure = go.Figure()

        # Get signal columns
        signals = self.config.get("signals", [])
        if not signals:
            # Use all columns except timestamp
            signals = [
                col for col in data.columns
                if col != self.timestamp_column
            ]

        # Get color sequence
        colors = self._get_color_sequence()

        # Add traces for each signal
        for i, signal in enumerate(signals):
            if signal not in data.columns:
                continue

            x_data = data[self.timestamp_column].values
            y_data = data[signal].values

            # Downsample if needed
            if len(x_data) > self.max_display_points:
                x_data, y_data = lttb_downsample(
                    x_data, y_data, self.max_display_points
                )

            color = colors[i % len(colors)]

            self.figure.add_trace(go.Scattergl(
                x=x_data,
                y=y_data,
                mode='lines',
                name=self._get_signal_name(signal),
                line=dict(
                    color=color,
                    width=self.config.get("line_width", DEFAULT_LINE_WIDTH)
                ),
                hovertemplate=(
                    f"<b>{signal}</b><br>"
                    "Time: %{x}<br>"
                    "Value: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            ))

        # Apply layout
        layout = self._create_base_layout()
        layout.update({
            "xaxis_title": self.config.get("x_label", "Time"),
            "yaxis_title": self.config.get("y_label", "Value"),
            "xaxis": {
                "rangeslider": {"visible": False},
                "type": "linear",
            },
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def update(self, data: pd.DataFrame) -> None:
        """
        Update plot with new data.

        Args:
            data: New DataFrame.
        """
        if self.figure is None:
            self.render(data)
            return

        self._original_data = data
        self._data = data

        signals = self.config.get("signals", [])
        if not signals:
            signals = [
                col for col in data.columns
                if col != self.timestamp_column
            ]

        # Update each trace
        for i, signal in enumerate(signals):
            if signal not in data.columns or i >= len(self.figure.data):
                continue

            x_data = data[self.timestamp_column].values
            y_data = data[signal].values

            if len(x_data) > self.max_display_points:
                x_data, y_data = lttb_downsample(
                    x_data, y_data, self.max_display_points
                )

            self.figure.data[i].x = x_data
            self.figure.data[i].y = y_data

    def add_time_range_highlight(
        self,
        start: float,
        end: float,
        color: str = "rgba(255, 200, 0, 0.2)",
        label: str = ""
    ) -> None:
        """
        Add highlighted time range.

        Args:
            start: Start timestamp.
            end: End timestamp.
            color: Fill color.
            label: Optional label.
        """
        if not self.figure:
            return

        self.figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )

    def add_horizontal_line(
        self,
        y_value: float,
        color: str = "#e74c3c",
        dash: str = "dash",
        label: str = ""
    ) -> None:
        """
        Add horizontal reference line.

        Args:
            y_value: Y-axis value.
            color: Line color.
            dash: Line dash style.
            label: Optional label.
        """
        if not self.figure:
            return

        self.figure.add_hline(
            y=y_value,
            line_dash=dash,
            line_color=color,
            annotation_text=label,
            annotation_position="right",
        )

    def add_statistics_overlay(
        self,
        signal: str,
        show_mean: bool = True,
        show_std: bool = True
    ) -> None:
        """
        Add statistical overlay (mean, std bands).

        Args:
            signal: Signal name.
            show_mean: Show mean line.
            show_std: Show std deviation bands.
        """
        if not self.figure or signal not in self._data.columns:
            return

        values = self._data[signal]
        mean_val = values.mean()
        std_val = values.std()

        if show_mean:
            self.add_horizontal_line(
                mean_val,
                color="#2ecc71",
                dash="dash",
                label=f"Mean: {mean_val:.2f}"
            )

        if show_std:
            self.add_horizontal_line(
                mean_val + std_val,
                color="#3498db",
                dash="dot",
                label=f"+1σ: {mean_val + std_val:.2f}"
            )
            self.add_horizontal_line(
                mean_val - std_val,
                color="#3498db",
                dash="dot",
                label=f"-1σ: {mean_val - std_val:.2f}"
            )

    def _get_signal_name(self, path: str) -> str:
        """Extract display name from signal path."""
        parts = path.split(".")
        return parts[-1] if parts else path

    def get_visible_range(self) -> Optional[tuple]:
        """Get current visible X-axis range."""
        if self.figure and self.figure.layout.xaxis.range:
            return tuple(self.figure.layout.xaxis.range)
        return None

