"""
Statistics visualization.

Provides statistical summary visualizations including box plots,
violin plots, and correlation matrices.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import List, Optional, Dict, Any

from ..base import BasePlot
from ...core.types import PlotConfig


class StatisticsPlot(BasePlot):
    """
    Statistical visualization plot.

    Supports multiple visualization types for statistical analysis.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize StatisticsPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.stat_type = config.get("stat_type", "boxplot")

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render statistics plot.

        Args:
            data: DataFrame with signal columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data

        if self.stat_type == "boxplot":
            return self._render_boxplot(data)
        elif self.stat_type == "violin":
            return self._render_violin(data)
        elif self.stat_type == "correlation":
            return self._render_correlation(data)
        else:
            return self._render_boxplot(data)

    def _render_boxplot(self, data: pd.DataFrame) -> go.Figure:
        """Render box plot."""
        self.figure = go.Figure()

        signals = self.config.get("signals", [])
        timestamp_col = self.config.get("x_axis", "timestamp")

        if not signals:
            signals = [
                col for col in data.columns
                if col != timestamp_col and data[col].dtype in [np.float64, np.int64]
            ][:10]

        colors = self._get_color_sequence()

        for i, signal in enumerate(signals):
            if signal not in data.columns:
                continue

            values = data[signal].dropna().values

            self.figure.add_trace(go.Box(
                y=values,
                name=self._get_signal_name(signal),
                marker_color=colors[i % len(colors)],
                boxpoints='outliers',
            ))

        layout = self._create_base_layout()
        layout.update({
            "yaxis_title": "Value",
            "showlegend": False,
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def _render_violin(self, data: pd.DataFrame) -> go.Figure:
        """Render violin plot."""
        self.figure = go.Figure()

        signals = self.config.get("signals", [])
        timestamp_col = self.config.get("x_axis", "timestamp")

        if not signals:
            signals = [
                col for col in data.columns
                if col != timestamp_col and data[col].dtype in [np.float64, np.int64]
            ][:10]

        colors = self._get_color_sequence()

        for i, signal in enumerate(signals):
            if signal not in data.columns:
                continue

            values = data[signal].dropna().values

            self.figure.add_trace(go.Violin(
                y=values,
                name=self._get_signal_name(signal),
                fillcolor=colors[i % len(colors)],
                line_color=colors[i % len(colors)],
                box_visible=True,
                meanline_visible=True,
            ))

        layout = self._create_base_layout()
        layout.update({
            "yaxis_title": "Value",
            "showlegend": False,
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def _render_correlation(self, data: pd.DataFrame) -> go.Figure:
        """Render correlation matrix heatmap."""
        signals = self.config.get("signals", [])
        timestamp_col = self.config.get("x_axis", "timestamp")

        if not signals:
            signals = [
                col for col in data.columns
                if col != timestamp_col and data[col].dtype in [np.float64, np.int64]
            ][:15]

        # Calculate correlation matrix
        subset = data[signals].dropna()
        corr_matrix = subset.corr()

        # Create heatmap
        self.figure = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[self._get_signal_name(s) for s in signals],
            y=[self._get_signal_name(s) for s in signals],
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate=(
                "%{x} vs %{y}<br>"
                "Correlation: %{z:.3f}<br>"
                "<extra></extra>"
            ),
        ))

        # Add annotations
        for i, row in enumerate(corr_matrix.values):
            for j, val in enumerate(row):
                self.figure.add_annotation(
                    x=j,
                    y=i,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(val) > 0.5 else "black",
                        size=10
                    ),
                )

        layout = self._create_base_layout()
        layout.update({
            "title": self.title or "Correlation Matrix",
            "xaxis": {"tickangle": 45},
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        self.render(data)

    def get_statistics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary for all signals.

        Returns:
            Dict mapping signal names to statistics dicts.
        """
        if self._data is None:
            return {}

        summary = {}
        signals = self.config.get("signals", list(self._data.columns))

        for signal in signals:
            if signal not in self._data.columns:
                continue

            values = self._data[signal].dropna()

            summary[signal] = {
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "25%": float(values.quantile(0.25)),
                "50%": float(values.quantile(0.50)),
                "75%": float(values.quantile(0.75)),
                "max": float(values.max()),
                "skewness": float(values.skew()),
                "kurtosis": float(values.kurtosis()),
            }

        return summary

    def _get_signal_name(self, path: str) -> str:
        """Extract display name from signal path."""
        parts = path.split(".")
        return parts[-1] if parts else path

