"""
X-Y plot implementation.

Provides scatter and line plots for visualizing relationships
between two signals.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from ..base import BasePlot
from ...core.types import PlotConfig, PlotType
from ...data.downsampling import lttb_downsample


class XYPlot(BasePlot):
    """
    X-Y scatter/line plot for signal relationships.

    Visualizes correlation between two signals with optional
    color coding by a third variable.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize XYPlot.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.max_display_points = config.get("max_points", 10000)

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render X-Y plot.

        Args:
            data: DataFrame with signal columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data
        self.figure = go.Figure()

        # Get X and Y signals
        x_signal = self.config.get("x_axis", "")
        y_signal = self.config.get("y_axis", "")
        color_signal = self.config.get("color_by", None)

        if x_signal not in data.columns or y_signal not in data.columns:
            return self.figure

        x_data = data[x_signal].values
        y_data = data[y_signal].values

        # Downsample if needed
        if len(x_data) > self.max_display_points:
            x_data, y_data = lttb_downsample(
                x_data, y_data, self.max_display_points
            )

        # Determine plot mode
        plot_type_str = self.config.get("plot_type", "XY_SCATTER")
        if plot_type_str == "XY_LINE":
            mode = "lines"
        else:
            mode = "markers"

        # Create trace
        trace_kwargs = {
            "x": x_data,
            "y": y_data,
            "mode": mode,
            "name": f"{y_signal} vs {x_signal}",
            "hovertemplate": (
                f"<b>{x_signal}</b>: %{{x:.4f}}<br>"
                f"<b>{y_signal}</b>: %{{y:.4f}}<br>"
                "<extra></extra>"
            ),
        }

        # Add color mapping if specified
        if color_signal and color_signal in data.columns:
            color_data = data[color_signal].values
            if len(color_data) > self.max_display_points:
                # Need to keep color data aligned with downsampled points
                indices = np.linspace(
                    0, len(color_data) - 1, self.max_display_points
                ).astype(int)
                color_data = color_data[indices]

            trace_kwargs["marker"] = {
                "color": color_data,
                "colorscale": self.config.get("colorscale", "Viridis"),
                "showscale": True,
                "colorbar": {"title": color_signal},
            }
        else:
            trace_kwargs["marker"] = {
                "color": self._get_color_sequence()[0],
                "size": self.config.get("marker_size", 5),
            }

        if mode == "lines":
            trace_kwargs["line"] = {
                "color": self._get_color_sequence()[0],
                "width": self.config.get("line_width", 1.5),
            }

        self.figure.add_trace(go.Scattergl(**trace_kwargs))

        # Add layout
        layout = self._create_base_layout()
        layout.update({
            "xaxis_title": self.config.get("x_label", x_signal),
            "yaxis_title": self.config.get("y_label", y_signal),
        })

        self.figure.update_layout(**layout)
        self._apply_theme()

        return self.figure

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        if self.figure is None:
            self.render(data)
            return

        x_signal = self.config.get("x_axis", "")
        y_signal = self.config.get("y_axis", "")

        if x_signal not in data.columns or y_signal not in data.columns:
            return

        x_data = data[x_signal].values
        y_data = data[y_signal].values

        if len(x_data) > self.max_display_points:
            x_data, y_data = lttb_downsample(
                x_data, y_data, self.max_display_points
            )

        if self.figure.data:
            self.figure.data[0].x = x_data
            self.figure.data[0].y = y_data

    def add_regression_line(self, degree: int = 1) -> None:
        """
        Add polynomial regression line.

        Args:
            degree: Polynomial degree (1 for linear).
        """
        if not self.figure or not self._data is not None:
            return

        x_signal = self.config.get("x_axis", "")
        y_signal = self.config.get("y_axis", "")

        if x_signal not in self._data.columns or y_signal not in self._data.columns:
            return

        x = self._data[x_signal].values
        y = self._data[y_signal].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return

        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)

        # Generate line points
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = poly(x_line)

        # Add trace
        self.figure.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Regression (deg={degree})",
            line=dict(
                color="#e74c3c",
                dash="dash",
                width=2,
            ),
        ))

    def add_confidence_ellipse(
        self,
        confidence: float = 0.95
    ) -> None:
        """
        Add confidence ellipse around data.

        Args:
            confidence: Confidence level (0-1).
        """
        if not self.figure or self._data is None:
            return

        x_signal = self.config.get("x_axis", "")
        y_signal = self.config.get("y_axis", "")

        if x_signal not in self._data.columns or y_signal not in self._data.columns:
            return

        x = self._data[x_signal].values
        y = self._data[y_signal].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            return

        # Calculate covariance
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Chi-squared value for confidence level
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence, df=2)

        # Generate ellipse points
        theta = np.linspace(0, 2 * np.pi, 100)
        a = np.sqrt(chi2_val * eigenvalues[0])
        b = np.sqrt(chi2_val * eigenvalues[1])

        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)

        # Rotate ellipse
        rotation_matrix = eigenvectors
        rotated = rotation_matrix @ np.vstack([ellipse_x, ellipse_y])

        # Translate to center
        ellipse_x = rotated[0, :] + mean_x
        ellipse_y = rotated[1, :] + mean_y

        self.figure.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name=f"{int(confidence * 100)}% Confidence",
            line=dict(color="#9b59b6", dash="dot", width=2),
            fill="toself",
            fillcolor="rgba(155, 89, 182, 0.1)",
        ))

