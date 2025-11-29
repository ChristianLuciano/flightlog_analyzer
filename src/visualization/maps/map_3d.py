"""
3D Map visualization.

Provides 3D geographic visualization with altitude dimension
and terrain rendering.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from ..base import BasePlot
from ...core.types import PlotConfig
from ...data.downsampling import douglas_peucker


class Map3D(BasePlot):
    """
    3D map for flight path visualization with altitude.

    Displays flight path in 3D space with altitude dimension
    and optional terrain overlay.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize Map3D.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.lat_column = config.get("lat_column", "lat")
        self.lon_column = config.get("lon_column", "lon")
        self.alt_column = config.get("alt_column", "altitude")
        self.max_path_points = config.get("max_path_points", 3000)

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render 3D map with flight path.

        Args:
            data: DataFrame with lat/lon/alt columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data
        self.figure = go.Figure()

        required_cols = [self.lat_column, self.lon_column, self.alt_column]
        if not all(col in data.columns for col in required_cols):
            return self.figure

        lat = data[self.lat_column].values
        lon = data[self.lon_column].values
        alt = data[self.alt_column].values

        # Remove NaN values
        valid_mask = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(alt))
        lat = lat[valid_mask]
        lon = lon[valid_mask]
        alt = alt[valid_mask]

        if len(lat) == 0:
            return self.figure

        # Convert to local coordinates for better 3D display
        lat_center = (lat.min() + lat.max()) / 2
        lon_center = (lon.min() + lon.max()) / 2

        # Convert to meters from center
        x = (lon - lon_center) * 111320 * np.cos(np.radians(lat_center))
        y = (lat - lat_center) * 110540
        z = alt

        # Simplify path if needed
        if len(x) > self.max_path_points:
            # Simplified 2D, then use same indices for z
            indices = np.linspace(0, len(x) - 1, self.max_path_points).astype(int)
            x = x[indices]
            y = y[indices]
            z = z[indices]

        # Get color mapping
        color_by = self.config.get("color_by", "altitude")
        if color_by == "altitude":
            color_data = z
            colorbar_title = "Altitude (m)"
        elif color_by in data.columns:
            color_data = data[color_by].values[valid_mask]
            if len(color_data) > len(x):
                indices = np.linspace(0, len(color_data) - 1, len(x)).astype(int)
                color_data = color_data[indices]
            colorbar_title = color_by
        else:
            color_data = z
            colorbar_title = "Altitude (m)"

        # Add 3D flight path
        self.figure.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(
                color=color_data,
                colorscale=self.config.get("colorscale", "Viridis"),
                width=4,
                colorbar=dict(title=colorbar_title),
            ),
            name='Flight Path',
            hovertemplate=(
                "X: %{x:.0f}m<br>"
                "Y: %{y:.0f}m<br>"
                "Alt: %{z:.0f}m<br>"
                "<extra></extra>"
            ),
        ))

        # Add ground projection
        if self.config.get("show_ground_projection", True):
            self.figure.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=np.zeros_like(z),
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=2),
                name='Ground Track',
                showlegend=True,
            ))

        # Add vertical lines to ground
        if self.config.get("show_altitude_lines", False):
            for i in range(0, len(x), max(1, len(x) // 20)):
                self.figure.add_trace(go.Scatter3d(
                    x=[x[i], x[i]],
                    y=[y[i], y[i]],
                    z=[0, z[i]],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    showlegend=False,
                ))

        # Add start/end markers
        self._add_markers(x, y, z)

        # Layout
        self.figure.update_layout(
            scene=dict(
                xaxis_title='East (m)',
                yaxis_title='North (m)',
                zaxis_title='Altitude (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            title=self.title or "3D Flight Path",
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return self.figure

    def _add_markers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> None:
        """Add start and end markers."""
        # Start marker
        self.figure.add_trace(go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode='markers+text',
            marker=dict(size=10, color='#2ecc71', symbol='circle'),
            text=['START'],
            name='Start',
        ))

        # End marker
        self.figure.add_trace(go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode='markers+text',
            marker=dict(size=10, color='#e74c3c', symbol='circle'),
            text=['END'],
            name='End',
        ))

    def add_current_position(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> None:
        """Add current position marker in 3D."""
        if not self.figure or self._data is None:
            return

        # Convert to local coordinates
        lat_center = (self._data[self.lat_column].min() +
                     self._data[self.lat_column].max()) / 2
        lon_center = (self._data[self.lon_column].min() +
                     self._data[self.lon_column].max()) / 2

        x = (lon - lon_center) * 111320 * np.cos(np.radians(lat_center))
        y = (lat - lat_center) * 110540

        # Remove previous current position if exists
        self.figure.data = [
            trace for trace in self.figure.data
            if trace.name != 'Current'
        ]

        self.figure.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[alt],
            mode='markers',
            marker=dict(size=15, color='#3498db', symbol='diamond'),
            name='Current',
        ))

    def update(self, data: pd.DataFrame) -> None:
        """Update plot with new data."""
        self.render(data)

    def set_camera_position(
        self,
        eye_x: float,
        eye_y: float,
        eye_z: float
    ) -> None:
        """
        Set 3D camera position.

        Args:
            eye_x: Camera X position.
            eye_y: Camera Y position.
            eye_z: Camera Z position.
        """
        if self.figure:
            self.figure.update_layout(
                scene_camera=dict(
                    eye=dict(x=eye_x, y=eye_y, z=eye_z)
                )
            )

    def set_follow_mode(self, enabled: bool = True) -> None:
        """Enable/disable camera follow mode."""
        self._follow_mode = enabled

