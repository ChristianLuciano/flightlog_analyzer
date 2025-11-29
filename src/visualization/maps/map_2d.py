"""
2D Map visualization.

Provides interactive 2D map display for flight paths with
various basemap options and overlay capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple

from ..base import BasePlot
from ...core.types import PlotConfig, MapConfig
from ...core.constants import DEFAULT_MAP_ZOOM, DEFAULT_MAP_CENTER
from ...data.downsampling import douglas_peucker


class Map2D(BasePlot):
    """
    2D interactive map for flight path visualization.

    Supports multiple basemap providers, path coloring by signal,
    markers, and various overlay options.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize Map2D.

        Args:
            config: Plot configuration.
        """
        super().__init__(config)
        self.lat_column = config.get("lat_column", "lat")
        self.lon_column = config.get("lon_column", "lon")
        self.max_path_points = config.get("max_path_points", 5000)
        self._current_position_idx = 0

    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render 2D map with flight path.

        Args:
            data: DataFrame with lat/lon columns.

        Returns:
            Plotly Figure object.
        """
        self._data = data
        self.figure = go.Figure()

        if self.lat_column not in data.columns or self.lon_column not in data.columns:
            return self.figure

        lat = data[self.lat_column].values
        lon = data[self.lon_column].values

        # Remove NaN values
        valid_mask = ~(np.isnan(lat) | np.isnan(lon))
        lat = lat[valid_mask]
        lon = lon[valid_mask]

        if len(lat) == 0:
            return self.figure

        # Simplify path if too many points
        if len(lat) > self.max_path_points:
            lat, lon = douglas_peucker(lat, lon, epsilon=0.00001)

        # Get color mapping if specified
        color_by = self.config.get("color_by")
        color_data = None
        if color_by and color_by in data.columns:
            color_data = data[color_by].values[valid_mask]
            if len(color_data) > len(lat):
                # Resample to match simplified path
                indices = np.linspace(0, len(color_data) - 1, len(lat)).astype(int)
                color_data = color_data[indices]

        # Add flight path
        self._add_flight_path(lat, lon, color_data)

        # Add markers
        self._add_start_marker(lat[0], lon[0])
        self._add_end_marker(lat[-1], lon[-1])

        # Calculate center and zoom
        center_lat = (lat.min() + lat.max()) / 2
        center_lon = (lon.min() + lon.max()) / 2
        zoom = self._calculate_zoom(lat, lon)

        # Apply layout
        self.figure.update_layout(
            map=dict(
                style=self._get_map_style(),
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            title=self.title,
            showlegend=True,
        )

        return self.figure

    def _add_flight_path(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        color_data: Optional[np.ndarray] = None
    ) -> None:
        """Add flight path trace."""
        if color_data is not None:
            # Colored path by signal value
            self.figure.add_trace(go.Scattermap(
                lat=lat,
                lon=lon,
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=color_data,
                    colorscale=self.config.get("colorscale", "Viridis"),
                    showscale=True,
                    colorbar=dict(
                        title=self.config.get("color_by", "Value"),
                        x=1.02,
                    ),
                ),
                line=dict(width=2),
                name='Flight Path',
                hovertemplate=(
                    "Lat: %{lat:.6f}<br>"
                    "Lon: %{lon:.6f}<br>"
                    "Value: %{marker.color:.2f}<br>"
                    "<extra></extra>"
                ),
            ))
        else:
            # Solid color path
            self.figure.add_trace(go.Scattermap(
                lat=lat,
                lon=lon,
                mode='lines',
                line=dict(
                    width=3,
                    color=self.config.get("path_color", "#3498db"),
                ),
                name='Flight Path',
                hovertemplate=(
                    "Lat: %{lat:.6f}<br>"
                    "Lon: %{lon:.6f}<br>"
                    "<extra></extra>"
                ),
            ))

    def _add_start_marker(self, lat: float, lon: float) -> None:
        """Add start position marker."""
        self.figure.add_trace(go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=dict(
                size=15,
                color='#2ecc71',
                symbol='circle',
            ),
            text=['START'],
            textposition='top center',
            name='Start',
            hovertemplate=(
                "<b>START</b><br>"
                "Lat: %{lat:.6f}<br>"
                "Lon: %{lon:.6f}<br>"
                "<extra></extra>"
            ),
        ))

    def _add_end_marker(self, lat: float, lon: float) -> None:
        """Add end position marker."""
        self.figure.add_trace(go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=dict(
                size=15,
                color='#e74c3c',
                symbol='circle',
            ),
            text=['END'],
            textposition='top center',
            name='End',
            hovertemplate=(
                "<b>END</b><br>"
                "Lat: %{lat:.6f}<br>"
                "Lon: %{lon:.6f}<br>"
                "<extra></extra>"
            ),
        ))

    def add_current_position(self, lat: float, lon: float, heading: float = 0) -> None:
        """
        Add/update current position marker.

        Args:
            lat: Current latitude.
            lon: Current longitude.
            heading: Current heading in degrees.
        """
        if not self.figure:
            return

        # Remove previous current position marker if exists
        self.figure.data = [
            trace for trace in self.figure.data
            if trace.name != 'Current Position'
        ]

        self.figure.add_trace(go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(
                size=20,
                color='#3498db',
                symbol='circle',
            ),
            name='Current Position',
            hovertemplate=(
                "<b>Current Position</b><br>"
                "Lat: %{lat:.6f}<br>"
                "Lon: %{lon:.6f}<br>"
                f"Heading: {heading:.1f}°<br>"
                "<extra></extra>"
            ),
        ))

    def add_event_markers(
        self,
        events: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon"
    ) -> None:
        """
        Add event markers to map.

        Args:
            events: Event DataFrame with position columns.
            lat_col: Latitude column name.
            lon_col: Longitude column name.
        """
        if not self.figure or events.empty:
            return

        if lat_col not in events.columns or lon_col not in events.columns:
            return

        self.figure.add_trace(go.Scattermap(
            lat=events[lat_col],
            lon=events[lon_col],
            mode='markers',
            marker=dict(
                size=10,
                color='#f39c12',
                symbol='star',
            ),
            text=events.get('event_type', ''),
            name='Events',
            hovertemplate=(
                "<b>Event</b><br>"
                "Type: %{text}<br>"
                "Lat: %{lat:.6f}<br>"
                "Lon: %{lon:.6f}<br>"
                "<extra></extra>"
            ),
        ))

    def update(self, data: pd.DataFrame) -> None:
        """Update map with new data."""
        self.render(data)

    def _get_map_style(self) -> str:
        """Get map style based on configuration."""
        style_map = {
            "OpenStreetMap": "open-street-map",
            "satellite": "satellite",
            "terrain": "stamen-terrain",
            "topo": "stamen-toner",
            "carto-positron": "carto-positron",
            "carto-darkmatter": "carto-darkmatter",
        }
        style = self.config.get("tile_provider", "OpenStreetMap")
        return style_map.get(style, "open-street-map")

    def _calculate_zoom(self, lat: np.ndarray, lon: np.ndarray) -> float:
        """Calculate appropriate zoom level for bounds."""
        lat_range = lat.max() - lat.min()
        lon_range = lon.max() - lon.min()
        max_range = max(lat_range, lon_range)

        if max_range < 0.01:
            return 15
        elif max_range < 0.1:
            return 12
        elif max_range < 1:
            return 9
        elif max_range < 10:
            return 6
        else:
            return 3

    def get_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get current map bounds.

        Returns:
            Tuple of (min_lat, max_lat, min_lon, max_lon) or None.
        """
        if self._data is None:
            return None

        lat = self._data[self.lat_column].dropna()
        lon = self._data[self.lon_column].dropna()

        return (lat.min(), lat.max(), lon.min(), lon.max())

