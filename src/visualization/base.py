"""
Base plot class.

Provides abstract base class and common functionality for all plot types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.types import PlotConfig, PlotType
from ..core.constants import DEFAULT_PLOT_HEIGHT, DEFAULT_PLOT_MARGIN, DEFAULT_LINE_WIDTH


class BasePlot(ABC):
    """
    Abstract base class for all plot types.

    Provides common interface and shared functionality for
    rendering and updating plots.
    """

    def __init__(self, config: PlotConfig):
        """
        Initialize base plot.

        Args:
            config: Plot configuration dictionary.
        """
        self.config = config
        self.figure: Optional[go.Figure] = None
        self._data: Optional[pd.DataFrame] = None
        self._theme: Dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Get plot ID."""
        return self.config.get("id", "")

    @property
    def plot_type(self) -> PlotType:
        """Get plot type."""
        type_str = self.config.get("plot_type", "TIME_SERIES")
        return PlotType[type_str.upper()]

    @property
    def title(self) -> str:
        """Get plot title."""
        return self.config.get("title", "")

    @abstractmethod
    def render(self, data: pd.DataFrame) -> go.Figure:
        """
        Render the plot with given data.

        Args:
            data: DataFrame containing signal data.

        Returns:
            Plotly Figure object.
        """
        pass

    @abstractmethod
    def update(self, data: pd.DataFrame) -> None:
        """
        Update plot with new data.

        Args:
            data: New DataFrame for the plot.
        """
        pass

    def set_theme(self, theme: Dict[str, Any]) -> None:
        """
        Set plot theme/colors.

        Args:
            theme: Theme configuration dictionary.
        """
        self._theme = theme
        if self.figure:
            self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply current theme to figure."""
        if not self.figure or not self._theme:
            return

        self.figure.update_layout(
            paper_bgcolor=self._theme.get("background", "#ffffff"),
            plot_bgcolor=self._theme.get("background", "#ffffff"),
            font_color=self._theme.get("text", "#2c3e50"),
        )

    def _create_base_layout(self) -> Dict[str, Any]:
        """Create base layout configuration."""
        return {
            "title": self.title,
            "height": self.config.get("height", DEFAULT_PLOT_HEIGHT),
            "margin": self.config.get("margin", DEFAULT_PLOT_MARGIN),
            "showlegend": True,
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            "hovermode": "x unified",
        }

    def _get_color_sequence(self) -> List[str]:
        """Get color sequence for multiple traces."""
        return [
            "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
            "#f39c12", "#1abc9c", "#e91e63", "#00bcd4",
            "#8bc34a", "#ff5722", "#607d8b", "#795548"
        ]

    def add_time_cursor(self, timestamp: float) -> None:
        """
        Add vertical time cursor line.

        Args:
            timestamp: Timestamp for cursor position.
        """
        if not self.figure:
            return

        self.figure.add_vline(
            x=timestamp,
            line_dash="dash",
            line_color=self._theme.get("accent", "#e74c3c"),
            line_width=2,
            annotation_text="",
        )

    def add_event_markers(
        self,
        events: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> None:
        """
        Add event markers to plot.

        Args:
            events: DataFrame with event data.
            timestamp_col: Name of timestamp column.
        """
        if not self.figure or events.empty:
            return

        for _, event in events.iterrows():
            self.figure.add_vline(
                x=event[timestamp_col],
                line_dash="dot",
                line_color="#f39c12",
                line_width=1,
                annotation_text=event.get("event_type", ""),
                annotation_position="top",
            )

    def set_x_range(self, x_min: float, x_max: float) -> None:
        """Set X-axis range."""
        if self.figure:
            self.figure.update_xaxes(range=[x_min, x_max])

    def set_y_range(self, y_min: float, y_max: float) -> None:
        """Set Y-axis range."""
        if self.figure:
            self.figure.update_yaxes(range=[y_min, y_max])

    def reset_zoom(self) -> None:
        """Reset zoom to auto-scale."""
        if self.figure:
            self.figure.update_xaxes(autorange=True)
            self.figure.update_yaxes(autorange=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert figure to dictionary for Dash."""
        if self.figure:
            return self.figure.to_dict()
        return {}

    def to_json(self) -> str:
        """Convert figure to JSON string."""
        if self.figure:
            return self.figure.to_json()
        return "{}"

