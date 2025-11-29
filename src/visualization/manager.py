"""
Plot manager.

Manages creation, updates, and synchronization of multiple plots
in the dashboard grid.
"""

from typing import Dict, List, Optional, Any, Type
import pandas as pd
import logging

from .base import BasePlot
from .plots.time_series import TimeSeriesPlot
from .plots.xy_plot import XYPlot
from .plots.fft import FFTPlot
from .plots.histogram import HistogramPlot
from .maps.map_2d import Map2D
from ..core.types import PlotConfig, PlotType, GridConfig
from ..core.exceptions import PlotConfigError

logger = logging.getLogger(__name__)


# Plot type registry
PLOT_REGISTRY: Dict[PlotType, Type[BasePlot]] = {
    PlotType.TIME_SERIES: TimeSeriesPlot,
    PlotType.XY_SCATTER: XYPlot,
    PlotType.XY_LINE: XYPlot,
    PlotType.FFT: FFTPlot,
    PlotType.HISTOGRAM: HistogramPlot,
    PlotType.MAP_2D: Map2D,
}


class PlotManager:
    """
    Manages all plots in the dashboard.

    Handles plot creation, updates, synchronization, and lifecycle.
    """

    def __init__(self):
        """Initialize PlotManager."""
        self._plots: Dict[str, BasePlot] = {}
        self._grid_config: Optional[GridConfig] = None
        self._current_time: float = 0.0
        self._time_range: tuple = (0.0, 1.0)
        self._theme: Dict[str, Any] = {}

    def create_plot(self, config: PlotConfig) -> BasePlot:
        """
        Create a new plot from configuration.

        Args:
            config: Plot configuration dictionary.

        Returns:
            Created plot instance.

        Raises:
            PlotConfigError: If plot type is not supported.
        """
        plot_type_str = config.get("plot_type", "TIME_SERIES")

        try:
            plot_type = PlotType[plot_type_str.upper()]
        except KeyError:
            raise PlotConfigError(f"Unknown plot type: {plot_type_str}")

        if plot_type not in PLOT_REGISTRY:
            raise PlotConfigError(f"Plot type not implemented: {plot_type}")

        plot_class = PLOT_REGISTRY[plot_type]
        plot = plot_class(config)
        plot.set_theme(self._theme)

        plot_id = config.get("id", f"plot_{len(self._plots)}")
        self._plots[plot_id] = plot

        logger.info(f"Created plot '{plot_id}' of type {plot_type}")
        return plot

    def get_plot(self, plot_id: str) -> Optional[BasePlot]:
        """Get plot by ID."""
        return self._plots.get(plot_id)

    def remove_plot(self, plot_id: str) -> bool:
        """
        Remove plot by ID.

        Args:
            plot_id: ID of plot to remove.

        Returns:
            True if plot was removed.
        """
        if plot_id in self._plots:
            del self._plots[plot_id]
            logger.info(f"Removed plot '{plot_id}'")
            return True
        return False

    def update_plot(self, plot_id: str, data: pd.DataFrame) -> None:
        """
        Update specific plot with new data.

        Args:
            plot_id: ID of plot to update.
            data: New data for the plot.
        """
        plot = self._plots.get(plot_id)
        if plot:
            plot.update(data)

    def update_all_plots(self, data_provider) -> None:
        """
        Update all plots with current data.

        Args:
            data_provider: Data provider instance.
        """
        for plot_id, plot in self._plots.items():
            try:
                signals = plot.config.get("signals", [])
                # Get data for signals and update
                # Implementation depends on data provider interface
            except Exception as e:
                logger.error(f"Error updating plot '{plot_id}': {e}")

    def set_time_cursor(self, timestamp: float) -> None:
        """
        Set time cursor position on all time-based plots.

        Args:
            timestamp: Current timestamp.
        """
        self._current_time = timestamp
        for plot in self._plots.values():
            if hasattr(plot, 'add_time_cursor'):
                plot.add_time_cursor(timestamp)

    def set_time_range(self, start: float, end: float) -> None:
        """
        Set visible time range on all time-based plots.

        Args:
            start: Start timestamp.
            end: End timestamp.
        """
        self._time_range = (start, end)
        for plot in self._plots.values():
            if hasattr(plot, 'set_x_range'):
                plot.set_x_range(start, end)

    def set_theme(self, theme: Dict[str, Any]) -> None:
        """
        Set theme for all plots.

        Args:
            theme: Theme configuration.
        """
        self._theme = theme
        for plot in self._plots.values():
            plot.set_theme(theme)

    def set_grid_config(self, config: GridConfig) -> None:
        """
        Set grid layout configuration.

        Args:
            config: Grid configuration.
        """
        self._grid_config = config

    def get_all_figures(self) -> Dict[str, Any]:
        """
        Get all plot figures as dictionaries.

        Returns:
            Dict mapping plot IDs to figure dicts.
        """
        return {
            plot_id: plot.to_dict()
            for plot_id, plot in self._plots.items()
        }

    def list_plots(self) -> List[str]:
        """Get list of all plot IDs."""
        return list(self._plots.keys())

    def clear_all(self) -> None:
        """Remove all plots."""
        self._plots.clear()
        logger.info("Cleared all plots")

    def reset_zoom_all(self) -> None:
        """Reset zoom on all plots."""
        for plot in self._plots.values():
            plot.reset_zoom()

    def add_events_to_all(self, events: pd.DataFrame) -> None:
        """
        Add event markers to all plots.

        Args:
            events: Event DataFrame.
        """
        for plot in self._plots.values():
            if hasattr(plot, 'add_event_markers'):
                plot.add_event_markers(events)

    def get_plot_count(self) -> int:
        """Get total number of plots."""
        return len(self._plots)

