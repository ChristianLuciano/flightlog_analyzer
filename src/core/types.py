"""
Type definitions for the Flight Log Analysis Dashboard.

Defines TypedDict, Protocol, and other type hints used throughout
the application for better type safety and documentation.
"""

from typing import (
    TypedDict, Protocol, Union, List, Dict, Any, Optional,
    Callable, Tuple, Literal, TypeVar
)
from dataclasses import dataclass, field
from enum import Enum, auto
import pandas as pd
import numpy as np
from numpy.typing import NDArray


# Type Aliases
FlightDataDict = Dict[str, Union[pd.DataFrame, "FlightDataDict"]]
SignalPath = str  # e.g., "Sensors.IMU.Accelerometer.accel_x"
Timestamp = Union[float, int, np.datetime64, pd.Timestamp]
NumericArray = NDArray[np.floating[Any]]


class PlotType(Enum):
    """Supported plot types."""
    TIME_SERIES = auto()
    XY_SCATTER = auto()
    XY_LINE = auto()
    FFT = auto()
    PSD = auto()
    SPECTROGRAM = auto()
    HISTOGRAM = auto()
    MAP_2D = auto()
    MAP_3D = auto()
    SURFACE_3D = auto()
    SCATTER_3D = auto()
    WATERFALL = auto()


class InterpolationMethod(Enum):
    """Interpolation methods for signal alignment."""
    LINEAR = "linear"
    NEAREST = "nearest"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    SPLINE = "spline"
    ZERO = "zero"


class DownsamplingMethod(Enum):
    """Downsampling algorithms."""
    LTTB = "lttb"  # Largest Triangle Three Buckets
    M4 = "m4"      # Min-Max-Mean-Median
    DOUGLAS_PEUCKER = "douglas_peucker"
    SIMPLE = "simple"  # Every Nth point


class EventSeverity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PlaybackState(Enum):
    """Playback control states."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


# TypedDicts for structured data

class SignalMetadata(TypedDict, total=False):
    """Metadata for a signal."""
    name: str
    path: str
    unit: str
    sampling_rate: float
    data_type: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    completeness: float
    gap_count: int
    is_computed: bool
    formula: Optional[str]


class EventData(TypedDict, total=False):
    """Structure for event log entries."""
    timestamp: Timestamp
    event_type: str
    description: str
    severity: str
    category: str
    duration: float
    metadata: Dict[str, Any]


class PlotConfig(TypedDict, total=False):
    """Configuration for a plot panel."""
    id: str
    plot_type: str
    signals: List[SignalPath]
    title: str
    x_axis: str
    y_axis: str
    row: int
    col: int
    row_span: int
    col_span: int
    options: Dict[str, Any]


class GridConfig(TypedDict, total=False):
    """Grid layout configuration."""
    rows: int
    cols: int
    gap: int
    plots: List[PlotConfig]


class TabConfig(TypedDict, total=False):
    """Tab configuration."""
    id: str
    name: str
    grid: GridConfig
    active: bool


class ComputedSignalConfig(TypedDict, total=False):
    """Configuration for a computed signal."""
    name: str
    formula: str
    inputs: List[SignalPath]
    parameters: Dict[str, float]
    dataframe_path: str
    unit: str
    description: str
    sampling_strategy: str


class MapConfig(TypedDict, total=False):
    """Map visualization configuration."""
    center_lat: float
    center_lon: float
    zoom: int
    tile_provider: str
    show_path: bool
    path_color_by: Optional[str]
    show_markers: bool
    show_events: bool
    colormap: str


class DashboardConfig(TypedDict, total=False):
    """Complete dashboard configuration."""
    version: str
    tabs: List[TabConfig]
    computed_signals: Dict[str, ComputedSignalConfig]
    map_config: MapConfig
    theme: str
    timestamp_column: str
    path_delimiter: str


# Dataclasses for internal use

@dataclass
class TimeRange:
    """Represents a time range."""
    start: Timestamp
    end: Timestamp

    @property
    def duration(self) -> float:
        """Return duration in seconds."""
        return float(self.end - self.start)


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def center(self) -> Tuple[float, float]:
        """Return center point."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    size_bytes: int
    access_count: int = 0
    last_access: float = 0.0


@dataclass
class SignalInfo:
    """Complete information about a signal."""
    name: str
    path: str
    dataframe_path: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    metadata: SignalMetadata = field(default_factory=dict)
    is_computed: bool = False


# Protocols for interfaces

class DataProvider(Protocol):
    """Protocol for data providers."""

    def get_signal(self, path: SignalPath) -> pd.Series:
        """Get signal data by path."""
        ...

    def get_dataframe(self, path: str) -> pd.DataFrame:
        """Get DataFrame by path."""
        ...

    def list_signals(self) -> List[SignalInfo]:
        """List all available signals."""
        ...


class PlotRenderer(Protocol):
    """Protocol for plot renderers."""

    def render(self, data: pd.DataFrame, config: PlotConfig) -> Any:
        """Render the plot with given data and configuration."""
        ...

    def update(self, data: pd.DataFrame) -> None:
        """Update the plot with new data."""
        ...


class Exporter(Protocol):
    """Protocol for data exporters."""

    def export(self, data: Any, path: str, **options) -> None:
        """Export data to file."""
        ...

    def supported_formats(self) -> List[str]:
        """Return list of supported formats."""
        ...

