"""
Custom exceptions for the Flight Log Analysis Dashboard.

Defines all application-specific exceptions for better error handling
and more informative error messages.
"""


class FlightLogException(Exception):
    """Base exception for all Flight Log Dashboard errors."""
    pass


# Data Exceptions
class DataLoadError(FlightLogException):
    """Raised when data cannot be loaded or parsed."""
    pass


class InvalidDataStructure(FlightLogException):
    """Raised when data structure doesn't match expected format."""
    pass


class TimestampError(FlightLogException):
    """Raised when timestamp column is missing or invalid."""
    pass


class HierarchyError(FlightLogException):
    """Raised when hierarchical path resolution fails."""
    pass


class DataValidationError(FlightLogException):
    """Raised when data validation fails."""
    pass


# Computed Signal Exceptions
class ComputedSignalError(FlightLogException):
    """Base exception for computed signal errors."""
    pass


class FormulaParseError(ComputedSignalError):
    """Raised when formula parsing fails."""
    pass


class FormulaSyntaxError(ComputedSignalError):
    """Raised when formula has syntax errors."""
    pass


class CircularDependencyError(ComputedSignalError):
    """Raised when circular dependencies are detected."""
    pass


class SignalNotFoundError(ComputedSignalError):
    """Raised when a referenced signal doesn't exist."""
    pass


class ComputationError(ComputedSignalError):
    """Raised when signal computation fails at runtime."""
    pass


class ComputationTimeoutError(ComputedSignalError):
    """Raised when computation exceeds time limit."""
    pass


# Visualization Exceptions
class VisualizationError(FlightLogException):
    """Base exception for visualization errors."""
    pass


class PlotConfigError(VisualizationError):
    """Raised when plot configuration is invalid."""
    pass


class RenderError(VisualizationError):
    """Raised when plot rendering fails."""
    pass


# Map Exceptions
class MapError(FlightLogException):
    """Base exception for map-related errors."""
    pass


class CoordinateError(MapError):
    """Raised when coordinates are invalid."""
    pass


class TileLoadError(MapError):
    """Raised when map tiles cannot be loaded."""
    pass


class ProjectionError(MapError):
    """Raised when coordinate projection fails."""
    pass


# Configuration Exceptions
class ConfigError(FlightLogException):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


class ConfigVersionError(ConfigError):
    """Raised when configuration version is incompatible."""
    pass


# Export Exceptions
class ExportError(FlightLogException):
    """Base exception for export errors."""
    pass


class UnsupportedFormatError(ExportError):
    """Raised when export format is not supported."""
    pass


class ExportWriteError(ExportError):
    """Raised when writing export file fails."""
    pass


# Cache Exceptions
class CacheError(FlightLogException):
    """Base exception for cache errors."""
    pass


class CacheMissError(CacheError):
    """Raised when cache lookup fails."""
    pass


class CacheFullError(CacheError):
    """Raised when cache capacity is exceeded."""
    pass

