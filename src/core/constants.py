"""
Application-wide constants.

Defines all constant values used throughout the application including
default settings, limits, and configuration values.
"""

# Application Info
APP_TITLE = "Flight Log Analysis Dashboard"
APP_VERSION = "0.1.0"
DEBUG_MODE = False

# Data Constants
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_PATH_DELIMITER = "."
MAX_HIERARCHY_DEPTH = 20

# Performance Thresholds
MAX_POINTS_DISPLAY = 10000  # Max points before downsampling
TARGET_FPS = 30
CACHE_SIZE_MB = 512
MAX_COMPUTATION_TIME_SEC = 30

# Downsampling
LTTB_DEFAULT_THRESHOLD = 5000
DOUGLAS_PEUCKER_EPSILON = 0.0001

# Map Defaults
DEFAULT_MAP_ZOOM = 12
DEFAULT_MAP_CENTER = {"lat": 0, "lon": 0}
DEFAULT_TILE_PROVIDER = "OpenStreetMap"

# Plot Defaults
DEFAULT_PLOT_HEIGHT = 400
DEFAULT_PLOT_MARGIN = {"l": 50, "r": 30, "t": 40, "b": 40}
DEFAULT_LINE_WIDTH = 1.5

# Playback
MIN_PLAYBACK_SPEED = 0.1
MAX_PLAYBACK_SPEED = 10.0
DEFAULT_PLAYBACK_SPEED = 1.0
PLAYBACK_FRAME_RATE = 30

# FFT Defaults
DEFAULT_FFT_WINDOW_SIZE = 1024
DEFAULT_FFT_OVERLAP = 0.5
DEFAULT_FFT_WINDOW_TYPE = "hanning"

# Color Schemes
COLOR_SCHEME_LIGHT = {
    "background": "#ffffff",
    "text": "#2c3e50",
    "primary": "#3498db",
    "secondary": "#2ecc71",
    "accent": "#e74c3c",
    "grid": "#ecf0f1",
}

COLOR_SCHEME_DARK = {
    "background": "#1a1a2e",
    "text": "#eaeaea",
    "primary": "#00d4ff",
    "secondary": "#00ff88",
    "accent": "#ff6b6b",
    "grid": "#2d2d44",
}

# Event Colors by Severity
EVENT_COLORS = {
    "critical": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
    "debug": "#95a5a6",
}

# Marker Styles
MARKER_STYLES = {
    "start": {"color": "#2ecc71", "symbol": "circle", "size": 12},
    "end": {"color": "#e74c3c", "symbol": "square", "size": 12},
    "current": {"color": "#3498db", "symbol": "triangle-up", "size": 14},
    "event": {"color": "#f39c12", "symbol": "diamond", "size": 10},
}

# File Extensions
SUPPORTED_EXPORT_FORMATS = {
    "image": ["png", "svg", "pdf", "jpeg"],
    "geo": ["kml", "kmz", "geojson", "gpx", "shp"],
    "data": ["csv", "xlsx", "json", "mat"],
    "config": ["yaml", "json"],
}

# Keyboard Shortcuts
KEYBOARD_SHORTCUTS = {
    "play_pause": " ",  # Space
    "step_forward": "ArrowRight",
    "step_backward": "ArrowLeft",
    "zoom_in": "+",
    "zoom_out": "-",
    "reset_zoom": "Home",
    "save": "ctrl+s",
    "open": "ctrl+o",
    "export": "ctrl+e",
}

