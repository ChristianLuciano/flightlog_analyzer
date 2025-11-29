"""
Built-in functions for computed signals.

Provides mathematical, signal processing, and geographic functions
for use in formula expressions.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d
from typing import Union, Optional

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, list, float]


# Mathematical functions
def sqrt(x: ArrayLike) -> np.ndarray:
    """Square root."""
    return np.sqrt(x)


def abs_val(x: ArrayLike) -> np.ndarray:
    """Absolute value."""
    return np.abs(x)


def sin(x: ArrayLike) -> np.ndarray:
    """Sine (radians)."""
    return np.sin(x)


def cos(x: ArrayLike) -> np.ndarray:
    """Cosine (radians)."""
    return np.cos(x)


def tan(x: ArrayLike) -> np.ndarray:
    """Tangent (radians)."""
    return np.tan(x)


def asin(x: ArrayLike) -> np.ndarray:
    """Arc sine."""
    return np.arcsin(x)


def acos(x: ArrayLike) -> np.ndarray:
    """Arc cosine."""
    return np.arccos(x)


def atan(x: ArrayLike) -> np.ndarray:
    """Arc tangent."""
    return np.arctan(x)


def atan2(y: ArrayLike, x: ArrayLike) -> np.ndarray:
    """Two-argument arc tangent."""
    return np.arctan2(y, x)


def exp(x: ArrayLike) -> np.ndarray:
    """Exponential."""
    return np.exp(x)


def log(x: ArrayLike) -> np.ndarray:
    """Natural logarithm."""
    return np.log(x)


def log10(x: ArrayLike) -> np.ndarray:
    """Base-10 logarithm."""
    return np.log10(x)


def log2(x: ArrayLike) -> np.ndarray:
    """Base-2 logarithm."""
    return np.log2(x)


def floor(x: ArrayLike) -> np.ndarray:
    """Floor function."""
    return np.floor(x)


def ceil(x: ArrayLike) -> np.ndarray:
    """Ceiling function."""
    return np.ceil(x)


def round_val(x: ArrayLike, decimals: int = 0) -> np.ndarray:
    """Round to decimals."""
    return np.round(x, decimals)


def clip(x: ArrayLike, x_min: float, x_max: float) -> np.ndarray:
    """Clip values to range."""
    return np.clip(x, x_min, x_max)


def degrees(x: ArrayLike) -> np.ndarray:
    """Radians to degrees."""
    return np.degrees(x)


def radians(x: ArrayLike) -> np.ndarray:
    """Degrees to radians."""
    return np.radians(x)


# Signal processing functions
def diff(x: ArrayLike) -> np.ndarray:
    """Numerical derivative (first difference)."""
    x = np.asarray(x)
    result = np.zeros_like(x)
    result[1:] = np.diff(x)
    return result


def cumsum(x: ArrayLike) -> np.ndarray:
    """Cumulative sum."""
    return np.cumsum(x)


def moving_avg(x: ArrayLike, window: int) -> np.ndarray:
    """Moving average filter."""
    return uniform_filter1d(np.asarray(x).astype(float), size=window)


def lowpass(x: ArrayLike, cutoff: float, fs: float = 1.0) -> np.ndarray:
    """Butterworth low-pass filter."""
    x = np.asarray(x)
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = scipy_signal.butter(4, min(normalized_cutoff, 0.99), btype='low')
    return scipy_signal.filtfilt(b, a, x)


def highpass(x: ArrayLike, cutoff: float, fs: float = 1.0) -> np.ndarray:
    """Butterworth high-pass filter."""
    x = np.asarray(x)
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = scipy_signal.butter(4, max(normalized_cutoff, 0.01), btype='high')
    return scipy_signal.filtfilt(b, a, x)


# Statistical functions
def mean(x: ArrayLike) -> float:
    """Mean value."""
    return float(np.nanmean(x))


def std(x: ArrayLike) -> float:
    """Standard deviation."""
    return float(np.nanstd(x))


def var(x: ArrayLike) -> float:
    """Variance."""
    return float(np.nanvar(x))


def min_val(x: ArrayLike) -> float:
    """Minimum value."""
    return float(np.nanmin(x))


def max_val(x: ArrayLike) -> float:
    """Maximum value."""
    return float(np.nanmax(x))


def median(x: ArrayLike) -> float:
    """Median value."""
    return float(np.nanmedian(x))


# Geographic functions
def haversine(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> np.ndarray:
    """Haversine distance in meters."""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def cumulative_distance(lat: ArrayLike, lon: ArrayLike) -> np.ndarray:
    """Cumulative distance along path in meters."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    distances = np.zeros(len(lat))
    if len(lat) > 1:
        segment_dist = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        distances[1:] = np.cumsum(segment_dist)
    return distances


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing between two points in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing_rad = np.arctan2(x, y)
    return float((np.degrees(bearing_rad) + 360) % 360)


# Conditional function
def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Conditional selection (like numpy.where)."""
    return np.where(condition, x, y)


# Build function registry
BUILTIN_FUNCTIONS = {
    # Math
    "sqrt": sqrt,
    "abs": abs_val,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "atan2": atan2,
    "exp": exp,
    "log": log,
    "log10": log10,
    "log2": log2,
    "floor": floor,
    "ceil": ceil,
    "round": round_val,
    "clip": clip,
    "degrees": degrees,
    "radians": radians,
    # Signal processing
    "diff": diff,
    "cumsum": cumsum,
    "moving_avg": moving_avg,
    "lowpass": lowpass,
    "highpass": highpass,
    # Statistics
    "mean": mean,
    "std": std,
    "var": var,
    "min": min_val,
    "max": max_val,
    "median": median,
    # Geographic
    "haversine": haversine,
    "cumulative_distance": cumulative_distance,
    "bearing": bearing,
    # Conditional
    "where": where,
}

