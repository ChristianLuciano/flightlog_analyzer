"""
Interpolation utility functions.

Provides various interpolation methods for signal alignment.
"""

import numpy as np
from scipy import interpolate
from typing import Optional, Tuple


def linear_interp(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Linear interpolation.

    Args:
        x_new: New x values.
        x: Original x values.
        y: Original y values.
        fill_value: Value for extrapolation.

    Returns:
        Interpolated y values.
    """
    return np.interp(x_new, x, y, left=fill_value, right=fill_value)


def spline_interp(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    kind: str = 'cubic'
) -> np.ndarray:
    """
    Spline interpolation.

    Args:
        x_new: New x values.
        x: Original x values.
        y: Original y values.
        kind: Spline type ('linear', 'cubic', 'quadratic').

    Returns:
        Interpolated y values.
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 4:
        return linear_interp(x_new, x_clean, y_clean)

    try:
        f = interpolate.interp1d(
            x_clean, y_clean,
            kind=kind,
            bounds_error=False,
            fill_value=np.nan
        )
        return f(x_new)
    except Exception:
        return linear_interp(x_new, x_clean, y_clean)


def resample_signal(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_rate: float,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample signal to target rate.

    Args:
        timestamps: Original timestamps.
        values: Original values.
        target_rate: Target sample rate in Hz.
        method: Interpolation method.

    Returns:
        Tuple of (new_timestamps, new_values).
    """
    if len(timestamps) < 2:
        return timestamps.copy(), values.copy()

    t_start = timestamps[0]
    t_end = timestamps[-1]
    dt = 1.0 / target_rate

    new_timestamps = np.arange(t_start, t_end, dt)

    if method == 'linear':
        new_values = linear_interp(new_timestamps, timestamps, values)
    elif method in ['cubic', 'spline']:
        new_values = spline_interp(new_timestamps, timestamps, values)
    else:
        new_values = linear_interp(new_timestamps, timestamps, values)

    return new_timestamps, new_values


def nearest_interp(
    x_new: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Nearest neighbor interpolation.

    Args:
        x_new: New x values.
        x: Original x values.
        y: Original y values.

    Returns:
        Interpolated y values.
    """
    indices = np.searchsorted(x, x_new)
    indices = np.clip(indices, 0, len(x) - 1)

    # Check which neighbor is closer
    left_dist = np.abs(x_new - x[np.maximum(indices - 1, 0)])
    right_dist = np.abs(x_new - x[indices])
    use_left = (indices > 0) & (left_dist < right_dist)
    indices = np.where(use_left, indices - 1, indices)

    result = y[indices]

    # Mask values outside range
    outside = (x_new < x[0]) | (x_new > x[-1])
    result[outside] = np.nan

    return result

