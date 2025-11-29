"""
Time utility functions.

Provides timestamp parsing, formatting, and conversion.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional


def parse_timestamp(
    value: Union[str, float, int, datetime, np.datetime64, pd.Timestamp]
) -> float:
    """
    Parse various timestamp formats to float seconds.

    Args:
        value: Timestamp in various formats.

    Returns:
        Timestamp as float seconds since epoch.
    """
    if isinstance(value, (float, int)):
        return float(value)
    elif isinstance(value, datetime):
        return value.timestamp()
    elif isinstance(value, pd.Timestamp):
        return value.timestamp()
    elif isinstance(value, np.datetime64):
        return float(pd.Timestamp(value).timestamp())
    elif isinstance(value, str):
        return pd.Timestamp(value).timestamp()
    else:
        raise ValueError(f"Cannot parse timestamp: {type(value)}")


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.
        precision: Decimal places for seconds.

    Returns:
        Formatted string like "05:23.45".
    """
    if seconds < 0:
        sign = "-"
        seconds = abs(seconds)
    else:
        sign = ""

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{sign}{hours:02d}:{minutes:02d}:{secs:0{3 + precision}.{precision}f}"
    else:
        return f"{sign}{minutes:02d}:{secs:0{3 + precision}.{precision}f}"


def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert timestamp to datetime object."""
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to timestamp."""
    return dt.timestamp()


def get_time_range(
    timestamps: np.ndarray
) -> tuple:
    """
    Get time range from timestamp array.

    Returns:
        Tuple of (start, end, duration).
    """
    start = float(np.nanmin(timestamps))
    end = float(np.nanmax(timestamps))
    duration = end - start
    return start, end, duration


def time_to_index(
    timestamp: float,
    timestamps: np.ndarray
) -> int:
    """
    Find nearest index for timestamp.

    Args:
        timestamp: Target timestamp.
        timestamps: Array of timestamps.

    Returns:
        Index of nearest timestamp.
    """
    return int(np.argmin(np.abs(timestamps - timestamp)))


def sample_rate_from_timestamps(timestamps: np.ndarray) -> float:
    """
    Estimate sample rate from timestamps.

    Args:
        timestamps: Array of timestamps.

    Returns:
        Estimated sample rate in Hz.
    """
    if len(timestamps) < 2:
        return 0.0

    diffs = np.diff(timestamps)
    median_dt = np.median(diffs)

    if median_dt > 0:
        return 1.0 / median_dt
    return 0.0

