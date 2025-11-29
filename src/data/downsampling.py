"""
Downsampling algorithms for efficient data visualization.

Implements LTTB (Largest Triangle Three Buckets), M4, and
Douglas-Peucker algorithms for reducing data points while
preserving visual fidelity.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from numba import jit
import logging

from ..core.types import DownsamplingMethod
from ..core.constants import LTTB_DEFAULT_THRESHOLD, DOUGLAS_PEUCKER_EPSILON

logger = logging.getLogger(__name__)


def downsample(
    x: np.ndarray,
    y: np.ndarray,
    target_points: int,
    method: DownsamplingMethod = DownsamplingMethod.LTTB
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample data using specified method.

    Args:
        x: X-axis values (typically timestamps).
        y: Y-axis values (signal data).
        target_points: Target number of points.
        method: Downsampling algorithm to use.

    Returns:
        Tuple of (downsampled_x, downsampled_y).
    """
    if len(x) <= target_points:
        return x, y

    if method == DownsamplingMethod.LTTB:
        return lttb_downsample(x, y, target_points)
    elif method == DownsamplingMethod.M4:
        return m4_downsample(x, y, target_points)
    elif method == DownsamplingMethod.SIMPLE:
        return simple_downsample(x, y, target_points)
    else:
        return lttb_downsample(x, y, target_points)


def lttb_downsample(
    x: np.ndarray,
    y: np.ndarray,
    threshold: int = LTTB_DEFAULT_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets downsampling.

    Preserves visual appearance by selecting points that maximize
    triangle area with neighboring selected points.

    Args:
        x: X-axis values.
        y: Y-axis values.
        threshold: Target number of points.

    Returns:
        Tuple of downsampled (x, y) arrays.
    """
    data_length = len(x)

    if threshold >= data_length or threshold <= 2:
        return x.copy(), y.copy()

    # Bucket size
    bucket_size = (data_length - 2) / (threshold - 2)

    # Always include first and last points
    sampled_indices = [0]

    # Previous selected point
    a = 0

    for i in range(threshold - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1

        if bucket_end > data_length - 1:
            bucket_end = data_length - 1

        # Calculate average point of next bucket
        next_bucket_start = bucket_end
        next_bucket_end = int((i + 3) * bucket_size) + 1

        if next_bucket_end > data_length - 1:
            next_bucket_end = data_length - 1

        if next_bucket_end > next_bucket_start:
            avg_x = np.mean(x[next_bucket_start:next_bucket_end])
            avg_y = np.mean(y[next_bucket_start:next_bucket_end])
        else:
            avg_x = x[data_length - 1]
            avg_y = y[data_length - 1]

        # Find point in current bucket with max triangle area
        max_area = -1
        max_area_point = bucket_start

        for j in range(bucket_start, bucket_end):
            # Calculate triangle area
            area = abs(
                (x[a] - avg_x) * (y[j] - y[a]) -
                (x[a] - x[j]) * (avg_y - y[a])
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = j

        sampled_indices.append(max_area_point)
        a = max_area_point

    # Always include last point
    sampled_indices.append(data_length - 1)

    indices = np.array(sampled_indices)
    return x[indices], y[indices]


def m4_downsample(
    x: np.ndarray,
    y: np.ndarray,
    target_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M4 downsampling (Min, Max, First, Last per bucket).

    Good for preserving extremes in the data.

    Args:
        x: X-axis values.
        y: Y-axis values.
        target_points: Target number of points.

    Returns:
        Tuple of downsampled (x, y) arrays.
    """
    data_length = len(x)

    if target_points >= data_length:
        return x.copy(), y.copy()

    # Number of buckets (each bucket produces up to 4 points)
    n_buckets = target_points // 4
    if n_buckets < 1:
        n_buckets = 1

    bucket_size = data_length // n_buckets

    sampled_x = []
    sampled_y = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, data_length)

        if start >= end:
            continue

        bucket_x = x[start:end]
        bucket_y = y[start:end]

        # First point
        sampled_x.append(bucket_x[0])
        sampled_y.append(bucket_y[0])

        if len(bucket_y) > 1:
            # Min point
            min_idx = np.argmin(bucket_y)
            if min_idx != 0 and min_idx != len(bucket_y) - 1:
                sampled_x.append(bucket_x[min_idx])
                sampled_y.append(bucket_y[min_idx])

            # Max point
            max_idx = np.argmax(bucket_y)
            if max_idx != 0 and max_idx != len(bucket_y) - 1 and max_idx != min_idx:
                sampled_x.append(bucket_x[max_idx])
                sampled_y.append(bucket_y[max_idx])

            # Last point
            sampled_x.append(bucket_x[-1])
            sampled_y.append(bucket_y[-1])

    # Sort by x values
    indices = np.argsort(sampled_x)
    return np.array(sampled_x)[indices], np.array(sampled_y)[indices]


def simple_downsample(
    x: np.ndarray,
    y: np.ndarray,
    target_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple every-Nth-point downsampling.

    Args:
        x: X-axis values.
        y: Y-axis values.
        target_points: Target number of points.

    Returns:
        Tuple of downsampled (x, y) arrays.
    """
    if len(x) <= target_points:
        return x.copy(), y.copy()

    step = len(x) // target_points
    indices = np.arange(0, len(x), step)

    # Always include last point
    if indices[-1] != len(x) - 1:
        indices = np.append(indices, len(x) - 1)

    return x[indices], y[indices]


def douglas_peucker(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = DOUGLAS_PEUCKER_EPSILON
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Douglas-Peucker line simplification algorithm.

    Good for geographic paths where preserving shape is important.

    Args:
        x: X-axis values.
        y: Y-axis values.
        epsilon: Maximum perpendicular distance threshold.

    Returns:
        Tuple of simplified (x, y) arrays.
    """
    if len(x) <= 2:
        return x.copy(), y.copy()

    # Convert to points array
    points = np.column_stack((x, y))

    # Run Douglas-Peucker
    mask = _douglas_peucker_recursive(points, 0, len(points) - 1, epsilon)

    # Always include endpoints
    mask[0] = True
    mask[-1] = True

    return x[mask], y[mask]


def _douglas_peucker_recursive(
    points: np.ndarray,
    start: int,
    end: int,
    epsilon: float
) -> np.ndarray:
    """Recursive Douglas-Peucker implementation."""
    mask = np.zeros(len(points), dtype=bool)

    if end - start <= 1:
        return mask

    # Find point with maximum distance from line
    line_vec = points[end] - points[start]
    line_len = np.sqrt(np.sum(line_vec ** 2))

    if line_len < 1e-10:
        return mask

    line_unit = line_vec / line_len

    max_dist = 0
    max_idx = start

    for i in range(start + 1, end):
        # Calculate perpendicular distance
        point_vec = points[i] - points[start]
        proj_length = np.dot(point_vec, line_unit)
        proj_point = points[start] + proj_length * line_unit
        dist = np.sqrt(np.sum((points[i] - proj_point) ** 2))

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # If max distance exceeds epsilon, recursively simplify
    if max_dist > epsilon:
        mask[max_idx] = True
        mask |= _douglas_peucker_recursive(points, start, max_idx, epsilon)
        mask |= _douglas_peucker_recursive(points, max_idx, end, epsilon)

    return mask


def adaptive_downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    preserve_peaks: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive downsampling based on data characteristics.

    Chooses optimal algorithm based on data properties.

    Args:
        x: X-axis values.
        y: Y-axis values.
        max_points: Maximum number of points.
        preserve_peaks: Whether to preserve local extrema.

    Returns:
        Tuple of downsampled (x, y) arrays.
    """
    if len(x) <= max_points:
        return x.copy(), y.copy()

    # Calculate data variability
    y_range = np.ptp(y)
    y_std = np.std(y)

    if y_range < 1e-10:
        # Constant signal - simple downsample
        return simple_downsample(x, y, max_points)

    # Calculate signal-to-noise ratio approximation
    snr = y_range / (y_std + 1e-10)

    if preserve_peaks and snr > 5:
        # High SNR with peaks - use M4 to preserve extremes
        return m4_downsample(x, y, max_points)
    else:
        # General case - use LTTB
        return lttb_downsample(x, y, max_points)

