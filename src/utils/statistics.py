"""
Statistical utility functions.

Provides statistical calculations for signal analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a signal.

    Args:
        values: Signal values.

    Returns:
        Dictionary of statistics.
    """
    values = np.asarray(values)
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {'count': 0}

    return {
        'count': len(valid),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'var': float(np.var(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'range': float(np.ptp(valid)),
        'median': float(np.median(valid)),
        'q25': float(np.percentile(valid, 25)),
        'q75': float(np.percentile(valid, 75)),
        'q05': float(np.percentile(valid, 5)),
        'q95': float(np.percentile(valid, 95)),
        'rms': float(np.sqrt(np.mean(valid ** 2))),
        'skewness': float(stats.skew(valid)),
        'kurtosis': float(stats.kurtosis(valid)),
    }


def rolling_statistics(
    values: np.ndarray,
    window: int,
    stat: str = 'mean'
) -> np.ndarray:
    """
    Compute rolling statistics.

    Args:
        values: Signal values.
        window: Window size.
        stat: Statistic to compute ('mean', 'std', 'min', 'max', 'median').

    Returns:
        Rolling statistic array.
    """
    from scipy.ndimage import uniform_filter1d

    values = np.asarray(values, dtype=float)

    if stat == 'mean':
        return uniform_filter1d(values, size=window, mode='nearest')
    elif stat == 'std':
        mean = uniform_filter1d(values, size=window, mode='nearest')
        sq_mean = uniform_filter1d(values ** 2, size=window, mode='nearest')
        return np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    elif stat == 'min':
        from scipy.ndimage import minimum_filter1d
        return minimum_filter1d(values, size=window, mode='nearest')
    elif stat == 'max':
        from scipy.ndimage import maximum_filter1d
        return maximum_filter1d(values, size=window, mode='nearest')
    elif stat == 'median':
        from scipy.ndimage import median_filter
        return median_filter(values, size=window, mode='nearest')
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Args:
        x, y: Input arrays.

    Returns:
        Tuple of (correlation, p-value).
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return np.nan, np.nan

    r, p = stats.pearsonr(x_clean, y_clean)
    return float(r), float(p)


def detect_outliers(
    values: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.

    Args:
        values: Signal values.
        method: Detection method ('zscore', 'iqr', 'mad').
        threshold: Detection threshold.

    Returns:
        Boolean mask where True indicates outlier.
    """
    values = np.asarray(values, dtype=float)

    if method == 'zscore':
        mean = np.nanmean(values)
        std = np.nanstd(values)
        z_scores = np.abs((values - mean) / (std + 1e-10))
        return z_scores > threshold

    elif method == 'iqr':
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (values < lower) | (values > upper)

    elif method == 'mad':
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        modified_z = 0.6745 * (values - median) / (mad + 1e-10)
        return np.abs(modified_z) > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

