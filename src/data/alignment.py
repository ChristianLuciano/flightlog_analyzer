"""
Time alignment for cross-DataFrame signals.

Provides interpolation and alignment utilities for combining
signals from different DataFrames with different sampling rates.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Any, Tuple, Optional, Union
from scipy import interpolate


class AlignmentMethod(Enum):
    """Interpolation methods for time alignment."""
    LINEAR = 'linear'
    NEAREST = 'nearest'
    FORWARD_FILL = 'ffill'
    BACKWARD_FILL = 'bfill'
    SPLINE = 'spline'
    ZERO = 'zero'


class TimeAligner:
    """
    Aligns signals from different DataFrames to a common timebase.
    
    Supports multiple interpolation methods and handles edge cases
    like gaps and out-of-range values.
    """
    
    def __init__(
        self,
        method: AlignmentMethod = AlignmentMethod.LINEAR,
        fill_value: Optional[float] = np.nan
    ):
        """
        Initialize TimeAligner.
        
        Args:
            method: Interpolation method to use.
            fill_value: Value to use for out-of-range points.
        """
        self.method = method
        self.fill_value = fill_value
    
    def align(
        self,
        source: pd.DataFrame,
        target_times: np.ndarray,
        value_column: str,
        timestamp_column: str = 'timestamp'
    ) -> np.ndarray:
        """
        Align a signal to target timestamps.
        
        Args:
            source: Source DataFrame with signal data.
            target_times: Target timestamps to interpolate to.
            value_column: Name of the value column to align.
            timestamp_column: Name of the timestamp column.
            
        Returns:
            Aligned signal values at target timestamps.
        """
        if source.empty:
            return np.full(len(target_times), np.nan)
        
        source_times = source[timestamp_column].values
        source_values = source[value_column].values
        
        # Remove NaN timestamps
        valid_mask = ~np.isnan(source_times)
        source_times = source_times[valid_mask]
        source_values = source_values[valid_mask]
        
        if len(source_times) == 0:
            return np.full(len(target_times), np.nan)
        
        if len(source_times) == 1:
            # Single point: use nearest
            return self._align_single_point(source_times[0], source_values[0], target_times)
        
        if self.method == AlignmentMethod.LINEAR:
            return self._linear_interpolate(source_times, source_values, target_times)
        elif self.method == AlignmentMethod.NEAREST:
            return self._nearest_interpolate(source_times, source_values, target_times)
        elif self.method == AlignmentMethod.FORWARD_FILL:
            return self._forward_fill(source_times, source_values, target_times)
        elif self.method == AlignmentMethod.BACKWARD_FILL:
            return self._backward_fill(source_times, source_values, target_times)
        elif self.method == AlignmentMethod.SPLINE:
            return self._spline_interpolate(source_times, source_values, target_times)
        else:
            return self._linear_interpolate(source_times, source_values, target_times)
    
    def _linear_interpolate(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Linear interpolation."""
        return np.interp(
            target_times,
            source_times,
            source_values,
            left=self.fill_value,
            right=self.fill_value
        )
    
    def _nearest_interpolate(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        interp_func = interpolate.interp1d(
            source_times,
            source_values,
            kind='nearest',
            bounds_error=False,
            fill_value=self.fill_value
        )
        return interp_func(target_times)
    
    def _forward_fill(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Forward fill (step) interpolation."""
        result = np.full(len(target_times), self.fill_value)
        
        for i, t in enumerate(target_times):
            # Find the last source time <= target time
            mask = source_times <= t
            if np.any(mask):
                idx = np.where(mask)[0][-1]
                result[i] = source_values[idx]
        
        return result
    
    def _backward_fill(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Backward fill interpolation."""
        result = np.full(len(target_times), self.fill_value)
        
        for i, t in enumerate(target_times):
            # Find the first source time >= target time
            mask = source_times >= t
            if np.any(mask):
                idx = np.where(mask)[0][0]
                result[i] = source_values[idx]
        
        return result
    
    def _spline_interpolate(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Cubic spline interpolation."""
        if len(source_times) < 4:
            return self._linear_interpolate(source_times, source_values, target_times)
        
        try:
            spline = interpolate.CubicSpline(source_times, source_values)
            result = spline(target_times)
            
            # Clip to source range
            mask = (target_times < source_times[0]) | (target_times > source_times[-1])
            result[mask] = self.fill_value
            
            return result
        except Exception:
            return self._linear_interpolate(source_times, source_values, target_times)
    
    def _align_single_point(
        self,
        source_time: float,
        source_value: float,
        target_times: np.ndarray
    ) -> np.ndarray:
        """Handle single point case."""
        result = np.full(len(target_times), self.fill_value)
        
        # Find closest target time
        idx = np.argmin(np.abs(target_times - source_time))
        result[idx] = source_value
        
        return result


def align_signals(
    signals: Dict[str, Tuple[pd.DataFrame, str]],
    target_times: np.ndarray,
    method: AlignmentMethod = AlignmentMethod.LINEAR,
    tolerance: Optional[float] = None,
    timestamp_column: str = 'timestamp'
) -> Dict[str, np.ndarray]:
    """
    Align multiple signals to a common timebase.
    
    Args:
        signals: Dict mapping signal names to (DataFrame, column_name) tuples.
        target_times: Target timestamps to align to.
        method: Interpolation method.
        tolerance: Time tolerance for snapping.
        timestamp_column: Name of timestamp column.
        
    Returns:
        Dict mapping signal names to aligned arrays.
    """
    aligner = TimeAligner(method=method)
    result = {}
    
    for name, (df, column) in signals.items():
        if tolerance is not None:
            # Snap target times to nearest source times within tolerance
            source_times = df[timestamp_column].values
            snapped_times = _snap_to_nearest(target_times, source_times, tolerance)
            result[name] = aligner.align(df, snapped_times, column, timestamp_column)
        else:
            result[name] = aligner.align(df, target_times, column, timestamp_column)
    
    return result


def _snap_to_nearest(
    target_times: np.ndarray,
    source_times: np.ndarray,
    tolerance: float
) -> np.ndarray:
    """Snap target times to nearest source times within tolerance."""
    result = target_times.copy()
    
    for i, t in enumerate(target_times):
        distances = np.abs(source_times - t)
        min_dist = np.min(distances)
        
        if min_dist <= tolerance:
            result[i] = source_times[np.argmin(distances)]
    
    return result
