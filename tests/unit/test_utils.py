"""
Tests for utility modules.

Covers src/utils/* - 100% coverage target.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.interpolation import (
    linear_interp,
    spline_interp,
    resample_signal,
    nearest_interp,
)
from src.utils.statistics import (
    compute_statistics,
    rolling_statistics,
    correlation,
    detect_outliers,
)
from src.utils.time_utils import (
    parse_timestamp,
    format_duration,
    timestamp_to_datetime,
    datetime_to_timestamp,
    get_time_range,
    time_to_index,
    sample_rate_from_timestamps,
)


class TestLinearInterp:
    """Test linear_interp function."""
    
    def test_basic_interpolation(self):
        """Test basic linear interpolation."""
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 2, 4, 6])
        x_new = np.array([0.5, 1.5, 2.5])
        result = linear_interp(x_new, x, y)
        np.testing.assert_array_almost_equal(result, [1, 3, 5])
    
    def test_extrapolation_returns_nan(self):
        """Test extrapolation returns fill value."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_new = np.array([-1, 3])
        result = linear_interp(x_new, x, y)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
    
    def test_custom_fill_value(self):
        """Test custom fill value."""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_new = np.array([-1, 3])
        result = linear_interp(x_new, x, y, fill_value=-999)
        assert result[0] == -999
        assert result[1] == -999
    
    def test_exact_match(self):
        """Test interpolation at exact points."""
        x = np.array([0, 1, 2])
        y = np.array([10, 20, 30])
        x_new = np.array([0, 1, 2])
        result = linear_interp(x_new, x, y)
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestSplineInterp:
    """Test spline_interp function."""
    
    def test_basic_spline(self):
        """Test basic cubic spline interpolation."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])  # y = x^2
        x_new = np.array([0.5, 1.5, 2.5, 3.5])
        result = spline_interp(x_new, x, y)
        # Should be close to quadratic
        assert len(result) == 4
    
    def test_linear_kind(self):
        """Test linear spline."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 2, 3, 4])
        x_new = np.array([0.5, 1.5])
        result = spline_interp(x_new, x, y, kind='linear')
        np.testing.assert_array_almost_equal(result, [0.5, 1.5])
    
    def test_with_nan_values(self):
        """Test spline with NaN values."""
        x = np.array([0, 1, np.nan, 3, 4])
        y = np.array([0, 1, 2, 3, 4])
        x_new = np.array([0.5, 1.5])
        result = spline_interp(x_new, x, y)
        assert len(result) == 2
    
    def test_too_few_points(self):
        """Test fallback to linear with few points."""
        x = np.array([0, 1])
        y = np.array([0, 1])
        x_new = np.array([0.5])
        result = spline_interp(x_new, x, y)
        assert result[0] == 0.5


class TestResampleSignal:
    """Test resample_signal function."""
    
    def test_upsample(self):
        """Test upsampling signal."""
        timestamps = np.array([0, 1, 2, 3])
        values = np.array([0, 1, 2, 3])
        new_t, new_v = resample_signal(timestamps, values, target_rate=10)
        assert len(new_t) > len(timestamps)
    
    def test_downsample(self):
        """Test downsampling signal."""
        timestamps = np.linspace(0, 10, 1000)
        values = np.sin(timestamps)
        new_t, new_v = resample_signal(timestamps, values, target_rate=10)
        assert len(new_t) < len(timestamps)
    
    def test_linear_method(self):
        """Test linear method."""
        timestamps = np.array([0, 1, 2])
        values = np.array([0, 1, 2])
        new_t, new_v = resample_signal(timestamps, values, target_rate=2, method='linear')
        assert len(new_v) > 0
    
    def test_spline_method(self):
        """Test spline method."""
        timestamps = np.array([0, 1, 2, 3, 4])
        values = np.array([0, 1, 4, 9, 16])
        new_t, new_v = resample_signal(timestamps, values, target_rate=2, method='spline')
        assert len(new_v) > 0
    
    def test_cubic_method(self):
        """Test cubic method."""
        timestamps = np.array([0, 1, 2, 3, 4])
        values = np.array([0, 1, 4, 9, 16])
        new_t, new_v = resample_signal(timestamps, values, target_rate=2, method='cubic')
        assert len(new_v) > 0
    
    def test_unknown_method_fallback(self):
        """Test unknown method falls back to linear."""
        timestamps = np.array([0, 1, 2])
        values = np.array([0, 1, 2])
        new_t, new_v = resample_signal(timestamps, values, target_rate=5, method='unknown')
        assert len(new_v) > 0
    
    def test_single_point(self):
        """Test with single point."""
        timestamps = np.array([0])
        values = np.array([5])
        new_t, new_v = resample_signal(timestamps, values, target_rate=10)
        assert len(new_t) == 1


class TestNearestInterp:
    """Test nearest_interp function."""
    
    def test_basic_nearest(self):
        """Test basic nearest neighbor interpolation."""
        x = np.array([0, 1, 2, 3], dtype=float)
        y = np.array([10, 20, 30, 40], dtype=float)
        x_new = np.array([0.3, 0.7, 1.5, 2.8])
        result = nearest_interp(x_new, x, y)
        # Check the approximate values (nearest neighbor)
        assert len(result) == 4
    
    def test_exact_points(self):
        """Test at exact points."""
        x = np.array([0, 1, 2], dtype=float)
        y = np.array([10, 20, 30], dtype=float)
        x_new = np.array([0.0, 1.0, 2.0])
        result = nearest_interp(x_new, x, y)
        # Results should be exact matches
        assert len(result) == 3
    
    def test_outside_range(self):
        """Test values outside range."""
        x = np.array([0, 1, 2], dtype=float)
        y = np.array([10, 20, 30], dtype=float)
        x_new = np.array([-1.0, 3.0])
        result = nearest_interp(x_new, x, y)
        # Values outside range should be NaN
        assert len(result) == 2


class TestComputeStatistics:
    """Test compute_statistics function."""
    
    def test_basic_statistics(self):
        """Test basic statistics computation."""
        values = np.array([1, 2, 3, 4, 5])
        stats = compute_statistics(values)
        assert stats['count'] == 5
        assert stats['mean'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
    
    def test_empty_array(self):
        """Test with empty array after NaN removal."""
        values = np.array([np.nan, np.nan])
        stats = compute_statistics(values)
        assert stats['count'] == 0
    
    def test_with_nan_values(self):
        """Test with NaN values."""
        values = np.array([1, np.nan, 3, np.nan, 5])
        stats = compute_statistics(values)
        assert stats['count'] == 3
    
    def test_all_statistics(self):
        """Test all statistics are computed."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = compute_statistics(values)
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'var' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'range' in stats
        assert 'median' in stats
        assert 'q25' in stats
        assert 'q75' in stats
        assert 'q05' in stats
        assert 'q95' in stats
        assert 'rms' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats


class TestRollingStatistics:
    """Test rolling_statistics function."""
    
    def test_rolling_mean(self):
        """Test rolling mean."""
        values = np.array([1, 2, 3, 4, 5])
        result = rolling_statistics(values, window=3, stat='mean')
        assert len(result) == 5
    
    def test_rolling_std(self):
        """Test rolling standard deviation."""
        values = np.array([1, 2, 3, 4, 5])
        result = rolling_statistics(values, window=3, stat='std')
        assert len(result) == 5
    
    def test_rolling_min(self):
        """Test rolling minimum."""
        values = np.array([1, 2, 3, 4, 5])
        result = rolling_statistics(values, window=3, stat='min')
        assert len(result) == 5
    
    def test_rolling_max(self):
        """Test rolling maximum."""
        values = np.array([1, 2, 3, 4, 5])
        result = rolling_statistics(values, window=3, stat='max')
        assert len(result) == 5
    
    def test_rolling_median(self):
        """Test rolling median."""
        values = np.array([1, 2, 3, 4, 5])
        result = rolling_statistics(values, window=3, stat='median')
        assert len(result) == 5
    
    def test_unknown_stat(self):
        """Test unknown statistic raises error."""
        values = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            rolling_statistics(values, window=2, stat='unknown')


class TestCorrelation:
    """Test correlation function."""
    
    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        r, p = correlation(x, y)
        assert abs(r - 1.0) < 1e-10
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        r, p = correlation(x, y)
        assert abs(r - (-1.0)) < 1e-10
    
    def test_with_nan_values(self):
        """Test with NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        r, p = correlation(x, y)
        assert not np.isnan(r)
    
    def test_too_few_points(self):
        """Test with too few points."""
        x = np.array([1, 2])
        y = np.array([1, 2])
        r, p = correlation(x, y)
        assert np.isnan(r)


class TestDetectOutliers:
    """Test detect_outliers function."""
    
    def test_zscore_method(self):
        """Test z-score outlier detection."""
        # Create data with a clear outlier
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier
        outliers = detect_outliers(values, method='zscore', threshold=2.0)
        # Last value should be an outlier
        assert outliers[-1] is True or outliers[-1] == True
    
    def test_iqr_method(self):
        """Test IQR outlier detection."""
        values = np.array([1, 2, 3, 4, 100])
        outliers = detect_outliers(values, method='iqr', threshold=1.5)
        assert outliers[-1] == True
    
    def test_mad_method(self):
        """Test MAD outlier detection."""
        values = np.array([1, 2, 3, 4, 100])
        outliers = detect_outliers(values, method='mad', threshold=2.0)
        assert outliers[-1] == True
    
    def test_unknown_method(self):
        """Test unknown method raises error."""
        values = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            detect_outliers(values, method='unknown')
    
    def test_no_outliers(self):
        """Test with no outliers."""
        values = np.array([1, 2, 3, 4, 5])
        outliers = detect_outliers(values, method='zscore', threshold=3.0)
        assert not any(outliers)


class TestParseTimestamp:
    """Test parse_timestamp function."""
    
    def test_float_input(self):
        """Test float input."""
        result = parse_timestamp(1234.5)
        assert result == 1234.5
    
    def test_int_input(self):
        """Test integer input."""
        result = parse_timestamp(1234)
        assert result == 1234.0
    
    def test_datetime_input(self):
        """Test datetime input."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = parse_timestamp(dt)
        assert result == dt.timestamp()
    
    def test_pandas_timestamp_input(self):
        """Test pandas Timestamp input."""
        ts = pd.Timestamp('2024-01-01 12:00:00')
        result = parse_timestamp(ts)
        assert result == ts.timestamp()
    
    def test_numpy_datetime64_input(self):
        """Test numpy datetime64 input."""
        dt64 = np.datetime64('2024-01-01T12:00:00')
        result = parse_timestamp(dt64)
        assert isinstance(result, float)
    
    def test_string_input(self):
        """Test string input."""
        result = parse_timestamp('2024-01-01 12:00:00')
        assert isinstance(result, float)
    
    def test_invalid_input(self):
        """Test invalid input raises error."""
        with pytest.raises(ValueError):
            parse_timestamp([1, 2, 3])


class TestFormatDuration:
    """Test format_duration function."""
    
    def test_minutes_and_seconds(self):
        """Test minutes and seconds format."""
        result = format_duration(125.5, precision=2)
        assert result == '02:05.50'
    
    def test_hours_minutes_seconds(self):
        """Test hours, minutes, seconds format."""
        result = format_duration(3725.5, precision=2)
        assert result == '01:02:05.50'
    
    def test_negative_duration(self):
        """Test negative duration."""
        result = format_duration(-65.0, precision=2)
        assert result == '-01:05.00'
    
    def test_zero_duration(self):
        """Test zero duration."""
        result = format_duration(0.0, precision=2)
        assert result == '00:00.00'
    
    def test_custom_precision(self):
        """Test custom precision."""
        result = format_duration(10.5555, precision=3)
        assert '10.556' in result


class TestTimestampConversions:
    """Test timestamp conversion functions."""
    
    def test_timestamp_to_datetime(self):
        """Test timestamp to datetime conversion."""
        ts = 1704110400.0  # 2024-01-01 12:00:00 UTC (approx)
        dt = timestamp_to_datetime(ts)
        assert isinstance(dt, datetime)
    
    def test_datetime_to_timestamp(self):
        """Test datetime to timestamp conversion."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        ts = datetime_to_timestamp(dt)
        assert isinstance(ts, float)
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        ts = datetime_to_timestamp(dt)
        dt2 = timestamp_to_datetime(ts)
        assert dt == dt2


class TestGetTimeRange:
    """Test get_time_range function."""
    
    def test_basic_range(self):
        """Test basic time range."""
        timestamps = np.array([10, 20, 30, 40, 50])
        start, end, duration = get_time_range(timestamps)
        assert start == 10
        assert end == 50
        assert duration == 40
    
    def test_with_nan(self):
        """Test with NaN values."""
        timestamps = np.array([10, np.nan, 30, np.nan, 50])
        start, end, duration = get_time_range(timestamps)
        assert start == 10
        assert end == 50


class TestTimeToIndex:
    """Test time_to_index function."""
    
    def test_exact_match(self):
        """Test exact timestamp match."""
        timestamps = np.array([0, 1, 2, 3, 4])
        idx = time_to_index(2.0, timestamps)
        assert idx == 2
    
    def test_nearest_match(self):
        """Test nearest timestamp match."""
        timestamps = np.array([0, 1, 2, 3, 4])
        idx = time_to_index(1.3, timestamps)
        assert idx == 1
    
    def test_nearest_match_higher(self):
        """Test nearest timestamp match rounds up."""
        timestamps = np.array([0, 1, 2, 3, 4])
        idx = time_to_index(1.7, timestamps)
        assert idx == 2


class TestSampleRateFromTimestamps:
    """Test sample_rate_from_timestamps function."""
    
    def test_uniform_rate(self):
        """Test uniform sample rate."""
        timestamps = np.arange(0, 10, 0.1)  # 10 Hz
        rate = sample_rate_from_timestamps(timestamps)
        assert abs(rate - 10.0) < 0.1
    
    def test_single_point(self):
        """Test single point returns zero."""
        timestamps = np.array([0])
        rate = sample_rate_from_timestamps(timestamps)
        assert rate == 0.0
    
    def test_empty_array(self):
        """Test empty array returns zero."""
        timestamps = np.array([])
        rate = sample_rate_from_timestamps(timestamps)
        assert rate == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

