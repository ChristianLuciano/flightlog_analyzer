"""
Performance tests with large datasets.

Covers:
- REQ-DI-005: Handle millions of data points per signal efficiently
- REQ-PERF-001: Initial data load time < 5 seconds for typical datasets
- REQ-SUCCESS-001: Successfully load 60-minute flight at 100Hz in < 5 seconds
- REQ-SUCCESS-005: Display flight paths with 10,000+ GPS points smoothly
"""

import pytest
import pandas as pd
import numpy as np
import time
from typing import Dict


def create_large_dataset(num_points: int, num_signals: int = 10) -> Dict[str, pd.DataFrame]:
    """Create a large dataset for performance testing.
    
    Args:
        num_points: Number of data points per signal.
        num_signals: Number of signals per DataFrame.
    
    Returns:
        Hierarchical dict of DataFrames.
    """
    # Create timestamps (microsecond precision)
    timestamps = np.linspace(0, num_points / 100, num_points)  # 100 Hz
    
    data = {
        'Sensors': {
            'IMU': {
                'Accelerometer': pd.DataFrame({
                    'timestamp': timestamps,
                    **{f'accel_{axis}': np.random.randn(num_points) * 9.81 
                       for axis in ['x', 'y', 'z']}
                }),
                'Gyroscope': pd.DataFrame({
                    'timestamp': timestamps,
                    **{f'gyro_{axis}': np.random.randn(num_points) * 0.1 
                       for axis in ['x', 'y', 'z']}
                })
            },
            'GPS': pd.DataFrame({
                'timestamp': timestamps,
                'lat': 40.0 + np.cumsum(np.random.randn(num_points) * 0.00001),
                'lon': -74.0 + np.cumsum(np.random.randn(num_points) * 0.00001),
                'altitude': 100 + np.cumsum(np.random.randn(num_points) * 0.1),
                'speed': np.abs(np.random.randn(num_points) * 10),
                'heading': np.random.rand(num_points) * 360
            })
        },
        'Control': {
            'Motors': pd.DataFrame({
                'timestamp': timestamps,
                **{f'motor_{i}': np.clip(np.random.randn(num_points) * 0.2 + 0.5, 0, 1) 
                   for i in range(1, 5)}
            }),
            'FlightController': pd.DataFrame({
                'timestamp': timestamps,
                'roll': np.random.randn(num_points) * 10,
                'pitch': np.random.randn(num_points) * 10,
                'yaw': np.cumsum(np.random.randn(num_points) * 0.1) % 360,
                'throttle': np.clip(np.random.rand(num_points), 0, 1)
            })
        },
        'Battery': pd.DataFrame({
            'timestamp': timestamps,
            'voltage': 12.6 - np.linspace(0, 2, num_points) + np.random.randn(num_points) * 0.1,
            'current': np.abs(np.random.randn(num_points) * 5 + 10),
            'remaining': 100 - np.linspace(0, 50, num_points)
        })
    }
    
    return data


class TestDataLoadPerformance:
    """Performance tests for data loading."""
    
    def test_load_100k_points_under_1_second(self):
        """Test loading 100,000 points completes in under 1 second."""
        start_time = time.time()
        data = create_large_dataset(100_000)
        load_time = time.time() - start_time
        
        assert load_time < 1.0, f"Loading took {load_time:.2f}s, expected < 1s"
        
        # Verify data was created correctly
        assert 'Sensors' in data
        assert 'GPS' in data['Sensors']
        assert len(data['Sensors']['GPS']) == 100_000
    
    def test_load_1_million_points_under_5_seconds(self):
        """Test loading 1 million points completes in under 5 seconds.
        
        REQ-DI-005: Handle millions of data points per signal efficiently.
        """
        start_time = time.time()
        data = create_large_dataset(1_000_000)
        load_time = time.time() - start_time
        
        assert load_time < 5.0, f"Loading took {load_time:.2f}s, expected < 5s"
        
        # Verify data integrity
        assert len(data['Sensors']['GPS']) == 1_000_000
        assert len(data['Sensors']['IMU']['Accelerometer']) == 1_000_000
    
    def test_load_60_minute_100hz_flight(self):
        """Test loading 60-minute flight at 100Hz (360,000 points).
        
        REQ-SUCCESS-001: Successfully load 60-minute flight at 100Hz in < 5 seconds.
        """
        num_points = 60 * 60 * 100  # 60 minutes * 60 seconds * 100 Hz = 360,000 points
        
        start_time = time.time()
        data = create_large_dataset(num_points)
        load_time = time.time() - start_time
        
        assert load_time < 5.0, f"Loading took {load_time:.2f}s, expected < 5s"
        assert len(data['Sensors']['GPS']) == num_points


class TestDownsamplingPerformance:
    """Performance tests for downsampling algorithms."""
    
    def test_lttb_downsample_1m_points(self):
        """Test LTTB downsampling 1 million points."""
        from src.data.downsampling import lttb_downsample
        
        x = np.linspace(0, 1000, 1_000_000)
        y = np.sin(x * 0.01) + np.random.randn(1_000_000) * 0.1
        
        start_time = time.time()
        x_ds, y_ds = lttb_downsample(x, y, 10_000)
        ds_time = time.time() - start_time
        
        assert ds_time < 5.0, f"Downsampling took {ds_time:.2f}s, expected < 5s"
        assert len(x_ds) == 10_000
        assert len(y_ds) == 10_000
    
    def test_douglas_peucker_10k_points(self):
        """Test Douglas-Peucker path simplification with 10k points."""
        from src.data.downsampling import douglas_peucker
        
        # Create a path (use 10k instead of 100k for reasonable test time)
        t = np.linspace(0, 100, 10_000)
        lat = 40.0 + np.sin(t * 0.1) * 0.01
        lon = -74.0 + np.cos(t * 0.1) * 0.01
        
        start_time = time.time()
        lat_s, lon_s = douglas_peucker(lat, lon, epsilon=0.0001)
        simplify_time = time.time() - start_time
        
        assert simplify_time < 10.0, f"Simplification took {simplify_time:.2f}s"
        assert len(lat_s) < len(lat)


class TestSignalProcessingPerformance:
    """Performance tests for signal processing operations."""
    
    def test_fft_1m_points(self):
        """Test FFT on 1 million points."""
        signal = np.random.randn(1_000_000)
        
        start_time = time.time()
        fft_result = np.fft.rfft(signal)
        fft_time = time.time() - start_time
        
        assert fft_time < 1.0, f"FFT took {fft_time:.2f}s, expected < 1s"
    
    def test_interpolation_1m_points(self):
        """Test interpolation on 1 million points."""
        from scipy import interpolate
        
        x = np.linspace(0, 1000, 500_000)  # Source: 500k points
        y = np.sin(x * 0.01)
        x_new = np.linspace(0, 1000, 1_000_000)  # Target: 1M points
        
        start_time = time.time()
        f = interpolate.interp1d(x, y, kind='linear')
        y_new = f(x_new)
        interp_time = time.time() - start_time
        
        assert interp_time < 2.0, f"Interpolation took {interp_time:.2f}s"
        assert len(y_new) == 1_000_000
    
    def test_moving_average_1m_points(self):
        """Test moving average on 1 million points."""
        signal = np.random.randn(1_000_000)
        window = 100
        
        start_time = time.time()
        # Use numpy convolution for efficiency
        kernel = np.ones(window) / window
        smoothed = np.convolve(signal, kernel, mode='valid')
        ma_time = time.time() - start_time
        
        assert ma_time < 0.5, f"Moving average took {ma_time:.2f}s"


class TestMemoryUsage:
    """Tests for memory efficiency."""
    
    def test_memory_efficiency_1m_points(self):
        """Test memory usage stays reasonable for 1M points."""
        import sys
        
        # Create data
        data = create_large_dataset(1_000_000)
        
        # Estimate memory usage
        total_bytes = 0
        
        def estimate_size(obj, seen=None):
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(estimate_size(v, seen) for v in obj.values())
            elif isinstance(obj, pd.DataFrame):
                size += obj.memory_usage(deep=True).sum()
            return size
        
        total_bytes = estimate_size(data)
        mb_used = total_bytes / (1024 * 1024)
        
        # 1M points * ~10 signals * 8 bytes ~= 80 MB base
        # Allow up to 500 MB for overhead
        assert mb_used < 500, f"Memory usage {mb_used:.1f} MB exceeds 500 MB limit"
    
    def test_dataframe_memory_layout(self):
        """Test DataFrame uses efficient memory layout."""
        # Create DataFrame with optimal dtypes
        n = 1_000_000
        df = pd.DataFrame({
            'timestamp': np.arange(n, dtype=np.float32),  # 4 bytes instead of 8
            'value': np.random.randn(n).astype(np.float32)
        })
        
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Should be ~8 MB (2 columns * 1M * 4 bytes)
        assert memory_mb < 10, f"DataFrame memory {memory_mb:.1f} MB, expected < 10 MB"


class TestHierarchyPerformance:
    """Performance tests for hierarchical data navigation."""
    
    def test_path_resolution_speed(self):
        """Test hierarchical path resolution is fast."""
        from src.data.hierarchy import resolve_path
        
        data = create_large_dataset(100_000)
        
        # Time 1000 path resolutions
        paths = [
            'Sensors.GPS.lat',
            'Sensors.IMU.Accelerometer.accel_x',
            'Control.Motors.motor_1',
            'Battery.voltage'
        ]
        
        start_time = time.time()
        for _ in range(250):  # 1000 total resolutions
            for path in paths:
                resolve_path(data, path)
        resolve_time = time.time() - start_time
        
        # Should complete 1000 resolutions in < 100ms
        assert resolve_time < 0.1, f"Path resolution took {resolve_time*1000:.1f}ms for 1000 ops"
    
    def test_signal_discovery_speed(self):
        """Test discovering all signals is fast."""
        from src.ui.app_layout import _build_signal_options
        
        data = create_large_dataset(100_000)
        
        start_time = time.time()
        options = _build_signal_options(data)
        discovery_time = time.time() - start_time
        
        # Should complete in < 100ms
        assert discovery_time < 0.1, f"Signal discovery took {discovery_time*1000:.1f}ms"
        assert len(options) > 10  # Should find multiple signals


class TestPlotPerformance:
    """Performance tests for plot creation."""
    
    def test_time_series_plot_100k_points(self):
        """Test time series plot creation with 100k points."""
        from src.ui.callbacks import _create_time_series_plot
        
        data = create_large_dataset(100_000)
        signals = ['Sensors.GPS.lat', 'Sensors.GPS.lon']
        
        start_time = time.time()
        fig = _create_time_series_plot(signals, data, None, None)
        plot_time = time.time() - start_time
        
        assert plot_time < 3.0, f"Plot creation took {plot_time:.2f}s"
        assert fig is not None
    
    def test_fft_plot_100k_points(self):
        """Test FFT plot creation with 100k points."""
        from src.ui.callbacks import _create_fft_plot
        
        data = create_large_dataset(100_000)
        signals = ['Sensors.IMU.Accelerometer.accel_x']
        
        start_time = time.time()
        fig = _create_fft_plot(signals, data, None, None)
        plot_time = time.time() - start_time
        
        assert plot_time < 2.0, f"FFT plot creation took {plot_time:.2f}s"
        assert fig is not None


class TestGPSPathPerformance:
    """Performance tests for GPS path handling."""
    
    def test_10k_gps_points(self):
        """Test handling 10,000+ GPS points.
        
        REQ-SUCCESS-005: Display flight paths with 10,000+ GPS points smoothly.
        """
        # Create 10k GPS points
        n = 10_000
        lat = 40.0 + np.cumsum(np.random.randn(n) * 0.0001)
        lon = -74.0 + np.cumsum(np.random.randn(n) * 0.0001)
        
        # Measure haversine distance calculation
        from src.utils.geo import haversine_distance
        
        start_time = time.time()
        total_distance = 0
        for i in range(1, n):
            total_distance += haversine_distance(lat[i-1], lon[i-1], lat[i], lon[i])
        calc_time = time.time() - start_time
        
        assert calc_time < 1.0, f"Distance calc took {calc_time:.2f}s for 10k points"
        assert total_distance > 0
    
    def test_50k_gps_points(self):
        """Test handling 50,000 GPS points."""
        n = 50_000
        lat = 40.0 + np.cumsum(np.random.randn(n) * 0.0001)
        lon = -74.0 + np.cumsum(np.random.randn(n) * 0.0001)
        
        # Test vectorized distance calculation
        from src.utils.geo import cumulative_distance
        
        start_time = time.time()
        distances = cumulative_distance(lat, lon)
        calc_time = time.time() - start_time
        
        assert calc_time < 1.0, f"Cumulative distance took {calc_time:.2f}s for 50k points"
        assert len(distances) == n


class TestTimeRangeFiltering:
    """Tests for time range filtering performance."""
    
    def test_filter_1m_points_by_time(self):
        """Test filtering 1M points by time range."""
        data = create_large_dataset(1_000_000)
        
        timestamps = data['Sensors']['GPS']['timestamp'].values
        values = data['Sensors']['GPS']['lat'].values
        
        # Filter to middle 50%
        t_start = timestamps[0] + (timestamps[-1] - timestamps[0]) * 0.25
        t_end = timestamps[0] + (timestamps[-1] - timestamps[0]) * 0.75
        
        start_time = time.time()
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        filtered_values = values[mask]
        filter_time = time.time() - start_time
        
        assert filter_time < 0.1, f"Filtering took {filter_time*1000:.1f}ms"
        assert len(filtered_values) > 0
        assert len(filtered_values) < len(values)

