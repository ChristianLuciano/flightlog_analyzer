"""
Tests for time alignment functionality.

Tests REQ-VIS-013 through REQ-VIS-020: Cross-DataFrame time alignment.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.alignment import TimeAligner, align_signals, AlignmentMethod


class TestTimeAligner:
    """Tests for TimeAligner class."""

    def test_linear_interpolation(self):
        """Test linear interpolation (REQ-VIS-016)."""
        aligner = TimeAligner(method=AlignmentMethod.LINEAR)
        
        # Source data at 1 Hz
        source = pd.DataFrame({
            'timestamp': [0, 1, 2, 3, 4],
            'value': [0, 10, 20, 30, 40],
        })
        
        # Target timestamps at 2 Hz
        target_times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        
        result = aligner.align(source, target_times, 'value')
        
        # Check interpolated values
        assert result[0] == 0
        assert result[1] == 5  # Midpoint between 0 and 10
        assert result[2] == 10
        assert result[3] == 15  # Midpoint between 10 and 20

    def test_nearest_neighbor_interpolation(self):
        """Test nearest neighbor interpolation (REQ-VIS-016)."""
        aligner = TimeAligner(method=AlignmentMethod.NEAREST)
        
        source = pd.DataFrame({
            'timestamp': [0, 1, 2, 3],
            'value': [0, 10, 20, 30],
        })
        
        target_times = np.array([0.4, 0.6, 1.4, 1.6])
        
        result = aligner.align(source, target_times, 'value')
        
        # Nearest neighbor should snap to closest
        assert result[0] == 0   # Closer to 0
        assert result[1] == 10  # Closer to 1
        assert result[2] == 10  # Closer to 1
        assert result[3] == 20  # Closer to 2

    def test_forward_fill(self):
        """Test forward fill interpolation (REQ-VIS-016)."""
        aligner = TimeAligner(method=AlignmentMethod.FORWARD_FILL)
        
        source = pd.DataFrame({
            'timestamp': [0, 2, 4],
            'value': [0, 20, 40],
        })
        
        target_times = np.array([0, 1, 2, 3, 4])
        
        result = aligner.align(source, target_times, 'value')
        
        assert result[0] == 0
        assert result[1] == 0   # Forward fill from 0
        assert result[2] == 20
        assert result[3] == 20  # Forward fill from 20
        assert result[4] == 40

    def test_backward_fill(self):
        """Test backward fill interpolation (REQ-VIS-016)."""
        aligner = TimeAligner(method=AlignmentMethod.BACKWARD_FILL)
        
        source = pd.DataFrame({
            'timestamp': [0, 2, 4],
            'value': [0, 20, 40],
        })
        
        target_times = np.array([0, 1, 2, 3, 4])
        
        result = aligner.align(source, target_times, 'value')
        
        assert result[0] == 0
        assert result[1] == 20  # Backward fill from 20
        assert result[2] == 20
        assert result[3] == 40  # Backward fill from 40
        assert result[4] == 40

    def test_different_sampling_rates(self, multi_rate_data):
        """Test alignment of signals with different sampling rates (REQ-VIS-014)."""
        aligner = TimeAligner()
        
        imu_data = multi_rate_data['IMU']
        gps_data = multi_rate_data['GPS']
        
        # Align GPS to IMU timestamps
        imu_times = imu_data['timestamp'].values
        aligned_lat = aligner.align(gps_data, imu_times, 'lat')
        
        assert len(aligned_lat) == len(imu_times)
        assert not np.any(np.isnan(aligned_lat))  # No NaN in interpolated range


class TestAlignSignals:
    """Tests for align_signals function."""

    def test_align_multiple_signals(self):
        """Test aligning multiple signals to common timebase."""
        df1 = pd.DataFrame({
            'timestamp': [0, 1, 2, 3, 4],
            'signal_a': [0, 10, 20, 30, 40],
        })
        
        df2 = pd.DataFrame({
            'timestamp': [0, 2, 4],
            'signal_b': [0, 200, 400],
        })
        
        signals = {
            'a': (df1, 'signal_a'),
            'b': (df2, 'signal_b'),
        }
        
        # Align to df1's timestamps
        aligned = align_signals(signals, df1['timestamp'].values)
        
        assert 'a' in aligned
        assert 'b' in aligned
        assert len(aligned['a']) == 5
        assert len(aligned['b']) == 5
        assert aligned['b'][1] == 100  # Interpolated

    def test_align_with_tolerance(self):
        """Test alignment with tolerance (REQ-VIS-020)."""
        df = pd.DataFrame({
            'timestamp': [0, 1.001, 2.002, 3.003],  # Slightly off
            'value': [0, 10, 20, 30],
        })
        
        target_times = np.array([0, 1, 2, 3])
        
        aligned = align_signals(
            {'signal': (df, 'value')},
            target_times,
            tolerance=0.01
        )
        
        # Should snap to nearest within tolerance
        assert len(aligned['signal']) == 4


class TestEdgeCases:
    """Tests for edge cases in alignment."""

    def test_align_with_nan_values(self, data_with_nan):
        """Test alignment handles NaN values gracefully."""
        aligner = TimeAligner()
        
        target_times = np.linspace(0, 10, 50)
        result = aligner.align(data_with_nan, target_times, 'value')
        
        # Result should exist but may contain NaN where source had NaN
        assert len(result) == 50

    def test_align_extrapolation_bounds(self):
        """Test alignment doesn't extrapolate beyond bounds."""
        aligner = TimeAligner()
        
        source = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'value': [10, 20, 30],
        })
        
        target_times = np.array([0, 1, 2, 3, 4])  # Extends beyond source
        
        result = aligner.align(source, target_times, 'value')
        
        # Values outside range should be NaN or clamped
        assert np.isnan(result[0]) or result[0] == 10  # Before range
        assert np.isnan(result[4]) or result[4] == 30  # After range

    def test_empty_dataframe(self):
        """Test alignment with empty DataFrame."""
        aligner = TimeAligner()
        
        source = pd.DataFrame({'timestamp': [], 'value': []})
        target_times = np.array([0, 1, 2])
        
        result = aligner.align(source, target_times, 'value')
        
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_single_point(self):
        """Test alignment with single data point."""
        aligner = TimeAligner(method=AlignmentMethod.NEAREST)
        
        source = pd.DataFrame({
            'timestamp': [5],
            'value': [100],
        })
        
        target_times = np.array([0, 5, 10])
        
        result = aligner.align(source, target_times, 'value')
        
        assert result[1] == 100  # Exact match

