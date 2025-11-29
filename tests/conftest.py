"""
Pytest configuration and shared fixtures.

Provides common test fixtures for the Flight Log Analysis Dashboard.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any


@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps for testing."""
    return np.linspace(0, 300, 3000)  # 5 minutes, 10Hz


@pytest.fixture
def sample_gps_data(sample_timestamps):
    """Generate sample GPS data."""
    n = len(sample_timestamps)
    t = np.linspace(0, 4 * np.pi, n)
    
    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'lat': 37.7749 + 0.005 * np.cos(t),
        'lon': -122.4194 + 0.005 * np.sin(t),
        'altitude': 100 + 50 * np.sin(t / 2),
        'ground_speed': 10 + np.random.randn(n) * 0.5,
        'heading': np.degrees(np.arctan2(np.cos(t), np.sin(t))) % 360,
    })


@pytest.fixture
def sample_imu_data(sample_timestamps):
    """Generate sample IMU data."""
    n = len(sample_timestamps)
    
    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'accel_x': np.random.randn(n) * 0.1,
        'accel_y': np.random.randn(n) * 0.1,
        'accel_z': np.random.randn(n) * 0.1 + 9.81,
        'gyro_x': np.random.randn(n) * 0.02,
        'gyro_y': np.random.randn(n) * 0.02,
        'gyro_z': np.random.randn(n) * 0.02,
    })


@pytest.fixture
def sample_battery_data(sample_timestamps):
    """Generate sample battery data."""
    n = len(sample_timestamps)
    
    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'voltage': 16.8 - sample_timestamps * 0.003,
        'current': 5 + np.random.randn(n) * 0.5,
        'remaining': 100 - sample_timestamps / 3,
    })


@pytest.fixture
def sample_events_data():
    """Generate sample events data."""
    return pd.DataFrame({
        'timestamp': [0, 30, 150, 280, 299],
        'event_type': ['startup', 'takeoff', 'waypoint', 'landing', 'shutdown'],
        'description': [
            'System initialized',
            'Aircraft took off',
            'Reached waypoint 1',
            'Landing initiated',
            'System shutdown'
        ],
        'severity': ['info', 'info', 'info', 'info', 'info'],
    })


@pytest.fixture
def flat_flight_data(sample_gps_data, sample_imu_data, sample_battery_data):
    """Create flat (non-hierarchical) flight data structure."""
    return {
        'GPS': sample_gps_data,
        'IMU': sample_imu_data,
        'Battery': sample_battery_data,
    }


@pytest.fixture
def hierarchical_flight_data(
    sample_timestamps,
    sample_gps_data, 
    sample_imu_data, 
    sample_battery_data,
    sample_events_data
):
    """Create hierarchical flight data structure (REQ-DI-001, REQ-DI-002)."""
    n = len(sample_timestamps)
    t = np.linspace(0, 4 * np.pi, n)
    
    return {
        'Sensors': {
            'IMU': {
                'Accelerometer': pd.DataFrame({
                    'timestamp': sample_timestamps,
                    'accel_x': np.random.randn(n) * 0.1,
                    'accel_y': np.random.randn(n) * 0.1,
                    'accel_z': np.random.randn(n) * 0.1 + 9.81,
                }),
                'Gyroscope': pd.DataFrame({
                    'timestamp': sample_timestamps,
                    'gyro_x': np.random.randn(n) * 0.02,
                    'gyro_y': np.random.randn(n) * 0.02,
                    'gyro_z': np.sin(t) * 0.1,
                }),
            },
            'GPS': sample_gps_data.copy(),
            'Barometer': pd.DataFrame({
                'timestamp': sample_timestamps,
                'pressure': 101325 - (100 + 50 * np.sin(t / 2)) * 12,
                'temperature': 20 + np.random.randn(n) * 0.5,
            }),
        },
        'Control': {
            'FlightController': pd.DataFrame({
                'timestamp': sample_timestamps,
                'roll': np.sin(t) * 10 + np.random.randn(n) * 1,
                'pitch': np.cos(t) * 5 + np.random.randn(n) * 1,
                'yaw': np.degrees(t) % 360,
                'throttle': 0.6 + np.sin(t / 2) * 0.1,
            }),
            'Motors': pd.DataFrame({
                'timestamp': sample_timestamps,
                'motor1': 1500 + np.random.randn(n) * 50,
                'motor2': 1500 + np.random.randn(n) * 50,
                'motor3': 1500 + np.random.randn(n) * 50,
                'motor4': 1500 + np.random.randn(n) * 50,
            }),
        },
        'Battery': sample_battery_data.copy(),
        'Events': sample_events_data.copy(),
    }


@pytest.fixture
def deeply_nested_data(sample_timestamps):
    """Create deeply nested data structure for testing (REQ-DI-002)."""
    n = len(sample_timestamps)
    
    return {
        'Level1': {
            'Level2': {
                'Level3': {
                    'Level4': {
                        'Level5': {
                            'Data': pd.DataFrame({
                                'timestamp': sample_timestamps,
                                'value': np.random.randn(n),
                            })
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_flight_data(hierarchical_flight_data):
    """Alias for hierarchical_flight_data for backward compatibility."""
    return hierarchical_flight_data


@pytest.fixture
def sample_dataframe(sample_timestamps):
    """Create a simple sample DataFrame for basic tests."""
    n = len(sample_timestamps)
    t = np.linspace(0, 4 * np.pi, n)
    
    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'value': np.sin(t) + np.random.randn(n) * 0.1,
        'sin_wave': np.sin(t),
        'cos_wave': np.cos(t),
        'signal': np.sin(t) * 2,
        'accel_x': np.random.randn(n) * 0.1,
        'accel_y': np.random.randn(n) * 0.1,
        'accel_z': np.random.randn(n) * 0.1 + 9.81,
        'lat': 37.7749 + 0.005 * np.cos(t),
        'lon': -122.4194 + 0.005 * np.sin(t),
        'altitude': 100 + 50 * np.sin(t / 2),
    })


@pytest.fixture
def multi_rate_data():
    """Create data with different sampling rates (REQ-DI-020)."""
    # High rate IMU (100 Hz)
    imu_timestamps = np.linspace(0, 10, 1000)
    imu_data = pd.DataFrame({
        'timestamp': imu_timestamps,
        'accel_x': np.random.randn(1000) * 0.1,
    })
    
    # Low rate GPS (10 Hz)
    gps_timestamps = np.linspace(0, 10, 100)
    gps_data = pd.DataFrame({
        'timestamp': gps_timestamps,
        'lat': 37.7749 + np.random.randn(100) * 0.0001,
        'lon': -122.4194 + np.random.randn(100) * 0.0001,
    })
    
    # Very low rate battery (1 Hz)
    battery_timestamps = np.linspace(0, 10, 10)
    battery_data = pd.DataFrame({
        'timestamp': battery_timestamps,
        'voltage': 16.8 - battery_timestamps * 0.01,
    })
    
    return {
        'IMU': imu_data,
        'GPS': gps_data,
        'Battery': battery_data,
    }


@pytest.fixture
def data_with_gaps():
    """Create data with timestamp gaps (REQ-DI-019)."""
    # Normal data with a gap in the middle
    t1 = np.linspace(0, 50, 500)
    t2 = np.linspace(60, 100, 400)  # 10 second gap
    timestamps = np.concatenate([t1, t2])
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'value': np.random.randn(len(timestamps)),
    })


@pytest.fixture
def data_with_nan():
    """Create data with NaN values."""
    n = 100
    values = np.random.randn(n)
    values[10:15] = np.nan  # Some NaN values
    
    return pd.DataFrame({
        'timestamp': np.linspace(0, 10, n),
        'value': values,
    })


# Markers for slow tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
