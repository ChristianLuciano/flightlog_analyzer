"""
Sample data generation utilities.

Provides functions to generate realistic flight log data for testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def generate_circular_flight(
    duration: float = 300,
    sample_rate: float = 10,
    center_lat: float = 37.7749,
    center_lon: float = -122.4194,
    radius_m: float = 500,
    altitude_m: float = 100
) -> Dict[str, Any]:
    """
    Generate data for a circular flight pattern.

    Args:
        duration: Flight duration in seconds.
        sample_rate: Sample rate in Hz.
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_m: Circle radius in meters.
        altitude_m: Base altitude in meters.

    Returns:
        Hierarchical flight data dictionary.
    """
    n = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n)

    # Convert radius to degrees (approximate)
    radius_deg = radius_m / 111320

    # Circular path
    t = np.linspace(0, 4 * np.pi, n)
    lat = center_lat + radius_deg * np.cos(t)
    lon = center_lon + radius_deg * np.sin(t) / np.cos(np.radians(center_lat))
    altitude = altitude_m + 20 * np.sin(t / 2)

    return _build_flight_structure(timestamps, lat, lon, altitude, t)


def generate_figure_eight_flight(
    duration: float = 300,
    sample_rate: float = 10,
    center_lat: float = 37.7749,
    center_lon: float = -122.4194,
    size_m: float = 300,
    altitude_m: float = 100
) -> Dict[str, Any]:
    """
    Generate data for a figure-8 flight pattern.

    Args:
        duration: Flight duration in seconds.
        sample_rate: Sample rate in Hz.
        center_lat: Center latitude.
        center_lon: Center longitude.
        size_m: Pattern size in meters.
        altitude_m: Base altitude in meters.

    Returns:
        Hierarchical flight data dictionary.
    """
    n = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n)

    size_deg = size_m / 111320
    t = np.linspace(0, 4 * np.pi, n)

    # Figure-8 (lemniscate)
    lat = center_lat + size_deg * np.sin(t)
    lon = center_lon + size_deg * np.sin(t) * np.cos(t) / np.cos(np.radians(center_lat))
    altitude = altitude_m + 30 * np.sin(t)

    return _build_flight_structure(timestamps, lat, lon, altitude, t)


def generate_survey_pattern(
    duration: float = 600,
    sample_rate: float = 10,
    start_lat: float = 37.7749,
    start_lon: float = -122.4194,
    width_m: float = 500,
    height_m: float = 400,
    altitude_m: float = 50
) -> Dict[str, Any]:
    """
    Generate data for a lawn-mower survey pattern.

    Args:
        duration: Flight duration in seconds.
        sample_rate: Sample rate in Hz.
        start_lat: Starting latitude.
        start_lon: Starting longitude.
        width_m: Survey width in meters.
        height_m: Survey height in meters.
        altitude_m: Survey altitude in meters.

    Returns:
        Hierarchical flight data dictionary.
    """
    n = int(duration * sample_rate)
    timestamps = np.linspace(0, duration, n)

    width_deg = width_m / 111320
    height_deg = height_m / 110540

    # Create lawn-mower pattern
    num_passes = 8
    t_normalized = timestamps / duration

    lat = np.zeros(n)
    lon = np.zeros(n)

    for i in range(n):
        pass_num = int(t_normalized[i] * num_passes)
        pass_progress = (t_normalized[i] * num_passes) % 1

        lon[i] = start_lon + (pass_num / num_passes) * width_deg
        if pass_num % 2 == 0:
            lat[i] = start_lat + pass_progress * height_deg
        else:
            lat[i] = start_lat + (1 - pass_progress) * height_deg

    altitude = np.full(n, altitude_m) + np.random.randn(n) * 0.5
    t = timestamps / 10  # For heading calculation

    return _build_flight_structure(timestamps, lat, lon, altitude, t)


def _build_flight_structure(
    timestamps: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    altitude: np.ndarray,
    phase: np.ndarray
) -> Dict[str, Any]:
    """Build hierarchical flight data structure."""
    n = len(timestamps)

    return {
        'Sensors': {
            'IMU': {
                'Accelerometer': pd.DataFrame({
                    'timestamp': timestamps,
                    'accel_x': np.random.randn(n) * 0.1,
                    'accel_y': np.random.randn(n) * 0.1,
                    'accel_z': 9.81 + np.random.randn(n) * 0.1,
                }),
                'Gyroscope': pd.DataFrame({
                    'timestamp': timestamps,
                    'gyro_x': np.random.randn(n) * 0.02,
                    'gyro_y': np.random.randn(n) * 0.02,
                    'gyro_z': np.gradient(phase) + np.random.randn(n) * 0.01,
                }),
            },
            'GPS': pd.DataFrame({
                'timestamp': timestamps,
                'lat': lat + np.random.randn(n) * 0.00001,
                'lon': lon + np.random.randn(n) * 0.00001,
                'altitude': altitude + np.random.randn(n) * 0.5,
            }),
        },
        'Control': {
            'FlightController': pd.DataFrame({
                'timestamp': timestamps,
                'roll': np.sin(phase) * 5 + np.random.randn(n) * 0.5,
                'pitch': np.cos(phase) * 3 + np.random.randn(n) * 0.5,
                'yaw': np.degrees(phase) % 360,
            }),
        },
        'Battery': pd.DataFrame({
            'timestamp': timestamps,
            'voltage': 16.8 - timestamps * 0.002,
            'current': 5 + np.abs(np.sin(phase)) * 2,
        }),
    }

