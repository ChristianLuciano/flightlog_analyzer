"""
Basic usage example for Flight Log Analysis Dashboard.

Demonstrates how to load flight data and launch the dashboard.
"""

import pandas as pd
import numpy as np

# Import the dashboard components
from src.core.app import create_app, run_app
from src.data.loader import load_flight_data
from src.config.settings import Settings


def create_sample_flight_data():
    """
    Create sample flight data for demonstration.

    Returns:
        Hierarchical dictionary containing DataFrames.
    """
    # Generate timestamps for a 5-minute flight
    n_samples = 3000
    duration = 300  # seconds
    timestamps = np.linspace(0, duration, n_samples)

    # Generate circular flight path
    t = np.linspace(0, 4 * np.pi, n_samples)
    center_lat = 37.7749
    center_lon = -122.4194
    radius = 0.005  # degrees

    lat = center_lat + radius * np.cos(t)
    lon = center_lon + radius * np.sin(t)
    altitude = 100 + 50 * np.sin(t / 2) + np.random.randn(n_samples) * 2

    # Create hierarchical data structure
    flight_data = {
        'Sensors': {
            'IMU': {
                'Accelerometer': pd.DataFrame({
                    'timestamp': timestamps,
                    'accel_x': np.random.randn(n_samples) * 0.1,
                    'accel_y': np.random.randn(n_samples) * 0.1,
                    'accel_z': np.random.randn(n_samples) * 0.1 + 9.81,
                }),
                'Gyroscope': pd.DataFrame({
                    'timestamp': timestamps,
                    'gyro_x': np.random.randn(n_samples) * 0.02,
                    'gyro_y': np.random.randn(n_samples) * 0.02,
                    'gyro_z': np.sin(t) * 0.1,  # Turning rate
                }),
            },
            'GPS': pd.DataFrame({
                'timestamp': timestamps,
                'lat': lat,
                'lon': lon,
                'altitude': altitude,
                'ground_speed': 10 + np.random.randn(n_samples) * 0.5,
                'heading': np.degrees(np.arctan2(np.cos(t), np.sin(t))) % 360,
            }),
            'Barometer': pd.DataFrame({
                'timestamp': timestamps,
                'pressure': 101325 - altitude * 12,
                'temperature': 20 + np.random.randn(n_samples) * 0.5,
            }),
        },
        'Control': {
            'FlightController': pd.DataFrame({
                'timestamp': timestamps,
                'roll': np.sin(t) * 10 + np.random.randn(n_samples) * 1,
                'pitch': np.cos(t) * 5 + np.random.randn(n_samples) * 1,
                'yaw': np.degrees(t) % 360,
                'throttle': 0.6 + np.sin(t / 2) * 0.1,
            }),
            'Motors': pd.DataFrame({
                'timestamp': timestamps,
                'motor1': 1500 + np.random.randn(n_samples) * 50,
                'motor2': 1500 + np.random.randn(n_samples) * 50,
                'motor3': 1500 + np.random.randn(n_samples) * 50,
                'motor4': 1500 + np.random.randn(n_samples) * 50,
            }),
        },
        'Battery': pd.DataFrame({
            'timestamp': timestamps,
            'voltage': 16.8 - timestamps * 0.003,  # Slowly decreasing
            'current': 5 + np.random.randn(n_samples) * 0.5,
            'remaining': 100 - timestamps / 3,  # Percentage
        }),
        'Events': pd.DataFrame({
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
        }),
    }

    return flight_data


def main():
    """Main function to run the dashboard."""
    print("Creating sample flight data...")
    flight_data = create_sample_flight_data()

    print("Loading data...")
    data_loader = load_flight_data(flight_data)

    print(f"Loaded {len(data_loader.list_dataframes())} DataFrames")
    print(f"Found {len(data_loader.list_signals())} signals")

    # Display some info
    print("\nDataFrames:")
    for path in data_loader.list_dataframes():
        df = data_loader.get_dataframe(path)
        print(f"  - {path}: {len(df)} rows, {len(df.columns)} columns")

    # Create and run the app
    print("\nStarting dashboard...")
    settings = Settings(theme='dark')
    app = create_app(settings=settings, flight_data=flight_data)

    print("Dashboard running at http://127.0.0.1:8050")
    run_app(app, debug=True)


if __name__ == '__main__':
    main()

