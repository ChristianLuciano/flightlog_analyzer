# Flight Log Analysis Dashboard

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-910%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-74%25-yellow.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

A high-performance interactive dashboard for analyzing flight log data from pre-parsed pandas DataFrames. Designed to handle long-duration (30-60 minute), high-frequency (microsecond-level) multi-signal datasets with advanced geospatial visualization and computed signal capabilities.

![Dashboard Screenshot](docs/screenshot.png)

## Quick Install

```bash
# Clone the repository
git clone https://github.com/ChristianLuciano/flightlog_analyzer.git
cd flightlog_analyzer

# Create virtual environment
python -m venv flog
source flog/bin/activate  # On Windows: flog\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Features

### Data Handling
- **Hierarchical Data Structure**: Support for nested dictionary structures containing pandas DataFrames
- **Automatic Discovery**: Recursive traversal to discover all DataFrames and signals
- **Path-Based Access**: Access data using hierarchical paths (e.g., `Sensors.IMU.Accelerometer.accel_x`)
- **Cross-DataFrame Alignment**: Automatic time-based alignment of signals with different sampling rates
- **Folder Import**: Recursively import all CSV files from a folder structure
- **Signal Assignment**: Map raw columns to standard flight dynamics variables with conversion factors
- **Multi-Source Signals**: Group measurement, command, and estimated signals for comparison plots

### Visualization
- **Time Series Plots**: Interactive plots with zoom, pan, and linked cursors
- **FFT Analysis**: Frequency domain visualization with configurable windowing
- **Geographic Maps**: 2D flight path visualization with multiple basemap options
- **Statistical Plots**: Histograms, scatter plots, X-Y plots
- **3D Visualization**: 3D trajectory and signal plots
- **Signal Grouping**: Automatic grouping of related signals (measurement vs command vs estimated)

### User Interface
- **Collapsible Sidebar**: All sidebar sections are collapsible for more screen space
  - Data Summary
  - Quick Select
  - Signal Selector
  - Plot Builder
  - Events Filter
- **Dark Theme**: Modern dark theme optimized for long analysis sessions
- **Responsive Layout**: Adapts to different screen sizes

### Computed Signals
- **Formula-Based Computation**: Create derived signals using mathematical expressions
- **Built-in Functions**: Math, signal processing, and geographic operations
- **Dependency Resolution**: Automatic handling of signal dependencies
- **Caching**: Smart caching for performance optimization

### Performance
- **LTTB Downsampling**: Efficient visualization of large datasets (tested with millions of points)
- **Douglas-Peucker Simplification**: Path simplification for geographic data
- **Lazy Loading**: Load data on-demand for better memory usage
- **WebGL Rendering**: Hardware-accelerated graphics for smooth interactions

## Quick Start

```python
import pandas as pd
import numpy as np
from src.core.app import create_app, run_app

# Create sample data
flight_data = {
    'Sensors': {
        'GPS': pd.DataFrame({
            'timestamp': np.linspace(0, 300, 1000),
            'lat': 37.7749 + np.random.randn(1000) * 0.001,
            'lon': -122.4194 + np.random.randn(1000) * 0.001,
            'altitude': 100 + np.random.randn(1000) * 5,
        }),
        'IMU': pd.DataFrame({
            'timestamp': np.linspace(0, 300, 1000),
            'accel_x': np.random.randn(1000) * 0.1,
            'accel_y': np.random.randn(1000) * 0.1,
            'accel_z': 9.81 + np.random.randn(1000) * 0.1,
        }),
    },
}

# Create and run app
app = create_app(flight_data=flight_data)
run_app(app)
```

Then open http://127.0.0.1:8050 in your browser.

Or run the example:
```bash
python examples/basic_usage.py
```

## Folder Import

Import all CSV files from a directory structure:

```python
from src.data.folder_importer import FolderImporter, import_flight_folder

# Quick import
data = import_flight_folder("C:/Flights/Flight_001")

# Or with more control
importer = FolderImporter(extensions=['.csv', '.tsv'])
data = importer.import_folder(
    "C:/Flights/Flight_001",
    include_root_name=True,  # Include root folder in hierarchy
    flatten=False            # Keep nested structure
)

# Check import summary
summary = importer.get_summary()
print(f"Loaded: {summary['loaded_count']} files")
```

Example folder structure:
```
Flight_001/
├── CU/
│   ├── sensors/
│   │   ├── imu.csv
│   │   └── gps.csv
│   └── control.csv
├── MU/
│   └── motors.csv
└── NU/
    └── nav.csv
```

Results in:
```python
{
    'CU': {
        'sensors': {'imu': DataFrame, 'gps': DataFrame},
        'control': DataFrame
    },
    'MU': {'motors': DataFrame},
    'NU': {'nav': DataFrame}
}
```

## Signal Assignment

Map raw columns to standard flight dynamics variables with conversion factors:

```python
from src.data.signal_assignment import (
    AssignmentConfig, SignalAssigner, StandardSignal, SignalSource
)

# Create assignment configuration
config = AssignmentConfig(name="My Drone Config", version="1.0")

# Add mappings with conversion presets and source types
config.add_mapping(
    source_column='gps_lat',
    target_signal=StandardSignal.POSITION_LATITUDE,
    conversion='gps_1e7_to_degrees',
    signal_source=SignalSource.MEASUREMENT  # Raw sensor data
)

config.add_mapping(
    source_column='ekf_lat',
    target_signal=StandardSignal.POSITION_LATITUDE,
    conversion='gps_1e7_to_degrees',
    signal_source=SignalSource.ESTIMATED    # Filtered/Kalman estimate
)

config.add_mapping(
    source_column='cmd_lat',
    target_signal=StandardSignal.POSITION_LATITUDE,
    conversion='gps_1e7_to_degrees',
    signal_source=SignalSource.COMMAND      # Commanded setpoint
)

# Save for reuse
config.save("configs/my_drone.json")

# Load and apply to data
config = AssignmentConfig.load("configs/my_drone.json")
assigner = SignalAssigner(config)
data_converted = assigner.apply_to_hierarchy(flight_data)
```

### Signal Source Types

When plotting, signals are automatically grouped by their base type and styled by source:

| Source | Description | Line Style |
|--------|-------------|------------|
| `measurement` | Raw sensor data | Solid |
| `command` | Commanded/setpoint values | Dashed |
| `estimated` | Filtered/Kalman estimates | Dotted |
| `reference` | Reference/target values | Dash-dot |
| `simulated` | Simulation output | Long dash |
| `raw` | Unprocessed data | Thin solid |

### Available Standard Signals

| Category | Signals |
|----------|---------|
| Position | `position.latitude`, `position.longitude`, `position.altitude`, `position.altitude_msl`, `position.altitude_agl` |
| Velocity | `velocity.north`, `velocity.east`, `velocity.down`, `velocity.ground_speed`, `velocity.airspeed` |
| Attitude | `attitude.roll`, `attitude.pitch`, `attitude.yaw`, `attitude.heading` |
| Angular Rates | `gyro.x`, `gyro.y`, `gyro.z`, `gyro.roll_rate`, `gyro.pitch_rate`, `gyro.yaw_rate` |
| Acceleration | `accel.x`, `accel.y`, `accel.z` |
| Battery | `battery.voltage`, `battery.current`, `battery.remaining` |

### Common Conversion Presets

| Preset | Scale | Description |
|--------|-------|-------------|
| `gps_1e7_to_degrees` | 1e-7 | MAVLink GPS coordinates |
| `mm_to_meters` | 0.001 | Millimeters to meters |
| `rad_to_deg` | 57.2958 | Radians to degrees |
| `cm_s_to_m_s` | 0.01 | cm/s to m/s |
| `mg_to_m_s2` | 0.00981 | milli-G to m/s² |
| `us_to_s` | 1e-6 | Microseconds to seconds |

## Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| MAVLink Telemetry | `.tlog` | Standard MAVLink log files |
| PX4 ULog | `.ulg` | PX4 binary logs |
| ArduPilot Binary | `.bin` | ArduPilot binary logs |
| CSV | `.csv` | Comma-separated values |
| Excel | `.xlsx`, `.xls` | Microsoft Excel |
| JSON | `.json` | JavaScript Object Notation |
| MATLAB | `.mat` | MATLAB data files |

## Project Structure

```
flightlog_analyzer/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Project configuration
├── src/
│   ├── core/             # Core application
│   │   ├── app.py        # Dash app factory
│   │   ├── constants.py  # Application constants
│   │   ├── exceptions.py # Custom exceptions
│   │   └── version.py    # Version management
│   ├── data/             # Data handling
│   │   ├── loader.py     # Data loading
│   │   ├── hierarchy.py  # Hierarchical navigation
│   │   ├── folder_importer.py  # Folder import
│   │   ├── signal_assignment.py # Signal mapping
│   │   ├── signal_groups.py    # Multi-source grouping
│   │   ├── validator.py  # Data validation
│   │   ├── alignment.py  # Signal alignment
│   │   └── downsampling.py # Downsampling algorithms
│   ├── visualization/    # Plotting
│   │   ├── plots/        # Plot implementations
│   │   └── maps/         # Map visualizations
│   ├── computed_signals/ # Computed signals engine
│   ├── ui/               # User interface
│   │   ├── app_layout.py # Main layout
│   │   ├── callbacks.py  # Dash callbacks
│   │   └── components/   # UI components
│   ├── config/           # Configuration
│   ├── utils/            # Utilities
│   └── export/           # Export functionality
├── tests/                # Test suite (910 tests)
├── examples/             # Example scripts
├── assets/               # Static assets (CSS)
└── docs/                 # Documentation
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_folder_import.py -v

# Format code
black src tests

# Type checking
mypy src
```

## Building Standalone Executables

```bash
# Install build dependencies
pip install pyinstaller

# Build for your platform
python scripts/build.py

# Build single-file executable
python scripts/build.py --onefile

# Build and create archive
python scripts/build.py --archive
```

Output will be in `dist/FlightLogDashboard/`.

## Configuration

Configuration can be provided via:
- Python dictionary
- YAML file
- JSON file
- Environment variables

Example configuration:
```yaml
theme: dark
timestamp_column: timestamp
max_display_points: 10000
cache_size_mb: 512
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Changelog](CHANGELOG.md)
- [User Guide](docs/user_guide/README.md)
- [API Reference](docs/api/README.md)
- [Architecture](docs/architecture/README.md)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Author

Christian Luciano - [GitHub](https://github.com/ChristianLuciano)
