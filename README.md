# Flight Log Analysis Dashboard

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

A high-performance interactive dashboard for analyzing flight log data from pre-parsed pandas DataFrames. Designed to handle long-duration (30-60 minute), high-frequency (microsecond-level) multi-signal datasets with advanced geospatial visualization, computed signal capabilities, and real-time playback features.

## Quick Install

```bash
# Option 1: pip install (recommended)
pip install flight-log-dashboard

# Option 2: From source
git clone https://github.com/flight-log/dashboard.git
cd dashboard
pip install -e .
```

**Standalone executables** available for Windows, macOS, and Linux - see [Releases](https://github.com/flight-log/dashboard/releases).

## Features

### Data Handling
- **Hierarchical Data Structure**: Support for nested dictionary structures containing pandas DataFrames
- **Automatic Discovery**: Recursive traversal to discover all DataFrames and signals
- **Path-Based Access**: Access data using hierarchical paths (e.g., `Sensors.IMU.Accelerometer.accel_x`)
- **Cross-DataFrame Alignment**: Automatic time-based alignment of signals with different sampling rates
- **Folder Import**: Recursively import all CSV files from a folder structure (e.g., `Flight_001/COMP1/sensors/*.csv`)
- **Signal Assignment**: Map raw columns to standard flight dynamics variables with conversion factors

### Visualization
- **Time Series Plots**: Interactive plots with zoom, pan, and linked cursors
- **FFT Analysis**: Frequency domain visualization with configurable windowing
- **Geographic Maps**: 2D and 3D flight path visualization with multiple basemap options
- **Statistical Plots**: Histograms, box plots, correlation matrices
- **Event Annotations**: Automatic event detection and display on all plots

### Computed Signals
- **Formula-Based Computation**: Create derived signals using mathematical expressions
- **Built-in Functions**: Math, signal processing, and geographic operations
- **Dependency Resolution**: Automatic handling of signal dependencies
- **Caching**: Smart caching for performance optimization

### Performance
- **LTTB Downsampling**: Efficient visualization of large datasets
- **Lazy Loading**: Load data on-demand for better memory usage
- **WebGL Rendering**: Hardware-accelerated graphics for smooth interactions
- **Caching**: Multi-level caching for computed results

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/flight-log-dashboard.git
cd flight-log-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
import pandas as pd
import numpy as np
from src.core.app import create_app, run_app
from src.data.loader import load_flight_data

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

# Load data
loader = load_flight_data(flight_data)

# Create and run app
app = create_app(flight_data=flight_data)
run_app(app)
```

Then open http://127.0.0.1:8050 in your browser.

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
├── COMP1/
│   ├── sensors/
│   │   ├── imu.csv
│   │   └── gps.csv
│   └── control.csv
├── COMP2/
│   └── motors.csv
└── COMP3/
    └── nav.csv
```

Results in:
```python
{
    'COMP1': {
        'sensors': {'imu': DataFrame, 'gps': DataFrame},
        'control': DataFrame
    },
    'COMP2': {'motors': DataFrame},
    'COMP3': {'nav': DataFrame}
}
```

## Signal Assignment

Map raw columns to standard flight dynamics variables with conversion factors:

```python
from src.data.signal_assignment import (
    AssignmentConfig, SignalAssigner, StandardSignal
)

# Create assignment configuration
config = AssignmentConfig(name="My Drone Config", version="1.0")

# Add mappings with conversion presets
config.add_mapping(
    source_column='lat_raw',
    target_signal=StandardSignal.POSITION_LATITUDE,
    conversion='gps_1e7_to_degrees'  # Common MAVLink scaling
)

config.add_mapping(
    source_column='roll_rad',
    target_signal=StandardSignal.ATTITUDE_ROLL,
    conversion='rad_to_deg'
)

# Save for reuse
config.save("configs/my_drone.json")

# Load and apply to data
config = AssignmentConfig.load("configs/my_drone.json")
assigner = SignalAssigner(config)

# Apply to a DataFrame
df_converted = assigner.apply(raw_df)

# Or apply to hierarchical data
data_converted = assigner.apply_to_hierarchy(flight_data)
```

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

## Project Structure

```
Flight_Log/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Project configuration
├── src/
│   ├── core/             # Core application
│   │   ├── app.py        # Dash app factory
│   │   ├── constants.py  # Application constants
│   │   ├── exceptions.py # Custom exceptions
│   │   └── types.py      # Type definitions
│   ├── data/             # Data handling
│   │   ├── loader.py     # Data loading
│   │   ├── hierarchy.py  # Hierarchical navigation
│   │   ├── validator.py  # Data validation
│   │   ├── cache.py      # Caching
│   │   ├── alignment.py  # Signal alignment
│   │   └── downsampling.py # Downsampling algorithms
│   ├── visualization/    # Plotting
│   │   ├── base.py       # Base plot class
│   │   ├── manager.py    # Plot management
│   │   ├── theme.py      # Theming
│   │   ├── plots/        # Plot implementations
│   │   └── maps/         # Map visualizations
│   ├── computed_signals/ # Computed signals
│   │   ├── engine.py     # Computation engine
│   │   ├── parser.py     # Formula parser
│   │   ├── functions.py  # Built-in functions
│   │   └── dependencies.py # Dependency resolution
│   ├── ui/               # User interface
│   │   ├── app_layout.py # Main layout
│   │   ├── callbacks.py  # Dash callbacks
│   │   ├── components/   # UI components
│   │   └── layouts/      # Layout management
│   ├── config/           # Configuration
│   ├── utils/            # Utilities
│   └── export/           # Export functionality
├── tests/                # Test suite
├── examples/             # Example scripts
├── assets/               # Static assets
└── docs/                 # Documentation
```

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

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Format code
black src tests

# Type checking
mypy src
```

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

