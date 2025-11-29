# Installation Guide

This guide covers all installation methods for the Flight Log Analysis Dashboard.

## Quick Start

### Option 1: pip install (Recommended for Python users)

```bash
# Install from PyPI (when published)
pip install flight-log-dashboard

# Or install from source
git clone https://github.com/flight-log/dashboard.git
cd dashboard
pip install -e .
```

### Option 2: Standalone Executable

Download the appropriate version for your platform:

| Platform | Download |
|----------|----------|
| Windows  | `FlightLogDashboard-windows-x64.zip` |
| macOS    | `FlightLogDashboard-macos-x64.dmg` |
| Linux    | `FlightLogDashboard-linux-x64.tar.gz` |

## Detailed Installation

### Prerequisites

- **Python**: 3.8 or higher (for pip installation)
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB for installation, plus space for flight logs

### Method 1: Development Installation

For developers who want to modify the code:

```bash
# Clone the repository
git clone https://github.com/flight-log/dashboard.git
cd dashboard

# Create virtual environment
python -m venv flog
source flog/bin/activate  # On Windows: .\flog\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,geo]"

# Run the dashboard
python examples/basic_usage.py
```

### Method 2: Production Installation

For end users:

```bash
# Create virtual environment
python -m venv flight-log-env
source flight-log-env/bin/activate

# Install the package
pip install flight-log-dashboard

# Run the dashboard
flight-log-dashboard
```

### Method 3: Standalone Executable

#### Windows

1. Download `FlightLogDashboard-windows-x64.zip`
2. Extract to a folder (e.g., `C:\Program Files\FlightLogDashboard`)
3. Run `FlightLogDashboard.exe`

**Silent Installation (Enterprise):**
```cmd
FlightLogDashboard-installer.exe /S /D=C:\Program Files\FlightLogDashboard
```

#### macOS

1. Download `FlightLogDashboard-macos-x64.dmg`
2. Open the DMG file
3. Drag `FlightLogDashboard.app` to Applications
4. Right-click and select "Open" (first time only, due to Gatekeeper)

#### Linux

```bash
# Download and extract
wget https://github.com/flight-log/releases/FlightLogDashboard-linux-x64.tar.gz
tar -xzf FlightLogDashboard-linux-x64.tar.gz
cd FlightLogDashboard

# Run
./FlightLogDashboard
```

**Desktop Integration (Ubuntu/Debian):**
```bash
# Create desktop entry
cat > ~/.local/share/applications/flight-log-dashboard.desktop << EOF
[Desktop Entry]
Name=Flight Log Dashboard
Exec=/path/to/FlightLogDashboard/FlightLogDashboard
Icon=/path/to/FlightLogDashboard/icon.png
Type=Application
Categories=Science;Engineering;
EOF
```

## Optional Dependencies

### MAVLink Support

For .tlog file support:
```bash
pip install pymavlink
```

### PX4 ULog Support

For .ulg file support:
```bash
pip install pyulog
```

### Advanced Geographic Features

For shapefile export and advanced coordinate operations:
```bash
pip install geopandas shapely pyproj
```

### Performance Optimization

For faster signal processing with large datasets:
```bash
pip install numba
```

## Building from Source

### Building Standalone Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build for your platform
pyinstaller flight_log_dashboard.spec

# Output will be in dist/FlightLogDashboard/
```

### Building pip Package

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Output will be in dist/
# - flight_log_dashboard-0.1.0-py3-none-any.whl
# - flight_log_dashboard-0.1.0.tar.gz
```

## Verifying Installation

```bash
# Check version
python -c "from src.core.version import get_version; print(get_version())"

# Run tests
pytest tests/ -v

# Start the dashboard
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Reinstall in development mode
pip install -e .
```

**Permission denied (Linux/macOS):**
```bash
chmod +x FlightLogDashboard
```

**Port already in use:**
```bash
# The dashboard runs on port 8050 by default
# Check if another process is using it
netstat -tulpn | grep 8050

# Or specify a different port in your code
app.run(port=8051)
```

**pymavlink installation fails on Windows:**
```bash
# Install Visual C++ Build Tools first
# Then retry:
pip install pymavlink
```

## Updating

### pip Installation

```bash
pip install --upgrade flight-log-dashboard
```

### Standalone Executable

1. Download the new version
2. Replace the old installation folder
3. Your configurations are preserved (stored in user directory)

## Uninstallation

### pip Installation

```bash
pip uninstall flight-log-dashboard
```

### Standalone (Windows)

Use "Add or Remove Programs" or delete the installation folder.

### Standalone (macOS)

Drag the app from Applications to Trash.

### Standalone (Linux)

Delete the installation folder and desktop entry.

## Support

- **Documentation**: https://flight-log.github.io/dashboard/
- **Issues**: https://github.com/flight-log/dashboard/issues
- **Discussions**: https://github.com/flight-log/dashboard/discussions

