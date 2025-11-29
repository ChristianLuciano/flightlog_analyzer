# Changelog

All notable changes to the Flight Log Analysis Dashboard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation
- Deployment documentation

---

## [0.1.0] - 2024-11-29

### Added

#### Core Features
- High-performance interactive dashboard for flight log analysis
- Hierarchical data structure support (nested DataFrames)
- Real-time signal plotting with Plotly
- Dark theme UI with modern design

#### Data Loading
- CSV, Excel, JSON file support
- MAVLink .tlog file support (via pymavlink)
- PX4 .ulg file support (via pyulog)
- ArduPilot .bin file support
- Drag-and-drop file upload
- Hierarchical data organization from flat file structures

#### Visualization
- **Overview Tab**: Auto-generated summary plots
  - Flight path map (OpenStreetMap)
  - Altitude profile
  - Attitude/Gyroscope data
  - Accelerometer data
  - Battery status
- **Time Series Tab**: Multi-signal time plots
- **Map Tab**: Full-screen flight path visualization
- **Analysis Tab**: Flight statistics and metrics
- Custom plot builder with multiple plot types:
  - Time Series
  - X-Y Scatter
  - 3D plots
  - FFT (frequency analysis)
  - Histogram
  - Scatter

#### Signal Management
- Signal selector with search/filter
- Quick select buttons (IMU, GPS, Battery, Motors, Control)
- Collapsible data tree with signal tooltips
- Time range selection for focused analysis

#### Playback Controls
- Play/Pause with adjustable speed (0.5x - 4x)
- Step forward/backward
- Jump to start/end
- Time slider for navigation

#### Export Features
- CSV export
- Excel export (multiple sheets)
- JSON export
- Configuration save/load

#### Performance
- Efficient handling of large datasets
- WebGL rendering for smooth plots
- Downsampling for overview plots

### Technical
- Built with Dash and Plotly
- Bootstrap-based responsive layout
- Comprehensive test suite (26+ tests)
- Python 3.8+ support

### Known Issues
- Computed signals not yet implemented
- FFT plot needs more configuration options
- Map offline mode not implemented

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-11-29 | Initial release with core visualization features |

---

## Upgrade Notes

### Upgrading to 0.1.0

This is the initial release. No upgrade path required.

### Configuration Migration

The dashboard automatically migrates configurations from older versions.
If you encounter issues, delete your configuration file and reconfigure.

Configuration files are stored in:
- **Windows**: `%APPDATA%\FlightLogDashboard\config.yaml`
- **macOS**: `~/Library/Application Support/FlightLogDashboard/config.yaml`
- **Linux**: `~/.config/FlightLogDashboard/config.yaml`

---

## Roadmap

### Version 0.2.0 (Planned)
- Computed signals with formula editor
- Event detection and annotation
- Improved 3D visualization
- KML/GPX export

### Version 0.3.0 (Planned)
- Real-time streaming support
- Multi-flight comparison
- Report generation
- Cloud storage integration

### Version 1.0.0 (Planned)
- Stable API
- Plugin system
- Full documentation
- Video tutorials

---

[Unreleased]: https://github.com/flight-log/dashboard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/flight-log/dashboard/releases/tag/v0.1.0

