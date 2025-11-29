# User Guide

## Getting Started

### Installation

1. Ensure Python 3.8+ is installed
2. Clone the repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run the dashboard: `python app.py`

### Loading Data

The dashboard accepts hierarchical dictionary structures containing pandas DataFrames:

```python
flight_data = {
    'Sensors': {
        'GPS': pd.DataFrame({...}),
        'IMU': {
            'Accelerometer': pd.DataFrame({...}),
            'Gyroscope': pd.DataFrame({...}),
        }
    },
    'Events': pd.DataFrame({...}),
}
```

## Interface Overview

### Main Components

1. **Header** - Application title and main actions
2. **Sidebar** - Data browser, signal selector, plot builder
3. **Main Area** - Plot grid with tabs
4. **Playback Controls** - Time navigation and playback
5. **Status Bar** - Current status and info

### Data Browser

Navigate the hierarchical data structure:
- Expand/collapse nodes to explore
- Search for signals by name
- Click signals to select them

### Creating Plots

1. Select signals from the browser or dropdown
2. Choose plot type (Time Series, FFT, Map, etc.)
3. Click "Add Plot"
4. Drag to reposition in grid

### Playback

- **Play/Pause**: Animate through time
- **Speed**: Adjust playback speed (0.1x to 10x)
- **Slider**: Scrub to any time point
- **Step**: Move frame by frame

### Computed Signals

Create derived signals using formulas:

1. Open Formula Editor
2. Enter signal name
3. Write formula using available functions
4. Select input signals
5. Click "Create Signal"

Available functions:
- Math: `sqrt`, `sin`, `cos`, `abs`, `exp`, `log`
- Signal: `diff`, `cumsum`, `moving_avg`, `lowpass`
- Geographic: `haversine`, `cumulative_distance`, `bearing`

### Export

Export data and plots:
- **Images**: PNG, SVG, PDF
- **Geographic**: KML, GeoJSON, GPX
- **Data**: CSV, Excel

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| ← → | Step back/forward |
| Home | Reset zoom |
| +/- | Zoom in/out |
| Ctrl+S | Save configuration |

## Tips

- Use downsampling for large datasets
- Bookmark frequently used signals
- Save configurations for repeated analyses
- Use dark theme for long sessions

