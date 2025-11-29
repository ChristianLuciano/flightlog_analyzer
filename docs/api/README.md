# API Documentation

This document provides API reference for the Flight Log Analysis Dashboard.

## Core Modules

### `src.core.app`
Main application factory.

```python
from src.core.app import create_app, run_app

# Create application
app = create_app(settings=Settings(), flight_data=data)

# Run server
run_app(app, host="127.0.0.1", port=8050, debug=True)
```

### `src.data.loader`
Data loading functionality.

```python
from src.data.loader import DataLoader, load_flight_data

# Using class
loader = DataLoader(timestamp_column='timestamp')
loader.load(flight_data)

# Using convenience function
loader = load_flight_data(flight_data)
```

#### Methods:
- `load(data)` - Load hierarchical data structure
- `get_dataframe(path)` - Get DataFrame by path
- `get_signal(path)` - Get signal Series by path
- `list_dataframes()` - List all DataFrame paths
- `list_signals()` - List all signal information

### `src.data.hierarchy`
Hierarchical data navigation.

```python
from src.data.hierarchy import HierarchyNavigator

nav = HierarchyNavigator(data)
gps_df = nav.get_dataframe('Sensors.GPS')
lat = nav.get_signal('Sensors.GPS.lat')
```

### `src.computed_signals.engine`
Computed signal engine.

```python
from src.computed_signals.engine import ComputedSignalEngine

engine = ComputedSignalEngine(data_provider)
engine.register_signal('speed', {
    'formula': 'sqrt(vx**2 + vy**2)',
    'inputs': ['velocity.vx', 'velocity.vy'],
    'unit': 'm/s'
})
result = engine.compute('speed')
```

## Visualization

### Plot Types
- `TimeSeriesPlot` - Time series visualization
- `XYPlot` - X-Y scatter/line plots
- `FFTPlot` - Frequency domain analysis
- `HistogramPlot` - Distribution visualization
- `Map2D` - 2D geographic map
- `Map3D` - 3D geographic map

## Export Functions

```python
from src.export.images import export_plot_image
from src.export.geo_formats import export_kml, export_geojson
from src.export.data_export import export_csv

export_plot_image(figure, 'plot.png', width=1200, height=800)
export_kml(lat, lon, alt, 'flight.kml')
export_csv(df, 'data.csv')
```

## Configuration

```python
from src.config.settings import Settings
from src.config.loader import load_config, save_config

settings = Settings(theme='dark', cache_size_mb=512)
config = load_config('config.yaml')
save_config(config, 'config.json')
```

