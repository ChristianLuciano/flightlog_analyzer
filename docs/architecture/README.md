# Architecture Documentation

## Overview

The Flight Log Analysis Dashboard is built using a modular architecture with clear separation of concerns.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Dash Application                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    UI       │  │  Callbacks  │  │    State Manager    │ │
│  │ Components  │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│ Visualization │   │ Computed Signals│   │     Data      │
│    Module     │   │     Engine      │   │    Module     │
└───────────────┘   └─────────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Data Cache    │
                    └─────────────────┘
```

## Module Responsibilities

### Core (`src/core/`)
- Application initialization
- Constants and configuration
- Custom exceptions
- Type definitions

### Data (`src/data/`)
- **loader.py**: Load and index hierarchical data
- **hierarchy.py**: Navigate nested structures
- **validator.py**: Validate data integrity
- **cache.py**: LRU caching implementation
- **alignment.py**: Cross-signal time alignment
- **downsampling.py**: LTTB, M4, Douglas-Peucker

### Visualization (`src/visualization/`)
- **base.py**: Abstract plot interface
- **manager.py**: Plot lifecycle management
- **theme.py**: Theming support
- **plots/**: Time series, FFT, histograms
- **maps/**: 2D and 3D geographic views

### Computed Signals (`src/computed_signals/`)
- **engine.py**: Signal computation orchestration
- **parser.py**: Safe formula parsing via AST
- **functions.py**: Built-in function library
- **dependencies.py**: Dependency graph management
- **cache.py**: Result caching

### UI (`src/ui/`)
- **app_layout.py**: Main layout structure
- **callbacks.py**: Dash callback definitions
- **state.py**: Application state management
- **components/**: Reusable UI components
- **layouts/**: Grid and tab management

## Design Patterns

### Factory Pattern
Used in `create_app()` for application initialization.

### Observer Pattern
State changes propagate to subscribed components.

### Strategy Pattern
Interchangeable algorithms for downsampling, interpolation.

### Decorator Pattern
Caching decorators for computed signals.

## Data Flow

1. **Load**: Data loaded via `DataLoader`
2. **Index**: Hierarchy indexed for fast access
3. **Cache**: Results cached for performance
4. **Visualize**: Plots render via Plotly
5. **Interact**: Callbacks handle user input
6. **Update**: State changes trigger re-renders

## Performance Considerations

### Downsampling
- LTTB for time series (preserves visual shape)
- Douglas-Peucker for paths (preserves geometry)

### Caching Strategy
- LRU eviction policy
- Separate caches for DataFrames, signals, FFT
- Cache invalidation on data changes

### Lazy Loading
- Tabs load on first activation
- Computed signals evaluated on demand
- Progressive rendering for large datasets

## Security

### Formula Evaluation
- AST-based parsing (no `eval()`)
- Whitelisted functions only
- Resource limits (time, memory)

## Extensibility

### Adding Plot Types
1. Create class extending `BasePlot`
2. Implement `render()` and `update()`
3. Register in `PLOT_REGISTRY`

### Adding Functions
1. Define function in `functions.py`
2. Add to `BUILTIN_FUNCTIONS` dict

### Custom Themes
1. Create `Theme` instance
2. Register via `register_theme()`

