"""
Main application layout.

Defines the overall structure and layout of the dashboard UI.
"""

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from typing import Optional
import plotly.graph_objects as go


def create_layout(app: Dash) -> dbc.Container:
    """
    Create the main application layout.

    Args:
        app: Dash application instance.

    Returns:
        Root layout component.
    """
    # Get flight data from app
    flight_data = getattr(app, 'flight_data', None)
    
    # Build signal options from flight data
    signal_options = []
    if flight_data:
        signal_options = _build_signal_options(flight_data)
    
    return dbc.Container([
        # Store components for state management
        dcc.Store(id='app-state', storage_type='memory', data={
            'flight_data_loaded': flight_data is not None,
        }),
        dcc.Store(id='selected-signals', data=[]),
        dcc.Store(id='plot-counter', data=0),
        
        # Download component for exports
        dcc.Download(id='export-download'),

        # Header
        _create_header(),

        # Main content
        dbc.Row([
            # Sidebar
            dbc.Col([
                _create_sidebar(signal_options, flight_data)
            ], width=3, className='sidebar-col'),

            # Main plot area
            dbc.Col([
                _create_main_content(app),
                # Custom plot container for user-selected signals
                html.Hr(className='my-3'),
                html.Div([
                    html.H6([
                        html.I(className='fas fa-chart-area me-2'), 
                        "Custom Plots"
                    ], className='text-light mb-3 d-inline-block'),
                ]),
                html.Div(id='custom-plots-list', children=[
                    dbc.Alert([
                        html.I(className='fas fa-info-circle me-2'),
                        "Select signals, choose plot type, optionally set time range, and click 'Add Plot'"
                    ], color='secondary', className='mb-0')
                ]),
            ], width=9, className='main-col'),
        ], className='main-row'),

        # Footer / Status bar
        _create_status_bar(flight_data),
        
        # Export Modal
        _create_export_modal(),
        
        # Load/Import Modal
        _create_load_modal(),
        
        # Signal Assignment Modal
        _create_assignment_modal(),
    ], fluid=True, className='app-container')


def _build_signal_options(data, prefix=''):
    """Recursively build signal options from hierarchical data."""
    import pandas as pd
    options = []
    
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                if col != 'timestamp':
                    full_path = f"{path}.{col}"
                    options.append({
                        'label': f"{path} → {col}",
                        'value': full_path
                    })
        elif isinstance(value, dict):
            options.extend(_build_signal_options(value, path))
    
    return options


def _create_header() -> dbc.Navbar:
    """Create application header."""
    return dbc.Navbar([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className='fas fa-plane-departure me-2'),
                        html.Span("Flight Log Analysis Dashboard", className='navbar-brand-text')
                    ], className='d-flex align-items-center')
                ], width='auto'),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([html.I(className='fas fa-folder-open me-1'), "Load"], 
                                  id='btn-load', color='primary', outline=True, size='sm'),
                        dbc.Button([html.I(className='fas fa-save me-1'), "Save"], 
                                  id='btn-save', color='secondary', outline=True, size='sm'),
                        dbc.Button([html.I(className='fas fa-download me-1'), "Export"], 
                                  id='btn-export', color='success', outline=True, size='sm'),
                    ])
                ], width='auto', className='ms-auto'),
            ], align='center', className='w-100'),
        ], fluid=True),
    ], color='dark', dark=True, className='mb-3')


def _create_sidebar(signal_options, flight_data) -> html.Div:
    """Create sidebar with data browser and controls."""
    # Build data tree summary
    data_summary = []
    if flight_data:
        data_summary = _build_data_tree_summary(flight_data)
    
    return html.Div([
        # Data Summary Card (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className='fas fa-database me-2'),
                    "Data Summary",
                    html.I(
                        id='data-summary-chevron',
                        className='fas fa-chevron-down ms-auto',
                        style={'transition': 'transform 0.3s'}
                    ),
                ], className='d-flex align-items-center w-100', 
                   style={'cursor': 'pointer'},
                   id='data-summary-toggle')
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    html.Div(data_summary if data_summary else "No data loaded", 
                            id='data-tree', className='data-tree-content')
                ])
            ], id='data-summary-collapse', is_open=True)
        ], className='mb-3'),

        # Quick Actions Card (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className='fas fa-bolt me-2'),
                    "Quick Select",
                    html.I(
                        id='quick-select-chevron',
                        className='fas fa-chevron-down ms-auto',
                        style={'transition': 'transform 0.3s'}
                    ),
                ], className='d-flex align-items-center w-100', 
                   style={'cursor': 'pointer'},
                   id='quick-select-toggle')
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    dbc.ButtonGroup([
                        dbc.Button("IMU", id='btn-quick-imu', size='sm', outline=True, color='info'),
                        dbc.Button("GPS", id='btn-quick-gps', size='sm', outline=True, color='info'),
                        dbc.Button("Battery", id='btn-quick-battery', size='sm', outline=True, color='info'),
                    ], className='w-100 mb-2'),
                    dbc.ButtonGroup([
                        dbc.Button("Motors", id='btn-quick-motors', size='sm', outline=True, color='warning'),
                        dbc.Button("Control", id='btn-quick-control', size='sm', outline=True, color='warning'),
                        dbc.Button([
                            html.I(className='fas fa-times me-1'),
                            "Clear"
                        ], id='btn-clear-selection', size='sm', outline=True, color='danger'),
                    ], className='w-100'),
                ])
            ], id='quick-select-collapse', is_open=True)
        ], className='mb-3'),

        # Signal Selector Card (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className='fas fa-chart-line me-2'),
                    "Signal Selector",
                    html.I(
                        id='signal-selector-chevron',
                        className='fas fa-chevron-down ms-auto',
                        style={'transition': 'transform 0.3s'}
                    ),
                ], className='d-flex align-items-center w-100', 
                   style={'cursor': 'pointer'},
                   id='signal-selector-toggle')
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    # Search filter
                    dbc.Input(
                        id='signal-search',
                        type='text',
                        placeholder='Filter signals...',
                        className='mb-2',
                        style={'backgroundColor': '#1a2744', 'color': '#ffffff', 
                               'border': '1px solid #2d4a7c'}
                    ),
                    # Scrollable signal checklist
                    html.Div(
                        dbc.Checklist(
                            id='signal-checklist',
                            options=signal_options,
                            value=[],
                            className='signal-checklist',
                            inline=False,
                            style={'color': '#ffffff'}
                        ),
                        style={
                            'maxHeight': '200px', 
                            'overflowY': 'auto',
                            'backgroundColor': '#1a2744',
                            'borderRadius': '4px',
                            'padding': '8px',
                            'border': '1px solid #2d4a7c'
                        }
                    ),
                    # Signal count with inline clear button
                    html.Div([
                        html.Small(
                            id='signal-count',
                            children='0 signals selected',
                            className='text-muted'
                        ),
                        dbc.Button(
                            "Select All",
                            id='btn-select-all',
                            size='sm',
                            color='link',
                            className='p-0 ms-2 text-info',
                            style={'fontSize': '0.75rem'}
                        ),
                    ], className='d-flex align-items-center mt-1 mb-2'),
                ])
            ], id='signal-selector-collapse', is_open=True)
        ], className='mb-3'),

        # Plot Builder Card (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className='fas fa-plus-circle me-2'),
                    "Plot Builder",
                    html.I(
                        id='plot-builder-chevron',
                        className='fas fa-chevron-down ms-auto',
                        style={'transition': 'transform 0.3s'}
                    ),
                ], className='d-flex align-items-center w-100', 
                   style={'cursor': 'pointer'},
                   id='plot-builder-toggle')
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    # Plot type selector
                    dbc.Label("Plot Type", className='small text-muted'),
                    dbc.Select(
                        id='plot-type-select',
                        options=[
                            {'label': '📈 Time Series', 'value': 'timeseries'},
                            {'label': '📊 X-Y Plot', 'value': 'xy'},
                            {'label': '🎲 3D Plot', 'value': '3d'},
                            {'label': '〰️ FFT Spectrum', 'value': 'fft'},
                            {'label': '📊 Histogram', 'value': 'histogram'},
                            {'label': '⚫ Scatter', 'value': 'scatter'},
                        ],
                        value='timeseries',
                        className='mb-2'
                    ),
                    
                    # Time range inputs
                    dbc.Label("Time Range (optional)", className='small text-muted'),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id='time-range-start',
                                type='number',
                                placeholder='Start (s)',
                                size='sm'
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Input(
                                id='time-range-end',
                                type='number',
                                placeholder='End (s)',
                                size='sm'
                            )
                        ], width=6),
                    ], className='mb-2'),
                    
                    # Conversion Factor Section (collapsible)
                    html.Details([
                        html.Summary([
                            html.I(className='fas fa-calculator me-2'),
                            "Signal Conversion",
                        ], className='text-muted small mb-2', style={'cursor': 'pointer'}),
                        html.Div([
                            # Preset conversions dropdown
                            dbc.Label("Preset", className='small text-muted'),
                            dbc.Select(
                                id='conversion-preset',
                                options=[
                                    {'label': 'None (raw values)', 'value': 'none'},
                                    {'label': '── Length ──', 'value': '', 'disabled': True},
                                    {'label': 'mm → m (÷1000)', 'value': 'mm_to_m'},
                                    {'label': 'cm → m (÷100)', 'value': 'cm_to_m'},
                                    {'label': 'm → km (÷1000)', 'value': 'm_to_km'},
                                    {'label': 'in → m (×0.0254)', 'value': 'in_to_m'},
                                    {'label': 'ft → m (×0.3048)', 'value': 'ft_to_m'},
                                    {'label': '── Angles ──', 'value': '', 'disabled': True},
                                    {'label': 'rad → deg (×57.2958)', 'value': 'rad_to_deg'},
                                    {'label': 'deg → rad (÷57.2958)', 'value': 'deg_to_rad'},
                                    {'label': 'mrad → deg (×0.0573)', 'value': 'mrad_to_deg'},
                                    {'label': '── MAVLink GPS ──', 'value': '', 'disabled': True},
                                    {'label': 'lat/lon ÷1e7 (MAVLink)', 'value': 'mavlink_latlon'},
                                    {'label': 'altitude mm → m', 'value': 'mavlink_alt'},
                                    {'label': '── Time ──', 'value': '', 'disabled': True},
                                    {'label': 'μs → s (÷1e6)', 'value': 'us_to_s'},
                                    {'label': 'ms → s (÷1000)', 'value': 'ms_to_s'},
                                    {'label': '── Velocity ──', 'value': '', 'disabled': True},
                                    {'label': 'cm/s → m/s (÷100)', 'value': 'cms_to_ms'},
                                    {'label': 'm/s → km/h (×3.6)', 'value': 'ms_to_kmh'},
                                    {'label': 'knots → m/s (×0.5144)', 'value': 'knots_to_ms'},
                                    {'label': '── Custom ──', 'value': '', 'disabled': True},
                                    {'label': 'Custom (enter below)', 'value': 'custom'},
                                ],
                                value='none',
                                size='sm',
                                className='mb-2'
                            ),
                            
                            # Custom conversion: y = scale * x + offset
                            dbc.Label("Custom: y = scale × x + offset", className='small text-muted'),
                            dbc.Row([
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.InputGroupText("×", className='input-group-text-sm'),
                                        dbc.Input(
                                            id='conversion-scale',
                                            type='number',
                                            placeholder='1.0',
                                            value=1.0,
                                            size='sm',
                                            style={'backgroundColor': '#1a2744', 'color': '#fff'}
                                        ),
                                    ], size='sm')
                                ], width=6),
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.InputGroupText("+", className='input-group-text-sm'),
                                        dbc.Input(
                                            id='conversion-offset',
                                            type='number',
                                            placeholder='0.0',
                                            value=0.0,
                                            size='sm',
                                            style={'backgroundColor': '#1a2744', 'color': '#fff'}
                                        ),
                                    ], size='sm')
                                ], width=6),
                            ], className='mb-1'),
                            
                            # Unit label
                            dbc.InputGroup([
                                dbc.InputGroupText("Unit label", className='input-group-text-sm'),
                                dbc.Input(
                                    id='conversion-unit',
                                    type='text',
                                    placeholder='e.g., m/s, deg, km',
                                    size='sm',
                                    style={'backgroundColor': '#1a2744', 'color': '#fff'}
                                ),
                            ], size='sm', className='mb-2'),
                        ], className='ps-3')
                    ], className='mb-2'),
                    
                    # Action buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                html.I(className='fas fa-plus me-1'),
                                "Add Plot"
                            ], id='btn-add-plot', color='success', className='w-100', size='sm')
                        ], width=6),
                        dbc.Col([
                            dbc.Button([
                                html.I(className='fas fa-trash me-1'),
                                "Clear All"
                            ], id='btn-clear-plots', color='danger', outline=True, className='w-100', size='sm')
                        ], width=6),
                    ]),
                ])
            ], id='plot-builder-collapse', is_open=True)
        ], className='mb-3'),

        # Events Card (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className='fas fa-exclamation-triangle me-2'),
                    "Events Filter",
                    html.I(
                        id='events-filter-chevron',
                        className='fas fa-chevron-down ms-auto',
                        style={'transition': 'transform 0.3s'}
                    ),
                ], className='d-flex align-items-center w-100', 
                   style={'cursor': 'pointer'},
                   id='events-filter-toggle')
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    dbc.Checklist(
                        id='event-filter',
                        options=[
                            {'label': ' Critical', 'value': 'critical'},
                            {'label': ' Warning', 'value': 'warning'},
                            {'label': ' Info', 'value': 'info'},
                        ],
                        value=['critical', 'warning'],
                        inline=True
                    ),
                ])
            ], id='events-filter-collapse', is_open=True)
        ]),
    ], className='sidebar-content')


def _build_data_tree_summary(data, level=0, parent_id=''):
    """Build a collapsible tree view of the data using Bootstrap Accordion."""
    import pandas as pd
    items = []
    
    # Time columns to exclude from display
    time_columns = {'timestamp', 'time', 'time_boot_ms', 'time_usec', 'TimeUS', 'mavpackettype'}
    
    for idx, (key, value) in enumerate(data.items()):
        item_id = f"{parent_id}-{key}".replace('.', '-').replace(' ', '-')
        
        if isinstance(value, pd.DataFrame):
            # DataFrame leaf node - show columns (exclude time columns, show all on hover)
            all_columns = [col for col in value.columns if col.lower() not in {c.lower() for c in time_columns}]
            numeric_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(value[col])]
            
            # Short preview for display
            columns_preview = ', '.join(numeric_columns[:4])
            if len(numeric_columns) > 4:
                columns_preview += f'... (+{len(numeric_columns) - 4} more)'
            
            # Full list for tooltip
            full_columns = '\n'.join(numeric_columns)
            
            items.append(
                html.Div([
                    html.Div([
                        html.I(className='fas fa-table text-success me-2'),
                        html.Strong(key, className='text-light'),
                        html.Small(f" ({len(value):,} rows, {len(numeric_columns)} signals)", className='text-muted ms-2'),
                    ], className='d-flex align-items-center py-1'),
                    html.Div([
                        html.Small([
                            html.I(className='fas fa-columns text-info me-1'),
                            columns_preview
                        ], className='text-muted d-block ps-4', title=full_columns)
                    ])
                ], className='tree-leaf mb-1', style={'paddingLeft': f'{level * 20}px'}, title=f"Signals:\n{full_columns}")
            )
        elif isinstance(value, dict):
            # Folder node - collapsible
            child_count = _count_items(value)
            children = _build_data_tree_summary(value, level + 1, item_id)
            
            items.append(
                html.Details([
                    html.Summary([
                        html.I(className='fas fa-folder-open text-warning me-2', id=f'icon-{item_id}'),
                        html.Strong(key, className='text-light'),
                        html.Small(f" ({child_count} items)", className='text-muted ms-2'),
                    ], className='tree-folder-header py-1'),
                    html.Div(children, className='tree-folder-content')
                ], open=level < 2, className='tree-folder mb-1', style={'paddingLeft': f'{level * 20}px'})
            )
    
    return items


def _count_items(data):
    """Count total DataFrames in nested structure."""
    import pandas as pd
    count = 0
    for value in data.values():
        if isinstance(value, pd.DataFrame):
            count += 1
        elif isinstance(value, dict):
            count += _count_items(value)
    return count


def _create_main_content(app) -> html.Div:
    """Create main content area with tabs and plots."""
    flight_data = getattr(app, 'flight_data', None)
    
    return html.Div([
        dbc.Tabs([
            dbc.Tab(label='📊 Overview', tab_id='tab-overview', 
                   children=_create_overview_tab(flight_data)),
            dbc.Tab(label='📈 Time Series', tab_id='tab-timeseries',
                   children=_create_timeseries_tab(flight_data)),
            dbc.Tab(label='🗺️ Map', tab_id='tab-map',
                   children=_create_map_tab(flight_data)),
            dbc.Tab(label='📉 Analysis', tab_id='tab-analysis',
                   children=_create_analysis_tab(flight_data)),
        ], id='main-tabs', active_tab='tab-overview', className='nav-tabs-custom'),
    ])


def _create_overview_tab(flight_data) -> html.Div:
    """Create overview tab with summary plots - always creates all plot containers."""
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    # Create empty figures with dark theme
    def empty_figure(title="No data"):
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=40, b=40),
            title=title
        )
        return fig
    
    # Initialize all figures (will be populated if data exists)
    overview_fig = empty_figure("Data Overview - Load data to see plots")
    map_fig = empty_figure("Flight Path - Load GPS data")
    altitude_fig = empty_figure("Altitude - Load GPS data")
    attitude_fig = empty_figure("Attitude - Load attitude data")
    imu_fig = empty_figure("Accelerometer - Load IMU data")
    battery_fig = empty_figure("Battery - Load battery data")
    
    if flight_data:
        # Add signals to overview figure
        signals_added = 0
        def add_to_overview(data, prefix=''):
            nonlocal signals_added
            for key, value in data.items():
                if signals_added >= 4:
                    return
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, pd.DataFrame):
                    # Find time column
                    time_col = _find_time_column(value)
                    if time_col:
                        time_data = _normalize_time(value[time_col])
                        for col in value.columns:
                            if col != time_col and signals_added < 4:
                                try:
                                    overview_fig.add_trace(go.Scattergl(
                                        x=time_data,
                                        y=value[col],
                                        mode='lines',
                                        name=f"{key}.{col}"
                                    ))
                                    signals_added += 1
                                except:
                                    pass
                elif isinstance(value, dict):
                    add_to_overview(value, path)
        
        add_to_overview(flight_data)
        if signals_added > 0:
            overview_fig.update_layout(title="Data Overview")
        
        # Find GPS data and create map
        gps_df = _find_dataframe(flight_data, ['GPS', 'Position', 'gps', 'GLOBAL_POSITION_INT', 'GPS_RAW_INT'])
        if gps_df is not None:
            lat_col = next((c for c in gps_df.columns if c.lower() in ['lat', 'latitude']), None)
            lon_col = next((c for c in gps_df.columns if c.lower() in ['lon', 'lng', 'longitude']), None)
            
            if lat_col and lon_col:
                map_fig = _create_map_figure(gps_df, lat_col, lon_col)
            
            alt_col = next((c for c in gps_df.columns if c.lower() in ['altitude', 'alt', 'relative_alt']), None)
            if alt_col:
                altitude_fig = _create_altitude_figure(gps_df, alt_col)
        
        # Find attitude data (includes Gyroscope for sample data)
        attitude_df = _find_dataframe(flight_data, ['ATTITUDE', 'Attitude', 'orientation', 'Gyroscope', 'gyro'])
        if attitude_df is not None:
            attitude_fig = _create_attitude_figure(attitude_df)
        
        # Find IMU data
        imu_df = _find_dataframe(flight_data, ['Accelerometer', 'IMU', 'accel', 'RAW_IMU', 'SCALED_IMU'])
        if imu_df is not None:
            imu_fig = _create_imu_figure(imu_df)
        
        # Find battery data
        battery_df = _find_dataframe(flight_data, ['Battery', 'battery', 'Power', 'SYS_STATUS', 'BATTERY_STATUS'])
        if battery_df is not None:
            battery_fig = _create_battery_figure(battery_df)
    
    # ALWAYS create all plot containers so callbacks can update them
    return html.Div([
        # Main overview plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=overview_fig, id='overview-plot',
                         config={'displayModeBar': True, 'scrollZoom': True},
                         style={'height': '300px'})
            ], width=12)
        ], className='mb-2'),
        # Grid of specific plots - always present
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=map_fig, id='overview-map',
                         config={'displayModeBar': True, 'scrollZoom': True},
                         style={'height': '350px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=altitude_fig, id='overview-altitude',
                         config={'displayModeBar': True},
                         style={'height': '350px'})
            ], width=6),
        ], className='g-2 mb-2'),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=attitude_fig, id='overview-attitude',
                         config={'displayModeBar': True},
                         style={'height': '350px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=imu_fig, id='overview-imu',
                         config={'displayModeBar': True},
                         style={'height': '350px'})
            ], width=6),
        ], className='g-2 mb-2'),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=battery_fig, id='overview-battery',
                         config={'displayModeBar': True},
                         style={'height': '350px'})
            ], width=6),
        ], className='g-2'),
    ], className='p-2')


def _find_time_column(df):
    """Find the time/timestamp column in a DataFrame."""
    time_cols = ['timestamp', 'time', 'time_boot_ms', 'time_usec', 'TimeUS']
    for col in time_cols:
        if col in df.columns:
            return col
    return None


def _normalize_time(time_series):
    """Normalize time series to seconds from start."""
    import numpy as np
    if len(time_series) == 0:
        return time_series
    
    time_data = time_series.copy()
    
    # Check if time is in microseconds (> 1e12)
    if time_data.max() > 1e12:
        time_data = (time_data - time_data.min()) / 1e6
    # Check if time is in milliseconds (> 1e9)
    elif time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e3
    # Otherwise assume it's already in reasonable units
    elif time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3
    
    return time_data


def _create_timeseries_tab(flight_data) -> html.Div:
    """Create time series tab with updateable content."""
    return html.Div(id='timeseries-tab-content', children=_build_timeseries_content(flight_data))


def _build_timeseries_content(flight_data):
    """Build the time series tab content."""
    if not flight_data:
        return dbc.Alert([
            html.I(className='fas fa-info-circle me-2'),
            "No data loaded. Load flight data to view time series plots."
        ], color='info', className='m-3')
    
    plots = []
    
    # Motors plot
    motors_df = _find_dataframe(flight_data, ['Motors', 'motor', 'Motor', 'SERVO_OUTPUT_RAW', 'RC_CHANNELS'])
    if motors_df is not None:
        fig = _create_motors_figure(motors_df)
        plots.append(dbc.Col([
            dcc.Graph(figure=fig, id='ts-motors', style={'height': '300px'})
        ], width=12))
    
    # Control signals
    control_df = _find_dataframe(flight_data, ['FlightController', 'Control', 'control', 'RC_CHANNELS', 'MANUAL_CONTROL'])
    if control_df is not None:
        fig = _create_control_figure(control_df)
        plots.append(dbc.Col([
            dcc.Graph(figure=fig, id='ts-control', style={'height': '300px'})
        ], width=12))
    
    # Gyroscope
    gyro_df = _find_dataframe(flight_data, ['Gyroscope', 'gyro', 'Gyro', 'RAW_IMU', 'SCALED_IMU'])
    if gyro_df is not None:
        fig = _create_gyro_figure(gyro_df)
        plots.append(dbc.Col([
            dcc.Graph(figure=fig, id='ts-gyro', style={'height': '300px'})
        ], width=12))
    
    # If no specific plots found, create generic signal plots
    if not plots:
        # Find any DataFrame and plot its signals
        all_dfs = _get_all_dataframes(flight_data)
        for name, df in list(all_dfs.items())[:3]:
            fig = _create_generic_timeseries_figure(df, name)
            plots.append(dbc.Col([
                dcc.Graph(figure=fig, style={'height': '300px'})
            ], width=12))
    
    if not plots:
        return dbc.Alert([
            html.I(className='fas fa-exclamation-triangle me-2'),
            "No plottable time series data found."
        ], color='warning', className='m-3')
    
    return html.Div([
        dbc.Row(plots, className='g-2 p-2')
    ])


def _get_all_dataframes(data, prefix=''):
    """Get all DataFrames from nested structure."""
    import pandas as pd
    result = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, pd.DataFrame):
            result[path] = value
        elif isinstance(value, dict):
            result.update(_get_all_dataframes(value, path))
    return result


def _create_generic_timeseries_figure(df, name):
    """Create a generic time series figure for any DataFrame."""
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    
    fig = go.Figure()
    
    time_col = _find_time_column(df)
    time_data = _normalize_time(df[time_col]) if time_col else np.arange(len(df))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot first 5 numeric columns
    numeric_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    
    for i, col in enumerate(numeric_cols[:5]):
        fig.add_trace(go.Scattergl(
            x=time_data,
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=1)
        ))
    
    fig.update_layout(
        title=name,
        xaxis_title='Time (s)',
        yaxis_title='Value',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def _create_map_tab(flight_data) -> html.Div:
    """Create map tab with updateable content."""
    return html.Div(id='map-tab-content', children=_build_map_content(flight_data))


def _build_map_content(flight_data):
    """Build the map tab content."""
    import plotly.graph_objects as go
    
    if not flight_data:
        return dbc.Alert([
            html.I(className='fas fa-info-circle me-2'),
            "No data loaded. Load flight data to view map."
        ], color='info', className='m-3')
    
    gps_df = _find_dataframe(flight_data, ['GPS', 'Position', 'gps', 'GLOBAL_POSITION_INT', 'GPS_RAW_INT'])
    
    if gps_df is None:
        return dbc.Alert([
            html.I(className='fas fa-exclamation-triangle me-2'),
            "No GPS data found in the loaded data."
        ], color='warning', className='m-3')
    
    # Find lat/lon columns
    lat_col = next((c for c in gps_df.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in gps_df.columns if c.lower() in ['lon', 'lng', 'longitude']), None)
    
    if lat_col and lon_col:
        fig = _create_map_figure(gps_df, lat_col, lon_col, height=600)
        return html.Div([
            dcc.Graph(figure=fig, id='main-map', style={'height': '600px'},
                     config={'displayModeBar': True, 'scrollZoom': True})
        ], className='p-2')
    
    return dbc.Alert([
        html.I(className='fas fa-times-circle me-2'),
        f"GPS data found but no lat/lon columns. Available columns: {', '.join(gps_df.columns[:10])}"
    ], color='danger', className='m-3')


def _create_analysis_tab(flight_data) -> html.Div:
    """Create analysis tab with updateable content."""
    return html.Div(id='analysis-tab-content', children=_build_analysis_content(flight_data))


def _build_analysis_content(flight_data):
    """Build the analysis tab content."""
    if not flight_data:
        return dbc.Alert([
            html.I(className='fas fa-info-circle me-2'),
            "No data loaded. Load flight data to view analysis."
        ], color='info', className='m-3')
    
    # Create statistics cards
    stats = _calculate_flight_stats(flight_data)
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Flight Duration"),
                    dbc.CardBody([
                        html.H3(stats.get('duration', 'N/A'), className='text-primary')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Max Altitude"),
                    dbc.CardBody([
                        html.H3(stats.get('max_altitude', 'N/A'), className='text-success')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Total Distance"),
                    dbc.CardBody([
                        html.H3(stats.get('total_distance', 'N/A'), className='text-info')
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Avg Speed"),
                    dbc.CardBody([
                        html.H3(stats.get('avg_speed', 'N/A'), className='text-warning')
                    ])
                ])
            ], width=3),
        ], className='g-3 p-3'),
        
        # Events table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Flight Events"),
                    dbc.CardBody([
                        _create_events_table(flight_data)
                    ])
                ])
            ])
        ], className='p-3')
    ])


def _find_dataframe(data, keywords, prefix=''):
    """Find a DataFrame containing any of the keywords."""
    import pandas as pd
    
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            if any(kw.lower() in key.lower() for kw in keywords):
                return value
        elif isinstance(value, dict):
            result = _find_dataframe(value, keywords)
            if result is not None:
                return result
    return None


def _create_map_figure(gps_df, lat_col='lat', lon_col='lon', height=350):
    """Create a map figure from GPS data."""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Get lat/lon data, handle potential scaling (MAVLink uses 1e7 scaling)
    lat_data = gps_df[lat_col].copy()
    lon_data = gps_df[lon_col].copy()
    
    # Auto-detect and handle MAVLink scaling (values > 1000 are likely scaled)
    if lat_data.abs().max() > 1000:
        lat_data = lat_data / 1e7
    if lon_data.abs().max() > 1000:
        lon_data = lon_data / 1e7
    
    # Filter out invalid coordinates
    valid_mask = (lat_data != 0) & (lon_data != 0) & np.isfinite(lat_data) & np.isfinite(lon_data)
    lat_data = lat_data[valid_mask]
    lon_data = lon_data[valid_mask]
    
    if len(lat_data) == 0:
        fig.update_layout(title='No valid GPS data')
        return fig
    
    # Add flight path
    fig.add_trace(go.Scattermap(
        lat=lat_data,
        lon=lon_data,
        mode='lines',
        line=dict(width=3, color='#3498db'),
        name='Flight Path'
    ))
    
    # Add start marker
    fig.add_trace(go.Scattermap(
        lat=[lat_data.iloc[0]],
        lon=[lon_data.iloc[0]],
        mode='markers',
        marker=dict(size=15, color='#2ecc71'),
        name='Start'
    ))
    
    # Add end marker
    fig.add_trace(go.Scattermap(
        lat=[lat_data.iloc[-1]],
        lon=[lon_data.iloc[-1]],
        mode='markers',
        marker=dict(size=15, color='#e74c3c'),
        name='End'
    ))
    
    center_lat = lat_data.mean()
    center_lon = lon_data.mean()
    
    fig.update_layout(
        map=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=14
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title='Flight Path',
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig


def _create_altitude_figure(gps_df, alt_col='altitude'):
    """Create altitude over time figure."""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Get altitude data
    alt_data = gps_df[alt_col].copy()
    
    # Handle MAVLink scaling (relative_alt is in mm)
    if alt_col == 'relative_alt' or alt_data.abs().max() > 10000:
        alt_data = alt_data / 1000  # Convert mm to m
    
    # Get timestamp - try different column names
    time_col = 'timestamp'
    if 'timestamp' not in gps_df.columns:
        for col in ['time_boot_ms', 'time_usec', 'time']:
            if col in gps_df.columns:
                time_col = col
                break
    
    time_data = gps_df[time_col] if time_col in gps_df.columns else np.arange(len(alt_data))
    
    # Normalize time if in microseconds or milliseconds
    if isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e6  # usec to sec
    elif isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3  # ms to sec
    
    fig.add_trace(go.Scatter(
        x=time_data,
        y=alt_data,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#9b59b6', width=2),
        name='Altitude'
    ))
    
    fig.update_layout(
        title='Altitude Profile',
        xaxis_title='Time (s)',
        yaxis_title='Altitude (m)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def _create_imu_figure(imu_df):
    """Create IMU accelerometer figure."""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    # Try different column name patterns
    axis_patterns = [
        ['accel_x', 'accel_y', 'accel_z'],
        ['xacc', 'yacc', 'zacc'],
        ['ax', 'ay', 'az'],
        ['AccX', 'AccY', 'AccZ'],
    ]
    
    labels = ['X', 'Y', 'Z']
    
    # Get timestamp
    time_col = 'timestamp'
    if 'timestamp' not in imu_df.columns:
        for col in ['time_boot_ms', 'time_usec', 'time']:
            if col in imu_df.columns:
                time_col = col
                break
    
    time_data = imu_df[time_col] if time_col in imu_df.columns else np.arange(len(imu_df))
    
    # Normalize time if in microseconds or milliseconds
    if isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e6
    elif isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3
    
    # Find which pattern matches
    axes = None
    for pattern in axis_patterns:
        if all(col in imu_df.columns for col in pattern):
            axes = pattern
            break
    
    if axes is None:
        # Just plot any numeric columns
        numeric_cols = imu_df.select_dtypes(include=[np.number]).columns.tolist()
        axes = [c for c in numeric_cols if c != time_col][:3]
        labels = axes
    
    for col, color, label in zip(axes, colors, labels):
        if col in imu_df.columns:
            fig.add_trace(go.Scatter(
                x=time_data,
                y=imu_df[col],
                mode='lines',
                line=dict(color=color, width=1),
                name=f'{label}'
            ))
    
    fig.update_layout(
        title='Accelerometer Data',
        xaxis_title='Time (s)',
        yaxis_title='Acceleration',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def _create_attitude_figure(attitude_df):
    """Create attitude (roll, pitch, yaw) or gyroscope figure."""
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    
    fig = go.Figure()
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    # Get timestamp
    time_col = 'timestamp'
    if 'timestamp' not in attitude_df.columns:
        for col in ['time_boot_ms', 'time_usec', 'time']:
            if col in attitude_df.columns:
                time_col = col
                break
    
    time_data = attitude_df[time_col] if time_col in attitude_df.columns else np.arange(len(attitude_df))
    
    # Normalize time
    if len(time_data) > 0 and isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e6
    elif len(time_data) > 0 and isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3
    
    traces_added = 0
    title = 'Attitude / Angular Rates'
    
    # Try to find roll, pitch, yaw columns first
    for col, color, label in [('roll', '#e74c3c', 'Roll'), ('pitch', '#2ecc71', 'Pitch'), ('yaw', '#3498db', 'Yaw')]:
        if col in attitude_df.columns:
            values = attitude_df[col].copy()
            if pd.api.types.is_numeric_dtype(values):
                # Convert from radians to degrees if values are small
                if values.abs().max() < 10:
                    values = np.degrees(values)
                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=values,
                    mode='lines',
                    line=dict(color=color, width=1),
                    name=label
                ))
                traces_added += 1
                title = 'Attitude (Roll, Pitch, Yaw)'
    
    # If no attitude columns, try gyroscope columns (gyro_x, gyro_y, gyro_z)
    if traces_added == 0:
        for col, color, label in [('gyro_x', '#e74c3c', 'Gyro X'), ('gyro_y', '#2ecc71', 'Gyro Y'), ('gyro_z', '#3498db', 'Gyro Z')]:
            if col in attitude_df.columns:
                values = attitude_df[col].copy()
                if pd.api.types.is_numeric_dtype(values):
                    fig.add_trace(go.Scatter(
                        x=time_data,
                        y=values,
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=label
                    ))
                    traces_added += 1
                    title = 'Gyroscope (Angular Rates)'
    
    # If still no traces, try any numeric columns
    if traces_added == 0:
        numeric_cols = [c for c in attitude_df.columns if c != time_col and pd.api.types.is_numeric_dtype(attitude_df[c])]
        for i, col in enumerate(numeric_cols[:3]):
            fig.add_trace(go.Scatter(
                x=time_data,
                y=attitude_df[col],
                mode='lines',
                line=dict(color=colors[i], width=1),
                name=col
            ))
            traces_added += 1
            title = 'Angular Data'
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Value',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def _create_battery_figure(battery_df):
    """Create battery voltage figure."""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Get timestamp
    time_col = 'timestamp'
    if 'timestamp' not in battery_df.columns:
        for col in ['time_boot_ms', 'time_usec', 'time']:
            if col in battery_df.columns:
                time_col = col
                break
    
    time_data = battery_df[time_col] if time_col in battery_df.columns else np.arange(len(battery_df))
    
    # Normalize time
    if isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e6
    elif isinstance(time_data.iloc[0], (int, float)) and time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3
    
    # Try to find voltage column
    voltage_col = next((c for c in battery_df.columns if 'voltage' in c.lower() or 'volt' in c.lower()), None)
    if voltage_col:
        voltage = battery_df[voltage_col].copy()
        # Handle mV to V conversion
        if voltage.max() > 100:
            voltage = voltage / 1000
        fig.add_trace(go.Scatter(
            x=time_data,
            y=voltage,
            mode='lines',
            line=dict(color='#f39c12', width=2),
            name='Voltage'
        ))
    
    # Try to find remaining/percentage column
    remaining_col = next((c for c in battery_df.columns if 'remain' in c.lower() or 'percent' in c.lower() or 'soc' in c.lower()), None)
    if remaining_col:
        fig.add_trace(go.Scatter(
            x=time_data,
            y=battery_df[remaining_col],
            mode='lines',
            line=dict(color='#2ecc71', width=2),
            name='Remaining %',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title='Battery Status',
        xaxis_title='Time (s)',
        yaxis_title='Voltage (V)',
        yaxis2=dict(title='%', overlaying='y', side='right'),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=40, b=40),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def _create_motors_figure(motors_df):
    """Create motors figure."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, col in enumerate([c for c in motors_df.columns if 'motor' in c.lower()]):
        fig.add_trace(go.Scatter(
            x=motors_df['timestamp'],
            y=motors_df[col],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1),
            name=col
        ))
    
    fig.update_layout(
        title='Motor Commands',
        xaxis_title='Time (s)',
        yaxis_title='PWM',
        template='plotly_white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def _create_control_figure(control_df):
    """Create control signals figure."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    signal_colors = {
        'roll': '#e74c3c',
        'pitch': '#2ecc71',
        'yaw': '#3498db',
        'throttle': '#f39c12'
    }
    
    for col in ['roll', 'pitch', 'yaw', 'throttle']:
        if col in control_df.columns:
            fig.add_trace(go.Scatter(
                x=control_df['timestamp'],
                y=control_df[col],
                mode='lines',
                line=dict(color=signal_colors.get(col, '#888'), width=1),
                name=col.capitalize()
            ))
    
    fig.update_layout(
        title='Flight Control Signals',
        xaxis_title='Time (s)',
        yaxis_title='Value',
        template='plotly_white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def _create_gyro_figure(gyro_df):
    """Create gyroscope figure."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for col, color, label in zip(['gyro_x', 'gyro_y', 'gyro_z'], colors, ['X', 'Y', 'Z']):
        if col in gyro_df.columns:
            fig.add_trace(go.Scatter(
                x=gyro_df['timestamp'],
                y=gyro_df[col],
                mode='lines',
                line=dict(color=color, width=1),
                name=f'Gyro {label}'
            ))
    
    fig.update_layout(
        title='Gyroscope Data',
        xaxis_title='Time (s)',
        yaxis_title='Angular Rate (rad/s)',
        template='plotly_white',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def _calculate_flight_stats(flight_data):
    """Calculate flight statistics."""
    import numpy as np
    
    stats = {}
    
    # Find GPS data
    gps_df = _find_dataframe(flight_data, ['GPS', 'Position'])
    if gps_df is not None:
        if 'timestamp' in gps_df.columns:
            duration = gps_df['timestamp'].max() - gps_df['timestamp'].min()
            mins = int(duration // 60)
            secs = int(duration % 60)
            stats['duration'] = f"{mins}m {secs}s"
        
        if 'altitude' in gps_df.columns:
            stats['max_altitude'] = f"{gps_df['altitude'].max():.1f} m"
        
        if 'ground_speed' in gps_df.columns:
            stats['avg_speed'] = f"{gps_df['ground_speed'].mean():.1f} m/s"
        
        # Calculate total distance
        if 'lat' in gps_df.columns and 'lon' in gps_df.columns:
            from ..utils.geo import cumulative_distance
            try:
                distances = cumulative_distance(
                    gps_df['lat'].values, 
                    gps_df['lon'].values
                )
                total_dist = distances[-1] if len(distances) > 0 else 0
                if total_dist > 1000:
                    stats['total_distance'] = f"{total_dist/1000:.2f} km"
                else:
                    stats['total_distance'] = f"{total_dist:.0f} m"
            except:
                stats['total_distance'] = "N/A"
    
    return stats


def _create_events_table(flight_data):
    """Create events table."""
    import pandas as pd
    
    events_df = _find_dataframe(flight_data, ['Events', 'events', 'Event'])
    
    if events_df is None or events_df.empty:
        return html.P("No events recorded", className='text-muted')
    
    rows = []
    for _, event in events_df.iterrows():
        severity_color = {
            'info': 'info',
            'warning': 'warning', 
            'error': 'danger',
            'critical': 'danger'
        }.get(event.get('severity', 'info'), 'secondary')
        
        rows.append(html.Tr([
            html.Td(f"{event.get('timestamp', 0):.1f}s"),
            html.Td(dbc.Badge(event.get('event_type', ''), color=severity_color)),
            html.Td(event.get('description', ''))
        ]))
    
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Time"),
            html.Th("Type"),
            html.Th("Description")
        ])),
        html.Tbody(rows)
    ], striped=True, hover=True, size='sm')


def _create_status_bar(flight_data) -> html.Div:
    """Create status bar."""
    data_info = "No data loaded"
    if flight_data:
        df_count = _count_dataframes(flight_data)
        data_info = f"Loaded: {df_count} DataFrames"
    
    return html.Footer([
        dbc.Row([
            dbc.Col([
                html.Small(id='status-message', children="Ready", className='text-muted')
            ]),
            dbc.Col([
                html.Small(id='data-info', children=data_info, className='text-muted text-center')
            ]),
            dbc.Col([
                html.Small(id='memory-usage', children="", className='text-muted text-end')
            ]),
        ], className='px-3 py-2 bg-light border-top')
    ])


def _count_dataframes(data):
    """Count DataFrames in hierarchy."""
    import pandas as pd
    count = 0
    for value in data.values():
        if isinstance(value, pd.DataFrame):
            count += 1
        elif isinstance(value, dict):
            count += _count_dataframes(value)
    return count


def _create_export_modal() -> dbc.Modal:
    """Create export options modal."""
    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className='fas fa-download me-2'),
            dbc.ModalTitle("Export Data")
        ]),
        dbc.ModalBody([
            # Export format selection
            dbc.Label("Export Format", className='fw-bold'),
            dbc.RadioItems(
                id='export-format',
                options=[
                    {'label': ' CSV (Comma Separated Values)', 'value': 'csv'},
                    {'label': ' Excel (Multiple Sheets)', 'value': 'excel'},
                    {'label': ' JSON (JavaScript Object Notation)', 'value': 'json'},
                    {'label': ' MATLAB (.mat)', 'value': 'matlab'},
                ],
                value='csv',
                className='mb-3'
            ),
            
            html.Hr(),
            
            # Export scope
            dbc.Label("Export Scope", className='fw-bold'),
            dbc.RadioItems(
                id='export-scope',
                options=[
                    {'label': ' All Data', 'value': 'all'},
                    {'label': ' Selected Signals Only', 'value': 'selected'},
                ],
                value='all',
                className='mb-3'
            ),
            
            html.Hr(),
            
            # Additional options
            dbc.Label("Additional Options", className='fw-bold'),
            dbc.Checklist(
                id='export-options',
                options=[
                    {'label': ' Include computed signals', 'value': 'computed'},
                    {'label': ' Include statistics', 'value': 'stats'},
                    {'label': ' Include timestamps', 'value': 'timestamps'},
                ],
                value=['timestamps'],
                className='mb-3'
            ),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id='btn-export-close', color='secondary', outline=True),
            dbc.Button([html.I(className='fas fa-download me-2'), "Export"], 
                      id='btn-export-confirm', color='success'),
        ])
    ], id='export-modal', is_open=False, size='md')


def _create_load_modal() -> dbc.Modal:
    """Create load/import options modal.
    
    REQ-INT-001 to REQ-INT-008: Import capabilities
    REQ-INT-006: Support drag-and-drop file import
    REQ-INT-007: Provide import wizard with auto-detection
    """
    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className='fas fa-folder-open me-2'),
            dbc.ModalTitle("Load Flight Data")
        ]),
        dbc.ModalBody([
            # Tabs for file vs folder import
            dbc.Tabs([
                dbc.Tab([
                    # Drag and drop upload area
                    dcc.Upload(
                        id='file-upload',
                        children=html.Div([
                            html.I(className='fas fa-cloud-upload-alt fa-3x mb-3', style={'color': '#3498db'}),
                            html.H5("Drag & Drop or Click to Select", className='text-light'),
                            html.P("Supported formats: CSV, Excel, MATLAB (.mat), MAVLink (.tlog), PX4 (.ulg), ArduPilot (.bin)", 
                                   className='text-muted small')
                        ], className='text-center py-4'),
                        style={
                            'width': '100%',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'borderColor': '#3498db',
                            'backgroundColor': 'rgba(52, 152, 219, 0.1)',
                            'cursor': 'pointer',
                            'marginBottom': '15px'
                        },
                        multiple=True  # Allow multiple files
                    ),
                ], label="Files", tab_id="tab-files", className='pt-3'),
                
                dbc.Tab([
                    # Folder path input
                    html.Div([
                        html.I(className='fas fa-folder-tree fa-3x mb-3', style={'color': '#f39c12'}),
                        html.H5("Import from Folder", className='text-light'),
                        html.P("Recursively imports all CSV files from a folder structure", 
                               className='text-muted small mb-3')
                    ], className='text-center py-3'),
                    
                    dbc.Label("Folder Path", className='fw-bold'),
                    dbc.InputGroup([
                        dbc.Input(
                            id='folder-path-input',
                            type='text',
                            placeholder='C:\\Users\\...\\Flight_001 or paste folder path',
                            className='bg-dark text-light'
                        ),
                        dbc.Button(
                            [html.I(className='fas fa-folder-open')],
                            id='btn-browse-folder',
                            color='secondary',
                            title='Browse...'
                        ),
                    ], className='mb-3'),
                    
                    dbc.Label("File Extensions", className='small text-muted'),
                    dbc.Checklist(
                        id='folder-extensions',
                        options=[
                            {'label': ' .csv', 'value': '.csv'},
                            {'label': ' .txt', 'value': '.txt'},
                            {'label': ' .tsv', 'value': '.tsv'},
                        ],
                        value=['.csv'],
                        inline=True,
                        className='mb-3'
                    ),
                    
                    dbc.Checklist(
                        id='folder-options',
                        options=[
                            {'label': ' Include root folder name in hierarchy', 'value': 'include_root'},
                            {'label': ' Flatten structure (dot-separated paths)', 'value': 'flatten'},
                        ],
                        value=[],
                        className='mb-3'
                    ),
                ], label="Folder", tab_id="tab-folder", className='pt-3'),
            ], id='load-tabs', active_tab='tab-files'),
            
            # Upload status
            html.Div(id='upload-status', className='mb-3'),
            
            html.Hr(),
            
            # Import options
            dbc.Label("Import Options", className='fw-bold'),
            dbc.Checklist(
                id='import-options',
                options=[
                    {'label': ' Auto-detect format', 'value': 'auto'},
                    {'label': ' Merge with existing data', 'value': 'merge'},
                    {'label': ' Validate timestamps', 'value': 'validate'},
                    {'label': ' Apply signal assignment', 'value': 'apply_assignment'},
                ],
                value=['auto', 'validate'],
                className='mb-3'
            ),
            
            # Signal assignment config selector (shown when apply_assignment is checked)
            html.Div([
                dbc.Label("Signal Assignment Config", className='fw-bold'),
                dbc.Select(
                    id='assignment-config-select',
                    options=[
                        {'label': '(None - keep original columns)', 'value': ''},
                    ],
                    value='',
                    className='mb-2'
                ),
                dbc.Button(
                    [html.I(className='fas fa-cog me-2'), "Configure Assignments"],
                    id='btn-open-assignment',
                    color='info',
                    size='sm',
                    outline=True
                ),
            ], id='assignment-config-section', style={'display': 'none'}),
            
            html.Hr(),
            
            # Timestamp column config
            dbc.Label("Timestamp Column", className='fw-bold mt-2'),
            dbc.Input(
                id='timestamp-column-input',
                type='text',
                value='timestamp',
                placeholder='timestamp',
                size='sm',
                className='mb-3'
            ),
            
            # Recent files (if any)
            html.Div(id='recent-files-list'),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id='btn-load-close', color='secondary', outline=True),
            dbc.Button([html.I(className='fas fa-folder-plus me-2'), "Import Folder"], 
                      id='btn-load-folder', color='warning', style={'display': 'none'}),
            dbc.Button([html.I(className='fas fa-check me-2'), "Load Selected"], 
                      id='btn-load-confirm', color='primary', disabled=True),
        ])
    ], id='load-modal', is_open=False, size='lg')


def _create_assignment_modal() -> dbc.Modal:
    """Create signal assignment configuration modal.
    
    Allows mapping CSV columns to standard flight dynamics variables.
    """
    from .signal_assignment_presets import STANDARD_SIGNALS, CONVERSION_PRESETS
    
    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className='fas fa-link me-2'),
            dbc.ModalTitle("Signal Assignment Configuration")
        ]),
        dbc.ModalBody([
            # Config name
            dbc.Row([
                dbc.Col([
                    dbc.Label("Configuration Name", className='fw-bold'),
                    dbc.Input(
                        id='assignment-name',
                        type='text',
                        placeholder='My Custom Mapping',
                        value='',
                        className='mb-3'
                    ),
                ], width=8),
                dbc.Col([
                    dbc.Label("Version", className='fw-bold'),
                    dbc.Input(
                        id='assignment-version',
                        type='text',
                        value='1.0',
                        className='mb-3',
                        size='sm'
                    ),
                ], width=4),
            ]),
            
            dbc.Label("Description", className='fw-bold'),
            dbc.Textarea(
                id='assignment-description',
                placeholder='Optional description for this mapping configuration...',
                rows=2,
                className='mb-3'
            ),
            
            html.Hr(),
            
            # Column mappings header
            html.Div([
                dbc.Label("Column Mappings", className='fw-bold'),
                dbc.Button(
                    [html.I(className='fas fa-magic me-2'), "Auto-Suggest"],
                    id='btn-auto-suggest',
                    color='info',
                    size='sm',
                    outline=True,
                    className='float-end'
                ),
            ], className='d-flex justify-content-between align-items-center mb-3'),
            
            # Detected columns (will be populated from loaded data)
            html.Div([
                dbc.Alert(
                    [html.I(className='fas fa-info-circle me-2'),
                     "Load data first to see available columns, or add mappings manually below."],
                    color='info',
                    className='mb-3'
                ),
            ], id='detected-columns-info'),
            
            # Mapping list (scrollable)
            html.Div([
                # Header row
                dbc.Row([
                    dbc.Col(html.Strong("Source Column"), width=2),
                    dbc.Col(html.Strong("Source Type"), width=2),
                    dbc.Col(html.Strong("Target Signal"), width=3),
                    dbc.Col(html.Strong("Conversion"), width=2),
                    dbc.Col(html.Strong("Unit"), width=2),
                    dbc.Col(width=1),
                ], className='mb-2 text-muted small'),
                
                # Mapping rows container
                html.Div(id='mapping-rows-container', children=[
                    # Will be populated dynamically
                ]),
                
            ], style={'maxHeight': '300px', 'overflowY': 'auto'}, className='mb-3'),
            
            # Add mapping button
            dbc.Button(
                [html.I(className='fas fa-plus me-2'), "Add Mapping"],
                id='btn-add-mapping',
                color='success',
                size='sm',
                outline=True,
                className='mb-3'
            ),
            
            html.Hr(),
            
            # Save/Load config section
            html.Div([
                dbc.Label("Load Existing Config", className='fw-bold'),
                dbc.InputGroup([
                    dbc.Select(
                        id='saved-configs-list',
                        options=[
                            {'label': '(Select saved config)', 'value': ''},
                        ],
                        value='',
                    ),
                    dbc.Button(
                        [html.I(className='fas fa-upload')],
                        id='btn-import-config',
                        color='secondary',
                        title='Import from file'
                    ),
                ], className='mb-3'),
            ]),
            
        ]),
        dbc.ModalFooter([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className='fas fa-save me-2'), "Save Config"],
                    id='btn-save-assignment',
                    color='success',
                    outline=True
                ),
                dbc.Button(
                    [html.I(className='fas fa-download me-2'), "Export"],
                    id='btn-export-assignment',
                    color='info',
                    outline=True
                ),
            ], className='me-auto'),
            dbc.Button("Cancel", id='btn-assignment-close', color='secondary', outline=True),
            dbc.Button(
                [html.I(className='fas fa-check me-2'), "Apply"],
                id='btn-assignment-apply',
                color='primary'
            ),
        ])
    ], id='assignment-modal', is_open=False, size='xl')
