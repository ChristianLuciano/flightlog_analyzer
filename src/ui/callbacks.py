"""
Dash callbacks for UI interactivity.

Defines all callback functions for handling user interactions
and updating the dashboard state.
"""

from dash import Input, Output, State, callback, ctx, no_update, ALL, MATCH, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import logging

logger = logging.getLogger(__name__)


def register_callbacks(app):
    """
    Register all callbacks with the Dash app.

    Args:
        app: Dash application instance.
    """
    
    # NOTE: Always use app.flight_data inside callbacks to get current data
    # Do NOT capture flight_data at registration time as it becomes stale

    # ========================================
    # Signal Selection Callbacks
    # ========================================

    @app.callback(
        Output('signal-checklist', 'options'),
        Input('signal-search', 'value'),
        State('signal-checklist', 'options'),
        prevent_initial_call=True
    )
    def filter_signals(search_value, all_options):
        """Filter signals based on search input."""
        current_data = getattr(app, 'flight_data', {})
        if not search_value:
            return _build_signal_options(current_data)
        
        search_lower = search_value.lower()
        original_options = _build_signal_options(current_data)
        filtered = [
            opt for opt in original_options 
            if search_lower in opt['label'].lower() or search_lower in opt['value'].lower()
        ]
        return filtered

    @app.callback(
        Output('signal-count', 'children'),
        Input('signal-checklist', 'value')
    )
    def update_signal_count(selected):
        """Update the signal count display."""
        count = len(selected) if selected else 0
        return f"{count} signal{'s' if count != 1 else ''} selected"

    # ========================================
    # Data Summary Collapse Toggle
    # ========================================

    @app.callback(
        Output('data-summary-collapse', 'is_open'),
        Output('data-summary-chevron', 'style'),
        Input('data-summary-toggle', 'n_clicks'),
        State('data-summary-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_data_summary(n_clicks, is_open):
        """Toggle data summary collapse."""
        if not n_clicks:
            raise PreventUpdate
        
        new_is_open = not is_open
        chevron_style = {
            'transition': 'transform 0.3s',
            'transform': 'rotate(0deg)' if new_is_open else 'rotate(-90deg)'
        }
        return new_is_open, chevron_style

    @app.callback(
        Output('signal-selector-collapse', 'is_open'),
        Output('signal-selector-chevron', 'style'),
        Input('signal-selector-toggle', 'n_clicks'),
        State('signal-selector-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_signal_selector(n_clicks, is_open):
        """Toggle signal selector collapse."""
        if not n_clicks:
            raise PreventUpdate
        
        new_is_open = not is_open
        chevron_style = {
            'transition': 'transform 0.3s',
            'transform': 'rotate(0deg)' if new_is_open else 'rotate(-90deg)'
        }
        return new_is_open, chevron_style

    @app.callback(
        Output('plot-builder-collapse', 'is_open'),
        Output('plot-builder-chevron', 'style'),
        Input('plot-builder-toggle', 'n_clicks'),
        State('plot-builder-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_plot_builder(n_clicks, is_open):
        """Toggle plot builder collapse."""
        if not n_clicks:
            raise PreventUpdate
        
        new_is_open = not is_open
        chevron_style = {
            'transition': 'transform 0.3s',
            'transform': 'rotate(0deg)' if new_is_open else 'rotate(-90deg)'
        }
        return new_is_open, chevron_style

    @app.callback(
        Output('quick-select-collapse', 'is_open'),
        Output('quick-select-chevron', 'style'),
        Input('quick-select-toggle', 'n_clicks'),
        State('quick-select-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_quick_select(n_clicks, is_open):
        """Toggle quick select collapse."""
        if not n_clicks:
            raise PreventUpdate
        
        new_is_open = not is_open
        chevron_style = {
            'transition': 'transform 0.3s',
            'transform': 'rotate(0deg)' if new_is_open else 'rotate(-90deg)'
        }
        return new_is_open, chevron_style

    @app.callback(
        Output('events-filter-collapse', 'is_open'),
        Output('events-filter-chevron', 'style'),
        Input('events-filter-toggle', 'n_clicks'),
        State('events-filter-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_events_filter(n_clicks, is_open):
        """Toggle events filter collapse."""
        if not n_clicks:
            raise PreventUpdate
        
        new_is_open = not is_open
        chevron_style = {
            'transition': 'transform 0.3s',
            'transform': 'rotate(0deg)' if new_is_open else 'rotate(-90deg)'
        }
        return new_is_open, chevron_style

    # ========================================
    # Conversion Preset Callback
    # ========================================
    
    @app.callback(
        Output('conversion-scale', 'value'),
        Output('conversion-offset', 'value'),
        Output('conversion-unit', 'value'),
        Input('conversion-preset', 'value'),
        prevent_initial_call=True
    )
    def update_conversion_from_preset(preset):
        """Update scale/offset/unit based on preset selection."""
        presets = {
            'none': (1.0, 0.0, ''),
            # Length
            'mm_to_m': (0.001, 0.0, 'm'),
            'cm_to_m': (0.01, 0.0, 'm'),
            'm_to_km': (0.001, 0.0, 'km'),
            'in_to_m': (0.0254, 0.0, 'm'),
            'ft_to_m': (0.3048, 0.0, 'm'),
            # Angles
            'rad_to_deg': (57.2957795, 0.0, 'deg'),
            'deg_to_rad': (0.01745329, 0.0, 'rad'),
            'mrad_to_deg': (0.0572958, 0.0, 'deg'),
            # MAVLink
            'mavlink_latlon': (1e-7, 0.0, 'deg'),
            'mavlink_alt': (0.001, 0.0, 'm'),
            # Time
            'us_to_s': (1e-6, 0.0, 's'),
            'ms_to_s': (0.001, 0.0, 's'),
            # Velocity
            'cms_to_ms': (0.01, 0.0, 'm/s'),
            'ms_to_kmh': (3.6, 0.0, 'km/h'),
            'knots_to_ms': (0.5144, 0.0, 'm/s'),
            # Custom - don't change values
            'custom': (no_update, no_update, no_update),
        }
        
        if preset in presets:
            return presets[preset]
        return 1.0, 0.0, ''

    # ========================================
    # Custom Plot Management
    # ========================================

    @app.callback(
        Output('custom-plots-list', 'children'),
        Output('plot-counter', 'data'),
        Input('btn-add-plot', 'n_clicks'),
        Input('btn-clear-plots', 'n_clicks'),
        Input({'type': 'remove-plot-btn', 'index': ALL}, 'n_clicks'),
        State('signal-checklist', 'value'),
        State('plot-type-select', 'value'),
        State('time-range-start', 'value'),
        State('time-range-end', 'value'),
        State('conversion-scale', 'value'),
        State('conversion-offset', 'value'),
        State('conversion-unit', 'value'),
        State('custom-plots-list', 'children'),
        State('plot-counter', 'data'),
        prevent_initial_call=True
    )
    def manage_custom_plots(add_clicks, clear_clicks, remove_clicks, 
                            selected_signals, plot_type, time_start, time_end,
                            conv_scale, conv_offset, conv_unit,
                            current_plots, counter):
        """Add, remove, or clear custom plots."""
        trigger = ctx.triggered_id
        
        if current_plots is None:
            current_plots = []
        if counter is None:
            counter = 0
        
        # Clear all plots
        if trigger == 'btn-clear-plots':
            return [], 0
        
        # Remove specific plot
        if isinstance(trigger, dict) and trigger.get('type') == 'remove-plot-btn':
            plot_index = trigger.get('index')
            current_plots = [p for p in current_plots 
                           if not (isinstance(p, dict) and p.get('props', {}).get('id', {}).get('index') == plot_index)]
            return current_plots, counter
        
        # Add new plot
        if trigger == 'btn-add-plot':
            if not selected_signals:
                return current_plots, counter
            
            counter += 1
            current_data = getattr(app, 'flight_data', {})
            
            # Build conversion config
            conversion = None
            if conv_scale is not None and conv_scale != 1.0:
                conversion = {
                    'scale': float(conv_scale) if conv_scale else 1.0,
                    'offset': float(conv_offset) if conv_offset else 0.0,
                    'unit': conv_unit or ''
                }
            elif conv_offset is not None and conv_offset != 0.0:
                conversion = {
                    'scale': float(conv_scale) if conv_scale else 1.0,
                    'offset': float(conv_offset) if conv_offset else 0.0,
                    'unit': conv_unit or ''
                }
            
            new_plot = _create_plot_card(
                plot_id=counter,
                signals=selected_signals,
                plot_type=plot_type or 'timeseries',
                time_start=time_start,
                time_end=time_end,
                flight_data=current_data,
                conversion=conversion
            )
            current_plots.append(new_plot)
            return current_plots, counter
        
        return current_plots, counter

    # ========================================
    # Quick Action Callbacks
    # ========================================

    @app.callback(
        Output('signal-checklist', 'value'),
        Input('btn-quick-imu', 'n_clicks'),
        Input('btn-quick-gps', 'n_clicks'),
        Input('btn-quick-battery', 'n_clicks'),
        Input('btn-quick-motors', 'n_clicks'),
        Input('btn-quick-control', 'n_clicks'),
        Input('btn-clear-selection', 'n_clicks'),
        Input('btn-select-all', 'n_clicks'),
        State('signal-checklist', 'value'),
        State('signal-checklist', 'options'),
        prevent_initial_call=True
    )
    def quick_select_signals(imu_click, gps_click, battery_click, motors_click, control_click, 
                             clear_click, select_all_click, current_selection, current_options):
        """Quickly select or deselect signal groups."""
        trigger = ctx.triggered_id
        current_data = getattr(app, 'flight_data', {})
        all_signals = _build_signal_options(current_data)
        signal_values = [s['value'] for s in all_signals]
        
        new_selection = current_selection or []
        
        # Clear all selections
        if trigger == 'btn-clear-selection':
            return []
        
        # Select all visible signals
        if trigger == 'btn-select-all':
            if current_options:
                return [opt['value'] for opt in current_options]
            return signal_values
        
        if trigger == 'btn-quick-imu':
            imu_signals = [s for s in signal_values if 'IMU' in s or 'accel' in s.lower() or 'gyro' in s.lower() or 'xacc' in s.lower()]
            new_selection = list(set(new_selection + imu_signals))
        elif trigger == 'btn-quick-gps':
            gps_signals = [s for s in signal_values if 'GPS' in s or 'lat' in s.lower() or 'lon' in s.lower() or 'alt' in s.lower() or 'POSITION' in s]
            new_selection = list(set(new_selection + gps_signals))
        elif trigger == 'btn-quick-battery':
            battery_signals = [s for s in signal_values if 'Battery' in s or 'batt' in s.lower() or 'volt' in s.lower() or 'SYS_STATUS' in s]
            new_selection = list(set(new_selection + battery_signals))
        elif trigger == 'btn-quick-motors':
            motor_signals = [s for s in signal_values if 'Motor' in s or 'motor' in s.lower() or 'SERVO' in s or 'RC_CHANNELS' in s]
            new_selection = list(set(new_selection + motor_signals))
        elif trigger == 'btn-quick-control':
            control_signals = [s for s in signal_values if 'Control' in s or 'Flight' in s or 'ATTITUDE' in s or 'MANUAL' in s]
            new_selection = list(set(new_selection + control_signals))
        
        return new_selection

    # ========================================
    # Export Callbacks
    # ========================================

    @app.callback(
        Output('export-modal', 'is_open'),
        Input('btn-export', 'n_clicks'),
        Input('btn-export-close', 'n_clicks'),
        Input('btn-export-confirm', 'n_clicks'),
        State('export-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_export_modal(export_click, close_click, confirm_click, is_open):
        """Open/close export modal."""
        trigger = ctx.triggered_id
        if trigger == 'btn-export':
            return True
        return False

    @app.callback(
        Output('export-download', 'data'),
        Output('status-message', 'children'),
        Input('btn-export-confirm', 'n_clicks'),
        State('export-format', 'value'),
        State('export-scope', 'value'),
        State('signal-checklist', 'value'),
        State('time-range-start', 'value'),
        State('time-range-end', 'value'),
        prevent_initial_call=True
    )
    def perform_export(n_clicks, format_type, scope, selected_signals, time_start, time_end):
        """Actually perform the data export."""
        if not n_clicks:
            raise PreventUpdate
        
        current_data = getattr(app, 'flight_data', {})
        
        try:
            # Gather data to export
            if scope == 'selected' and selected_signals:
                export_data = {}
                for signal_path in selected_signals:
                    data = _get_signal_data(current_data, signal_path, time_start, time_end)
                    if data is not None:
                        timestamps, values = data
                        df_name = signal_path.rsplit('.', 1)[0]
                        if df_name not in export_data:
                            export_data[df_name] = {'timestamp': timestamps}
                        export_data[df_name][signal_path.split('.')[-1]] = values
            else:
                export_data = _flatten_flight_data(current_data)
            
            if not export_data:
                return no_update, "No data to export"
            
            # Create export based on format
            if format_type == 'csv':
                all_dfs = []
                for name, data in export_data.items():
                    df = pd.DataFrame(data)
                    df['source'] = name
                    all_dfs.append(df)
                
                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    return dcc.send_data_frame(combined.to_csv, "flight_data.csv", index=False), "Export complete: CSV"
            
            elif format_type == 'excel':
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for name, data in export_data.items():
                        df = pd.DataFrame(data)
                        sheet_name = name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                return dcc.send_bytes(buffer.getvalue(), "flight_data.xlsx"), "Export complete: Excel"
            
            elif format_type == 'json':
                json_data = {}
                for name, data in export_data.items():
                    json_data[name] = {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v 
                                       for k, v in data.items()}
                return dcc.send_string(json.dumps(json_data, indent=2, default=str), "flight_data.json"), "Export complete: JSON"
            
            elif format_type == 'matlab':
                from src.export.data_export import export_matlab
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
                    dfs = {name: pd.DataFrame(data) for name, data in export_data.items()}
                    export_matlab(dfs, f.name)
                    f.seek(0)
                    return dcc.send_file(f.name, filename="flight_data.mat"), "Export complete: MATLAB"
            
            return no_update, f"Exported data ({format_type})"
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return no_update, f"Export failed: {str(e)}"

    # ========================================
    # Load/Import Callbacks
    # ========================================

    @app.callback(
        Output('load-modal', 'is_open'),
        Input('btn-load', 'n_clicks'),
        Input('btn-load-close', 'n_clicks'),
        Input('btn-load-confirm', 'n_clicks'),
        Input('btn-load-folder', 'n_clicks'),
        State('load-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_load_modal(load_click, close_click, confirm_click, folder_click, is_open):
        """Open/close load modal."""
        trigger = ctx.triggered_id
        if trigger == 'btn-load':
            return True
        return False

    # Show/hide folder import button based on active tab
    @app.callback(
        Output('btn-load-folder', 'style'),
        Output('btn-load-confirm', 'style'),
        Input('load-tabs', 'active_tab'),
        prevent_initial_call=True
    )
    def toggle_load_buttons(active_tab):
        """Show folder button on folder tab, file button on files tab."""
        if active_tab == 'tab-folder':
            return {'display': 'inline-block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'inline-block'}

    # Show/hide assignment config section
    @app.callback(
        Output('assignment-config-section', 'style'),
        Input('import-options', 'value'),
        prevent_initial_call=True
    )
    def toggle_assignment_section(options):
        """Show assignment config when 'apply_assignment' is checked."""
        if options and 'apply_assignment' in options:
            return {'display': 'block'}
        return {'display': 'none'}

    # Folder import handler
    @app.callback(
        Output('upload-status', 'children', allow_duplicate=True),
        Output('btn-load-confirm', 'disabled', allow_duplicate=True),
        Output('app-state', 'data', allow_duplicate=True),
        Input('btn-load-folder', 'n_clicks'),
        State('folder-path-input', 'value'),
        State('folder-extensions', 'value'),
        State('folder-options', 'value'),
        State('import-options', 'value'),
        State('app-state', 'data'),
        prevent_initial_call=True
    )
    def handle_folder_import(n_clicks, folder_path, extensions, folder_opts, import_opts, app_state):
        """Handle folder import."""
        if not n_clicks or not folder_path:
            raise PreventUpdate
        
        try:
            from src.data.folder_importer import FolderImporter
            
            # Check if folder exists
            folder = Path(folder_path)
            if not folder.exists():
                return dbc.Alert([
                    html.I(className='fas fa-exclamation-triangle me-2'),
                    f"Folder not found: {folder_path}"
                ], color='danger'), True, app_state or {}
            
            if not folder.is_dir():
                return dbc.Alert([
                    html.I(className='fas fa-exclamation-triangle me-2'),
                    f"Path is not a directory: {folder_path}"
                ], color='danger'), True, app_state or {}
            
            # Import from folder
            importer = FolderImporter(extensions=extensions or ['.csv'])
            include_root = 'include_root' in (folder_opts or [])
            flatten = 'flatten' in (folder_opts or [])
            
            data = importer.import_folder(
                folder_path,
                include_root_name=include_root,
                flatten=flatten
            )
            
            summary = importer.get_summary()
            
            # Convert DataFrames to JSON-serializable format
            loaded_data = {}
            def serialize_data(d, prefix=''):
                for key, value in d.items():
                    path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, pd.DataFrame):
                        loaded_data[path] = value.to_dict('records')
                    elif isinstance(value, dict):
                        serialize_data(value, path)
            
            serialize_data(data)
            
            # Update app state
            if app_state is None:
                app_state = {}
            app_state['uploaded_data'] = loaded_data
            app_state['folder_path'] = folder_path
            
            # Build status message
            status_items = [
                dbc.Alert([
                    html.I(className='fas fa-folder-open me-2'),
                    html.Strong(f"Imported from: {folder.name}/"),
                    html.Br(),
                    f"✓ {summary['loaded_count']} files loaded"
                ], color='success', className='mb-2')
            ]
            
            if summary['failed_count'] > 0:
                status_items.append(dbc.Alert([
                    html.I(className='fas fa-exclamation-triangle me-2'),
                    f"⚠ {summary['failed_count']} files failed"
                ], color='warning', className='mb-1'))
            
            return html.Div(status_items), False, app_state
            
        except Exception as e:
            logger.error(f"Folder import error: {e}")
            return dbc.Alert([
                html.I(className='fas fa-times-circle me-2'),
                f"Import error: {str(e)}"
            ], color='danger'), True, app_state or {}

    @app.callback(
        Output('upload-status', 'children'),
        Output('btn-load-confirm', 'disabled'),
        Output('app-state', 'data'),
        Input('file-upload', 'contents'),
        State('file-upload', 'filename'),
        State('import-options', 'value'),
        State('timestamp-column-input', 'value'),
        State('app-state', 'data'),
        prevent_initial_call=True
    )
    def handle_file_upload(contents_list, filenames, import_options, ts_column, app_state):
        """Handle file upload and parse data."""
        if contents_list is None:
            raise PreventUpdate
        
        if not isinstance(contents_list, list):
            contents_list = [contents_list]
            filenames = [filenames]
        
        results = []
        loaded_data = {}
        auto_detect = 'auto' in (import_options or [])
        
        for contents, filename in zip(contents_list, filenames):
            try:
                # Decode the file content
                content_type, content_string = contents.split(',')
                import base64
                decoded = base64.b64decode(content_string)
                
                # Determine file type and parse
                ext = Path(filename).suffix.lower()
                
                if ext == '.csv':
                    import io
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    loaded_data[filename] = df
                    results.append(dbc.Alert([
                        html.I(className='fas fa-check-circle me-2'),
                        f"✓ {filename}: {len(df):,} rows, {len(df.columns)} columns"
                    ], color='success', className='mb-1 py-2'))
                    
                elif ext in ['.xlsx', '.xls']:
                    import io
                    excel_file = pd.ExcelFile(io.BytesIO(decoded))
                    for sheet in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet)
                        loaded_data[f"{filename}.{sheet}"] = df
                    results.append(dbc.Alert([
                        html.I(className='fas fa-check-circle me-2'),
                        f"✓ {filename}: {len(excel_file.sheet_names)} sheets loaded"
                    ], color='success', className='mb-1 py-2'))
                    
                elif ext == '.mat':
                    try:
                        from scipy.io import loadmat
                        import io
                        mat_data = loadmat(io.BytesIO(decoded), squeeze_me=True)
                        for key, value in mat_data.items():
                            if not key.startswith('_'):
                                if isinstance(value, np.ndarray):
                                    if value.ndim == 2:
                                        df = pd.DataFrame(value)
                                        loaded_data[f"{filename}.{key}"] = df
                        results.append(dbc.Alert([
                            html.I(className='fas fa-check-circle me-2'),
                            f"✓ {filename}: MATLAB file loaded"
                        ], color='success', className='mb-1 py-2'))
                    except Exception as e:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-times-circle me-2'),
                            f"✗ {filename}: {str(e)}"
                        ], color='danger', className='mb-1 py-2'))
                        
                elif ext == '.json':
                    import io
                    json_data = json.loads(decoded.decode('utf-8'))
                    if isinstance(json_data, dict):
                        for key, value in json_data.items():
                            if isinstance(value, (list, dict)):
                                df = pd.DataFrame(value)
                                loaded_data[f"{filename}.{key}"] = df
                    results.append(dbc.Alert([
                        html.I(className='fas fa-check-circle me-2'),
                        f"✓ {filename}: JSON file loaded"
                    ], color='success', className='mb-1 py-2'))
                
                elif ext == '.tlog':
                    # MAVLink telemetry log
                    try:
                        from pymavlink import mavutil
                        import tempfile
                        import os
                        
                        # Write to temp file (mavutil needs file path)
                        tmp_path = tempfile.mktemp(suffix='.tlog')
                        with open(tmp_path, 'wb') as tmp:
                            tmp.write(decoded)
                        
                        mlog = mavutil.mavlink_connection(tmp_path)
                        messages = {}
                        
                        while True:
                            msg = mlog.recv_match(blocking=False)
                            if msg is None:
                                break
                            msg_type = msg.get_type()
                            if msg_type == 'BAD_DATA':
                                continue
                            if msg_type not in messages:
                                messages[msg_type] = []
                            msg_dict = msg.to_dict()
                            msg_dict['timestamp'] = getattr(msg, '_timestamp', 0)
                            messages[msg_type].append(msg_dict)
                        
                        # Close the connection before deleting
                        mlog.close()
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass  # Ignore cleanup errors
                        
                        for msg_type, msg_list in messages.items():
                            if msg_list and len(msg_list) > 10:  # Only include message types with data
                                df = pd.DataFrame(msg_list)
                                loaded_data[f"{filename}.{msg_type}"] = df
                        
                        results.append(dbc.Alert([
                            html.I(className='fas fa-check-circle me-2'),
                            f"✓ {filename}: MAVLink log loaded ({len(messages)} message types)"
                        ], color='success', className='mb-1 py-2'))
                        
                    except ImportError:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-exclamation-triangle me-2'),
                            f"⚠ {filename}: pymavlink not installed. Run: pip install pymavlink"
                        ], color='warning', className='mb-1 py-2'))
                    except Exception as e:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-times-circle me-2'),
                            f"✗ {filename}: Error parsing MAVLink - {str(e)}"
                        ], color='danger', className='mb-1 py-2'))
                
                elif ext == '.ulg':
                    # PX4 ULog format
                    try:
                        from pyulog import ULog
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(suffix='.ulg', delete=False) as tmp:
                            tmp.write(decoded)
                            tmp_path = tmp.name
                        
                        ulog = ULog(tmp_path)
                        
                        import os
                        os.unlink(tmp_path)
                        
                        for data in ulog.data_list:
                            df_data = {'timestamp': data.data['timestamp']}
                            for field in data.data:
                                if field != 'timestamp':
                                    df_data[field] = data.data[field]
                            df = pd.DataFrame(df_data)
                            loaded_data[f"{filename}.{data.name}"] = df
                        
                        results.append(dbc.Alert([
                            html.I(className='fas fa-check-circle me-2'),
                            f"✓ {filename}: PX4 ULog loaded ({len(ulog.data_list)} topics)"
                        ], color='success', className='mb-1 py-2'))
                        
                    except ImportError:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-exclamation-triangle me-2'),
                            f"⚠ {filename}: pyulog not installed. Run: pip install pyulog"
                        ], color='warning', className='mb-1 py-2'))
                    except Exception as e:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-times-circle me-2'),
                            f"✗ {filename}: Error parsing ULog - {str(e)}"
                        ], color='danger', className='mb-1 py-2'))
                
                elif ext == '.bin':
                    # ArduPilot binary log
                    try:
                        from pymavlink import DFReader
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
                            tmp.write(decoded)
                            tmp_path = tmp.name
                        
                        mlog = DFReader.DFReader_binary(tmp_path)
                        messages = {}
                        
                        while True:
                            msg = mlog.recv_msg()
                            if msg is None:
                                break
                            msg_type = msg.get_type()
                            if msg_type not in messages:
                                messages[msg_type] = []
                            msg_dict = {}
                            for field in msg._fieldnames:
                                msg_dict[field] = getattr(msg, field)
                            msg_dict['timestamp'] = msg._timestamp
                            messages[msg_type].append(msg_dict)
                        
                        import os
                        os.unlink(tmp_path)
                        
                        for msg_type, msg_list in messages.items():
                            if msg_list and len(msg_list) > 10:
                                df = pd.DataFrame(msg_list)
                                loaded_data[f"{filename}.{msg_type}"] = df
                        
                        results.append(dbc.Alert([
                            html.I(className='fas fa-check-circle me-2'),
                            f"✓ {filename}: ArduPilot log loaded ({len(messages)} message types)"
                        ], color='success', className='mb-1 py-2'))
                        
                    except ImportError:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-exclamation-triangle me-2'),
                            f"⚠ {filename}: pymavlink not installed. Run: pip install pymavlink"
                        ], color='warning', className='mb-1 py-2'))
                    except Exception as e:
                        results.append(dbc.Alert([
                            html.I(className='fas fa-times-circle me-2'),
                            f"✗ {filename}: Error parsing ArduPilot log - {str(e)}"
                        ], color='danger', className='mb-1 py-2'))
                    
                else:
                    results.append(dbc.Alert([
                        html.I(className='fas fa-exclamation-triangle me-2'),
                        f"⚠ {filename}: Unsupported format ({ext})"
                    ], color='warning', className='mb-1 py-2'))
                    
            except Exception as e:
                results.append(dbc.Alert([
                    html.I(className='fas fa-times-circle me-2'),
                    f"✗ {filename}: Error - {str(e)}"
                ], color='danger', className='mb-1 py-2'))
        
        # Update app state with loaded data info
        if app_state is None:
            app_state = {}
        
        app_state['uploaded_files'] = list(loaded_data.keys())
        app_state['uploaded_data'] = {k: v.to_dict('list') for k, v in loaded_data.items()}
        
        # Enable confirm button if we have data
        confirm_disabled = len(loaded_data) == 0
        
        return html.Div(results), confirm_disabled, app_state

    @app.callback(
        Output('status-message', 'children', allow_duplicate=True),
        Output('data-info', 'children'),
        Output('signal-checklist', 'options', allow_duplicate=True),
        Output('signal-checklist', 'value', allow_duplicate=True),
        Output('data-tree', 'children'),
        Output('custom-plots-list', 'children', allow_duplicate=True),
        Output('overview-plot', 'figure', allow_duplicate=True),
        Output('overview-map', 'figure', allow_duplicate=True),
        Output('overview-altitude', 'figure', allow_duplicate=True),
        Output('overview-attitude', 'figure', allow_duplicate=True),
        Output('overview-imu', 'figure', allow_duplicate=True),
        Output('overview-battery', 'figure', allow_duplicate=True),
        Output('timeseries-tab-content', 'children', allow_duplicate=True),
        Output('map-tab-content', 'children', allow_duplicate=True),
        Output('analysis-tab-content', 'children', allow_duplicate=True),
        Input('btn-load-confirm', 'n_clicks'),
        State('app-state', 'data'),
        prevent_initial_call=True
    )
    def confirm_load(n_clicks, app_state):
        """Confirm and apply loaded data - REPLACES old data and clears ALL plots."""
        if not n_clicks or not app_state:
            raise PreventUpdate
        
        uploaded_data = app_state.get('uploaded_data', {})
        if not uploaded_data:
            return ("No data to load",) + (no_update,) * 14
        
        # Convert back to DataFrames and organize hierarchically
        new_data = _organize_data_hierarchically(uploaded_data)
        
        # REPLACE the app's flight_data (clear old data completely)
        app.flight_data = new_data
        
        # Rebuild signal options using updated flight_data
        options = _build_signal_options(app.flight_data)
        
        # Rebuild data tree
        from src.ui.app_layout import (
            _build_data_tree_summary, 
            _build_timeseries_content,
            _build_map_content, 
            _build_analysis_content
        )
        tree = _build_data_tree_summary(app.flight_data)
        
        # Clear custom plots (start fresh with new data)
        empty_plots = [dbc.Alert([
            html.I(className='fas fa-check-circle me-2'),
            "Data loaded successfully! Select signals from the new data and add plots."
        ], color='success', className='mb-0')]
        
        # Create all overview figures with new data
        overview_fig = _create_overview_figure(app.flight_data)
        map_fig = _create_map_figure_from_data(app.flight_data)
        altitude_fig = _create_altitude_figure_from_data(app.flight_data)
        attitude_fig = _create_attitude_figure_from_data(app.flight_data)
        imu_fig = _create_imu_figure_from_data(app.flight_data)
        battery_fig = _create_battery_figure_from_data(app.flight_data)
        
        # Update other tabs
        timeseries_content = _build_timeseries_content(app.flight_data)
        map_content = _build_map_content(app.flight_data)
        analysis_content = _build_analysis_content(app.flight_data)
        
        total_dfs = _count_dataframes_in_dict(app.flight_data)
        total_rows = _count_rows_in_dict(app.flight_data)
        status = f"✓ Loaded {total_dfs} DataFrames ({total_rows:,} total rows)"
        data_info = f"{total_dfs} DataFrames | {len(options)} signals"
        
        # Clear selected signals (value=[]) to ensure fresh start
        return (status, data_info, options, [], tree, empty_plots, 
                overview_fig, map_fig, altitude_fig, attitude_fig, imu_fig, battery_fig,
                timeseries_content, map_content, analysis_content)

    # ========================================
    # Signal Assignment Modal Callbacks
    # ========================================

    @app.callback(
        Output('assignment-modal', 'is_open'),
        Input('btn-open-assignment', 'n_clicks'),
        Input('btn-assignment-close', 'n_clicks'),
        Input('btn-assignment-apply', 'n_clicks'),
        State('assignment-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_assignment_modal(open_click, close_click, apply_click, is_open):
        """Open/close signal assignment modal."""
        trigger = ctx.triggered_id
        if trigger == 'btn-open-assignment':
            return True
        return False

    @app.callback(
        Output('mapping-rows-container', 'children'),
        Input('btn-add-mapping', 'n_clicks'),
        State('mapping-rows-container', 'children'),
        prevent_initial_call=True
    )
    def add_mapping_row(n_clicks, current_rows):
        """Add a new mapping row."""
        if not n_clicks:
            raise PreventUpdate
        
        from src.ui.signal_assignment_presets import get_signal_options, get_conversion_options, get_source_options
        
        row_id = n_clicks
        new_row = dbc.Row([
            dbc.Col([
                dbc.Input(
                    id={'type': 'mapping-source', 'index': row_id},
                    placeholder='column_name',
                    size='sm'
                )
            ], width=2),
            dbc.Col([
                dbc.Select(
                    id={'type': 'mapping-source-type', 'index': row_id},
                    options=get_source_options(),
                    value='',
                    size='sm'
                )
            ], width=2),
            dbc.Col([
                dbc.Select(
                    id={'type': 'mapping-target', 'index': row_id},
                    options=get_signal_options(),
                    value='',
                    size='sm'
                )
            ], width=3),
            dbc.Col([
                dbc.Select(
                    id={'type': 'mapping-conversion', 'index': row_id},
                    options=get_conversion_options(),
                    value='none',
                    size='sm'
                )
            ], width=2),
            dbc.Col([
                dbc.Input(
                    id={'type': 'mapping-unit', 'index': row_id},
                    placeholder='unit',
                    size='sm'
                )
            ], width=2),
            dbc.Col([
                dbc.Button(
                    html.I(className='fas fa-trash'),
                    id={'type': 'mapping-delete', 'index': row_id},
                    color='danger',
                    size='sm',
                    outline=True
                )
            ], width=1),
        ], className='mb-2', id={'type': 'mapping-row', 'index': row_id})
        
        if current_rows is None:
            current_rows = []
        
        return current_rows + [new_row]

    @app.callback(
        Output('mapping-rows-container', 'children', allow_duplicate=True),
        Input({'type': 'mapping-delete', 'index': ALL}, 'n_clicks'),
        State('mapping-rows-container', 'children'),
        prevent_initial_call=True
    )
    def delete_mapping_row(delete_clicks, current_rows):
        """Delete a mapping row."""
        if not any(delete_clicks):
            raise PreventUpdate
        
        # Find which button was clicked
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate
        
        row_index = trigger['index']
        
        # Remove the row with matching index
        new_rows = [
            row for row in (current_rows or [])
            if row.get('props', {}).get('id', {}).get('index') != row_index
        ]
        
        return new_rows

    @app.callback(
        Output('mapping-rows-container', 'children', allow_duplicate=True),
        Output('assignment-name', 'value'),
        Output('assignment-description', 'value'),
        Output('assignment-version', 'value'),
        Input('btn-auto-suggest', 'n_clicks'),
        State('app-state', 'data'),
        prevent_initial_call=True
    )
    def auto_suggest_mappings(n_clicks, app_state):
        """Auto-suggest mappings based on column names and source types."""
        if not n_clicks:
            raise PreventUpdate
        
        current_data = getattr(app, 'flight_data', {})
        if not current_data:
            return [], '', '', '1.0'
        
        from src.data.signal_assignment import suggest_mappings
        from src.data.signal_groups import SignalGroupManager
        from src.ui.signal_assignment_presets import get_signal_options, get_conversion_options, get_source_options
        
        # Collect all columns with path info
        all_columns = []
        def collect_columns(data, prefix=''):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, pd.DataFrame):
                    for col in value.columns:
                        all_columns.append((col, f"{path}.{col}", path))
                elif isinstance(value, dict):
                    collect_columns(value, path)
        
        collect_columns(current_data)
        
        # Use signal group manager to detect source types
        group_manager = SignalGroupManager()
        
        # Get suggestions for all columns
        rows = []
        row_id = 0
        for col, full_path, df_path in all_columns[:20]:  # Limit to 20 suggestions
            # Create a simple DataFrame with this column to get suggestions
            temp_df = pd.DataFrame({col: [0]})
            suggestions = suggest_mappings(temp_df)
            
            suggested_target = ''
            if suggestions:
                suggested_target = suggestions[0]['suggested_signal']
            
            # Detect source type from path
            parsed = group_manager.parse_signal(full_path)
            source_type = parsed.source.value if parsed.source.value else ''
            
            row_id += 1
            new_row = dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id={'type': 'mapping-source', 'index': row_id},
                        value=col,
                        size='sm'
                    )
                ], width=2),
                dbc.Col([
                    dbc.Select(
                        id={'type': 'mapping-source-type', 'index': row_id},
                        options=get_source_options(),
                        value=source_type,
                        size='sm'
                    )
                ], width=2),
                dbc.Col([
                    dbc.Select(
                        id={'type': 'mapping-target', 'index': row_id},
                        options=get_signal_options(),
                        value=suggested_target,
                        size='sm'
                    )
                ], width=3),
                dbc.Col([
                    dbc.Select(
                        id={'type': 'mapping-conversion', 'index': row_id},
                        options=get_conversion_options(),
                        value='none',
                        size='sm'
                    )
                ], width=2),
                dbc.Col([
                    dbc.Input(
                        id={'type': 'mapping-unit', 'index': row_id},
                        placeholder='unit',
                        size='sm'
                    )
                ], width=2),
                dbc.Col([
                    dbc.Button(
                        html.I(className='fas fa-trash'),
                        id={'type': 'mapping-delete', 'index': row_id},
                        color='danger',
                        size='sm',
                        outline=True
                    )
                ], width=1),
            ], className='mb-2', id={'type': 'mapping-row', 'index': row_id})
            
            rows.append(new_row)
        
        return rows, 'Auto-generated', 'Auto-suggested mappings based on column names', '1.0'

    @app.callback(
        Output('upload-status', 'children', allow_duplicate=True),
        Input('btn-save-assignment', 'n_clicks'),
        State('assignment-name', 'value'),
        State('assignment-description', 'value'),
        State('assignment-version', 'value'),
        State({'type': 'mapping-source', 'index': ALL}, 'value'),
        State({'type': 'mapping-source-type', 'index': ALL}, 'value'),
        State({'type': 'mapping-target', 'index': ALL}, 'value'),
        State({'type': 'mapping-conversion', 'index': ALL}, 'value'),
        State({'type': 'mapping-unit', 'index': ALL}, 'value'),
        prevent_initial_call=True
    )
    def save_assignment_config(n_clicks, name, description, version,
                               sources, source_types, targets, conversions, units):
        """Save assignment configuration to file."""
        if not n_clicks or not name:
            raise PreventUpdate
        
        try:
            from src.data.signal_assignment import AssignmentConfig, SignalMapping, CONVERSION_PRESETS
            
            config = AssignmentConfig(
                name=name,
                description=description or '',
                version=version or '1.0'
            )
            
            for source, source_type, target, conv, unit in zip(sources, source_types, targets, conversions, units):
                if source and target:
                    config.add_mapping(
                        source_column=source,
                        target_signal=target,
                        conversion=conv if conv != 'none' else None,
                        signal_source=source_type or '',
                        description=f"Unit: {unit}" if unit else ''
                    )
            
            # Save to configs folder
            config_dir = Path('configs/assignments')
            config_dir.mkdir(parents=True, exist_ok=True)
            
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            config_path = config_dir / f"{safe_name}.json"
            config.save(config_path)
            
            return dbc.Alert([
                html.I(className='fas fa-check-circle me-2'),
                f"✓ Saved config: {config_path}"
            ], color='success', className='mb-2')
            
        except Exception as e:
            logger.error(f"Save assignment error: {e}")
            return dbc.Alert([
                html.I(className='fas fa-times-circle me-2'),
                f"Error saving config: {str(e)}"
            ], color='danger')

    @app.callback(
        Output('saved-configs-list', 'options'),
        Input('assignment-modal', 'is_open'),
        prevent_initial_call=True
    )
    def refresh_saved_configs(is_open):
        """Refresh list of saved assignment configs."""
        if not is_open:
            raise PreventUpdate
        
        options = [{'label': '(Select saved config)', 'value': ''}]
        
        config_dir = Path('configs/assignments')
        if config_dir.exists():
            for config_file in config_dir.glob('*.json'):
                options.append({
                    'label': config_file.stem,
                    'value': str(config_file)
                })
            for config_file in config_dir.glob('*.yaml'):
                options.append({
                    'label': config_file.stem,
                    'value': str(config_file)
                })
        
        return options


def _organize_data_hierarchically(uploaded_data):
    """Convert flat data dict to hierarchical structure based on filenames."""
    hierarchical = {}
    
    for name, data_dict in uploaded_data.items():
        df = pd.DataFrame(data_dict)
        
        # Split name by dots to create hierarchy
        parts = name.split('.')
        
        if len(parts) == 1:
            # Simple name, add directly
            hierarchical[name] = df
        else:
            # Create nested structure
            # e.g., "filename.ATTITUDE" -> {"filename": {"ATTITUDE": df}}
            current = hierarchical
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = df
    
    return hierarchical


def _count_rows_in_dict(data):
    """Count total rows in nested dict of DataFrames."""
    count = 0
    for value in data.values():
        if isinstance(value, pd.DataFrame):
            count += len(value)
        elif isinstance(value, dict):
            count += _count_rows_in_dict(value)
    return count


def _find_time_column(df):
    """Find the time/timestamp column in a DataFrame."""
    time_cols = ['timestamp', 'time', 'time_boot_ms', 'time_usec', 'TimeUS']
    for col in time_cols:
        if col in df.columns:
            return col
    return None


def _normalize_time(time_series):
    """Normalize time series to seconds from start."""
    if len(time_series) == 0:
        return time_series
    
    time_data = time_series.copy()
    
    # Check if time is in microseconds (> 1e12)
    if time_data.max() > 1e12:
        time_data = (time_data - time_data.min()) / 1e6
    # Check if time is in milliseconds (> 1e9)
    elif time_data.max() > 1e9:
        time_data = (time_data - time_data.min()) / 1e3
    elif time_data.max() > 1e6:
        time_data = (time_data - time_data.min()) / 1e3
    
    return time_data


def _empty_figure(title="No data"):
    """Create an empty figure with dark theme."""
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=40, b=40),
        title=title
    )
    return fig


def _find_dataframe(data, keywords, prefix=''):
    """Find a DataFrame containing any of the keywords."""
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            if any(kw.lower() in key.lower() for kw in keywords):
                return value
        elif isinstance(value, dict):
            result = _find_dataframe(value, keywords)
            if result is not None:
                return result
    return None


def _create_overview_figure(flight_data):
    """Create overview figure from loaded data."""
    fig = go.Figure()
    
    # Find some signals to plot
    signals_added = 0
    def find_signals(data, prefix=''):
        nonlocal signals_added
        for key, value in data.items():
            if signals_added >= 4:
                return
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, pd.DataFrame):
                time_col = _find_time_column(value)
                if time_col:
                    time_data = _normalize_time(value[time_col])
                    for col in value.columns:
                        if col != time_col and signals_added < 4:
                            try:
                                # Only plot numeric columns
                                if pd.api.types.is_numeric_dtype(value[col]):
                                    fig.add_trace(go.Scattergl(
                                        x=time_data,
                                        y=value[col],
                                        mode='lines',
                                        name=f"{key}.{col}"
                                    ))
                                    signals_added += 1
                            except:
                                pass
            elif isinstance(value, dict):
                find_signals(value, path)
    
    find_signals(flight_data)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        title="Data Overview" if signals_added > 0 else "Data Overview - No plottable signals found"
    )
    
    return fig


def _create_map_figure_from_data(flight_data):
    """Create map figure from flight data."""
    gps_df = _find_dataframe(flight_data, ['GPS', 'Position', 'gps', 'GLOBAL_POSITION_INT', 'GPS_RAW_INT'])
    
    if gps_df is None:
        return _empty_figure("Flight Path - No GPS data")
    
    lat_col = next((c for c in gps_df.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in gps_df.columns if c.lower() in ['lon', 'lng', 'longitude']), None)
    
    if not lat_col or not lon_col:
        return _empty_figure("Flight Path - No lat/lon columns found")
    
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
        return _empty_figure("Flight Path - No valid GPS coordinates")
    
    fig = go.Figure()
    
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
    
    fig.update_layout(
        map=dict(
            style='open-street-map',
            center=dict(lat=lat_data.mean(), lon=lon_data.mean()),
            zoom=14
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title='Flight Path',
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig


def _create_altitude_figure_from_data(flight_data):
    """Create altitude figure from flight data."""
    gps_df = _find_dataframe(flight_data, ['GPS', 'Position', 'gps', 'GLOBAL_POSITION_INT', 'GPS_RAW_INT'])
    
    if gps_df is None:
        return _empty_figure("Altitude - No GPS data")
    
    alt_col = next((c for c in gps_df.columns if c.lower() in ['altitude', 'alt', 'relative_alt']), None)
    
    if not alt_col:
        return _empty_figure("Altitude - No altitude column found")
    
    alt_data = gps_df[alt_col].copy()
    
    # Handle MAVLink scaling (relative_alt is in mm)
    if alt_col == 'relative_alt' or alt_data.abs().max() > 10000:
        alt_data = alt_data / 1000
    
    time_col = _find_time_column(gps_df)
    time_data = _normalize_time(gps_df[time_col]) if time_col else np.arange(len(alt_data))
    
    fig = go.Figure()
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


def _create_attitude_figure_from_data(flight_data):
    """Create attitude figure from flight data."""
    # Try to find attitude data, also check Gyroscope as fallback
    attitude_df = _find_dataframe(flight_data, ['ATTITUDE', 'Attitude', 'orientation', 'Gyroscope', 'gyro'])
    
    if attitude_df is None:
        return _empty_figure("Attitude - No attitude data")
    
    fig = go.Figure()
    
    time_col = _find_time_column(attitude_df)
    time_data = _normalize_time(attitude_df[time_col]) if time_col else np.arange(len(attitude_df))
    
    # Try roll/pitch/yaw first, then try gyro columns
    attitude_cols = [
        ('roll', '#e74c3c', 'Roll'),
        ('pitch', '#2ecc71', 'Pitch'),
        ('yaw', '#3498db', 'Yaw'),
    ]
    
    gyro_cols = [
        ('gyro_x', '#e74c3c', 'Gyro X'),
        ('gyro_y', '#2ecc71', 'Gyro Y'),
        ('gyro_z', '#3498db', 'Gyro Z'),
        ('rollspeed', '#e74c3c', 'Roll Rate'),
        ('pitchspeed', '#2ecc71', 'Pitch Rate'),
        ('yawspeed', '#3498db', 'Yaw Rate'),
    ]
    
    traces_added = 0
    
    # Try attitude columns first
    for col, color, name in attitude_cols:
        if col in attitude_df.columns:
            values = attitude_df[col].copy()
            # Convert from radians to degrees if values are small
            if pd.api.types.is_numeric_dtype(values) and values.abs().max() < 10:
                values = np.degrees(values)
            fig.add_trace(go.Scatter(
                x=time_data,
                y=values,
                mode='lines',
                line=dict(color=color, width=1),
                name=name
            ))
            traces_added += 1
    
    # If no attitude columns found, try gyro columns
    if traces_added == 0:
        for col, color, name in gyro_cols:
            if col in attitude_df.columns:
                values = attitude_df[col].copy()
                if pd.api.types.is_numeric_dtype(values):
                    fig.add_trace(go.Scatter(
                        x=time_data,
                        y=values,
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=name
                    ))
                    traces_added += 1
    
    # If still no traces, just plot first 3 numeric columns
    if traces_added == 0:
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        numeric_cols = [c for c in attitude_df.columns if c != time_col and pd.api.types.is_numeric_dtype(attitude_df[c])]
        for i, col in enumerate(numeric_cols[:3]):
            fig.add_trace(go.Scatter(
                x=time_data,
                y=attitude_df[col],
                mode='lines',
                line=dict(color=colors[i], width=1),
                name=col
            ))
    
    fig.update_layout(
        title='Attitude / Angular Rates',
        xaxis_title='Time (s)',
        yaxis_title='Value',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def _create_imu_figure_from_data(flight_data):
    """Create IMU figure from flight data."""
    imu_df = _find_dataframe(flight_data, ['Accelerometer', 'IMU', 'accel', 'RAW_IMU', 'SCALED_IMU'])
    
    if imu_df is None:
        return _empty_figure("Accelerometer - No IMU data")
    
    fig = go.Figure()
    
    time_col = _find_time_column(imu_df)
    time_data = _normalize_time(imu_df[time_col]) if time_col else np.arange(len(imu_df))
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    axis_patterns = [
        ['accel_x', 'accel_y', 'accel_z'],
        ['xacc', 'yacc', 'zacc'],
        ['ax', 'ay', 'az'],
        ['AccX', 'AccY', 'AccZ'],
    ]
    
    axes = None
    for pattern in axis_patterns:
        if all(col in imu_df.columns for col in pattern):
            axes = pattern
            break
    
    if axes is None:
        # Just plot any numeric columns
        numeric_cols = imu_df.select_dtypes(include=[np.number]).columns.tolist()
        axes = [c for c in numeric_cols if c != time_col][:3]
    
    labels = ['X', 'Y', 'Z'] if len(axes) == 3 else axes
    
    for i, col in enumerate(axes):
        if col in imu_df.columns:
            fig.add_trace(go.Scatter(
                x=time_data,
                y=imu_df[col],
                mode='lines',
                line=dict(color=colors[i % 3], width=1),
                name=labels[i] if i < len(labels) else col
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


def _create_battery_figure_from_data(flight_data):
    """Create battery figure from flight data."""
    battery_df = _find_dataframe(flight_data, ['Battery', 'battery', 'Power', 'SYS_STATUS', 'BATTERY_STATUS'])
    
    if battery_df is None:
        return _empty_figure("Battery - No battery data")
    
    fig = go.Figure()
    
    time_col = _find_time_column(battery_df)
    time_data = _normalize_time(battery_df[time_col]) if time_col else np.arange(len(battery_df))
    
    # Try to find voltage column
    voltage_col = next((c for c in battery_df.columns if 'voltage' in c.lower() or 'volt' in c.lower()), None)
    if voltage_col:
        voltage = battery_df[voltage_col].copy()
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
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def _count_dataframes_in_dict(data):
    """Count DataFrames in nested dict."""
    count = 0
    for value in data.values():
        if isinstance(value, pd.DataFrame):
            count += 1
        elif isinstance(value, dict):
            count += _count_dataframes_in_dict(value)
    return count

    # ========================================
    # Status Updates
    # ========================================

    @app.callback(
        Output('status-message', 'children', allow_duplicate=True),
        Input('btn-save', 'n_clicks'),
        prevent_initial_call=True
    )
    def update_status_save(save_click):
        """Update status message for save."""
        return "Configuration saved"


# ========================================
# Plot Creation Functions
# ========================================

def _create_plot_card(plot_id: int, signals: List[str], plot_type: str,
                      time_start: Optional[float], time_end: Optional[float],
                      flight_data: Dict, conversion: Optional[Dict] = None) -> dbc.Card:
    """
    Create a plot card with the specified configuration.
    
    Args:
        plot_id: Unique identifier for the plot
        signals: List of signal paths to plot
        plot_type: Type of plot (timeseries, xy, 3d, fft, histogram, scatter)
        time_start: Optional start time filter
        time_end: Optional end time filter
        flight_data: Hierarchical flight data dictionary
        conversion: Optional conversion config {'scale': float, 'offset': float, 'unit': str}
    """
    
    # Create figure based on plot type
    if plot_type == 'timeseries':
        fig = _create_time_series_plot(signals, flight_data, time_start, time_end, conversion)
    elif plot_type == 'xy':
        fig = _create_xy_plot(signals, flight_data, time_start, time_end, conversion)
    elif plot_type == '3d':
        fig = _create_3d_plot(signals, flight_data, time_start, time_end, conversion)
    elif plot_type == 'fft':
        fig = _create_fft_plot(signals, flight_data, time_start, time_end, conversion)
    elif plot_type == 'histogram':
        fig = _create_histogram_plot(signals, flight_data, time_start, time_end, conversion)
    elif plot_type == 'scatter':
        fig = _create_scatter_plot(signals, flight_data, time_start, time_end, conversion)
    else:
        fig = _create_time_series_plot(signals, flight_data, time_start, time_end, conversion)
    
    # Time range label
    time_label = ""
    if time_start is not None or time_end is not None:
        start_str = f"{time_start:.1f}s" if time_start else "start"
        end_str = f"{time_end:.1f}s" if time_end else "end"
        time_label = f" [{start_str} - {end_str}]"
    
    # Conversion label
    conv_label = ""
    if conversion and (conversion.get('scale', 1.0) != 1.0 or conversion.get('offset', 0.0) != 0.0):
        conv_label = f" [{conversion.get('unit', '')}]" if conversion.get('unit') else " [converted]"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span([
                    html.I(className=f'fas fa-{_get_plot_icon(plot_type)} me-2'),
                    f"Plot {plot_id}: {plot_type.title()}{time_label}{conv_label}"
                ]),
                html.Span([
                    html.Small(f"{len(signals)} signals", className='text-muted me-2'),
                    dbc.Button(
                        html.I(className='fas fa-times'),
                        id={'type': 'remove-plot-btn', 'index': plot_id},
                        color='danger',
                        size='sm',
                        outline=True,
                        className='ms-2'
                    )
                ])
            ], className='d-flex justify-content-between align-items-center')
        ]),
        dbc.CardBody([
            dcc.Graph(
                figure=fig,
                style={'height': '350px'},
                config={'displayModeBar': True, 'scrollZoom': True}
            )
        ])
    ], id={'type': 'plot-card', 'index': plot_id}, className='mb-3')


def _get_plot_icon(plot_type: str) -> str:
    """Get icon for plot type."""
    icons = {
        'timeseries': 'chart-line',
        'xy': 'chart-scatter',
        '3d': 'cube',
        'fft': 'wave-square',
        'histogram': 'chart-bar',
        'scatter': 'braille'
    }
    return icons.get(plot_type, 'chart-line')


def _apply_conversion(values, conversion: Optional[Dict]):
    """Apply scale and offset conversion to values."""
    if conversion is None:
        return values
    
    scale = conversion.get('scale', 1.0)
    offset = conversion.get('offset', 0.0)
    
    if scale == 1.0 and offset == 0.0:
        return values
    
    return values * scale + offset


def _create_time_series_plot(signals: List[str], flight_data: Dict,
                             time_start: Optional[float], time_end: Optional[float],
                             conversion: Optional[Dict] = None,
                             use_grouping: bool = True) -> go.Figure:
    """Create time series plot with optional conversion and signal grouping.
    
    When use_grouping=True, signals are grouped by base type and styled
    to distinguish measurement, command, estimated, etc. sources.
    """
    fig = go.Figure()
    
    unit_label = conversion.get('unit', '') if conversion else ''
    
    if use_grouping and len(signals) > 1:
        # Use signal grouping for coordinated styling
        from src.data.signal_groups import SignalGroupManager
        
        manager = SignalGroupManager()
        groups = manager.group_signals(signals)
        
        for base_signal, group in groups.items():
            for parsed in group.signals:
                data = _get_signal_data(flight_data, parsed.full_path, time_start, time_end)
                if data is not None:
                    timestamps, values = data
                    values = _apply_conversion(values, conversion)
                    
                    # Get styled configuration for this signal
                    config = manager.get_plot_config(parsed.full_path, group.color_index)
                    
                    hover_unit = f" {unit_label}" if unit_label else ""
                    source_label = config['name']
                    
                    fig.add_trace(go.Scattergl(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name=source_label,
                        line=config['line'],
                        opacity=config.get('opacity', 1.0),
                        legendgroup=config.get('legendgroup', ''),
                        legendgrouptitle_text=config.get('legendgrouptitle_text'),
                        hovertemplate=f'<b>{source_label}</b><br>Time: %{{x:.2f}}s<br>Value: %{{y:.4f}}{hover_unit}<extra></extra>'
                    ))
    else:
        # Simple plotting without grouping
        for signal_path in signals:
            data = _get_signal_data(flight_data, signal_path, time_start, time_end)
            if data is not None:
                timestamps, values = data
                values = _apply_conversion(values, conversion)
                
                # Update hover template with unit if provided
                hover_unit = f" {unit_label}" if unit_label else ""
                fig.add_trace(go.Scattergl(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=signal_path.split('.')[-1],
                    hovertemplate=f'<b>{signal_path}</b><br>Time: %{{x:.2f}}s<br>Value: %{{y:.4f}}{hover_unit}<extra></extra>'
                ))
    
    y_axis_title = f'Value ({unit_label})' if unit_label else 'Value'
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_title='Time (s)',
        yaxis_title=y_axis_title
    )
    
    return fig


def _create_xy_plot(signals: List[str], flight_data: Dict,
                    time_start: Optional[float], time_end: Optional[float],
                    conversion: Optional[Dict] = None) -> go.Figure:
    """Create X-Y scatter plot with optional conversion."""
    fig = go.Figure()
    
    unit_label = conversion.get('unit', '') if conversion else ''
    
    if len(signals) >= 2:
        x_data = _get_signal_data(flight_data, signals[0], time_start, time_end)
        y_data = _get_signal_data(flight_data, signals[1], time_start, time_end)
        
        if x_data and y_data:
            _, x_values = x_data
            _, y_values = y_data
            
            # Apply conversion
            x_values = _apply_conversion(x_values, conversion)
            y_values = _apply_conversion(y_values, conversion)
            
            # Ensure same length
            min_len = min(len(x_values), len(y_values))
            
            fig.add_trace(go.Scattergl(
                x=x_values[:min_len],
                y=y_values[:min_len],
                mode='markers',
                name=f'{signals[0].split(".")[-1]} vs {signals[1].split(".")[-1]}',
                marker=dict(size=4, opacity=0.6)
            ))
    
    x_title = signals[0].split('.')[-1] if signals else 'X'
    y_title = signals[1].split('.')[-1] if len(signals) > 1 else 'Y'
    if unit_label:
        x_title += f' ({unit_label})'
        y_title += f' ({unit_label})'
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    
    return fig


def _create_3d_plot(signals: List[str], flight_data: Dict,
                    time_start: Optional[float], time_end: Optional[float],
                    conversion: Optional[Dict] = None) -> go.Figure:
    """Create 3D scatter/line plot with optional conversion."""
    fig = go.Figure()
    unit_label = conversion.get('unit', '') if conversion else ''
    
    if len(signals) >= 3:
        x_data = _get_signal_data(flight_data, signals[0], time_start, time_end)
        y_data = _get_signal_data(flight_data, signals[1], time_start, time_end)
        z_data = _get_signal_data(flight_data, signals[2], time_start, time_end)
        
        if x_data and y_data and z_data:
            _, x_values = x_data
            _, y_values = y_data
            _, z_values = z_data
            
            # Apply conversion
            x_values = _apply_conversion(x_values, conversion)
            y_values = _apply_conversion(y_values, conversion)
            z_values = _apply_conversion(z_values, conversion)
            
            min_len = min(len(x_values), len(y_values), len(z_values))
            
            fig.add_trace(go.Scatter3d(
                x=x_values[:min_len],
                y=y_values[:min_len],
                z=z_values[:min_len],
                mode='lines',
                name='3D Path',
                line=dict(width=3, color=np.arange(min_len), colorscale='Viridis')
            ))
    
    suffix = f' ({unit_label})' if unit_label else ''
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title=(signals[0].split('.')[-1] if signals else 'X') + suffix,
            yaxis_title=(signals[1].split('.')[-1] if len(signals) > 1 else 'Y') + suffix,
            zaxis_title=(signals[2].split('.')[-1] if len(signals) > 2 else 'Z') + suffix,
        )
    )
    
    return fig


def _create_fft_plot(signals: List[str], flight_data: Dict,
                     time_start: Optional[float], time_end: Optional[float],
                     conversion: Optional[Dict] = None) -> go.Figure:
    """Create FFT frequency domain plot."""
    fig = go.Figure()
    
    for signal_path in signals[:3]:  # Limit to 3 signals for FFT
        data = _get_signal_data(flight_data, signal_path, time_start, time_end)
        if data is not None:
            timestamps, values = data
            
            # Calculate sample rate
            if len(timestamps) > 1:
                dt = np.mean(np.diff(timestamps))
                sample_rate = 1.0 / dt if dt > 0 else 100
            else:
                sample_rate = 100
            
            # Compute FFT
            n = len(values)
            fft_values = np.fft.rfft(values - np.mean(values))
            freqs = np.fft.rfftfreq(n, 1/sample_rate)
            magnitude = np.abs(fft_values) * 2 / n
            
            fig.add_trace(go.Scattergl(
                x=freqs,
                y=magnitude,
                mode='lines',
                name=signal_path.split('.')[-1]
            ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        xaxis_type='log',
        yaxis_type='log'
    )
    
    return fig


def _create_histogram_plot(signals: List[str], flight_data: Dict,
                           time_start: Optional[float], time_end: Optional[float],
                           conversion: Optional[Dict] = None) -> go.Figure:
    """Create histogram plot with optional conversion."""
    fig = go.Figure()
    unit_label = conversion.get('unit', '') if conversion else ''
    
    for signal_path in signals:
        data = _get_signal_data(flight_data, signal_path, time_start, time_end)
        if data is not None:
            _, values = data
            values = _apply_conversion(values, conversion)
            
            fig.add_trace(go.Histogram(
                x=values,
                name=signal_path.split('.')[-1],
                opacity=0.7
            ))
    
    x_title = f'Value ({unit_label})' if unit_label else 'Value'
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        barmode='overlay',
        xaxis_title=x_title,
        yaxis_title='Count'
    )
    
    return fig


def _create_scatter_plot(signals: List[str], flight_data: Dict,
                         time_start: Optional[float], time_end: Optional[float],
                         conversion: Optional[Dict] = None) -> go.Figure:
    """Create scatter plot with points and optional conversion."""
    fig = go.Figure()
    unit_label = conversion.get('unit', '') if conversion else ''
    
    for signal_path in signals:
        data = _get_signal_data(flight_data, signal_path, time_start, time_end)
        if data is not None:
            timestamps, values = data
            values = _apply_conversion(values, conversion)
            
            fig.add_trace(go.Scattergl(
                x=timestamps,
                y=values,
                mode='markers',
                name=signal_path.split('.')[-1],
                marker=dict(size=3, opacity=0.5)
            ))
    
    y_title = f'Value ({unit_label})' if unit_label else 'Value'
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title='Time (s)',
        yaxis_title=y_title
    )
    
    return fig


# ========================================
# Helper Functions
# ========================================

def _build_signal_options(data, prefix=''):
    """Recursively build signal options from hierarchical data."""
    options = []
    
    if not data:
        return options
    
    # Columns to exclude (time-related columns)
    time_columns = {'timestamp', 'time', 'time_boot_ms', 'time_usec', 'TimeUS', 'mavpackettype'}
    
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                if col.lower() not in {c.lower() for c in time_columns}:
                    # Only include numeric columns
                    if pd.api.types.is_numeric_dtype(value[col]):
                        full_path = f"{path}.{col}"
                        options.append({
                            'label': f"{path} → {col}",
                            'value': full_path
                        })
        elif isinstance(value, dict):
            options.extend(_build_signal_options(value, path))
    
    return options


def _get_signal_data(data, signal_path, time_start=None, time_end=None):
    """Get signal data from hierarchical structure with optional time filtering."""
    parts = signal_path.split('.')
    signal_name = parts[-1]
    df_path = parts[:-1]
    
    current = data
    for part in df_path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    if isinstance(current, pd.DataFrame):
        if signal_name in current.columns:
            # Find time column
            time_col = _find_time_column(current)
            if time_col:
                timestamps = _normalize_time(current[time_col]).values
            else:
                timestamps = np.arange(len(current))
            
            values = current[signal_name].values
            
            # Apply time filtering
            if time_start is not None or time_end is not None:
                mask = np.ones(len(timestamps), dtype=bool)
                if time_start is not None:
                    mask &= timestamps >= time_start
                if time_end is not None:
                    mask &= timestamps <= time_end
                timestamps = timestamps[mask]
                values = values[mask]
            
            return timestamps, values
    
    return None


def _flatten_flight_data(data, prefix=''):
    """Flatten hierarchical flight data."""
    result = {}
    
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, pd.DataFrame):
            result[path] = value.to_dict('list')
        elif isinstance(value, dict):
            result.update(_flatten_flight_data(value, path))
    
    return result
