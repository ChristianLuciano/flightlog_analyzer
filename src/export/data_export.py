"""
Data export functionality.

Provides export to CSV, Excel, MATLAB, and other data formats.

Covers requirements:
- REQ-EXP-010: Export visible data range as CSV
- REQ-EXP-011: Export computed signals as CSV
- REQ-EXP-013: Export statistical summaries as CSV/Excel
- REQ-INT-009: Export to MATLAB .mat format
- REQ-INT-010: Export to Octave format
- REQ-INT-011: Export to Python Jupyter notebook format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import logging
import json

from ..core.exceptions import ExportError

logger = logging.getLogger(__name__)


def export_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    include_index: bool = False,
    **kwargs
) -> None:
    """
    Export DataFrame to CSV.

    Args:
        df: DataFrame to export.
        path: Output path.
        include_index: Whether to include index column.
        **kwargs: Additional pandas to_csv arguments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=include_index, **kwargs)
        logger.info(f"Exported CSV to {path} ({len(df)} rows)")
    except Exception as e:
        raise ExportError(f"Failed to export CSV: {e}")


def export_excel(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    path: Union[str, Path],
    include_index: bool = False
) -> None:
    """
    Export DataFrame(s) to Excel.

    Args:
        data: Single DataFrame or dict of sheet_name: DataFrame.
        path: Output path.
        include_index: Whether to include index column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name='Data', index=include_index)
            else:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=include_index)

        logger.info(f"Exported Excel to {path}")

    except ImportError:
        raise ExportError("openpyxl not installed. Install with: pip install openpyxl")
    except Exception as e:
        raise ExportError(f"Failed to export Excel: {e}")


def export_signals(
    signals: Dict[str, pd.Series],
    timestamps: pd.Series,
    path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Export multiple signals to file.

    Args:
        signals: Dict of signal_name: series.
        timestamps: Timestamp series.
        path: Output path.
        format: Output format ('csv', 'excel').
    """
    # Build DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    for name, series in signals.items():
        df[name] = series.values

    if format == 'csv':
        export_csv(df, path)
    elif format in ['excel', 'xlsx']:
        export_excel(df, path)
    else:
        raise ExportError(f"Unsupported format: {format}")


def export_statistics(
    stats: Dict[str, Dict[str, float]],
    path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Export statistics summary to file.

    Args:
        stats: Dict of signal_name: statistics_dict.
        path: Output path.
        format: Output format.
    """
    df = pd.DataFrame(stats).T
    df.index.name = 'signal'

    if format == 'csv':
        export_csv(df, path, include_index=True)
    elif format in ['excel', 'xlsx']:
        export_excel(df, path, include_index=True)
    else:
        raise ExportError(f"Unsupported format: {format}")


def export_matlab(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    path: Union[str, Path],
    variable_name: str = 'flight_data'
) -> None:
    """
    Export data to MATLAB .mat format.
    
    REQ-INT-009: Export to MATLAB .mat format
    
    Args:
        data: Single DataFrame or dict of DataFrames.
        path: Output path (.mat file).
        variable_name: MATLAB variable name for single DataFrame.
    """
    try:
        from scipy.io import savemat
    except ImportError:
        raise ExportError("scipy not installed. Install with: pip install scipy")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        mat_dict = {}
        
        if isinstance(data, pd.DataFrame):
            # Convert single DataFrame to MATLAB struct
            mat_dict[variable_name] = _dataframe_to_matlab_struct(data)
        else:
            # Convert dict of DataFrames
            for name, df in data.items():
                # Clean name for MATLAB variable (alphanumeric + underscore)
                clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
                clean_name = clean_name.lstrip('0123456789')  # Can't start with number
                if not clean_name:
                    clean_name = f'data_{hash(name) % 10000}'
                mat_dict[clean_name] = _dataframe_to_matlab_struct(df)
        
        savemat(str(path), mat_dict)
        logger.info(f"Exported MATLAB file to {path}")
        
    except Exception as e:
        raise ExportError(f"Failed to export MATLAB file: {e}")


def _dataframe_to_matlab_struct(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Convert DataFrame to MATLAB-compatible dict of arrays."""
    result = {}
    for col in df.columns:
        # Clean column name for MATLAB
        clean_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in col)
        clean_col = clean_col.lstrip('0123456789')
        if not clean_col:
            clean_col = f'col_{hash(col) % 10000}'
        
        # Convert to numpy array
        values = df[col].values
        if pd.api.types.is_datetime64_any_dtype(values):
            # Convert datetime to numeric (seconds since epoch)
            values = values.astype('datetime64[s]').astype(np.float64)
        elif pd.api.types.is_object_dtype(values):
            # Convert strings to object array
            values = values.astype(str)
        
        result[clean_col] = values
    
    return result


def export_octave(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    path: Union[str, Path],
    variable_name: str = 'flight_data'
) -> None:
    """
    Export data to Octave format (same as MATLAB .mat v5).
    
    REQ-INT-010: Export to Octave format
    
    Args:
        data: Single DataFrame or dict of DataFrames.
        path: Output path (.mat file).
        variable_name: Variable name for single DataFrame.
    """
    # Octave can read MATLAB v5 format
    export_matlab(data, path, variable_name)
    logger.info(f"Exported Octave-compatible file to {path}")


def export_jupyter_notebook(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    path: Union[str, Path],
    title: str = "Flight Log Analysis",
    include_plots: bool = True
) -> None:
    """
    Export data as Python Jupyter notebook.
    
    REQ-INT-011: Export to Python Jupyter notebook format
    
    Args:
        data: Single DataFrame or dict of DataFrames.
        path: Output path (.ipynb file).
        title: Notebook title.
        include_plots: Whether to include plotting code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", "\nAutomatically generated from Flight Log Dashboard"]
    })
    
    # Import cell
    import_code = """import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
"""
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": import_code.split('\n'),
        "outputs": [],
        "execution_count": None
    })
    
    # Data loading cell with embedded data
    if isinstance(data, pd.DataFrame):
        data_dict = {'flight_data': data}
    else:
        data_dict = data
    
    for name, df in data_dict.items():
        # Clean name
        var_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        var_name = var_name.lstrip('0123456789') or 'df'
        
        # Create DataFrame from dict representation
        df_dict = df.to_dict('list')
        data_code = f"# {name}\n{var_name} = pd.DataFrame({json.dumps(df_dict, default=str)})\n"
        data_code += f"print(f'{var_name}: {{len({var_name})}} rows, {{{var_name}.columns.tolist()}}')\n"
        data_code += f"{var_name}.head()"
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {name}\n"]
        })
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "source": data_code.split('\n'),
            "outputs": [],
            "execution_count": None
        })
        
        # Add plotting code if requested
        if include_plots and len(df.columns) > 1:
            timestamp_col = None
            for col in ['timestamp', 'time', 't']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != timestamp_col][:3]
                
                if numeric_cols:
                    plot_code = f"""# Time series plot
fig = go.Figure()
"""
                    for col in numeric_cols:
                        plot_code += f"fig.add_trace(go.Scatter(x={var_name}['{timestamp_col}'], y={var_name}['{col}'], name='{col}', mode='lines'))\n"
                    
                    plot_code += f"""fig.update_layout(title='{name}', xaxis_title='Time', yaxis_title='Value', template='plotly_dark')
fig.show()"""
                    
                    cells.append({
                        "cell_type": "code",
                        "metadata": {},
                        "source": plot_code.split('\n'),
                        "outputs": [],
                        "execution_count": None
                    })
    
    # Statistics cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Statistics\n"]
    })
    
    stats_code = "# Calculate statistics for all numeric columns\n"
    for name, df in data_dict.items():
        var_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        var_name = var_name.lstrip('0123456789') or 'df'
        stats_code += f"print('\\n{name} Statistics:')\n"
        stats_code += f"display({var_name}.describe())\n"
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": stats_code.split('\n'),
        "outputs": [],
        "execution_count": None
    })
    
    # Build notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "cells": cells
    }
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, default=str)
        logger.info(f"Exported Jupyter notebook to {path}")
    except Exception as e:
        raise ExportError(f"Failed to export Jupyter notebook: {e}")


def export_fft_results(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    phases: Optional[np.ndarray] = None,
    path: Union[str, Path] = None,
    format: str = 'csv'
) -> None:
    """
    Export FFT results to file.
    
    REQ-EXP-012: Export FFT results (frequencies, magnitudes, phases)
    
    Args:
        frequencies: Frequency array.
        magnitudes: Magnitude array.
        phases: Optional phase array.
        path: Output path.
        format: Output format ('csv', 'excel', 'mat').
    """
    df = pd.DataFrame({
        'frequency_hz': frequencies,
        'magnitude': magnitudes
    })
    
    if phases is not None:
        df['phase_rad'] = phases
    
    path = Path(path)
    
    if format == 'csv':
        export_csv(df, path)
    elif format in ['excel', 'xlsx']:
        export_excel(df, path)
    elif format in ['mat', 'matlab']:
        export_matlab(df, path, variable_name='fft_results')
    else:
        raise ExportError(f"Unsupported format: {format}")


def export_computed_signals(
    signals: Dict[str, pd.Series],
    definitions: Dict[str, Dict[str, Any]],
    path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Export computed signal definitions (shareable).
    
    REQ-EXP-015: Export computed signal definitions
    
    Args:
        signals: Dict of signal_name: computed series.
        definitions: Dict of signal definitions (formula, inputs, etc.).
        path: Output path.
        format: Output format ('json', 'yaml').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        'version': '1.0',
        'computed_signals': {}
    }
    
    for name, definition in definitions.items():
        export_data['computed_signals'][name] = {
            'formula': definition.get('formula', ''),
            'inputs': definition.get('inputs', []),
            'unit': definition.get('unit', ''),
            'description': definition.get('description', ''),
            'parameters': definition.get('parameters', {})
        }
    
    try:
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'yaml':
            try:
                import yaml
                with open(path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            except ImportError:
                raise ExportError("pyyaml not installed. Install with: pip install pyyaml")
        else:
            raise ExportError(f"Unsupported format: {format}")
        
        logger.info(f"Exported computed signal definitions to {path}")
    except Exception as e:
        raise ExportError(f"Failed to export computed signals: {e}")

