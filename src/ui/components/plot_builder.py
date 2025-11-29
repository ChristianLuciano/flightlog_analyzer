"""
Plot builder component.

Provides UI for creating and configuring new plots.
"""

from dash import html, dcc
from typing import List, Dict, Any, Optional


class PlotBuilder:
    """
    Plot builder component.

    Provides interface for creating new plots with configuration.
    """

    PLOT_TYPES = [
        {'label': 'Time Series', 'value': 'TIME_SERIES'},
        {'label': 'X-Y Scatter', 'value': 'XY_SCATTER'},
        {'label': 'X-Y Line', 'value': 'XY_LINE'},
        {'label': 'FFT', 'value': 'FFT'},
        {'label': 'Spectrogram', 'value': 'SPECTROGRAM'},
        {'label': 'Histogram', 'value': 'HISTOGRAM'},
        {'label': 'Box Plot', 'value': 'BOXPLOT'},
        {'label': '2D Map', 'value': 'MAP_2D'},
        {'label': '3D Map', 'value': 'MAP_3D'},
    ]

    def __init__(self):
        pass

    def render(self) -> html.Div:
        """Render the plot builder component."""
        return html.Div([
            html.H4("Add New Plot"),

            # Plot type selection
            html.Div([
                html.Label("Plot Type"),
                dcc.Dropdown(
                    id='new-plot-type',
                    options=self.PLOT_TYPES,
                    placeholder='Select plot type...',
                ),
            ], className='form-group'),

            # Signal selection
            html.Div([
                html.Label("Signals"),
                dcc.Dropdown(
                    id='new-plot-signals',
                    multi=True,
                    placeholder='Select signals...',
                ),
            ], className='form-group'),

            # Grid position
            html.Div([
                html.Label("Position"),
                html.Div([
                    dcc.Input(id='new-plot-row', type='number', placeholder='Row', min=0, max=10),
                    dcc.Input(id='new-plot-col', type='number', placeholder='Col', min=0, max=10),
                ], className='position-inputs'),
            ], className='form-group'),

            # Create button
            html.Button("Create Plot", id='btn-create-plot', className='btn btn-primary'),

        ], className='plot-builder')


def create_plot_grid(tab_id: str) -> html.Div:
    """
    Create plot grid for a tab.

    Args:
        tab_id: Tab identifier.

    Returns:
        Grid layout component.
    """
    return html.Div([
        html.Div([
            # Placeholder for plots
            html.Div([
                dcc.Graph(
                    id={'type': 'plot', 'index': f'{tab_id}-1'},
                    className='plot-item',
                    config={'displayModeBar': True, 'scrollZoom': True}
                ),
            ], className='plot-cell'),
            html.Div([
                dcc.Graph(
                    id={'type': 'plot', 'index': f'{tab_id}-2'},
                    className='plot-item',
                    config={'displayModeBar': True, 'scrollZoom': True}
                ),
            ], className='plot-cell'),
        ], className='plot-row'),
        html.Div([
            html.Div([
                dcc.Graph(
                    id={'type': 'plot', 'index': f'{tab_id}-3'},
                    className='plot-item',
                    config={'displayModeBar': True, 'scrollZoom': True}
                ),
            ], className='plot-cell'),
            html.Div([
                dcc.Graph(
                    id={'type': 'plot', 'index': f'{tab_id}-4'},
                    className='plot-item',
                    config={'displayModeBar': True, 'scrollZoom': True}
                ),
            ], className='plot-cell'),
        ], className='plot-row'),
    ], className='plot-grid', id=f'grid-{tab_id}')

