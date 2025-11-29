"""
Formula editor component.

Provides interface for creating and editing computed signal formulas.
"""

from dash import html, dcc
from typing import List


class FormulaEditor:
    """
    Formula editor component.

    Provides syntax-aware editing for computed signal formulas.
    """

    COMMON_FUNCTIONS = [
        'sqrt', 'abs', 'sin', 'cos', 'tan', 'exp', 'log', 'log10',
        'diff', 'cumsum', 'moving_avg', 'lowpass', 'highpass',
        'mean', 'std', 'min', 'max', 'haversine', 'where'
    ]

    def __init__(self):
        pass

    def render(self) -> html.Div:
        """Render formula editor."""
        return html.Div([
            html.H4("Computed Signal Editor"),

            # Name input
            html.Div([
                html.Label("Signal Name"),
                dcc.Input(
                    id='formula-name',
                    type='text',
                    placeholder='my_signal',
                    className='formula-input'
                ),
            ], className='form-group'),

            # Formula input
            html.Div([
                html.Label("Formula"),
                dcc.Textarea(
                    id='formula-input',
                    placeholder='sqrt(x**2 + y**2)',
                    className='formula-textarea'
                ),
            ], className='form-group'),

            # Function reference
            html.Div([
                html.Label("Available Functions"),
                html.Div([
                    html.Button(
                        f,
                        id={'type': 'func-btn', 'func': f},
                        className='func-btn'
                    )
                    for f in self.COMMON_FUNCTIONS
                ], className='func-list'),
            ], className='form-group'),

            # Input signals
            html.Div([
                html.Label("Input Signals"),
                dcc.Dropdown(
                    id='formula-inputs',
                    multi=True,
                    placeholder='Select input signals...',
                ),
            ], className='form-group'),

            # Unit
            html.Div([
                html.Label("Unit"),
                dcc.Input(
                    id='formula-unit',
                    type='text',
                    placeholder='m/s',
                    className='formula-input'
                ),
            ], className='form-group'),

            # Validation status
            html.Div(id='formula-validation', className='validation-status'),

            # Preview
            html.Div([
                html.Label("Preview (first 5 values)"),
                html.Div(id='formula-preview', className='formula-preview'),
            ], className='form-group'),

            # Actions
            html.Div([
                html.Button("Validate", id='btn-validate-formula', className='btn btn-secondary'),
                html.Button("Create Signal", id='btn-create-signal', className='btn btn-primary'),
            ], className='formula-actions'),

        ], className='formula-editor')

