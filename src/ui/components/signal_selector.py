"""
Signal selector component.

Provides multi-select dropdown for choosing signals to display.
"""

from dash import html, dcc
from typing import List, Dict, Any


class SignalSelector:
    """
    Signal selection component.

    Provides searchable multi-select for signals.
    """

    def __init__(self, signals: List[str] = None):
        self._signals = signals or []

    def set_signals(self, signals: List[str]) -> None:
        """Set available signals."""
        self._signals = signals

    def render(self) -> html.Div:
        """Render the signal selector."""
        return html.Div([
            html.Label("Select Signals", className='selector-label'),
            dcc.Dropdown(
                id='signal-selector',
                options=[{'label': s.split('.')[-1], 'value': s} for s in self._signals],
                multi=True,
                placeholder='Search and select signals...',
                searchable=True,
                className='signal-dropdown'
            ),
            html.Div([
                html.Button("Clear All", id='btn-clear-signals', className='btn btn-sm'),
                html.Button("Select All", id='btn-select-all', className='btn btn-sm'),
            ], className='selector-actions'),
        ], className='signal-selector')

    @staticmethod
    def group_by_category(signals: List[str]) -> Dict[str, List[str]]:
        """Group signals by top-level category."""
        groups = {}
        for signal in signals:
            parts = signal.split('.')
            category = parts[0] if parts else 'Other'
            if category not in groups:
                groups[category] = []
            groups[category].append(signal)
        return groups

