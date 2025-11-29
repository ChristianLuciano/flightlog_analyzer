"""
Event panel component.

Displays and filters event log entries.
"""

from dash import html, dcc, dash_table
from typing import List, Dict, Any
import pandas as pd


class EventPanel:
    """
    Event log panel component.

    Displays events with filtering and navigation.
    """

    def __init__(self, events: pd.DataFrame = None):
        self._events = events

    def set_events(self, events: pd.DataFrame) -> None:
        """Set event data."""
        self._events = events

    def render(self) -> html.Div:
        """Render event panel."""
        return html.Div([
            html.H4("Events"),

            # Filters
            html.Div([
                dcc.Dropdown(
                    id='event-type-filter',
                    multi=True,
                    placeholder='Filter by type...',
                    className='event-filter'
                ),
                dcc.Dropdown(
                    id='event-severity-filter',
                    options=[
                        {'label': 'Critical', 'value': 'critical'},
                        {'label': 'Warning', 'value': 'warning'},
                        {'label': 'Info', 'value': 'info'},
                        {'label': 'Debug', 'value': 'debug'},
                    ],
                    multi=True,
                    placeholder='Filter by severity...',
                    value=['critical', 'warning'],
                    className='event-filter'
                ),
                dcc.Input(
                    id='event-search',
                    type='text',
                    placeholder='Search events...',
                    className='event-search'
                ),
            ], className='event-filters'),

            # Event table
            dash_table.DataTable(
                id='event-table',
                columns=[
                    {'name': 'Time', 'id': 'timestamp'},
                    {'name': 'Type', 'id': 'event_type'},
                    {'name': 'Severity', 'id': 'severity'},
                    {'name': 'Description', 'id': 'description'},
                ],
                data=[],
                row_selectable='single',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{severity} = critical'},
                     'backgroundColor': '#ffebee', 'color': '#c62828'},
                    {'if': {'filter_query': '{severity} = warning'},
                     'backgroundColor': '#fff3e0', 'color': '#e65100'},
                ],
            ),

            # Navigation buttons
            html.Div([
                html.Button("⬆ Previous", id='btn-prev-event', className='btn btn-sm'),
                html.Button("⬇ Next", id='btn-next-event', className='btn btn-sm'),
                html.Button("Jump to Selected", id='btn-jump-event', className='btn btn-sm'),
            ], className='event-nav'),

        ], className='event-panel')

