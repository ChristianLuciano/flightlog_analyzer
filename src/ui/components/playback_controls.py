"""
Playback controls component.

Provides play/pause, speed control, and time navigation.
"""

from dash import html, dcc


class PlaybackControls:
    """
    Playback control component.

    Provides controls for time-based playback and navigation.
    """

    SPEED_OPTIONS = [
        {'label': '0.1x', 'value': 0.1},
        {'label': '0.25x', 'value': 0.25},
        {'label': '0.5x', 'value': 0.5},
        {'label': '1x', 'value': 1.0},
        {'label': '2x', 'value': 2.0},
        {'label': '5x', 'value': 5.0},
        {'label': '10x', 'value': 10.0},
    ]

    def __init__(self, time_range: tuple = (0, 100)):
        self.time_range = time_range

    def render(self) -> html.Div:
        """Render playback controls."""
        return html.Div([
            # Time slider
            html.Div([
                dcc.Slider(
                    id='playback-slider',
                    min=self.time_range[0],
                    max=self.time_range[1],
                    value=self.time_range[0],
                    tooltip={'placement': 'top', 'always_visible': True},
                    className='playback-slider'
                ),
            ], className='slider-container'),

            # Control buttons
            html.Div([
                html.Button("⏮", id='btn-to-start', className='btn-control', title='Go to start'),
                html.Button("⏪", id='btn-step-back', className='btn-control', title='Step back'),
                html.Button("▶", id='btn-play-pause', className='btn-control btn-play', title='Play/Pause'),
                html.Button("⏩", id='btn-step-forward', className='btn-control', title='Step forward'),
                html.Button("⏭", id='btn-to-end', className='btn-control', title='Go to end'),

                # Speed selector
                html.Div([
                    html.Label("Speed:", className='speed-label'),
                    dcc.Dropdown(
                        id='playback-speed',
                        options=self.SPEED_OPTIONS,
                        value=1.0,
                        clearable=False,
                        className='speed-dropdown'
                    ),
                ], className='speed-container'),

                # Time display
                html.Div([
                    html.Span(id='time-current', children='00:00.00'),
                    html.Span(' / '),
                    html.Span(id='time-total', children='00:00.00'),
                ], className='time-display'),

                # Loop toggle
                dcc.Checklist(
                    id='loop-toggle',
                    options=[{'label': 'Loop', 'value': 'loop'}],
                    value=[],
                    className='loop-toggle'
                ),
            ], className='controls-row'),

        ], className='playback-controls')

