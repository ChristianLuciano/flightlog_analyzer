"""
Data browser component.

Provides hierarchical tree view for navigating data structure.
"""

from dash import html, dcc
from typing import Dict, Any, List, Optional


class DataBrowser:
    """
    Hierarchical data browser component.

    Displays nested data structure as expandable tree.
    """

    def __init__(self, data_hierarchy: Optional[Dict] = None):
        self._hierarchy = data_hierarchy or {}

    def set_data(self, hierarchy: Dict) -> None:
        """Set data hierarchy."""
        self._hierarchy = hierarchy

    def render(self) -> html.Div:
        """Render the data browser component."""
        if not self._hierarchy:
            return html.Div("No data loaded", className='no-data-message')

        return html.Div([
            html.Div([
                dcc.Input(
                    id='tree-search',
                    type='text',
                    placeholder='Search signals...',
                    className='tree-search'
                ),
            ], className='tree-header'),
            html.Div(
                self._build_tree(self._hierarchy, ""),
                id='tree-container',
                className='tree-container'
            ),
        ], className='data-browser')

    def _build_tree(self, data: Any, path: str, level: int = 0) -> List:
        """Recursively build tree nodes."""
        nodes = []

        if isinstance(data, dict):
            for key, value in data.items():
                node_path = f"{path}.{key}" if path else key
                is_leaf = not isinstance(value, dict)

                node = html.Div([
                    html.Div([
                        html.Span(
                            "📁 " if not is_leaf else "📊 ",
                            className='node-icon'
                        ),
                        html.Span(key, className='node-label'),
                    ], className='node-header', id={'type': 'tree-node', 'path': node_path}),
                    html.Div(
                        self._build_tree(value, node_path, level + 1),
                        className='node-children'
                    ) if not is_leaf else None,
                ], className=f'tree-node level-{level}', style={'marginLeft': f'{level * 16}px'})

                nodes.append(node)

        return nodes

    @staticmethod
    def get_component_id() -> str:
        """Get component ID for callbacks."""
        return 'data-browser'

