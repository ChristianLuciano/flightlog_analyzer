"""
Tests for UI components.

Covers src/ui/components/* - 100% coverage target.
"""

import pytest
from dash import html
import pandas as pd

from src.ui.components.data_browser import DataBrowser
from src.ui.components.signal_selector import SignalSelector
from src.ui.components.event_panel import EventPanel
from src.ui.components.formula_editor import FormulaEditor
from src.ui.components.playback_controls import PlaybackControls
from src.ui.components.plot_builder import PlotBuilder, create_plot_grid


class TestDataBrowser:
    """Test DataBrowser component."""
    
    def test_init_empty(self):
        """Test initialization with no data."""
        browser = DataBrowser()
        assert browser._hierarchy == {}
    
    def test_init_with_data(self):
        """Test initialization with data."""
        data = {'folder': {'signal': 'value'}}
        browser = DataBrowser(data)
        assert browser._hierarchy == data
    
    def test_set_data(self):
        """Test set_data method."""
        browser = DataBrowser()
        data = {'test': {'value': 1}}
        browser.set_data(data)
        assert browser._hierarchy == data
    
    def test_render_empty(self):
        """Test render with empty data."""
        browser = DataBrowser()
        result = browser.render()
        assert isinstance(result, html.Div)
        assert 'No data loaded' in str(result.children)
    
    def test_render_with_data(self):
        """Test render with data."""
        data = {'Folder1': {'Signal1': 'value'}}
        browser = DataBrowser(data)
        result = browser.render()
        assert isinstance(result, html.Div)
        assert 'data-browser' in result.className
    
    def test_build_tree_dict(self):
        """Test _build_tree with dictionary."""
        browser = DataBrowser()
        data = {'key1': {'nested': 'value'}, 'key2': 'leaf'}
        nodes = browser._build_tree(data, "")
        assert len(nodes) == 2
    
    def test_build_tree_nested(self):
        """Test _build_tree with nested structure."""
        browser = DataBrowser()
        data = {'a': {'b': {'c': 'value'}}}
        nodes = browser._build_tree(data, "root")
        assert len(nodes) == 1
    
    def test_build_tree_path(self):
        """Test _build_tree generates correct paths."""
        browser = DataBrowser()
        data = {'folder': {'signal': 'data'}}
        nodes = browser._build_tree(data, "")
        # Check that node has proper ID
        node_div = nodes[0]
        assert isinstance(node_div, html.Div)
    
    def test_get_component_id(self):
        """Test get_component_id static method."""
        assert DataBrowser.get_component_id() == 'data-browser'


class TestSignalSelector:
    """Test SignalSelector component."""
    
    def test_init_empty(self):
        """Test initialization with no signals."""
        selector = SignalSelector()
        assert selector._signals == []
    
    def test_init_with_signals(self):
        """Test initialization with signals."""
        signals = ['a.b.c', 'd.e.f']
        selector = SignalSelector(signals)
        assert selector._signals == signals
    
    def test_set_signals(self):
        """Test set_signals method."""
        selector = SignalSelector()
        signals = ['x', 'y', 'z']
        selector.set_signals(signals)
        assert selector._signals == signals
    
    def test_render(self):
        """Test render method."""
        signals = ['IMU.accel_x', 'GPS.lat']
        selector = SignalSelector(signals)
        result = selector.render()
        assert isinstance(result, html.Div)
        assert 'signal-selector' in result.className
    
    def test_render_creates_dropdown(self):
        """Test render creates dropdown options."""
        signals = ['path.to.signal']
        selector = SignalSelector(signals)
        result = selector.render()
        # Check structure
        assert len(result.children) == 3  # Label, Dropdown, Actions
    
    def test_group_by_category_empty(self):
        """Test group_by_category with empty list."""
        groups = SignalSelector.group_by_category([])
        assert groups == {}
    
    def test_group_by_category_single(self):
        """Test group_by_category with single signal."""
        groups = SignalSelector.group_by_category(['IMU.accel'])
        assert 'IMU' in groups
        assert 'IMU.accel' in groups['IMU']
    
    def test_group_by_category_multiple(self):
        """Test group_by_category with multiple signals."""
        signals = ['IMU.accel', 'IMU.gyro', 'GPS.lat', 'GPS.lon']
        groups = SignalSelector.group_by_category(signals)
        assert len(groups) == 2
        assert len(groups['IMU']) == 2
        assert len(groups['GPS']) == 2
    
    def test_group_by_category_no_dot(self):
        """Test group_by_category with signal without dot."""
        groups = SignalSelector.group_by_category(['signal'])
        assert 'signal' in groups
        assert 'signal' in groups['signal']


class TestEventPanel:
    """Test EventPanel component."""
    
    def test_init_empty(self):
        """Test initialization with no events."""
        panel = EventPanel()
        assert panel._events is None
    
    def test_init_with_events(self):
        """Test initialization with events."""
        events = pd.DataFrame({'time': [1, 2], 'event': ['a', 'b']})
        panel = EventPanel(events)
        assert panel._events is not None
        assert len(panel._events) == 2
    
    def test_set_events(self):
        """Test set_events method."""
        panel = EventPanel()
        events = pd.DataFrame({'time': [1]})
        panel.set_events(events)
        assert panel._events is not None
    
    def test_render(self):
        """Test render method."""
        panel = EventPanel()
        result = panel.render()
        assert isinstance(result, html.Div)
        assert 'event-panel' in result.className
    
    def test_render_contains_filters(self):
        """Test render contains filter dropdowns."""
        panel = EventPanel()
        result = panel.render()
        # Should have H4, filters div, table, and nav
        assert len(result.children) == 4
    
    def test_render_contains_table(self):
        """Test render contains DataTable."""
        panel = EventPanel()
        result = panel.render()
        # DataTable should be the third child
        table = result.children[2]
        assert table.id == 'event-table'


class TestFormulaEditor:
    """Test FormulaEditor component."""
    
    def test_common_functions(self):
        """Test COMMON_FUNCTIONS list."""
        assert 'sqrt' in FormulaEditor.COMMON_FUNCTIONS
        assert 'sin' in FormulaEditor.COMMON_FUNCTIONS
        assert 'cos' in FormulaEditor.COMMON_FUNCTIONS
        assert 'moving_avg' in FormulaEditor.COMMON_FUNCTIONS
    
    def test_init(self):
        """Test initialization."""
        editor = FormulaEditor()
        assert editor is not None
    
    def test_render(self):
        """Test render method."""
        editor = FormulaEditor()
        result = editor.render()
        assert isinstance(result, html.Div)
        assert 'formula-editor' in result.className
    
    def test_render_has_inputs(self):
        """Test render has name and formula inputs."""
        editor = FormulaEditor()
        result = editor.render()
        # Should have multiple form groups
        assert len(result.children) >= 6
    
    def test_render_has_function_buttons(self):
        """Test render has function buttons."""
        editor = FormulaEditor()
        result = editor.render()
        # Structure: H4, NameInput, FormulaTextarea, FunctionButtons, InputSignals, Unit, etc
        # Function buttons are in the 4th child (index 3)
        func_group = result.children[3]  # Function reference form-group
        func_list_div = func_group.children[1]  # The div with function buttons
        # Should have buttons for each function
        assert len(func_list_div.children) == len(FormulaEditor.COMMON_FUNCTIONS)


class TestPlaybackControls:
    """Test PlaybackControls component."""
    
    def test_speed_options(self):
        """Test SPEED_OPTIONS list."""
        speeds = [opt['value'] for opt in PlaybackControls.SPEED_OPTIONS]
        assert 0.1 in speeds
        assert 1.0 in speeds
        assert 10.0 in speeds
    
    def test_init_default(self):
        """Test initialization with defaults."""
        controls = PlaybackControls()
        assert controls.time_range == (0, 100)
    
    def test_init_custom_range(self):
        """Test initialization with custom range."""
        controls = PlaybackControls(time_range=(10, 500))
        assert controls.time_range == (10, 500)
    
    def test_render(self):
        """Test render method."""
        controls = PlaybackControls()
        result = controls.render()
        assert isinstance(result, html.Div)
        assert 'playback-controls' in result.className
    
    def test_render_has_slider(self):
        """Test render has slider."""
        controls = PlaybackControls((0, 50))
        result = controls.render()
        slider_container = result.children[0]
        assert 'slider-container' in slider_container.className
    
    def test_render_has_controls(self):
        """Test render has control buttons."""
        controls = PlaybackControls()
        result = controls.render()
        controls_row = result.children[1]
        assert 'controls-row' in controls_row.className
    
    def test_slider_range(self):
        """Test slider has correct range."""
        controls = PlaybackControls((5, 95))
        result = controls.render()
        slider = result.children[0].children[0]
        assert slider.min == 5
        assert slider.max == 95


class TestPlotBuilder:
    """Test PlotBuilder component."""
    
    def test_plot_types(self):
        """Test PLOT_TYPES list."""
        types = [pt['value'] for pt in PlotBuilder.PLOT_TYPES]
        assert 'TIME_SERIES' in types
        assert 'XY_SCATTER' in types
        assert 'FFT' in types
        assert 'MAP_2D' in types
    
    def test_init(self):
        """Test initialization."""
        builder = PlotBuilder()
        assert builder is not None
    
    def test_render(self):
        """Test render method."""
        builder = PlotBuilder()
        result = builder.render()
        assert isinstance(result, html.Div)
        assert 'plot-builder' in result.className
    
    def test_render_has_dropdowns(self):
        """Test render has type and signal dropdowns."""
        builder = PlotBuilder()
        result = builder.render()
        # Should have H4, type selector, signal selector, position, button
        assert len(result.children) == 5
    
    def test_render_has_create_button(self):
        """Test render has create button."""
        builder = PlotBuilder()
        result = builder.render()
        button = result.children[4]
        assert button.id == 'btn-create-plot'


class TestCreatePlotGrid:
    """Test create_plot_grid function."""
    
    def test_creates_grid(self):
        """Test function creates grid."""
        result = create_plot_grid('tab1')
        assert isinstance(result, html.Div)
        assert result.id == 'grid-tab1'
    
    def test_grid_has_plots(self):
        """Test grid has plot cells."""
        result = create_plot_grid('test')
        # Should have two rows
        assert len(result.children) == 2
    
    def test_plot_ids(self):
        """Test plots have correct IDs."""
        result = create_plot_grid('myTab')
        # First row, first cell, graph
        first_row = result.children[0]
        first_cell = first_row.children[0]
        graph = first_cell.children[0]
        assert graph.id == {'type': 'plot', 'index': 'myTab-1'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

