"""
Tests for UI layout components.

Covers src/ui/layouts/* - 100% coverage target.
"""

import pytest

from src.ui.layouts.grid_layout import GridCell, GridLayout
from src.ui.layouts.tab_manager import Tab, TabManager


class TestGridCell:
    """Test GridCell dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        cell = GridCell(row=0, col=0)
        assert cell.row == 0
        assert cell.col == 0
        assert cell.row_span == 1
        assert cell.col_span == 1
        assert cell.plot_id is None
    
    def test_custom_values(self):
        """Test custom values."""
        cell = GridCell(row=1, col=2, row_span=2, col_span=3, plot_id='plot1')
        assert cell.row == 1
        assert cell.col == 2
        assert cell.row_span == 2
        assert cell.col_span == 3
        assert cell.plot_id == 'plot1'


class TestGridLayout:
    """Test GridLayout dataclass."""
    
    def test_default_values(self):
        """Test default grid values."""
        grid = GridLayout()
        assert grid.rows == 2
        assert grid.cols == 2
        assert grid.gap == 10
        assert grid.cells == {}
    
    def test_custom_values(self):
        """Test custom grid values."""
        grid = GridLayout(rows=3, cols=4, gap=20)
        assert grid.rows == 3
        assert grid.cols == 4
        assert grid.gap == 20
    
    def test_add_plot_success(self):
        """Test adding plot successfully."""
        grid = GridLayout()
        result = grid.add_plot('plot1', row=0, col=0)
        assert result is True
        assert 'plot1' in grid.cells
    
    def test_add_plot_out_of_bounds(self):
        """Test adding plot out of bounds."""
        grid = GridLayout(rows=2, cols=2)
        result = grid.add_plot('plot1', row=2, col=0)
        assert result is False
    
    def test_add_plot_with_span_out_of_bounds(self):
        """Test adding plot with span going out of bounds."""
        grid = GridLayout(rows=2, cols=2)
        result = grid.add_plot('plot1', row=0, col=0, row_span=3)
        assert result is False
    
    def test_add_plot_overlap(self):
        """Test adding overlapping plots."""
        grid = GridLayout()
        grid.add_plot('plot1', row=0, col=0)
        result = grid.add_plot('plot2', row=0, col=0)
        assert result is False
    
    def test_add_plot_adjacent(self):
        """Test adding adjacent plots."""
        grid = GridLayout()
        grid.add_plot('plot1', row=0, col=0)
        result = grid.add_plot('plot2', row=0, col=1)
        assert result is True
    
    def test_add_plot_with_spanning(self):
        """Test adding plot with row and column spanning."""
        grid = GridLayout(rows=4, cols=4)
        result = grid.add_plot('plot1', row=0, col=0, row_span=2, col_span=2)
        assert result is True
        assert grid.cells['plot1'].row_span == 2
        assert grid.cells['plot1'].col_span == 2
    
    def test_remove_plot_exists(self):
        """Test removing existing plot."""
        grid = GridLayout()
        grid.add_plot('plot1', row=0, col=0)
        result = grid.remove_plot('plot1')
        assert result is True
        assert 'plot1' not in grid.cells
    
    def test_remove_plot_not_exists(self):
        """Test removing non-existent plot."""
        grid = GridLayout()
        result = grid.remove_plot('nonexistent')
        assert result is False
    
    def test_move_plot_success(self):
        """Test moving plot successfully."""
        grid = GridLayout()
        grid.add_plot('plot1', row=0, col=0)
        result = grid.move_plot('plot1', new_row=1, new_col=1)
        assert result is True
        assert grid.cells['plot1'].row == 1
        assert grid.cells['plot1'].col == 1
    
    def test_move_plot_not_exists(self):
        """Test moving non-existent plot."""
        grid = GridLayout()
        result = grid.move_plot('nonexistent', new_row=0, new_col=0)
        assert result is False
    
    def test_move_plot_out_of_bounds(self):
        """Test moving plot out of bounds."""
        grid = GridLayout(rows=2, cols=2)
        grid.add_plot('plot1', row=0, col=0)
        result = grid.move_plot('plot1', new_row=2, new_col=0)
        assert result is False
    
    def test_move_plot_overlap(self):
        """Test moving plot to occupied position."""
        grid = GridLayout()
        grid.add_plot('plot1', row=0, col=0)
        grid.add_plot('plot2', row=1, col=1)
        result = grid.move_plot('plot1', new_row=1, new_col=1)
        assert result is False
    
    def test_resize_grid_success(self):
        """Test resizing grid successfully."""
        grid = GridLayout(rows=2, cols=2)
        result = grid.resize_grid(rows=4, cols=4)
        assert result is True
        assert grid.rows == 4
        assert grid.cols == 4
    
    def test_resize_grid_too_small(self):
        """Test resizing grid smaller than plots."""
        grid = GridLayout(rows=4, cols=4)
        grid.add_plot('plot1', row=2, col=2)
        result = grid.resize_grid(rows=2, cols=2)
        assert result is False
        assert grid.rows == 4  # Unchanged
    
    def test_cells_overlap_no_overlap(self):
        """Test _cells_overlap with non-overlapping cells."""
        grid = GridLayout()
        cell = GridCell(row=0, col=0)
        result = grid._cells_overlap(cell, 1, 1, 1, 1)
        assert result is False
    
    def test_cells_overlap_partial_overlap(self):
        """Test _cells_overlap with partial overlap."""
        grid = GridLayout()
        cell = GridCell(row=0, col=0, row_span=2, col_span=2)
        result = grid._cells_overlap(cell, 1, 1, 1, 1)
        assert result is True
    
    def test_get_css_grid_template(self):
        """Test getting CSS grid template."""
        grid = GridLayout(rows=3, cols=4, gap=15)
        styles = grid.get_css_grid_template()
        assert styles['display'] == 'grid'
        assert 'repeat(3, 1fr)' in styles['gridTemplateRows']
        assert 'repeat(4, 1fr)' in styles['gridTemplateColumns']
        assert styles['gap'] == '15px'
    
    def test_get_cell_style_exists(self):
        """Test getting cell style for existing plot."""
        grid = GridLayout(rows=5, cols=6)  # Make grid large enough
        result = grid.add_plot('plot1', row=1, col=2, row_span=2, col_span=3)
        assert result is True
        styles = grid.get_cell_style('plot1')
        assert styles['gridRow'] == '2 / span 2'  # 1-based
        assert styles['gridColumn'] == '3 / span 3'
    
    def test_get_cell_style_not_exists(self):
        """Test getting cell style for non-existent plot."""
        grid = GridLayout()
        styles = grid.get_cell_style('nonexistent')
        assert styles == {}
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        grid = GridLayout(rows=3, cols=3, gap=5)
        grid.add_plot('plot1', row=0, col=0)
        result = grid.to_dict()
        assert result['rows'] == 3
        assert result['cols'] == 3
        assert result['gap'] == 5
        assert 'plot1' in result['cells']
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'rows': 4,
            'cols': 5,
            'gap': 8,
            'cells': {
                'plot1': {'row': 0, 'col': 0, 'row_span': 1, 'col_span': 1, 'plot_id': 'plot1'}
            }
        }
        grid = GridLayout.from_dict(data)
        assert grid.rows == 4
        assert grid.cols == 5
        assert grid.gap == 8
        assert 'plot1' in grid.cells
    
    def test_from_dict_empty(self):
        """Test creating from empty dictionary."""
        grid = GridLayout.from_dict({})
        assert grid.rows == 2
        assert grid.cols == 2
        assert grid.gap == 10


class TestTab:
    """Test Tab dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        tab = Tab(id='tab1', name='Tab 1')
        assert tab.id == 'tab1'
        assert tab.name == 'Tab 1'
        assert isinstance(tab.grid, GridLayout)
        assert tab.is_active is False
        assert tab.is_loaded is False


class TestTabManager:
    """Test TabManager class."""
    
    def test_init(self):
        """Test initialization."""
        manager = TabManager()
        assert manager._tabs == {}
        assert manager._order == []
        assert manager._active_tab is None
    
    def test_add_tab(self):
        """Test adding tab."""
        manager = TabManager()
        tab = manager.add_tab('tab1', 'Tab 1')
        assert tab.id == 'tab1'
        assert tab.name == 'Tab 1'
        assert 'tab1' in manager._tabs
    
    def test_add_first_tab_becomes_active(self):
        """Test first tab becomes active."""
        manager = TabManager()
        tab = manager.add_tab('tab1', 'Tab 1')
        assert tab.is_active is True
        assert manager._active_tab == 'tab1'
    
    def test_add_second_tab_not_active(self):
        """Test second tab is not active."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        tab2 = manager.add_tab('tab2', 'Tab 2')
        assert tab2.is_active is False
    
    def test_remove_tab_exists(self):
        """Test removing existing tab."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        result = manager.remove_tab('tab1')
        assert result is True
        assert 'tab1' not in manager._tabs
    
    def test_remove_tab_not_exists(self):
        """Test removing non-existent tab."""
        manager = TabManager()
        result = manager.remove_tab('nonexistent')
        assert result is False
    
    def test_remove_active_tab(self):
        """Test removing active tab switches to next."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        manager.remove_tab('tab1')
        assert manager._active_tab == 'tab2'
    
    def test_remove_last_tab(self):
        """Test removing last tab clears active."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.remove_tab('tab1')
        assert manager._active_tab is None
    
    def test_get_tab_exists(self):
        """Test getting existing tab."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        tab = manager.get_tab('tab1')
        assert tab is not None
        assert tab.id == 'tab1'
    
    def test_get_tab_not_exists(self):
        """Test getting non-existent tab."""
        manager = TabManager()
        tab = manager.get_tab('nonexistent')
        assert tab is None
    
    def test_set_active_success(self):
        """Test setting active tab."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        result = manager.set_active('tab2')
        assert result is True
        assert manager._active_tab == 'tab2'
        assert manager._tabs['tab2'].is_active is True
        assert manager._tabs['tab1'].is_active is False
    
    def test_set_active_not_exists(self):
        """Test setting non-existent tab as active."""
        manager = TabManager()
        result = manager.set_active('nonexistent')
        assert result is False
    
    def test_set_active_marks_loaded(self):
        """Test setting active marks tab as loaded."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        tab2 = manager.add_tab('tab2', 'Tab 2')
        assert tab2.is_loaded is False
        manager.set_active('tab2')
        assert tab2.is_loaded is True
    
    def test_get_active_exists(self):
        """Test getting active tab."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        active = manager.get_active()
        assert active is not None
        assert active.id == 'tab1'
    
    def test_get_active_none(self):
        """Test getting active when none."""
        manager = TabManager()
        active = manager.get_active()
        assert active is None
    
    def test_reorder_success(self):
        """Test reordering tabs."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        manager.add_tab('tab3', 'Tab 3')
        result = manager.reorder(['tab3', 'tab1', 'tab2'])
        assert result is True
        assert manager._order == ['tab3', 'tab1', 'tab2']
    
    def test_reorder_invalid(self):
        """Test reordering with invalid list."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        result = manager.reorder(['tab1', 'tab3'])  # tab3 doesn't exist
        assert result is False
    
    def test_duplicate_tab_success(self):
        """Test duplicating tab."""
        manager = TabManager()
        tab1 = manager.add_tab('tab1', 'Tab 1')
        tab1.grid.add_plot('plot1', row=0, col=0)
        
        dup = manager.duplicate_tab('tab1', 'Tab 1 Copy')
        assert dup is not None
        assert dup.name == 'Tab 1 Copy'
        assert 'tab1_copy' in manager._tabs
    
    def test_duplicate_tab_not_exists(self):
        """Test duplicating non-existent tab."""
        manager = TabManager()
        dup = manager.duplicate_tab('nonexistent', 'Copy')
        assert dup is None
    
    def test_duplicate_tab_unique_id(self):
        """Test duplicate generates unique IDs."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.duplicate_tab('tab1', 'Copy 1')
        dup2 = manager.duplicate_tab('tab1', 'Copy 2')
        assert dup2.id == 'tab1_copy_1'
    
    def test_get_tab_order(self):
        """Test getting tab order."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        order = manager.get_tab_order()
        assert order == ['tab1', 'tab2']
        # Should be a copy
        order.append('tab3')
        assert manager._order == ['tab1', 'tab2']
    
    def test_list_tabs(self):
        """Test listing all tabs."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        manager.add_tab('tab2', 'Tab 2')
        tabs = manager.list_tabs()
        assert len(tabs) == 2
        assert tabs[0].id == 'tab1'
        assert tabs[1].id == 'tab2'
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        manager = TabManager()
        manager.add_tab('tab1', 'Tab 1')
        result = manager.to_dict()
        assert 'tabs' in result
        assert 'order' in result
        assert 'active' in result
        assert 'tab1' in result['tabs']
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'tabs': {
                'tab1': {'id': 'tab1', 'name': 'Tab 1', 'grid': {}},
                'tab2': {'id': 'tab2', 'name': 'Tab 2', 'grid': {}}
            },
            'order': ['tab1', 'tab2'],
            'active': 'tab2'
        }
        manager = TabManager.from_dict(data)
        assert len(manager._tabs) == 2
        assert manager._order == ['tab1', 'tab2']
        assert manager._active_tab == 'tab2'
        assert manager._tabs['tab2'].is_active is True
    
    def test_from_dict_empty(self):
        """Test creating from empty dictionary."""
        manager = TabManager.from_dict({})
        assert manager._tabs == {}
        assert manager._order == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

