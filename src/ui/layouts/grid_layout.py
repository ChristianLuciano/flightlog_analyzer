"""
Grid layout management.

Provides flexible grid system for arranging plots.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class GridCell:
    """Represents a cell in the grid."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    plot_id: Optional[str] = None


@dataclass
class GridLayout:
    """
    Flexible grid layout for plots.

    Supports arbitrary dimensions and spanning.
    """

    rows: int = 2
    cols: int = 2
    gap: int = 10
    cells: Dict[str, GridCell] = field(default_factory=dict)

    def add_plot(
        self,
        plot_id: str,
        row: int,
        col: int,
        row_span: int = 1,
        col_span: int = 1
    ) -> bool:
        """
        Add a plot to the grid.

        Returns True if successful.
        """
        # Check bounds
        if row + row_span > self.rows or col + col_span > self.cols:
            return False

        # Check for overlaps
        for cell in self.cells.values():
            if self._cells_overlap(cell, row, col, row_span, col_span):
                return False

        self.cells[plot_id] = GridCell(row, col, row_span, col_span, plot_id)
        return True

    def remove_plot(self, plot_id: str) -> bool:
        """Remove a plot from the grid."""
        if plot_id in self.cells:
            del self.cells[plot_id]
            return True
        return False

    def move_plot(self, plot_id: str, new_row: int, new_col: int) -> bool:
        """Move a plot to new position."""
        if plot_id not in self.cells:
            return False

        cell = self.cells[plot_id]
        row_span, col_span = cell.row_span, cell.col_span

        # Check bounds
        if new_row + row_span > self.rows or new_col + col_span > self.cols:
            return False

        # Check for overlaps (excluding self)
        for pid, c in self.cells.items():
            if pid != plot_id and self._cells_overlap(c, new_row, new_col, row_span, col_span):
                return False

        cell.row = new_row
        cell.col = new_col
        return True

    def resize_grid(self, rows: int, cols: int) -> bool:
        """Resize the grid."""
        # Check if all plots still fit
        for cell in self.cells.values():
            if cell.row + cell.row_span > rows or cell.col + cell.col_span > cols:
                return False

        self.rows = rows
        self.cols = cols
        return True

    def _cells_overlap(
        self,
        cell: GridCell,
        row: int,
        col: int,
        row_span: int,
        col_span: int
    ) -> bool:
        """Check if cell overlaps with given position."""
        return not (
            cell.row + cell.row_span <= row or
            row + row_span <= cell.row or
            cell.col + cell.col_span <= col or
            col + col_span <= cell.col
        )

    def get_css_grid_template(self) -> Dict[str, str]:
        """Get CSS grid template styles."""
        return {
            'display': 'grid',
            'gridTemplateRows': f'repeat({self.rows}, 1fr)',
            'gridTemplateColumns': f'repeat({self.cols}, 1fr)',
            'gap': f'{self.gap}px',
        }

    def get_cell_style(self, plot_id: str) -> Dict[str, str]:
        """Get CSS style for a cell."""
        if plot_id not in self.cells:
            return {}

        cell = self.cells[plot_id]
        return {
            'gridRow': f'{cell.row + 1} / span {cell.row_span}',
            'gridColumn': f'{cell.col + 1} / span {cell.col_span}',
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rows': self.rows,
            'cols': self.cols,
            'gap': self.gap,
            'cells': {k: vars(v) for k, v in self.cells.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridLayout':
        """Create from dictionary."""
        layout = cls(
            rows=data.get('rows', 2),
            cols=data.get('cols', 2),
            gap=data.get('gap', 10)
        )
        for plot_id, cell_data in data.get('cells', {}).items():
            layout.cells[plot_id] = GridCell(**cell_data)
        return layout

