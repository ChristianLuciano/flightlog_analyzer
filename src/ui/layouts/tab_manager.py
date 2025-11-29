"""
Tab management.

Handles creation and management of analysis tabs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .grid_layout import GridLayout


@dataclass
class Tab:
    """Represents an analysis tab."""
    id: str
    name: str
    grid: GridLayout = field(default_factory=GridLayout)
    is_active: bool = False
    is_loaded: bool = False


class TabManager:
    """
    Manages dashboard tabs.

    Handles tab creation, switching, and lazy loading.
    """

    def __init__(self):
        self._tabs: Dict[str, Tab] = {}
        self._order: List[str] = []
        self._active_tab: Optional[str] = None

    def add_tab(self, tab_id: str, name: str) -> Tab:
        """Add a new tab."""
        tab = Tab(id=tab_id, name=name)
        self._tabs[tab_id] = tab
        self._order.append(tab_id)

        if self._active_tab is None:
            self._active_tab = tab_id
            tab.is_active = True

        return tab

    def remove_tab(self, tab_id: str) -> bool:
        """Remove a tab."""
        if tab_id not in self._tabs:
            return False

        del self._tabs[tab_id]
        self._order.remove(tab_id)

        if self._active_tab == tab_id:
            self._active_tab = self._order[0] if self._order else None

        return True

    def get_tab(self, tab_id: str) -> Optional[Tab]:
        """Get tab by ID."""
        return self._tabs.get(tab_id)

    def set_active(self, tab_id: str) -> bool:
        """Set active tab."""
        if tab_id not in self._tabs:
            return False

        if self._active_tab:
            self._tabs[self._active_tab].is_active = False

        self._active_tab = tab_id
        self._tabs[tab_id].is_active = True
        self._tabs[tab_id].is_loaded = True  # Mark as loaded on first activation

        return True

    def get_active(self) -> Optional[Tab]:
        """Get active tab."""
        return self._tabs.get(self._active_tab) if self._active_tab else None

    def reorder(self, new_order: List[str]) -> bool:
        """Reorder tabs."""
        if set(new_order) != set(self._order):
            return False
        self._order = new_order
        return True

    def duplicate_tab(self, tab_id: str, new_name: str) -> Optional[Tab]:
        """Duplicate a tab."""
        source = self._tabs.get(tab_id)
        if not source:
            return None

        new_id = f"{tab_id}_copy"
        counter = 1
        while new_id in self._tabs:
            new_id = f"{tab_id}_copy_{counter}"
            counter += 1

        new_tab = Tab(
            id=new_id,
            name=new_name,
            grid=GridLayout.from_dict(source.grid.to_dict())
        )
        self._tabs[new_id] = new_tab
        self._order.append(new_id)

        return new_tab

    def get_tab_order(self) -> List[str]:
        """Get current tab order."""
        return self._order.copy()

    def list_tabs(self) -> List[Tab]:
        """Get all tabs in order."""
        return [self._tabs[tid] for tid in self._order]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tabs': {tid: {
                'id': t.id,
                'name': t.name,
                'grid': t.grid.to_dict()
            } for tid, t in self._tabs.items()},
            'order': self._order,
            'active': self._active_tab
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TabManager':
        """Create from dictionary."""
        manager = cls()
        for tab_data in data.get('tabs', {}).values():
            tab = Tab(
                id=tab_data['id'],
                name=tab_data['name'],
                grid=GridLayout.from_dict(tab_data.get('grid', {}))
            )
            manager._tabs[tab.id] = tab

        manager._order = data.get('order', list(manager._tabs.keys()))
        manager._active_tab = data.get('active')
        if manager._active_tab and manager._active_tab in manager._tabs:
            manager._tabs[manager._active_tab].is_active = True

        return manager

