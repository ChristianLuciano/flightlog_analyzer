"""
Application state management.

Provides centralized state management for the dashboard.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json


@dataclass
class AppState:
    """
    Application state container.

    Holds all state information for the dashboard including
    loaded data references, UI state, and configuration.
    """

    # Data state
    data_loaded: bool = False
    data_path: str = ""
    time_range: tuple = (0.0, 0.0)
    current_time: float = 0.0

    # Playback state
    is_playing: bool = False
    playback_speed: float = 1.0

    # Selection state
    selected_signals: List[str] = field(default_factory=list)
    selected_tab: str = "tab-1"

    # View state
    zoom_range: Optional[tuple] = None
    map_center: Optional[tuple] = None
    map_zoom: int = 12

    # Configuration
    theme: str = "light"
    show_events: bool = True
    event_filters: List[str] = field(default_factory=lambda: ["critical", "warning"])

    # Plot state
    plots: Dict[str, Any] = field(default_factory=dict)
    grid_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert state to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppState":
        """Create state from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "AppState":
        """Create state from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def update(self, **kwargs) -> "AppState":
        """Update state with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class StateManager:
    """Manages application state with history."""

    def __init__(self, max_history: int = 50):
        self._state = AppState()
        self._history: List[AppState] = []
        self._max_history = max_history
        self._subscribers: List[callable] = []

    @property
    def state(self) -> AppState:
        """Get current state."""
        return self._state

    def update(self, **kwargs) -> None:
        """Update state and notify subscribers."""
        # Save to history
        self._history.append(AppState(**asdict(self._state)))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Update state
        self._state.update(**kwargs)

        # Notify subscribers
        for callback in self._subscribers:
            callback(self._state)

    def undo(self) -> bool:
        """Undo last state change."""
        if self._history:
            self._state = self._history.pop()
            return True
        return False

    def subscribe(self, callback: callable) -> None:
        """Subscribe to state changes."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: callable) -> None:
        """Unsubscribe from state changes."""
        self._subscribers.remove(callback)

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = AppState()
        self._history.clear()

