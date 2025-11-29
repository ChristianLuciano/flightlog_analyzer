"""
Theme management for visualizations.

Provides theming support for plots including light/dark modes
and color scheme customization.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.constants import COLOR_SCHEME_LIGHT, COLOR_SCHEME_DARK


class ThemeMode(Enum):
    """Theme mode options."""
    LIGHT = "light"
    DARK = "dark"
    CUSTOM = "custom"


@dataclass
class Theme:
    """
    Theme configuration for visualizations.

    Contains color schemes and styling options for plots and UI.
    """

    mode: ThemeMode = ThemeMode.LIGHT
    name: str = "default"

    # Main colors
    background: str = "#ffffff"
    paper_background: str = "#ffffff"
    text: str = "#2c3e50"
    text_secondary: str = "#7f8c8d"

    # Accent colors
    primary: str = "#3498db"
    secondary: str = "#2ecc71"
    accent: str = "#e74c3c"
    warning: str = "#f39c12"

    # Plot colors
    grid: str = "#ecf0f1"
    axis: str = "#bdc3c7"
    plot_background: str = "#ffffff"

    # Signal colors (for multiple traces)
    signal_colors: List[str] = field(default_factory=lambda: [
        "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
        "#f39c12", "#1abc9c", "#e91e63", "#00bcd4",
        "#8bc34a", "#ff5722", "#607d8b", "#795548"
    ])

    # Event colors by severity
    event_colors: Dict[str, str] = field(default_factory=lambda: {
        "critical": "#e74c3c",
        "warning": "#f39c12",
        "info": "#3498db",
        "debug": "#95a5a6",
    })

    # Colorscales for heatmaps/spectrograms
    sequential_colorscale: str = "Viridis"
    diverging_colorscale: str = "RdBu"

    # Font settings
    font_family: str = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    font_size: int = 12
    title_font_size: int = 14

    def to_plotly_template(self) -> Dict[str, Any]:
        """
        Convert theme to Plotly template format.

        Returns:
            Plotly template dictionary.
        """
        return {
            "layout": {
                "paper_bgcolor": self.paper_background,
                "plot_bgcolor": self.plot_background,
                "font": {
                    "family": self.font_family,
                    "size": self.font_size,
                    "color": self.text,
                },
                "title": {
                    "font": {
                        "size": self.title_font_size,
                        "color": self.text,
                    }
                },
                "xaxis": {
                    "gridcolor": self.grid,
                    "linecolor": self.axis,
                    "zerolinecolor": self.axis,
                },
                "yaxis": {
                    "gridcolor": self.grid,
                    "linecolor": self.axis,
                    "zerolinecolor": self.axis,
                },
                "colorway": self.signal_colors,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert theme to dictionary."""
        return {
            "mode": self.mode.value,
            "name": self.name,
            "background": self.background,
            "paper_background": self.paper_background,
            "text": self.text,
            "text_secondary": self.text_secondary,
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "warning": self.warning,
            "grid": self.grid,
            "axis": self.axis,
            "plot_background": self.plot_background,
            "signal_colors": self.signal_colors,
            "event_colors": self.event_colors,
            "font_family": self.font_family,
            "font_size": self.font_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theme":
        """Create theme from dictionary."""
        mode = ThemeMode(data.get("mode", "light"))
        return cls(
            mode=mode,
            name=data.get("name", "custom"),
            background=data.get("background", "#ffffff"),
            paper_background=data.get("paper_background", "#ffffff"),
            text=data.get("text", "#2c3e50"),
            text_secondary=data.get("text_secondary", "#7f8c8d"),
            primary=data.get("primary", "#3498db"),
            secondary=data.get("secondary", "#2ecc71"),
            accent=data.get("accent", "#e74c3c"),
            warning=data.get("warning", "#f39c12"),
            grid=data.get("grid", "#ecf0f1"),
            axis=data.get("axis", "#bdc3c7"),
            plot_background=data.get("plot_background", "#ffffff"),
            signal_colors=data.get("signal_colors", []),
            event_colors=data.get("event_colors", {}),
            font_family=data.get("font_family", "Inter, sans-serif"),
            font_size=data.get("font_size", 12),
        )


# Predefined themes
LIGHT_THEME = Theme(
    mode=ThemeMode.LIGHT,
    name="Light",
    background="#ffffff",
    paper_background="#ffffff",
    text="#2c3e50",
    text_secondary="#7f8c8d",
    primary="#3498db",
    secondary="#2ecc71",
    accent="#e74c3c",
    grid="#ecf0f1",
    axis="#bdc3c7",
    plot_background="#ffffff",
)

DARK_THEME = Theme(
    mode=ThemeMode.DARK,
    name="Dark",
    background="#1a1a2e",
    paper_background="#16213e",
    text="#eaeaea",
    text_secondary="#a0a0a0",
    primary="#00d4ff",
    secondary="#00ff88",
    accent="#ff6b6b",
    grid="#2d2d44",
    axis="#3d3d5c",
    plot_background="#16213e",
    signal_colors=[
        "#00d4ff", "#ff6b6b", "#00ff88", "#d68fff",
        "#ffb347", "#00ffff", "#ff69b4", "#87ceeb",
        "#98fb98", "#ffa07a", "#b0c4de", "#deb887"
    ],
)

# Aviation-inspired theme
AVIATION_THEME = Theme(
    mode=ThemeMode.DARK,
    name="Aviation",
    background="#0a1628",
    paper_background="#0d1f3c",
    text="#00ff00",
    text_secondary="#00cc00",
    primary="#00ff00",
    secondary="#00ffff",
    accent="#ff0000",
    warning="#ffff00",
    grid="#1a3a5c",
    axis="#2a5a8c",
    plot_background="#0d1f3c",
    signal_colors=[
        "#00ff00", "#00ffff", "#ffff00", "#ff00ff",
        "#ff8000", "#00ff80", "#8080ff", "#ff0080"
    ],
)

# Theme registry
THEMES: Dict[str, Theme] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "aviation": AVIATION_THEME,
}


def get_theme(name: str = "light") -> Theme:
    """
    Get theme by name.

    Args:
        name: Theme name ('light', 'dark', 'aviation').

    Returns:
        Theme instance.
    """
    return THEMES.get(name.lower(), LIGHT_THEME)


def register_theme(name: str, theme: Theme) -> None:
    """
    Register a custom theme.

    Args:
        name: Theme name.
        theme: Theme instance.
    """
    THEMES[name.lower()] = theme


def list_themes() -> List[str]:
    """Get list of available theme names."""
    return list(THEMES.keys())

