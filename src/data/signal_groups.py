"""
Signal grouping and multi-source signal management.

Groups signals by type (measurement, command, estimated) for 
coordinated plotting on the same axis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
import re


class SignalSource(Enum):
    """Source type for a signal."""
    MEASUREMENT = "measurement"    # Raw sensor data
    COMMAND = "command"           # Commanded/setpoint values
    ESTIMATED = "estimated"       # Filtered/estimated values (e.g., Kalman)
    REFERENCE = "reference"       # Reference/target values
    SIMULATED = "simulated"       # Simulation output
    RAW = "raw"                   # Unprocessed raw data
    UNKNOWN = ""                  # Unknown/default source


# Display properties for each source type
SOURCE_STYLES = {
    SignalSource.MEASUREMENT: {
        'color_index': 0,
        'dash': 'solid',
        'opacity': 1.0,
        'width': 2,
        'marker': 'circle',
        'label_suffix': ' (meas)',
        'legend_group': 'Measurement'
    },
    SignalSource.COMMAND: {
        'color_index': 1,
        'dash': 'dash',
        'opacity': 0.9,
        'width': 2,
        'marker': 'square',
        'label_suffix': ' (cmd)',
        'legend_group': 'Command'
    },
    SignalSource.ESTIMATED: {
        'color_index': 2,
        'dash': 'dot',
        'opacity': 0.9,
        'width': 2.5,
        'marker': 'diamond',
        'label_suffix': ' (est)',
        'legend_group': 'Estimated'
    },
    SignalSource.REFERENCE: {
        'color_index': 3,
        'dash': 'dashdot',
        'opacity': 0.8,
        'width': 1.5,
        'marker': 'triangle-up',
        'label_suffix': ' (ref)',
        'legend_group': 'Reference'
    },
    SignalSource.SIMULATED: {
        'color_index': 4,
        'dash': 'longdash',
        'opacity': 0.7,
        'width': 1.5,
        'marker': 'star',
        'label_suffix': ' (sim)',
        'legend_group': 'Simulated'
    },
    SignalSource.RAW: {
        'color_index': 5,
        'dash': 'solid',
        'opacity': 0.5,
        'width': 1,
        'marker': 'x',
        'label_suffix': ' (raw)',
        'legend_group': 'Raw'
    },
    SignalSource.UNKNOWN: {
        'color_index': 0,
        'dash': 'solid',
        'opacity': 1.0,
        'width': 2,
        'marker': 'circle',
        'label_suffix': '',
        'legend_group': 'Data'
    },
}

# Color palette for signal base types (same color for grouped signals)
SIGNAL_COLOR_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf',  # Cyan
]

# Source-specific color variations (lighter/darker shades)
SOURCE_COLOR_MODIFIERS = {
    SignalSource.MEASUREMENT: 1.0,    # Base color
    SignalSource.COMMAND: 0.7,        # Darker
    SignalSource.ESTIMATED: 1.2,      # Lighter
    SignalSource.REFERENCE: 0.5,      # Much darker
    SignalSource.SIMULATED: 1.4,      # Much lighter
    SignalSource.RAW: 0.85,           # Slightly darker
    SignalSource.UNKNOWN: 1.0,
}


@dataclass
class ParsedSignal:
    """Parsed signal path with source and base type."""
    full_path: str
    source: SignalSource
    base_signal: str  # e.g., "position.latitude"
    column_name: str  # Original column name
    dataframe_path: str  # Path to containing DataFrame
    
    @property
    def group_key(self) -> str:
        """Key for grouping related signals."""
        return self.base_signal
    
    def get_style(self) -> Dict:
        """Get display style for this signal."""
        return SOURCE_STYLES.get(self.source, SOURCE_STYLES[SignalSource.UNKNOWN])


@dataclass 
class SignalGroup:
    """A group of related signals to be plotted together."""
    base_signal: str
    signals: List[ParsedSignal] = field(default_factory=list)
    color_index: int = 0
    
    def add_signal(self, signal: ParsedSignal):
        """Add a signal to this group."""
        self.signals.append(signal)
    
    def get_sources(self) -> List[SignalSource]:
        """Get all source types in this group."""
        return [s.source for s in self.signals]
    
    @property
    def base_color(self) -> str:
        """Get base color for this signal group."""
        return SIGNAL_COLOR_PALETTE[self.color_index % len(SIGNAL_COLOR_PALETTE)]


class SignalGroupManager:
    """
    Manages signal grouping for coordinated multi-source plotting.
    
    Parses signal paths to identify source types and groups related
    signals together for plotting on the same axis.
    """
    
    # Patterns to identify source types in signal paths
    # These patterns match both at start/end AND in the middle of paths
    SOURCE_PATTERNS = {
        SignalSource.MEASUREMENT: [
            r'(?:^|[_\.])meas(?:urement)?(?:[_\.]|$)',
            r'(?:^|[_\.])sensor[s]?(?:[_\.]|$)',
            r'(?:^|[_\.])actual(?:[_\.]|$)',
            r'(?:^|[_\.])measured(?:[_\.]|$)',
        ],
        SignalSource.COMMAND: [
            r'(?:^|[_\.])cmd(?:[_\.]|$)',
            r'(?:^|[_\.])command(?:[_\.]|$)',
            r'(?:^|[_\.])setpoint[s]?(?:[_\.]|$)',
            r'(?:^|[_\.])sp(?:[_\.]|$)',
            r'(?:^|[_\.])target(?:[_\.]|$)',
            r'(?:^|[_\.])desired(?:[_\.]|$)',
        ],
        SignalSource.ESTIMATED: [
            r'(?:^|[_\.])est(?:imated)?(?:[_\.]|$)',
            r'(?:^|[_\.])filtered(?:[_\.]|$)',
            r'(?:^|[_\.])filt(?:[_\.]|$)',
            r'(?:^|[_\.])ekf(?:[_\.]|$)',
            r'(?:^|[_\.])kalman(?:[_\.]|$)',
            r'(?:^|[_\.])fused(?:[_\.]|$)',
            r'(?:^|[_\.])state(?:[_\.]|$)',
            r'(?:^|[_\.])estimation(?:[_\.]|$)',
        ],
        SignalSource.REFERENCE: [
            r'(?:^|[_\.])ref(?:erence)?(?:[_\.]|$)',
            r'(?:^|[_\.])nominal(?:[_\.]|$)',
        ],
        SignalSource.SIMULATED: [
            r'(?:^|[_\.])sim(?:ulated)?(?:[_\.]|$)',
            r'(?:^|[_\.])model(?:[_\.]|$)',
            r'(?:^|[_\.])predicted(?:[_\.]|$)',
        ],
        SignalSource.RAW: [
            r'(?:^|[_\.])raw(?:[_\.]|$)',
            r'(?:^|[_\.])unfiltered(?:[_\.]|$)',
        ],
    }
    
    def __init__(self):
        self._groups: Dict[str, SignalGroup] = {}
        self._parsed_signals: Dict[str, ParsedSignal] = {}
        self._color_counter = 0
    
    def parse_signal(self, full_path: str) -> ParsedSignal:
        """
        Parse a signal path to extract source type and base signal.
        
        Args:
            full_path: Full path like "measurement.sensors.gps.latitude"
            
        Returns:
            ParsedSignal with source type and base signal identified
        """
        if full_path in self._parsed_signals:
            return self._parsed_signals[full_path]
        
        # Split path into parts
        parts = full_path.split('.')
        
        # Detect source type
        source = self._detect_source(full_path)
        
        # Extract base signal (remove source prefix/suffix)
        base_signal = self._extract_base_signal(parts, source)
        
        # Get column name (last part)
        column_name = parts[-1] if parts else full_path
        
        # Get dataframe path (everything except last part)
        df_path = '.'.join(parts[:-1]) if len(parts) > 1 else ''
        
        parsed = ParsedSignal(
            full_path=full_path,
            source=source,
            base_signal=base_signal,
            column_name=column_name,
            dataframe_path=df_path
        )
        
        self._parsed_signals[full_path] = parsed
        return parsed
    
    def _detect_source(self, path: str) -> SignalSource:
        """Detect signal source type from path."""
        path_lower = path.lower()
        
        for source, patterns in self.SOURCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path_lower):
                    return source
        
        return SignalSource.UNKNOWN
    
    def _extract_base_signal(self, parts: List[str], source: SignalSource) -> str:
        """Extract base signal name, removing source prefix/suffix."""
        if source == SignalSource.UNKNOWN:
            return '.'.join(parts)
        
        # Filter out source-related parts
        source_keywords = {
            SignalSource.MEASUREMENT: {'meas', 'measurement', 'sensor', 'sensors', 'actual', 'measured'},
            SignalSource.COMMAND: {'cmd', 'command', 'setpoint', 'sp', 'target', 'desired'},
            SignalSource.ESTIMATED: {'est', 'estimated', 'filtered', 'filt', 'ekf', 'kalman', 'fused', 'state'},
            SignalSource.REFERENCE: {'ref', 'reference', 'nominal'},
            SignalSource.SIMULATED: {'sim', 'simulated', 'model', 'predicted'},
            SignalSource.RAW: {'raw', 'unfiltered'},
        }
        
        keywords = source_keywords.get(source, set())
        filtered_parts = [p for p in parts if p.lower() not in keywords]
        
        return '.'.join(filtered_parts) if filtered_parts else '.'.join(parts)
    
    def group_signals(self, signal_paths: List[str]) -> Dict[str, SignalGroup]:
        """
        Group a list of signal paths by their base signal type.
        
        Args:
            signal_paths: List of full signal paths
            
        Returns:
            Dictionary of SignalGroups keyed by base signal
        """
        self._groups.clear()
        self._color_counter = 0
        
        for path in signal_paths:
            parsed = self.parse_signal(path)
            
            if parsed.base_signal not in self._groups:
                self._groups[parsed.base_signal] = SignalGroup(
                    base_signal=parsed.base_signal,
                    color_index=self._color_counter
                )
                self._color_counter += 1
            
            self._groups[parsed.base_signal].add_signal(parsed)
        
        return self._groups
    
    def get_group(self, base_signal: str) -> Optional[SignalGroup]:
        """Get signal group by base signal name."""
        return self._groups.get(base_signal)
    
    def get_all_groups(self) -> Dict[str, SignalGroup]:
        """Get all signal groups."""
        return self._groups.copy()
    
    def get_plot_config(self, signal_path: str, group_color_index: int = 0) -> Dict:
        """
        Get Plotly trace configuration for a signal.
        
        Args:
            signal_path: Full signal path
            group_color_index: Color index for the signal's group
            
        Returns:
            Dict with Plotly trace properties
        """
        parsed = self.parse_signal(signal_path)
        style = parsed.get_style()
        
        # Get base color from group
        base_color = SIGNAL_COLOR_PALETTE[group_color_index % len(SIGNAL_COLOR_PALETTE)]
        
        # Modify color based on source type
        modifier = SOURCE_COLOR_MODIFIERS.get(parsed.source, 1.0)
        color = self._modify_color(base_color, modifier)
        
        return {
            'name': f"{parsed.column_name}{style['label_suffix']}",
            'line': {
                'color': color,
                'dash': style['dash'],
                'width': style['width'],
            },
            'opacity': style['opacity'],
            'legendgroup': style['legend_group'],
            'legendgrouptitle_text': style['legend_group'] if style['legend_group'] else None,
        }
    
    def _modify_color(self, hex_color: str, modifier: float) -> str:
        """Modify a hex color by a brightness modifier."""
        # Parse hex color
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Apply modifier
        if modifier > 1:
            # Lighten
            factor = modifier - 1
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)
        else:
            # Darken
            r = int(r * modifier)
            g = int(g * modifier)
            b = int(b * modifier)
        
        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f'#{r:02x}{g:02x}{b:02x}'


def group_signals_for_plotting(signal_paths: List[str]) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Convenience function to group signals and get plot configurations.
    
    Args:
        signal_paths: List of signal paths to group
        
    Returns:
        Dictionary mapping base signal names to list of (path, config) tuples
    """
    manager = SignalGroupManager()
    groups = manager.group_signals(signal_paths)
    
    result = {}
    for base_signal, group in groups.items():
        configs = []
        for parsed in group.signals:
            config = manager.get_plot_config(parsed.full_path, group.color_index)
            configs.append((parsed.full_path, config))
        result[base_signal] = configs
    
    return result


def suggest_grouped_plots(signal_paths: List[str]) -> List[Dict]:
    """
    Suggest which signals should be plotted together.
    
    Args:
        signal_paths: All available signal paths
        
    Returns:
        List of plot suggestions with signal groups
    """
    manager = SignalGroupManager()
    groups = manager.group_signals(signal_paths)
    
    suggestions = []
    for base_signal, group in groups.items():
        if len(group.signals) > 1:
            # Multiple sources for same signal - suggest grouped plot
            sources = [s.source.value for s in group.signals if s.source != SignalSource.UNKNOWN]
            suggestions.append({
                'title': f"{base_signal}",
                'description': f"Compare {', '.join(set(sources))} signals",
                'signals': [s.full_path for s in group.signals],
                'type': 'comparison',
                'priority': len(group.signals)  # More signals = higher priority
            })
    
    # Sort by priority (most signals first)
    suggestions.sort(key=lambda x: x['priority'], reverse=True)
    
    return suggestions

