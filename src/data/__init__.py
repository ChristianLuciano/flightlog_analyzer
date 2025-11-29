"""
Data handling module.

Provides functionality for loading, navigating, validating, caching,
aligning, importing, and exporting flight log data from hierarchical 
DataFrame structures.
"""

from .loader import DataLoader, load_flight_data
from .hierarchy import HierarchyNavigator, resolve_path
from .validator import DataValidator, EventDetector
from .cache import DataCache
from .alignment import TimeAligner, AlignmentMethod, align_signals
from .downsampling import downsample, lttb_downsample, douglas_peucker
from .importers import (
    AutoImporter,
    CSVImporter,
    ExcelImporter,
    MATLABImporter,
    MAVLinkImporter,
    PX4Importer,
    ArduPilotImporter,
    import_flight_data,
)
from .folder_importer import FolderImporter, import_flight_folder
from .signal_assignment import (
    SignalAssigner,
    SignalMapping,
    AssignmentConfig,
    ConversionFactor,
    StandardSignal,
    SignalSource,
    CONVERSION_PRESETS,
    get_standard_signals,
    get_conversion_presets,
    suggest_mappings,
)
from .signal_groups import (
    SignalGroupManager,
    SignalGroup,
    ParsedSignal,
    SignalSource as GroupSignalSource,
    SOURCE_STYLES,
    SIGNAL_COLOR_PALETTE,
    group_signals_for_plotting,
    suggest_grouped_plots,
)

