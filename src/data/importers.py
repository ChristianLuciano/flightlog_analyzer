"""
Data import functionality.

Provides import from various flight log formats.

Covers requirements:
- REQ-INT-001: Import from MAVLink .tlog files
- REQ-INT-002: Import from PX4 .ulg files
- REQ-INT-003: Import from ArduPilot .bin files
- REQ-INT-004: Import from CSV with configurable column mapping
- REQ-INT-005: Import from Excel with sheet selection
- REQ-INT-006: Support drag-and-drop file import
- REQ-INT-007: Provide import wizard with auto-detection
- REQ-INT-008: Support batch import of multiple files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple
import logging
import struct
from abc import ABC, abstractmethod

from ..core.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class BaseImporter(ABC):
    """Base class for data importers."""
    
    @abstractmethod
    def can_import(self, path: Path) -> bool:
        """Check if this importer can handle the file."""
        pass
    
    @abstractmethod
    def import_file(self, path: Path, **kwargs) -> Dict[str, pd.DataFrame]:
        """Import file and return dict of DataFrames."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass


class CSVImporter(BaseImporter):
    """
    Import from CSV files with configurable column mapping.
    
    REQ-INT-004: Import from CSV with configurable column mapping
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.csv', '.txt', '.tsv']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(
        self,
        path: Path,
        column_mapping: Optional[Dict[str, str]] = None,
        timestamp_column: str = 'timestamp',
        delimiter: str = ',',
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Import CSV file.
        
        Args:
            path: File path.
            column_mapping: Dict mapping old column names to new names.
            timestamp_column: Name of timestamp column.
            delimiter: Column delimiter.
            **kwargs: Additional pandas read_csv arguments.
        
        Returns:
            Dict with single DataFrame under filename key.
        """
        try:
            df = pd.read_csv(path, delimiter=delimiter, **kwargs)
            
            # Apply column mapping
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Auto-detect timestamp column if not present
            if timestamp_column not in df.columns:
                for col in ['time', 't', 'Time', 'TIMESTAMP', 'epoch']:
                    if col in df.columns:
                        df = df.rename(columns={col: timestamp_column})
                        break
            
            name = path.stem
            logger.info(f"Imported CSV: {name} ({len(df)} rows, {len(df.columns)} columns)")
            
            return {name: df}
            
        except Exception as e:
            raise DataLoadError(f"Failed to import CSV {path}: {e}")


class ExcelImporter(BaseImporter):
    """
    Import from Excel files with sheet selection.
    
    REQ-INT-005: Import from Excel with sheet selection
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.xlsx', '.xls', '.xlsm']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(
        self,
        path: Path,
        sheets: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Import Excel file.
        
        Args:
            path: File path.
            sheets: List of sheet names to import (None = all sheets).
            **kwargs: Additional pandas read_excel arguments.
        
        Returns:
            Dict of sheet_name: DataFrame.
        """
        try:
            excel_file = pd.ExcelFile(path)
            available_sheets = excel_file.sheet_names
            
            if sheets is None:
                sheets = available_sheets
            else:
                sheets = [s for s in sheets if s in available_sheets]
            
            result = {}
            for sheet in sheets:
                df = pd.read_excel(excel_file, sheet_name=sheet, **kwargs)
                result[sheet] = df
                logger.info(f"Imported sheet '{sheet}': {len(df)} rows")
            
            return result
            
        except Exception as e:
            raise DataLoadError(f"Failed to import Excel {path}: {e}")


class MATLABImporter(BaseImporter):
    """
    Import from MATLAB .mat files.
    
    REQ-INT-009: Support MATLAB format (bidirectional)
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.mat']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(self, path: Path, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Import MATLAB .mat file.
        
        Args:
            path: File path.
            **kwargs: Additional options.
        
        Returns:
            Dict of variable_name: DataFrame.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise DataLoadError("scipy not installed. Install with: pip install scipy")
        
        try:
            mat_data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
            
            result = {}
            for key, value in mat_data.items():
                if key.startswith('_'):
                    continue
                
                df = self._convert_to_dataframe(value)
                if df is not None:
                    result[key] = df
                    logger.info(f"Imported MATLAB variable '{key}': {len(df)} rows")
            
            return result
            
        except Exception as e:
            raise DataLoadError(f"Failed to import MATLAB file {path}: {e}")
    
    def _convert_to_dataframe(self, data) -> Optional[pd.DataFrame]:
        """Convert MATLAB data to DataFrame."""
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                return pd.DataFrame(data)
            elif data.ndim == 1:
                return pd.DataFrame({'value': data})
        elif hasattr(data, '_fieldnames'):
            # MATLAB struct
            result = {}
            for field in data._fieldnames:
                value = getattr(data, field)
                if isinstance(value, np.ndarray):
                    result[field] = value
            if result:
                return pd.DataFrame(result)
        return None


class MAVLinkImporter(BaseImporter):
    """
    Import from MAVLink .tlog files.
    
    REQ-INT-001: Import from MAVLink .tlog files
    
    Note: Full implementation requires pymavlink library.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.tlog', '.mavlink']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(self, path: Path, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Import MAVLink telemetry log.
        
        Args:
            path: File path.
            **kwargs: Additional options.
        
        Returns:
            Dict of message_type: DataFrame.
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            raise DataLoadError(
                "pymavlink not installed. Install with: pip install pymavlink"
            )
        
        try:
            mlog = mavutil.mavlink_connection(str(path))
            
            messages = {}
            while True:
                msg = mlog.recv_match(blocking=False)
                if msg is None:
                    break
                
                msg_type = msg.get_type()
                if msg_type == 'BAD_DATA':
                    continue
                
                if msg_type not in messages:
                    messages[msg_type] = []
                
                msg_dict = msg.to_dict()
                msg_dict['timestamp'] = getattr(msg, '_timestamp', 0)
                messages[msg_type].append(msg_dict)
            
            result = {}
            for msg_type, msg_list in messages.items():
                if msg_list:
                    df = pd.DataFrame(msg_list)
                    result[msg_type] = df
                    logger.info(f"Imported MAVLink message '{msg_type}': {len(df)} messages")
            
            return result
            
        except Exception as e:
            raise DataLoadError(f"Failed to import MAVLink file {path}: {e}")


class PX4Importer(BaseImporter):
    """
    Import from PX4 .ulg files.
    
    REQ-INT-002: Import from PX4 .ulg files
    
    Note: Full implementation requires pyulog library.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.ulg']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(self, path: Path, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Import PX4 ULog file.
        
        Args:
            path: File path.
            **kwargs: Additional options.
        
        Returns:
            Dict of topic_name: DataFrame.
        """
        try:
            from pyulog import ULog
        except ImportError:
            raise DataLoadError(
                "pyulog not installed. Install with: pip install pyulog"
            )
        
        try:
            ulog = ULog(str(path))
            
            result = {}
            for data in ulog.data_list:
                topic = data.name
                
                df_data = {'timestamp': data.data['timestamp']}
                for field in data.data:
                    if field != 'timestamp':
                        df_data[field] = data.data[field]
                
                df = pd.DataFrame(df_data)
                result[topic] = df
                logger.info(f"Imported PX4 topic '{topic}': {len(df)} rows")
            
            return result
            
        except Exception as e:
            raise DataLoadError(f"Failed to import PX4 file {path}: {e}")


class ArduPilotImporter(BaseImporter):
    """
    Import from ArduPilot .bin files.
    
    REQ-INT-003: Import from ArduPilot .bin files
    
    Note: Full implementation requires pymavlink library.
    """
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.bin', '.log']
    
    def can_import(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    def import_file(self, path: Path, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Import ArduPilot binary log.
        
        Args:
            path: File path.
            **kwargs: Additional options.
        
        Returns:
            Dict of message_type: DataFrame.
        """
        try:
            from pymavlink import mavutil
            from pymavlink import DFReader
        except ImportError:
            raise DataLoadError(
                "pymavlink not installed. Install with: pip install pymavlink"
            )
        
        try:
            if path.suffix.lower() == '.bin':
                mlog = DFReader.DFReader_binary(str(path))
            else:
                mlog = DFReader.DFReader_text(str(path))
            
            messages = {}
            while True:
                msg = mlog.recv_msg()
                if msg is None:
                    break
                
                msg_type = msg.get_type()
                if msg_type not in messages:
                    messages[msg_type] = []
                
                msg_dict = {}
                for field in msg._fieldnames:
                    msg_dict[field] = getattr(msg, field)
                msg_dict['timestamp'] = msg._timestamp
                messages[msg_type].append(msg_dict)
            
            result = {}
            for msg_type, msg_list in messages.items():
                if msg_list:
                    df = pd.DataFrame(msg_list)
                    result[msg_type] = df
                    logger.info(f"Imported ArduPilot message '{msg_type}': {len(df)} messages")
            
            return result
            
        except Exception as e:
            raise DataLoadError(f"Failed to import ArduPilot file {path}: {e}")


class AutoImporter:
    """
    Auto-detect file format and import.
    
    REQ-INT-007: Provide import wizard with auto-detection
    """
    
    def __init__(self):
        self.importers = [
            CSVImporter(),
            ExcelImporter(),
            MATLABImporter(),
            MAVLinkImporter(),
            PX4Importer(),
            ArduPilotImporter(),
        ]
    
    def detect_format(self, path: Union[str, Path]) -> Optional[BaseImporter]:
        """
        Detect file format and return appropriate importer.
        
        Args:
            path: File path.
        
        Returns:
            Importer instance or None if not supported.
        """
        path = Path(path)
        
        for importer in self.importers:
            if importer.can_import(path):
                return importer
        
        return None
    
    def import_file(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Auto-detect format and import file.
        
        Args:
            path: File path.
            **kwargs: Import options.
        
        Returns:
            Dict of DataFrames.
        """
        path = Path(path)
        
        importer = self.detect_format(path)
        if importer is None:
            raise DataLoadError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Auto-detected format: {importer.__class__.__name__}")
        return importer.import_file(path, **kwargs)
    
    def import_batch(
        self,
        paths: List[Union[str, Path]],
        merge_strategy: str = 'separate',
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Import multiple files.
        
        REQ-INT-008: Support batch import of multiple files
        
        Args:
            paths: List of file paths.
            merge_strategy: 'separate' (keep separate) or 'merge' (combine same types).
            **kwargs: Import options.
        
        Returns:
            Dict of DataFrames.
        """
        result = {}
        
        for path in paths:
            path = Path(path)
            try:
                data = self.import_file(path, **kwargs)
                
                for name, df in data.items():
                    if merge_strategy == 'separate':
                        # Prefix with filename
                        key = f"{path.stem}.{name}"
                    else:
                        key = name
                    
                    if key in result and merge_strategy == 'merge':
                        # Concatenate DataFrames
                        result[key] = pd.concat([result[key], df], ignore_index=True)
                    else:
                        result[key] = df
                        
            except Exception as e:
                logger.warning(f"Failed to import {path}: {e}")
        
        return result
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get dict of importer name to supported extensions."""
        return {
            importer.__class__.__name__: importer.supported_extensions
            for importer in self.importers
        }


# Convenience function
def import_flight_data(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Import flight data from various formats.
    
    Args:
        path: File path.
        format: Force specific format (None for auto-detect).
        **kwargs: Format-specific options.
    
    Returns:
        Dict of DataFrames.
    """
    path = Path(path)
    
    if format is None:
        return AutoImporter().import_file(path, **kwargs)
    
    importers = {
        'csv': CSVImporter(),
        'excel': ExcelImporter(),
        'matlab': MATLABImporter(),
        'mavlink': MAVLinkImporter(),
        'px4': PX4Importer(),
        'ardupilot': ArduPilotImporter(),
    }
    
    format = format.lower()
    if format not in importers:
        raise DataLoadError(f"Unknown format: {format}")
    
    return importers[format].import_file(path, **kwargs)

