"""
Folder-based data importer.

Recursively imports CSV files from a directory structure,
organizing them hierarchically based on folder names.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import logging

from ..core.types import FlightDataDict
from ..core.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class FolderImporter:
    """
    Imports flight data from a folder structure.
    
    Recursively scans directories and imports all CSV files,
    organizing them into a hierarchical dictionary based on
    the folder structure.
    
    Example structure:
        Flight_001/
        ├── CU/
        │   ├── sensors/
        │   │   ├── imu.csv
        │   │   └── gps.csv
        │   └── control.csv
        ├── MU/
        │   └── motors.csv
        └── NU/
            └── nav.csv
    
    Results in:
        {
            'CU': {
                'sensors': {
                    'imu': DataFrame,
                    'gps': DataFrame
                },
                'control': DataFrame
            },
            'MU': {
                'motors': DataFrame
            },
            'NU': {
                'nav': DataFrame
            }
        }
    """
    
    SUPPORTED_EXTENSIONS = ['.csv', '.txt', '.tsv']
    
    def __init__(
        self,
        extensions: Optional[List[str]] = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        skip_errors: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize FolderImporter.
        
        Args:
            extensions: List of file extensions to import (default: csv, txt, tsv)
            delimiter: CSV delimiter character
            encoding: File encoding
            skip_errors: If True, skip files that fail to load
            progress_callback: Optional callback(filename, current, total)
        """
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS
        self.delimiter = delimiter
        self.encoding = encoding
        self.skip_errors = skip_errors
        self.progress_callback = progress_callback
        
        self._loaded_files: List[str] = []
        self._failed_files: List[tuple] = []
    
    def import_folder(
        self,
        folder_path: Union[str, Path],
        flatten: bool = False,
        include_root_name: bool = False
    ) -> FlightDataDict:
        """
        Import all data files from a folder.
        
        Args:
            folder_path: Path to the root folder
            flatten: If True, flatten the hierarchy (use dot-separated paths)
            include_root_name: If True, include root folder name in hierarchy
            
        Returns:
            Dictionary of DataFrames organized by folder structure
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise DataLoadError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise DataLoadError(f"Path is not a directory: {folder_path}")
        
        self._loaded_files = []
        self._failed_files = []
        
        # Find all matching files
        all_files = self._find_files(folder_path)
        total_files = len(all_files)
        
        if total_files == 0:
            logger.warning(f"No matching files found in {folder_path}")
            return {}
        
        logger.info(f"Found {total_files} files to import from {folder_path}")
        
        # Build the hierarchical structure
        data: FlightDataDict = {}
        
        for idx, file_path in enumerate(all_files):
            if self.progress_callback:
                self.progress_callback(file_path.name, idx + 1, total_files)
            
            try:
                df = self._load_file(file_path)
                
                # Get relative path from root folder
                rel_path = file_path.relative_to(folder_path)
                
                # Build hierarchy path
                if include_root_name:
                    parts = [folder_path.name] + list(rel_path.parts[:-1])
                else:
                    parts = list(rel_path.parts[:-1])
                
                # Use filename without extension as final key
                name = file_path.stem
                
                if flatten:
                    # Create dot-separated path
                    full_path = '.'.join(parts + [name]) if parts else name
                    data[full_path] = df
                else:
                    # Build nested dictionary
                    self._set_nested(data, parts, name, df)
                
                self._loaded_files.append(str(file_path))
                logger.debug(f"Loaded: {file_path}")
                
            except Exception as e:
                self._failed_files.append((str(file_path), str(e)))
                if self.skip_errors:
                    logger.warning(f"Failed to load {file_path}: {e}")
                else:
                    raise DataLoadError(f"Failed to load {file_path}: {e}")
        
        logger.info(
            f"Import complete: {len(self._loaded_files)} loaded, "
            f"{len(self._failed_files)} failed"
        )
        
        return data
    
    def _find_files(self, folder_path: Path) -> List[Path]:
        """Find all matching files recursively."""
        files = []
        
        for ext in self.extensions:
            # Handle both with and without leading dot
            ext = ext if ext.startswith('.') else f'.{ext}'
            files.extend(folder_path.rglob(f'*{ext}'))
        
        # Sort for consistent ordering
        return sorted(files)
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single file."""
        # Detect delimiter for txt/tsv files
        delimiter = self.delimiter
        if file_path.suffix.lower() in ['.tsv', '.txt']:
            # Try to detect delimiter
            with open(file_path, 'r', encoding=self.encoding) as f:
                first_line = f.readline()
                if '\t' in first_line:
                    delimiter = '\t'
                elif ';' in first_line:
                    delimiter = ';'
        
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=self.encoding
        )
        
        return df
    
    def _set_nested(
        self,
        data: dict,
        path: List[str],
        name: str,
        value: pd.DataFrame
    ) -> None:
        """Set a value in a nested dictionary."""
        current = data
        
        for part in path:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[name] = value
    
    @property
    def loaded_files(self) -> List[str]:
        """Get list of successfully loaded files."""
        return self._loaded_files.copy()
    
    @property
    def failed_files(self) -> List[tuple]:
        """Get list of failed files with error messages."""
        return self._failed_files.copy()
    
    def get_summary(self) -> Dict:
        """Get import summary."""
        return {
            'loaded_count': len(self._loaded_files),
            'failed_count': len(self._failed_files),
            'loaded_files': self._loaded_files,
            'failed_files': self._failed_files
        }


def import_flight_folder(
    folder_path: Union[str, Path],
    **kwargs
) -> FlightDataDict:
    """
    Convenience function to import a flight folder.
    
    Args:
        folder_path: Path to flight data folder
        **kwargs: Arguments passed to FolderImporter
        
    Returns:
        Hierarchical dictionary of DataFrames
    """
    importer = FolderImporter(**kwargs)
    return importer.import_folder(folder_path)

