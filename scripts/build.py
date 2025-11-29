#!/usr/bin/env python3
"""
Build script for Flight Log Analysis Dashboard.

Creates standalone executables for Windows, macOS, and Linux.

Usage:
    python scripts/build.py [--onefile] [--clean]

Implements:
- REQ-DEPLOY-001: Standalone executable for Windows
- REQ-DEPLOY-002: Standalone executable for macOS  
- REQ-DEPLOY-003: Standalone executable for Linux
- REQ-DEPLOY-005: Include all dependencies
"""

import subprocess
import sys
import shutil
import platform
from pathlib import Path
import argparse


def get_version():
    """Get current version from version module."""
    try:
        from src.core.version import get_version as gv
        return gv()
    except ImportError:
        return "0.1.0"


def clean_build_dirs():
    """Remove build and dist directories."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Cleaned: {dir_path}")


def install_dependencies():
    """Install build dependencies."""
    print("Installing build dependencies...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'pyinstaller', 'wheel', 'build'
    ], check=True)


def build_executable(onefile=False):
    """Build standalone executable using PyInstaller."""
    print(f"\nBuilding executable for {platform.system()}...")
    
    cmd = [sys.executable, '-m', 'PyInstaller', 'flight_log_dashboard.spec']
    
    if onefile:
        cmd.append('--onefile')
    
    subprocess.run(cmd, check=True)
    
    # Get output location
    if onefile:
        output_dir = Path('dist')
    else:
        output_dir = Path('dist/FlightLogDashboard')
    
    print(f"\n✓ Build complete: {output_dir}")
    return output_dir


def build_wheel():
    """Build Python wheel package."""
    print("\nBuilding Python wheel...")
    subprocess.run([sys.executable, '-m', 'build'], check=True)
    print("\n✓ Wheel built in dist/")


def create_archive(output_dir):
    """Create distributable archive."""
    version = get_version()
    system = platform.system().lower()
    arch = platform.machine()
    
    archive_name = f"FlightLogDashboard-{version}-{system}-{arch}"
    
    if system == 'windows':
        # Create ZIP for Windows
        shutil.make_archive(
            f'dist/{archive_name}',
            'zip',
            output_dir.parent,
            output_dir.name
        )
        print(f"✓ Created: dist/{archive_name}.zip")
        
    elif system == 'darwin':
        # Create DMG for macOS (simplified - just tar.gz for now)
        shutil.make_archive(
            f'dist/{archive_name}',
            'gztar',
            output_dir.parent,
            output_dir.name
        )
        print(f"✓ Created: dist/{archive_name}.tar.gz")
        
    else:
        # Create tar.gz for Linux
        shutil.make_archive(
            f'dist/{archive_name}',
            'gztar',
            output_dir.parent,
            output_dir.name
        )
        print(f"✓ Created: dist/{archive_name}.tar.gz")


def main():
    parser = argparse.ArgumentParser(description='Build Flight Log Dashboard')
    parser.add_argument('--onefile', action='store_true', 
                       help='Create single-file executable')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build directories before building')
    parser.add_argument('--wheel', action='store_true',
                       help='Build Python wheel package only')
    parser.add_argument('--archive', action='store_true',
                       help='Create distributable archive')
    
    args = parser.parse_args()
    
    print(f"Flight Log Dashboard Build Script")
    print(f"Version: {get_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("-" * 40)
    
    if args.clean:
        clean_build_dirs()
    
    install_dependencies()
    
    if args.wheel:
        build_wheel()
    else:
        output_dir = build_executable(args.onefile)
        
        if args.archive:
            create_archive(output_dir)
    
    print("\n" + "=" * 40)
    print("Build complete!")
    print("=" * 40)


if __name__ == '__main__':
    main()

