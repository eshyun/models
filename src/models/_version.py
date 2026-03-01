"""Utility to read version from pyproject.toml."""

import sys
from pathlib import Path

def get_version() -> str:
    """
    Read the version from pyproject.toml.
    
    Returns:
        str: The version string from pyproject.toml
    """
    try:
        # Try to import tomllib (Python 3.11+)
        import tomllib
    except ImportError:
        # Fall back to tomli for Python < 3.11
        import tomli as tomllib
    
    # Navigate up from the current file to find pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    
    try:
        with open(pyproject_path, 'rb') as f:
            pyproject = tomllib.load(f)
        return pyproject['project']['version']
    except Exception as e:
        print(f"Error reading version from pyproject.toml: {e}", file=sys.stderr)
        return "0.0.0"  # Fallback version

# Set the version when this module is imported
__version__ = get_version()
