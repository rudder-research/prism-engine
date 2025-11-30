"""
PRISM Engine Configuration
==========================

Project metadata and version information.
"""

__version__ = "0.1.0"
__author__ = "PRISM Contributors"
__project__ = "PRISM Engine"

# Project paths are computed relative to this file
from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get project root directory, works in scripts, notebooks, and Colab.

    Returns:
        Path to the project root directory
    """
    # Check if running in Colab
    if 'COLAB_GPU' in os.environ or Path('/content').exists():
        # Colab with Drive mounted
        drive_path = Path('/content/drive/MyDrive/prism-engine')
        if drive_path.exists():
            return drive_path
        # Colab without Drive - check if prism-engine was cloned
        content_path = Path('/content/prism-engine')
        if content_path.exists():
            return content_path

    # Standard case: relative to this config file
    return Path(__file__).parent.resolve()


PROJECT_ROOT = get_project_root()

# Standard project directories
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / '06_output'
LOGS_DIR = PROJECT_ROOT / 'logs'


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
