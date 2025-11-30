"""
Logging configuration for PRISM Engine
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Get project root (parent of utils directory)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent


def setup_logging(
    level: str = "INFO",
    log_file: bool = True,
    log_dir: Path = None
) -> logging.Logger:
    """
    Set up logging for PRISM Engine.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Whether to also log to file
        log_dir: Directory for log files (defaults to PROJECT_ROOT/logs)

    Returns:
        Root logger
    """
    # Default to logs directory in project root
    if log_dir is None:
        log_dir = _PROJECT_ROOT / "logs"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Get root logger
    logger = logging.getLogger("prism_engine")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(
            log_dir / f"prism_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
