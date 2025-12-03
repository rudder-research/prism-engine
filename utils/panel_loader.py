"""
Panel Loader - Registry-driven panel data loading utility

Provides centralized panel loading using system_registry.json configuration.
All engines should use this module instead of hardcoded paths.

Usage:
    from utils.panel_loader import load_panel, get_registry, get_panel_path

    # Load default panel
    df = load_panel()

    # Load specific panel
    df = load_panel(panel_name="climate")

    # Get panel path for custom loading
    path = get_panel_path("default")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)

# Project root detection
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Registry paths
SYSTEM_REGISTRY_PATH = _PROJECT_ROOT / "data" / "registry" / "system_registry.json"
METRIC_REGISTRY_PATH = _PROJECT_ROOT / "data" / "registry" / "metric_registry.json"


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class PanelLoadError(Exception):
    """Exception raised for panel loading errors."""
    pass


def get_project_root() -> Path:
    """Get the project root directory."""
    return _PROJECT_ROOT


def get_registry(registry_type: str = "system") -> Dict[str, Any]:
    """
    Load a registry file.

    Args:
        registry_type: "system" or "metrics"

    Returns:
        Dictionary with registry contents

    Raises:
        RegistryError: If registry cannot be loaded
    """
    if registry_type == "system":
        registry_path = SYSTEM_REGISTRY_PATH
    elif registry_type == "metrics":
        registry_path = METRIC_REGISTRY_PATH
    else:
        raise RegistryError(f"Unknown registry type: {registry_type}")

    if not registry_path.exists():
        raise RegistryError(f"Registry not found: {registry_path}")

    try:
        with open(registry_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RegistryError(f"Invalid JSON in registry: {e}")


def get_panel_path(panel_name: str = "default") -> Path:
    """
    Get the path to a panel file from the registry.

    Args:
        panel_name: Name of the panel ("default", "climate", "global", "test")

    Returns:
        Absolute path to the panel file

    Raises:
        RegistryError: If panel configuration not found
    """
    registry = get_registry("system")

    # Check for specific panel configuration
    if "panels" in registry and panel_name in registry["panels"]:
        relative_path = registry["panels"][panel_name]["path"]
    elif panel_name == "default" and "panel_master_path" in registry:
        relative_path = registry["panel_master_path"]
    else:
        raise RegistryError(f"Panel '{panel_name}' not found in registry")

    return _PROJECT_ROOT / relative_path


def get_panel_format(panel_name: str = "default") -> str:
    """
    Get the format of a panel file.

    Args:
        panel_name: Name of the panel

    Returns:
        Format string ("csv" or "parquet")
    """
    registry = get_registry("system")

    if "panels" in registry and panel_name in registry["panels"]:
        return registry["panels"][panel_name].get("format", "csv")

    return registry.get("panel_format", "csv")


def load_panel(
    panel_name: str = "default",
    columns: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_na: bool = True,
    custom_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a panel DataFrame from the registry-configured path.

    Args:
        panel_name: Name of the panel to load ("default", "climate", etc.)
        columns: Optional list of columns to load (loads all if None)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        fill_na: Whether to forward-fill then backward-fill NaN values
        custom_path: Override registry path with custom path (for backward compatibility)

    Returns:
        DataFrame with date index and indicator columns

    Raises:
        PanelLoadError: If panel cannot be loaded
    """
    # Get path
    if custom_path is not None:
        panel_path = Path(custom_path)
        logger.warning(f"Using custom path instead of registry: {panel_path}")
    else:
        panel_path = get_panel_path(panel_name)

    if not panel_path.exists():
        raise PanelLoadError(f"Panel file not found: {panel_path}")

    # Get format
    panel_format = get_panel_format(panel_name) if custom_path is None else "csv"

    # Load data
    try:
        if panel_format == "parquet":
            df = pd.read_parquet(panel_path)
        else:
            df = pd.read_csv(panel_path, index_col=0, parse_dates=True)

        logger.info(f"Loaded panel '{panel_name}' from {panel_path}: {df.shape}")
    except Exception as e:
        raise PanelLoadError(f"Failed to load panel: {e}")

    # Reset index to get date as column
    if df.index.name is None:
        df.index.name = "date"
    df = df.reset_index()

    # Ensure date column
    registry = get_registry("system")
    date_col = registry.get("column_mappings", {}).get("date_column", "date")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        date_col = "date"

    # Filter columns
    if columns is not None:
        available_cols = [c for c in columns if c in df.columns]
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            logger.warning(f"Requested columns not found: {missing_cols}")
        df = df[[date_col] + available_cols]

    # Filter dates
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # Fill NaN values
    if fill_na:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


def get_engine_indicators(engine_type: str) -> List[str]:
    """
    Get the default indicators for a specific engine type.

    Args:
        engine_type: "macro", "market", "stress", or "rates"

    Returns:
        List of indicator column names
    """
    registry = get_registry("system")
    defaults = registry.get("engine_defaults", {})

    if engine_type not in defaults:
        logger.warning(f"No default indicators for engine type '{engine_type}'")
        return []

    return defaults[engine_type].get("indicators", [])


def get_metric_registry() -> List[Dict[str, str]]:
    """
    Load the metric registry.

    Returns:
        List of metric definitions with keys: key, source, ticker
    """
    return get_registry("metrics")


def get_metric_by_key(key: str) -> Optional[Dict[str, str]]:
    """
    Get metric definition by key.

    Args:
        key: Metric key (e.g., "spy", "cpi")

    Returns:
        Metric definition dict or None if not found
    """
    metrics = get_metric_registry()
    for metric in metrics:
        if metric.get("key") == key:
            return metric
    return None


def list_available_panels() -> List[str]:
    """List all available panel names in the registry."""
    registry = get_registry("system")
    return list(registry.get("panels", {}).keys())


def validate_panel(panel_name: str = "default") -> Dict[str, Any]:
    """
    Validate a panel's configuration and data.

    Args:
        panel_name: Name of the panel to validate

    Returns:
        Dictionary with validation results
    """
    result = {
        "panel_name": panel_name,
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    # Check registry configuration
    try:
        path = get_panel_path(panel_name)
        result["info"]["path"] = str(path)
    except RegistryError as e:
        result["valid"] = False
        result["errors"].append(f"Registry error: {e}")
        return result

    # Check file exists
    if not path.exists():
        result["valid"] = False
        result["errors"].append(f"Panel file not found: {path}")
        return result

    # Load and check data
    try:
        df = load_panel(panel_name)
        result["info"]["shape"] = df.shape
        result["info"]["columns"] = list(df.columns)
        result["info"]["date_range"] = [
            str(df["date"].min()) if "date" in df.columns else "N/A",
            str(df["date"].max()) if "date" in df.columns else "N/A"
        ]

        # Check for excessive NaN
        nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if nan_pct > 50:
            result["warnings"].append(f"High NaN percentage: {nan_pct:.1f}%")
        result["info"]["nan_percentage"] = nan_pct

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Load error: {e}")

    return result
