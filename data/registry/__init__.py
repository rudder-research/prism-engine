"""
PRISM Registry Module
=====================

Loads and validates the metric registry for the PRISM Engine.

The registry defines all indicators, their sources, frequencies, and metadata.

Supports two registry formats:
1. YAML-based Full Institutional Pack (250+ indicators) - Primary
2. Legacy JSON format for backward compatibility

The YAML registry is the default and recommended format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent
YAML_DIR = REGISTRY_DIR / "yaml"


def load_metric_registry(use_yaml: bool = True) -> Dict[str, Any]:
    """
    Load the complete metric registry.

    By default, loads from the YAML-based Full Institutional Pack.
    Falls back to legacy JSON if YAML is not available.

    Args:
        use_yaml: If True, prefer YAML registry (default: True)

    Returns:
        Dictionary containing all registry data with keys:
            - version: Registry version string
            - market: List of market indicator definitions
            - economic: List of economic indicator definitions
            - synthetic: List of synthetic indicator definitions
            - technical: List of technical indicator definitions
            - yaml_registry: Full YAML registry if loaded (optional)
            - technical_rules: Technical auto-apply rules (optional)
    """
    # Try YAML registry first
    if use_yaml and YAML_DIR.exists():
        try:
            from data.registry.yaml_loader import (
                load_all_yaml_registries,
                convert_to_legacy_format,
            )

            yaml_registry = load_all_yaml_registries()
            registry = convert_to_legacy_format(yaml_registry)

            # Store the full YAML registry for advanced use
            registry["yaml_registry"] = yaml_registry

            logger.info(
                f"Loaded YAML registry v{registry['version']}: "
                f"{len(registry['market'])} market, "
                f"{len(registry['economic'])} economic, "
                f"{len(registry['synthetic'])} synthetic, "
                f"{len(registry['technical'])} technical"
            )

            return registry

        except Exception as e:
            logger.warning(f"Failed to load YAML registry, falling back to JSON: {e}")

    # Fall back to legacy JSON format
    return _load_legacy_json_registry()


def _load_legacy_json_registry() -> Dict[str, Any]:
    """Load the legacy JSON-based registry."""
    registry: Dict[str, Any] = {
        "version": "1.0.0",
        "market": [],
        "economic": [],
        "synthetic": [],
        "technical": [],
    }

    # Load market registry
    market_path = REGISTRY_DIR / "market_registry.json"
    if market_path.exists():
        with open(market_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                registry["market"] = data
            elif isinstance(data, dict):
                registry["market"] = data.get("indicators", data.get("market", []))
                if "version" in data:
                    registry["version"] = data["version"]

    # Load economic registry
    economic_path = REGISTRY_DIR / "economic_registry.json"
    if economic_path.exists():
        with open(economic_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                registry["economic"] = data
            elif isinstance(data, dict):
                registry["economic"] = data.get("indicators", data.get("economic", []))

    # Load metric registry (may contain synthetic/technical)
    metric_path = REGISTRY_DIR / "metric_registry.json"
    if metric_path.exists():
        with open(metric_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                registry["synthetic"] = data.get("synthetic", [])
                registry["technical"] = data.get("technical", [])
                if "version" in data:
                    registry["version"] = data["version"]

    # Load system registry for additional metadata
    system_path = REGISTRY_DIR / "system_registry.json"
    if system_path.exists():
        with open(system_path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                registry["system"] = data

    logger.info(
        f"Loaded legacy JSON registry v{registry['version']}: "
        f"{len(registry['market'])} market, "
        f"{len(registry['economic'])} economic, "
        f"{len(registry['synthetic'])} synthetic, "
        f"{len(registry['technical'])} technical"
    )

    return registry


def validate_registry(registry: Dict[str, Any]) -> bool:
    """
    Validate the metric registry structure and contents.

    Supports both YAML-based and legacy JSON registries.

    Args:
        registry: The registry dictionary to validate

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with details about the issue
    """
    # If YAML registry is present, use its validation
    if "yaml_registry" in registry:
        try:
            from data.registry.yaml_loader import validate_yaml_registry
            return validate_yaml_registry(registry["yaml_registry"])
        except ImportError:
            pass  # Fall through to legacy validation

    # Legacy validation
    required_keys = ["version", "market", "economic"]

    for key in required_keys:
        if key not in registry:
            raise ValueError(f"Registry missing required key: {key}")

    # Validate market indicators
    for i, indicator in enumerate(registry.get("market", [])):
        if not isinstance(indicator, dict):
            raise ValueError(f"Market indicator {i} is not a dictionary")
        if "name" not in indicator and "ticker" not in indicator:
            raise ValueError(f"Market indicator {i} missing 'name' or 'ticker'")

    # Validate economic indicators
    for i, indicator in enumerate(registry.get("economic", [])):
        if not isinstance(indicator, dict):
            raise ValueError(f"Economic indicator {i} is not a dictionary")
        if "name" not in indicator and "series_id" not in indicator:
            raise ValueError(f"Economic indicator {i} missing 'name' or 'series_id'")

    # Check for duplicate IDs
    all_ids = set()
    for section in ["market", "economic", "synthetic", "technical"]:
        for ind in registry.get(section, []):
            name = ind.get("name")
            if name in all_ids:
                raise ValueError(f"Duplicate indicator ID: {name}")
            all_ids.add(name)

    total_count = sum(len(registry.get(s, [])) for s in ["market", "economic", "synthetic", "technical"])
    logger.info(f"Registry validation passed: {total_count} indicators")
    return True


def get_market_tickers(registry: Dict[str, Any]) -> List[str]:
    """
    Extract Yahoo Finance tickers from the market registry.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of ticker symbols
    """
    tickers = []
    for indicator in registry.get("market", []):
        ticker = indicator.get("ticker") or indicator.get("name")
        if ticker:
            tickers.append(ticker)
    return tickers


def get_fred_series(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract FRED series information from the economic registry.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of dictionaries with 'name' and 'series_id' keys
    """
    series = []
    for indicator in registry.get("economic", []):
        name = indicator.get("name", "").lower()
        series_id = indicator.get("series_id") or indicator.get("fred_id")
        if name and series_id:
            series.append({"name": name, "series_id": series_id})
    return series


def get_synthetic_definitions(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract synthetic indicator definitions.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of synthetic indicator definitions with formula and inputs
    """
    synthetics = []
    for indicator in registry.get("synthetic", []):
        name = indicator.get("name")
        formula = indicator.get("formula")
        inputs = indicator.get("inputs") or indicator.get("depends_on", [])
        if name:
            synthetics.append({
                "name": name,
                "formula": formula,
                "inputs": inputs,
                "group": indicator.get("group"),
            })
    return synthetics


def get_technical_definitions(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract technical indicator definitions.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of technical indicator definitions
    """
    technicals = []
    for indicator in registry.get("technical", []):
        name = indicator.get("name")
        base = indicator.get("base")
        tech_type = indicator.get("type")
        params = indicator.get("params", {})
        if name and base:
            technicals.append({
                "name": name,
                "base": base,
                "type": tech_type,
                "params": params,
                "group": indicator.get("group"),
            })
    return technicals


def get_technical_auto_apply_rules(registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the technical indicator auto-apply rules.

    These rules specify which indicator types should be automatically
    applied to which asset groups.

    Args:
        registry: The loaded registry dictionary

    Returns:
        Dictionary with auto-apply rules
    """
    return registry.get("technical_rules", {})


def get_all_indicator_ids(registry: Dict[str, Any]) -> List[str]:
    """
    Get all indicator IDs from the registry.

    Args:
        registry: The loaded registry dictionary

    Returns:
        List of all indicator IDs
    """
    ids = []
    for section in ["market", "economic", "synthetic", "technical"]:
        for ind in registry.get(section, []):
            name = ind.get("name")
            if name:
                ids.append(name)
    return ids


def get_indicator_count(registry: Dict[str, Any]) -> Dict[str, int]:
    """
    Get indicator counts by category.

    Args:
        registry: The loaded registry dictionary

    Returns:
        Dictionary mapping category to count
    """
    return {
        "market": len(registry.get("market", [])),
        "economic": len(registry.get("economic", [])),
        "synthetic": len(registry.get("synthetic", [])),
        "technical": len(registry.get("technical", [])),
        "total": sum(
            len(registry.get(s, []))
            for s in ["market", "economic", "synthetic", "technical"]
        ),
    }


def print_registry_summary(registry: Dict[str, Any]) -> None:
    """Print a summary of the loaded registry."""
    counts = get_indicator_count(registry)
    print("\n" + "=" * 50)
    print("PRISM METRIC REGISTRY")
    print("=" * 50)
    print(f"Version: {registry.get('version', 'unknown')}")
    print()
    print("Indicator Counts:")
    print(f"  Market:      {counts['market']:4d}")
    print(f"  Economic:    {counts['economic']:4d}")
    print(f"  Synthetic:   {counts['synthetic']:4d}")
    print(f"  Technical:   {counts['technical']:4d}")
    print("-" * 30)
    print(f"  TOTAL:       {counts['total']:4d}")
    print()
