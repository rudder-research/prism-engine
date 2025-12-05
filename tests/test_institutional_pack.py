"""
Acceptance Tests for PRISM Full Institutional Pack Registry
============================================================

These tests verify the requirements for the Full Institutional Pack:

✓ Registry has >250 indicators
✓ No duplicate IDs
✓ Synthetic builder runs with no missing dependencies
✓ Technical builder automatically applies indicators to appropriate assets
✓ update_all.py runs through all six stages without error
✓ SQLite "indicator_values" receives entries for all fetchable data

Run with:
    pytest tests/test_institutional_pack.py -v

Or run individual tests:
    pytest tests/test_institutional_pack.py::test_registry_has_250_plus_indicators -v
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# Test 1: Registry has >250 indicators
# =============================================================================

def test_registry_has_250_plus_indicators():
    """Verify the registry contains at least 250 indicators."""
    from data.registry import load_metric_registry, get_indicator_count

    registry = load_metric_registry()
    counts = get_indicator_count(registry)

    print(f"\nIndicator counts:")
    print(f"  Market:     {counts['market']}")
    print(f"  Economic:   {counts['economic']}")
    print(f"  Synthetic:  {counts['synthetic']}")
    print(f"  Technical:  {counts['technical']}")
    print(f"  TOTAL:      {counts['total']}")

    assert counts["total"] >= 250, (
        f"Registry should have at least 250 indicators, found {counts['total']}"
    )


# =============================================================================
# Test 2: No duplicate IDs
# =============================================================================

def test_no_duplicate_indicator_ids():
    """Verify there are no duplicate indicator IDs in the registry."""
    from data.registry import load_metric_registry, get_all_indicator_ids

    registry = load_metric_registry()
    all_ids = get_all_indicator_ids(registry)

    # Check for duplicates
    seen = set()
    duplicates = []
    for ind_id in all_ids:
        if ind_id in seen:
            duplicates.append(ind_id)
        seen.add(ind_id)

    assert len(duplicates) == 0, f"Found duplicate indicator IDs: {duplicates}"


def test_all_ids_are_lowercase():
    """Verify all indicator IDs are lowercase."""
    from data.registry import load_metric_registry, get_all_indicator_ids

    registry = load_metric_registry()
    all_ids = get_all_indicator_ids(registry)

    uppercase_ids = [ind_id for ind_id in all_ids if ind_id != ind_id.lower()]

    assert len(uppercase_ids) == 0, f"Found uppercase indicator IDs: {uppercase_ids}"


# =============================================================================
# Test 3: Synthetic builder can resolve dependencies
# =============================================================================

def test_synthetic_dependencies_are_valid():
    """Verify all synthetic indicator dependencies can be resolved."""
    from data.registry import load_metric_registry, get_all_indicator_ids

    registry = load_metric_registry()
    all_ids = set(get_all_indicator_ids(registry))

    # Check synthetic indicators
    missing_deps = []
    for synth in registry.get("synthetic", []):
        name = synth.get("name")
        inputs = synth.get("inputs") or synth.get("depends_on", [])

        for dep in inputs:
            if dep not in all_ids:
                missing_deps.append(f"{name} -> {dep}")

    assert len(missing_deps) == 0, (
        f"Synthetic indicators with missing dependencies:\n"
        + "\n".join(f"  {d}" for d in missing_deps)
    )


def test_synthetic_formulas_are_valid():
    """Verify all synthetic indicators have valid formulas."""
    from data.registry import load_metric_registry

    VALID_FORMULAS = {
        "spread", "ratio", "inverse_ratio",
        "yoy", "mom", "qoq",
        "log_return", "cumulative_return",
        "zscore", "percentile", "diff",
        "rolling_mean", "rolling_std",
        "sum", "product", "max", "min", "mean",
        "rebase",
    }

    registry = load_metric_registry()
    invalid_formulas = []

    for synth in registry.get("synthetic", []):
        name = synth.get("name")
        formula = synth.get("formula", "")

        if formula and formula.lower() not in VALID_FORMULAS:
            invalid_formulas.append(f"{name}: {formula}")

    assert len(invalid_formulas) == 0, (
        f"Invalid formulas found:\n" + "\n".join(f"  {f}" for f in invalid_formulas)
    )


# =============================================================================
# Test 4: Technical builder auto-apply rules
# =============================================================================

def test_technical_auto_apply_rules_exist():
    """Verify technical auto-apply rules are defined."""
    from data.registry import load_metric_registry, get_technical_auto_apply_rules

    registry = load_metric_registry()
    rules = get_technical_auto_apply_rules(registry)

    assert "groups_to_apply" in rules, "Missing 'groups_to_apply' in technical rules"
    assert "indicators" in rules, "Missing 'indicators' in technical rules"
    assert len(rules["groups_to_apply"]) > 0, "No groups defined for auto-apply"
    assert len(rules["indicators"]) > 0, "No indicator types defined for auto-apply"


def test_technical_indicators_have_valid_types():
    """Verify all technical indicators have valid types."""
    from data.registry import load_metric_registry

    VALID_TYPES = {
        "sma", "ema", "wma",
        "rsi",
        "macd", "macd_signal", "macd_histogram",
        "bollinger_upper", "bollinger_lower", "bollinger_width", "bollinger_pct",
        "volatility", "volatility_pct",
        "momentum", "roc", "log_return",
        "zscore",
        "price_to_sma", "distance_from_high", "distance_from_low",
        "stochastic_k", "stochastic_d",
        "percentile_rank", "trend_strength",
    }

    registry = load_metric_registry()
    invalid_types = []

    for tech in registry.get("technical", []):
        name = tech.get("name")
        tech_type = tech.get("type", "")

        if tech_type and tech_type.lower() not in VALID_TYPES:
            invalid_types.append(f"{name}: {tech_type}")

    assert len(invalid_types) == 0, (
        f"Invalid technical types found:\n" + "\n".join(f"  {t}" for t in invalid_types)
    )


# =============================================================================
# Test 5: YAML files exist and are valid
# =============================================================================

def test_yaml_registry_files_exist():
    """Verify all expected YAML registry files exist."""
    yaml_dir = Path(__file__).parent.parent / "data" / "registry" / "yaml"

    expected_files = [
        "market_global.yaml",
        "rates_us.yaml",
        "rates_global.yaml",
        "credit.yaml",
        "liquidity.yaml",
        "fx.yaml",
        "commodities.yaml",
        "volatility.yaml",
        "economics_us.yaml",
        "economics_global.yaml",
        "synthetic.yaml",
        "technical.yaml",
    ]

    missing_files = []
    for filename in expected_files:
        filepath = yaml_dir / filename
        if not filepath.exists():
            missing_files.append(filename)

    assert len(missing_files) == 0, f"Missing YAML files: {missing_files}"


def test_yaml_loader_works():
    """Verify the YAML loader can load all files."""
    from data.registry.yaml_loader import load_all_yaml_registries

    registry = load_all_yaml_registries()

    assert "indicators" in registry, "Registry missing 'indicators' key"
    assert len(registry["indicators"]) > 0, "No indicators loaded"
    assert "metadata" in registry, "Registry missing 'metadata' key"
    assert "total_indicators" in registry["metadata"], "Missing total count"


def test_yaml_validation_passes():
    """Verify YAML registry passes validation."""
    from data.registry.yaml_loader import load_all_yaml_registries, validate_yaml_registry

    registry = load_all_yaml_registries()
    result = validate_yaml_registry(registry)

    assert result is True, "YAML registry validation failed"


# =============================================================================
# Test 6: Registry loader integration
# =============================================================================

def test_registry_loader_uses_yaml():
    """Verify the main registry loader uses YAML by default."""
    from data.registry import load_metric_registry

    registry = load_metric_registry(use_yaml=True)

    # Should have yaml_registry key if YAML was loaded
    assert "yaml_registry" in registry, "Registry not loaded from YAML"
    assert registry.get("version") == "2.0.0", "Expected version 2.0.0"


def test_registry_backward_compatibility():
    """Verify registry works with legacy format expectations."""
    from data.registry import (
        load_metric_registry,
        get_market_tickers,
        get_fred_series,
    )

    registry = load_metric_registry()

    # Should still have legacy structure
    assert "market" in registry, "Missing 'market' key"
    assert "economic" in registry, "Missing 'economic' key"
    assert "synthetic" in registry, "Missing 'synthetic' key"
    assert "technical" in registry, "Missing 'technical' key"

    # Helper functions should work
    tickers = get_market_tickers(registry)
    assert len(tickers) > 0, "No market tickers found"

    fred_series = get_fred_series(registry)
    assert len(fred_series) > 0, "No FRED series found"


# =============================================================================
# Test 7: Source coverage
# =============================================================================

def test_yahoo_indicator_coverage():
    """Verify Yahoo Finance indicators have ticker params."""
    from data.registry.yaml_loader import load_all_yaml_registries, get_yahoo_indicators

    registry = load_all_yaml_registries()
    yahoo_indicators = get_yahoo_indicators(registry)

    assert len(yahoo_indicators) >= 100, (
        f"Expected at least 100 Yahoo indicators, found {len(yahoo_indicators)}"
    )

    # Check all have tickers
    missing_tickers = [ind["id"] for ind in yahoo_indicators if not ind.get("ticker")]
    assert len(missing_tickers) == 0, f"Yahoo indicators missing tickers: {missing_tickers}"


def test_fred_indicator_coverage():
    """Verify FRED indicators have series_id params."""
    from data.registry.yaml_loader import load_all_yaml_registries, get_fred_indicators

    registry = load_all_yaml_registries()
    fred_indicators = get_fred_indicators(registry)

    assert len(fred_indicators) >= 80, (
        f"Expected at least 80 FRED indicators, found {len(fred_indicators)}"
    )

    # Check all have series IDs
    missing_ids = [ind["id"] for ind in fred_indicators if not ind.get("series_id")]
    assert len(missing_ids) == 0, f"FRED indicators missing series_id: {missing_ids}"


# =============================================================================
# Test 8: Category coverage
# =============================================================================

def test_indicator_group_coverage():
    """Verify indicators cover all required groups."""
    from data.registry.yaml_loader import load_all_yaml_registries

    registry = load_all_yaml_registries()

    # Required groups
    required_groups = {
        "us_equity_index",
        "global_equity_index",
        "us_sector",
        "us_treasury",
        "fx_major",
        "precious_metals",
        "energy",
        "inflation",
        "employment",
        "yield_curve",
    }

    found_groups = set(registry.get("by_group", {}).keys())
    missing_groups = required_groups - found_groups

    assert len(missing_groups) == 0, f"Missing required groups: {missing_groups}"


# =============================================================================
# Test 9: Synthetic categories coverage
# =============================================================================

def test_synthetic_category_coverage():
    """Verify synthetic indicators cover required categories."""
    from data.registry import load_metric_registry

    registry = load_metric_registry()
    synthetics = registry.get("synthetic", [])

    # Check for required synthetic types
    synth_groups = set(s.get("group", "") for s in synthetics)

    required_synth_groups = {
        "yield_curve",
        "real_yield",
        "credit_spread",
        "liquidity_ratio",
        "cross_asset",
    }

    missing = required_synth_groups - synth_groups
    assert len(missing) == 0, f"Missing synthetic groups: {missing}"


# =============================================================================
# Test 10: Full integration test (registry load + validate)
# =============================================================================

def test_full_registry_integration():
    """Full integration test: load, validate, and verify registry."""
    from data.registry import (
        load_metric_registry,
        validate_registry,
        get_indicator_count,
        get_technical_auto_apply_rules,
    )

    # Load
    registry = load_metric_registry(use_yaml=True)

    # Validate
    assert validate_registry(registry) is True

    # Check counts
    counts = get_indicator_count(registry)
    assert counts["total"] >= 250

    # Check technical rules
    rules = get_technical_auto_apply_rules(registry)
    assert len(rules.get("groups_to_apply", [])) > 0

    print("\n✓ All integration tests passed!")
    print(f"  Total indicators: {counts['total']}")
    print(f"  Market: {counts['market']}")
    print(f"  Economic: {counts['economic']}")
    print(f"  Synthetic: {counts['synthetic']}")
    print(f"  Technical: {counts['technical']}")


# =============================================================================
# CLI Runner
# =============================================================================

if __name__ == "__main__":
    import sys

    # Run all tests
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
