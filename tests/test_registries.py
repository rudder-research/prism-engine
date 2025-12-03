"""
Test Registry Loading and Validation

Tests for:
- Registry JSON can be loaded
- Required keys are present in each entry
- Invalid registry entries raise clear errors
"""

import json
import pytest
from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def registry_path():
    """Path to the metric registry JSON file."""
    return Path(__file__).parent.parent / "data" / "registry" / "metric_registry.json"


@pytest.fixture
def valid_registry():
    """Example of a valid registry for testing."""
    return [
        {"key": "test1", "source": "fred", "ticker": "TEST1"},
        {"key": "test2", "source": "yahoo", "ticker": "TEST2"},
    ]


@pytest.fixture
def registry_data(registry_path):
    """Load the actual registry data."""
    if not registry_path.exists():
        pytest.skip("Registry file not found")
    with open(registry_path) as f:
        return json.load(f)


# ============================================================================
# Registry Loading Tests
# ============================================================================

class TestRegistryLoading:
    """Tests for loading registry files."""

    def test_registry_file_exists(self, registry_path):
        """Registry file should exist at expected path."""
        assert registry_path.exists(), f"Registry file not found at {registry_path}"

    def test_registry_is_valid_json(self, registry_path):
        """Registry file should be valid JSON."""
        with open(registry_path) as f:
            data = json.load(f)
        assert isinstance(data, list), "Registry should be a JSON array"

    def test_registry_is_not_empty(self, registry_data):
        """Registry should contain at least one entry."""
        assert len(registry_data) > 0, "Registry should not be empty"

    def test_registry_entries_are_dicts(self, registry_data):
        """Each registry entry should be a dictionary."""
        for i, entry in enumerate(registry_data):
            assert isinstance(entry, dict), f"Entry {i} should be a dictionary"


# ============================================================================
# Registry Schema Tests
# ============================================================================

class TestRegistrySchema:
    """Tests for registry entry schema validation."""

    REQUIRED_KEYS = ["key", "source", "ticker"]
    VALID_SOURCES = ["fred", "yahoo", "climate", "custom"]

    def test_entries_have_required_keys(self, registry_data):
        """Each entry must have all required keys."""
        for i, entry in enumerate(registry_data):
            for key in self.REQUIRED_KEYS:
                assert key in entry, f"Entry {i} missing required key '{key}'"

    def test_entry_keys_are_strings(self, registry_data):
        """All entry values should be strings."""
        for i, entry in enumerate(registry_data):
            for key in self.REQUIRED_KEYS:
                assert isinstance(entry[key], str), \
                    f"Entry {i}: '{key}' should be a string, got {type(entry[key])}"

    def test_entry_keys_not_empty(self, registry_data):
        """Entry values should not be empty strings."""
        for i, entry in enumerate(registry_data):
            for key in self.REQUIRED_KEYS:
                assert entry[key].strip(), \
                    f"Entry {i}: '{key}' should not be empty"

    def test_source_is_valid(self, registry_data):
        """Source should be one of the valid source types."""
        for i, entry in enumerate(registry_data):
            assert entry["source"] in self.VALID_SOURCES, \
                f"Entry {i}: invalid source '{entry['source']}'"

    def test_keys_are_unique(self, registry_data):
        """All keys should be unique across the registry."""
        keys = [entry["key"] for entry in registry_data]
        duplicates = [k for k in keys if keys.count(k) > 1]
        assert len(duplicates) == 0, f"Duplicate keys found: {set(duplicates)}"


# ============================================================================
# Registry Validation Function Tests
# ============================================================================

class TestRegistryValidation:
    """Tests for registry validation logic."""

    def validate_registry(self, registry):
        """
        Validate registry entries.

        Args:
            registry: List of registry entries

        Raises:
            ValueError: If validation fails
        """
        required_fields = ["key", "source", "ticker"]
        valid_sources = ["fred", "yahoo", "climate", "custom"]
        seen_keys = set()

        for i, entry in enumerate(registry):
            # Check required fields
            for field in required_fields:
                if field not in entry:
                    raise ValueError(f"Entry {i}: missing required field '{field}'")
                if not isinstance(entry[field], str):
                    raise ValueError(f"Entry {i}: field '{field}' must be a string")
                if not entry[field].strip():
                    raise ValueError(f"Entry {i}: field '{field}' cannot be empty")

            # Check source
            if entry["source"] not in valid_sources:
                raise ValueError(f"Entry {i}: invalid source '{entry['source']}'")

            # Check uniqueness
            if entry["key"] in seen_keys:
                raise ValueError(f"Entry {i}: duplicate key '{entry['key']}'")
            seen_keys.add(entry["key"])

        return True

    def test_valid_registry_passes(self, valid_registry):
        """Valid registry should pass validation."""
        assert self.validate_registry(valid_registry) is True

    def test_missing_key_raises_error(self):
        """Missing required field should raise ValueError."""
        invalid_registry = [
            {"source": "fred", "ticker": "TEST"}  # missing 'key'
        ]
        with pytest.raises(ValueError, match="missing required field 'key'"):
            self.validate_registry(invalid_registry)

    def test_missing_source_raises_error(self):
        """Missing source field should raise ValueError."""
        invalid_registry = [
            {"key": "test", "ticker": "TEST"}  # missing 'source'
        ]
        with pytest.raises(ValueError, match="missing required field 'source'"):
            self.validate_registry(invalid_registry)

    def test_missing_ticker_raises_error(self):
        """Missing ticker field should raise ValueError."""
        invalid_registry = [
            {"key": "test", "source": "fred"}  # missing 'ticker'
        ]
        with pytest.raises(ValueError, match="missing required field 'ticker'"):
            self.validate_registry(invalid_registry)

    def test_invalid_source_raises_error(self):
        """Invalid source should raise ValueError."""
        invalid_registry = [
            {"key": "test", "source": "invalid_source", "ticker": "TEST"}
        ]
        with pytest.raises(ValueError, match="invalid source 'invalid_source'"):
            self.validate_registry(invalid_registry)

    def test_duplicate_key_raises_error(self):
        """Duplicate keys should raise ValueError."""
        invalid_registry = [
            {"key": "test", "source": "fred", "ticker": "TEST1"},
            {"key": "test", "source": "yahoo", "ticker": "TEST2"},  # duplicate key
        ]
        with pytest.raises(ValueError, match="duplicate key 'test'"):
            self.validate_registry(invalid_registry)

    def test_empty_key_raises_error(self):
        """Empty key should raise ValueError."""
        invalid_registry = [
            {"key": "", "source": "fred", "ticker": "TEST"}
        ]
        with pytest.raises(ValueError, match="cannot be empty"):
            self.validate_registry(invalid_registry)

    def test_non_string_value_raises_error(self):
        """Non-string values should raise ValueError."""
        invalid_registry = [
            {"key": 123, "source": "fred", "ticker": "TEST"}  # key is int
        ]
        with pytest.raises(ValueError, match="must be a string"):
            self.validate_registry(invalid_registry)


# ============================================================================
# Registry Content Tests
# ============================================================================

class TestRegistryContent:
    """Tests for expected registry content."""

    def test_has_fred_indicators(self, registry_data):
        """Registry should contain FRED indicators."""
        fred_entries = [e for e in registry_data if e["source"] == "fred"]
        assert len(fred_entries) > 0, "Registry should have FRED indicators"

    def test_has_yahoo_indicators(self, registry_data):
        """Registry should contain Yahoo indicators."""
        yahoo_entries = [e for e in registry_data if e["source"] == "yahoo"]
        assert len(yahoo_entries) > 0, "Registry should have Yahoo indicators"

    def test_has_common_indicators(self, registry_data):
        """Registry should contain common financial indicators."""
        keys = {e["key"] for e in registry_data}

        # Check for some expected indicators
        expected = ["spy", "dgs10"]
        for indicator in expected:
            assert indicator in keys, f"Missing expected indicator: {indicator}"
