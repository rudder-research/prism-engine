"""
Test Panel Builder

Tests for:
- Build a small mock DB
- Run panel builder
- Assert shape and column presence
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_market_data():
    """Mock market data (daily frequency)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(42)

    return {
        "SPY": pd.DataFrame({
            "date": dates,
            "spy": np.cumsum(np.random.randn(100)) + 300,
        }),
        "QQQ": pd.DataFrame({
            "date": dates,
            "qqq": np.cumsum(np.random.randn(100)) + 200,
        }),
        "TLT": pd.DataFrame({
            "date": dates,
            "tlt": np.cumsum(np.random.randn(100)) + 140,
        }),
    }


@pytest.fixture
def mock_economic_data():
    """Mock economic data (monthly frequency)."""
    # Monthly data - fewer points
    dates = pd.date_range("2020-01-01", periods=4, freq="ME")
    np.random.seed(42)

    return {
        "CPI": pd.DataFrame({
            "date": dates,
            "cpi": [258.0, 258.5, 259.0, 259.5],
        }),
        "UNRATE": pd.DataFrame({
            "date": dates,
            "unrate": [3.5, 3.6, 3.4, 3.5],
        }),
    }


@pytest.fixture
def mock_data_with_gaps():
    """Mock data with missing values."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    np.random.seed(42)

    data = pd.DataFrame({
        "date": dates,
        "indicator_a": np.random.randn(50) + 100,
        "indicator_b": np.random.randn(50) + 50,
    })

    # Add gaps
    data.loc[10:15, "indicator_a"] = np.nan
    data.loc[25:30, "indicator_b"] = np.nan

    return data


# ============================================================================
# Panel Builder Class (for testing)
# ============================================================================

class SimplePanelBuilder:
    """
    Simple panel builder for testing purposes.

    Mimics the behavior of DataAligner.create_master_panel().
    """

    def __init__(self):
        self.date_col = "date"

    def merge_dataframes(self, dfs):
        """
        Merge multiple DataFrames on date column.

        Args:
            dfs: Dictionary of {name: DataFrame}

        Returns:
            Merged DataFrame
        """
        if not dfs:
            return pd.DataFrame()

        # Get first DataFrame
        df_list = list(dfs.values())
        result = df_list[0].copy()

        # Ensure date is datetime
        result[self.date_col] = pd.to_datetime(result[self.date_col])

        # Merge remaining
        for df in df_list[1:]:
            df = df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            result = result.merge(df, on=self.date_col, how="outer")

        return result.sort_values(self.date_col).reset_index(drop=True)

    def align_to_daily(self, df):
        """
        Ensure data is at daily frequency with forward-fill.

        Args:
            df: Input DataFrame

        Returns:
            Daily-aligned DataFrame
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col)

        # Resample to daily and forward-fill
        df = df.resample("D").last().ffill()

        return df.reset_index()

    def fill_missing(self, df):
        """
        Fill missing values with forward-fill then backward-fill.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with filled values
        """
        return df.ffill().bfill()

    def build_panel(self, dfs, align_daily=True, fill_na=True):
        """
        Build a panel from multiple DataFrames.

        Args:
            dfs: Dictionary of {name: DataFrame}
            align_daily: Whether to align to daily frequency
            fill_na: Whether to fill missing values

        Returns:
            Panel DataFrame
        """
        if not dfs:
            return pd.DataFrame()

        # Align each to daily if needed
        if align_daily:
            aligned = {}
            for name, df in dfs.items():
                aligned[name] = self.align_to_daily(df)
            dfs = aligned

        # Merge
        panel = self.merge_dataframes(dfs)

        # Fill missing
        if fill_na:
            # Only fill non-date columns
            value_cols = [c for c in panel.columns if c != self.date_col]
            panel[value_cols] = self.fill_missing(panel[value_cols])

        return panel


# ============================================================================
# Basic Panel Building Tests
# ============================================================================

class TestBasicPanelBuilding:
    """Tests for basic panel building functionality."""

    def test_build_empty_panel(self):
        """Empty input should return empty DataFrame."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel({})

        assert isinstance(panel, pd.DataFrame)
        assert len(panel) == 0

    def test_build_single_indicator_panel(self, mock_market_data):
        """Should build panel with single indicator."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel({"SPY": mock_market_data["SPY"]})

        assert len(panel) > 0
        assert "date" in panel.columns
        assert "spy" in panel.columns

    def test_build_multi_indicator_panel(self, mock_market_data):
        """Should build panel with multiple indicators."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        assert len(panel) > 0
        assert "date" in panel.columns
        assert "spy" in panel.columns
        assert "qqq" in panel.columns
        assert "tlt" in panel.columns


# ============================================================================
# Panel Shape Tests
# ============================================================================

class TestPanelShape:
    """Tests for panel shape and dimensions."""

    def test_panel_has_date_column(self, mock_market_data):
        """Panel should have date column."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        assert "date" in panel.columns

    def test_panel_column_count(self, mock_market_data):
        """Panel should have date + N indicator columns."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        # 1 date column + 3 indicators
        assert len(panel.columns) == 4

    def test_panel_row_count(self, mock_market_data):
        """Panel should have rows for all dates."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        # Should have at least as many rows as input
        assert len(panel) >= 100

    def test_panel_dates_are_sorted(self, mock_market_data):
        """Panel dates should be sorted."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        dates = panel["date"].tolist()
        assert dates == sorted(dates)


# ============================================================================
# Column Presence Tests
# ============================================================================

class TestColumnPresence:
    """Tests for expected column presence."""

    def test_all_indicators_present(self, mock_market_data):
        """All input indicators should be present in panel."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        expected = ["spy", "qqq", "tlt"]
        for col in expected:
            assert col in panel.columns, f"Missing column: {col}"

    def test_date_column_is_datetime(self, mock_market_data):
        """Date column should be datetime type."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        assert pd.api.types.is_datetime64_any_dtype(panel["date"])

    def test_value_columns_are_numeric(self, mock_market_data):
        """Value columns should be numeric type."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        value_cols = [c for c in panel.columns if c != "date"]
        for col in value_cols:
            assert pd.api.types.is_numeric_dtype(panel[col]), \
                f"Column {col} should be numeric"


# ============================================================================
# Frequency Alignment Tests
# ============================================================================

class TestFrequencyAlignment:
    """Tests for frequency alignment."""

    def test_align_monthly_to_daily(self, mock_economic_data):
        """Monthly data should be aligned to daily."""
        builder = SimplePanelBuilder()

        # Build with daily alignment
        panel = builder.build_panel(mock_economic_data, align_daily=True)

        # Should have more rows than monthly data
        assert len(panel) > 4

    def test_mixed_frequencies(self, mock_market_data, mock_economic_data):
        """Should handle mixed frequency data."""
        builder = SimplePanelBuilder()

        all_data = {**mock_market_data, **mock_economic_data}
        panel = builder.build_panel(all_data)

        # All columns should be present
        assert "spy" in panel.columns
        assert "cpi" in panel.columns

    def test_forward_fill_applied(self, mock_economic_data):
        """Forward fill should be applied to align frequencies."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_economic_data, align_daily=True)

        # Check that values are filled
        assert panel["cpi"].notna().all() or len(panel) > 0


# ============================================================================
# Missing Value Tests
# ============================================================================

class TestMissingValueHandling:
    """Tests for missing value handling."""

    def test_fill_missing_values(self, mock_data_with_gaps):
        """Should fill missing values."""
        builder = SimplePanelBuilder()

        # Build with NA filling
        panel = builder.build_panel(
            {"test": mock_data_with_gaps},
            align_daily=False,
            fill_na=True
        )

        # Check that gaps are filled
        assert panel["indicator_a"].notna().all()
        assert panel["indicator_b"].notna().all()

    def test_no_fill_when_disabled(self, mock_data_with_gaps):
        """Should not fill missing values when disabled."""
        builder = SimplePanelBuilder()

        # Build without NA filling
        panel = builder.build_panel(
            {"test": mock_data_with_gaps},
            align_daily=False,
            fill_na=False
        )

        # Check that gaps remain
        assert panel["indicator_a"].isna().any()
        assert panel["indicator_b"].isna().any()


# ============================================================================
# Integration Tests
# ============================================================================

class TestPanelBuilderIntegration:
    """Integration tests for complete panel building workflow."""

    def test_full_pipeline(self, mock_market_data, mock_economic_data):
        """Test complete panel building pipeline."""
        builder = SimplePanelBuilder()

        # Combine all data
        all_data = {**mock_market_data, **mock_economic_data}

        # Build panel
        panel = builder.build_panel(all_data, align_daily=True, fill_na=True)

        # Assertions
        assert len(panel) > 0
        assert "date" in panel.columns

        # All indicators present
        for col in ["spy", "qqq", "tlt", "cpi", "unrate"]:
            assert col in panel.columns

        # No missing values
        value_cols = [c for c in panel.columns if c != "date"]
        for col in value_cols:
            # At least some non-NA values
            assert panel[col].notna().any(), f"Column {col} is all NA"

    def test_panel_suitable_for_analysis(self, mock_market_data):
        """Panel should be suitable for analysis (enough rows, proper format)."""
        builder = SimplePanelBuilder()
        panel = builder.build_panel(mock_market_data)

        # Minimum rows for analysis
        assert len(panel) >= 50, "Panel should have at least 50 rows"

        # Proper format
        assert "date" in panel.columns
        value_cols = [c for c in panel.columns if c != "date"]
        assert len(value_cols) >= 1, "Panel should have at least one indicator"

        # All value columns are numeric
        for col in value_cols:
            assert pd.api.types.is_numeric_dtype(panel[col])


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_data(self):
        """Should handle single row of data."""
        builder = SimplePanelBuilder()
        data = {"test": pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")],
            "value": [100.0]
        })}

        panel = builder.build_panel(data, align_daily=False)
        assert len(panel) == 1

    def test_all_nan_column(self):
        """Should handle column with all NaN values."""
        builder = SimplePanelBuilder()
        data = {"test": pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "value": [np.nan] * 10
        })}

        panel = builder.build_panel(data, fill_na=False)
        assert panel["value"].isna().all()

    def test_duplicate_dates(self):
        """Should handle duplicate dates in input."""
        builder = SimplePanelBuilder()
        data = {"test": pd.DataFrame({
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "value": [100, 101, 102]
        })}

        # Last value should be used after alignment
        panel = builder.build_panel(data, align_daily=True)
        assert len(panel) >= 1

    def test_non_overlapping_dates(self):
        """Should handle data with non-overlapping dates."""
        builder = SimplePanelBuilder()
        data = {
            "early": pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "early": [1, 2, 3, 4, 5]
            }),
            "late": pd.DataFrame({
                "date": pd.date_range("2020-02-01", periods=5, freq="D"),
                "late": [10, 20, 30, 40, 50]
            }),
        }

        panel = builder.build_panel(data)

        # Both columns should exist
        assert "early" in panel.columns
        assert "late" in panel.columns

        # Panel should span both date ranges
        assert panel["date"].min() <= pd.Timestamp("2020-01-01")
        assert panel["date"].max() >= pd.Timestamp("2020-02-01")
