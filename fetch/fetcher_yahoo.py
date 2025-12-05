"""
YahooFetcher - Fetch market data from Yahoo Finance
===================================================

This module provides a production-ready fetcher for Yahoo Finance data.
It integrates with the unified database schema via data.sql.db.

Usage:
    from fetch.fetcher_yahoo import YahooFetcher

    fetcher = YahooFetcher()

    # Fetch single ticker
    df = fetcher.fetch_single("SPY")

    # Fetch all enabled tickers from registry and write to database
    results = fetcher.fetch_all()

    # Test connection
    if fetcher.test_connection():
        print("Yahoo Finance is accessible")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from .fetcher_base import BaseFetcher


logger = logging.getLogger(__name__)

# Project root for finding registries
PROJECT_ROOT = Path(__file__).parent.parent


class YahooFetcher(BaseFetcher):
    """
    Production-ready Yahoo Finance Fetcher.

    Handles:
    - MultiIndex columns from yfinance
    - Missing 'Close' columns (VIX, DXY, some indices)
    - Always returns: date, value, adjusted_value

    Registry entry example:
    {
        "key": "spy",
        "ticker": "SPY",
        "source": "yahoo",
        "enabled": true,
        "name": "S&P 500 ETF",
        "frequency": "daily"
    }
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Yahoo fetcher.

        Args:
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(checkpoint_dir)

    def validate_response(self, df: Any) -> bool:
        """Validate Yahoo Finance response."""
        return df is not None and not df.empty

    def _flatten(self, cols) -> List[str]:
        """Flatten MultiIndex columns to simple lowercase strings."""
        out = []
        for c in cols:
            if isinstance(c, tuple):
                flat = "_".join([str(x) for x in c if x])
            else:
                flat = str(c)
            out.append(flat.lower())
        return out

    def _detect_close_column(self, df: pd.DataFrame) -> str:
        """
        Try to find the best price column.

        Yahoo can return:
          - 'close'
          - 'close_*'
          - 'adjclose'
          - 'adj_close'
          - 'vixclose' (weird VIX cases)
        """
        candidates = [
            "close",
            "adjclose",
            "adj_close",
            "close_price",
            "last",
        ]

        # direct match
        for c in candidates:
            if c in df.columns:
                return c

        # anything containing 'close'
        for col in df.columns:
            if "close" in col:
                return col

        # last resort: first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"[YahooFetcher] Using fallback column: {col}")
                return col

        raise ValueError("[YahooFetcher] Could not detect close column")

    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single Yahoo Finance ticker.

        Args:
            ticker: Yahoo ticker symbol (e.g., 'SPY', 'AAPL', '^VIX')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date, value, value_2, adjusted_value columns
        """
        import yfinance as yf

        logger.info(f"[YahooFetcher] Requesting {ticker} ...")

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            if not self.validate_response(df):
                logger.error(f"[YahooFetcher] Invalid response for {ticker}")
                return None

            df = df.reset_index()
            df.columns = self._flatten(df.columns)

            if "date" not in df.columns:
                raise ValueError(f"[YahooFetcher] No 'date' column for {ticker}")

            # pick the best available 'close-like' column
            close_col = self._detect_close_column(df)

            # pick adjusted close if present
            adj_col = "adjclose" if "adjclose" in df.columns else close_col

            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["date"])
            out["value"] = df[close_col]
            out["value_2"] = None
            out["adjusted_value"] = df[adj_col]

            logger.info(f"[YahooFetcher] {ticker} -> {len(out)} rows")
            return out

        except Exception as e:
            logger.error(f"[YahooFetcher] Error fetching {ticker}: {e}")
            return None

    def fetch_single_close_only(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch only closing prices (lighter weight).

        Args:
            ticker: Yahoo ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value only
        """
        import yfinance as yf

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                return None

            df = df.reset_index()
            df.columns = self._flatten(df.columns)

            if "date" not in df.columns:
                return None

            close_col = self._detect_close_column(df)

            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df["date"])
            out["value"] = df[close_col]

            return out

        except Exception as e:
            logger.error(f"[YahooFetcher] Error for {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_single(ticker, start_date, end_date)
            if df is not None and not df.empty:
                results[ticker] = df
        return results

    def load_market_registry(self) -> List[Dict[str, Any]]:
        """
        Load the market registry from data/registry/.

        Returns:
            List of instrument configurations
        """
        registry_path = PROJECT_ROOT / "data" / "registry" / "market_registry.json"

        if not registry_path.exists():
            logger.error(f"Market registry not found: {registry_path}")
            return []

        with open(registry_path, "r") as f:
            registry = json.load(f)

        return registry.get("instruments", [])

    def fetch_all(
        self,
        write_to_db: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all enabled instruments from the market registry.

        Args:
            write_to_db: If True, write results to database
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping indicator keys to DataFrames
        """
        instruments = self.load_market_registry()
        results = {}

        # Import database module only if needed
        db = None
        if write_to_db:
            try:
                from data.sql.db import add_indicator, write_dataframe, log_fetch, init_database
                init_database()  # Ensure tables exist
                db = True
            except ImportError as e:
                logger.warning(f"Database module not available: {e}")
                db = None

        for instrument in instruments:
            if not instrument.get("enabled", True):
                logger.debug(f"Skipping disabled instrument: {instrument.get('key')}")
                continue

            # Only process yahoo-sourced instruments
            if instrument.get("source", "yahoo") != "yahoo":
                continue

            key = instrument.get("key")
            ticker = instrument.get("ticker", key.upper())
            frequency = instrument.get("frequency", "daily")
            source = "yahoo"

            logger.info(f"Fetching market instrument: {key} ({ticker})")

            try:
                df = self.fetch_single(ticker, start_date, end_date)

                if df is not None and not df.empty:
                    results[key] = df

                    if db:
                        # Register indicator
                        add_indicator(
                            name=key,
                            system="market",
                            frequency=frequency,
                            source=source,
                            description=instrument.get("name")
                        )

                        # Write data
                        rows = write_dataframe(df, indicator=key, system="market")

                        # Log fetch
                        log_fetch(
                            indicator=key,
                            system="market",
                            source=source,
                            rows_fetched=rows,
                            status="success"
                        )

                        logger.info(f"  -> Wrote {rows} rows to database for {key}")
                else:
                    logger.warning(f"  -> No data returned for {key}")

                    if db:
                        log_fetch(
                            indicator=key,
                            system="market",
                            source=source,
                            rows_fetched=0,
                            status="error",
                            error_message="No data returned"
                        )

            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")

                if db:
                    log_fetch(
                        indicator=key,
                        system="market",
                        source=source,
                        rows_fetched=0,
                        status="error",
                        error_message=str(e)
                    )

        logger.info(f"Completed: {len(results)} market instruments fetched")
        return results

    def test_connection(self) -> bool:
        """
        Test the Yahoo Finance connection.

        Returns:
            True if connection successful
        """
        try:
            df = self.fetch_single("SPY")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Common Yahoo Finance tickers for financial analysis
COMMON_YAHOO_TICKERS = {
    # Major Indices
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX Volatility",

    # Sector ETFs
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",

    # Commodities
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
    "SI=F": "Silver Futures",

    # Currencies
    "DX-Y.NYB": "US Dollar Index",
    "EURUSD=X": "EUR/USD",
    "JPYUSD=X": "JPY/USD",

    # Bonds
    "TLT": "20+ Year Treasury ETF",
    "IEF": "7-10 Year Treasury ETF",
    "HYG": "High Yield Bond ETF",
}


# Backward compatibility aliases
FetcherYahoo = YahooFetcher
