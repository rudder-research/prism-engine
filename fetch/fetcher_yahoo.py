"""
Yahoo Finance Fetcher - Stock, ETF, and market data
"""

from pathlib import Path
from typing import Optional, Any, List
import pandas as pd
import logging

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class YahooFetcher(BaseFetcher):
    """
    Fetcher for Yahoo Finance data.

    Supports stocks, ETFs, indices, currencies, and commodities.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Yahoo fetcher.

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(checkpoint_dir)

    def validate_response(self, response: Any) -> bool:
        """Validate Yahoo Finance response."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single Yahoo Finance ticker.

        Args:
            ticker: Yahoo ticker symbol (e.g., 'SPY', 'AAPL', '^VIX')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with OHLCV data
        """
        import yfinance as yf

        try:
            # Download data
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                logger.warning(f"No data returned for {ticker}")
                return None

            # Reset index to get date as column
            df = df.reset_index()

            # Rename columns
            column_map = {
                "Date": "date",
                "Open": f"{ticker.lower()}_open",
                "High": f"{ticker.lower()}_high",
                "Low": f"{ticker.lower()}_low",
                "Close": ticker.lower(),  # Main column is just the ticker
                "Volume": f"{ticker.lower()}_volume"
            }
            df = df.rename(columns=column_map)

            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
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
            DataFrame with date and close price only
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

            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", ticker.lower()]

            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers efficiently in one call.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and all ticker close prices
        """
        import yfinance as yf

        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                return pd.DataFrame()

            # Handle MultiIndex columns from multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                # Get just Close prices
                df = df["Close"]

            df = df.reset_index()
            df.columns = ["date"] + [t.lower() for t in tickers]

            return df

        except Exception as e:
            logger.error(f"Yahoo batch error: {e}")
            return pd.DataFrame()


def fetch_registry_market_data(
    registry: dict,
    conn,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch market data for all enabled instruments in the registry.

    This is the main entry point for registry-driven Yahoo Finance fetching,
    designed to be called from update_all.py or similar pipeline scripts.

    Args:
        registry: Market registry dictionary with instruments list
        conn: Database connection for storing results
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        Number of instruments successfully fetched
    """
    fetcher = YahooFetcher()
    success_count = 0

    instruments = registry.get("instruments", [])
    enabled_instruments = [i for i in instruments if i.get("enabled", True)]

    logger.info(f"Fetching {len(enabled_instruments)} enabled instruments from registry")

    for instrument in enabled_instruments:
        key = instrument.get("key", "")
        ticker = instrument.get("ticker", key.upper())

        try:
            logger.info(f"Fetching {key} ({ticker})...")
            df = fetcher.fetch_single(
                ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if df is not None and not df.empty:
                # Store in database if connection provided
                if conn is not None:
                    try:
                        # Prepare data for database storage
                        # Standardize column names for DB
                        db_df = df.copy()

                        # Ensure we have a date column
                        if "date" in db_df.columns:
                            db_df = db_df.set_index("date")

                        # Write to database using the key as table/identifier
                        table_name = f"market_{key}"
                        db_df.to_sql(
                            table_name,
                            conn,
                            if_exists="replace",
                            index=True,
                        )
                        logger.info(f"  -> Stored {len(df)} rows for {key}")
                    except Exception as db_err:
                        logger.warning(f"  -> DB write failed for {key}: {db_err}")

                success_count += 1
                logger.info(f"  -> Fetched {len(df)} rows for {key}")
            else:
                logger.warning(f"  -> No data returned for {key}")

        except Exception as e:
            logger.error(f"  -> Error fetching {key}: {e}")
            continue

    logger.info(f"Registry fetch complete: {success_count}/{len(enabled_instruments)} successful")
    return success_count


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
