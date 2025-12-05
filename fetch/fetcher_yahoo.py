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


    def fetch_all(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        write_to_db: bool = False,
        system: str = "finance",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch all tickers and optionally write to database.

        This method fetches multiple tickers, combines them into a single
        DataFrame, and optionally persists the data to the PRISM database.

        Args:
            tickers: List of ticker symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            write_to_db: If True, write results to database
            system: System type for database storage (default: 'finance')

        Returns:
            DataFrame with date and all ticker close prices

        Example:
            fetcher = YahooFetcher()
            df = fetcher.fetch_all(
                ['SPY', 'QQQ', 'IWM'],
                start_date='2020-01-01',
                write_to_db=True
            )
        """
        # Fetch all tickers
        df = self.fetch_multiple(tickers, start_date, end_date, **kwargs)

        if df.empty:
            logger.warning("No data fetched for any ticker")
            return df

        # Write to database if requested
        if write_to_db:
            try:
                from data.sql.db import write_dataframe, log_fetch
                from datetime import datetime

                started_at = datetime.now().isoformat()

                for ticker in tickers:
                    ticker_lower = ticker.lower()
                    if ticker_lower not in df.columns:
                        logger.warning(f"Ticker {ticker} not found in results")
                        continue

                    # Create single-column dataframe for this ticker
                    ticker_df = df[["date", ticker_lower]].copy()
                    ticker_df = ticker_df.rename(columns={ticker_lower: "value"})
                    ticker_df = ticker_df.dropna()

                    if ticker_df.empty:
                        continue

                    try:
                        rows = write_dataframe(
                            ticker_df,
                            indicator_name=ticker,
                            system=system,
                            source="Yahoo Finance",
                            frequency="daily",
                        )

                        log_fetch(
                            source="yahoo",
                            entity=ticker,
                            operation="fetch",
                            status="success",
                            rows_fetched=len(ticker_df),
                            rows_inserted=rows,
                            started_at=started_at,
                        )

                        logger.info(f"Wrote {rows} rows for {ticker}")

                    except Exception as e:
                        log_fetch(
                            source="yahoo",
                            entity=ticker,
                            operation="fetch",
                            status="error",
                            error_message=str(e),
                            started_at=started_at,
                        )
                        logger.error(f"Failed to write {ticker}: {e}")

            except ImportError:
                logger.error(
                    "Database modules not available. "
                    "Install with: pip install -e .[db]"
                )
                raise

        return df


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
