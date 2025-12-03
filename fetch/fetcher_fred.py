"""
FREDFetcher - Fetch economic time series from FRED

This file replaces all previous versions and removes:
✓ circular imports
✓ self-imports
✓ broken class names
✓ duplicate loaders

It integrates cleanly with FetcherBase and the new registry format.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests

from fetch.fetcher_base import FetcherBase

logger = logging.getLogger(__name__)


class FREDFetcher(FetcherBase):
    """
    Fetch economic data from FRED (Federal Reserve Bank of St. Louis).

    registry entry example:
    {
        "key": "dgs10",
        "ticker": "DGS10",
        "source": "fred",
        "enabled": true,
        "name": "10-Year Treasury Constant Maturity Rate",
        "frequency": "daily"
    }
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FRED_API_KEY")

        if not self.api_key:
            logger.warning("No FRED API key found. FRED requests may fail.")

    # ----------------------------------------------------------------------
    # Core fetch function
    # ----------------------------------------------------------------------
    def fetch(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch a FRED series defined in the registry.
        """

        series_id = config.get("ticker")
        if not series_id:
            raise ValueError("Economic registry entry missing 'ticker'")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": "1900-01-01",
        }

        logger.info(f"Fetching FRED series: {series_id}")

        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch {series_id} from FRED: "
                f"{response.status_code} {response.text}"
            )

        data = response.json().get("observations", [])
        if not data:
            raise RuntimeError(f"FRED returned no data for {series_id}")

        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "timestamp", "value": series_id})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

        df = df.dropna(subset=["timestamp"])

        df = df.set_index("timestamp").sort_index()

        logger.info(f"Fetched {len(df)} rows for {series_id}")

        return df

    # ----------------------------------------------------------------------
    # Helper to save raw data
    # ----------------------------------------------------------------------
    def save_raw(self, key: str, df: pd.DataFrame, folder: Path):
        """
        Save raw CSV files into data/raw
        """
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{key}.csv"
        df.to_csv(out_path)
        logger.info(f"Saved raw FRED CSV: {out_path}")
