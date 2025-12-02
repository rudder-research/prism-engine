"""
PRISM Data Fetch Module
=======================

Registry-driven data fetching with normalization and validation.

This module provides robust data fetching capabilities for market and economic
data, with strict validation and normalization before database insertion.

Modules:
    - fetch_market_data: Market data fetching (equities, bonds, commodities)
    - fetch_economic_data: Economic data fetching (FRED series)

Usage:
    from data_fetch.fetch_market_data import fetch_all_market_data
    from data_fetch.fetch_economic_data import fetch_all_economic_data

    # Fetch market data for all enabled instruments
    market_results = fetch_all_market_data()

    # Fetch economic data for all enabled series
    economic_results = fetch_all_economic_data()
"""

from .fetch_market_data import (
    fetch_all_market_data,
    fetch_single_market_instrument,
    load_market_registry,
)

from .fetch_economic_data import (
    fetch_all_economic_data,
    fetch_single_economic_series,
    load_economic_registry,
)

__all__ = [
    "fetch_all_market_data",
    "fetch_single_market_instrument",
    "load_market_registry",
    "fetch_all_economic_data",
    "fetch_single_economic_series",
    "load_economic_registry",
]
