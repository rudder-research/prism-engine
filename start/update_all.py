#!/usr/bin/env python3
"""
PRISM Data Update Script
========================

This script fetches all market and economic data from configured registries
and writes them to the PRISM database.

Usage:
    # Fetch all data (market + economic)
    python start/update_all.py

    # Fetch only market data
    python start/update_all.py --market

    # Fetch only economic data
    python start/update_all.py --economic

    # Show status/stats only
    python start/update_all.py --status

    # Test connections only
    python start/update_all.py --test

CLI Arguments:
    --market     Fetch market data only (Yahoo Finance)
    --economic   Fetch economic data only (FRED)
    --all        Fetch all data (default)
    --status     Show database statistics
    --test       Test API connections
    --no-db      Fetch but don't write to database
    --start      Start date for data (YYYY-MM-DD)
    --end        End date for data (YYYY-MM-DD)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_market_data(write_to_db: bool = True, start_date: str = None, end_date: str = None):
    """Fetch all market data from Yahoo Finance."""
    print("\n" + "=" * 60)
    print("FETCHING MARKET DATA")
    print("=" * 60)

    from fetch.fetcher_yahoo import YahooFetcher

    fetcher = YahooFetcher()
    results = fetcher.fetch_all(
        write_to_db=write_to_db,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\nMarket data: {len(results)} instruments fetched")
    return results


def fetch_economic_data(write_to_db: bool = True, start_date: str = None, end_date: str = None):
    """Fetch all economic data from FRED."""
    print("\n" + "=" * 60)
    print("FETCHING ECONOMIC DATA")
    print("=" * 60)

    from fetch.fetcher_fred import FREDFetcher

    fetcher = FREDFetcher()
    results = fetcher.fetch_all(
        write_to_db=write_to_db,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\nEconomic data: {len(results)} series fetched")
    return results


def show_status():
    """Show database statistics."""
    print("\n" + "=" * 60)
    print("PRISM DATABASE STATUS")
    print("=" * 60)

    try:
        from data.sql.db import get_db_path, get_table_stats, list_indicators, database_stats

        db_path = get_db_path()
        print(f"\nDatabase: {db_path}")

        # Check if DB exists
        if not Path(db_path).exists():
            print("  [Database not initialized]")
            print("  Run: python start/update_all.py --all")
            return

        stats = database_stats()

        print(f"\nTable Statistics:")
        for table, count in stats.get("table_stats", {}).items():
            print(f"  {table}: {count} rows")

        date_range = stats.get("date_range", (None, None))
        if date_range[0]:
            print(f"\nDate Range: {date_range[0]} to {date_range[1]}")

        systems = stats.get("systems", [])
        if systems:
            print(f"\nSystems: {', '.join(systems)}")

        # Show indicator summary
        indicators = list_indicators()
        if indicators:
            print(f"\nIndicators by system:")
            by_system = {}
            for ind in indicators:
                sys = ind.get("system", "unknown")
                by_system[sys] = by_system.get(sys, 0) + 1
            for sys, count in by_system.items():
                print(f"  {sys}: {count}")

    except ImportError as e:
        print(f"Error loading database module: {e}")
    except Exception as e:
        print(f"Error: {e}")


def test_connections():
    """Test all API connections."""
    print("\n" + "=" * 60)
    print("TESTING API CONNECTIONS")
    print("=" * 60)

    results = {}

    # Test Yahoo Finance
    print("\nTesting Yahoo Finance...")
    try:
        from fetch.fetcher_yahoo import YahooFetcher
        fetcher = YahooFetcher()
        if fetcher.test_connection():
            print("  Yahoo Finance: OK")
            results["yahoo"] = True
        else:
            print("  Yahoo Finance: FAILED")
            results["yahoo"] = False
    except Exception as e:
        print(f"  Yahoo Finance: ERROR - {e}")
        results["yahoo"] = False

    # Test FRED
    print("\nTesting FRED API...")
    try:
        from fetch.fetcher_fred import FREDFetcher
        fetcher = FREDFetcher()
        if fetcher.test_connection():
            print("  FRED API: OK")
            results["fred"] = True
        else:
            print("  FRED API: FAILED")
            print("  (Check FRED_API_KEY environment variable)")
            results["fred"] = False
    except Exception as e:
        print(f"  FRED API: ERROR - {e}")
        results["fred"] = False

    # Test database connection
    print("\nTesting Database...")
    try:
        from data.sql.db import connect, get_db_path
        db_path = get_db_path()
        conn = connect()
        conn.close()
        print(f"  Database: OK ({db_path})")
        results["database"] = True
    except Exception as e:
        print(f"  Database: ERROR - {e}")
        results["database"] = False

    # Summary
    print("\n" + "-" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Connection tests: {passed}/{total} passed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Data Update Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--market", action="store_true", help="Fetch market data only")
    parser.add_argument("--economic", action="store_true", help="Fetch economic data only")
    parser.add_argument("--all", action="store_true", help="Fetch all data (default)")
    parser.add_argument("--status", action="store_true", help="Show database status")
    parser.add_argument("--test", action="store_true", help="Test API connections")
    parser.add_argument("--no-db", action="store_true", help="Don't write to database")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    print("=" * 60)
    print("PRISM ENGINE - Data Update")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Handle special modes
    if args.status:
        show_status()
        return

    if args.test:
        test_connections()
        return

    # Determine what to fetch
    fetch_market = args.market or args.all or (not args.market and not args.economic)
    fetch_economic = args.economic or args.all or (not args.market and not args.economic)
    write_to_db = not args.no_db

    market_results = {}
    economic_results = {}

    if fetch_market:
        market_results = fetch_market_data(
            write_to_db=write_to_db,
            start_date=args.start,
            end_date=args.end
        )

    if fetch_economic:
        economic_results = fetch_economic_data(
            write_to_db=write_to_db,
            start_date=args.start,
            end_date=args.end
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Market instruments: {len(market_results)}")
    print(f"Economic series:    {len(economic_results)}")
    print(f"Total indicators:   {len(market_results) + len(economic_results)}")
    print(f"Write to DB:        {'Yes' if write_to_db else 'No'}")

    if write_to_db:
        show_status()


if __name__ == "__main__":
    main()
