
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
"""
Full Update Orchestration for PRISM Engine.

Orchestrates the complete data pipeline:
1. Load and validate registry
2. Run all fetchers (Yahoo, FRED)
3. Build synthetic time series
4. Build technical indicators
5. Prepare geometry engine inputs

CLI Usage:
    python -m start.update_all
    python -m start.update_all --skip-fetch  # Skip data fetching
    python -m start.update_all --start-date 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the update script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def step_header(step_num: int, title: str) -> None:
    """Print a step header."""
    print()
    print("=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


def load_and_validate_registry() -> dict:
    """Load and validate the metric registry."""
    from data.registry import load_metric_registry, validate_registry

    registry = load_metric_registry()
    validate_registry(registry)
    return registry


def run_yahoo_fetcher(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch market data from Yahoo Finance and store in database.

    Returns number of indicators updated.
    """
    from fetch.fetcher_yahoo import fetch_registry_market_data
    from data.sql.prism_db import write_dataframe

    logger.info("Fetching market data from Yahoo Finance...")

    df = fetch_registry_market_data(registry, start_date, end_date)

    if df.empty:
        logger.warning("No data returned from Yahoo Finance")
        return 0

    # Write each column (except date) as an indicator
    count = 0
    date_col = df["date"]

    for col in df.columns:
        if col == "date":
            continue

        indicator_df = df[["date", col]].copy()
        indicator_df.columns = ["date", "value"]
        indicator_df = indicator_df.dropna()

        if indicator_df.empty:
            logger.warning(f"No data for {col}")
            continue

        try:
            # Use the prism_db write function
            cursor = conn.cursor()

            # Get or create indicator
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (col,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (col, "finance", "daily", "yahoo"),
                )
                indicator_id = cursor.lastrowid

            # Write values
            for _, r in indicator_df.iterrows():
                date_str = r["date"].strftime("%Y-%m-%d")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(r["value"])),
                )

            conn.commit()
            count += 1
            logger.info(f"  {col}: {len(indicator_df)} rows")

        except Exception as e:
            logger.error(f"Error writing {col}: {e}")

    return count


def run_fred_fetcher(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """
    Fetch economic data from FRED and store in database.

    Returns number of indicators updated.
    """
    try:
        from fetch.fetcher_fred import FREDFetcher
    except ImportError:
        logger.warning("FRED fetcher not available, skipping")
        return 0

    logger.info("Fetching economic data from FRED...")

    # Get economic metrics from registry
    economic_metrics = registry.get("economic", [])

    # FRED series mapping (lowercase registry name -> FRED series ID)
    REGISTRY_TO_FRED = {
        "cpi": "CPIAUCSL",
        "ppi": "PPIACO",
        "unrate": "UNRATE",
        "payems": "PAYEMS",
        "m2": "M2SL",
        "gdp": "GDP",
        "dgs10": "DGS10",
        "dgs2": "DGS2",
        "dgs3mo": "DGS3MO",
        "baaffm": "BAAFFM",
        "baa10ym": "BAA10YM",
    }

    fetcher = FREDFetcher()
    count = 0

    for metric in economic_metrics:
        name = metric.get("name", "").lower()
        freq = metric.get("frequency", "daily")

        fred_id = REGISTRY_TO_FRED.get(name)
        if not fred_id:
            logger.warning(f"No FRED mapping for {name}")
            continue

        try:
            df = fetcher.fetch_single(fred_id, start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                logger.warning(f"No data for {name} ({fred_id})")
                continue

            # Standardize column names
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            if fred_id in df.columns:
                df = df.rename(columns={fred_id: "value"})
            elif "value" not in df.columns:
                # Take the first non-date column
                value_col = [c for c in df.columns if c != "date"][0]
                df = df.rename(columns={value_col: "value"})

            df = df[["date", "value"]].dropna()

            if df.empty:
                continue

            # Write to database
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, "finance", freq, "fred"),
                )
                indicator_id = cursor.lastrowid

            for _, r in df.iterrows():
                date_str = r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])[:10]
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(r["value"])),
                )

            conn.commit()
            count += 1
            logger.info(f"  {name}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")

    return count


def run_synthetic_builder(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, int]:
    """Build synthetic time series."""
    from data.sql.synthetic_pipeline import build_synthetic_timeseries

    logger.info("Building synthetic time series...")
    return build_synthetic_timeseries(registry, conn, start_date, end_date)


def run_technical_builder(
    registry: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, int]:
    """Build technical indicators."""
    from engine_core.metrics.sector_technicals import build_technical_indicators

    logger.info("Building technical indicators...")
    return build_technical_indicators(registry, conn, start_date, end_date)


def verify_geometry_inputs(
    registry: dict,
    conn: sqlite3.Connection,
) -> int:
    """Verify geometry inputs are available."""
    from engine_core.geometry.inputs import get_geometry_input_series

    logger.info("Verifying geometry engine inputs...")
    inputs = get_geometry_input_series(conn, registry)
    return len(inputs)


def main() -> int:
    """Main entry point for update_all."""
    parser = argparse.ArgumentParser(
        description="Full data pipeline update for PRISM Engine"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching (Yahoo/FRED)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Start date for data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    print()
    print("=" * 60)
    print("PRISM ENGINE - FULL UPDATE PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start date: {args.start_date}")
    print(f"End date: {args.end_date or 'today'}")
    print(f"Skip fetch: {args.skip_fetch}")

    try:
        # Step 1: Load and validate registry
        step_header(1, "LOAD AND VALIDATE REGISTRY")
        registry = load_and_validate_registry()
        print(f"Registry version: {registry.get('version')}")
        print(f"Market metrics: {len(registry.get('market', []))}")
        print(f"Economic metrics: {len(registry.get('economic', []))}")
        print(f"Synthetic metrics: {len(registry.get('synthetic', []))}")
        print(f"Technical metrics: {len(registry.get('technical', []))}")

        # Initialize database connection
        from data.sql.prism_db import get_db_path, initialize_db

        db_path = get_db_path()
        initialize_db()
        print(f"Database: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Step 2: Run fetchers
        if not args.skip_fetch:
            step_header(2, "FETCH DATA")

            yahoo_count = run_yahoo_fetcher(
                registry, conn, args.start_date, args.end_date
            )
            print(f"Yahoo Finance: {yahoo_count} indicators updated")

            fred_count = run_fred_fetcher(
                registry, conn, args.start_date, args.end_date
            )
            print(f"FRED: {fred_count} indicators updated")
        else:
            step_header(2, "FETCH DATA (SKIPPED)")
            print("Skipping data fetch as requested")

        # Step 3: Build synthetics
        step_header(3, "BUILD SYNTHETIC SERIES")
        synthetic_results = run_synthetic_builder(
            registry, conn, args.start_date, args.end_date
        )
        synthetic_ok = sum(1 for v in synthetic_results.values() if v > 0)
        print(f"Synthetic series built: {synthetic_ok}/{len(synthetic_results)}")

        # Step 4: Build technicals
        step_header(4, "BUILD TECHNICAL INDICATORS")
        technical_results = run_technical_builder(
            registry, conn, args.start_date, args.end_date
        )
        technical_ok = sum(1 for v in technical_results.values() if v > 0)
        print(f"Technical indicators built: {technical_ok}/{len(technical_results)}")

        # Step 5: Verify geometry inputs
        step_header(5, "VERIFY GEOMETRY INPUTS")
        geometry_count = verify_geometry_inputs(registry, conn)
        print(f"Geometry input series available: {geometry_count}")

        # Summary
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        conn.close()
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
