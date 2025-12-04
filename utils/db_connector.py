import sqlite3
import logging
import json
from pathlib import Path
from typing import Optional, Sequence, Mapping, Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _get_root_dir() -> Path:
    """Return the project root (one level above utils/)."""
    return Path(__file__).resolve().parents[1]


def _load_system_registry() -> dict:
    """Load system registry JSON from data_fetch/system_registry.json."""
    root = _get_root_dir()
    registry_path = root / "data_fetch" / "system_registry.json"
    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_db_path() -> Path:
    """Resolve the SQLite database path from the system registry."""
    registry = _load_system_registry()
    db_rel = registry.get("paths", {}).get("db_path", "data/sql/prism.db")
    root = _get_root_dir()
    db_path = root / db_rel
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_connection(conn_or_path: Optional[Any] = None) -> tuple[sqlite3.Connection, bool]:
    """
    Normalize a connection argument.

    Returns (connection, should_close).
    """
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    if isinstance(conn_or_path, (str, Path)):
        conn = sqlite3.connect(str(conn_or_path))
        return conn, True

    # Default: open from registry
    conn = sqlite3.connect(str(_get_db_path()))
    return conn, True


# ---------------------------------------------------------------------------
# Schema / migrations
# ---------------------------------------------------------------------------


def init_database(conn: Optional[Any] = None) -> None:
    """
    Initialize the SQLite database with minimal required tables.
    Safe to call multiple times.
    """
    db_path = _get_db_path()
    LOGGER.info("Initializing database at %s", db_path)

    conn, should_close = _get_connection(conn)
    try:
        cur = conn.cursor()

        # Market prices
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_prices (
                ticker     TEXT NOT NULL,
                date       TEXT NOT NULL,
                field      TEXT NOT NULL,
                value      REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date, field)
            )
            """
        )

        # Economic series
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS econ_values (
                series_id  TEXT NOT NULL,
                date       TEXT NOT NULL,
                value      REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_id, date)
            )
            """
        )

        # Fetch log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fetch_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                source     TEXT,
                status     TEXT,
                message    TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
    finally:
        if should_close:
            conn.close()


def run_pending_migrations(conn: Optional[Any] = None) -> None:
    """
    Placeholder for a full migration system.

    For now, this simply ensures core tables exist.
    """
    init_database(conn)


# ---------------------------------------------------------------------------
# Fetch logging
# ---------------------------------------------------------------------------


def log_fetch(source: str, status: str, message: str = "", conn: Optional[Any] = None) -> None:
    """Record a fetch event in the fetch_log table."""
    conn, should_close = _get_connection(conn)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fetch_log (source, status, message)
            VALUES (?, ?, ?)
            """,
            (source, status, message),
        )
        conn.commit()
    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Market price operations
# ---------------------------------------------------------------------------


def insert_market_price(
    conn_or_path: Optional[Any],
    ticker: str,
    date: str,
    field: str,
    value: float,
) -> None:
    """Insert or replace a single market price observation."""
    conn, should_close = _get_connection(conn_or_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO market_prices (ticker, date, field, value)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, date, field, float(value) if value is not None else None),
        )
        conn.commit()
    finally:
        if should_close:
            conn.close()


def insert_market_dividend(
    conn_or_path: Optional[Any],
    ticker: str,
    date: str,
    value: float,
) -> None:
    """
    Convenience wrapper: store dividends as field='dividend'
    in the market_prices table.
    """
    insert_market_price(conn_or_path, ticker, date, "dividend", value)


def insert_market_tri(
    conn_or_path: Optional[Any],
    ticker: str,
    date: str,
    value: float,
) -> None:
    """
    Convenience wrapper: store total-return index as field='tri'
    in the market_prices table.
    """
    insert_market_price(conn_or_path, ticker, date, "tri", value)


def upsert_market_prices(
    conn_or_path: Optional[Any],
    df: pd.DataFrame,
    table_name: str = "market_prices",
    **kwargs: Any,
) -> None:
    """
    Upsert a DataFrame of market prices.

    Expected columns:
      - 'ticker'
      - 'date'
      - EITHER:
          - 'field' and 'value'
        OR
          - wide columns like 'close', 'open', etc. which will be melted.
    """
    if df is None or df.empty:
        LOGGER.warning("upsert_market_prices called with empty DataFrame")
        return

    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("DataFrame must contain at least 'ticker' and 'date' columns")

    work = df.copy()

    if "field" in work.columns and "value" in work.columns:
        long_df = work[["ticker", "date", "field", "value"]]
    else:
        value_cols = [c for c in work.columns if c not in ("ticker", "date")]
        if not value_cols:
            raise ValueError("No value columns found for market prices")
        long_df = work.melt(
            id_vars=["ticker", "date"],
            value_vars=value_cols,
            var_name="field",
            value_name="value",
        )

    conn, should_close = _get_connection(conn_or_path)
    try:
        cur = conn.cursor()
        records = [
            (
                str(row["ticker"]),
                str(row["date"]),
                str(row["field"]),
                None if pd.isna(row["value"]) else float(row["value"]),
            )
            for _, row in long_df.iterrows()
        ]
        cur.executemany(
            f"""
            INSERT OR REPLACE INTO {table_name} (ticker, date, field, value)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    finally:
        if should_close:
            conn.close()


def load_market_prices(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    field: Optional[str] = None,
    conn: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Load market prices from database.
    
    Args:
        ticker: Filter by ticker symbol (optional)
        start_date: Filter by start date YYYY-MM-DD (optional)
        end_date: Filter by end date YYYY-MM-DD (optional)
        field: Filter by field type e.g. 'close', 'open' (optional)
        conn: Optional database connection
        
    Returns:
        DataFrame with columns: ticker, date, field, value
    """
    connection, should_close = _get_connection(conn)
    try:
        sql = "SELECT ticker, date, field, value FROM market_prices WHERE 1=1"
        params = []
        
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker)
        if field:
            sql += " AND field = ?"
            params.append(field)
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        
        sql += " ORDER BY ticker, date, field"
        
        return pd.read_sql_query(sql, connection, params=params)
    finally:
        if should_close:
            connection.close()


# ---------------------------------------------------------------------------
# Economic series operations
# ---------------------------------------------------------------------------


def insert_econ_value(
    conn_or_path: Optional[Any],
    series_id: str,
    date: str,
    value: float,
) -> None:
    """Insert or replace a single economic time-series observation."""
    conn, should_close = _get_connection(conn_or_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO econ_values (series_id, date, value)
            VALUES (?, ?, ?)
            """,
            (series_id, date, float(value) if value is not None else None),
        )
        conn.commit()
    finally:
        if should_close:
            conn.close()


def upsert_econ_values(
    conn_or_path: Optional[Any] = None,
    df: Optional[pd.DataFrame] = None,
    table_name: str = "econ_values",
) -> None:
    """
    Upsert a DataFrame of economic values.

    Expected DataFrame columns:
      - 'series_id'
      - 'date'
      - 'value'
    """
    if df is None or df.empty:
        LOGGER.warning("upsert_econ_values called with empty DataFrame")
        return

    required = {"series_id", "date", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    conn, should_close = _get_connection(conn_or_path)
    try:
        cur = conn.cursor()

        # Convert rows
        rows = []
        for _, row in df.iterrows():
            rows.append(
                (
                    str(row["series_id"]),
                    str(row["date"]),
                    None if pd.isna(row["value"]) else float(row["value"])
                )
            )

        cur.executemany(
            f"""
            INSERT OR REPLACE INTO {table_name} (series_id, date, value)
            VALUES (?, ?, ?)
            """,
            rows,
        )

        conn.commit()
    finally:
        if should_close:
            conn.close()


def load_econ_values(
    series_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Load economic values from database.
    
    Args:
        series_id: Filter by series ID e.g. 'GDP', 'UNRATE' (optional)
        start_date: Filter by start date YYYY-MM-DD (optional)
        end_date: Filter by end date YYYY-MM-DD (optional)
        conn: Optional database connection
        
    Returns:
        DataFrame with columns: series_id, date, value
    """
    connection, should_close = _get_connection(conn)
    try:
        sql = "SELECT series_id, date, value FROM econ_values WHERE 1=1"
        params = []
        
        if series_id:
            sql += " AND series_id = ?"
            params.append(series_id)
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        
        sql += " ORDER BY series_id, date"
        
        return pd.read_sql_query(sql, connection, params=params)
    finally:
        if should_close:
            connection.close()


# ---------------------------------------------------------------------------
# Statistics functions
# ---------------------------------------------------------------------------


def get_market_stats(conn: Optional[Any] = None) -> dict:
    """
    Get summary statistics for market data.
    
    Returns:
        Dictionary with keys: total_rows, unique_tickers, unique_fields, 
        min_date, max_date
    """
    connection, should_close = _get_connection(conn)
    try:
        stats = {}
        cur = connection.cursor()
        
        # Count rows
        cur.execute("SELECT COUNT(*) FROM market_prices")
        stats["total_rows"] = cur.fetchone()[0]
        
        # Count unique tickers
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM market_prices")
        stats["unique_tickers"] = cur.fetchone()[0]
        
        # Count unique fields
        cur.execute("SELECT COUNT(DISTINCT field) FROM market_prices")
        stats["unique_fields"] = cur.fetchone()[0]
        
        # Date range
        cur.execute("SELECT MIN(date), MAX(date) FROM market_prices")
        row = cur.fetchone()
        stats["min_date"] = row[0]
        stats["max_date"] = row[1]
        
        return stats
    except Exception as e:
        LOGGER.warning(f"get_market_stats error: {e}")
        return {
            "total_rows": 0, 
            "unique_tickers": 0, 
            "unique_fields": 0,
            "min_date": None, 
            "max_date": None
        }
    finally:
        if should_close:
            connection.close()


def get_econ_stats(conn: Optional[Any] = None) -> dict:
    """
    Get summary statistics for economic data.
    
    Returns:
        Dictionary with keys: total_rows, unique_series, min_date, max_date
    """
    connection, should_close = _get_connection(conn)
    try:
        stats = {}
        cur = connection.cursor()
        
        # Count rows
        cur.execute("SELECT COUNT(*) FROM econ_values")
        stats["total_rows"] = cur.fetchone()[0]
        
        # Count unique series
        cur.execute("SELECT COUNT(DISTINCT series_id) FROM econ_values")
        stats["unique_series"] = cur.fetchone()[0]
        
        # Date range
        cur.execute("SELECT MIN(date), MAX(date) FROM econ_values")
        row = cur.fetchone()
        stats["min_date"] = row[0]
        stats["max_date"] = row[1]
        
        return stats
    except Exception as e:
        LOGGER.warning(f"get_econ_stats error: {e}")
        return {
            "total_rows": 0, 
            "unique_series": 0, 
            "min_date": None, 
            "max_date": None
        }
    finally:
        if should_close:
            connection.close()