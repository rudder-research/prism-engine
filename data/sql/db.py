"""
Unified Database API for PRISM Engine.

This module provides a compatibility layer that exposes a unified API
for database operations. It re-exports functions from prism_db.py and
db_connector.py to provide a single import point.

Usage:
    from data.sql.db import (
        # Connection management
        get_connection,
        init_db,

        # Indicator management
        add_indicator,
        get_indicator,
        list_indicators,

        # Data operations
        write_dataframe,
        load_indicator,
        load_multiple_indicators,

        # Fetch logging
        log_fetch,
    )

Example:
    from data.sql.db import init_db, add_indicator, write_dataframe

    # Initialize database
    init_db()

    # Add and write data
    add_indicator("SPY", system="finance", source="Yahoo")
    write_dataframe(df, "SPY", system="finance")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Import from prism_db (primary database module)
from .prism_db import (
    # Connection
    get_connection,
    get_db_path,
    init_db,

    # Indicator management
    add_indicator,
    get_indicator,
    list_indicators,
    update_indicator,
    delete_indicator,

    # Data operations
    write_dataframe,
    load_indicator,
    load_multiple_indicators,
    load_system_indicators,

    # Utilities
    run_migration,
    get_db_stats,

    # Constants
    system_types,
)

# Alias for backward compatibility
initialize_db = init_db

_logger = logging.getLogger(__name__)


# =============================================================================
# FETCH LOGGING
# =============================================================================

def log_fetch(
    source: str,
    entity: str,
    operation: str,
    status: str,
    *,
    rows_fetched: int = 0,
    rows_inserted: int = 0,
    rows_updated: int = 0,
    error_message: Optional[str] = None,
    started_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Log a fetch operation to the database.

    This function records fetch operations for auditing and debugging.
    It requires the fetch_log table to exist (created by migration 006).

    Args:
        source: Data source (e.g., 'yahoo', 'fred', 'custom')
        entity: Ticker or series code fetched (e.g., 'SPY', 'GDP')
        operation: Operation type ('fetch', 'update', 'backfill')
        status: Status ('success', 'error', 'partial')
        rows_fetched: Number of rows retrieved from source
        rows_inserted: Number of new rows inserted
        rows_updated: Number of existing rows updated
        error_message: Error details if status='error'
        started_at: Start time (ISO format, defaults to now)
        duration_ms: Duration in milliseconds
        db_path: Optional path to database

    Returns:
        Row ID of the log entry

    Example:
        log_fetch(
            source="fred",
            entity="GDP",
            operation="fetch",
            status="success",
            rows_fetched=100,
            rows_inserted=100
        )
    """
    if started_at is None:
        started_at = datetime.now().isoformat()

    with get_connection(db_path) as conn:
        # Check if fetch_log table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
        )
        if cursor.fetchone() is None:
            _logger.warning(
                "fetch_log table not found. Run migrations to create it. "
                "Skipping log entry."
            )
            return -1

        cursor = conn.execute(
            """
            INSERT INTO fetch_log
                (source, entity, operation, status, rows_fetched, rows_inserted,
                 rows_updated, error_message, started_at, completed_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (source, entity, operation, status, rows_fetched, rows_inserted,
             rows_updated, error_message, started_at, duration_ms),
        )
        conn.commit()
        return cursor.lastrowid


def get_fetch_history(
    entity: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Get fetch history from the log.

    Args:
        entity: Filter by entity (ticker/series code)
        source: Filter by source
        limit: Maximum number of records to return
        db_path: Optional path to database

    Returns:
        DataFrame with fetch log entries
    """
    with get_connection(db_path) as conn:
        # Check if fetch_log table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
        )
        if cursor.fetchone() is None:
            return pd.DataFrame()

        query = "SELECT * FROM fetch_log WHERE 1=1"
        params = []

        if entity:
            query += " AND entity = ?"
            params.append(entity)
        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        return pd.read_sql_query(query, conn, params=params)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_write(
    df: pd.DataFrame,
    indicator_name: str,
    system: str,
    source: str,
    *,
    frequency: str = "daily",
    date_column: str = "date",
    value_column: str = "value",
    db_path: Optional[Path] = None,
) -> int:
    """
    Quick write with automatic indicator creation and fetch logging.

    This is a convenience function that:
    1. Creates the indicator if it doesn't exist
    2. Writes the data
    3. Logs the fetch operation

    Args:
        df: DataFrame with the data
        indicator_name: Name of the indicator
        system: System/domain type
        source: Data source name
        frequency: Data frequency
        date_column: Name of date column in df
        value_column: Name of value column in df
        db_path: Optional path to database

    Returns:
        Number of rows written
    """
    started_at = datetime.now().isoformat()

    try:
        rows = write_dataframe(
            df,
            indicator_name,
            system=system,
            source=source,
            frequency=frequency,
            date_column=date_column,
            value_column=value_column,
            db_path=db_path,
        )

        log_fetch(
            source=source.lower(),
            entity=indicator_name,
            operation="fetch",
            status="success",
            rows_fetched=len(df),
            rows_inserted=rows,
            started_at=started_at,
            db_path=db_path,
        )

        return rows

    except Exception as e:
        log_fetch(
            source=source.lower(),
            entity=indicator_name,
            operation="fetch",
            status="error",
            error_message=str(e),
            started_at=started_at,
            db_path=db_path,
        )
        raise


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Connection management
    "get_connection",
    "get_db_path",
    "init_db",
    "initialize_db",  # Alias

    # Indicator management
    "add_indicator",
    "get_indicator",
    "list_indicators",
    "update_indicator",
    "delete_indicator",

    # Data operations
    "write_dataframe",
    "load_indicator",
    "load_multiple_indicators",
    "load_system_indicators",

    # Fetch logging
    "log_fetch",
    "get_fetch_history",

    # Convenience
    "quick_write",

    # Utilities
    "run_migration",
    "get_db_stats",

    # Constants
    "system_types",
]
