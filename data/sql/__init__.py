"""
PRISM SQL Database Module
=========================

SQLite-based storage for indicators and time series data.

Two modules are available:

1. data.sql.db - Modern, unified API (recommended)
    from data.sql.db import init_database, add_indicator, write_dataframe, load_indicator

2. data.sql.prism_db - Legacy API with panel/system support
    from data.sql.prism_db import add_indicator, write_dataframe, load_indicator
"""

# Import from the new unified db module
from .db import (
    get_db_path,
    connect,
    init_database,
    init_db,
    query,
    add_indicator,
    get_indicator,
    list_indicators,
    write_dataframe,
    load_indicator,
    load_multiple_indicators,
    log_fetch,
    get_date_range,
    export_to_csv,
    get_table_stats,
    database_stats,
)

# Also expose prism_db for backward compatibility
from . import prism_db

__all__ = [
    # From db module
    "get_db_path",
    "connect",
    "init_database",
    "init_db",
    "query",
    "add_indicator",
    "get_indicator",
    "list_indicators",
    "write_dataframe",
    "load_indicator",
    "load_multiple_indicators",
    "log_fetch",
    "get_date_range",
    "export_to_csv",
    "get_table_stats",
    "database_stats",
    # Legacy module
    "prism_db",
]
