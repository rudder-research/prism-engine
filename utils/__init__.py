"""
PRISM Engine - Utilities
"""

# Logging utilities
from .logging_config import setup_logging

# Checkpoints
from .checkpoint_manager import CheckpointManager

# DB Manager (temporal DB)
from .db_manager import (
    TemporalDB,
    init_db,
    query_indicator_history,
    query_window_top_n,
    export_to_csv,
)

# DB Connector (migration runner, inserts, loads)
from .db_connector import (
    run_pending_migrations,
    init_database,
    insert_market_price,
    insert_market_dividend,
    insert_market_tri,
    upsert_market_prices,
    insert_econ_value,
    upsert_econ_values,
    load_market_prices,
    load_econ_values,
    log_fetch,
    get_market_stats,
    get_econ_stats,
)

# Registry-based DB helpers
from .db_registry import (
    load_system_registry,
    get_db_path,
    get_path,
    get_connection,
    execute_query,
    execute_script,
    table_exists,
    get_table_info,
    get_all_tables,
)

# Date cleaning utilities
from .date_cleaner import (
    parse_date_strict,
    fix_two_digit_year,
    strip_time_and_timezone,
    validate_date_range,
    clean_date_column,
    to_iso_date,
)

# Number cleaning utilities
from .number_cleaner import (
    parse_numeric,
    is_numeric_value,
    handle_percentage,
    clean_numeric_column,
    coerce_to_float_series,
    detect_numeric_garbage,
    summarize_numeric_issues,
)

# Fetch validation utilities
from .fetch_validator import (
    ValidationError,
    ValidationWarning,
    validate_dataframe_shape,
    detect_footer_garbage,
    remove_footer_garbage,
    validate_no_duplicate_dates,
    remove_duplicate_dates,
    validate_date_sequence,
    validate_frequency,
    validate_numeric_columns,
    validate_no_future_dates,
    comprehensive_validate,
)

# Registry validation utilities
from .registry_validator import (
    validate_all_registries,
    validate_system_registry,
    validate_market_registry,
    validate_economic_registry,
    load_validated_registry,
    get_enabled_tickers,
    registries_are_valid,
)

__all__ = [
    # Logging
    'setup_logging',

    # Checkpoint
    'CheckpointManager',

    # DB Manager
    'TemporalDB',
    'init_db',
    'query_indicator_history',
    'query_window_top_n',
    'export_to_csv',

    # DB Connector
    'run_pending_migrations',
    'init_database',
    'insert_market_price',
    'insert_market_dividend',
    'insert_market_tri',
    'upsert_market_prices',
    'insert_econ_value',
    'upsert_econ_values',
    'load_market_prices',
    'load_econ_values',
    'log_fetch',
    'get_market_stats',
    'get_econ_stats',

    # Registry helpers
    'load_system_registry',
    'get_db_path',
    'get_path',
    'get_connection',
    'execute_query',
    'execute_script',
    'table_exists',
    'get_table_info',
    'get_all_tables',

    # Date cleaning
    'parse_date_strict',
    'fix_two_digit_year',
    'strip_time_and_timezone',
    'validate_date_range',
    'clean_date_column',
    'to_iso_date',

    # Number cleaning
    'parse_numeric',
    'is_numeric_value',
    'handle_percentage',
    'clean_numeric_column',
    'coerce_to_float_series',
    'detect_numeric_garbage',
    'summarize_numeric_issues',

    # Fetch validation
    'ValidationError',
    'ValidationWarning',
    'validate_dataframe_shape',
    'detect_footer_garbage',
    'remove_footer_garbage',
    'validate_no_duplicate_dates',
    'remove_duplicate_dates',
    'validate_date_sequence',
    'validate_frequency',
    'validate_numeric_columns',
    'validate_no_future_dates',
    'comprehensive_validate',

    # Registry validation
    'validate_all_registries',
    'validate_system_registry',
    'validate_market_registry',
    'validate_economic_registry',
    'load_validated_registry',
    'get_enabled_tickers',
    'registries_are_valid',
]

# Sorting helper used by loaders/fetchers
DEFAULT_SORT_CONFIG = {
    "sort_by": "date",
    "ascending": True,
    "drop_duplicates": True,
}
