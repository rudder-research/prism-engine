"""
PRISM Engine - Utilities
"""

from .logging_config import setup_logging
from .checkpoint_manager import CheckpointManager
from .db_manager import TemporalDB, init_db, query_indicator_history, query_window_top_n, export_to_csv

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
    sort_by_date,
    validate_frequency,
    validate_numeric_columns,
    validate_no_future_dates,
    comprehensive_validate,
)

__all__ = [
    # Existing exports
    'setup_logging',
    'CheckpointManager',
    'TemporalDB',
    'init_db',
    'query_indicator_history',
    'query_window_top_n',
    'export_to_csv',
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
    'sort_by_date',
    'validate_frequency',
    'validate_numeric_columns',
    'validate_no_future_dates',
    'comprehensive_validate',
]
