# Database Schema

This document describes the SQLite database schema used by PRISM for persistent storage of indicators and time series data.

## Overview

PRISM uses SQLite for lightweight, file-based storage. The database supports:

- Multiple domain systems (finance, climate, chemistry, etc.)
- Flexible indicator metadata
- Efficient time series storage with unique constraints

## Schema Location

- **Schema Definition**: `data/sql/prism_schema.sql`
- **Python API**: `data/sql/prism_db.py`
- **Migrations**: `data/sql/migrations/`
- **Default Database**: `data/sql/prism.db`

## Tables

### indicators

Stores metadata about each indicator (time series).

```sql
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,           -- Unique identifier (e.g., "SPY", "GDP")
    system TEXT NOT NULL,                -- Domain: finance, climate, chemistry, etc.
    frequency TEXT NOT NULL,             -- daily, weekly, monthly, quarterly, yearly
    source TEXT,                         -- Data source (FRED, Yahoo, etc.)
    units TEXT,                          -- Unit of measurement
    description TEXT,                    -- Human-readable description
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Columns**:

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique row identifier |
| name | TEXT | NOT NULL, UNIQUE | Indicator name (e.g., "SPY") |
| system | TEXT | NOT NULL | Domain classification |
| frequency | TEXT | NOT NULL | Data frequency |
| source | TEXT | - | Data provider |
| units | TEXT | - | Unit of measurement |
| description | TEXT | - | Human-readable description |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Creation timestamp |
| updated_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Last update timestamp |

**Indexes**:

```sql
CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);
```

### indicator_values

Stores the actual time series data points.

```sql
CREATE TABLE IF NOT EXISTS indicator_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,       -- Foreign key to indicators
    date TEXT NOT NULL,                  -- ISO format: YYYY-MM-DD
    value REAL NOT NULL,                 -- Numeric value
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
    UNIQUE(indicator_id, date)           -- One value per indicator per date
);
```

**Columns**:

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique row identifier |
| indicator_id | INTEGER | NOT NULL, FOREIGN KEY | Reference to indicators table |
| date | TEXT | NOT NULL | Date in ISO format (YYYY-MM-DD) |
| value | REAL | NOT NULL | Numeric value |
| created_at | TEXT | DEFAULT CURRENT_TIMESTAMP | Creation timestamp |

**Indexes**:

```sql
CREATE INDEX IF NOT EXISTS idx_indicator_values_date ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_indicator_id ON indicator_values(indicator_id);
```

**Unique Constraint**:

The `UNIQUE(indicator_id, date)` constraint ensures:
- Only one value per indicator per date
- Duplicate inserts will fail (use INSERT OR REPLACE)
- Data integrity for time series

## Valid System Types

The `system` column accepts the following values:

| System | Description |
|--------|-------------|
| finance | Financial markets, equities, bonds, commodities |
| climate | Climate and environmental indicators |
| chemistry | Chemical process data |
| anthropology | Demographic and social indicators |
| biology | Biological and health data |
| physics | Physical measurements |

**Note**: Validation is enforced at the application layer in `prism_db.py`.

## Triggers

### update_indicator_timestamp

Automatically updates `updated_at` when an indicator is modified:

```sql
CREATE TRIGGER IF NOT EXISTS update_indicator_timestamp
    AFTER UPDATE ON indicators
    FOR EACH ROW
BEGIN
    UPDATE indicators SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;
```

## Migrations

Migrations are stored in `data/sql/migrations/` and applied sequentially.

### Migration History

| Migration | Description |
|-----------|-------------|
| 001_rename_panel_to_system.sql | Renamed `panel` column to `system` |

### Running Migrations

```python
from data.sql.prism_db import run_migration
from pathlib import Path

migration_path = Path("data/sql/migrations/001_rename_panel_to_system.sql")
run_migration(migration_path)
```

## Python API

### Initialization

```python
from data.sql.prism_db import init_db, get_db_path

# Initialize database (creates tables if not exist)
init_db()

# Or specify custom path
init_db(db_path=Path("/custom/path/prism.db"))

# Get current database path
print(get_db_path())  # Uses PRISM_DB env var or default
```

### Adding Indicators

```python
from data.sql.prism_db import add_indicator

indicator_id = add_indicator(
    name="SPY",
    system="finance",
    frequency="daily",
    source="Yahoo Finance",
    units="USD",
    description="S&P 500 ETF"
)
```

### Writing Data

```python
import pandas as pd
from data.sql.prism_db import write_dataframe

df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "value": [100.0, 101.5, 99.8]
})

rows_written = write_dataframe(
    df,
    indicator_name="SPY",
    system="finance",
    date_column="date",
    value_column="value",
    create_if_missing=True
)
```

### Loading Data

```python
from data.sql.prism_db import load_indicator, load_multiple_indicators

# Load single indicator
spy_data = load_indicator("SPY")
spy_data = load_indicator("SPY", start_date="2020-01-01", end_date="2024-01-01")

# Load multiple indicators
data = load_multiple_indicators(["SPY", "QQQ", "TLT"])

# Load all indicators for a system
from data.sql.prism_db import load_system_indicators
finance_data = load_system_indicators("finance")
```

### Listing Indicators

```python
from data.sql.prism_db import list_indicators

# List all indicators
all_indicators = list_indicators()

# List by system
finance_indicators = list_indicators(system="finance")
```

### Updating and Deleting

```python
from data.sql.prism_db import update_indicator, delete_indicator

# Update indicator metadata
update_indicator("SPY", description="Updated description")

# Delete indicator and all its values
delete_indicator("SPY")
```

### Database Statistics

```python
from data.sql.prism_db import get_db_stats

stats = get_db_stats()
# Returns:
# {
#     "indicator_count": 35,
#     "value_count": 125000,
#     "systems": {"finance": 30, "climate": 5}
# }
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PRISM_DB | Custom database path | `data/sql/prism.db` |

## Constraints Behavior

### Unique Constraint on indicator_values

```python
# First insert succeeds
write_dataframe(df1, "SPY", system="finance")

# Duplicate date insert replaces existing value
write_dataframe(df2, "SPY", system="finance")  # Uses INSERT OR REPLACE
```

### Foreign Key Constraint

```python
# Deleting an indicator cascades to delete all its values
delete_indicator("SPY")  # Also deletes all SPY values
```

### Unique Name Constraint

```python
# Second add with same name raises IntegrityError
add_indicator("SPY", system="finance")
add_indicator("SPY", system="finance")  # sqlite3.IntegrityError
```

## Query Examples

### Get indicator with date range

```sql
SELECT iv.date, iv.value
FROM indicator_values iv
JOIN indicators i ON iv.indicator_id = i.id
WHERE i.name = 'SPY'
  AND iv.date >= '2020-01-01'
  AND iv.date <= '2024-01-01'
ORDER BY iv.date;
```

### Get all indicators for a system

```sql
SELECT name, frequency, source, description
FROM indicators
WHERE system = 'finance'
ORDER BY name;
```

### Count values per indicator

```sql
SELECT i.name, COUNT(iv.id) as value_count
FROM indicators i
LEFT JOIN indicator_values iv ON i.id = iv.indicator_id
GROUP BY i.id
ORDER BY value_count DESC;
```

## Best Practices

1. **Initialize First**: Always call `init_db()` before database operations
2. **Use Context Manager**: The `get_connection()` context manager handles cleanup
3. **Batch Writes**: For large datasets, use `write_dataframe()` rather than individual inserts
4. **Date Format**: Always use ISO format (YYYY-MM-DD) for dates
5. **System Validation**: Validate system types before inserting
