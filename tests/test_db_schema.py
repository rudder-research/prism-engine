"""
Test Database Schema

Tests for:
- Tables exist with required columns
- Unique constraints behave correctly (duplicate inserts fail)
- Foreign key constraints work
- Data types are correct
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from contextlib import contextmanager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def schema_path():
    """Path to the schema SQL file."""
    return Path(__file__).parent.parent / "data" / "sql" / "prism_schema.sql"


@pytest.fixture
def schema_sql(schema_path):
    """Load schema SQL content."""
    if not schema_path.exists():
        pytest.skip("Schema file not found")
    return schema_path.read_text()


@pytest.fixture
def temp_db(schema_sql):
    """Create a temporary database with schema applied."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()

    yield conn

    conn.close()
    db_path.unlink()


@pytest.fixture
def db_with_data(temp_db):
    """Database with sample data inserted."""
    cursor = temp_db.cursor()

    # Insert sample indicator
    cursor.execute("""
        INSERT INTO indicators (name, system, frequency, source, units, description)
        VALUES ('SPY', 'finance', 'daily', 'Yahoo', 'USD', 'S&P 500 ETF')
    """)

    # Get indicator ID
    indicator_id = cursor.lastrowid

    # Insert sample values
    cursor.executemany("""
        INSERT INTO indicator_values (indicator_id, date, value)
        VALUES (?, ?, ?)
    """, [
        (indicator_id, "2020-01-01", 100.0),
        (indicator_id, "2020-01-02", 101.5),
        (indicator_id, "2020-01-03", 99.8),
    ])

    temp_db.commit()

    return temp_db


# ============================================================================
# Schema File Tests
# ============================================================================

class TestSchemaFile:
    """Tests for schema file existence and validity."""

    def test_schema_file_exists(self, schema_path):
        """Schema file should exist."""
        assert schema_path.exists(), f"Schema file not found at {schema_path}"

    def test_schema_is_valid_sql(self, schema_sql, temp_db):
        """Schema should be valid SQL that can be executed."""
        # If we get here, the schema was successfully applied in the fixture
        cursor = temp_db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert len(tables) > 0, "Schema should create at least one table"


# ============================================================================
# Table Existence Tests
# ============================================================================

class TestTableExistence:
    """Tests for required tables."""

    def test_indicators_table_exists(self, temp_db):
        """The 'indicators' table should exist."""
        cursor = temp_db.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='indicators'
        """)
        assert cursor.fetchone() is not None, "indicators table should exist"

    def test_indicator_values_table_exists(self, temp_db):
        """The 'indicator_values' table should exist."""
        cursor = temp_db.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='indicator_values'
        """)
        assert cursor.fetchone() is not None, "indicator_values table should exist"


# ============================================================================
# Column Tests
# ============================================================================

class TestIndicatorsColumns:
    """Tests for indicators table columns."""

    REQUIRED_COLUMNS = ["id", "name", "system", "frequency"]
    OPTIONAL_COLUMNS = ["source", "units", "description", "created_at", "updated_at"]

    def get_column_info(self, db, table_name):
        """Get column information for a table."""
        cursor = db.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]}
                for row in cursor.fetchall()}

    def test_has_required_columns(self, temp_db):
        """Indicators table should have all required columns."""
        columns = self.get_column_info(temp_db, "indicators")

        for col in self.REQUIRED_COLUMNS:
            assert col in columns, f"Missing required column: {col}"

    def test_has_optional_columns(self, temp_db):
        """Indicators table should have optional columns."""
        columns = self.get_column_info(temp_db, "indicators")

        for col in self.OPTIONAL_COLUMNS:
            assert col in columns, f"Missing optional column: {col}"

    def test_id_is_primary_key(self, temp_db):
        """id column should be primary key."""
        columns = self.get_column_info(temp_db, "indicators")
        assert columns["id"]["pk"] == 1, "id should be primary key"

    def test_name_is_not_null(self, temp_db):
        """name column should be NOT NULL."""
        columns = self.get_column_info(temp_db, "indicators")
        assert columns["name"]["notnull"] == 1, "name should be NOT NULL"

    def test_system_is_not_null(self, temp_db):
        """system column should be NOT NULL."""
        columns = self.get_column_info(temp_db, "indicators")
        assert columns["system"]["notnull"] == 1, "system should be NOT NULL"


class TestIndicatorValuesColumns:
    """Tests for indicator_values table columns."""

    REQUIRED_COLUMNS = ["id", "indicator_id", "date", "value"]

    def get_column_info(self, db, table_name):
        """Get column information for a table."""
        cursor = db.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]}
                for row in cursor.fetchall()}

    def test_has_required_columns(self, temp_db):
        """indicator_values table should have all required columns."""
        columns = self.get_column_info(temp_db, "indicator_values")

        for col in self.REQUIRED_COLUMNS:
            assert col in columns, f"Missing required column: {col}"

    def test_indicator_id_is_not_null(self, temp_db):
        """indicator_id column should be NOT NULL."""
        columns = self.get_column_info(temp_db, "indicator_values")
        assert columns["indicator_id"]["notnull"] == 1, "indicator_id should be NOT NULL"

    def test_date_is_not_null(self, temp_db):
        """date column should be NOT NULL."""
        columns = self.get_column_info(temp_db, "indicator_values")
        assert columns["date"]["notnull"] == 1, "date should be NOT NULL"

    def test_value_is_not_null(self, temp_db):
        """value column should be NOT NULL."""
        columns = self.get_column_info(temp_db, "indicator_values")
        assert columns["value"]["notnull"] == 1, "value should be NOT NULL"


# ============================================================================
# Index Tests
# ============================================================================

class TestIndexes:
    """Tests for database indexes."""

    def get_indexes(self, db, table_name):
        """Get index names for a table."""
        cursor = db.cursor()
        cursor.execute(f"""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='{table_name}'
        """)
        return [row[0] for row in cursor.fetchall()]

    def test_indicators_has_system_index(self, temp_db):
        """indicators table should have index on system column."""
        indexes = self.get_indexes(temp_db, "indicators")
        assert any("system" in idx for idx in indexes), \
            "Should have index on system column"

    def test_indicators_has_name_index(self, temp_db):
        """indicators table should have index on name column."""
        indexes = self.get_indexes(temp_db, "indicators")
        assert any("name" in idx for idx in indexes), \
            "Should have index on name column"

    def test_indicator_values_has_date_index(self, temp_db):
        """indicator_values table should have index on date column."""
        indexes = self.get_indexes(temp_db, "indicator_values")
        assert any("date" in idx for idx in indexes), \
            "Should have index on date column"

    def test_indicator_values_has_indicator_id_index(self, temp_db):
        """indicator_values table should have index on indicator_id column."""
        indexes = self.get_indexes(temp_db, "indicator_values")
        assert any("indicator_id" in idx for idx in indexes), \
            "Should have index on indicator_id column"


# ============================================================================
# Unique Constraint Tests
# ============================================================================

class TestUniqueConstraints:
    """Tests for unique constraints."""

    def test_indicator_name_is_unique(self, temp_db):
        """Indicator names should be unique."""
        cursor = temp_db.cursor()

        # Insert first indicator
        cursor.execute("""
            INSERT INTO indicators (name, system, frequency)
            VALUES ('TEST', 'finance', 'daily')
        """)
        temp_db.commit()

        # Try to insert duplicate
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO indicators (name, system, frequency)
                VALUES ('TEST', 'finance', 'daily')
            """)

    def test_indicator_date_is_unique(self, db_with_data):
        """Each indicator can only have one value per date."""
        cursor = db_with_data.cursor()

        # Get existing indicator ID
        cursor.execute("SELECT id FROM indicators WHERE name='SPY'")
        indicator_id = cursor.fetchone()[0]

        # Try to insert duplicate date (2020-01-01 already exists)
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO indicator_values (indicator_id, date, value)
                VALUES (?, '2020-01-01', 999.0)
            """, (indicator_id,))


# ============================================================================
# Foreign Key Tests
# ============================================================================

class TestForeignKeys:
    """Tests for foreign key constraints."""

    def test_values_require_valid_indicator(self, temp_db):
        """indicator_values should require valid indicator_id."""
        cursor = temp_db.cursor()

        # Try to insert value with non-existent indicator_id
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO indicator_values (indicator_id, date, value)
                VALUES (99999, '2020-01-01', 100.0)
            """)

    def test_cascade_delete(self, db_with_data):
        """Deleting indicator should cascade delete values."""
        cursor = db_with_data.cursor()

        # Count values before delete
        cursor.execute("SELECT COUNT(*) FROM indicator_values")
        count_before = cursor.fetchone()[0]
        assert count_before > 0

        # Delete the indicator
        cursor.execute("DELETE FROM indicators WHERE name='SPY'")
        db_with_data.commit()

        # Values should be deleted
        cursor.execute("SELECT COUNT(*) FROM indicator_values")
        count_after = cursor.fetchone()[0]
        assert count_after == 0


# ============================================================================
# Data Type Tests
# ============================================================================

class TestDataTypes:
    """Tests for data type handling."""

    def test_value_stores_float(self, db_with_data):
        """Value column should store float values correctly."""
        cursor = db_with_data.cursor()
        cursor.execute("SELECT value FROM indicator_values WHERE date='2020-01-02'")
        value = cursor.fetchone()[0]

        assert value == 101.5
        assert isinstance(value, float)

    def test_date_stores_iso_format(self, db_with_data):
        """Date column should store ISO format strings."""
        cursor = db_with_data.cursor()
        cursor.execute("SELECT date FROM indicator_values ORDER BY date LIMIT 1")
        date = cursor.fetchone()[0]

        assert date == "2020-01-01"

    def test_timestamps_auto_populate(self, temp_db):
        """created_at should auto-populate."""
        cursor = temp_db.cursor()
        cursor.execute("""
            INSERT INTO indicators (name, system, frequency)
            VALUES ('TEST', 'finance', 'daily')
        """)
        temp_db.commit()

        cursor.execute("SELECT created_at FROM indicators WHERE name='TEST'")
        created_at = cursor.fetchone()[0]

        assert created_at is not None


# ============================================================================
# Query Tests
# ============================================================================

class TestQueries:
    """Tests for common query patterns."""

    def test_select_by_system(self, db_with_data):
        """Should be able to filter by system."""
        cursor = db_with_data.cursor()
        cursor.execute("SELECT * FROM indicators WHERE system='finance'")
        results = cursor.fetchall()

        assert len(results) >= 1

    def test_select_date_range(self, db_with_data):
        """Should be able to filter by date range."""
        cursor = db_with_data.cursor()
        cursor.execute("""
            SELECT * FROM indicator_values
            WHERE date >= '2020-01-01' AND date <= '2020-01-02'
        """)
        results = cursor.fetchall()

        assert len(results) == 2

    def test_join_indicators_and_values(self, db_with_data):
        """Should be able to join tables."""
        cursor = db_with_data.cursor()
        cursor.execute("""
            SELECT i.name, iv.date, iv.value
            FROM indicators i
            JOIN indicator_values iv ON i.id = iv.indicator_id
            ORDER BY iv.date
        """)
        results = cursor.fetchall()

        assert len(results) == 3
        assert results[0][0] == "SPY"


# ============================================================================
# CRUD Operations Tests
# ============================================================================

class TestCRUDOperations:
    """Tests for Create, Read, Update, Delete operations."""

    def test_create_indicator(self, temp_db):
        """Should be able to create an indicator."""
        cursor = temp_db.cursor()
        cursor.execute("""
            INSERT INTO indicators (name, system, frequency, source)
            VALUES ('NEW_IND', 'climate', 'monthly', 'NOAA')
        """)
        temp_db.commit()

        cursor.execute("SELECT * FROM indicators WHERE name='NEW_IND'")
        result = cursor.fetchone()

        assert result is not None

    def test_read_indicator(self, db_with_data):
        """Should be able to read an indicator."""
        cursor = db_with_data.cursor()
        cursor.execute("SELECT name, system, frequency FROM indicators WHERE name='SPY'")
        result = cursor.fetchone()

        assert result[0] == "SPY"
        assert result[1] == "finance"
        assert result[2] == "daily"

    def test_update_indicator(self, db_with_data):
        """Should be able to update an indicator."""
        cursor = db_with_data.cursor()
        cursor.execute("""
            UPDATE indicators
            SET description = 'Updated description'
            WHERE name = 'SPY'
        """)
        db_with_data.commit()

        cursor.execute("SELECT description FROM indicators WHERE name='SPY'")
        result = cursor.fetchone()

        assert result[0] == "Updated description"

    def test_delete_indicator(self, db_with_data):
        """Should be able to delete an indicator."""
        cursor = db_with_data.cursor()
        cursor.execute("DELETE FROM indicators WHERE name='SPY'")
        db_with_data.commit()

        cursor.execute("SELECT * FROM indicators WHERE name='SPY'")
        result = cursor.fetchone()

        assert result is None
