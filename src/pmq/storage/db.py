"""SQLite database connection and initialization."""

import sqlite3
from pathlib import Path
from typing import Any

from pmq.config import Settings, get_settings
from pmq.logging import get_logger

logger = get_logger("storage.db")

# Path to schema file
SCHEMA_FILE = Path(__file__).parent / "schema.sql"


class Database:
    """SQLite database wrapper with connection management."""

    def __init__(self, db_path: Path | None = None, settings: Settings | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file (overrides settings)
            settings: Application settings
        """
        self._settings = settings or get_settings()
        self._db_path = db_path or self._settings.db_path
        self._conn: sqlite3.Connection | None = None

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            logger.debug(f"Opening database: {self._db_path}")
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            logger.debug("Closing database connection")
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def initialize(self) -> None:
        """Initialize database schema from SQL file."""
        if not SCHEMA_FILE.exists():
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

        schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
        logger.info("Initializing database schema")

        conn = self.connection
        # Execute schema statements
        conn.executescript(schema_sql)
        logger.info("Database schema initialized")

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> sqlite3.Cursor:
        """Execute a SQL statement.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Cursor with results
        """
        conn = self.connection
        if params:
            return conn.execute(sql, params)
        return conn.execute(sql)

    def execute_many(
        self,
        sql: str,
        params_list: list[tuple[Any, ...]] | list[dict[str, Any]],
    ) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples/dicts

        Returns:
            Cursor with results
        """
        return self.connection.executemany(sql, params_list)

    def fetch_one(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> sqlite3.Row | None:
        """Fetch a single row.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Row or None
        """
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> list[sqlite3.Row]:
        """Fetch all rows.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of rows
        """
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        result = self.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return result is not None


# Global database instance (lazy loaded)
_database: Database | None = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _database
    if _database is None:
        _database = Database()
        _database.initialize()
    return _database


def reset_database() -> None:
    """Reset global database instance (useful for testing)."""
    global _database
    if _database is not None:
        _database.close()
    _database = None
