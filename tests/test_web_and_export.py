"""Tests for web API, export functionality, and new DAO helpers."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pmq.models import GammaMarket, SignalType
from pmq.storage.dao import DAO
from pmq.storage.db import Database
from pmq.web import create_app

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path=db_path)
        db.initialize()
        yield db
        db.close()


@pytest.fixture
def dao(temp_db):
    """Create a DAO with temporary database."""
    return DAO(db=temp_db)


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    app = create_app()
    return TestClient(app)


def create_test_market(market_id: str, question: str = "Test?") -> GammaMarket:
    """Create a test market for fixtures."""
    return GammaMarket(
        id=market_id,
        question=question,
        slug=market_id,
        active=True,
        closed=False,
        liquidity=1000.0,
        volume=5000.0,
        volume24hr=500.0,
    )


# =============================================================================
# Runtime State Tests
# =============================================================================


class TestRuntimeState:
    """Tests for runtime state DAO methods."""

    def test_set_and_get_runtime_state(self, dao):
        """Test setting and getting runtime state."""
        dao.set_runtime_state("test_key", "test_value")
        value = dao.get_runtime_state("test_key")
        assert value == "test_value"

    def test_get_nonexistent_runtime_state(self, dao):
        """Test getting non-existent runtime state returns None."""
        value = dao.get_runtime_state("nonexistent_key")
        assert value is None

    def test_update_runtime_state(self, dao):
        """Test updating existing runtime state."""
        dao.set_runtime_state("key", "value1")
        dao.set_runtime_state("key", "value2")
        value = dao.get_runtime_state("key")
        assert value == "value2"

    def test_get_all_runtime_state(self, dao):
        """Test getting all runtime state values."""
        dao.set_runtime_state("key1", "value1")
        dao.set_runtime_state("key2", "value2")
        all_state = dao.get_all_runtime_state()
        assert all_state == {"key1": "value1", "key2": "value2"}


# =============================================================================
# Export Helpers Tests
# =============================================================================


class TestExportHelpers:
    """Tests for export DAO methods."""

    def test_get_signals_for_export_empty(self, dao):
        """Test export with no signals returns empty list."""
        signals = dao.get_signals_for_export(limit=100)
        assert signals == []

    def test_get_trades_for_export_empty(self, dao):
        """Test export with no trades returns empty list."""
        trades = dao.get_trades_for_export(limit=100)
        assert trades == []

    def test_get_positions_for_export_empty(self, dao):
        """Test export with no positions returns empty list."""
        positions = dao.get_positions_for_export()
        assert positions == []

    def test_get_signals_for_export_with_data(self, dao):
        """Test export signals with data."""
        # Create a signal
        signal_id = dao.save_signal(
            SignalType.ARBITRAGE,
            ["market1"],
            {"key": "value", "profit_potential": 0.05},
            profit_potential=0.05,
        )

        signals = dao.get_signals_for_export(limit=100)
        assert len(signals) == 1
        assert signals[0]["id"] == signal_id
        assert signals[0]["type"] == "ARBITRAGE"

    def test_get_signals_for_export_with_type_filter(self, dao):
        """Test export signals with type filter."""
        dao.save_signal(SignalType.ARBITRAGE, ["m1"], {}, 0.01)
        dao.save_signal(SignalType.STAT_ARB, ["m2", "m3"], {}, 0.02)

        arb_signals = dao.get_signals_for_export(signal_type=SignalType.ARBITRAGE)
        assert len(arb_signals) == 1
        assert arb_signals[0]["type"] == "ARBITRAGE"


# =============================================================================
# Web API Tests
# =============================================================================


class TestWebAPI:
    """Tests for web API endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "time" in data
        assert "version" in data

    def test_get_signals_empty(self, test_client):
        """Test signals endpoint with no data."""
        response = test_client.get("/api/signals")
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "count" in data

    def test_get_signals_with_limit(self, test_client):
        """Test signals endpoint with limit parameter."""
        response = test_client.get("/api/signals?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "signals" in data

    def test_get_trades_empty(self, test_client):
        """Test trades endpoint with no data."""
        response = test_client.get("/api/trades")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data
        assert "count" in data

    def test_get_positions_empty(self, test_client):
        """Test positions endpoint with no data."""
        response = test_client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert "count" in data

    def test_get_summary(self, test_client):
        """Test summary endpoint."""
        response = test_client.get("/api/summary")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "runtime" in data
        assert "timestamp" in data

    def test_dashboard_loads(self, test_client):
        """Test dashboard HTML loads."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Polymarket Quant Lab" in response.content


# =============================================================================
# CSV Export Tests
# =============================================================================


class TestCSVExport:
    """Tests for CSV export functionality."""

    def test_write_csv(self, tmp_path):
        """Test CSV writing helper."""
        from pmq.cli import _write_csv

        data = [
            {"col1": "val1", "col2": 123},
            {"col1": "val2", "col2": 456},
        ]
        filepath = tmp_path / "test.csv"
        _write_csv(filepath, data)

        assert filepath.exists()
        content = filepath.read_text()
        assert "col1,col2" in content
        assert "val1,123" in content
        assert "val2,456" in content

    def test_write_csv_with_nested_data(self, tmp_path):
        """Test CSV writing flattens nested data."""
        from pmq.cli import _write_csv

        data = [{"col1": "val1", "nested": {"key": "value"}, "list_col": [1, 2, 3]}]
        filepath = tmp_path / "test.csv"
        _write_csv(filepath, data)

        assert filepath.exists()
        content = filepath.read_text()
        # Nested data should be stringified
        assert "{'key': 'value'}" in content or '{"key": "value"}' in content

    def test_write_csv_empty(self, tmp_path):
        """Test CSV writing with empty data does nothing."""
        from pmq.cli import _write_csv

        filepath = tmp_path / "test.csv"
        _write_csv(filepath, [])

        assert not filepath.exists()
