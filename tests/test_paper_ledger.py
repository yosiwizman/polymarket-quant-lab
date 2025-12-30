"""Tests for paper trading ledger and safety guardrails."""

import tempfile
from pathlib import Path

import pytest

from pmq.config import SafetyConfig
from pmq.models import ArbitrageSignal, GammaMarket, Outcome, PaperPosition, Side
from pmq.storage.dao import DAO
from pmq.storage.db import Database
from pmq.strategies.paper import PaperLedger, SafetyError, SafetyGuard


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
def safety_config():
    """Safety configuration for testing."""
    return SafetyConfig(
        kill_switch=False,
        max_positions=10,
        max_notional_per_market=100.0,
        max_trades_per_hour=20,
    )


@pytest.fixture
def ledger(dao, safety_config):
    """Create a paper ledger for testing."""
    safety = SafetyGuard(config=safety_config, dao=dao)
    return PaperLedger(dao=dao, safety=safety)


class TestSafetyGuard:
    """Tests for SafetyGuard."""

    def test_kill_switch_inactive(self, dao, safety_config):
        """Test normal operation when kill switch is off."""
        guard = SafetyGuard(config=safety_config, dao=dao)

        # Should not raise
        guard.check_kill_switch()

    def test_kill_switch_active(self, dao):
        """Test that kill switch blocks operations."""
        config = SafetyConfig(kill_switch=True)
        guard = SafetyGuard(config=config, dao=dao)

        with pytest.raises(SafetyError, match="Kill switch"):
            guard.check_kill_switch()

    def test_notional_limit_ok(self, dao, safety_config):
        """Test notional check when within limits."""
        guard = SafetyGuard(config=safety_config, dao=dao)

        # Should not raise for small notional
        guard.check_notional_limit("market1", 50.0)

    def test_notional_limit_exceeded(self, dao, safety_config):
        """Test notional check when limit would be exceeded."""
        guard = SafetyGuard(config=safety_config, dao=dao)

        with pytest.raises(SafetyError, match="Notional limit"):
            guard.check_notional_limit("market1", 150.0)

    def test_rate_limit_ok(self, dao, safety_config):
        """Test rate limit when within limits."""
        guard = SafetyGuard(config=safety_config, dao=dao)

        # Should not raise with no recent trades
        guard.check_rate_limit()

    def test_validate_trade_all_pass(self, dao, safety_config):
        """Test full validation when all checks pass."""
        guard = SafetyGuard(config=safety_config, dao=dao)

        # Should not raise
        guard.validate_trade("market1", 50.0)


class TestPaperLedger:
    """Tests for PaperLedger."""

    def test_execute_trade(self, ledger, dao):
        """Test executing a simple paper trade."""
        # Insert market to satisfy foreign key constraint
        dao.upsert_market(create_test_market("test_market", "Test question?"))

        trade = ledger.execute_trade(
            strategy="test",
            market_id="test_market",
            market_question="Test question?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=10.0,
        )

        assert trade.strategy == "test"
        assert trade.market_id == "test_market"
        assert trade.side == Side.BUY
        assert trade.outcome == Outcome.YES
        assert trade.price == 0.50
        assert trade.quantity == 10.0
        assert trade.notional == 5.0

    def test_position_update_on_buy(self, ledger, dao):
        """Test that position is updated after buy."""
        dao.upsert_market(create_test_market("market1"))

        ledger.execute_trade(
            strategy="test",
            market_id="market1",
            market_question="Test?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=10.0,
        )

        position = ledger.get_position("market1")

        assert position is not None
        assert position.yes_quantity == 10.0
        assert position.avg_price_yes == 0.50
        assert position.no_quantity == 0.0

    def test_position_update_on_sell(self, ledger, dao):
        """Test that position is updated after sell."""
        dao.upsert_market(create_test_market("market1"))

        # First buy
        ledger.execute_trade(
            strategy="test",
            market_id="market1",
            market_question="Test?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=10.0,
        )

        # Then sell at higher price (profit)
        ledger.execute_trade(
            strategy="test",
            market_id="market1",
            market_question="Test?",
            side=Side.SELL,
            outcome=Outcome.YES,
            price=0.60,
            quantity=5.0,
        )

        position = ledger.get_position("market1")

        assert position is not None
        assert position.yes_quantity == 5.0
        assert position.realized_pnl == pytest.approx(0.50, abs=0.01)  # (0.60-0.50)*5

    def test_execute_arb_trade(self, ledger, dao):
        """Test executing an arbitrage trade."""
        dao.upsert_market(create_test_market("arb_market", "Arb test?"))

        signal = ArbitrageSignal(
            market_id="arb_market",
            market_question="Arb test?",
            yes_price=0.45,
            no_price=0.45,
            combined_price=0.90,
            profit_potential=0.10,
            liquidity=1000.0,
        )

        yes_trade, no_trade = ledger.execute_arb_trade(signal, quantity=10.0)

        assert yes_trade.outcome == Outcome.YES
        assert yes_trade.price == 0.45
        assert no_trade.outcome == Outcome.NO
        assert no_trade.price == 0.45

        position = ledger.get_position("arb_market")
        assert position.yes_quantity == 10.0
        assert position.no_quantity == 10.0

    def test_get_all_positions(self, ledger, dao):
        """Test getting all positions."""
        dao.upsert_market(create_test_market("m1", "M1?"))
        dao.upsert_market(create_test_market("m2", "M2?"))

        # Create positions in multiple markets
        ledger.execute_trade(
            strategy="test",
            market_id="m1",
            market_question="M1?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=5.0,
        )
        ledger.execute_trade(
            strategy="test",
            market_id="m2",
            market_question="M2?",
            side=Side.BUY,
            outcome=Outcome.NO,
            price=0.40,
            quantity=10.0,
        )

        positions = ledger.get_all_positions()

        assert len(positions) == 2

    def test_get_trades(self, ledger, dao):
        """Test getting trade history."""
        for i in range(3):
            dao.upsert_market(create_test_market(f"m{i}", f"Q{i}?"))
            ledger.execute_trade(
                strategy="test",
                market_id=f"m{i}",
                market_question=f"Q{i}?",
                side=Side.BUY,
                outcome=Outcome.YES,
                price=0.50,
                quantity=1.0,
            )

        trades = ledger.get_trades(limit=10)
        assert len(trades) == 3

        # Test filtering by strategy
        trades = ledger.get_trades(strategy="test")
        assert len(trades) == 3

        trades = ledger.get_trades(strategy="other")
        assert len(trades) == 0

    def test_calculate_pnl(self, ledger, dao):
        """Test PnL calculation."""
        dao.upsert_market(create_test_market("m1", "Q1?"))

        # Buy and hold
        ledger.execute_trade(
            strategy="test",
            market_id="m1",
            market_question="Q1?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.40,
            quantity=10.0,
        )

        pnl = ledger.calculate_pnl()

        assert pnl["position_count"] == 1
        assert "total_pnl" in pnl
        assert "positions" in pnl

    def test_get_stats(self, ledger, dao):
        """Test getting trading statistics."""
        dao.upsert_market(create_test_market("m1", "Q1?"))

        ledger.execute_trade(
            strategy="test",
            market_id="m1",
            market_question="Q1?",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=10.0,
        )

        stats = ledger.get_stats()

        assert stats["total_trades"] == 1
        assert stats["total_notional"] == 5.0
        assert stats["unique_markets"] == 1


class TestPaperPosition:
    """Tests for PaperPosition model."""

    def test_has_position_true(self):
        """Test has_position when there's a position."""
        pos = PaperPosition(
            market_id="test",
            yes_quantity=10.0,
            no_quantity=0.0,
        )
        assert pos.has_position is True

    def test_has_position_false(self):
        """Test has_position when empty."""
        pos = PaperPosition(
            market_id="test",
            yes_quantity=0.0,
            no_quantity=0.0,
        )
        assert pos.has_position is False

    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        pos = PaperPosition(
            market_id="test",
            yes_quantity=10.0,
            no_quantity=5.0,
            avg_price_yes=0.40,
            avg_price_no=0.30,
        )

        # Current prices: YES=0.50, NO=0.35
        unrealized = pos.unrealized_pnl(0.50, 0.35)

        # YES: (0.50 - 0.40) * 10 = 1.0
        # NO: (0.35 - 0.30) * 5 = 0.25
        assert unrealized == pytest.approx(1.25, abs=0.01)

    def test_total_pnl(self):
        """Test total PnL calculation."""
        pos = PaperPosition(
            market_id="test",
            yes_quantity=10.0,
            no_quantity=0.0,
            avg_price_yes=0.40,
            realized_pnl=0.50,
        )

        total = pos.total_pnl(0.50, 0.50)

        # Realized: 0.50
        # Unrealized YES: (0.50 - 0.40) * 10 = 1.0
        assert total == pytest.approx(1.50, abs=0.01)
