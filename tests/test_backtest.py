"""Tests for backtesting framework.

Tests cover:
- Snapshot storage correctness
- Deterministic replay (same input → same output)
- Metrics correctness
- BacktestEngine position management
"""

import tempfile
from pathlib import Path

import pytest

from pmq.backtest.engine import BacktestEngine, BacktestPosition
from pmq.backtest.metrics import BacktestMetrics, MetricsCalculator
from pmq.backtest.runner import BacktestRunner
from pmq.models import GammaMarket, Outcome, Side
from pmq.storage.dao import DAO
from pmq.storage.db import Database


def create_test_market(
    market_id: str,
    yes_price: float = 0.5,
    no_price: float = 0.5,
    liquidity: float = 1000.0,
) -> GammaMarket:
    """Create a test market for fixtures."""
    return GammaMarket(
        id=market_id,
        question=f"Test market {market_id}?",
        slug=market_id,
        active=True,
        closed=False,
        outcome_prices=f"[{yes_price}, {no_price}]",
        liquidity=liquidity,
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


class TestBacktestPosition:
    """Tests for BacktestPosition."""

    def test_update_buy_yes(self):
        """Test buying YES updates position correctly."""
        pos = BacktestPosition(market_id="test")
        pos.update_buy(Outcome.YES, price=0.50, quantity=10.0)

        assert pos.yes_quantity == 10.0
        assert pos.avg_price_yes == 0.50
        assert pos.no_quantity == 0.0

    def test_update_buy_no(self):
        """Test buying NO updates position correctly."""
        pos = BacktestPosition(market_id="test")
        pos.update_buy(Outcome.NO, price=0.40, quantity=20.0)

        assert pos.no_quantity == 20.0
        assert pos.avg_price_no == 0.40
        assert pos.yes_quantity == 0.0

    def test_update_buy_multiple(self):
        """Test multiple buys update average price correctly."""
        pos = BacktestPosition(market_id="test")
        pos.update_buy(Outcome.YES, price=0.50, quantity=10.0)  # Cost: 5.0
        pos.update_buy(Outcome.YES, price=0.60, quantity=10.0)  # Cost: 6.0

        assert pos.yes_quantity == 20.0
        # Average: (0.50*10 + 0.60*10) / 20 = 0.55
        assert pos.avg_price_yes == pytest.approx(0.55, abs=0.001)

    def test_update_sell_realized_pnl(self):
        """Test selling realizes PnL correctly."""
        pos = BacktestPosition(market_id="test")
        pos.update_buy(Outcome.YES, price=0.50, quantity=10.0)
        pos.update_sell(Outcome.YES, price=0.60, quantity=5.0)

        assert pos.yes_quantity == 5.0
        # PnL: (0.60 - 0.50) * 5 = 0.50
        assert pos.realized_pnl == pytest.approx(0.50, abs=0.001)

    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        pos = BacktestPosition(market_id="test")
        pos.update_buy(Outcome.YES, price=0.50, quantity=10.0)
        pos.update_buy(Outcome.NO, price=0.40, quantity=10.0)

        # Current prices: YES=0.60, NO=0.35
        # Unrealized YES: (0.60 - 0.50) * 10 = 1.0
        # Unrealized NO: (0.35 - 0.40) * 10 = -0.5
        # Total: 0.5
        unrealized = pos.unrealized_pnl(0.60, 0.35)
        assert unrealized == pytest.approx(0.50, abs=0.001)


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initial_state(self):
        """Test engine initializes with correct state."""
        engine = BacktestEngine(initial_balance=10000.0)

        assert engine.initial_balance == 10000.0
        assert engine.balance == 10000.0
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0

    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        engine = BacktestEngine(initial_balance=1000.0)
        engine.set_time("2024-01-01T00:00:00")

        trade = engine.execute_trade(
            market_id="market1",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=10.0,
        )

        assert trade is not None
        assert trade.market_id == "market1"
        assert trade.notional == 5.0
        assert engine.balance == 995.0  # 1000 - 5

    def test_execute_trade_insufficient_funds(self):
        """Test trade fails with insufficient funds."""
        engine = BacktestEngine(initial_balance=100.0)

        trade = engine.execute_trade(
            market_id="market1",
            side=Side.BUY,
            outcome=Outcome.YES,
            price=0.50,
            quantity=500.0,  # Would cost 250, but only have 100
        )

        assert trade is None
        assert engine.balance == 100.0  # Unchanged

    def test_execute_arb_trade(self):
        """Test executing arbitrage trade."""
        engine = BacktestEngine(initial_balance=1000.0)
        engine.set_time("2024-01-01T00:00:00")

        yes_trade, no_trade = engine.execute_arb_trade(
            market_id="arb_market",
            yes_price=0.45,
            no_price=0.45,
            quantity=10.0,
        )

        assert yes_trade is not None
        assert no_trade is not None
        # Cost: (0.45 + 0.45) * 10 = 9.0
        assert engine.balance == pytest.approx(991.0, abs=0.01)

    def test_deterministic_replay(self):
        """Test that same inputs produce same outputs."""
        # Run 1
        engine1 = BacktestEngine(initial_balance=1000.0)
        engine1.set_time("2024-01-01T00:00:00")
        engine1.execute_trade("m1", Side.BUY, Outcome.YES, 0.50, 10.0)
        engine1.execute_trade("m1", Side.BUY, Outcome.NO, 0.45, 10.0)

        # Run 2 - same operations
        engine2 = BacktestEngine(initial_balance=1000.0)
        engine2.set_time("2024-01-01T00:00:00")
        engine2.execute_trade("m1", Side.BUY, Outcome.YES, 0.50, 10.0)
        engine2.execute_trade("m1", Side.BUY, Outcome.NO, 0.45, 10.0)

        # Results should be identical
        assert engine1.balance == engine2.balance
        assert len(engine1.trades) == len(engine2.trades)
        assert engine1.positions["m1"].yes_quantity == engine2.positions["m1"].yes_quantity

    def test_reset(self):
        """Test engine reset."""
        engine = BacktestEngine(initial_balance=1000.0)
        engine.execute_trade("m1", Side.BUY, Outcome.YES, 0.50, 10.0)
        engine.reset()

        assert engine.balance == 1000.0
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_drawdown_no_drawdown(self):
        """Test drawdown when equity only increases."""
        calc = MetricsCalculator()

        equity_curve = [
            ("2024-01-01T00:00:00", 1000.0),
            ("2024-01-01T01:00:00", 1050.0),
            ("2024-01-01T02:00:00", 1100.0),
        ]

        dd, peak, low = calc._calculate_drawdown(equity_curve)

        assert dd == 0.0
        assert peak == 1100.0
        assert low == 1000.0

    def test_calculate_drawdown_with_drawdown(self):
        """Test drawdown calculation with actual drawdown."""
        calc = MetricsCalculator()

        equity_curve = [
            ("2024-01-01T00:00:00", 1000.0),
            ("2024-01-01T01:00:00", 1100.0),  # Peak
            ("2024-01-01T02:00:00", 990.0),   # Drawdown
            ("2024-01-01T03:00:00", 1050.0),
        ]

        dd, peak, low = calc._calculate_drawdown(equity_curve)

        # Max drawdown: (1100 - 990) / 1100 = 0.10
        assert dd == pytest.approx(0.10, abs=0.001)
        assert peak == 1100.0
        assert low == 990.0

    def test_calculate_trades_per_day(self):
        """Test trades per day calculation."""
        calc = MetricsCalculator()

        trades_per_day = calc._calculate_trades_per_day(
            "2024-01-01", "2024-01-08", total_trades=14
        )

        # 7 days, 14 trades = 2 trades/day
        assert trades_per_day == 2.0

    def test_calculate_full_metrics(self):
        """Test full metrics calculation."""
        engine = BacktestEngine(initial_balance=1000.0)
        engine.set_time("2024-01-01T00:00:00")

        # Execute some trades
        engine.execute_trade("m1", Side.BUY, Outcome.YES, 0.45, 10.0)
        engine.execute_trade("m1", Side.BUY, Outcome.NO, 0.45, 10.0)

        # Record equity
        engine.record_equity({"m1": {"yes_price": 0.50, "no_price": 0.50}})
        engine.set_time("2024-01-02T00:00:00")
        engine.record_equity({"m1": {"yes_price": 0.55, "no_price": 0.45}})

        calc = MetricsCalculator()
        metrics = calc.calculate(engine, "2024-01-01", "2024-01-02")

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_trades == 2
        assert metrics.final_balance < 1000.0  # Spent on trades


class TestSnapshotStorage:
    """Tests for snapshot storage in DAO."""

    def test_save_and_get_snapshot(self, dao):
        """Test saving and retrieving snapshots."""
        # Insert market first
        dao.upsert_market(create_test_market("snap_market"))

        snapshot_time = "2024-01-01T12:00:00"
        dao.save_snapshot(
            market_id="snap_market",
            yes_price=0.55,
            no_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
            snapshot_time=snapshot_time,
        )

        snapshots = dao.get_snapshots("2024-01-01T00:00:00", "2024-01-02T00:00:00")

        assert len(snapshots) == 1
        assert snapshots[0]["market_id"] == "snap_market"
        assert snapshots[0]["yes_price"] == 0.55
        assert snapshots[0]["no_price"] == 0.45

    def test_save_snapshots_bulk(self, dao):
        """Test bulk snapshot saving."""
        markets = [
            create_test_market("m1", yes_price=0.50, no_price=0.50),
            create_test_market("m2", yes_price=0.60, no_price=0.40),
        ]

        for m in markets:
            dao.upsert_market(m)

        snapshot_time = "2024-01-01T12:00:00"
        count = dao.save_snapshots_bulk(markets, snapshot_time)

        assert count == 2

        snapshots = dao.get_snapshots("2024-01-01T00:00:00", "2024-01-02T00:00:00")
        assert len(snapshots) == 2

    def test_get_snapshot_times(self, dao):
        """Test getting distinct snapshot times."""
        dao.upsert_market(create_test_market("m1"))

        # Save snapshots at different times
        dao.save_snapshot("m1", 0.50, 0.50, 1000.0, 5000.0, "2024-01-01T12:00:00")
        dao.save_snapshot("m1", 0.55, 0.45, 1000.0, 5000.0, "2024-01-01T13:00:00")
        dao.save_snapshot("m1", 0.60, 0.40, 1000.0, 5000.0, "2024-01-01T14:00:00")

        times = dao.get_snapshot_times("2024-01-01T00:00:00", "2024-01-02T00:00:00")

        assert len(times) == 3
        assert times == [
            "2024-01-01T12:00:00",
            "2024-01-01T13:00:00",
            "2024-01-01T14:00:00",
        ]


class TestBacktestRunner:
    """Tests for BacktestRunner."""

    def test_run_arb_backtest_no_snapshots(self, dao):
        """Test backtest with no snapshots returns empty metrics."""
        runner = BacktestRunner(dao=dao, initial_balance=10000.0)

        run_id, metrics = runner.run_arb_backtest(
            start_date="2024-01-01",
            end_date="2024-01-07",
        )

        assert run_id is not None
        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0

    def test_run_arb_backtest_with_snapshots(self, dao):
        """Test backtest with snapshots produces trades."""
        # Create market and snapshots
        dao.upsert_market(create_test_market("arb_m", yes_price=0.45, no_price=0.45, liquidity=500.0))

        # Save snapshots with arb opportunity (YES+NO < 0.99)
        dao.save_snapshot("arb_m", 0.45, 0.45, 500.0, 5000.0, "2024-01-01T12:00:00")
        dao.save_snapshot("arb_m", 0.46, 0.44, 500.0, 5000.0, "2024-01-01T13:00:00")

        runner = BacktestRunner(dao=dao, initial_balance=10000.0)

        run_id, metrics = runner.run_arb_backtest(
            start_date="2024-01-01",
            end_date="2024-01-02",
            quantity=10.0,
        )

        assert run_id is not None
        assert metrics.total_trades > 0  # Should have executed trades

    def test_deterministic_backtest_results(self, dao):
        """Test that running same backtest twice produces identical results."""
        # Setup
        dao.upsert_market(create_test_market("det_m", yes_price=0.45, no_price=0.45, liquidity=500.0))
        dao.save_snapshot("det_m", 0.45, 0.45, 500.0, 5000.0, "2024-01-01T12:00:00")

        # Run 1
        runner1 = BacktestRunner(dao=dao, initial_balance=10000.0)
        run_id1, metrics1 = runner1.run_arb_backtest(
            start_date="2024-01-01",
            end_date="2024-01-02",
            quantity=10.0,
        )

        # Run 2 - same parameters
        runner2 = BacktestRunner(dao=dao, initial_balance=10000.0)
        run_id2, metrics2 = runner2.run_arb_backtest(
            start_date="2024-01-01",
            end_date="2024-01-02",
            quantity=10.0,
        )

        # Results should be identical (except run_id)
        assert run_id1 != run_id2  # Different run IDs
        assert metrics1.total_trades == metrics2.total_trades
        assert metrics1.total_pnl == metrics2.total_pnl
        assert metrics1.final_balance == metrics2.final_balance

    def test_get_run_report(self, dao):
        """Test getting backtest run report."""
        dao.upsert_market(create_test_market("rep_m", yes_price=0.45, no_price=0.45, liquidity=500.0))
        dao.save_snapshot("rep_m", 0.45, 0.45, 500.0, 5000.0, "2024-01-01T12:00:00")

        runner = BacktestRunner(dao=dao, initial_balance=10000.0)
        run_id, _ = runner.run_arb_backtest(
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

        report = runner.get_run_report(run_id)

        assert report is not None
        assert "run" in report
        assert "metrics" in report
        assert "trades" in report
        assert report["run"]["strategy"] == "arb"

    def test_list_runs(self, dao):
        """Test listing backtest runs."""
        dao.upsert_market(create_test_market("list_m", yes_price=0.45, no_price=0.45, liquidity=500.0))
        dao.save_snapshot("list_m", 0.45, 0.45, 500.0, 5000.0, "2024-01-01T12:00:00")

        runner = BacktestRunner(dao=dao, initial_balance=10000.0)

        # Run multiple backtests
        runner.run_arb_backtest("2024-01-01", "2024-01-02")
        runner.run_arb_backtest("2024-01-01", "2024-01-02")

        runs = runner.list_runs(limit=10)

        assert len(runs) >= 2
