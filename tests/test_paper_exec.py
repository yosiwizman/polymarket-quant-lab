"""Tests for paper execution module.

Phase 6.1: Unit tests for PaperExecutor with mocked dependencies.
Tests cover:
- Paper execution disabled → no trades executed
- Paper execution enabled → signals detected and trades executed
- Risk gate blocks trades when not approved / limits exceeded
- max_trades_per_tick enforced
- Minimum edge filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pmq.models import ArbitrageSignal
from pmq.ops.paper_exec import PaperExecConfig, PaperExecResult, PaperExecutor

# =============================================================================
# Mock Implementations
# =============================================================================


class FakeDAO:
    """Fake DAO for testing paper execution."""

    def __init__(self) -> None:
        self.paper_trades: list[Any] = []
        self.positions: dict[str, dict[str, Any]] = {}
        self.audit_log: list[dict[str, Any]] = []
        self.trades_in_window: int = 0

    def save_paper_trade(self, trade: Any) -> int:
        """Save paper trade and return ID."""
        trade_id = len(self.paper_trades) + 1
        trade.id = trade_id
        self.paper_trades.append(trade)
        return trade_id

    def get_position(self, market_id: str) -> Any:
        """Get position by market ID."""
        return self.positions.get(market_id)

    def upsert_position(self, position: Any) -> None:
        """Upsert position."""
        self.positions[position.market_id] = position

    def get_all_positions(self) -> list[Any]:
        """Get all positions."""
        return list(self.positions.values())

    def log_audit(
        self,
        event_type: str,
        market_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log audit event."""
        self.audit_log.append(
            {"event_type": event_type, "market_id": market_id, "details": details}
        )

    def count_positions(self) -> int:
        """Count positions with holdings."""
        return len(
            [
                p
                for p in self.positions.values()
                if (getattr(p, "yes_quantity", 0) > 0 or getattr(p, "no_quantity", 0) > 0)
            ]
        )

    def count_trades_in_window(self, hours: int = 1) -> int:  # noqa: ARG002
        """Count trades in time window."""
        return self.trades_in_window

    def get_paper_trades(
        self,
        strategy: str | None = None,
        limit: int = 100,  # noqa: ARG002
    ) -> list[Any]:
        """Get paper trades."""
        return self.paper_trades[:limit]

    def get_trading_stats(self) -> dict[str, Any]:
        """Get trading stats."""
        return {
            "total_trades": len(self.paper_trades),
            "total_notional": sum(getattr(t, "notional", 0) for t in self.paper_trades),
            "unique_markets": len({getattr(t, "market_id", "") for t in self.paper_trades}),
            "open_positions": self.count_positions(),
            "total_signals": 0,
        }


class FakeArbitrageScanner:
    """Fake arbitrage scanner for testing."""

    def __init__(self, signals: list[ArbitrageSignal] | None = None) -> None:
        self._signals = signals or []
        self._config = FakeScannerConfig()

    def scan_from_db(
        self,
        markets_data: list[dict[str, Any]],
        top_n: int | None = None,  # noqa: ARG002
    ) -> list[ArbitrageSignal]:
        """Return preconfigured signals."""
        if top_n is not None:
            return self._signals[:top_n]
        return self._signals


@dataclass
class FakeScannerConfig:
    """Fake scanner config."""

    threshold: float = 0.99
    min_liquidity: float = 100.0


class FakeRiskGate:
    """Fake risk gate for testing."""

    def __init__(
        self,
        approved: bool = True,
        trade_allowed: bool = True,
        block_reason: str = "Blocked",
    ) -> None:
        self._approved = approved
        self._trade_allowed = trade_allowed
        self._block_reason = block_reason
        self._trades_recorded = 0
        self._current_limits = None

    def check_approval(self, strategy_name: str) -> Any:
        """Check if strategy is approved."""

        @dataclass
        class ApprovalStatus:
            approved: bool
            strategy_name: str
            strategy_version: str | None
            approval_id: int | None
            limits: Any
            reason: str

        return ApprovalStatus(
            approved=self._approved,
            strategy_name=strategy_name,
            strategy_version="1.0",
            approval_id=1 if self._approved else None,
            limits=None,
            reason="Approved" if self._approved else "Not approved",
        )

    def check_trade_limit(
        self,
        market_id: str,  # noqa: ARG002
        notional: float,  # noqa: ARG002
        current_positions: int,  # noqa: ARG002
        current_total_notional: float,  # noqa: ARG002
    ) -> tuple[bool, str]:
        """Check if trade is within limits."""
        if self._trade_allowed:
            return True, "OK"
        return False, self._block_reason

    def record_trade(self) -> None:
        """Record trade execution."""
        self._trades_recorded += 1


class FakePaperLedger:
    """Fake paper ledger for testing."""

    def __init__(self, dao: FakeDAO | None = None) -> None:
        self._dao = dao or FakeDAO()
        self._positions: list[Any] = []
        self._trades: list[Any] = []
        self._pnl = {
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "position_count": 0,
            "positions": [],
        }
        self.execute_called = 0
        self.should_raise_safety = False

    def execute_arb_trade(self, signal: ArbitrageSignal, quantity: float = 10.0) -> tuple[Any, Any]:
        """Execute arb trade and return fake trades."""
        from pmq.strategies.paper import SafetyError

        if self.should_raise_safety:
            raise SafetyError("Safety blocked")

        self.execute_called += 1

        @dataclass
        class FakeTrade:
            id: int
            strategy: str
            market_id: str
            outcome: str
            price: float
            quantity: float
            notional: float

        yes_trade = FakeTrade(
            id=len(self._trades) + 1,
            strategy="arb",
            market_id=signal.market_id,
            outcome="YES",
            price=signal.yes_price,
            quantity=quantity,
            notional=signal.yes_price * quantity,
        )
        no_trade = FakeTrade(
            id=len(self._trades) + 2,
            strategy="arb",
            market_id=signal.market_id,
            outcome="NO",
            price=signal.no_price,
            quantity=quantity,
            notional=signal.no_price * quantity,
        )
        self._trades.extend([yes_trade, no_trade])
        return yes_trade, no_trade

    def get_all_positions(self) -> list[Any]:
        """Get all positions."""
        return self._positions

    def get_trades(self, limit: int = 100) -> list[Any]:
        """Get trades."""
        return self._trades[:limit]

    def calculate_pnl(
        self,
        markets_data: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Calculate PnL."""
        return self._pnl

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return {
            "total_trades": len(self._trades),
            "total_notional": sum(t.notional for t in self._trades),
            "unique_markets": len({t.market_id for t in self._trades}),
            "open_positions": len(self._positions),
            "total_signals": 0,
        }


def make_signal(
    market_id: str,
    yes_price: float,
    no_price: float,
    liquidity: float = 1000.0,
) -> ArbitrageSignal:
    """Create a test arbitrage signal."""
    return ArbitrageSignal(
        market_id=market_id,
        market_question=f"Test market {market_id}",
        yes_price=yes_price,
        no_price=no_price,
        combined_price=yes_price + no_price,
        profit_potential=1.0 - (yes_price + no_price),
        liquidity=liquidity,
        detected_at=datetime.now(UTC),
    )


# =============================================================================
# Tests: PaperExecConfig
# =============================================================================


class TestPaperExecConfig:
    """Tests for PaperExecConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have safe defaults."""
        config = PaperExecConfig()
        assert config.enabled is False
        assert config.max_trades_per_tick == 3
        assert config.max_markets_scanned == 200
        assert config.min_signal_edge_bps == 50.0
        assert config.require_approval is True
        assert config.trade_quantity == 10.0

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = PaperExecConfig(
            enabled=True,
            max_trades_per_tick=5,
            min_signal_edge_bps=100.0,
        )
        assert config.enabled is True
        assert config.max_trades_per_tick == 5
        assert config.min_signal_edge_bps == 100.0


# =============================================================================
# Tests: PaperExecResult
# =============================================================================


class TestPaperExecResult:
    """Tests for PaperExecResult dataclass."""

    def test_default_values(self) -> None:
        """Result should initialize with zeros."""
        result = PaperExecResult()
        assert result.signals_found == 0
        assert result.trades_executed == 0
        assert result.blocked_by_risk == 0
        assert result.errors == 0
        assert result.total_pnl == 0.0
        assert result.executed_signals == []
        assert result.blocked_reasons == []


# =============================================================================
# Tests: PaperExecutor - Disabled Mode
# =============================================================================


class TestPaperExecutorDisabled:
    """Tests for PaperExecutor when disabled."""

    def test_disabled_returns_pnl_only(self) -> None:
        """When disabled, should only return PnL snapshot."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Set some fake PnL
        ledger._pnl = {
            "total_realized_pnl": 10.0,
            "total_unrealized_pnl": 5.0,
            "total_pnl": 15.0,
            "position_count": 2,
            "positions": [],
        }

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
        )

        markets_data = [{"id": "m1", "last_price_yes": 0.4, "last_price_no": 0.5, "active": True}]

        result = executor.execute_tick(markets_data)

        # Should return PnL but no trades
        assert result.signals_found == 0
        assert result.trades_executed == 0
        assert result.total_pnl == 15.0
        assert result.position_count == 2
        assert ledger.execute_called == 0

    def test_disabled_no_scanner_called(self) -> None:
        """When disabled, should not call scanner."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()

        # Scanner that would return signals
        scanner = FakeArbitrageScanner(
            signals=[make_signal("m1", 0.4, 0.5)]  # 10% edge
        )

        executor = PaperExecutor(
            config=config,
            dao=dao,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": "m1", "active": True}])

        assert result.signals_found == 0  # Scanner wasn't called


# =============================================================================
# Tests: PaperExecutor - Enabled Mode
# =============================================================================


class TestPaperExecutorEnabled:
    """Tests for PaperExecutor when enabled."""

    def test_enabled_executes_signals(self) -> None:
        """When enabled, should execute signals with sufficient edge."""
        config = PaperExecConfig(
            enabled=True,
            min_signal_edge_bps=50.0,  # 0.5%
            require_approval=False,  # Skip approval check
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Signal with 10% edge (1000 bps) - should be executed
        signal = make_signal("m1", 0.4, 0.5)  # combined=0.9, edge=10%
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        markets_data = [
            {
                "id": "m1",
                "last_price_yes": 0.4,
                "last_price_no": 0.5,
                "active": True,
                "closed": False,
                "liquidity": 1000.0,
            }
        ]

        result = executor.execute_tick(markets_data)

        assert result.signals_found == 1
        assert result.signals_evaluated == 1
        assert result.trades_executed == 2  # YES + NO trade
        assert ledger.execute_called == 1

    def test_edge_filter(self) -> None:
        """Signals below min edge should be filtered out."""
        config = PaperExecConfig(
            enabled=True,
            min_signal_edge_bps=200.0,  # 2% minimum edge
            require_approval=False,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Signal with only 1% edge (100 bps) - should be filtered
        signal = make_signal("m1", 0.495, 0.495)  # combined=0.99, edge=1%
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": "m1", "active": True}])

        assert result.signals_found == 1
        assert result.signals_evaluated == 0  # Filtered by edge
        assert result.trades_executed == 0
        assert ledger.execute_called == 0

    def test_max_trades_per_tick(self) -> None:
        """Should respect max_trades_per_tick limit."""
        config = PaperExecConfig(
            enabled=True,
            max_trades_per_tick=2,  # Only 2 trades per tick
            min_signal_edge_bps=50.0,
            require_approval=False,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Multiple signals with good edge
        signals = [
            make_signal("m1", 0.4, 0.5),
            make_signal("m2", 0.35, 0.55),
            make_signal("m3", 0.3, 0.6),
            make_signal("m4", 0.45, 0.45),
        ]
        scanner = FakeArbitrageScanner(signals=signals)

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": f"m{i}", "active": True} for i in range(4)])

        # Should only execute 2 trades (max_trades_per_tick)
        assert result.signals_found == 4
        assert ledger.execute_called == 2
        assert result.trades_executed == 4  # 2 arb trades = 4 individual trades


# =============================================================================
# Tests: PaperExecutor - Risk Gate
# =============================================================================


class TestPaperExecutorRiskGate:
    """Tests for PaperExecutor with risk gate."""

    def test_approval_required_blocks_when_not_approved(self) -> None:
        """Should block execution when approval required but not approved."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        risk_gate = FakeRiskGate(approved=False)

        signal = make_signal("m1", 0.4, 0.5)
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": "m1", "active": True}])

        assert result.signals_found == 0  # Blocked before scanning
        assert result.trades_executed == 0
        assert result.blocked_by_risk == 1
        assert "Not approved" in result.blocked_reasons[0]

    def test_trade_limit_blocks_individual_trades(self) -> None:
        """Should block individual trades when risk limit exceeded."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        risk_gate = FakeRiskGate(
            approved=True,
            trade_allowed=False,
            block_reason="Position limit reached",
        )

        signal = make_signal("m1", 0.4, 0.5)
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": "m1", "active": True}])

        assert result.signals_found == 1
        assert result.signals_evaluated == 1
        assert result.trades_executed == 0  # All blocked
        assert result.blocked_by_risk == 1
        assert "Position limit" in result.blocked_reasons[0]


# =============================================================================
# Tests: PaperExecutor - Safety Guard
# =============================================================================


class TestPaperExecutorSafety:
    """Tests for PaperExecutor with SafetyGuard."""

    def test_safety_error_blocks_trade(self) -> None:
        """SafetyError from ledger should be caught and recorded."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        ledger.should_raise_safety = True  # Make ledger raise SafetyError

        signal = make_signal("m1", 0.4, 0.5)
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        result = executor.execute_tick([{"id": "m1", "active": True}])

        assert result.signals_found == 1
        assert result.signals_evaluated == 1
        assert result.trades_executed == 0
        assert result.blocked_by_safety == 1
        assert "Safety" in result.blocked_reasons[0]


# =============================================================================
# Tests: PaperExecutor - Helper Methods
# =============================================================================


class TestPaperExecutorHelpers:
    """Tests for PaperExecutor helper methods."""

    def test_get_pnl_snapshot(self) -> None:
        """get_pnl_snapshot should return ledger PnL."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        ledger._pnl = {
            "total_realized_pnl": 100.0,
            "total_unrealized_pnl": 50.0,
            "total_pnl": 150.0,
            "position_count": 5,
            "positions": [],
        }

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
        )

        pnl = executor.get_pnl_snapshot()
        assert pnl["total_pnl"] == 150.0
        assert pnl["position_count"] == 5

    def test_get_positions(self) -> None:
        """get_positions should return ledger positions."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        @dataclass
        class FakePosition:
            market_id: str
            yes_quantity: float
            no_quantity: float

        ledger._positions = [
            FakePosition("m1", 10.0, 5.0),
            FakePosition("m2", 20.0, 0.0),
        ]

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
        )

        positions = executor.get_positions()
        assert len(positions) == 2

    def test_get_recent_trades(self) -> None:
        """get_recent_trades should return ledger trades."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
        )

        # Execute a trade first
        config.enabled = True
        config.require_approval = False
        executor.config = config

        signal = make_signal("m1", 0.4, 0.5)
        scanner = FakeArbitrageScanner(signals=[signal])
        executor._arb_scanner = scanner

        executor.execute_tick([{"id": "m1", "active": True}])

        trades = executor.get_recent_trades(limit=10)
        assert len(trades) == 2  # YES + NO trade

    def test_get_stats(self) -> None:
        """get_stats should return ledger stats."""
        config = PaperExecConfig(enabled=False)
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
        )

        stats = executor.get_stats()
        assert "total_trades" in stats
        assert "total_notional" in stats
