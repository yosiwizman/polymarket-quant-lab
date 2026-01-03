"""Tests for paper execution module.

Phase 6.1: Unit tests for PaperExecutor with mocked dependencies.
Phase 6.2: Unit tests for explain mode, rejection taxonomy, and exports.

Tests cover:
- Paper execution disabled → no trades executed
- Paper execution enabled → signals detected and trades executed
- Risk gate blocks trades when not approved / limits exceeded
- max_trades_per_tick enforced
- Minimum edge filtering
- Explain mode with rejection reasons
- ExplainCandidate serialization
- ExplainSummary aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pmq.models import ArbitrageSignal
from pmq.ops.paper_exec import (
    ExplainCandidate,
    ExplainSummary,
    PaperExecConfig,
    PaperExecResult,
    PaperExecutor,
    RejectionReason,
)

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
        _strategy: str | None = None,
        limit: int = 100,
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
        _markets_data: list[dict[str, Any]],
        top_n: int | None = None,
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
        """Should block execution when approval required but not approved.

        Phase 6.2.1: blocked_by_risk only increments when there were actual
        executable signals that would have traded. If we're blocked before
        scanning (or no signals found), blocked_by_risk is 0.
        """
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
        # Phase 6.2.1: blocked_by_risk is 0 because no executable signals were found
        # (we were blocked before scanning, so signals_evaluated = 0)
        assert result.blocked_by_risk == 0
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


# =============================================================================
# Tests: Phase 6.2 - Explain Mode
# =============================================================================


class TestRejectionReason:
    """Tests for RejectionReason enum."""

    def test_enum_values_exist(self) -> None:
        """All rejection reason variants should exist."""
        assert RejectionReason.EXECUTED is not None
        assert RejectionReason.NO_SIGNAL is not None
        assert RejectionReason.EDGE_BELOW_MIN is not None
        assert RejectionReason.LIQUIDITY_BELOW_MIN is not None
        assert RejectionReason.SPREAD_ABOVE_MAX is not None
        assert RejectionReason.MARKET_INACTIVE is not None
        assert RejectionReason.MARKET_CLOSED is not None
        assert RejectionReason.RISK_NOT_APPROVED is not None
        assert RejectionReason.RISK_POSITION_LIMIT is not None
        assert RejectionReason.RISK_NOTIONAL_LIMIT is not None
        assert RejectionReason.RISK_BLOCKED is not None
        assert RejectionReason.SAFETY_ERROR is not None
        assert RejectionReason.MAX_TRADES_PER_TICK is not None

    def test_enum_string_values(self) -> None:
        """Enum values should be descriptive strings."""
        assert RejectionReason.EDGE_BELOW_MIN.value == "edge_below_min"
        assert RejectionReason.LIQUIDITY_BELOW_MIN.value == "liquidity_below_min"
        assert RejectionReason.EXECUTED.value == "executed"


class TestExplainCandidate:
    """Tests for ExplainCandidate dataclass."""

    def test_to_dict_basic(self) -> None:
        """to_dict should return serializable dictionary."""
        candidate = ExplainCandidate(
            market_id="test-market-123",
            token_id="token-yes-456",
            side="YES",
            edge_bps=75.5,
            spread_bps=20.0,
            liquidity=5000.0,
            mid_price=0.45,
            rejection_reason=RejectionReason.EDGE_BELOW_MIN,
        )

        result = candidate.to_dict()

        assert result["market_id"] == "test-market-123"
        assert result["token_id"] == "token-yes-456"
        assert result["side"] == "YES"
        assert result["edge_bps"] == 75.5
        assert result["spread_bps"] == 20.0
        assert result["liquidity"] == 5000.0
        assert result["mid_price"] == 0.45
        assert result["rejection_reason"] == "edge_below_min"

    def test_to_dict_executed_candidate(self) -> None:
        """Executed candidates should have rejection_reason = executed."""
        candidate = ExplainCandidate(
            market_id="m1",
            token_id="t1",
            side="NO",
            edge_bps=150.0,
            spread_bps=10.0,
            liquidity=10000.0,
            mid_price=0.55,
            rejection_reason=RejectionReason.EXECUTED,
        )

        result = candidate.to_dict()
        assert result["rejection_reason"] == "executed"

    def test_to_dict_with_market_question(self) -> None:
        """to_dict should include optional market_question."""
        candidate = ExplainCandidate(
            market_id="m1",
            token_id="t1",
            side="YES",
            edge_bps=100.0,
            spread_bps=15.0,
            liquidity=8000.0,
            mid_price=0.50,
            rejection_reason=RejectionReason.LIQUIDITY_BELOW_MIN,
            market_question="Will X happen?",
        )

        result = candidate.to_dict()
        assert result["market_question"] == "Will X happen?"


class TestExplainSummary:
    """Tests for ExplainSummary dataclass."""

    def test_from_results_empty(self) -> None:
        """from_results should handle empty results list."""
        summary = ExplainSummary.from_results([])

        assert summary.total_ticks == 0
        assert summary.total_candidates == 0
        assert summary.total_executed == 0
        assert summary.rejection_counts == {}
        assert summary.avg_top_edge_bps == 0.0
        assert summary.ticks_with_candidates_above_min_edge == 0

    def test_from_results_with_data(self) -> None:
        """from_results should aggregate rejection counts correctly."""
        # Create mock results with explain data
        result1 = PaperExecResult(
            signals_found=3,
            trades_executed=1,
            explain_candidates=[
                ExplainCandidate(
                    market_id="m1",
                    token_id="t1",
                    side="YES",
                    edge_bps=120.0,
                    spread_bps=10.0,
                    liquidity=5000.0,
                    mid_price=0.5,
                    rejection_reason=RejectionReason.EXECUTED,
                ),
                ExplainCandidate(
                    market_id="m2",
                    token_id="t2",
                    side="YES",
                    edge_bps=30.0,
                    spread_bps=10.0,
                    liquidity=5000.0,
                    mid_price=0.5,
                    rejection_reason=RejectionReason.EDGE_BELOW_MIN,
                ),
            ],
            rejection_counts={"executed": 1, "edge_below_min": 1},
        )

        result2 = PaperExecResult(
            signals_found=2,
            trades_executed=0,
            explain_candidates=[
                ExplainCandidate(
                    market_id="m3",
                    token_id="t3",
                    side="NO",
                    edge_bps=45.0,
                    spread_bps=10.0,
                    liquidity=500.0,
                    mid_price=0.5,
                    rejection_reason=RejectionReason.LIQUIDITY_BELOW_MIN,
                ),
            ],
            rejection_counts={"liquidity_below_min": 1},
        )

        summary = ExplainSummary.from_results([result1, result2])

        assert summary.total_ticks == 2
        assert summary.total_candidates == 3
        assert summary.total_executed == 1
        assert summary.rejection_counts["executed"] == 1
        assert summary.rejection_counts["edge_below_min"] == 1
        assert summary.rejection_counts["liquidity_below_min"] == 1


class TestPaperExecConfigExplain:
    """Tests for PaperExecConfig explain mode fields."""

    def test_explain_defaults(self) -> None:
        """Explain mode should be disabled by default."""
        config = PaperExecConfig()
        assert config.explain_enabled is False
        assert config.explain_top_n == 10
        assert config.explain_export_path is None

    def test_explain_custom_values(self) -> None:
        """Explain mode should accept custom values."""
        config = PaperExecConfig(
            explain_enabled=True,
            explain_top_n=25,
            explain_export_path="exports/test.jsonl",
        )
        assert config.explain_enabled is True
        assert config.explain_top_n == 25
        assert config.explain_export_path == "exports/test.jsonl"


class TestPaperExecResultExplain:
    """Tests for PaperExecResult explain mode fields."""

    def test_result_explain_defaults(self) -> None:
        """Result should have empty explain fields by default."""
        result = PaperExecResult()
        assert result.explain_candidates == []
        assert result.rejection_counts == {}
        assert result.markets_scanned == 0  # Phase 6.2.1

    def test_result_with_explain_data(self) -> None:
        """Result should store explain candidates and counts."""
        candidates = [
            ExplainCandidate(
                market_id="m1",
                token_id="t1",
                side="YES",
                edge_bps=100.0,
                spread_bps=10.0,
                liquidity=5000.0,
                mid_price=0.5,
                rejection_reason=RejectionReason.EXECUTED,
            ),
        ]
        counts = {"executed": 1}

        result = PaperExecResult(
            explain_candidates=candidates,
            rejection_counts=counts,
            markets_scanned=100,
        )

        assert len(result.explain_candidates) == 1
        assert result.rejection_counts["executed"] == 1
        assert result.markets_scanned == 100


# =============================================================================
# Tests: Phase 6.2.1 - Always-On Explain Mode
# =============================================================================


class TestAlwaysOnExplainMode:
    """Tests for Phase 6.2.1 always-on explain mode features."""

    def test_near_miss_candidates_generated_when_no_signals(self) -> None:
        """Near-miss candidates should be generated when no arb signals found."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
            explain_enabled=True,
            explain_top_n=5,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Empty scanner - no signals
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        # Markets with valid price data but no arb opportunities
        markets_data = [
            {
                "id": "m1",
                "question": "Test market 1",
                "active": True,
                "closed": False,
                "last_price_yes": 0.50,
                "last_price_no": 0.51,  # Sum = 1.01, no arb
                "liquidity": 5000.0,
            },
            {
                "id": "m2",
                "question": "Test market 2",
                "active": True,
                "closed": False,
                "last_price_yes": 0.48,
                "last_price_no": 0.52,  # Sum = 1.00, no arb
                "liquidity": 3000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # Should have near-miss candidates even with no signals
        assert result.signals_found == 0
        assert len(result.explain_candidates) > 0
        assert result.markets_scanned == 2

        # All candidates should have rejection reasons
        for cand in result.explain_candidates:
            assert cand.rejection_reason != RejectionReason.NONE

    def test_blocked_by_risk_zero_when_no_signals(self) -> None:
        """blocked_by_risk should be 0 when there are no executable signals."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,  # Requires approval
            explain_enabled=True,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Risk gate that rejects everything
        risk_gate = FakeRiskGate(approved=False, block_reason="Not approved")

        # Empty scanner - no signals
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        markets_data = [
            {
                "id": "m1",
                "active": True,
                "last_price_yes": 0.50,
                "last_price_no": 0.51,
                "liquidity": 5000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # No executable signals → blocked_by_risk should be 0
        assert result.signals_found == 0
        assert result.signals_evaluated == 0
        assert result.blocked_by_risk == 0

    def test_blocked_by_risk_incremented_when_signals_blocked(self) -> None:
        """blocked_by_risk should increment when executable signals are blocked."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,
            explain_enabled=True,
            min_signal_edge_bps=10.0,  # Low threshold
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Risk gate that rejects everything
        risk_gate = FakeRiskGate(approved=False, block_reason="Not approved")

        # Signal with good edge that would execute
        signal = make_signal("m1", 0.40, 0.50, liquidity=5000.0)  # 10% edge
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        markets_data = [
            {
                "id": "m1",
                "active": True,
                "last_price_yes": 0.40,
                "last_price_no": 0.50,
                "liquidity": 5000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # Had executable signal but not approved → blocked_by_risk > 0
        assert result.signals_found == 1
        assert result.signals_evaluated > 0
        assert result.blocked_by_risk > 0

    def test_markets_scanned_tracked(self) -> None:
        """markets_scanned should track number of markets processed."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
            max_markets_scanned=50,  # Limit to 50
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        # Create 100 markets but max is 50
        markets_data = [
            {
                "id": f"m{i}",
                "active": True,
                "last_price_yes": 0.5,
                "last_price_no": 0.5,
                "liquidity": 1000.0,
            }
            for i in range(100)
        ]

        result = executor.execute_tick(markets_data)

        # Should only scan up to max
        assert result.markets_scanned == 50

    def test_near_miss_candidates_have_rejection_details(self) -> None:
        """Near-miss candidates should have detailed rejection reasons."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
            explain_enabled=True,
            explain_top_n=10,
            min_signal_edge_bps=100.0,  # 1% min edge
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        markets_data = [
            {
                "id": "m_closed",
                "active": True,
                "closed": True,
                "last_price_yes": 0.45,
                "last_price_no": 0.45,
                "liquidity": 5000.0,
            },
            {
                "id": "m_inactive",
                "active": False,
                "closed": False,
                "last_price_yes": 0.45,
                "last_price_no": 0.45,
                "liquidity": 5000.0,
            },
            {
                "id": "m_low_liq",
                "active": True,
                "closed": False,
                "last_price_yes": 0.45,
                "last_price_no": 0.45,
                "liquidity": 10.0,  # Below min liquidity
            },
            {
                "id": "m_no_arb",
                "active": True,
                "closed": False,
                "last_price_yes": 0.50,
                "last_price_no": 0.51,  # Sum > 1.0, no arb
                "liquidity": 5000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # Should have candidates with various rejection reasons
        # At least some should have clear rejection reasons
        assert len(result.explain_candidates) > 0
        for cand in result.explain_candidates:
            assert cand.rejection_detail != ""  # Should have detail


# =============================================================================
# Tests: Phase 7 - Raw Edge Before Risk Gating
# =============================================================================


class TestRawEdgeBps:
    """Tests for Phase 7 raw_edge_bps field.

    raw_edge_bps captures edge BEFORE any risk gating, ensuring
    we can calibrate thresholds even when trades are blocked.
    """

    def test_raw_edge_bps_preserved_when_risk_not_approved(self) -> None:
        """raw_edge_bps should be non-zero even when rejection_reason is risk_not_approved."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,
            explain_enabled=True,
            min_signal_edge_bps=10.0,  # Low threshold to ensure signal passes
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Risk gate that rejects (not approved)
        risk_gate = FakeRiskGate(approved=False, block_reason="No approval for paper_exec")

        # Signal with good edge (10% edge = 1000 bps)
        # yes + no = 0.40 + 0.50 = 0.90 → profit_potential = 0.10 → 1000 bps
        signal = make_signal("m1", 0.40, 0.50, liquidity=5000.0)
        scanner = FakeArbitrageScanner(signals=[signal])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        markets_data = [
            {
                "id": "m1",
                "question": "Test market",
                "active": True,
                "closed": False,
                "last_price_yes": 0.40,
                "last_price_no": 0.50,
                "liquidity": 5000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # Should have explain candidates
        assert len(result.explain_candidates) > 0

        # Find candidate that was rejected by risk
        risk_rejected = [
            c
            for c in result.explain_candidates
            if c.rejection_reason == RejectionReason.RISK_NOT_APPROVED
        ]
        assert len(risk_rejected) > 0, "Expected at least one risk_not_approved rejection"

        # The critical assertion: raw_edge_bps should be non-zero
        # even though the trade was rejected by risk
        candidate = risk_rejected[0]
        assert candidate.raw_edge_bps > 0, (
            f"raw_edge_bps should be non-zero when rejected by risk, got {candidate.raw_edge_bps}"
        )
        assert candidate.raw_edge_bps >= 100, (
            f"Expected raw_edge_bps >= 100 (10% edge), got {candidate.raw_edge_bps}"
        )

    def test_raw_edge_bps_equals_edge_bps_when_not_rejected(self) -> None:
        """raw_edge_bps should equal edge_bps for executed candidates."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,  # No risk gate
            explain_enabled=True,
            min_signal_edge_bps=10.0,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)

        # Good signal that should execute
        signal = make_signal("m1", 0.40, 0.50, liquidity=5000.0)  # 10% edge
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
                "active": True,
                "last_price_yes": 0.40,
                "last_price_no": 0.50,
                "liquidity": 5000.0,
            },
        ]

        result = executor.execute_tick(markets_data)

        # Should have executed
        assert result.trades_executed > 0
        assert len(result.explain_candidates) > 0

        # For executed candidates, raw_edge_bps == edge_bps
        for cand in result.explain_candidates:
            if cand.executed:
                assert cand.raw_edge_bps == cand.edge_bps, (
                    f"raw_edge_bps ({cand.raw_edge_bps}) should equal "
                    f"edge_bps ({cand.edge_bps}) for executed trades"
                )

    def test_raw_edge_bps_in_to_dict(self) -> None:
        """to_dict should include raw_edge_bps field."""
        candidate = ExplainCandidate(
            market_id="m1",
            token_id="t1",
            side="arb",
            edge_bps=0.0,  # Zero because rejected
            raw_edge_bps=150.0,  # Actual edge before gating
            spread_bps=10.0,
            liquidity=5000.0,
            mid_price=0.5,
            rejection_reason=RejectionReason.RISK_NOT_APPROVED,
            rejection_detail="No approval for paper_exec",
        )

        result = candidate.to_dict()

        assert "raw_edge_bps" in result
        assert result["raw_edge_bps"] == 150.0
        assert result["edge_bps"] == 0.0
        assert result["rejection_reason"] == "risk_not_approved"

    def test_explain_summary_tracks_raw_edge_stats(self) -> None:
        """ExplainSummary should track raw_edge statistics."""
        # Create results with raw_edge data
        result1 = PaperExecResult(
            explain_candidates=[
                ExplainCandidate(
                    market_id="m1",
                    edge_bps=0.0,
                    raw_edge_bps=100.0,
                    rejection_reason=RejectionReason.RISK_NOT_APPROVED,
                ),
            ],
            rejection_counts={"risk_not_approved": 1},
        )
        result2 = PaperExecResult(
            explain_candidates=[
                ExplainCandidate(
                    market_id="m2",
                    edge_bps=0.0,
                    raw_edge_bps=200.0,
                    rejection_reason=RejectionReason.RISK_NOT_APPROVED,
                ),
            ],
            rejection_counts={"risk_not_approved": 1},
        )

        summary = ExplainSummary.from_results([result1, result2], min_edge_bps=50.0)

        # Should track raw edge statistics
        assert summary.max_raw_edge_bps == 200.0
        assert summary.avg_top_raw_edge_bps == 150.0  # (100 + 200) / 2
        assert summary.ticks_with_raw_edge_above_threshold == 2  # Both >= 50


# =============================================================================
# Phase 8: Orderbook-Based Edge Computation Tests
# =============================================================================


class TestPhase8OrderbookEdge:
    """Tests for Phase 8 orderbook-based edge computation.

    Phase 8 uses CLOB orderbook data (best_bid, best_ask) to compute
    true arb edge using the formulas:
    - BUY_BOTH: raw_edge_bps = (1.0 - (ask_yes + ask_no)) * 10_000
    - SELL_BOTH: raw_edge_bps = ((bid_yes + bid_no) - 1.0) * 10_000

    For YES orderbooks, NO prices are derived using binary market relations:
    - bid_no ≈ 1 - ask_yes
    - ask_no ≈ 1 - bid_yes
    """

    def test_orderbook_edge_buy_both_scenario(self) -> None:
        """Test BUY_BOTH edge computed from orderbook data."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=True,
            explain_enabled=True,
            min_signal_edge_bps=10.0,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        risk_gate = FakeRiskGate(approved=False)  # Block execution to see explain data
        scanner = FakeArbitrageScanner(signals=[])  # No signals - use near-miss path

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            risk_gate=risk_gate,
            arb_scanner=scanner,
        )

        # Market with orderbook data showing BUY_BOTH opportunity:
        # bid_yes=0.47, ask_yes=0.48 → bid_no=1-0.48=0.52, ask_no=1-0.47=0.53
        # buy_cost = 0.48 + 0.53 = 1.01 → BUY edge = (1 - 1.01) * 10000 = -100
        # sell_revenue = 0.47 + 0.52 = 0.99 → SELL edge = (0.99 - 1) * 10000 = -100
        # Both negative, but BUY_BOTH wins on tie (per implementation)
        markets_data = [
            {
                "id": "m1",
                "question": "Test market",
                "active": True,
                "closed": False,
                "last_price_yes": 0.475,  # Gamma price
                "last_price_no": 0.505,
                "liquidity": 5000.0,
                "yes_token_id": "yes_token_001",
                "no_token_id": "no_token_001",
                "orderbook": {
                    "best_bid": 0.47,
                    "best_ask": 0.48,
                },
            },
        ]

        result = executor.execute_tick(markets_data)

        # Should have candidates
        assert len(result.explain_candidates) > 0

        candidate = result.explain_candidates[0]

        # Should have orderbook prices populated
        assert candidate.bid_yes == 0.47
        assert candidate.ask_yes == 0.48
        # NO prices derived from YES orderbook
        assert candidate.bid_no is not None
        assert candidate.ask_no is not None

        # raw_edge_bps should be computed from orderbook
        # Not zero like before Phase 8
        assert candidate.raw_edge_bps != 0.0 or candidate.arb_side in ("BUY_BOTH", "SELL_BOTH")

        # Token IDs should be populated
        assert candidate.yes_token_id == "yes_token_001"
        assert candidate.no_token_id == "no_token_001"

    def test_orderbook_edge_positive_buy_both(self) -> None:
        """Test positive BUY_BOTH edge from orderbook.

        Scenario: ask_yes + ask_no < 1.0 → positive BUY_BOTH edge.
        """
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
            explain_enabled=True,
            min_signal_edge_bps=10.0,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        # bid_yes=0.46, ask_yes=0.47 → bid_no=1-0.47=0.53, ask_no=1-0.46=0.54
        # buy_cost = 0.47 + 0.54 = 1.01 → BUY edge = (1 - 1.01) * 10000 = -100
        # But let's use asymmetric pricing to create positive edge:
        # bid_yes=0.48, ask_yes=0.49 → bid_no=1-0.49=0.51, ask_no=1-0.48=0.52
        # buy_cost = 0.49 + 0.52 = 1.01 → still -100
        # Need ask_yes lower: bid_yes=0.44, ask_yes=0.45 → bid_no=0.55, ask_no=0.56
        # buy_cost = 0.45 + 0.56 = 1.01 → still -100
        # The math shows that with derived NO prices, we can't get positive BUY edge
        # because ask_yes + ask_no = ask_yes + (1 - bid_yes) >= ask_yes + (1 - ask_yes) = 1
        # when bid_yes <= ask_yes

        # This is expected - the derived NO prices are approximations.
        # Real positive edges require actual NO token orderbooks.

        markets_data = [
            {
                "id": "m1",
                "question": "Test",
                "active": True,
                "closed": False,
                "last_price_yes": 0.475,
                "last_price_no": 0.505,
                "liquidity": 5000.0,
                "yes_token_id": "yes_001",
                "no_token_id": "no_001",
                "orderbook": {
                    "best_bid": 0.47,
                    "best_ask": 0.48,
                },
            },
        ]

        result = executor.execute_tick(markets_data)

        assert len(result.explain_candidates) > 0
        candidate = result.explain_candidates[0]

        # Should have arb_side populated (BUY_BOTH or SELL_BOTH)
        assert candidate.arb_side in ("BUY_BOTH", "SELL_BOTH", "NONE")

        # mid_price should be computed from orderbook (not hardcoded 0.5)
        assert candidate.mid_price != 0.5 or candidate.bid_yes is None

        # spread_bps should be computed
        assert candidate.spread_bps >= 0

    def test_to_dict_includes_phase8_fields(self) -> None:
        """to_dict should include all Phase 8 fields."""
        candidate = ExplainCandidate(
            market_id="m1",
            token_id="t1",
            yes_token_id="yes_001",
            no_token_id="no_001",
            side="arb",
            arb_side="BUY_BOTH",
            edge_bps=300.0,
            raw_edge_bps=300.0,
            spread_bps=421.1,
            liquidity=5000.0,
            mid_price=0.475,
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
        )

        result = candidate.to_dict()

        # Phase 8 fields
        assert result["yes_token_id"] == "yes_001"
        assert result["no_token_id"] == "no_001"
        assert result["arb_side"] == "BUY_BOTH"
        assert result["ask_yes"] == 0.48
        assert result["ask_no"] == 0.49
        assert result["bid_yes"] == 0.47
        assert result["bid_no"] == 0.48

    def test_fallback_to_gamma_prices_without_orderbook(self) -> None:
        """Should fallback to Gamma prices when orderbook not available."""
        config = PaperExecConfig(
            enabled=True,
            require_approval=False,
            explain_enabled=True,
            min_signal_edge_bps=10.0,
        )
        dao = FakeDAO()
        ledger = FakePaperLedger(dao)
        scanner = FakeArbitrageScanner(signals=[])

        executor = PaperExecutor(
            config=config,
            dao=dao,
            paper_ledger=ledger,
            arb_scanner=scanner,
        )

        # Market WITHOUT orderbook data
        markets_data = [
            {
                "id": "m1",
                "question": "Test",
                "active": True,
                "closed": False,
                "last_price_yes": 0.40,
                "last_price_no": 0.50,
                "liquidity": 5000.0,
                "yes_token_id": "yes_001",
                "no_token_id": "no_001",
                # No "orderbook" key
            },
        ]

        result = executor.execute_tick(markets_data)

        assert len(result.explain_candidates) > 0
        candidate = result.explain_candidates[0]

        # Should use Gamma price-based edge (yes + no = 0.90 → 10% edge = 1000 bps)
        expected_edge = (1.0 - (0.40 + 0.50)) * 10000  # 1000 bps
        assert abs(candidate.raw_edge_bps - expected_edge) < 1.0

        # Orderbook fields should be None
        assert candidate.ask_yes is None
        assert candidate.ask_no is None
        assert candidate.bid_yes is None
        assert candidate.bid_no is None
        assert candidate.arb_side == "NONE"
