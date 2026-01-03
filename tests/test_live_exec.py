"""Tests for Phase 12 live execution module.

Tests cover:
- Safety checks (enabled, confirm, kill switch)
- TTL approval gating (separate from paper_exec)
- Rate and size limits
- Order plan construction
- Ledger writes with redaction
- Mock CLOB client behavior
- Dry-run mode

NO NETWORK CALLS - all tests use mocks.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# Mock the py_clob_client.order module before importing live_exec
# This prevents ImportError when running tests without py-clob-client installed
BUY = "BUY"
SELL = "SELL"


class MockOrderArgs:
    """Mock OrderArgs for testing."""

    def __init__(self, token_id: str, price: float, size: float, side: str):
        self.token_id = token_id
        self.price = price
        self.size = size
        self.side = side


# Create mock module
mock_order_module = Mock()
mock_order_module.BUY = BUY
mock_order_module.SELL = SELL
mock_order_module.OrderArgs = MockOrderArgs

# Inject mock into sys.modules
sys.modules["py_clob_client"] = Mock()
sys.modules["py_clob_client.order"] = mock_order_module

# Imports after mock setup (noqa E402 expected)
from pmq.ops.edge_calc import ArbSide  # noqa: E402
from pmq.ops.live_exec import (  # noqa: E402
    LiveExecConfig,
    LiveExecStatus,
    LiveExecutor,
    RejectionReason,
    TradePlan,
    create_trade_plan_from_edge,
)
from pmq.ops.live_ledger import LiveLedger, LiveOrderRecord  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_ledger_dir():
    """Create a temporary directory for ledger files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_kill_switch_dir():
    """Create a temporary directory for kill switch file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_trade_plan() -> TradePlan:
    """Create a sample trade plan for testing."""
    return TradePlan(
        market_id="market_123",
        yes_token_id="0xyes123abc",
        no_token_id="0xno456def",
        arb_side=ArbSide.BUY_BOTH,
        yes_limit_price=0.45,
        no_limit_price=0.52,
        size=5.0,
        gross_edge_bps=100.0,
        net_edge_bps=80.0,
        market_question="Test market?",
    )


class MockClobClient:
    """Mock CLOB client for testing."""

    def __init__(self, should_fail: bool = False, fail_message: str = ""):
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.orders_posted: list[dict[str, Any]] = []
        self._order_id_counter = 0

    def get_orders(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:  # noqa: ARG002
        """Return empty list of orders."""
        return []

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Return mock order."""
        return {"orderID": order_id, "status": "LIVE"}

    def create_and_post_order(self, order: Any) -> dict[str, Any]:
        """Mock order posting."""
        if self.should_fail:
            raise Exception(self.fail_message or "Mock order failure")

        self._order_id_counter += 1
        order_id = f"order_{self._order_id_counter}"
        self.orders_posted.append(
            {
                "orderID": order_id,
                "token_id": order.token_id,
                "price": order.price,
                "size": order.size,
                "side": order.side,
            }
        )
        return {"orderID": order_id}

    def cancel_order(self, order_id: str) -> dict[str, Any]:  # noqa: ARG002
        """Mock order cancellation."""
        return {"success": True}


def make_approval_checker(approved: bool, detail: str = ""):
    """Create a mock approval checker function."""

    def checker(scope: str, now: datetime | None = None) -> tuple[bool, str]:  # noqa: ARG001
        if approved:
            return True, detail or f"Approved for {scope}"
        return False, detail or f"Not approved for {scope}"

    return checker


# =============================================================================
# Test: LiveExecConfig Defaults
# =============================================================================


class TestLiveExecConfig:
    """Test LiveExecConfig default values."""

    def test_disabled_by_default(self):
        """Config is disabled by default."""
        config = LiveExecConfig()
        assert config.enabled is False

    def test_confirm_false_by_default(self):
        """Confirm is False by default (dry-run)."""
        config = LiveExecConfig()
        assert config.confirm is False

    def test_safe_limits_by_default(self):
        """Default limits are conservative."""
        config = LiveExecConfig()
        assert config.max_order_usd <= 10.0
        assert config.max_orders_per_hour <= 5

    def test_live_exec_scope_separate(self):
        """Approval scope is live_exec, not paper_exec."""
        config = LiveExecConfig()
        assert config.approval_scope == "live_exec"
        assert config.approval_scope != "paper_exec"


# =============================================================================
# Test: Safety Check - Not Enabled
# =============================================================================


class TestNotEnabled:
    """Test behavior when live exec is not enabled."""

    def test_returns_disabled_when_not_enabled(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path
    ):
        """Returns DISABLED status when enabled=False."""
        config = LiveExecConfig(
            enabled=False,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        executor = LiveExecutor(
            config=config,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.DISABLED
        assert result.rejection_reason == RejectionReason.NOT_ENABLED


# =============================================================================
# Test: Safety Check - Kill Switch
# =============================================================================


class TestKillSwitch:
    """Test kill switch behavior."""

    def test_rejects_when_kill_switch_exists(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path
    ):
        """Rejects execution when kill switch file exists."""
        kill_file = temp_kill_switch_dir / "KILL"
        kill_file.touch()  # Create kill switch file

        config = LiveExecConfig(
            enabled=True,
            kill_switch_path=kill_file,
        )
        executor = LiveExecutor(
            config=config,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.KILL_SWITCH

    def test_allows_when_kill_switch_missing(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Allows execution when kill switch file does not exist."""
        kill_file = temp_kill_switch_dir / "KILL"
        # Do NOT create the file

        config = LiveExecConfig(
            enabled=True,
            confirm=False,  # Dry run
            kill_switch_path=kill_file,
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        # Should pass kill switch check (may fail later for other reasons)
        assert result.rejection_reason != RejectionReason.KILL_SWITCH


# =============================================================================
# Test: Safety Check - TTL Approval
# =============================================================================


class TestTTLApproval:
    """Test TTL approval gating."""

    def test_rejects_when_approval_missing(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path
    ):
        """Rejects when no TTL approval exists."""
        config = LiveExecConfig(
            enabled=True,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        executor = LiveExecutor(
            config=config,
            approval_checker=make_approval_checker(
                False, "No TTL approval found for scope 'live_exec'"
            ),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.APPROVAL_MISSING

    def test_rejects_when_approval_expired(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path
    ):
        """Rejects when TTL approval has expired."""
        config = LiveExecConfig(
            enabled=True,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        executor = LiveExecutor(
            config=config,
            approval_checker=make_approval_checker(
                False, "TTL approval for 'live_exec' expired 300s ago"
            ),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.APPROVAL_EXPIRED

    def test_allows_when_approval_valid(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Allows execution when TTL approval is valid."""
        config = LiveExecConfig(
            enabled=True,
            confirm=False,  # Dry run
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True, "Approved for 3600s"),
        )

        result = executor.execute(sample_trade_plan)

        # Should pass approval check
        assert result.rejection_reason != RejectionReason.APPROVAL_MISSING
        assert result.rejection_reason != RejectionReason.APPROVAL_EXPIRED


# =============================================================================
# Test: Safety Check - SELL_BOTH Disabled
# =============================================================================


class TestSellBothDisabled:
    """Test that SELL_BOTH is rejected."""

    def test_rejects_sell_both(self, temp_kill_switch_dir: Path, temp_ledger_dir: Path):
        """SELL_BOTH is rejected because position mechanics not implemented."""
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="0xyes123",
            no_token_id="0xno456",
            arb_side=ArbSide.SELL_BOTH,  # This should be rejected
            yes_limit_price=0.55,
            no_limit_price=0.48,
            size=5.0,
            gross_edge_bps=100.0,
            net_edge_bps=80.0,
        )

        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.SELL_BOTH_DISABLED


# =============================================================================
# Test: Safety Check - Size Limit
# =============================================================================


class TestSizeLimit:
    """Test size limit enforcement."""

    def test_rejects_when_size_exceeds_limit(
        self, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Rejects when total notional exceeds max_order_usd."""
        # Plan with large size that will exceed $5 limit
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="0xyes123",
            no_token_id="0xno456",
            arb_side=ArbSide.BUY_BOTH,
            yes_limit_price=0.50,
            no_limit_price=0.50,
            size=20.0,  # 0.50 * 20 * 2 legs = $20, exceeds $5 limit
            gross_edge_bps=100.0,
            net_edge_bps=80.0,
        )

        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            max_order_usd=5.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.SIZE_LIMIT

    def test_allows_when_size_within_limit(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Allows when total notional is within limit."""
        # sample_trade_plan: 0.45*5 + 0.52*5 = $4.85, within $5 limit
        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            max_order_usd=10.0,  # Generous limit
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.rejection_reason != RejectionReason.SIZE_LIMIT


# =============================================================================
# Test: Safety Check - Rate Limit
# =============================================================================


class TestRateLimit:
    """Test rate limit enforcement."""

    def test_rejects_when_rate_limit_exceeded(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Rejects when orders in last hour >= limit."""
        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            max_orders_per_hour=2,
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)

        # Pre-populate ledger with 2 orders in last hour
        now = datetime.now(UTC)
        for i in range(2):
            record = LiveOrderRecord(
                order_id=f"existing_{i}",
                timestamp=now.isoformat(),
                market_id="market_old",
                token_id="0xtoken",
                outcome="YES",
                side="BUY",
                price=0.50,
                size=1.0,
                notional_usd=0.50,
                status="POSTED",
            )
            ledger.record_order(record)

        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan, now=now)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.RATE_LIMIT


# =============================================================================
# Test: Safety Check - Edge Minimum
# =============================================================================


class TestEdgeMinimum:
    """Test minimum net edge enforcement."""

    def test_rejects_when_edge_below_minimum(
        self, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Rejects when net edge is below minimum threshold."""
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="0xyes123",
            no_token_id="0xno456",
            arb_side=ArbSide.BUY_BOTH,
            yes_limit_price=0.45,
            no_limit_price=0.52,
            size=2.0,
            gross_edge_bps=15.0,
            net_edge_bps=5.0,  # Below default 10 bps minimum
        )

        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            min_net_edge_bps=10.0,
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.EDGE_TOO_LOW


# =============================================================================
# Test: Dry Run Mode
# =============================================================================


class TestDryRunMode:
    """Test dry-run behavior (confirm=False)."""

    def test_dry_run_logs_to_ledger(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Dry run records orders in ledger with DRY_RUN status."""
        config = LiveExecConfig(
            enabled=True,
            confirm=False,  # DRY RUN
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.DRY_RUN
        assert result.dry_run is True
        assert len(result.orders) == 2  # YES and NO

        # Check ledger
        recent = ledger.get_recent_orders(limit=10)
        assert len(recent) == 2
        for order in recent:
            assert order.status == "DRY_RUN"
            assert order.dry_run is True

    def test_dry_run_does_not_call_client(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Dry run does not call the CLOB client."""
        config = LiveExecConfig(
            enabled=True,
            confirm=False,  # DRY RUN
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        mock_client = MockClobClient()
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            client=mock_client,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.DRY_RUN
        assert len(mock_client.orders_posted) == 0  # No orders posted


# =============================================================================
# Test: Live Execution (confirm=True)
# =============================================================================


class TestLiveExecution:
    """Test actual order posting (confirm=True)."""

    def test_posts_orders_when_confirmed(
        self, sample_trade_plan: TradePlan, temp_kill_switch_dir: Path, temp_ledger_dir: Path
    ):
        """Posts orders to CLOB client when confirm=True."""
        config = LiveExecConfig(
            enabled=True,
            confirm=True,  # LIVE EXECUTION
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        mock_client = MockClobClient()
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            client=mock_client,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(sample_trade_plan)

        assert result.status == LiveExecStatus.SUCCESS
        assert result.orders_posted == 2
        assert len(mock_client.orders_posted) == 2

        # Check ledger has POSTED status
        recent = ledger.get_recent_orders(limit=10)
        assert len(recent) == 2
        for order in recent:
            assert order.status == "POSTED"
            assert order.order_id is not None

    def test_handles_partial_failure(self, temp_kill_switch_dir: Path, temp_ledger_dir: Path):
        """Handles partial failure (one order fails, one succeeds)."""
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="0xyes123",
            no_token_id="0xno456",
            arb_side=ArbSide.BUY_BOTH,
            yes_limit_price=0.45,
            no_limit_price=0.52,
            size=2.0,
            gross_edge_bps=100.0,
            net_edge_bps=80.0,
        )

        config = LiveExecConfig(
            enabled=True,
            confirm=True,
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )

        # Create client that fails after first order
        call_count = [0]

        class PartialFailClient(MockClobClient):
            def create_and_post_order(self, order: Any) -> dict[str, Any]:
                call_count[0] += 1
                if call_count[0] > 1:
                    raise Exception("Second order failed")
                return super().create_and_post_order(order)

        mock_client = PartialFailClient()
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            client=mock_client,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(plan)

        assert result.status == LiveExecStatus.ERROR
        assert result.orders_posted == 1  # Only first order succeeded


# =============================================================================
# Test: Ledger Redaction
# =============================================================================


class TestLedgerRedaction:
    """Test that sensitive data is redacted in ledger."""

    def test_token_ids_masked(self, temp_ledger_dir: Path):
        """Token IDs are masked in ledger records."""
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)

        record = LiveOrderRecord(
            order_id="test_order",
            timestamp=datetime.now(UTC).isoformat(),
            market_id="market_123",
            token_id="0x1234567890abcdef1234567890abcdef12345678",  # Long token ID
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
            notional_usd=5.0,
            status="POSTED",
        )
        ledger.record_order(record)

        # Check that token ID was masked
        recent = ledger.get_recent_orders(limit=1)
        assert len(recent) == 1
        saved_token_id = recent[0].token_id
        # Should be masked like "0x123456...12345678"
        assert "..." in saved_token_id
        assert len(saved_token_id) < 50

    def test_error_messages_redacted(self, temp_ledger_dir: Path):
        """Error messages with secrets are redacted."""
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)

        # Error with a fake private key in it
        record = LiveOrderRecord(
            order_id=None,
            timestamp=datetime.now(UTC).isoformat(),
            market_id="market_123",
            token_id="0xtoken",
            outcome="YES",
            side="BUY",
            price=0.50,
            size=10.0,
            notional_usd=5.0,
            status="ERROR",
            error_message="Auth failed with key 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        )
        ledger.record_order(record)

        recent = ledger.get_recent_orders(limit=1)
        assert len(recent) == 1
        # The 64-char hex key should be redacted
        assert "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef" not in (
            recent[0].error_message or ""
        )


# =============================================================================
# Test: TradePlan Creation
# =============================================================================


class TestTradePlanCreation:
    """Test TradePlan and create_trade_plan_from_edge helper."""

    def test_create_trade_plan_from_edge(self):
        """create_trade_plan_from_edge creates valid plan."""
        plan = create_trade_plan_from_edge(
            market_id="market_abc",
            yes_token_id="0xyes",
            no_token_id="0xno",
            arb_side=ArbSide.BUY_BOTH,
            ask_yes=0.48,
            ask_no=0.49,
            size=10.0,
            gross_edge_bps=100.0,
            net_edge_bps=85.0,
            market_question="Will it rain?",
        )

        assert plan.market_id == "market_abc"
        assert plan.arb_side == ArbSide.BUY_BOTH
        assert plan.yes_limit_price == 0.48
        assert plan.no_limit_price == 0.49
        assert plan.size == 10.0
        assert plan.gross_edge_bps == 100.0
        assert plan.net_edge_bps == 85.0

    def test_trade_plan_has_timestamp(self):
        """TradePlan auto-sets timestamp if not provided."""
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="0xyes",
            no_token_id="0xno",
            arb_side=ArbSide.BUY_BOTH,
            yes_limit_price=0.45,
            no_limit_price=0.52,
            size=5.0,
            gross_edge_bps=100.0,
            net_edge_bps=80.0,
        )

        assert plan.timestamp != ""
        # Should be ISO format
        datetime.fromisoformat(plan.timestamp.replace("Z", "+00:00"))


# =============================================================================
# Test: LiveLedger Statistics
# =============================================================================


class TestLiveLedgerStats:
    """Test LiveLedger statistics."""

    def test_stats_counts_by_status(self, temp_ledger_dir: Path):
        """Stats correctly count orders by status."""
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        now = datetime.now(UTC)

        # Add various orders
        for status in ["POSTED", "POSTED", "DRY_RUN", "REJECTED", "ERROR"]:
            record = LiveOrderRecord(
                order_id="id" if status == "POSTED" else None,
                timestamp=now.isoformat(),
                market_id="market",
                token_id="0xtoken",
                outcome="YES",
                side="BUY",
                price=0.50,
                size=1.0,
                notional_usd=0.50,
                status=status,
            )
            ledger.record_order(record)

        stats = ledger.get_stats()

        assert stats.total_orders == 5
        assert stats.posted_orders == 2
        assert stats.dry_run_orders == 1
        assert stats.rejected_orders == 1
        assert stats.error_orders == 1

    def test_stats_sums_notional(self, temp_ledger_dir: Path):
        """Stats correctly sum notional for POSTED orders only."""
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        now = datetime.now(UTC)

        # Add POSTED order
        ledger.record_order(
            LiveOrderRecord(
                order_id="id1",
                timestamp=now.isoformat(),
                market_id="market",
                token_id="0xtoken",
                outcome="YES",
                side="BUY",
                price=0.50,
                size=10.0,
                notional_usd=5.0,
                status="POSTED",
            )
        )

        # Add DRY_RUN order (should not count toward notional)
        ledger.record_order(
            LiveOrderRecord(
                order_id=None,
                timestamp=now.isoformat(),
                market_id="market",
                token_id="0xtoken",
                outcome="YES",
                side="BUY",
                price=0.50,
                size=10.0,
                notional_usd=5.0,
                status="DRY_RUN",
            )
        )

        stats = ledger.get_stats()

        assert stats.total_notional_usd == 5.0  # Only POSTED order


# =============================================================================
# Test: Missing Token IDs
# =============================================================================


class TestMissingTokenIds:
    """Test rejection when token IDs are missing."""

    def test_rejects_missing_yes_token(self, temp_kill_switch_dir: Path, temp_ledger_dir: Path):
        """Rejects when YES token ID is missing."""
        plan = TradePlan(
            market_id="market_123",
            yes_token_id="",  # Missing
            no_token_id="0xno456",
            arb_side=ArbSide.BUY_BOTH,
            yes_limit_price=0.45,
            no_limit_price=0.52,
            size=2.0,
            gross_edge_bps=100.0,
            net_edge_bps=80.0,
        )

        config = LiveExecConfig(
            enabled=True,
            confirm=False,
            max_order_usd=20.0,
            kill_switch_path=temp_kill_switch_dir / "KILL",
        )
        ledger = LiveLedger(ledger_dir=temp_ledger_dir)
        executor = LiveExecutor(
            config=config,
            ledger=ledger,
            approval_checker=make_approval_checker(True),
        )

        result = executor.execute(plan)

        assert result.status == LiveExecStatus.REJECTED
        assert result.rejection_reason == RejectionReason.MISSING_TOKEN_ID
