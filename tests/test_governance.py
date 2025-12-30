"""Tests for governance module (Phase 3).

Tests cover:
- Strategy scorecard computation
- Approval CRUD operations
- RiskGate enforcement
"""

import tempfile
from pathlib import Path

import pytest

from pmq.governance import RiskGate, RiskLimits, compute_scorecard
from pmq.storage import DAO
from pmq.storage.db import Database


class TestScorecard:
    """Tests for scorecard computation."""

    def test_compute_scorecard_passing(self) -> None:
        """Test scorecard with passing metrics."""
        scorecard = compute_scorecard(
            total_pnl=1000.0,
            max_drawdown=0.08,
            win_rate=0.65,
            sharpe_ratio=1.5,
            total_trades=50,
            trades_per_day=5.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is True
        assert scorecard.score >= 60
        assert any("PASS" in r for r in scorecard.reasons)
        assert isinstance(scorecard.recommended_limits, RiskLimits)

    def test_compute_scorecard_failing_pnl(self) -> None:
        """Test scorecard fails with negative PnL."""
        scorecard = compute_scorecard(
            total_pnl=-500.0,
            max_drawdown=0.05,
            win_rate=0.70,
            sharpe_ratio=2.0,
            total_trades=100,
            trades_per_day=10.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "PnL" in r for r in scorecard.reasons)

    def test_compute_scorecard_failing_drawdown(self) -> None:
        """Test scorecard fails with high drawdown."""
        scorecard = compute_scorecard(
            total_pnl=1000.0,
            max_drawdown=0.30,  # 30% - above 25% threshold
            win_rate=0.70,
            sharpe_ratio=2.0,
            total_trades=100,
            trades_per_day=10.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "drawdown" in r.lower() for r in scorecard.reasons)

    def test_compute_scorecard_failing_win_rate(self) -> None:
        """Test scorecard fails with low win rate."""
        scorecard = compute_scorecard(
            total_pnl=500.0,
            max_drawdown=0.05,
            win_rate=0.35,  # Below 40% threshold
            sharpe_ratio=2.0,
            total_trades=100,
            trades_per_day=10.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "Win rate" in r for r in scorecard.reasons)

    def test_compute_scorecard_failing_sharpe(self) -> None:
        """Test scorecard fails with low sharpe ratio."""
        scorecard = compute_scorecard(
            total_pnl=500.0,
            max_drawdown=0.05,
            win_rate=0.70,
            sharpe_ratio=0.3,  # Below 0.5 threshold
            total_trades=100,
            trades_per_day=10.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "Sharpe" in r for r in scorecard.reasons)

    def test_compute_scorecard_failing_trades(self) -> None:
        """Test scorecard fails with too few trades."""
        scorecard = compute_scorecard(
            total_pnl=500.0,
            max_drawdown=0.05,
            win_rate=0.70,
            sharpe_ratio=2.0,
            total_trades=3,  # Below 5 threshold
            trades_per_day=0.5,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=95.0,
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "trades" in r.lower() for r in scorecard.reasons)

    def test_compute_scorecard_failing_data_quality(self) -> None:
        """Test scorecard fails with low data quality."""
        scorecard = compute_scorecard(
            total_pnl=500.0,
            max_drawdown=0.05,
            win_rate=0.70,
            sharpe_ratio=2.0,
            total_trades=100,
            trades_per_day=10.0,
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=60.0,  # Below 70% threshold
        )

        assert scorecard.passed is False
        assert any("FAIL" in r and "quality" in r.lower() for r in scorecard.reasons)

    def test_scorecard_warnings(self) -> None:
        """Test scorecard generates warnings for borderline metrics."""
        scorecard = compute_scorecard(
            total_pnl=200.0,  # Low but positive
            max_drawdown=0.18,  # High but below threshold
            win_rate=0.55,
            sharpe_ratio=1.0,
            total_trades=20,
            trades_per_day=0.8,  # Low frequency
            capital_utilization=0.5,
            initial_balance=10000.0,
            data_quality_pct=78.0,  # Low quality
        )

        # Should have some warnings
        assert len(scorecard.warnings) > 0

    def test_recommended_limits_scale_with_performance(self) -> None:
        """Test that recommended limits scale with performance."""
        good_scorecard = compute_scorecard(
            total_pnl=2000.0,
            max_drawdown=0.05,
            win_rate=0.75,
            sharpe_ratio=2.5,
            total_trades=200,
            trades_per_day=20.0,
            capital_utilization=0.7,
            initial_balance=10000.0,
            data_quality_pct=98.0,
        )

        poor_scorecard = compute_scorecard(
            total_pnl=100.0,
            max_drawdown=0.15,
            win_rate=0.50,
            sharpe_ratio=0.8,
            total_trades=20,
            trades_per_day=2.0,
            capital_utilization=0.3,
            initial_balance=10000.0,
            data_quality_pct=80.0,
        )

        # Good performance should have higher limits
        assert (
            good_scorecard.recommended_limits.max_total_notional
            > poor_scorecard.recommended_limits.max_total_notional
        )


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_gov.db"
        db = Database(db_path=db_path)
        db.initialize()
        yield db
        db.close()


@pytest.fixture
def dao(temp_db):
    """Create a DAO with temporary database."""
    return DAO(db=temp_db)


class TestApprovalCRUD:
    """Tests for approval CRUD operations."""

    def test_save_and_get_approval(self, dao) -> None:
        """Test saving and retrieving an approval."""
        approval_id = dao.save_approval(
            strategy_name="test_strategy",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
            reasons=["PASS: Score 85/100"],
            limits={"max_notional_per_market": 1000.0},
            approved_by="test",
        )

        assert approval_id > 0

        approval = dao.get_approval(approval_id)
        assert approval is not None
        assert approval["strategy_name"] == "test_strategy"
        assert approval["strategy_version"] == "v1"
        assert approval["score"] == 85.0
        assert approval["status"] == "APPROVED"

    def test_get_active_approval(self, dao) -> None:
        """Test getting active approval for a strategy."""
        # Create approved
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=80.0,
            status="APPROVED",
        )

        # Create revoked (newer)
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v2",
            window_from="2024-02-01",
            window_to="2024-02-28",
            score=70.0,
            status="REVOKED",
        )

        # Active approval should be v1 (only approved one)
        active = dao.get_active_approval("arb")
        assert active is not None
        assert active["strategy_version"] == "v1"

        # Specific version query
        active_v2 = dao.get_active_approval("arb", "v2")
        assert active_v2 is None  # v2 is revoked

    def test_revoke_approval(self, dao) -> None:
        """Test revoking an approval."""
        approval_id = dao.save_approval(
            strategy_name="test",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=80.0,
            status="APPROVED",
        )

        dao.revoke_approval(approval_id, "Performance degraded")

        approval = dao.get_approval(approval_id)
        assert approval["status"] == "REVOKED"
        assert approval["revoke_reason"] == "Performance degraded"

    def test_get_approvals_list(self, dao) -> None:
        """Test listing approvals with filters."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=80.0,
            status="APPROVED",
        )
        dao.save_approval(
            strategy_name="statarb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=75.0,
            status="PENDING",
        )

        all_approvals = dao.get_approvals()
        assert len(all_approvals) == 2

        approved_only = dao.get_approvals(status="APPROVED")
        assert len(approved_only) == 1
        assert approved_only[0]["strategy_name"] == "arb"

        arb_only = dao.get_approvals(strategy_name="arb")
        assert len(arb_only) == 1


class TestRiskEvents:
    """Tests for risk event logging."""

    def test_save_and_get_risk_event(self, dao) -> None:
        """Test saving and retrieving risk events."""
        event_id = dao.save_risk_event(
            severity="WARN",
            event_type="LIMIT_BREACH",
            message="Trade exceeded per-market limit",
            strategy_name="arb",
            details={"market_id": "test-123", "notional": 1000.0},
        )

        assert event_id > 0

        events = dao.get_risk_events(limit=10)
        assert len(events) == 1
        assert events[0]["severity"] == "WARN"
        assert events[0]["event_type"] == "LIMIT_BREACH"
        assert events[0]["details"]["market_id"] == "test-123"

    def test_get_risk_events_with_filters(self, dao) -> None:
        """Test filtering risk events."""
        dao.save_risk_event(
            severity="INFO",
            event_type="APPROVAL_GRANTED",
            message="Strategy approved",
        )
        dao.save_risk_event(
            severity="WARN",
            event_type="LIMIT_BREACH",
            message="Limit breached",
            strategy_name="arb",
        )
        dao.save_risk_event(
            severity="CRITICAL",
            event_type="KILL_TRIGGERED",
            message="Stop loss hit",
            strategy_name="arb",
        )

        all_events = dao.get_risk_events()
        assert len(all_events) == 3

        warn_events = dao.get_risk_events(severity="WARN")
        assert len(warn_events) == 1

        arb_events = dao.get_risk_events(strategy_name="arb")
        assert len(arb_events) == 2


class TestRiskGate:
    """Tests for RiskGate enforcement."""

    def test_check_approval_no_approval(self, dao) -> None:
        """Test check_approval when no approval exists."""
        gate = RiskGate(dao=dao)
        status = gate.check_approval("unknown_strategy")

        assert status.approved is False
        assert "no active approval" in status.reason.lower()

    def test_check_approval_approved(self, dao) -> None:
        """Test check_approval when strategy is approved."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
            limits={"max_notional_per_market": 1000.0},
        )

        gate = RiskGate(dao=dao)
        status = gate.check_approval("arb")

        assert status.approved is True
        assert status.strategy_name == "arb"
        assert status.limits is not None

    def test_enforce_approval_blocks_unapproved(self, dao) -> None:
        """Test enforce_approval raises error for unapproved strategy."""
        gate = RiskGate(dao=dao)

        with pytest.raises(PermissionError) as exc_info:
            gate.enforce_approval("unapproved_strategy")

        assert "not approved" in str(exc_info.value).lower()

    def test_enforce_approval_allows_override(self, dao) -> None:
        """Test enforce_approval allows override."""
        gate = RiskGate(dao=dao)

        # Should not raise with override
        status = gate.enforce_approval("unapproved_strategy", allow_override=True)

        assert status.approved is False
        assert status.limits is not None  # Default limits applied

    def test_enforce_approval_success(self, dao) -> None:
        """Test enforce_approval succeeds for approved strategy."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
        )

        gate = RiskGate(dao=dao)
        status = gate.enforce_approval("arb")

        assert status.approved is True

    def test_check_trade_limit_within_limits(self, dao) -> None:
        """Test trade limit check passes within limits."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
            limits={"max_notional_per_market": 1000.0, "max_total_notional": 5000.0},
        )

        gate = RiskGate(dao=dao)
        gate.enforce_approval("arb")

        allowed, reason = gate.check_trade_limit(
            market_id="test",
            notional=500.0,
            current_positions=5,
            current_total_notional=2000.0,
        )

        assert allowed is True

    def test_check_trade_limit_exceeds_per_market(self, dao) -> None:
        """Test trade limit check fails when exceeding per-market limit."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
            limits={"max_notional_per_market": 500.0},
        )

        gate = RiskGate(dao=dao)
        gate.enforce_approval("arb")

        allowed, reason = gate.check_trade_limit(
            market_id="test",
            notional=800.0,  # Exceeds 500 limit
            current_positions=5,
            current_total_notional=2000.0,
        )

        assert allowed is False
        assert "per-market" in reason.lower()

    def test_check_drawdown_stop(self, dao) -> None:
        """Test drawdown stop-loss check."""
        dao.save_approval(
            strategy_name="arb",
            strategy_version="v1",
            window_from="2024-01-01",
            window_to="2024-01-31",
            score=85.0,
            status="APPROVED",
            limits={"stop_loss_pct": 0.10},
        )

        gate = RiskGate(dao=dao)
        gate.enforce_approval("arb")

        # Within limit (5% drawdown)
        triggered, _ = gate.check_drawdown_stop(
            current_pnl=-500.0,
            peak_equity=10500.0,
            initial_balance=10000.0,
        )
        assert triggered is False

        # Exceeds stop-loss (12% drawdown from peak)
        triggered, reason = gate.check_drawdown_stop(
            current_pnl=-200.0,
            peak_equity=11000.0,  # Peak was 11000, now at 9800
            initial_balance=10000.0,
        )
        assert triggered is True
        assert "stop-loss" in reason.lower()
