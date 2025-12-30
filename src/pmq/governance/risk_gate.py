"""Risk gate enforcement for strategy execution.

The RiskGate ensures that:
1. Only approved strategies can run (unless explicitly overridden)
2. Risk limits are enforced during execution
3. Kill conditions trigger automatic stops
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from pmq.governance.scorecard import RiskLimits as RiskLimits
from pmq.logging import get_logger
from pmq.storage.dao import DAO

logger = get_logger("governance.risk_gate")


@dataclass
class ApprovalStatus:
    """Status of a strategy's approval."""

    approved: bool
    strategy_name: str
    strategy_version: str | None
    approval_id: int | None
    limits: RiskLimits | None
    reason: str


class RiskGate:
    """Enforces risk limits and approval requirements.

    The RiskGate is the central point for all risk enforcement.
    It checks approvals, enforces limits, and logs risk events.
    """

    def __init__(self, dao: DAO | None = None) -> None:
        """Initialize risk gate.

        Args:
            dao: Data access object
        """
        self._dao = dao or DAO()
        self._current_limits: RiskLimits | None = None
        self._trades_this_hour: int = 0
        self._hour_start: datetime | None = None

    def check_approval(
        self,
        strategy_name: str,
        strategy_version: str | None = None,
    ) -> ApprovalStatus:
        """Check if a strategy is approved.

        Args:
            strategy_name: Name of the strategy
            strategy_version: Optional version (uses latest if not specified)

        Returns:
            ApprovalStatus with approval details
        """
        approval = self._dao.get_active_approval(strategy_name, strategy_version)

        if not approval:
            self._log_event(
                severity="WARN",
                event_type="APPROVAL_CHECK",
                strategy_name=strategy_name,
                message=f"Strategy '{strategy_name}' has no active approval",
            )
            return ApprovalStatus(
                approved=False,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                approval_id=None,
                limits=None,
                reason=f"No active approval found for '{strategy_name}'",
            )

        # Parse limits from JSON
        limits = self._parse_limits(approval.get("limits_json"))

        self._log_event(
            severity="INFO",
            event_type="APPROVAL_CHECK",
            strategy_name=strategy_name,
            message=f"Strategy '{strategy_name}' v{approval['strategy_version']} is APPROVED",
            details={"approval_id": approval["id"]},
        )

        return ApprovalStatus(
            approved=True,
            strategy_name=approval["strategy_name"],
            strategy_version=approval["strategy_version"],
            approval_id=approval["id"],
            limits=limits,
            reason="Approved",
        )

    def enforce_approval(
        self,
        strategy_name: str,
        strategy_version: str | None = None,
        allow_override: bool = False,
    ) -> ApprovalStatus:
        """Enforce approval requirement for execution.

        Args:
            strategy_name: Name of the strategy
            strategy_version: Optional version
            allow_override: If True, allows running without approval (with warning)

        Returns:
            ApprovalStatus

        Raises:
            PermissionError: If strategy not approved and override not allowed
        """
        status = self.check_approval(strategy_name, strategy_version)

        if not status.approved:
            if allow_override:
                self._log_event(
                    severity="WARN",
                    event_type="OVERRIDE_USED",
                    strategy_name=strategy_name,
                    message=f"Running unapproved strategy '{strategy_name}' with override",
                )
                logger.warning(f"OVERRIDE: Running unapproved strategy '{strategy_name}'")
                # Use default limits for override mode
                status.limits = RiskLimits()
                return status
            else:
                self._log_event(
                    severity="CRITICAL",
                    event_type="EXECUTION_BLOCKED",
                    strategy_name=strategy_name,
                    message=f"Blocked execution of unapproved strategy '{strategy_name}'",
                )
                raise PermissionError(
                    f"Strategy '{strategy_name}' is not approved. "
                    "Use --override-unsafe to bypass (not recommended)."
                )

        # Store limits for enforcement
        self._current_limits = status.limits
        return status

    def check_trade_limit(
        self,
        market_id: str,
        notional: float,
        current_positions: int,
        current_total_notional: float,
    ) -> tuple[bool, str]:
        """Check if a trade is within limits.

        Args:
            market_id: Market being traded
            notional: Trade notional amount
            current_positions: Current number of open positions
            current_total_notional: Current total notional exposure

        Returns:
            Tuple of (allowed, reason)
        """
        if self._current_limits is None:
            return True, "No limits set"

        limits = self._current_limits

        # Check per-market notional
        if notional > limits.max_notional_per_market:
            reason = f"Trade notional ${notional:.2f} exceeds per-market limit ${limits.max_notional_per_market:.2f}"
            self._log_event(
                severity="WARN",
                event_type="LIMIT_BREACH",
                message=reason,
                details={"market_id": market_id, "notional": notional},
            )
            return False, reason

        # Check total notional
        if current_total_notional + notional > limits.max_total_notional:
            reason = f"Total notional would exceed limit ${limits.max_total_notional:.2f}"
            self._log_event(
                severity="WARN",
                event_type="LIMIT_BREACH",
                message=reason,
                details={"new_total": current_total_notional + notional},
            )
            return False, reason

        # Check position count
        if current_positions >= limits.max_positions:
            reason = f"Position count {current_positions} at limit {limits.max_positions}"
            self._log_event(
                severity="WARN",
                event_type="LIMIT_BREACH",
                message=reason,
            )
            return False, reason

        # Check trades per hour
        self._update_hourly_counter()
        if self._trades_this_hour >= limits.max_trades_per_hour:
            reason = f"Trades this hour ({self._trades_this_hour}) at limit ({limits.max_trades_per_hour})"
            self._log_event(
                severity="WARN",
                event_type="LIMIT_BREACH",
                message=reason,
            )
            return False, reason

        return True, "OK"

    def record_trade(self) -> None:
        """Record that a trade was executed (for rate limiting)."""
        self._update_hourly_counter()
        self._trades_this_hour += 1

    def check_drawdown_stop(
        self,
        current_pnl: float,
        peak_equity: float,
        initial_balance: float,
    ) -> tuple[bool, str]:
        """Check if drawdown stop-loss should trigger.

        Args:
            current_pnl: Current P&L
            peak_equity: Peak equity value
            initial_balance: Initial balance

        Returns:
            Tuple of (should_stop, reason)
        """
        if self._current_limits is None:
            return False, "No limits set"

        current_equity = initial_balance + current_pnl
        if peak_equity <= 0:
            return False, "Invalid peak equity"

        drawdown = (peak_equity - current_equity) / peak_equity

        if drawdown >= self._current_limits.stop_loss_pct:
            reason = f"Drawdown {drawdown:.1%} exceeds stop-loss {self._current_limits.stop_loss_pct:.1%}"
            self._log_event(
                severity="CRITICAL",
                event_type="KILL_TRIGGERED",
                message=reason,
                details={
                    "drawdown": drawdown,
                    "limit": self._current_limits.stop_loss_pct,
                },
            )
            return True, reason

        return False, "OK"

    def check_data_quality(self) -> tuple[bool, str]:
        """Check if current data quality meets requirements.

        Returns:
            Tuple of (quality_ok, reason)
        """
        if self._current_limits is None:
            return True, "No limits set"

        # Get latest quality report
        report = self._dao.get_latest_quality_report()
        if not report:
            return True, "No quality report available"

        coverage_pct = report.get("coverage_pct", 100.0)
        min_required = self._current_limits.min_data_quality_pct

        if coverage_pct < min_required:
            reason = f"Data quality {coverage_pct:.1f}% below minimum {min_required:.1f}%"
            self._log_event(
                severity="WARN",
                event_type="DATA_QUALITY_FAIL",
                message=reason,
                details={"coverage_pct": coverage_pct, "min_required": min_required},
            )
            return False, reason

        return True, "OK"

    def _update_hourly_counter(self) -> None:
        """Reset hourly trade counter if hour has changed."""
        now = datetime.now(UTC)
        if self._hour_start is None or (now - self._hour_start).total_seconds() >= 3600:
            self._hour_start = now
            self._trades_this_hour = 0

    def _parse_limits(self, limits_json: str | None) -> RiskLimits:
        """Parse limits from JSON string."""
        if not limits_json:
            return RiskLimits()

        try:
            data = json.loads(limits_json)
            return RiskLimits(
                max_notional_per_market=data.get("max_notional_per_market", 500.0),
                max_total_notional=data.get("max_total_notional", 5000.0),
                max_positions=data.get("max_positions", 20),
                max_trades_per_hour=data.get("max_trades_per_hour", 50),
                stop_loss_pct=data.get("stop_loss_pct", 0.10),
                min_data_quality_pct=data.get("min_data_quality_pct", 80.0),
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse limits JSON, using defaults")
            return RiskLimits()

    def _log_event(
        self,
        severity: str,
        event_type: str,
        message: str,
        strategy_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a risk event to the database."""
        self._dao.save_risk_event(
            severity=severity,
            event_type=event_type,
            message=message,
            strategy_name=strategy_name,
            details=details,
        )


def limits_to_dict(limits: RiskLimits) -> dict[str, Any]:
    """Convert RiskLimits to dict for JSON serialization."""
    return asdict(limits)
