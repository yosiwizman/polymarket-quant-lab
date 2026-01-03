"""Live execution engine for real order placement.

Phase 12: Implements LIVE execution with multiple safety layers:

SAFETY MODEL:
1. Disabled by default (--live-exec flag required)
2. Dry-run by default (--live-exec-confirm required to post)
3. TTL approval required (scope: live_exec, separate from paper_exec)
4. Kill switch file check (~/.pmq/KILL)
5. Rate limits (max orders per hour)
6. Size limits (max USD per order)
7. Net edge minimum threshold

SUPPORTED OPERATIONS:
- BUY_BOTH: Place 2 BUY orders (YES + NO) when sum < 1.0
- SELL_BOTH: DISABLED by default (requires position mechanics not yet implemented)

USAGE:
    from pmq.ops.live_exec import LiveExecutor, LiveExecConfig

    config = LiveExecConfig(
        enabled=True,
        confirm=True,  # Actually post orders
        max_order_usd=5.0,
        max_orders_per_hour=2,
    )
    executor = LiveExecutor(config, client, ledger)
    result = executor.execute(trade_plan, now=datetime.now(UTC))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pmq.logging import get_logger
from pmq.ops.edge_calc import ArbSide

if TYPE_CHECKING:
    from pmq.auth.client_factory import ClobClientProtocol
    from pmq.ops.live_ledger import LiveLedger

logger = get_logger("ops.live_exec")

# Default kill switch file path
DEFAULT_KILL_SWITCH_PATH = Path.home() / ".pmq" / "KILL"

# Minimum net edge required (conservative default)
DEFAULT_MIN_NET_EDGE_BPS = 10.0

# Default rate/size limits
DEFAULT_MAX_ORDER_USD = 5.0
DEFAULT_MAX_ORDERS_PER_HOUR = 2


class LiveExecStatus(str, Enum):
    """Status of live execution attempt."""

    SUCCESS = "SUCCESS"  # Order(s) posted successfully
    DRY_RUN = "DRY_RUN"  # Would have posted (dry run mode)
    REJECTED = "REJECTED"  # Rejected by safety checks
    ERROR = "ERROR"  # Execution error
    DISABLED = "DISABLED"  # Live exec not enabled


class RejectionReason(str, Enum):
    """Reason for rejecting a live execution."""

    NOT_ENABLED = "not_enabled"  # --live-exec not set
    NOT_CONFIRMED = "not_confirmed"  # --live-exec-confirm not set
    APPROVAL_MISSING = "approval_missing"  # No TTL approval for live_exec
    APPROVAL_EXPIRED = "approval_expired"  # TTL approval expired
    KILL_SWITCH = "kill_switch"  # Kill switch file exists
    RATE_LIMIT = "rate_limit"  # Too many orders in last hour
    SIZE_LIMIT = "size_limit"  # Order size exceeds limit
    EDGE_TOO_LOW = "edge_too_low"  # Net edge below minimum
    SELL_BOTH_DISABLED = "sell_both_disabled"  # SELL_BOTH not supported
    NO_VALID_ARB = "no_valid_arb"  # No valid arbitrage opportunity
    MISSING_TOKEN_ID = "missing_token_id"  # Token ID not available
    ORDERBOOK_ERROR = "orderbook_error"  # Cannot compute order prices
    CLIENT_ERROR = "client_error"  # CLOB client error


@dataclass
class TradePlan:
    """Plan for a live trade.

    Contains all information needed to execute an arbitrage trade.
    """

    market_id: str
    yes_token_id: str
    no_token_id: str

    # Arbitrage type
    arb_side: ArbSide  # BUY_BOTH or SELL_BOTH

    # Computed prices from orderbook
    yes_limit_price: float  # Limit price for YES order
    no_limit_price: float  # Limit price for NO order

    # Size (same for both legs)
    size: float  # Quantity in shares

    # Edge metrics
    gross_edge_bps: float  # Edge before fees
    net_edge_bps: float  # Edge after fees/slippage

    # Context
    market_question: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class OrderResult:
    """Result of a single order placement."""

    token_id: str
    outcome: str  # "YES" or "NO"
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    order_id: str | None = None
    status: str = "PENDING"  # "POSTED", "REJECTED", "ERROR", "DRY_RUN"
    error: str | None = None


@dataclass
class LiveExecResult:
    """Result of live execution attempt."""

    status: LiveExecStatus
    rejection_reason: RejectionReason | None = None
    rejection_detail: str = ""

    # Orders placed (if any)
    orders: list[OrderResult] = field(default_factory=list)

    # Statistics
    total_notional_usd: float = 0.0
    orders_posted: int = 0

    # Context
    dry_run: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class LiveExecConfig:
    """Configuration for live execution.

    SAFE DEFAULTS:
    - enabled: False (must explicitly enable with --live-exec)
    - confirm: False (must explicitly confirm with --live-exec-confirm)
    """

    # Master switches
    enabled: bool = False  # Must be True to consider live execution
    confirm: bool = False  # Must be True to actually post orders (dry-run otherwise)

    # Limits
    max_order_usd: float = DEFAULT_MAX_ORDER_USD
    max_orders_per_hour: int = DEFAULT_MAX_ORDERS_PER_HOUR
    min_net_edge_bps: float = DEFAULT_MIN_NET_EDGE_BPS

    # Slippage guard: adjust limit price by this many bps
    slippage_guard_bps: float = 5.0

    # Kill switch
    kill_switch_path: Path = field(default_factory=lambda: DEFAULT_KILL_SWITCH_PATH)

    # TTL approval settings
    approval_scope: str = "live_exec"  # Must be separate from paper_exec


class ApprovalChecker(Protocol):
    """Protocol for checking TTL approvals."""

    def __call__(self, scope: str, now: datetime | None = None) -> tuple[bool, str]:
        """Check if scope is approved.

        Returns:
            Tuple of (is_approved, detail_message)
        """
        ...


def default_approval_checker(scope: str, now: datetime | None = None) -> tuple[bool, str]:
    """Default approval checker using risk/approval module.

    Returns:
        Tuple of (is_approved, detail_message)
    """
    from pmq.risk.approval import get_approval, is_approved

    if now is None:
        now = datetime.now(UTC)

    # Check if approved
    if not is_approved(scope, now=now):
        # Get approval to see why (expired vs missing)
        approval = get_approval(scope)
        if approval is None:
            return False, f"No TTL approval found for scope '{scope}'"
        elif not approval.is_valid(now):
            remaining = approval.time_remaining(now)
            return (
                False,
                f"TTL approval for '{scope}' expired {-remaining.total_seconds():.0f}s ago",
            )
        else:
            return False, f"TTL approval for '{scope}' is invalid"

    # Get approval details for logging
    approval = get_approval(scope)
    if approval:
        remaining = approval.time_remaining(now)
        return True, f"TTL approval valid for {remaining.total_seconds():.0f}s"

    return True, "Approved"


class LiveExecutor:
    """Executes live trades with safety checks.

    HARD RULES:
    1. No execution unless enabled=True AND confirm=True AND approval valid
    2. SELL_BOTH is DISABLED (requires position mechanics not implemented)
    3. Every attempt is logged to LiveLedger
    4. Kill switch file stops all execution
    """

    def __init__(
        self,
        config: LiveExecConfig,
        client: ClobClientProtocol | None = None,
        ledger: LiveLedger | None = None,
        approval_checker: ApprovalChecker | None = None,
    ) -> None:
        """Initialize live executor.

        Args:
            config: Execution configuration
            client: CLOB client for order placement (None for dry-run only)
            ledger: Ledger for recording orders
            approval_checker: Function to check TTL approvals
        """
        self.config = config
        self._client = client
        self._ledger = ledger
        self._approval_checker = approval_checker or default_approval_checker

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists.

        Returns:
            True if kill switch is active (file exists)
        """
        return self.config.kill_switch_path.exists()

    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows more orders.

        Returns:
            True if rate limit allows more orders
        """
        if self._ledger is None:
            return True

        orders_last_hour = self._ledger.get_orders_last_hour()
        return orders_last_hour < self.config.max_orders_per_hour

    def _check_size_limit(self, notional_usd: float) -> bool:
        """Check if order size is within limit.

        Args:
            notional_usd: Total notional value of order(s)

        Returns:
            True if size is within limit
        """
        return notional_usd <= self.config.max_order_usd

    def _compute_limit_prices(
        self,
        plan: TradePlan,
    ) -> tuple[float, float]:
        """Compute limit prices with slippage guard.

        For BUY orders, we add slippage to the ask price.
        For SELL orders, we subtract slippage from the bid price.

        Args:
            plan: Trade plan with base prices

        Returns:
            Tuple of (yes_limit_price, no_limit_price)
        """
        slippage_mult = self.config.slippage_guard_bps / 10000

        if plan.arb_side == ArbSide.BUY_BOTH:
            # Buying: willing to pay a bit more than the ask
            yes_price = min(0.99, plan.yes_limit_price * (1 + slippage_mult))
            no_price = min(0.99, plan.no_limit_price * (1 + slippage_mult))
        else:
            # Selling: willing to receive a bit less than the bid
            yes_price = max(0.01, plan.yes_limit_price * (1 - slippage_mult))
            no_price = max(0.01, plan.no_limit_price * (1 - slippage_mult))

        return yes_price, no_price

    def execute(
        self,
        plan: TradePlan,
        now: datetime | None = None,
    ) -> LiveExecResult:
        """Execute a live trade plan.

        Goes through multiple safety checks before placing orders.

        Args:
            plan: Trade plan to execute
            now: Current time (for TTL checks)

        Returns:
            LiveExecResult with status and details
        """
        if now is None:
            now = datetime.now(UTC)

        # =========================================
        # Safety Check 1: Is live exec enabled?
        # =========================================
        if not self.config.enabled:
            return LiveExecResult(
                status=LiveExecStatus.DISABLED,
                rejection_reason=RejectionReason.NOT_ENABLED,
                rejection_detail="Live execution not enabled. Use --live-exec flag.",
            )

        # =========================================
        # Safety Check 2: Kill switch
        # =========================================
        if self._check_kill_switch():
            logger.warning(f"Kill switch active: {self.config.kill_switch_path}")
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.KILL_SWITCH,
                rejection_detail=f"Kill switch file exists: {self.config.kill_switch_path}",
            )

        # =========================================
        # Safety Check 3: TTL Approval
        # =========================================
        approved, approval_detail = self._approval_checker(self.config.approval_scope, now)
        if not approved:
            logger.warning(f"TTL approval check failed: {approval_detail}")
            # Determine if missing or expired
            reason = RejectionReason.APPROVAL_MISSING
            if "expired" in approval_detail.lower():
                reason = RejectionReason.APPROVAL_EXPIRED
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=reason,
                rejection_detail=approval_detail,
            )

        # =========================================
        # Safety Check 4: SELL_BOTH disabled
        # =========================================
        if plan.arb_side == ArbSide.SELL_BOTH:
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.SELL_BOTH_DISABLED,
                rejection_detail=(
                    "SELL_BOTH is disabled. Polymarket position mechanics for selling "
                    "both sides of a position are not yet implemented."
                ),
            )

        # =========================================
        # Safety Check 5: Valid arb side
        # =========================================
        if plan.arb_side == ArbSide.NONE:
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.NO_VALID_ARB,
                rejection_detail="No valid arbitrage opportunity (arb_side=NONE)",
            )

        # =========================================
        # Safety Check 6: Minimum net edge
        # =========================================
        if plan.net_edge_bps < self.config.min_net_edge_bps:
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.EDGE_TOO_LOW,
                rejection_detail=(
                    f"Net edge {plan.net_edge_bps:.1f} bps < minimum "
                    f"{self.config.min_net_edge_bps:.1f} bps"
                ),
            )

        # =========================================
        # Safety Check 7: Token IDs present
        # =========================================
        if not plan.yes_token_id or not plan.no_token_id:
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.MISSING_TOKEN_ID,
                rejection_detail="YES or NO token ID is missing",
            )

        # =========================================
        # Safety Check 8: Size limit
        # =========================================
        # Compute total notional for both legs
        yes_price, no_price = self._compute_limit_prices(plan)
        total_notional = (yes_price * plan.size) + (no_price * plan.size)

        if not self._check_size_limit(total_notional):
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.SIZE_LIMIT,
                rejection_detail=(
                    f"Total notional ${total_notional:.2f} > max ${self.config.max_order_usd:.2f}"
                ),
            )

        # =========================================
        # Safety Check 9: Rate limit
        # =========================================
        if not self._check_rate_limit():
            stats = self._ledger.get_stats() if self._ledger else None
            orders_last_hour = stats.orders_last_hour if stats else "?"
            return LiveExecResult(
                status=LiveExecStatus.REJECTED,
                rejection_reason=RejectionReason.RATE_LIMIT,
                rejection_detail=(
                    f"Rate limit: {orders_last_hour} orders in last hour >= "
                    f"max {self.config.max_orders_per_hour}"
                ),
            )

        # =========================================
        # Safety Check 10: Confirm flag
        # =========================================
        if not self.config.confirm:
            # Dry run - log what would happen
            logger.info(
                f"DRY RUN: Would place BUY_BOTH orders: "
                f"YES @ {yes_price:.4f} x {plan.size:.2f}, "
                f"NO @ {no_price:.4f} x {plan.size:.2f}"
            )
            result = LiveExecResult(
                status=LiveExecStatus.DRY_RUN,
                dry_run=True,
                total_notional_usd=total_notional,
            )

            # Record dry run in ledger
            if self._ledger:
                from pmq.ops.live_ledger import LiveOrderRecord

                for outcome, token_id, price in [
                    ("YES", plan.yes_token_id, yes_price),
                    ("NO", plan.no_token_id, no_price),
                ]:
                    record = LiveOrderRecord(
                        order_id=None,
                        timestamp=now.isoformat(),
                        market_id=plan.market_id,
                        token_id=token_id,
                        outcome=outcome,
                        side="BUY",
                        price=price,
                        size=plan.size,
                        notional_usd=price * plan.size,
                        status="DRY_RUN",
                        arb_side=plan.arb_side.value,
                        edge_bps=plan.net_edge_bps,
                        dry_run=True,
                    )
                    self._ledger.record_order(record)
                    result.orders.append(
                        OrderResult(
                            token_id=token_id,
                            outcome=outcome,
                            side="BUY",
                            price=price,
                            size=plan.size,
                            status="DRY_RUN",
                        )
                    )

            return result

        # =========================================
        # All checks passed - EXECUTE ORDERS
        # =========================================
        if self._client is None:
            return LiveExecResult(
                status=LiveExecStatus.ERROR,
                rejection_reason=RejectionReason.CLIENT_ERROR,
                rejection_detail="No CLOB client configured for live execution",
            )

        result = LiveExecResult(
            status=LiveExecStatus.SUCCESS,
            total_notional_usd=total_notional,
        )

        # Place both orders
        orders_to_place = [
            ("YES", plan.yes_token_id, yes_price),
            ("NO", plan.no_token_id, no_price),
        ]

        for outcome, token_id, price in orders_to_place:
            order_result = self._place_order(
                token_id=token_id,
                outcome=outcome,
                side="BUY",
                price=price,
                size=plan.size,
                market_id=plan.market_id,
                arb_side=plan.arb_side,
                edge_bps=plan.net_edge_bps,
                now=now,
            )
            result.orders.append(order_result)
            if order_result.status == "POSTED":
                result.orders_posted += 1

        # If any order failed, mark result as error
        if result.orders_posted < len(orders_to_place):
            result.status = LiveExecStatus.ERROR
            errors = [o.error for o in result.orders if o.error]
            result.rejection_detail = "; ".join(errors) if errors else "Partial failure"

        return result

    def _place_order(
        self,
        token_id: str,
        outcome: str,
        side: str,
        price: float,
        size: float,
        market_id: str,
        arb_side: ArbSide,
        edge_bps: float,
        now: datetime,
    ) -> OrderResult:
        """Place a single order via CLOB client.

        Args:
            token_id: Token to trade
            outcome: "YES" or "NO"
            side: "BUY" or "SELL"
            price: Limit price
            size: Quantity
            market_id: Market ID for logging
            arb_side: Arbitrage type
            edge_bps: Edge at time of order
            now: Current timestamp

        Returns:
            OrderResult with status
        """
        from pmq.ops.live_ledger import LiveOrderRecord

        try:
            # Create order using py_clob_client
            # Note: This requires the py_clob_client.order module
            from py_clob_client.order import (
                BUY,
                SELL,
                OrderArgs,
            )

            order_side = BUY if side == "BUY" else SELL

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side,
            )

            # Post the order
            response = self._client.create_and_post_order(order_args)  # type: ignore

            # Extract order ID from response
            order_id = response.get("orderID") or response.get("order_id")

            logger.info(f"Order posted: {side} {outcome} {size:.2f} @ {price:.4f} -> {order_id}")

            # Record in ledger
            if self._ledger:
                record = LiveOrderRecord(
                    order_id=order_id,
                    timestamp=now.isoformat(),
                    market_id=market_id,
                    token_id=token_id,
                    outcome=outcome,
                    side=side,
                    price=price,
                    size=size,
                    notional_usd=price * size,
                    status="POSTED",
                    arb_side=arb_side.value,
                    edge_bps=edge_bps,
                    dry_run=False,
                )
                self._ledger.record_order(record)

            return OrderResult(
                token_id=token_id,
                outcome=outcome,
                side=side,
                price=price,
                size=size,
                order_id=order_id,
                status="POSTED",
            )

        except Exception as e:
            # Redact any secrets from error
            from pmq.auth.redact import redact_secrets

            safe_error = redact_secrets(str(e))
            logger.error(f"Order failed: {side} {outcome} @ {price:.4f}: {safe_error}")

            # Record error in ledger
            if self._ledger:
                record = LiveOrderRecord(
                    order_id=None,
                    timestamp=now.isoformat(),
                    market_id=market_id,
                    token_id=token_id,
                    outcome=outcome,
                    side=side,
                    price=price,
                    size=size,
                    notional_usd=price * size,
                    status="ERROR",
                    error_message=safe_error,
                    arb_side=arb_side.value,
                    edge_bps=edge_bps,
                    dry_run=False,
                )
                self._ledger.record_order(record)

            return OrderResult(
                token_id=token_id,
                outcome=outcome,
                side=side,
                price=price,
                size=size,
                status="ERROR",
                error=safe_error,
            )


def create_trade_plan_from_edge(
    market_id: str,
    yes_token_id: str,
    no_token_id: str,
    arb_side: ArbSide,
    ask_yes: float,
    ask_no: float,
    size: float,
    gross_edge_bps: float,
    net_edge_bps: float,
    market_question: str = "",
) -> TradePlan:
    """Create a trade plan from edge computation results.

    For BUY_BOTH, uses ask prices as limit prices.
    For SELL_BOTH, would use bid prices (but SELL_BOTH is disabled).

    Args:
        market_id: Market identifier
        yes_token_id: YES token ID
        no_token_id: NO token ID
        arb_side: BUY_BOTH or SELL_BOTH
        ask_yes: Best ask for YES
        ask_no: Best ask for NO
        size: Quantity per leg
        gross_edge_bps: Edge before fees
        net_edge_bps: Edge after fees
        market_question: Market description

    Returns:
        TradePlan ready for execution
    """
    return TradePlan(
        market_id=market_id,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        arb_side=arb_side,
        yes_limit_price=ask_yes,
        no_limit_price=ask_no,
        size=size,
        gross_edge_bps=gross_edge_bps,
        net_edge_bps=net_edge_bps,
        market_question=market_question,
    )
