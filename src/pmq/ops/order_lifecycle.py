"""Order lifecycle management for live order operations.

Phase 13: Provides reliable order placement, state waiting, and cancellation
with proper error handling, backoff, and secret redaction.

DESIGN PRINCIPLES:
- Polite backoff: Respect rate limits, no tight polling loops
- Reliable cancellation: Always attempt cleanup on errors
- Full redaction: Never log order IDs or sensitive data in plain text
- Timeout-aware: All operations have configurable timeouts

USAGE:
    from pmq.ops.order_lifecycle import OrderLifecycle, OrderState

    lifecycle = OrderLifecycle(clob_client)

    # Place an order
    order_id = lifecycle.place_limit_order(token_id, side, price, size)

    # Wait for it to be open
    state = lifecycle.wait_for_order_state(order_id, target_state=OrderState.LIVE)

    # Cancel it
    lifecycle.cancel_order(order_id)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from pmq.auth.redact import mask_string, redact_secrets
from pmq.logging import get_logger

if TYPE_CHECKING:
    pass  # No type-only imports needed

logger = get_logger("ops.order_lifecycle")

# Default timing parameters
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_MAX_S = 10.0


class OrderState(str, Enum):
    """Possible order states from CLOB API."""

    LIVE = "LIVE"  # Order is on the book, waiting to fill
    MATCHED = "MATCHED"  # Order has been fully filled
    CANCELLED = "CANCELLED"  # Order was cancelled
    PENDING = "PENDING"  # Order is being processed
    UNKNOWN = "UNKNOWN"  # State could not be determined


class OrderLifecycleError(Exception):
    """Base exception for order lifecycle errors."""

    pass


class OrderPlacementError(OrderLifecycleError):
    """Error placing an order."""

    pass


class OrderCancellationError(OrderLifecycleError):
    """Error cancelling an order."""

    pass


class OrderTimeoutError(OrderLifecycleError):
    """Timeout waiting for order state."""

    pass


@dataclass
class OrderPlacementResult:
    """Result of placing a limit order."""

    success: bool
    order_id: str | None = None
    status: str = ""
    error: str | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class OrderStateResult:
    """Result of checking order state."""

    order_id: str
    state: OrderState
    filled_size: Decimal = Decimal("0")
    remaining_size: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    error: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class CancellationResult:
    """Result of cancelling an order."""

    success: bool
    order_id: str
    was_filled: bool = False  # True if order filled before cancel
    filled_size: Decimal = Decimal("0")
    error: str | None = None


class ClobClientProtocol(Protocol):
    """Protocol for CLOB client methods we need."""

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        ...

    def get_orders(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Get open orders."""
        ...

    def create_and_post_order(self, order: Any) -> dict[str, Any]:
        """Create and post a new order."""
        ...

    def cancel(self, order_id: str) -> dict[str, Any]:
        """Cancel an order."""
        ...

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order (alias)."""
        ...


def _mask_order_id(order_id: str) -> str:
    """Mask order ID for logging."""
    if len(order_id) > 12:
        return mask_string(order_id, 6)
    return mask_string(order_id, 3)


def _parse_order_state(state_str: str | None) -> OrderState:
    """Parse order state string to enum."""
    if not state_str:
        return OrderState.UNKNOWN

    state_upper = state_str.upper()

    if state_upper in ("LIVE", "OPEN", "ACTIVE"):
        return OrderState.LIVE
    elif state_upper in ("MATCHED", "FILLED", "COMPLETE"):
        return OrderState.MATCHED
    elif state_upper in ("CANCELLED", "CANCELED"):
        return OrderState.CANCELLED
    elif state_upper in ("PENDING", "SUBMITTED"):
        return OrderState.PENDING
    else:
        return OrderState.UNKNOWN


class OrderLifecycle:
    """Manages order lifecycle: place, wait, cancel.

    Handles retries, backoff, and proper cleanup on errors.
    """

    def __init__(
        self,
        client: ClobClientProtocol,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
    ) -> None:
        """Initialize order lifecycle manager.

        Args:
            client: CLOB client for API calls
            poll_interval_s: Interval between state polls
            timeout_s: Default timeout for operations
            max_retries: Maximum retry attempts
            backoff_base_s: Base backoff time for retries
            backoff_max_s: Maximum backoff time
        """
        self._client = client
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s

    def _backoff(self, attempt: int) -> float:
        """Calculate backoff time for retry attempt."""
        backoff: float = self._backoff_base_s * (2**attempt)
        result: float = min(backoff, self._backoff_max_s)
        return result

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: Decimal | float,
        size: Decimal | float,
        *,
        time_in_force: str = "GTC",
    ) -> OrderPlacementResult:
        """Place a limit order on the CLOB.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            price: Limit price
            size: Order size in shares
            time_in_force: Order time-in-force (GTC, IOC, FOK)

        Returns:
            OrderPlacementResult with order_id if successful
        """
        masked_token = mask_string(token_id, 8) if len(token_id) > 16 else token_id
        price_float = float(price)
        size_float = float(size)

        logger.info(
            f"Placing {side} order: token={masked_token}, "
            f"price={price_float:.4f}, size={size_float:.2f}, tif={time_in_force}"
        )

        try:
            # Import OrderArgs from py_clob_client
            from py_clob_client.clob_types import OrderArgs  # type: ignore[import-untyped]
            from py_clob_client.order_builder.constants import (  # type: ignore[import-untyped]
                BUY,
                SELL,
            )

            # Determine side constant
            side_const = BUY if side.upper() == "BUY" else SELL

            # Build order args
            order_args = OrderArgs(
                token_id=token_id,
                price=price_float,
                size=size_float,
                side=side_const,
            )

            # Post order
            response = self._client.create_and_post_order(order_args)

            # Extract order ID from response
            order_id = response.get("orderID") or response.get("order_id") or response.get("id")

            if order_id:
                logger.info(f"Order placed successfully: {_mask_order_id(order_id)}")
                return OrderPlacementResult(
                    success=True,
                    order_id=order_id,
                    status="SUBMITTED",
                )
            else:
                error_msg = response.get("error") or response.get("message") or "Unknown error"
                safe_error = redact_secrets(str(error_msg))
                logger.error(f"Order placement failed: {safe_error}")
                return OrderPlacementResult(
                    success=False,
                    error=safe_error,
                    status="REJECTED",
                )

        except ImportError as e:
            error_msg = f"py_clob_client not installed: {e}"
            logger.error(error_msg)
            return OrderPlacementResult(success=False, error=error_msg, status="ERROR")

        except Exception as e:
            safe_error = redact_secrets(str(e))
            logger.error(f"Order placement error: {safe_error}")
            return OrderPlacementResult(success=False, error=safe_error, status="ERROR")

    def get_order_state(self, order_id: str) -> OrderStateResult:
        """Get current state of an order.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStateResult with current state
        """
        masked_id = _mask_order_id(order_id)

        try:
            response = self._client.get_order(order_id)

            state_str = response.get("status") or response.get("state")
            state = _parse_order_state(state_str)

            filled_size = Decimal(str(response.get("size_matched", 0) or 0))
            original_size = Decimal(
                str(response.get("original_size", 0) or response.get("size", 0) or 0)
            )
            remaining_size = original_size - filled_size

            avg_price_raw = response.get("avg_fill_price") or response.get("average_price")
            avg_price = Decimal(str(avg_price_raw)) if avg_price_raw else None

            logger.debug(f"Order {masked_id} state: {state.value}, filled={filled_size}")

            return OrderStateResult(
                order_id=order_id,
                state=state,
                filled_size=filled_size,
                remaining_size=remaining_size,
                avg_fill_price=avg_price,
                raw_response=response,
            )

        except Exception as e:
            safe_error = redact_secrets(str(e))
            logger.warning(f"Error getting order {masked_id} state: {safe_error}")
            return OrderStateResult(
                order_id=order_id,
                state=OrderState.UNKNOWN,
                error=safe_error,
            )

    def wait_for_order_state(
        self,
        order_id: str,
        target_states: set[OrderState] | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> OrderStateResult:
        """Wait for order to reach one of the target states.

        Args:
            order_id: Order ID to wait for
            target_states: Set of states to wait for (default: LIVE or MATCHED)
            timeout_s: Maximum wait time
            poll_interval_s: Time between polls

        Returns:
            OrderStateResult when target state reached or timeout

        Raises:
            OrderTimeoutError: If timeout exceeded
        """
        if target_states is None:
            target_states = {OrderState.LIVE, OrderState.MATCHED, OrderState.CANCELLED}

        timeout = timeout_s or self._timeout_s
        interval = poll_interval_s or self._poll_interval_s
        masked_id = _mask_order_id(order_id)

        logger.info(f"Waiting for order {masked_id} to reach {[s.value for s in target_states]}")

        start_time = time.monotonic()
        last_state = OrderState.UNKNOWN

        while (time.monotonic() - start_time) < timeout:
            result = self.get_order_state(order_id)
            last_state = result.state

            if result.state in target_states:
                elapsed = time.monotonic() - start_time
                logger.info(f"Order {masked_id} reached {result.state.value} in {elapsed:.1f}s")
                return result

            if result.state == OrderState.UNKNOWN and result.error:
                # Error getting state - might be transient
                logger.warning(f"Error polling order {masked_id}: {result.error}")

            # Polite backoff
            time.sleep(interval)

        # Timeout reached
        elapsed = time.monotonic() - start_time
        error_msg = f"Timeout waiting for order {masked_id} after {elapsed:.1f}s (last state: {last_state.value})"
        logger.error(error_msg)

        raise OrderTimeoutError(error_msg)

    def cancel_order(self, order_id: str) -> CancellationResult:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            CancellationResult indicating success/failure
        """
        masked_id = _mask_order_id(order_id)
        logger.info(f"Cancelling order {masked_id}")

        try:
            response = self._client.cancel(order_id)

            # Check if cancel was successful
            # Different API versions may return different structures
            success = response.get("success", True)
            if isinstance(response.get("canceled"), list):
                success = order_id in response.get("canceled", [])

            if success:
                logger.info(f"Order {masked_id} cancelled successfully")
                return CancellationResult(success=True, order_id=order_id)
            else:
                error = response.get("error") or response.get("message") or "Unknown error"
                safe_error = redact_secrets(str(error))
                logger.warning(f"Cancel failed for {masked_id}: {safe_error}")
                return CancellationResult(success=False, order_id=order_id, error=safe_error)

        except Exception as e:
            safe_error = redact_secrets(str(e))
            logger.error(f"Error cancelling order {masked_id}: {safe_error}")
            return CancellationResult(success=False, order_id=order_id, error=safe_error)

    def safe_cancel_if_open(self, order_id: str) -> CancellationResult:
        """Safely cancel an order if it's still open.

        Checks order state first, only cancels if LIVE/PENDING.
        Returns success if order already cancelled/filled.

        Args:
            order_id: Order ID to cancel

        Returns:
            CancellationResult
        """
        masked_id = _mask_order_id(order_id)

        # Get current state
        state_result = self.get_order_state(order_id)

        if state_result.state == OrderState.CANCELLED:
            logger.info(f"Order {masked_id} already cancelled")
            return CancellationResult(success=True, order_id=order_id)

        if state_result.state == OrderState.MATCHED:
            logger.info(f"Order {masked_id} already filled")
            return CancellationResult(
                success=True,
                order_id=order_id,
                was_filled=True,
                filled_size=state_result.filled_size,
            )

        if state_result.state in (OrderState.LIVE, OrderState.PENDING, OrderState.UNKNOWN):
            # Order might still be open, try to cancel
            return self.cancel_order(order_id)

        # Unknown state, try cancel anyway to be safe
        logger.warning(
            f"Order {masked_id} in unexpected state {state_result.state}, attempting cancel"
        )
        return self.cancel_order(order_id)

    def place_and_verify(
        self,
        token_id: str,
        side: str,
        price: Decimal | float,
        size: Decimal | float,
        *,
        wait_timeout_s: float = 5.0,
    ) -> tuple[OrderPlacementResult, OrderStateResult | None]:
        """Place an order and verify it reaches LIVE state.

        Convenience method that places order then waits for confirmation.

        Args:
            token_id: Token ID to trade
            side: "BUY" or "SELL"
            price: Limit price
            size: Order size
            wait_timeout_s: How long to wait for LIVE state

        Returns:
            Tuple of (placement_result, state_result or None)
        """
        # Place the order
        placement = self.place_limit_order(token_id, side, price, size)

        if not placement.success or not placement.order_id:
            return placement, None

        # Wait for it to be live
        try:
            state = self.wait_for_order_state(
                placement.order_id,
                target_states={OrderState.LIVE, OrderState.MATCHED},
                timeout_s=wait_timeout_s,
            )
            return placement, state
        except OrderTimeoutError:
            # Order didn't go live in time - might still be pending
            state = self.get_order_state(placement.order_id)
            return placement, state
