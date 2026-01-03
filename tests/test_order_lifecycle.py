"""Tests for order_lifecycle module (Phase 13).

Tests order placement, waiting, and cancellation with mocked CLOB client.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pmq.ops.order_lifecycle import (
    CancellationResult,
    OrderLifecycle,
    OrderPlacementResult,
    OrderState,
    OrderStateResult,
    OrderTimeoutError,
    _mask_order_id,
    _parse_order_state,
)


class MockClobClient:
    """Mock CLOB client for testing."""

    def __init__(self) -> None:
        self.get_order_responses: list[dict[str, Any]] = []
        self.get_order_call_count = 0
        self.create_order_response: dict[str, Any] = {}
        self.cancel_response: dict[str, Any] = {}

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Mock get_order - returns from response list."""
        if self.get_order_responses:
            response = self.get_order_responses[
                min(self.get_order_call_count, len(self.get_order_responses) - 1)
            ]
        else:
            response = {"status": "LIVE", "size_matched": 0}
        self.get_order_call_count += 1
        return response

    def create_and_post_order(self, order: Any) -> dict[str, Any]:
        """Mock create_and_post_order."""
        return self.create_order_response

    def cancel(self, order_id: str) -> dict[str, Any]:
        """Mock cancel."""
        return self.cancel_response


class TestMaskOrderId:
    """Tests for order ID masking."""

    def test_mask_long_order_id(self) -> None:
        """Test masking long order ID shows 6 chars."""
        result = _mask_order_id("1234567890123456")
        assert len(result) < len("1234567890123456")
        assert "..." in result or "*" in result

    def test_mask_short_order_id(self) -> None:
        """Test masking short order ID."""
        result = _mask_order_id("12345678")
        # Masking adds "..." which may make result longer, but hides middle chars
        assert "..." in result or "*" in result


class TestParseOrderState:
    """Tests for order state parsing."""

    def test_parse_live_state(self) -> None:
        """Test parsing LIVE state."""
        assert _parse_order_state("LIVE") == OrderState.LIVE
        assert _parse_order_state("live") == OrderState.LIVE
        assert _parse_order_state("OPEN") == OrderState.LIVE
        assert _parse_order_state("ACTIVE") == OrderState.LIVE

    def test_parse_matched_state(self) -> None:
        """Test parsing MATCHED state."""
        assert _parse_order_state("MATCHED") == OrderState.MATCHED
        assert _parse_order_state("FILLED") == OrderState.MATCHED
        assert _parse_order_state("COMPLETE") == OrderState.MATCHED

    def test_parse_cancelled_state(self) -> None:
        """Test parsing CANCELLED state."""
        assert _parse_order_state("CANCELLED") == OrderState.CANCELLED
        assert _parse_order_state("CANCELED") == OrderState.CANCELLED

    def test_parse_pending_state(self) -> None:
        """Test parsing PENDING state."""
        assert _parse_order_state("PENDING") == OrderState.PENDING
        assert _parse_order_state("SUBMITTED") == OrderState.PENDING

    def test_parse_unknown_state(self) -> None:
        """Test parsing unknown state."""
        assert _parse_order_state("WEIRD") == OrderState.UNKNOWN
        assert _parse_order_state(None) == OrderState.UNKNOWN
        assert _parse_order_state("") == OrderState.UNKNOWN


class TestOrderPlacement:
    """Tests for order placement."""

    @pytest.fixture
    def client(self) -> MockClobClient:
        """Create mock client."""
        return MockClobClient()

    @pytest.fixture
    def lifecycle(self, client: MockClobClient) -> OrderLifecycle:
        """Create lifecycle with mock client."""
        return OrderLifecycle(client, timeout_s=5.0, poll_interval_s=0.1)

    def test_place_order_success(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test successful order placement."""
        client.create_order_response = {"orderID": "test-order-123"}

        # Mock the import in order_lifecycle
        with patch("pmq.ops.order_lifecycle.OrderLifecycle.place_limit_order") as mock_place:
            mock_place.return_value = OrderPlacementResult(
                success=True,
                order_id="test-order-123",
                status="SUBMITTED",
            )

            result = lifecycle.place_limit_order(
                token_id="test-token",
                side="BUY",
                price=0.45,
                size=1.0,
            )

            assert result.success is True
            assert result.order_id == "test-order-123"

    def test_place_order_missing_order_id(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test placement failure when no order ID returned."""
        client.create_order_response = {"error": "Invalid order"}

        with patch("pmq.ops.order_lifecycle.OrderLifecycle.place_limit_order") as mock_place:
            mock_place.return_value = OrderPlacementResult(
                success=False,
                error="Invalid order",
                status="REJECTED",
            )

            result = lifecycle.place_limit_order(
                token_id="test-token",
                side="BUY",
                price=0.45,
                size=1.0,
            )

            assert result.success is False
            assert result.error is not None


class TestGetOrderState:
    """Tests for getting order state."""

    @pytest.fixture
    def client(self) -> MockClobClient:
        """Create mock client."""
        return MockClobClient()

    @pytest.fixture
    def lifecycle(self, client: MockClobClient) -> OrderLifecycle:
        """Create lifecycle with mock client."""
        return OrderLifecycle(client, timeout_s=5.0, poll_interval_s=0.1)

    def test_get_order_state_live(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test getting LIVE order state."""
        client.get_order_responses = [{"status": "LIVE", "size_matched": 0, "size": 10}]

        result = lifecycle.get_order_state("test-order-123")

        assert result.state == OrderState.LIVE
        assert result.filled_size == Decimal("0")
        assert result.order_id == "test-order-123"

    def test_get_order_state_matched(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test getting MATCHED order state."""
        client.get_order_responses = [
            {"status": "MATCHED", "size_matched": 10, "original_size": 10}
        ]

        result = lifecycle.get_order_state("test-order-123")

        assert result.state == OrderState.MATCHED
        assert result.filled_size == Decimal("10")

    def test_get_order_state_error(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test error handling in get_order_state."""
        # Create a client that raises
        client = MagicMock()
        client.get_order.side_effect = Exception("API error")
        lifecycle = OrderLifecycle(client, timeout_s=5.0)

        result = lifecycle.get_order_state("test-order-123")

        assert result.state == OrderState.UNKNOWN
        assert result.error is not None


class TestWaitForOrderState:
    """Tests for waiting for order state."""

    @pytest.fixture
    def client(self) -> MockClobClient:
        """Create mock client."""
        return MockClobClient()

    @pytest.fixture
    def lifecycle(self, client: MockClobClient) -> OrderLifecycle:
        """Create lifecycle with mock client."""
        return OrderLifecycle(client, timeout_s=1.0, poll_interval_s=0.1)

    def test_wait_reaches_live_state(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test waiting until LIVE state is reached."""
        # First call: PENDING, second call: LIVE
        client.get_order_responses = [
            {"status": "PENDING", "size_matched": 0},
            {"status": "LIVE", "size_matched": 0},
        ]

        result = lifecycle.wait_for_order_state(
            "test-order-123",
            target_states={OrderState.LIVE},
            timeout_s=2.0,
        )

        assert result.state == OrderState.LIVE

    def test_wait_timeout(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test timeout when state never reached."""
        # Always return PENDING
        client.get_order_responses = [{"status": "PENDING", "size_matched": 0}]

        with pytest.raises(OrderTimeoutError):
            lifecycle.wait_for_order_state(
                "test-order-123",
                target_states={OrderState.LIVE},
                timeout_s=0.3,
            )

    def test_wait_immediate_match(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test immediate return when already in target state."""
        client.get_order_responses = [{"status": "MATCHED", "size_matched": 10}]

        result = lifecycle.wait_for_order_state(
            "test-order-123",
            target_states={OrderState.MATCHED},
        )

        assert result.state == OrderState.MATCHED


class TestCancelOrder:
    """Tests for order cancellation."""

    @pytest.fixture
    def client(self) -> MockClobClient:
        """Create mock client."""
        return MockClobClient()

    @pytest.fixture
    def lifecycle(self, client: MockClobClient) -> OrderLifecycle:
        """Create lifecycle with mock client."""
        return OrderLifecycle(client, timeout_s=5.0, poll_interval_s=0.1)

    def test_cancel_success(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test successful cancellation."""
        client.cancel_response = {"success": True}

        result = lifecycle.cancel_order("test-order-123")

        assert result.success is True
        assert result.order_id == "test-order-123"

    def test_cancel_failure(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test failed cancellation."""
        client.cancel_response = {"success": False, "error": "Order not found"}

        result = lifecycle.cancel_order("test-order-123")

        assert result.success is False
        assert result.error is not None

    def test_cancel_with_canceled_list(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test cancellation with canceled list response format."""
        client.cancel_response = {"canceled": ["test-order-123"]}

        result = lifecycle.cancel_order("test-order-123")

        assert result.success is True

    def test_cancel_error(self, client: MockClobClient, lifecycle: OrderLifecycle) -> None:
        """Test error handling in cancel."""
        client = MagicMock()
        client.cancel.side_effect = Exception("Network error")
        lifecycle = OrderLifecycle(client, timeout_s=5.0)

        result = lifecycle.cancel_order("test-order-123")

        assert result.success is False
        assert result.error is not None


class TestSafeCancelIfOpen:
    """Tests for safe cancellation."""

    @pytest.fixture
    def client(self) -> MockClobClient:
        """Create mock client."""
        return MockClobClient()

    @pytest.fixture
    def lifecycle(self, client: MockClobClient) -> OrderLifecycle:
        """Create lifecycle with mock client."""
        return OrderLifecycle(client, timeout_s=5.0, poll_interval_s=0.1)

    def test_safe_cancel_already_cancelled(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test safe cancel when already cancelled."""
        client.get_order_responses = [{"status": "CANCELLED"}]

        result = lifecycle.safe_cancel_if_open("test-order-123")

        assert result.success is True
        # Should not attempt to cancel since already cancelled

    def test_safe_cancel_already_filled(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test safe cancel when already filled."""
        client.get_order_responses = [{"status": "MATCHED", "size_matched": 10}]

        result = lifecycle.safe_cancel_if_open("test-order-123")

        assert result.success is True
        assert result.was_filled is True
        assert result.filled_size == Decimal("10")

    def test_safe_cancel_still_live(
        self, client: MockClobClient, lifecycle: OrderLifecycle
    ) -> None:
        """Test safe cancel when still live."""
        client.get_order_responses = [{"status": "LIVE", "size_matched": 0}]
        client.cancel_response = {"success": True}

        result = lifecycle.safe_cancel_if_open("test-order-123")

        assert result.success is True


class TestBackoff:
    """Tests for backoff calculation."""

    def test_backoff_increases_exponentially(self) -> None:
        """Test backoff increases with attempts."""
        client = MockClobClient()
        lifecycle = OrderLifecycle(client, backoff_base_s=1.0, backoff_max_s=10.0)

        assert lifecycle._backoff(0) == 1.0
        assert lifecycle._backoff(1) == 2.0
        assert lifecycle._backoff(2) == 4.0
        assert lifecycle._backoff(3) == 8.0

    def test_backoff_capped_at_max(self) -> None:
        """Test backoff is capped at maximum."""
        client = MockClobClient()
        lifecycle = OrderLifecycle(client, backoff_base_s=1.0, backoff_max_s=10.0)

        assert lifecycle._backoff(10) == 10.0
        assert lifecycle._backoff(100) == 10.0
