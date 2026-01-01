"""Tests for Market WebSocket client.

Phase 5.0: WebSocket microstructure feed integration tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmq.markets.wss_market import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF_SECONDS,
    JITTER_FACTOR,
    MAX_BACKOFF_SECONDS,
    CacheEntry,
    MarketWssClient,
    WssStats,
)

# =============================================================================
# WssStats Tests
# =============================================================================


class TestWssStats:
    """Tests for WssStats dataclass."""

    def test_default_values(self) -> None:
        """Stats should initialize with zeros."""
        stats = WssStats()
        assert stats.messages_received == 0
        assert stats.reconnect_count == 0
        assert stats.parse_errors == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.cache_stale == 0

    def test_custom_values(self) -> None:
        """Stats should accept custom values."""
        stats = WssStats(
            messages_received=100,
            reconnect_count=2,
            parse_errors=5,
            cache_hits=80,
            cache_misses=10,
            cache_stale=3,
        )
        assert stats.messages_received == 100
        assert stats.reconnect_count == 2
        assert stats.parse_errors == 5
        assert stats.cache_hits == 80
        assert stats.cache_misses == 10
        assert stats.cache_stale == 3


# =============================================================================
# Message Parsing Tests
# =============================================================================


class TestMessageParsing:
    """Tests for WebSocket message parsing."""

    @pytest.fixture
    def client(self) -> MarketWssClient:
        """Create a fresh client for each test."""
        return MarketWssClient()

    @pytest.mark.asyncio
    async def test_handle_book_message_full(self, client: MarketWssClient) -> None:
        """Book message with bids and asks should update cache."""
        msg = {
            "event_type": "book",
            "asset_id": "0xtoken123",
            "market": "0xmarket456",
            "bids": [
                {"price": "0.55", "size": "1000"},
                {"price": "0.54", "size": "500"},
            ],
            "asks": [
                {"price": "0.56", "size": "800"},
                {"price": "0.57", "size": "600"},
            ],
            "timestamp": "1234567890",
        }

        await client._handle_book_message(msg)

        ob = client.get_orderbook("0xtoken123")
        assert ob is not None
        assert ob.token_id == "0xtoken123"
        assert ob.best_bid == 0.55
        assert ob.best_bid_size == 1000.0
        assert ob.best_ask == 0.56
        assert ob.best_ask_size == 800.0
        assert ob.mid_price is not None
        assert ob.spread_bps is not None

    @pytest.mark.asyncio
    async def test_handle_book_message_empty_sides(self, client: MarketWssClient) -> None:
        """Book message with empty bids/asks should still create entry."""
        msg = {
            "event_type": "book",
            "asset_id": "0xtoken_empty",
            "bids": [],
            "asks": [],
        }

        await client._handle_book_message(msg)

        ob = client.get_orderbook("0xtoken_empty")
        assert ob is not None
        assert ob.best_bid is None
        assert ob.best_ask is None

    @pytest.mark.asyncio
    async def test_handle_book_message_no_asset_id(self, client: MarketWssClient) -> None:
        """Book message without asset_id should be ignored."""
        msg = {
            "event_type": "book",
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        }

        await client._handle_book_message(msg)

        # Should not have added anything
        assert len(client.get_cached_token_ids()) == 0

    @pytest.mark.asyncio
    async def test_handle_price_change_buy(self, client: MarketWssClient) -> None:
        """Price change message for BUY side should update best bid."""
        msg = {
            "event_type": "price_change",
            "asset_id": "0xtoken_buy",
            "price": "0.52",
            "size": "500",
            "side": "BUY",
        }

        await client._handle_price_change(msg)

        ob = client.get_orderbook("0xtoken_buy")
        assert ob is not None
        assert ob.best_bid == 0.52
        assert ob.best_bid_size == 500.0
        assert ob.best_ask is None

    @pytest.mark.asyncio
    async def test_handle_price_change_sell(self, client: MarketWssClient) -> None:
        """Price change message for SELL side should update best ask."""
        msg = {
            "event_type": "price_change",
            "asset_id": "0xtoken_sell",
            "price": "0.58",
            "size": "300",
            "side": "SELL",
        }

        await client._handle_price_change(msg)

        ob = client.get_orderbook("0xtoken_sell")
        assert ob is not None
        assert ob.best_ask == 0.58
        assert ob.best_ask_size == 300.0
        assert ob.best_bid is None

    @pytest.mark.asyncio
    async def test_handle_price_change_updates_existing(
        self, client: MarketWssClient
    ) -> None:
        """Price change should update existing cache entry."""
        # First, create an entry via book message
        book_msg = {
            "event_type": "book",
            "asset_id": "0xtoken_update",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.60", "size": "100"}],
        }
        await client._handle_book_message(book_msg)

        # Then update via price_change
        change_msg = {
            "event_type": "price_change",
            "asset_id": "0xtoken_update",
            "price": "0.55",
            "size": "200",
            "side": "BUY",
        }
        await client._handle_price_change(change_msg)

        ob = client.get_orderbook("0xtoken_update")
        assert ob is not None
        assert ob.best_bid == 0.55  # Updated
        assert ob.best_bid_size == 200.0  # Updated
        assert ob.best_ask == 0.60  # Preserved
        assert ob.best_ask_size == 100.0  # Preserved

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self, client: MarketWssClient) -> None:
        """Invalid JSON should increment parse_errors."""
        await client._handle_message("not valid json {{{")

        stats = client.get_stats()
        assert stats.parse_errors == 1

    @pytest.mark.asyncio
    async def test_handle_message_bytes(self, client: MarketWssClient) -> None:
        """Bytes message should be decoded and processed."""
        msg = {
            "event_type": "book",
            "asset_id": "0xbytes_token",
            "bids": [{"price": "0.45", "size": "50"}],
            "asks": [],
        }
        raw = json.dumps(msg).encode("utf-8")

        await client._handle_message(raw)

        ob = client.get_orderbook("0xbytes_token")
        assert ob is not None
        assert ob.best_bid == 0.45

    @pytest.mark.asyncio
    async def test_process_message_unknown_type(self, client: MarketWssClient) -> None:
        """Unknown message types should be logged but not error."""
        msg = {"type": "unknown_event", "data": "something"}
        # Should not raise
        await client._process_message(msg)

    @pytest.mark.asyncio
    async def test_process_message_control_types(self, client: MarketWssClient) -> None:
        """Control messages (subscribed, connected, pong) should be handled."""
        for msg_type in ["subscribed", "connected", "pong"]:
            msg = {"type": msg_type}
            # Should not raise
            await client._process_message(msg)


# =============================================================================
# Cache Staleness Tests
# =============================================================================


class TestCacheStaleness:
    """Tests for cache staleness detection."""

    @pytest.fixture
    def client(self) -> MarketWssClient:
        """Create client with short staleness for testing."""
        return MarketWssClient(staleness_seconds=0.1)  # 100ms

    def test_fresh_data_not_stale(self, client: MarketWssClient) -> None:
        """Recently updated data should not be stale."""
        from pmq.markets.orderbook import OrderBookData

        client._update_cache(
            "0xfresh",
            OrderBookData(
                token_id="0xfresh",
                best_bid=0.5,
                best_ask=0.6,
            ),
        )

        assert not client.is_stale("0xfresh")
        ob = client.get_orderbook("0xfresh")
        assert ob is not None

    def test_missing_data_is_stale(self, client: MarketWssClient) -> None:
        """Non-existent token should be considered stale."""
        assert client.is_stale("0xnonexistent")
        assert client.get_orderbook("0xnonexistent") is None

    def test_old_data_becomes_stale(self, client: MarketWssClient) -> None:
        """Data older than staleness threshold should be stale."""
        from pmq.markets.orderbook import OrderBookData

        # Manually create an old cache entry
        old_time = time.monotonic() - 1.0  # 1 second ago
        client._cache["0xold"] = CacheEntry(
            data=OrderBookData(token_id="0xold", best_bid=0.4, best_ask=0.5),
            updated_at=old_time,
        )

        assert client.is_stale("0xold")
        # get_orderbook should return None for stale data
        assert client.get_orderbook("0xold") is None

        # But allow_stale=True should return the data
        ob = client.get_orderbook("0xold", allow_stale=True)
        assert ob is not None
        assert ob.best_bid == 0.4

    def test_stats_track_staleness(self, client: MarketWssClient) -> None:
        """Stats should track cache hits, misses, and stale reads."""
        from pmq.markets.orderbook import OrderBookData

        # Fresh data
        client._update_cache(
            "0xfresh",
            OrderBookData(token_id="0xfresh", best_bid=0.5, best_ask=0.6),
        )

        # Old data
        old_time = time.monotonic() - 1.0
        client._cache["0xold"] = CacheEntry(
            data=OrderBookData(token_id="0xold", best_bid=0.4, best_ask=0.5),
            updated_at=old_time,
        )

        # Access patterns
        client.get_orderbook("0xfresh")  # hit
        client.get_orderbook("0xfresh")  # hit
        client.get_orderbook("0xmissing")  # miss
        client.get_orderbook("0xold")  # stale

        stats = client.get_stats()
        assert stats.cache_hits == 2
        assert stats.cache_misses == 1
        assert stats.cache_stale == 1


# =============================================================================
# Reconnect Logic Tests
# =============================================================================


class TestReconnectLogic:
    """Tests for reconnection with exponential backoff."""

    def test_backoff_constants(self) -> None:
        """Verify backoff constants are reasonable."""
        assert INITIAL_BACKOFF_SECONDS > 0
        assert MAX_BACKOFF_SECONDS > INITIAL_BACKOFF_SECONDS
        assert BACKOFF_MULTIPLIER > 1
        assert 0 < JITTER_FACTOR < 1

    def test_backoff_calculation(self) -> None:
        """Verify exponential backoff math."""
        backoff = INITIAL_BACKOFF_SECONDS

        # First retry
        backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)
        assert backoff == INITIAL_BACKOFF_SECONDS * BACKOFF_MULTIPLIER

        # Continue until max
        for _ in range(20):
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

        assert backoff == MAX_BACKOFF_SECONDS

    @pytest.mark.asyncio
    async def test_reconnect_increments_stats(self) -> None:
        """Reconnection attempts should increment stats."""
        client = MarketWssClient()

        # Simulate reconnect stat increment
        with client._lock:
            client._stats.reconnect_count += 1

        stats = client.get_stats()
        assert stats.reconnect_count == 1


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================


class TestConnectionLifecycle:
    """Tests for connection lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_sets_running(self) -> None:
        """Connect should set running flag and create task."""
        client = MarketWssClient()

        # Patch the _connect_and_receive method to simulate connection failure
        async def mock_connect_and_receive() -> None:
            raise asyncio.CancelledError()

        with patch.object(client, "_connect_and_receive", mock_connect_and_receive):
            await client.connect()

            assert client._running is True
            assert client._task is not None

            # Clean up
            client._running = False
            client._stop_event.set()
            with contextlib.suppress(asyncio.CancelledError):
                await client._task

    @pytest.mark.asyncio
    async def test_connect_twice_ignored(self) -> None:
        """Second connect call should be ignored."""
        client = MarketWssClient()
        client._running = True

        # Should not create a new task
        await client.connect()

        assert client._task is None  # Was not created

    @pytest.mark.asyncio
    async def test_close_stops_client(self) -> None:
        """Close should stop the client and clean up."""
        client = MarketWssClient()
        client._running = True
        client._stop_event.clear()

        # Mock the task
        async def dummy_task() -> None:
            await asyncio.sleep(10)

        client._task = asyncio.create_task(dummy_task())

        await client.close()

        assert client._running is False
        assert client._stop_event.is_set()
        assert client._task is None

    @pytest.mark.asyncio
    async def test_close_when_not_running(self) -> None:
        """Close when not running should be a no-op."""
        client = MarketWssClient()
        client._running = False

        # Should not raise
        await client.close()


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for asset subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_adds_to_set(self) -> None:
        """Subscribe should add assets to subscribed set."""
        client = MarketWssClient()

        await client.subscribe(["0xtoken1", "0xtoken2"])

        assert "0xtoken1" in client._subscribed_assets
        assert "0xtoken2" in client._subscribed_assets

    @pytest.mark.asyncio
    async def test_subscribe_empty_list(self) -> None:
        """Empty subscription list should be no-op."""
        client = MarketWssClient()

        await client.subscribe([])

        assert len(client._subscribed_assets) == 0

    @pytest.mark.asyncio
    async def test_subscribe_deduplicates(self) -> None:
        """Subscribing to same asset twice should not duplicate."""
        client = MarketWssClient()

        await client.subscribe(["0xtoken1"])
        await client.subscribe(["0xtoken1", "0xtoken2"])

        assert len(client._subscribed_assets) == 2

    @pytest.mark.asyncio
    async def test_subscribe_sends_message_when_connected(self) -> None:
        """Subscribe should send message if WebSocket is connected."""
        client = MarketWssClient()

        mock_ws = AsyncMock()
        client._ws = mock_ws

        await client.subscribe(["0xnew_token"])

        mock_ws.send.assert_called_once()
        sent_msg = json.loads(mock_ws.send.call_args[0][0])
        assert sent_msg["type"] == "subscribe"
        assert sent_msg["channel"] == "book"
        assert "0xnew_token" in sent_msg["assets_ids"]


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe cache access."""

    def test_get_stats_returns_copy(self) -> None:
        """get_stats should return a copy, not the original."""
        client = MarketWssClient()

        stats1 = client.get_stats()
        stats1.messages_received = 999  # Modify the copy

        stats2 = client.get_stats()
        assert stats2.messages_received == 0  # Original unchanged

    def test_get_cached_token_ids_returns_copy(self) -> None:
        """get_cached_token_ids should return a copy."""
        from pmq.markets.orderbook import OrderBookData

        client = MarketWssClient()
        client._update_cache(
            "0xtoken",
            OrderBookData(token_id="0xtoken", best_bid=0.5, best_ask=0.6),
        )

        ids1 = client.get_cached_token_ids()
        ids1.append("0xfake")  # Modify the copy

        ids2 = client.get_cached_token_ids()
        assert "0xfake" not in ids2  # Original unchanged


# =============================================================================
# Integration-style Tests (Still Mocked)
# =============================================================================


class TestIntegration:
    """Integration-style tests with mocked WebSocket."""

    @pytest.mark.asyncio
    async def test_full_message_flow(self) -> None:
        """Test processing a sequence of messages."""
        client = MarketWssClient(staleness_seconds=60.0)

        # Simulate a book snapshot
        book_msg = {
            "event_type": "book",
            "asset_id": "0xintegration_token",
            "bids": [
                {"price": "0.50", "size": "1000"},
                {"price": "0.49", "size": "500"},
            ],
            "asks": [
                {"price": "0.55", "size": "800"},
                {"price": "0.56", "size": "400"},
            ],
        }
        await client._handle_book_message(book_msg)

        ob = client.get_orderbook("0xintegration_token")
        assert ob is not None
        assert ob.best_bid == 0.50
        assert ob.best_ask == 0.55
        # Spread should be (0.55 - 0.50) / 0.525 * 10000 ≈ 952 bps
        assert ob.spread_bps is not None
        assert 900 < ob.spread_bps < 1000

        # Simulate a price change (new bid)
        change_msg = {
            "event_type": "price_change",
            "asset_id": "0xintegration_token",
            "price": "0.52",
            "size": "2000",
            "side": "BUY",
        }
        await client._handle_price_change(change_msg)

        ob = client.get_orderbook("0xintegration_token")
        assert ob is not None
        assert ob.best_bid == 0.52  # Updated
        assert ob.best_ask == 0.55  # Unchanged
        # New spread: (0.55 - 0.52) / 0.535 * 10000 ≈ 560 bps
        assert ob.spread_bps is not None
        assert 500 < ob.spread_bps < 600

    @pytest.mark.asyncio
    async def test_multiple_tokens(self) -> None:
        """Test handling multiple tokens simultaneously."""
        client = MarketWssClient()

        tokens = ["0xtoken_a", "0xtoken_b", "0xtoken_c"]
        for i, token in enumerate(tokens):
            msg = {
                "event_type": "book",
                "asset_id": token,
                "bids": [{"price": str(0.4 + i * 0.1), "size": "100"}],
                "asks": [{"price": str(0.5 + i * 0.1), "size": "100"}],
            }
            await client._handle_book_message(msg)

        cached_ids = client.get_cached_token_ids()
        assert len(cached_ids) == 3
        for token in tokens:
            assert token in cached_ids
            ob = client.get_orderbook(token)
            assert ob is not None
