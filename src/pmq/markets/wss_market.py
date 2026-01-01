"""Market WebSocket client for real-time order book streaming.

Connects to Polymarket's Market WebSocket channel to receive
real-time order book updates. No authentication required (market
data only).

Phase 5.0: WebSocket microstructure feed integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pmq.logging import get_logger
from pmq.markets.orderbook import OrderBookData, compute_microstructure

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = get_logger("wss_market")

# Polymarket Market WebSocket endpoint (no auth required)
WSS_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Reconnection parameters
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.3  # ±30% jitter

# Cache staleness threshold
DEFAULT_STALENESS_SECONDS = 30.0


@dataclass
class CacheEntry:
    """Cached order book data with timestamp."""

    data: OrderBookData
    updated_at: float  # time.monotonic() value


@dataclass
class WssStats:
    """Statistics for WebSocket connection."""

    messages_received: int = 0
    reconnect_count: int = 0
    parse_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stale: int = 0


@dataclass
class MarketWssClient:
    """WebSocket client for Polymarket market data.

    Subscribes to order book updates and maintains an in-memory
    cache of latest OrderBookData per token_id. Thread-safe for
    reading from cache while the event loop runs.

    Example:
        client = MarketWssClient()
        await client.connect()
        await client.subscribe(["token_id_1", "token_id_2"])

        # In another context (or same event loop):
        data = client.get_orderbook("token_id_1")
        if data:
            print(f"Best bid: {data.best_bid}")

        await client.close()
    """

    staleness_seconds: float = DEFAULT_STALENESS_SECONDS
    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _stats: WssStats = field(default_factory=WssStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ws: ClientConnection | None = field(default=None, repr=False)
    _subscribed_assets: set[str] = field(default_factory=set)
    _running: bool = field(default=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _connected: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def connect(self) -> None:
        """Establish WebSocket connection.

        Non-blocking; starts background task for message processing.
        Use `wait_connected()` to block until connected.
        """
        if self._running:
            logger.warning("Already running, ignoring connect()")
            return

        self._running = True
        self._stop_event.clear()
        self._connected.clear()

        # Start background receiver task
        self._task = asyncio.create_task(self._run_forever())
        logger.info("WebSocket client started")

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        """Wait until WebSocket is connected.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if connected, False if timeout
        """
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def subscribe(self, asset_ids: list[str]) -> None:
        """Subscribe to order book updates for given assets.

        Args:
            asset_ids: List of token IDs to subscribe to
        """
        if not asset_ids:
            return

        new_assets = set(asset_ids) - self._subscribed_assets
        if not new_assets:
            logger.debug("Already subscribed to all requested assets")
            return

        self._subscribed_assets.update(new_assets)

        if self._ws is not None:
            await self._send_subscribe(list(new_assets))

    async def _send_subscribe(self, asset_ids: list[str]) -> None:
        """Send subscription message for assets."""
        if self._ws is None:
            return

        # Polymarket market channel subscription format
        msg = {
            "type": "subscribe",
            "channel": "book",
            "assets_ids": asset_ids,  # Note: API uses 'assets_ids' (plural)
        }

        try:
            await self._ws.send(json.dumps(msg))
            logger.info(f"Subscribed to {len(asset_ids)} assets")
        except Exception as e:
            logger.error(f"Failed to send subscribe: {e}")

    def get_orderbook(
        self, token_id: str, allow_stale: bool = False
    ) -> OrderBookData | None:
        """Get cached order book data for a token.

        Thread-safe. Returns None if not in cache or stale.

        Args:
            token_id: The token ID to look up
            allow_stale: If True, return stale data instead of None

        Returns:
            OrderBookData if available and fresh (or allow_stale=True)
        """
        with self._lock:
            entry = self._cache.get(token_id)
            if entry is None:
                self._stats.cache_misses += 1
                return None

            age = time.monotonic() - entry.updated_at
            if age > self.staleness_seconds and not allow_stale:
                self._stats.cache_stale += 1
                return None

            self._stats.cache_hits += 1
            return entry.data

    def is_stale(self, token_id: str) -> bool:
        """Check if cached data for token is stale or missing.

        Args:
            token_id: The token ID to check

        Returns:
            True if stale or missing
        """
        with self._lock:
            entry = self._cache.get(token_id)
            if entry is None:
                return True
            age = time.monotonic() - entry.updated_at
            return age > self.staleness_seconds

    def get_stats(self) -> WssStats:
        """Get connection statistics (thread-safe copy)."""
        with self._lock:
            return WssStats(
                messages_received=self._stats.messages_received,
                reconnect_count=self._stats.reconnect_count,
                parse_errors=self._stats.parse_errors,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                cache_stale=self._stats.cache_stale,
            )

    def get_cached_token_ids(self) -> list[str]:
        """Get list of token IDs currently in cache (thread-safe)."""
        with self._lock:
            return list(self._cache.keys())

    async def close(self) -> None:
        """Close WebSocket connection and stop background task."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self._ws = None

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

        logger.info("WebSocket client stopped")

    async def _run_forever(self) -> None:
        """Main loop: connect, receive, reconnect on failure."""
        backoff = INITIAL_BACKOFF_SECONDS

        while self._running and not self._stop_event.is_set():
            try:
                await self._connect_and_receive()
                # Clean disconnect, reset backoff
                backoff = INITIAL_BACKOFF_SECONDS
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break

                # Add jitter to backoff
                jitter = backoff * JITTER_FACTOR * (2 * random.random() - 1)
                sleep_time = backoff + jitter

                logger.warning(
                    f"WebSocket error: {e}. Reconnecting in {sleep_time:.1f}s"
                )

                with self._lock:
                    self._stats.reconnect_count += 1

                # Wait before reconnect (interruptible)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=sleep_time
                    )
                    # Stop event was set, exit
                    break
                except TimeoutError:
                    pass  # Normal timeout, proceed to reconnect

                # Exponential backoff
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

        self._connected.clear()

    async def _connect_and_receive(self) -> None:
        """Establish connection and process messages."""
        import websockets

        logger.info(f"Connecting to {WSS_ENDPOINT}")

        async with websockets.connect(
            WSS_ENDPOINT,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected.set()
            logger.info("WebSocket connected")

            # Re-subscribe to any previously subscribed assets
            if self._subscribed_assets:
                await self._send_subscribe(list(self._subscribed_assets))

            async for message in ws:
                if self._stop_event.is_set():
                    break

                await self._handle_message(message)

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse and process incoming WebSocket message."""
        with self._lock:
            self._stats.messages_received += 1

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            data = json.loads(raw)
            await self._process_message(data)

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            with self._lock:
                self._stats.parse_errors += 1
        except Exception as e:
            logger.debug(f"Message handling error: {e}")
            with self._lock:
                self._stats.parse_errors += 1

    async def _process_message(self, data: dict[str, Any]) -> None:
        """Process parsed message and update cache."""
        msg_type = data.get("event_type") or data.get("type")

        if msg_type == "book":
            await self._handle_book_message(data)
        elif msg_type == "price_change":
            await self._handle_price_change(data)
        elif msg_type in ("subscribed", "connected", "pong"):
            logger.debug(f"Control message: {msg_type}")
        else:
            # Log unknown message types at debug level
            logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_book_message(self, data: dict[str, Any]) -> None:
        """Handle order book snapshot/update message.

        Expected format:
        {
            "event_type": "book",
            "asset_id": "0x...",
            "market": "0x...",
            "bids": [{"price": "0.55", "size": "100"}, ...],
            "asks": [{"price": "0.56", "size": "50"}, ...],
            "timestamp": "1234567890",
            "hash": "..."
        }
        """
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Parse best bid/ask from top of book
        best_bid: float | None = None
        best_bid_size: float | None = None
        best_ask: float | None = None
        best_ask_size: float | None = None

        if bids:
            try:
                best_bid = float(bids[0].get("price", 0))
                best_bid_size = float(bids[0].get("size", 0))
            except (ValueError, TypeError, IndexError):
                pass

        if asks:
            try:
                best_ask = float(asks[0].get("price", 0))
                best_ask_size = float(asks[0].get("size", 0))
            except (ValueError, TypeError, IndexError):
                pass

        # Compute derived microstructure metrics
        mid_price, spread_bps, top_depth_usd = compute_microstructure(
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
        )

        orderbook = OrderBookData(
            token_id=asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            mid_price=mid_price,
            spread_bps=spread_bps,
            top_depth_usd=top_depth_usd,
        )

        self._update_cache(asset_id, orderbook)

    async def _handle_price_change(self, data: dict[str, Any]) -> None:
        """Handle price change message (simplified book update).

        Expected format:
        {
            "event_type": "price_change",
            "asset_id": "0x...",
            "price": "0.55",
            "side": "BUY",
            "size": "100",
            ...
        }

        These are incremental updates; we use them to update
        best bid/ask if applicable.
        """
        asset_id = data.get("asset_id")
        if not asset_id:
            return

        side = data.get("side", "").upper()
        try:
            price = float(data.get("price", 0))
            size = float(data.get("size", 0))
        except (ValueError, TypeError):
            return

        # Get existing cache entry or create new one
        with self._lock:
            entry = self._cache.get(asset_id)

        if entry is None:
            # Create new entry from price_change
            if side == "BUY":
                best_bid, best_bid_size = price, size
                best_ask, best_ask_size = None, None
            else:
                best_bid, best_bid_size = None, None
                best_ask, best_ask_size = price, size
        else:
            # Update existing entry
            best_bid = entry.data.best_bid
            best_bid_size = entry.data.best_bid_size
            best_ask = entry.data.best_ask
            best_ask_size = entry.data.best_ask_size

            if side == "BUY":
                best_bid, best_bid_size = price, size
            else:
                best_ask, best_ask_size = price, size

        # Compute derived metrics
        mid_price, spread_bps, top_depth_usd = compute_microstructure(
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
        )

        orderbook = OrderBookData(
            token_id=asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            mid_price=mid_price,
            spread_bps=spread_bps,
            top_depth_usd=top_depth_usd,
        )

        self._update_cache(asset_id, orderbook)

    def _update_cache(self, token_id: str, data: OrderBookData) -> None:
        """Update cache entry (thread-safe)."""
        with self._lock:
            self._cache[token_id] = CacheEntry(
                data=data,
                updated_at=time.monotonic(),
            )
