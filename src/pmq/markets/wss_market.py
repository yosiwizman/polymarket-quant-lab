"""Market WebSocket client for real-time order book streaming.

Connects to Polymarket's Market WebSocket channel to receive
real-time order book updates. No authentication required (market
data only).

Phase 5.0: WebSocket microstructure feed integration.
Phase 5.3: Application-level keepalive + adaptive staleness.
Phase 5.4: Connection-level health tracking for health-gated fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import statistics
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

# Cache staleness threshold (deprecated in favor of health-based model in Phase 5.4)
DEFAULT_STALENESS_SECONDS = 30.0

# Phase 5.3: Application-level keepalive (Polymarket requires "PING" text frames)
DEFAULT_KEEPALIVE_INTERVAL_SECONDS = 10.0

# Phase 5.4: Connection health timeout (unhealthy if no message/pong in this period)
DEFAULT_HEALTH_TIMEOUT_SECONDS = 60.0

# Phase 5.4: Maximum book age before considered "very old" (safety cap, not for frequent fallback)
DEFAULT_MAX_BOOK_AGE_SECONDS = 1800.0  # 30 minutes


@dataclass
class CacheEntry:
    """Cached order book data with timestamp."""

    data: OrderBookData
    updated_at: float  # time.monotonic() value


@dataclass
class CacheAgeStats:
    """Statistics about cache entry ages."""

    min_age: float = 0.0
    max_age: float = 0.0
    median_age: float = 0.0
    count: int = 0


@dataclass
class WssStats:
    """Statistics for WebSocket connection."""

    messages_received: int = 0
    reconnect_count: int = 0
    parse_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stale: int = 0
    # Phase 5.3: Keepalive stats
    keepalive_sent: int = 0
    keepalive_failures: int = 0
    # Phase 5.4: Health tracking
    pong_received: int = 0


@dataclass
class MarketWssClient:
    """WebSocket client for Polymarket market data.

    Subscribes to order book updates and maintains an in-memory
    cache of latest OrderBookData per token_id. Thread-safe for
    reading from cache while the event loop runs.

    Phase 5.3: Implements application-level keepalive ("PING" text frames)
    as required by Polymarket's WebSocket server for long-lived connections.

    Phase 5.4: Connection-level health tracking. Use is_healthy() to determine
    if the connection is alive (receiving messages/pongs). Markets that don't
    emit updates frequently should NOT trigger REST fallback - only unhealthy
    connections or missing cache entries should.

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
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL_SECONDS
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS  # Phase 5.4
    _cache: dict[str, CacheEntry] = field(default_factory=dict)
    _stats: WssStats = field(default_factory=WssStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ws: ClientConnection | None = field(default=None, repr=False)
    _subscribed_assets: set[str] = field(default_factory=set)
    _running: bool = field(default=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _connected: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Phase 5.4: Health tracking timestamps (monotonic time)
    _last_message_at: float = field(default=0.0)
    _last_pong_at: float = field(default=0.0)

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

    def get_orderbook(self, token_id: str, allow_stale: bool = False) -> OrderBookData | None:
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

    def is_healthy(self, health_timeout: float | None = None) -> bool:
        """Check if the WebSocket connection is healthy.

        Phase 5.4: Connection is healthy if we've received any message
        or PONG response within the health timeout period.

        This should be used to determine REST fallback, NOT per-market cache age.
        Markets with quiet cache (no recent updates) should still be considered
        healthy if the connection itself is alive.

        Args:
            health_timeout: Override health timeout (uses self.health_timeout_seconds if None)

        Returns:
            True if connection is healthy (recent activity)
        """
        timeout = health_timeout or self.health_timeout_seconds
        now = time.monotonic()

        # Check both message and pong timestamps
        with self._lock:
            last_message = self._last_message_at
            last_pong = self._last_pong_at

        # Healthy if either timestamp is within timeout
        message_age = now - last_message if last_message > 0 else float("inf")
        pong_age = now - last_pong if last_pong > 0 else float("inf")

        return min(message_age, pong_age) <= timeout

    def get_orderbook_if_healthy(
        self, token_id: str, max_book_age: float | None = None
    ) -> OrderBookData | None:
        """Get cached order book only if it exists (ignore staleness).

        Phase 5.4: For health-gated fallback model. Returns cached data
        regardless of age (since market quietness != staleness), but
        optionally enforces a maximum book age as a safety cap.

        Use is_healthy() separately to determine if REST fallback is needed
        due to connection issues.

        Args:
            token_id: The token ID to look up
            max_book_age: Optional maximum age in seconds (safety cap)

        Returns:
            OrderBookData if in cache (and within max_book_age if specified)
        """
        with self._lock:
            entry = self._cache.get(token_id)
            if entry is None:
                self._stats.cache_misses += 1
                return None

            # Apply optional safety cap on book age
            if max_book_age is not None:
                age = time.monotonic() - entry.updated_at
                if age > max_book_age:
                    self._stats.cache_stale += 1
                    return None

            self._stats.cache_hits += 1
            return entry.data

    def has_cached_book(self, token_id: str) -> bool:
        """Check if we have any cached data for this token (regardless of age).

        Phase 5.4: Used to determine if we need REST fetch for missing data
        vs. just using cached data for quiet markets.

        Args:
            token_id: The token ID to check

        Returns:
            True if token exists in cache
        """
        with self._lock:
            return token_id in self._cache

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
                keepalive_sent=self._stats.keepalive_sent,
                keepalive_failures=self._stats.keepalive_failures,
                pong_received=self._stats.pong_received,
            )

    def get_cache_ages(self, token_ids: list[str] | None = None) -> CacheAgeStats:
        """Get statistics about cache entry ages.

        Args:
            token_ids: Optional list of tokens to check (all cached if None)

        Returns:
            CacheAgeStats with min/max/median ages in seconds
        """
        now = time.monotonic()
        with self._lock:
            if token_ids is None:
                entries = list(self._cache.values())
            else:
                entries = [self._cache[tid] for tid in token_ids if tid in self._cache]

            if not entries:
                return CacheAgeStats()

            ages = [now - entry.updated_at for entry in entries]

        return CacheAgeStats(
            min_age=min(ages),
            max_age=max(ages),
            median_age=statistics.median(ages) if ages else 0.0,
            count=len(ages),
        )

    def get_cache_freshness(
        self, token_ids: list[str], staleness_threshold: float | None = None
    ) -> tuple[int, int, int]:
        """Get freshness breakdown for given tokens.

        Args:
            token_ids: List of tokens to check
            staleness_threshold: Override staleness threshold (uses self.staleness_seconds if None)

        Returns:
            Tuple of (fresh_count, stale_count, missing_count)
        """
        threshold = staleness_threshold or self.staleness_seconds
        now = time.monotonic()
        fresh = 0
        stale = 0
        missing = 0

        with self._lock:
            for token_id in token_ids:
                entry = self._cache.get(token_id)
                if entry is None:
                    missing += 1
                elif (now - entry.updated_at) > threshold:
                    stale += 1
                else:
                    fresh += 1

        return fresh, stale, missing

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

        # Stop keepalive task first
        await self._stop_keepalive()

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

    async def _start_keepalive(self) -> None:
        """Start the keepalive background task."""
        await self._stop_keepalive()  # Ensure no duplicate tasks
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.debug(f"Keepalive task started (interval={self.keepalive_interval}s)")

    async def _stop_keepalive(self) -> None:
        """Stop the keepalive background task."""
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
            logger.debug("Keepalive task stopped")

    async def _keepalive_loop(self) -> None:
        """Background task that sends 'PING' text frames periodically.

        Polymarket's WebSocket server expects application-level keepalive
        messages (literal "PING" text) to maintain long-lived connections.
        """
        while self._running and not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.keepalive_interval)

                if self._ws is not None and self._running:
                    await self._ws.send("PING")
                    with self._lock:
                        self._stats.keepalive_sent += 1
                    logger.debug("Sent keepalive PING")

            except asyncio.CancelledError:
                break
            except Exception as e:
                with self._lock:
                    self._stats.keepalive_failures += 1
                logger.debug(f"Keepalive send failed: {e}")

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

                # Stop keepalive on disconnect
                await self._stop_keepalive()

                # Add jitter to backoff
                jitter = backoff * JITTER_FACTOR * (2 * random.random() - 1)
                sleep_time = backoff + jitter

                logger.warning(f"WebSocket error: {e}. Reconnecting in {sleep_time:.1f}s")

                with self._lock:
                    self._stats.reconnect_count += 1

                # Wait before reconnect (interruptible)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_time)
                    # Stop event was set, exit
                    break
                except TimeoutError:
                    pass  # Normal timeout, proceed to reconnect

                # Exponential backoff
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

        # Cleanup on exit
        await self._stop_keepalive()
        self._connected.clear()

    async def _connect_and_receive(self) -> None:
        """Establish connection and process messages."""
        import websockets

        logger.info(f"Connecting to {WSS_ENDPOINT}")

        # Disable library-level ping since we use application-level keepalive
        async with websockets.connect(
            WSS_ENDPOINT,
            ping_interval=None,  # Disable websockets library ping
            ping_timeout=None,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected.set()
            logger.info("WebSocket connected")

            # Start application-level keepalive (Phase 5.3)
            await self._start_keepalive()

            # Re-subscribe to any previously subscribed assets
            if self._subscribed_assets:
                await self._send_subscribe(list(self._subscribed_assets))

            async for message in ws:
                if self._stop_event.is_set():
                    break

                await self._handle_message(message)

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse and process incoming WebSocket message."""
        now = time.monotonic()

        # Phase 5.4: Update health timestamp on every message
        with self._lock:
            self._stats.messages_received += 1
            self._last_message_at = now

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            # Phase 5.3/5.4: Handle non-JSON keepalive responses ("PONG", etc.)
            raw_stripped = raw.strip().upper()
            if raw_stripped in ("PONG", "PING"):
                # Phase 5.4: Track PONG specifically for health monitoring
                if raw_stripped == "PONG":
                    with self._lock:
                        self._last_pong_at = now
                        self._stats.pong_received += 1
                logger.debug(f"Keepalive response: {raw_stripped}")
                return

            data = json.loads(raw)
            await self._process_message(data)

        except json.JSONDecodeError as e:
            # Don't count simple text responses as parse errors
            if len(raw) < 20:  # Short non-JSON likely control message
                logger.debug(f"Non-JSON message: {raw!r}")
            else:
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
        elif msg_type in ("subscribed", "connected", "pong", "PONG"):
            # Control/keepalive response messages - log at debug level
            # Phase 5.4: Track JSON pong as well
            if msg_type in ("pong", "PONG"):
                with self._lock:
                    self._last_pong_at = time.monotonic()
                    self._stats.pong_received += 1
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

    def update_cache(self, token_id: str, data: OrderBookData) -> None:
        """Public method to update cache entry (for healing).

        Phase 5.5: Used by reconciliation to heal cache when drift is detected.
        Replaces the cached orderbook with REST-fetched data.

        Args:
            token_id: Token ID to update
            data: OrderBookData from REST (source of truth)
        """
        self._update_cache(token_id, data)
