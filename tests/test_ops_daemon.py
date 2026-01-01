"""Tests for ops daemon continuous capture.

Phase 5.1: Unit tests for DaemonRunner with mocked dependencies.
Tests cover:
- Graceful shutdown
- Daily rollover export
- REST fallback
- Coverage counters

Phase 5.2: Extended tests for:
- Snapshot export to gzip CSV
- Retention cleanup
- Ops status command
"""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pmq.ops.daemon import (
    DaemonConfig,
    DaemonRunner,
    DailyStats,
    RealClock,
    TickStats,
    real_sleep,
)

# =============================================================================
# Mock Implementations for Dependency Injection
# =============================================================================


@dataclass
class FakeClock:
    """Fake clock for testing."""

    _current_time: datetime
    _monotonic: float = 0.0
    _day_offset: int = 0  # For simulating day changes

    def now(self) -> datetime:
        """Return current fake time."""
        if self._day_offset > 0:
            return self._current_time.replace(day=self._current_time.day + self._day_offset)
        return self._current_time

    def monotonic(self) -> float:
        """Return monotonic counter."""
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance monotonic time."""
        self._monotonic += seconds

    def advance_day(self) -> None:
        """Advance to next day."""
        self._day_offset += 1


class FakeSleep:
    """Fake async sleep for testing."""

    def __init__(self) -> None:
        self.total_sleep = 0.0
        self.call_count = 0

    async def __call__(self, seconds: float) -> None:
        """Record sleep call without actual sleep."""
        self.total_sleep += seconds
        self.call_count += 1
        await asyncio.sleep(0)  # Yield to event loop


class FakeDAO:
    """Fake DAO for testing."""

    def __init__(self) -> None:
        self.upserted_markets: list[Any] = []
        self.saved_snapshots: list[tuple[Any, str, Any]] = []
        self.runtime_state: dict[str, str] = {}
        # Phase 5.2: Snapshot storage for export tests
        self._snapshots_db: list[dict[str, Any]] = []
        self.deleted_snapshots_before: list[str] = []

    def upsert_markets(self, markets: list[Any]) -> int:
        """Record upserted markets."""
        self.upserted_markets.extend(markets)
        return len(markets)

    def save_snapshots_bulk(
        self,
        markets: list[Any],
        snapshot_time: str,
        orderbook_data: dict[str, Any] | None = None,
    ) -> int:
        """Record saved snapshots."""
        self.saved_snapshots.append((markets, snapshot_time, orderbook_data))
        return len(markets)

    def set_runtime_state(self, key: str, value: str) -> None:
        """Store runtime state."""
        self.runtime_state[key] = value

    # Phase 5.2: New methods for snapshot export
    def get_snapshots_for_date(self, date_str: str) -> list[dict[str, Any]]:
        """Get all snapshots for a given date."""
        start = f"{date_str}T00:00:00"
        end = f"{date_str}T23:59:59"
        return [s for s in self._snapshots_db if start <= s.get("snapshot_time", "") <= end]

    def delete_snapshots_before(self, cutoff_time: str) -> int:
        """Delete snapshots older than cutoff."""
        self.deleted_snapshots_before.append(cutoff_time)
        before_count = len(self._snapshots_db)
        self._snapshots_db = [
            s for s in self._snapshots_db if s.get("snapshot_time", "") >= cutoff_time
        ]
        return before_count - len(self._snapshots_db)

    def add_test_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Add a test snapshot to the fake DB."""
        self._snapshots_db.append(snapshot)

    def count_snapshots(self) -> int:
        """Count total snapshots."""
        return len(self._snapshots_db)

    def get_latest_snapshot_time(self) -> str | None:
        """Get the most recent snapshot time."""
        if not self._snapshots_db:
            return None
        return max(s.get("snapshot_time", "") for s in self._snapshots_db)

    def get_runtime_state(self, key: str) -> str | None:
        """Get runtime state value."""
        return self.runtime_state.get(key)


class FakeGammaClient:
    """Fake Gamma client for testing."""

    def __init__(self, markets: list[Any] | None = None) -> None:
        self._markets = markets or []
        self.closed = False

    def list_markets(self, limit: int = 200) -> list[Any]:
        """Return fake markets."""
        return self._markets[:limit]

    def close(self) -> None:
        """Mark client as closed."""
        self.closed = True


class FakeMarket:
    """Fake market for testing."""

    def __init__(self, market_id: str, yes_token_id: str | None = None) -> None:
        self.id = market_id
        self.yes_token_id = yes_token_id or f"0xtoken_{market_id}"
        self.active = True
        self.closed = False
        self.liquidity = 1000.0
        self.volume24hr = 500.0


class FakeOrderBook:
    """Fake order book for testing."""

    def __init__(self, token_id: str, valid: bool = True) -> None:
        self.token_id = token_id
        self.has_valid_book = valid
        self.best_bid = 0.45 if valid else None
        self.best_ask = 0.55 if valid else None
        # Phase 5.5: Add attributes needed for drift detection
        if valid and self.best_bid is not None and self.best_ask is not None:
            self.mid_price: float | None = (self.best_bid + self.best_ask) / 2.0
            self.spread_bps: float | None = (
                (self.best_ask - self.best_bid) / self.mid_price * 10000
                if self.mid_price > 0
                else None
            )
        else:
            self.mid_price = None
            self.spread_bps = None
        # Empty depth levels (no drift on depth by default)
        self.bids: list[Any] = []
        self.asks: list[Any] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
        }


class FakeCacheAgeStats:
    """Fake cache age stats for testing."""

    def __init__(
        self,
        min_age: float | None = None,
        max_age: float | None = None,
        median_age: float | None = None,
        count: int = 0,
    ) -> None:
        self.min_age = min_age
        self.max_age = max_age
        self.median_age = median_age
        self.count = count


class FakeWssClient:
    """Fake WSS client for testing."""

    def __init__(self) -> None:
        self._orderbooks: dict[str, FakeOrderBook] = {}
        self._stale_tokens: set[str] = set()
        self._subscribed: set[str] = set()
        self._connected = True
        self.closed = False
        self.staleness_seconds: float = 60.0  # Phase 5.3: staleness threshold
        self._cache_ages: dict[str, float] = {}  # token_id -> age in seconds
        # Phase 5.4: Health tracking
        self.health_timeout_seconds: float = 60.0
        self._healthy: bool = True  # Simulate healthy connection

    async def connect(self) -> None:
        """Simulate connect."""
        self._connected = True

    async def wait_connected(self, timeout: float = 10.0) -> bool:  # noqa: ARG002
        """Simulate wait for connection."""
        return self._connected

    async def subscribe(self, token_ids: list[str]) -> None:
        """Record subscription."""
        self._subscribed.update(token_ids)

    def get_orderbook(self, token_id: str) -> FakeOrderBook | None:
        """Get orderbook from cache."""
        return self._orderbooks.get(token_id)

    def is_stale(self, token_id: str) -> bool:
        """Check if token is stale."""
        return token_id in self._stale_tokens

    def add_orderbook(self, token_id: str, valid: bool = True) -> None:
        """Add orderbook to cache."""
        self._orderbooks[token_id] = FakeOrderBook(token_id, valid)
        # Default to fresh cache age (0 seconds old)
        if token_id not in self._cache_ages:
            self._cache_ages[token_id] = 0.0

    def mark_stale(self, token_id: str) -> None:
        """Mark token as stale."""
        self._stale_tokens.add(token_id)

    def set_cache_age(self, token_id: str, age: float) -> None:
        """Set cache age for a token."""
        self._cache_ages[token_id] = age

    def get_cache_freshness(self, token_ids: list[str], threshold: float) -> tuple[int, int, int]:
        """Get cache freshness breakdown (Phase 5.3).

        Returns:
            Tuple of (fresh, stale, missing) counts
        """
        fresh = 0
        stale = 0
        missing = 0
        for token_id in token_ids:
            if token_id not in self._orderbooks:
                missing += 1
            elif token_id in self._stale_tokens or self._cache_ages.get(token_id, 0.0) > threshold:
                stale += 1
            else:
                fresh += 1
        return (fresh, stale, missing)

    def get_cache_ages(self, token_ids: list[str]) -> FakeCacheAgeStats:
        """Get cache age statistics (Phase 5.3)."""
        ages = [self._cache_ages.get(tid, 0.0) for tid in token_ids if tid in self._orderbooks]
        if not ages:
            return FakeCacheAgeStats()
        ages_sorted = sorted(ages)
        median = ages_sorted[len(ages_sorted) // 2]
        return FakeCacheAgeStats(
            min_age=min(ages),
            max_age=max(ages),
            median_age=median,
            count=len(ages),
        )

    # Phase 5.4: Health-gated methods
    def is_healthy(self, health_timeout: float | None = None) -> bool:  # noqa: ARG002
        """Check if WSS connection is healthy (Phase 5.4)."""
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Set health status for testing."""
        self._healthy = healthy

    def has_cached_book(self, token_id: str) -> bool:
        """Check if we have cached data for token (Phase 5.4)."""
        return token_id in self._orderbooks

    def get_orderbook_if_healthy(
        self, token_id: str, max_book_age: float | None = None
    ) -> FakeOrderBook | None:
        """Get orderbook regardless of staleness, with optional max age (Phase 5.4)."""
        if token_id not in self._orderbooks:
            return None
        # Check max_book_age if specified
        if max_book_age is not None:
            age = self._cache_ages.get(token_id, 0.0)
            if age > max_book_age:
                return None
        return self._orderbooks.get(token_id)

    async def close(self) -> None:
        """Close connection."""
        self._connected = False
        self.closed = True

    # Phase 5.5: Cache update for healing
    def update_cache(self, token_id: str, orderbook: FakeOrderBook) -> None:
        """Update cache with new orderbook (for healing)."""
        self._orderbooks[token_id] = orderbook
        self._cache_ages[token_id] = 0.0  # Reset age after heal


class FakeOrderBookFetcher:
    """Fake REST order book fetcher for testing."""

    def __init__(self) -> None:
        self._orderbooks: dict[str, FakeOrderBook] = {}
        self.closed = False
        self.fetch_count = 0

    def fetch_order_book(self, token_id: str) -> FakeOrderBook:
        """Fetch order book via REST."""
        self.fetch_count += 1
        if token_id in self._orderbooks:
            return self._orderbooks[token_id]
        return FakeOrderBook(token_id, valid=True)

    def add_orderbook(self, token_id: str, valid: bool = True) -> None:
        """Add orderbook to return."""
        self._orderbooks[token_id] = FakeOrderBook(token_id, valid)

    def close(self) -> None:
        """Close fetcher."""
        self.closed = True


# =============================================================================
# TickStats Tests
# =============================================================================


class TestTickStats:
    """Tests for TickStats dataclass."""

    def test_default_values(self) -> None:
        """TickStats should initialize with zeros and None error."""
        stats = TickStats(timestamp="2024-01-01T00:00:00Z")
        assert stats.timestamp == "2024-01-01T00:00:00Z"
        assert stats.markets_fetched == 0
        assert stats.snapshots_saved == 0
        assert stats.orderbooks_success == 0
        assert stats.wss_hits == 0
        assert stats.rest_fallbacks == 0
        assert stats.stale_count == 0
        assert stats.missing_count == 0
        assert stats.error is None

    def test_custom_values(self) -> None:
        """TickStats should accept custom values."""
        stats = TickStats(
            timestamp="2024-01-01T12:00:00Z",
            markets_fetched=100,
            snapshots_saved=95,
            orderbooks_success=90,
            wss_hits=80,
            rest_fallbacks=10,
            stale_count=5,
            missing_count=5,
            error="Test error",
        )
        assert stats.markets_fetched == 100
        assert stats.wss_hits == 80
        assert stats.error == "Test error"


# =============================================================================
# DailyStats Tests
# =============================================================================


class TestDailyStats:
    """Tests for DailyStats dataclass."""

    def test_default_values(self) -> None:
        """DailyStats should initialize with zeros and empty history."""
        stats = DailyStats(date="2024-01-01")
        assert stats.date == "2024-01-01"
        assert stats.total_ticks == 0
        assert stats.total_snapshots == 0
        assert stats.tick_history == []

    def test_tick_history_aggregation(self) -> None:
        """DailyStats can accumulate tick history."""
        stats = DailyStats(date="2024-01-01")
        tick1 = TickStats(timestamp="2024-01-01T00:00:00Z", snapshots_saved=10)
        tick2 = TickStats(timestamp="2024-01-01T01:00:00Z", snapshots_saved=15)

        stats.tick_history.append(tick1)
        stats.tick_history.append(tick2)
        stats.total_ticks = 2
        stats.total_snapshots = 25

        assert len(stats.tick_history) == 2
        assert stats.total_snapshots == 25


# =============================================================================
# RealClock Tests
# =============================================================================


class TestRealClock:
    """Tests for RealClock implementation."""

    def test_now_returns_utc(self) -> None:
        """RealClock.now() should return UTC datetime."""
        clock = RealClock()
        now = clock.now()
        assert now.tzinfo == UTC

    def test_monotonic_increases(self) -> None:
        """RealClock.monotonic() should return increasing values."""
        import time

        clock = RealClock()
        t1 = clock.monotonic()
        time.sleep(0.05)  # Sleep longer for Windows timer resolution
        t2 = clock.monotonic()
        assert t2 >= t1  # Allow equal on very fast machines


# =============================================================================
# DaemonRunner Core Tests
# =============================================================================


class TestDaemonRunnerInit:
    """Tests for DaemonRunner initialization."""

    def test_creates_export_dir(self, tmp_path: Path) -> None:
        """DaemonRunner should create export directory on init."""
        export_dir = tmp_path / "new_exports"
        assert not export_dir.exists()

        config = DaemonConfig(export_dir=export_dir)
        _ = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
        )

        assert export_dir.exists()

    def test_default_clock_and_sleep(self, tmp_path: Path) -> None:
        """DaemonRunner should use real clock/sleep if not provided."""
        config = DaemonConfig(export_dir=tmp_path)
        runner = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
        )

        assert isinstance(runner.clock, RealClock)
        assert runner.sleep_fn == real_sleep

    def test_injectable_dependencies(self, tmp_path: Path) -> None:
        """DaemonRunner should accept injectable dependencies."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_sleep = FakeSleep()

        config = DaemonConfig(export_dir=tmp_path)
        runner = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
            sleep_fn=fake_sleep,
        )

        assert runner.clock is fake_clock
        assert runner.sleep_fn is fake_sleep


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_request_shutdown_stops_loop(self, tmp_path: Path) -> None:
        """request_shutdown() should stop the daemon loop."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_sleep = FakeSleep()
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=fake_sleep,
        )

        # Run daemon in background task
        async def run_and_stop():
            # Give it time to start, then request shutdown
            await asyncio.sleep(0.01)
            runner.request_shutdown()

        # Use timeout to catch hangs as failures, not infinite waits
        await asyncio.wait_for(
            asyncio.gather(
                runner.run(),
                run_and_stop(),
            ),
            timeout=5.0,
        )

        # Should have stopped gracefully
        assert not runner.is_running
        assert runner._shutdown_requested

    @pytest.mark.asyncio
    async def test_finalize_closes_connections(self, tmp_path: Path) -> None:
        """Daemon finalize should close all connections."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_sleep = FakeSleep()
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])
        fake_wss = FakeWssClient()
        fake_ob = FakeOrderBookFetcher()

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            wss_client=fake_wss,
            ob_fetcher=fake_ob,
            clock=fake_clock,
            sleep_fn=fake_sleep,
        )

        # Request immediate shutdown and run with timeout
        runner.request_shutdown()
        await asyncio.wait_for(runner.run(), timeout=5.0)

        # All connections should be closed
        assert fake_gamma.closed
        assert fake_wss.closed
        assert fake_ob.closed

    @pytest.mark.asyncio
    async def test_finalize_is_idempotent(self, tmp_path: Path) -> None:
        """Calling finalize multiple times should be safe."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        # Call finalize directly multiple times
        await asyncio.wait_for(runner._finalize(), timeout=2.0)
        await asyncio.wait_for(runner._finalize(), timeout=2.0)  # Should be no-op

        assert fake_gamma.closed


# =============================================================================
# Tick Execution Tests
# =============================================================================


class TestTickExecution:
    """Tests for single tick execution."""

    @pytest.mark.asyncio
    async def test_tick_fetches_markets(self, tmp_path: Path) -> None:
        """Tick should fetch markets from gamma client."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        markets = [FakeMarket(f"m{i}") for i in range(5)]
        fake_gamma = FakeGammaClient(markets)

        config = DaemonConfig(
            interval_seconds=1,
            limit=10,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.markets_fetched == 5
        assert len(fake_dao.upserted_markets) == 5

    @pytest.mark.asyncio
    async def test_tick_saves_snapshots(self, tmp_path: Path) -> None:
        """Tick should save snapshots to DAO."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        markets = [FakeMarket(f"m{i}") for i in range(3)]
        fake_gamma = FakeGammaClient(markets)

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.snapshots_saved == 3
        assert len(fake_dao.saved_snapshots) == 1

    @pytest.mark.asyncio
    async def test_tick_updates_runtime_state(self, tmp_path: Path) -> None:
        """Tick should update runtime state in DAO."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        await runner._execute_tick()

        assert "daemon_last_tick" in fake_dao.runtime_state
        assert "daemon_total_ticks" in fake_dao.runtime_state
        assert fake_dao.runtime_state["daemon_total_ticks"] == "1"


# =============================================================================
# Coverage Counter Tests
# =============================================================================


class TestCoverageCounters:
    """Tests for WSS/REST coverage tracking."""

    @pytest.mark.asyncio
    async def test_wss_hits_counted(self, tmp_path: Path) -> None:
        """WSS cache hits should be counted."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        markets = [FakeMarket("m1", "0xtoken1"), FakeMarket("m2", "0xtoken2")]
        fake_gamma = FakeGammaClient(markets)
        fake_wss = FakeWssClient()
        fake_wss.add_orderbook("0xtoken1")
        fake_wss.add_orderbook("0xtoken2")

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            orderbook_source="wss",
            with_orderbook=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            wss_client=fake_wss,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.wss_hits == 2
        assert tick_stats.rest_fallbacks == 0

    @pytest.mark.asyncio
    async def test_rest_fallback_counted(self, tmp_path: Path) -> None:
        """REST fallbacks should be counted when WSS misses."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        markets = [FakeMarket("m1", "0xtoken1")]
        fake_gamma = FakeGammaClient(markets)
        fake_wss = FakeWssClient()  # No orderbooks in cache
        fake_ob = FakeOrderBookFetcher()

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            orderbook_source="wss",
            with_orderbook=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            wss_client=fake_wss,
            ob_fetcher=fake_ob,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.wss_hits == 0
        assert tick_stats.rest_fallbacks == 1
        assert fake_ob.fetch_count == 1

    @pytest.mark.asyncio
    async def test_stale_count_tracked(self, tmp_path: Path) -> None:
        """Stale WSS entries should be counted."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        markets = [FakeMarket("m1", "0xtoken1")]
        fake_gamma = FakeGammaClient(markets)
        fake_wss = FakeWssClient()
        fake_wss.mark_stale("0xtoken1")  # Mark as stale, no orderbook

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            orderbook_source="wss",
            with_orderbook=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            wss_client=fake_wss,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # No WSS hit (no orderbook), and stale flag set
        assert tick_stats.stale_count == 1

    @pytest.mark.asyncio
    async def test_missing_count_tracked(self, tmp_path: Path) -> None:
        """Markets without token_id should be counted as missing."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()
        market_no_token = FakeMarket("m1")
        market_no_token.yes_token_id = None
        fake_gamma = FakeGammaClient([market_no_token])

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.missing_count == 1


# =============================================================================
# Daily Rollover Export Tests
# =============================================================================


class TestDailyExport:
    """Tests for daily artifact export."""

    @pytest.mark.asyncio
    async def test_export_creates_coverage_json(self, tmp_path: Path) -> None:
        """Daily export should create coverage JSON."""
        stats = DailyStats(
            date="2024-01-15",
            total_ticks=100,
            total_snapshots=5000,
            total_orderbooks=4500,
            total_wss_hits=4000,
            total_rest_fallbacks=500,
            total_stale=50,
            total_missing=100,
            total_errors=2,
            start_time="2024-01-15T00:00:00Z",
            end_time="2024-01-15T23:59:59Z",
        )

        config = DaemonConfig(export_dir=tmp_path)
        runner = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
        )

        await runner._export_daily_artifacts(stats)

        coverage_path = tmp_path / "coverage_2024-01-15.json"
        assert coverage_path.exists()

        with open(coverage_path) as f:
            data = json.load(f)

        assert data["date"] == "2024-01-15"
        assert data["total_ticks"] == 100
        assert data["wss_hits"] == 4000
        assert data["rest_fallbacks"] == 500
        assert abs(data["wss_coverage_pct"] - 88.89) < 0.1  # 4000/(4000+500)

    @pytest.mark.asyncio
    async def test_export_creates_ticks_csv(self, tmp_path: Path) -> None:
        """Daily export should create ticks CSV."""
        tick1 = TickStats(timestamp="2024-01-15T00:00:00Z", snapshots_saved=50, wss_hits=40)
        tick2 = TickStats(timestamp="2024-01-15T01:00:00Z", snapshots_saved=55, wss_hits=45)

        stats = DailyStats(
            date="2024-01-15",
            tick_history=[tick1, tick2],
        )

        config = DaemonConfig(export_dir=tmp_path)
        runner = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
        )

        await runner._export_daily_artifacts(stats)

        csv_path = tmp_path / "ticks_2024-01-15.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            lines = f.readlines()

        assert len(lines) == 3  # Header + 2 rows
        assert "timestamp" in lines[0]
        assert "2024-01-15T00:00:00Z" in lines[1]

    @pytest.mark.asyncio
    async def test_export_creates_markdown_summary(self, tmp_path: Path) -> None:
        """Daily export should create markdown summary."""
        stats = DailyStats(
            date="2024-01-15",
            total_ticks=100,
            total_snapshots=5000,
            start_time="2024-01-15T00:00:00Z",
            end_time="2024-01-15T23:59:59Z",
        )

        config = DaemonConfig(export_dir=tmp_path)
        runner = DaemonRunner(
            config=config,
            dao=FakeDAO(),
            gamma_client=FakeGammaClient(),
        )

        await runner._export_daily_artifacts(stats)

        md_path = tmp_path / "daemon_summary_2024-01-15.md"
        assert md_path.exists()

        content = md_path.read_text()
        assert "# Daemon Summary - 2024-01-15" in content
        assert "Total Ticks:** 100" in content


# =============================================================================
# Max Hours Tests
# =============================================================================


class TestMaxHours:
    """Tests for max_hours time limit."""

    @pytest.mark.asyncio
    async def test_stops_after_max_hours(self, tmp_path: Path) -> None:
        """Daemon should stop after max_hours elapsed."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_sleep = FakeSleep()
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])

        config = DaemonConfig(
            interval_seconds=1,
            max_hours=0.001,  # ~3.6 seconds
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=fake_sleep,
        )

        # Advance time past max_hours on each sleep
        original_sleep = fake_sleep.__call__

        async def advancing_sleep(seconds: float) -> None:
            fake_clock.advance(10)  # Advance 10 seconds
            await original_sleep(seconds)

        fake_sleep.__call__ = advancing_sleep

        # Use timeout to catch hangs as failures
        await asyncio.wait_for(runner.run(), timeout=5.0)

        # Should have stopped due to max_hours
        assert not runner.is_running


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during tick execution."""

    @pytest.mark.asyncio
    async def test_tick_error_recorded(self, tmp_path: Path) -> None:
        """Tick errors should be recorded in stats."""
        fake_clock = FakeClock(datetime.now(UTC))
        fake_dao = FakeDAO()

        # Create a gamma client that raises an error
        class ErrorGammaClient:
            closed = False

            def list_markets(self, limit: int = 200) -> list[Any]:  # noqa: ARG002
                raise ValueError("Test API error")

            def close(self) -> None:
                self.closed = True

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=ErrorGammaClient(),
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        assert tick_stats.error is not None
        assert "Test API error" in tick_stats.error
        assert "daemon_last_error" in fake_dao.runtime_state


# =============================================================================
# Phase 5.2: Snapshot Export Tests
# =============================================================================


class TestSnapshotExport:
    """Tests for snapshot export to gzip CSV."""

    @pytest.mark.asyncio
    async def test_export_snapshots_creates_gzip_csv(self, tmp_path: Path) -> None:
        """_export_snapshots_for_day should create gzip CSV file."""
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()

        # Add test snapshots for the day
        for i in range(5):
            fake_dao.add_test_snapshot(
                {
                    "id": i + 1,
                    "market_id": f"market_{i}",
                    "yes_price": 0.5 + (i * 0.01),
                    "no_price": 0.5 - (i * 0.01),
                    "liquidity": 1000.0,
                    "volume": 500.0,
                    "snapshot_time": f"2024-01-15T12:0{i}:00",
                    "best_bid": 0.48,
                    "best_ask": 0.52,
                    "mid_price": 0.50,
                    "spread_bps": 400.0,
                    "top_depth_usd": 100.0,
                }
            )

        config = DaemonConfig(
            export_dir=tmp_path,
            snapshot_export=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        # Export snapshots
        count = await runner._export_snapshots_for_day("2024-01-15")

        assert count == 5
        output_path = tmp_path / "snapshots_2024-01-15.csv.gz"
        assert output_path.exists()

        # Verify file contents
        with gzip.open(output_path, "rt", encoding="utf-8") as f:
            lines = f.readlines()

        # Header + 5 data rows
        assert len(lines) == 6
        assert "id,market_id,yes_price" in lines[0]

    @pytest.mark.asyncio
    async def test_export_snapshots_disabled(self, tmp_path: Path) -> None:
        """_export_snapshots_for_day should skip when disabled."""
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()
        fake_dao.add_test_snapshot(
            {
                "id": 1,
                "market_id": "test",
                "yes_price": 0.5,
                "no_price": 0.5,
                "snapshot_time": "2024-01-15T12:00:00",
            }
        )

        config = DaemonConfig(
            export_dir=tmp_path,
            snapshot_export=False,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        count = await runner._export_snapshots_for_day("2024-01-15")

        assert count == 0
        output_path = tmp_path / "snapshots_2024-01-15.csv.gz"
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_export_snapshots_empty_date(self, tmp_path: Path) -> None:
        """_export_snapshots_for_day should handle empty date gracefully."""
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()  # No snapshots added

        config = DaemonConfig(
            export_dir=tmp_path,
            snapshot_export=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        count = await runner._export_snapshots_for_day("2024-01-15")

        assert count == 0
        output_path = tmp_path / "snapshots_2024-01-15.csv.gz"
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_export_snapshots_overwrites_existing(self, tmp_path: Path) -> None:
        """_export_snapshots_for_day should overwrite existing file (Windows)."""
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()
        fake_dao.add_test_snapshot(
            {
                "id": 1,
                "market_id": "test",
                "yes_price": 0.5,
                "no_price": 0.5,
                "snapshot_time": "2024-01-15T12:00:00",
            }
        )

        config = DaemonConfig(
            export_dir=tmp_path,
            snapshot_export=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        # Create existing file
        output_path = tmp_path / "snapshots_2024-01-15.csv.gz"
        output_path.write_bytes(b"old data")

        # Export should overwrite
        count = await runner._export_snapshots_for_day("2024-01-15")

        assert count == 1
        # Verify it's valid gzip now
        with gzip.open(output_path, "rt", encoding="utf-8") as f:
            content = f.read()
        assert "market_id" in content


# =============================================================================
# Phase 5.2: Retention Cleanup Tests
# =============================================================================


class TestRetentionCleanup:
    """Tests for retention cleanup."""

    @pytest.mark.asyncio
    async def test_retention_deletes_old_snapshots(self, tmp_path: Path) -> None:
        """_cleanup_retention should delete old snapshots."""
        # Set clock to 2024-01-15
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()

        # Add old snapshots (should be deleted with 7-day retention)
        for day in range(1, 10):
            fake_dao.add_test_snapshot(
                {
                    "id": day,
                    "market_id": f"market_{day}",
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "snapshot_time": f"2024-01-0{day}T12:00:00",
                }
            )

        config = DaemonConfig(
            export_dir=tmp_path,
            retention_days=7,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        # Run retention cleanup for day 14 (yesterday)
        await runner._cleanup_retention("2024-01-14")

        # Should have called delete_snapshots_before
        assert len(fake_dao.deleted_snapshots_before) == 1
        # Cutoff is 2024-01-08 (15 - 7 days)
        assert "2024-01-08T00:00:00" in fake_dao.deleted_snapshots_before[0]

    @pytest.mark.asyncio
    async def test_retention_disabled_by_default(self, tmp_path: Path) -> None:
        """_cleanup_retention should do nothing when retention_days is None."""
        fake_clock = FakeClock(datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()

        config = DaemonConfig(
            export_dir=tmp_path,
            retention_days=None,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        deleted = await runner._cleanup_retention("2024-01-14")

        assert deleted == 0
        assert len(fake_dao.deleted_snapshots_before) == 0

    @pytest.mark.asyncio
    async def test_retention_safety_check(self, tmp_path: Path) -> None:
        """_cleanup_retention should not delete if cutoff is not older than exported date."""
        # Set clock to 2024-01-10 (so cutoff with 1-day retention is 2024-01-09)
        # Trying to export 2024-01-09 would have cutoff >= exported_date
        fake_clock = FakeClock(datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC))
        fake_dao = FakeDAO()

        config = DaemonConfig(
            export_dir=tmp_path,
            retention_days=1,  # Very short retention: cutoff = 2024-01-09
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=FakeGammaClient(),
            clock=fake_clock,
        )

        # Try to run retention for the same day as cutoff (would delete today's data!)
        # Cutoff is 2024-01-09, exported_date is 2024-01-09, so cutoff >= exported_date
        deleted = await runner._cleanup_retention("2024-01-09")

        # Safety check should prevent deletion (cutoff not older than exported date)
        assert deleted == 0
        assert len(fake_dao.deleted_snapshots_before) == 0


# =============================================================================
# Phase 5.2: Day Rollover with Snapshot Export Tests
# =============================================================================


class TestDayRolloverWithSnapshotExport:
    """Tests for day rollover triggering snapshot export."""

    @pytest.mark.asyncio
    async def test_day_rollover_exports_snapshots(self, tmp_path: Path) -> None:
        """Day rollover should export snapshots for the completed day."""
        # Start on day 15
        fake_clock = FakeClock(datetime(2024, 1, 15, 23, 59, 0, tzinfo=UTC))
        fake_dao = FakeDAO()
        fake_gamma = FakeGammaClient([FakeMarket("m1")])

        # Add snapshot for day 15
        fake_dao.add_test_snapshot(
            {
                "id": 1,
                "market_id": "m1",
                "yes_price": 0.5,
                "no_price": 0.5,
                "snapshot_time": "2024-01-15T23:59:00",
            }
        )

        config = DaemonConfig(
            interval_seconds=1,
            export_dir=tmp_path,
            with_orderbook=False,
            snapshot_export=True,
            retention_days=7,
        )

        runner = DaemonRunner(
            config=config,
            dao=fake_dao,
            gamma_client=fake_gamma,
            clock=fake_clock,
            sleep_fn=FakeSleep(),
        )

        # Initialize daily stats for day 15
        runner._current_day = "2024-01-15"
        runner._daily_stats = DailyStats(
            date="2024-01-15",
            total_ticks=10,
            total_snapshots=100,
        )

        # Simulate day change to 16
        fake_clock.advance_day()

        # Trigger rollover check
        await runner._check_day_rollover()

        # Verify snapshot export was created
        snapshot_path = tmp_path / "snapshots_2024-01-15.csv.gz"
        assert snapshot_path.exists()

        # Verify retention was triggered
        assert len(fake_dao.deleted_snapshots_before) == 1


# =============================================================================
# Phase 5.2: DaemonConfig Tests
# =============================================================================


class TestDaemonConfigPhase52:
    """Tests for Phase 5.2 DaemonConfig options."""

    def test_default_snapshot_export_enabled(self) -> None:
        """snapshot_export should default to True."""
        config = DaemonConfig()
        assert config.snapshot_export is True

    def test_default_snapshot_export_format(self) -> None:
        """snapshot_export_format should default to csv_gz."""
        config = DaemonConfig()
        assert config.snapshot_export_format == "csv_gz"

    def test_default_retention_days_none(self) -> None:
        """retention_days should default to None (disabled)."""
        config = DaemonConfig()
        assert config.retention_days is None

    def test_custom_retention_days(self) -> None:
        """retention_days should accept custom value."""
        config = DaemonConfig(retention_days=30)
        assert config.retention_days == 30

    def test_snapshot_export_timeout_default(self) -> None:
        """snapshot_export_timeout should have reasonable default."""
        config = DaemonConfig()
        assert config.snapshot_export_timeout == 60.0


# =============================================================================
# Phase 5.3: Adaptive Staleness Tests
# =============================================================================


class TestAdaptiveStaleness:
    """Tests for Phase 5.3 adaptive staleness threshold."""

    def test_adaptive_staleness_default(self) -> None:
        """wss_staleness_seconds should default to None (adaptive)."""
        config = DaemonConfig()
        assert config.wss_staleness_seconds is None

    def test_adaptive_staleness_computation_60s_interval(self) -> None:
        """Adaptive staleness for 60s interval should be max(180, 60) = 180."""
        config = DaemonConfig(interval_seconds=60)
        assert config.get_effective_staleness() == 180.0

    def test_adaptive_staleness_computation_10s_interval(self) -> None:
        """Adaptive staleness for 10s interval should be max(30, 60) = 60."""
        config = DaemonConfig(interval_seconds=10)
        assert config.get_effective_staleness() == 60.0

    def test_adaptive_staleness_computation_120s_interval(self) -> None:
        """Adaptive staleness for 120s interval should be max(360, 60) = 360."""
        config = DaemonConfig(interval_seconds=120)
        assert config.get_effective_staleness() == 360.0

    def test_explicit_staleness_overrides_adaptive(self) -> None:
        """Explicit wss_staleness_seconds should override adaptive."""
        config = DaemonConfig(interval_seconds=60, wss_staleness_seconds=45.0)
        assert config.get_effective_staleness() == 45.0


# =============================================================================
# Phase 5.3: TickStats Enhanced Fields Tests
# =============================================================================


class TestTickStatsPhase53:
    """Tests for Phase 5.3 TickStats enhanced fields."""

    def test_tick_stats_has_cache_age_fields(self) -> None:
        """TickStats should have cache age tracking fields."""
        stats = TickStats(timestamp="2024-01-01T00:00:00Z")
        assert stats.wss_fresh == 0
        assert stats.wss_stale == 0
        assert stats.wss_missing == 0
        assert stats.cache_age_median == 0.0
        assert stats.cache_age_max == 0.0

    def test_tick_stats_cache_age_values(self) -> None:
        """TickStats should accept cache age values."""
        stats = TickStats(
            timestamp="2024-01-01T00:00:00Z",
            wss_fresh=150,
            wss_stale=30,
            wss_missing=20,
            cache_age_median=5.5,
            cache_age_max=25.0,
        )
        assert stats.wss_fresh == 150
        assert stats.wss_stale == 30
        assert stats.wss_missing == 20
        assert stats.cache_age_median == 5.5
        assert stats.cache_age_max == 25.0


# =============================================================================
# Phase 5.4: Health-Gated Fallback Tests
# =============================================================================


class TestPhase54Config:
    """Tests for Phase 5.4 DaemonConfig fields."""

    def test_default_health_timeout(self) -> None:
        """Default wss_health_timeout should be 60s."""
        config = DaemonConfig()
        assert config.wss_health_timeout == 60.0

    def test_default_max_book_age(self) -> None:
        """Default max_book_age should be 1800s (30 min)."""
        config = DaemonConfig()
        assert config.max_book_age == 1800.0

    def test_default_reconcile_settings(self) -> None:
        """Default reconcile settings should be set."""
        config = DaemonConfig()
        assert config.reconcile_sample == 10
        assert config.reconcile_min_age == 300.0
        assert config.reconcile_timeout == 5.0

    def test_custom_phase54_settings(self) -> None:
        """Custom Phase 5.4 settings should be accepted."""
        config = DaemonConfig(
            wss_health_timeout=30.0,
            max_book_age=3600.0,
            reconcile_sample=5,
            reconcile_min_age=600.0,
            reconcile_timeout=10.0,
        )
        assert config.wss_health_timeout == 30.0
        assert config.max_book_age == 3600.0
        assert config.reconcile_sample == 5
        assert config.reconcile_min_age == 600.0
        assert config.reconcile_timeout == 10.0


class TestTickStatsPhase54:
    """Tests for Phase 5.4 TickStats fields."""

    def test_tick_stats_has_phase54_fields(self) -> None:
        """TickStats should have Phase 5.4 health-gated fields."""
        stats = TickStats(timestamp="2024-01-01T00:00:00Z")
        assert stats.wss_cache_used == 0
        assert stats.wss_cache_quiet == 0
        assert stats.wss_cache_very_old == 0
        assert stats.wss_unhealthy_count == 0
        assert stats.wss_healthy is True
        assert stats.reconciled_count == 0
        assert stats.drift_count == 0
        assert stats.drift_max_spread == 0.0

    def test_tick_stats_phase54_values(self) -> None:
        """TickStats should accept Phase 5.4 values."""
        stats = TickStats(
            timestamp="2024-01-01T00:00:00Z",
            wss_cache_used=180,
            wss_cache_quiet=150,
            wss_cache_very_old=5,
            wss_unhealthy_count=0,
            wss_healthy=True,
            reconciled_count=10,
            drift_count=2,
            drift_max_spread=50.0,
        )
        assert stats.wss_cache_used == 180
        assert stats.wss_cache_quiet == 150
        assert stats.wss_cache_very_old == 5
        assert stats.wss_unhealthy_count == 0
        assert stats.wss_healthy is True
        assert stats.reconciled_count == 10
        assert stats.drift_count == 2
        assert stats.drift_max_spread == 50.0


class TestDailyStatsPhase54:
    """Tests for Phase 5.4 DailyStats fields."""

    def test_daily_stats_has_reconciliation_fields(self) -> None:
        """DailyStats should have Phase 5.4 reconciliation fields."""
        stats = DailyStats(date="2024-01-01")
        assert stats.total_reconciled == 0
        assert stats.total_drift == 0


class TestHealthGatedFallback:
    """Tests for Phase 5.4 health-gated fallback policy."""

    @pytest.fixture
    def dao(self) -> FakeDAO:
        """Create fake DAO."""
        return FakeDAO()

    @pytest.fixture
    def gamma_client(self) -> FakeGammaClient:
        """Create fake Gamma client with markets."""
        return FakeGammaClient(
            markets=[
                FakeMarket("market1", "0xtoken1"),
                FakeMarket("market2", "0xtoken2"),
            ]
        )

    @pytest.fixture
    def wss_client(self) -> FakeWssClient:
        """Create fake WSS client."""
        return FakeWssClient()

    @pytest.fixture
    def ob_fetcher(self) -> FakeOrderBookFetcher:
        """Create fake orderbook fetcher."""
        return FakeOrderBookFetcher()

    @pytest.mark.asyncio
    async def test_quiet_cache_no_rest_when_healthy(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """Quiet cache (old but present) should NOT trigger REST when WSS is healthy."""
        # Setup: add cached orderbooks with high age (quiet markets)
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.add_orderbook("0xtoken2", valid=True)
        wss_client.set_cache_age("0xtoken1", 500.0)  # 500s old (quiet)
        wss_client.set_cache_age("0xtoken2", 600.0)  # 600s old (quiet)
        wss_client.set_healthy(True)

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            wss_health_timeout=60.0,
            max_book_age=1800.0,  # 30 min
            reconcile_sample=0,  # Disable reconciliation for this test
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Should use WSS cache (quiet but OK), NOT REST
        assert tick_stats.wss_hits >= 2
        assert ob_fetcher.fetch_count == 0  # No REST calls

    @pytest.mark.asyncio
    async def test_unhealthy_wss_triggers_rest(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """Unhealthy WSS should trigger REST fallback for all markets."""
        # Setup: add cached orderbooks but mark WSS unhealthy
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.add_orderbook("0xtoken2", valid=True)
        wss_client.set_healthy(False)  # Unhealthy!

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            wss_health_timeout=60.0,
            reconcile_sample=0,
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Should use REST fallback due to unhealthy WSS
        assert tick_stats.wss_unhealthy_count >= 2
        assert tick_stats.wss_healthy is False
        assert ob_fetcher.fetch_count >= 2  # REST calls

    @pytest.mark.asyncio
    async def test_missing_cache_triggers_rest(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """Missing cache (not subscribed yet) should trigger REST."""
        # Setup: no cached orderbooks (first tick scenario)
        wss_client.set_healthy(True)
        # Don't add any orderbooks to cache

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            wss_health_timeout=60.0,
            reconcile_sample=0,
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Should use REST for missing cache entries
        assert tick_stats.wss_missing >= 2
        assert ob_fetcher.fetch_count >= 2  # REST calls

    @pytest.mark.asyncio
    async def test_very_old_cache_triggers_rest(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """Cache older than max_book_age should trigger REST."""
        # Setup: add very old cached orderbook
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.add_orderbook("0xtoken2", valid=True)
        wss_client.set_cache_age("0xtoken1", 2000.0)  # 33 min (> 30 min max)
        wss_client.set_cache_age("0xtoken2", 100.0)  # Fresh
        wss_client.set_healthy(True)

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            max_book_age=1800.0,  # 30 min
            reconcile_sample=0,
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # token1 is very old -> should use REST
        # token2 is fresh -> should use cache
        assert tick_stats.wss_cache_very_old >= 1
        assert ob_fetcher.fetch_count >= 1  # At least one REST call


# =============================================================================
# Phase 5.5: Drift Calibration and Cache Healing Tests
# =============================================================================


class TestPhase55Config:
    """Tests for Phase 5.5 DaemonConfig fields."""

    def test_default_drift_thresholds(self) -> None:
        """Default drift thresholds should be set."""
        config = DaemonConfig()
        assert config.reconcile_mid_bps == 25.0
        assert config.reconcile_spread_bps == 25.0
        assert config.reconcile_depth_pct == 50.0
        assert config.reconcile_depth_levels == 3

    def test_default_heal_settings(self) -> None:
        """Default healing should be enabled with cap."""
        config = DaemonConfig()
        assert config.reconcile_heal is True
        assert config.reconcile_max_heals == 25

    def test_custom_phase55_settings(self) -> None:
        """Custom Phase 5.5 settings should be accepted."""
        config = DaemonConfig(
            reconcile_mid_bps=50.0,
            reconcile_spread_bps=30.0,
            reconcile_depth_pct=75.0,
            reconcile_depth_levels=5,
            reconcile_heal=False,
            reconcile_max_heals=10,
        )
        assert config.reconcile_mid_bps == 50.0
        assert config.reconcile_spread_bps == 30.0
        assert config.reconcile_depth_pct == 75.0
        assert config.reconcile_depth_levels == 5
        assert config.reconcile_heal is False
        assert config.reconcile_max_heals == 10


class TestTickStatsPhase55:
    """Tests for Phase 5.5 TickStats fields."""

    def test_tick_stats_has_phase55_rest_buckets(self) -> None:
        """TickStats should have explicit REST fallback buckets."""
        stats = TickStats(timestamp="2024-01-01T00:00:00Z")
        assert stats.rest_missing == 0
        assert stats.rest_unhealthy == 0
        assert stats.rest_very_old == 0
        assert stats.rest_reconcile == 0

    def test_tick_stats_has_reconcile_fields(self) -> None:
        """TickStats should have reconciliation fields."""
        stats = TickStats(timestamp="2024-01-01T00:00:00Z")
        assert stats.reconcile_ok_count == 0
        assert stats.reconcile_healed_count == 0
        assert stats.drift_max_mid_bps == 0.0

    def test_tick_stats_phase55_values(self) -> None:
        """TickStats should accept Phase 5.5 values."""
        stats = TickStats(
            timestamp="2024-01-01T00:00:00Z",
            rest_missing=5,
            rest_unhealthy=0,
            rest_very_old=2,
            rest_reconcile=10,
            reconcile_ok_count=7,
            reconcile_healed_count=3,
            drift_max_mid_bps=45.5,
        )
        assert stats.rest_missing == 5
        assert stats.rest_very_old == 2
        assert stats.rest_reconcile == 10
        assert stats.reconcile_ok_count == 7
        assert stats.reconcile_healed_count == 3
        assert stats.drift_max_mid_bps == 45.5


class TestDailyStatsPhase55:
    """Tests for Phase 5.5 DailyStats fields."""

    def test_daily_stats_has_phase55_fields(self) -> None:
        """DailyStats should have Phase 5.5 REST/reconcile fields."""
        stats = DailyStats(date="2024-01-01")
        assert stats.total_rest_missing == 0
        assert stats.total_rest_unhealthy == 0
        assert stats.total_rest_very_old == 0
        assert stats.total_rest_reconcile == 0
        assert stats.total_wss_cache_fresh == 0
        assert stats.total_wss_cache_quiet == 0
        assert stats.total_reconcile_ok == 0
        assert stats.total_reconcile_healed == 0


class TestReconciliationHealing:
    """Tests for Phase 5.5 drift detection and cache healing."""

    @pytest.fixture
    def dao(self) -> FakeDAO:
        """Create fake DAO."""
        return FakeDAO()

    @pytest.fixture
    def gamma_client(self) -> FakeGammaClient:
        """Create fake Gamma client with markets."""
        return FakeGammaClient(
            markets=[
                FakeMarket("market1", "0xtoken1"),
                FakeMarket("market2", "0xtoken2"),
            ]
        )

    @pytest.fixture
    def wss_client(self) -> FakeWssClient:
        """Create fake WSS client."""
        return FakeWssClient()

    @pytest.fixture
    def ob_fetcher(self) -> FakeOrderBookFetcher:
        """Create fake orderbook fetcher."""
        return FakeOrderBookFetcher()

    @pytest.mark.asyncio
    async def test_reconciliation_counts_rest_reconcile(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """Reconciliation should increment rest_reconcile counter."""
        # Setup: old cache entries that qualify for reconciliation
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.add_orderbook("0xtoken2", valid=True)
        wss_client.set_cache_age("0xtoken1", 600.0)  # 10 min (> 5 min min_age)
        wss_client.set_cache_age("0xtoken2", 400.0)  # 6.7 min (> 5 min)
        wss_client.set_healthy(True)

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            reconcile_sample=2,
            reconcile_min_age=300.0,  # 5 min
            reconcile_heal=False,  # Don't heal for this test
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Should have REST reconcile calls
        assert tick_stats.rest_reconcile >= 1
        assert tick_stats.reconciled_count >= 1

    @pytest.mark.asyncio
    async def test_reconciliation_no_drift_increments_ok(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,  # noqa: ARG002
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """No drift should increment reconcile_ok_count."""
        # Setup: cache matches REST exactly
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.set_cache_age("0xtoken1", 600.0)  # Old enough to reconcile
        wss_client.set_healthy(True)

        # ob_fetcher returns similar orderbook (no drift)
        ob_fetcher.add_orderbook("0xtoken1", valid=True)

        # Single market for simplicity
        gamma_single = FakeGammaClient(markets=[FakeMarket("market1", "0xtoken1")])

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            reconcile_sample=1,
            reconcile_min_age=300.0,
            reconcile_mid_bps=25.0,  # Tight threshold but same data = no drift
            reconcile_heal=True,
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_single,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Should reconcile with no drift
        assert tick_stats.reconciled_count >= 1
        assert tick_stats.drift_count == 0
        assert tick_stats.reconcile_ok_count >= 1
        assert tick_stats.reconcile_healed_count == 0

    @pytest.mark.asyncio
    async def test_max_heals_cap_respected(
        self,
        dao: FakeDAO,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """reconcile_max_heals should cap healing per tick."""
        # Setup: many old cache entries
        markets = [FakeMarket(f"m{i}", f"0xtoken{i}") for i in range(10)]
        gamma_many = FakeGammaClient(markets=markets)

        for i in range(10):
            token_id = f"0xtoken{i}"
            wss_client.add_orderbook(token_id, valid=True)
            wss_client.set_cache_age(token_id, 1000.0)  # Very old
        wss_client.set_healthy(True)

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            reconcile_sample=10,  # Try to reconcile all
            reconcile_min_age=300.0,
            reconcile_heal=True,
            reconcile_max_heals=3,  # Cap at 3 heals per tick
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_many,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # Max heals should be capped
        # Note: actual heals depend on drift detection, but cap should be respected
        assert tick_stats.reconcile_healed_count <= 3

    @pytest.mark.asyncio
    async def test_heal_disabled_no_healing(
        self,
        dao: FakeDAO,
        gamma_client: FakeGammaClient,
        wss_client: FakeWssClient,
        ob_fetcher: FakeOrderBookFetcher,
    ) -> None:
        """reconcile_heal=False should disable healing."""
        # Setup: old cache
        wss_client.add_orderbook("0xtoken1", valid=True)
        wss_client.set_cache_age("0xtoken1", 600.0)
        wss_client.set_healthy(True)

        config = DaemonConfig(
            interval_seconds=60,
            orderbook_source="wss",
            reconcile_sample=1,
            reconcile_min_age=300.0,
            reconcile_heal=False,  # Disabled
        )

        runner = DaemonRunner(
            config=config,
            dao=dao,
            gamma_client=gamma_client,
            wss_client=wss_client,
            ob_fetcher=ob_fetcher,
            clock=FakeClock(datetime.now(UTC)),
            sleep_fn=FakeSleep(),
        )

        tick_stats = await runner._execute_tick()

        # No healing should occur
        assert tick_stats.reconcile_healed_count == 0
