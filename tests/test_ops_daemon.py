"""Tests for ops daemon continuous capture.

Phase 5.1: Unit tests for DaemonRunner with mocked dependencies.
Tests cover:
- Graceful shutdown
- Daily rollover export
- REST fallback
- Coverage counters
"""

from __future__ import annotations

import asyncio
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
        }


class FakeWssClient:
    """Fake WSS client for testing."""

    def __init__(self) -> None:
        self._orderbooks: dict[str, FakeOrderBook] = {}
        self._stale_tokens: set[str] = set()
        self._subscribed: set[str] = set()
        self._connected = True
        self.closed = False

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

    def mark_stale(self, token_id: str) -> None:
        """Mark token as stale."""
        self._stale_tokens.add(token_id)

    async def close(self) -> None:
        """Close connection."""
        self._connected = False
        self.closed = True


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
