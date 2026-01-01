"""Continuous snapshot capture daemon.

Phase 5.1: Production-grade continuous data capture with:
- Resilient WSS connection with REST fallback
- Coverage tracking per tick
- Daily export artifacts (CSV, JSON, markdown)
- Clean shutdown handling
"""

from __future__ import annotations

import asyncio
import csv
import json
import signal
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.gamma_client import GammaClient
    from pmq.markets.orderbook import OrderBookFetcher
    from pmq.markets.wss_market import MarketWssClient
    from pmq.storage import DAO

logger = get_logger("ops.daemon")


# =============================================================================
# Protocols for Dependency Injection
# =============================================================================


class ClockProtocol(Protocol):
    """Protocol for time provider."""

    def now(self) -> datetime:
        """Return current UTC datetime."""
        ...

    def monotonic(self) -> float:
        """Return monotonic time in seconds."""
        ...


class SleepProtocol(Protocol):
    """Protocol for async sleep function."""

    async def __call__(self, seconds: float) -> None:
        """Sleep for given seconds."""
        ...


class RealClock:
    """Real clock implementation using system time."""

    def now(self) -> datetime:
        """Return current UTC datetime."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return monotonic time."""
        import time

        return time.monotonic()


async def real_sleep(seconds: float) -> None:
    """Real async sleep."""
    await asyncio.sleep(seconds)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TickStats:
    """Statistics for a single tick."""

    timestamp: str
    markets_fetched: int = 0
    snapshots_saved: int = 0
    orderbooks_success: int = 0
    wss_hits: int = 0
    rest_fallbacks: int = 0
    stale_count: int = 0
    missing_count: int = 0
    error: str | None = None


@dataclass
class DailyStats:
    """Aggregated statistics for a day."""

    date: str
    total_ticks: int = 0
    total_snapshots: int = 0
    total_orderbooks: int = 0
    total_wss_hits: int = 0
    total_rest_fallbacks: int = 0
    total_stale: int = 0
    total_missing: int = 0
    total_errors: int = 0
    start_time: str = ""
    end_time: str = ""
    tick_history: list[TickStats] = field(default_factory=list)


@dataclass
class DaemonConfig:
    """Configuration for daemon runner."""

    interval_seconds: int = 60
    limit: int = 200
    orderbook_source: str = "wss"  # "rest" or "wss"
    wss_staleness_seconds: float = 30.0
    max_hours: float | None = None
    export_dir: Path = field(default_factory=lambda: Path("exports"))
    with_orderbook: bool = True


# =============================================================================
# Daemon Runner
# =============================================================================


class DaemonRunner:
    """Continuous snapshot capture daemon with dependency injection.

    Designed for testability with injectable:
    - Clock/time provider
    - Sleep function
    - WSS client
    - REST orderbook fetcher
    - DAO/database
    - Gamma client
    """

    def __init__(
        self,
        config: DaemonConfig,
        dao: DAO,
        gamma_client: GammaClient,
        wss_client: MarketWssClient | None = None,
        ob_fetcher: OrderBookFetcher | None = None,
        clock: ClockProtocol | None = None,
        sleep_fn: SleepProtocol | None = None,
    ) -> None:
        """Initialize daemon runner.

        Args:
            config: Daemon configuration
            dao: Data access object
            gamma_client: Gamma API client
            wss_client: WebSocket client (optional, for WSS mode)
            ob_fetcher: REST orderbook fetcher (for fallback)
            clock: Time provider (defaults to real clock)
            sleep_fn: Async sleep function (defaults to real sleep)
        """
        self.config = config
        self.dao = dao
        self.gamma_client = gamma_client
        self.wss_client = wss_client
        self.ob_fetcher = ob_fetcher
        self.clock = clock or RealClock()
        self.sleep_fn = sleep_fn or real_sleep

        # State
        self._running = False
        self._shutdown_requested = False
        self._finalized = False
        self._current_day: str | None = None
        self._daily_stats: DailyStats | None = None
        self._total_ticks = 0
        self._start_time: datetime | None = None

        # Ensure export directory exists
        self.config.export_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_running(self) -> bool:
        """Check if daemon is currently running."""
        return self._running

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

    async def run(self) -> None:
        """Main daemon loop."""
        self._running = True
        self._shutdown_requested = False
        self._start_time = self.clock.now()
        start_monotonic = self.clock.monotonic()

        # Calculate end time if max_hours specified
        end_monotonic = (
            start_monotonic + (self.config.max_hours * 3600)
            if self.config.max_hours
            else float("inf")
        )

        logger.info(
            f"Daemon starting: interval={self.config.interval_seconds}s, "
            f"source={self.config.orderbook_source}, "
            f"max_hours={self.config.max_hours or 'infinite'}"
        )

        # Connect WSS if enabled
        if self.wss_client and self.config.orderbook_source == "wss":
            await self.wss_client.connect()
            connected = await self.wss_client.wait_connected(timeout=10.0)
            if not connected:
                logger.warning("WSS connection timeout, will retry on first tick")

        try:
            while not self._shutdown_requested:
                # Check time limit
                if self.clock.monotonic() >= end_monotonic:
                    logger.info("Max hours reached, stopping")
                    break

                # Execute tick
                tick_stats = await self._execute_tick()

                # Track daily stats
                self._update_daily_stats(tick_stats)

                # Check for day rollover
                await self._check_day_rollover()

                # Wait for next tick (interruptible)
                if not self._shutdown_requested:
                    await self.sleep_fn(self.config.interval_seconds)

        except asyncio.CancelledError:
            logger.info("Daemon cancelled")
        finally:
            # Final export on shutdown
            await self._finalize()
            self._running = False

    async def _execute_tick(self) -> TickStats:
        """Execute a single snapshot tick."""
        now = self.clock.now()
        tick_stats = TickStats(timestamp=now.isoformat())
        self._total_ticks += 1

        try:
            # Fetch markets
            markets = self.gamma_client.list_markets(limit=self.config.limit)
            tick_stats.markets_fetched = len(markets)

            # Upsert market data
            self.dao.upsert_markets(markets)

            # Subscribe to WSS if enabled
            if self.wss_client:
                token_ids = [m.yes_token_id for m in markets if m.yes_token_id]
                if token_ids:
                    await self.wss_client.subscribe(token_ids)

            # Fetch order books
            orderbook_data: dict[str, Any] | None = None
            if self.config.with_orderbook:
                orderbook_data = {}
                for market in markets:
                    token_id = market.yes_token_id
                    if not token_id:
                        tick_stats.missing_count += 1
                        continue

                    ob = None

                    # Try WSS cache first
                    if self.wss_client and self.config.orderbook_source == "wss":
                        ob = self.wss_client.get_orderbook(token_id)
                        if ob and ob.has_valid_book:
                            tick_stats.wss_hits += 1
                        elif self.wss_client.is_stale(token_id):
                            tick_stats.stale_count += 1

                    # Fallback to REST
                    if ob is None and self.ob_fetcher:
                        try:
                            ob = self.ob_fetcher.fetch_order_book(token_id)
                            if self.config.orderbook_source == "wss":
                                tick_stats.rest_fallbacks += 1
                        except Exception as e:
                            logger.debug(f"REST fallback failed for {market.id}: {e}")
                            tick_stats.missing_count += 1

                    if ob and ob.has_valid_book:
                        orderbook_data[market.id] = ob.to_dict()
                        tick_stats.orderbooks_success += 1

            # Save snapshots
            snapshot_time = now.isoformat()
            snapshot_count = self.dao.save_snapshots_bulk(markets, snapshot_time, orderbook_data)
            tick_stats.snapshots_saved = snapshot_count

            # Update runtime state
            self.dao.set_runtime_state("daemon_last_tick", snapshot_time)
            self.dao.set_runtime_state("daemon_total_ticks", str(self._total_ticks))

            logger.info(
                f"Tick {self._total_ticks}: {tick_stats.snapshots_saved} snapshots, "
                f"{tick_stats.orderbooks_success} orderbooks "
                f"(WSS:{tick_stats.wss_hits}, REST:{tick_stats.rest_fallbacks})"
            )

        except Exception as e:
            tick_stats.error = str(e)
            logger.error(f"Tick error: {e}")
            self.dao.set_runtime_state("daemon_last_error", f"{now.isoformat()}: {e}")

        return tick_stats

    def _update_daily_stats(self, tick_stats: TickStats) -> None:
        """Update daily statistics with tick data."""
        today = self.clock.now().strftime("%Y-%m-%d")

        if self._current_day != today:
            # New day - initialize stats
            self._current_day = today
            self._daily_stats = DailyStats(
                date=today,
                start_time=tick_stats.timestamp,
            )

        if self._daily_stats:
            self._daily_stats.total_ticks += 1
            self._daily_stats.total_snapshots += tick_stats.snapshots_saved
            self._daily_stats.total_orderbooks += tick_stats.orderbooks_success
            self._daily_stats.total_wss_hits += tick_stats.wss_hits
            self._daily_stats.total_rest_fallbacks += tick_stats.rest_fallbacks
            self._daily_stats.total_stale += tick_stats.stale_count
            self._daily_stats.total_missing += tick_stats.missing_count
            if tick_stats.error:
                self._daily_stats.total_errors += 1
            self._daily_stats.end_time = tick_stats.timestamp
            self._daily_stats.tick_history.append(tick_stats)

    async def _check_day_rollover(self) -> None:
        """Check if day has changed and export previous day's data."""
        today = self.clock.now().strftime("%Y-%m-%d")

        if self._current_day and self._current_day != today and self._daily_stats:
            # Day changed - export previous day
            await self._export_daily_artifacts(self._daily_stats)
            self._daily_stats = None

    async def _export_daily_artifacts(self, stats: DailyStats) -> None:
        """Export daily artifacts: CSV, JSON, markdown."""
        date_str = stats.date
        export_dir = self.config.export_dir

        # Export coverage JSON
        coverage_path = export_dir / f"coverage_{date_str}.json"
        coverage_data = {
            "date": stats.date,
            "total_ticks": stats.total_ticks,
            "total_snapshots": stats.total_snapshots,
            "total_orderbooks": stats.total_orderbooks,
            "wss_hits": stats.total_wss_hits,
            "rest_fallbacks": stats.total_rest_fallbacks,
            "stale_count": stats.total_stale,
            "missing_count": stats.total_missing,
            "errors": stats.total_errors,
            "start_time": stats.start_time,
            "end_time": stats.end_time,
            "wss_coverage_pct": (
                (stats.total_wss_hits / (stats.total_wss_hits + stats.total_rest_fallbacks) * 100)
                if (stats.total_wss_hits + stats.total_rest_fallbacks) > 0
                else 0.0
            ),
        }
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(coverage_data, f, indent=2)
        logger.info(f"Exported coverage to {coverage_path}")

        # Export tick history CSV
        csv_path = export_dir / f"ticks_{date_str}.csv"
        if stats.tick_history:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "markets_fetched",
                        "snapshots_saved",
                        "orderbooks_success",
                        "wss_hits",
                        "rest_fallbacks",
                        "stale_count",
                        "missing_count",
                        "error",
                    ],
                )
                writer.writeheader()
                for tick in stats.tick_history:
                    writer.writerow(
                        {
                            "timestamp": tick.timestamp,
                            "markets_fetched": tick.markets_fetched,
                            "snapshots_saved": tick.snapshots_saved,
                            "orderbooks_success": tick.orderbooks_success,
                            "wss_hits": tick.wss_hits,
                            "rest_fallbacks": tick.rest_fallbacks,
                            "stale_count": tick.stale_count,
                            "missing_count": tick.missing_count,
                            "error": tick.error or "",
                        }
                    )
            logger.info(f"Exported ticks to {csv_path}")

        # Export markdown summary
        md_path = export_dir / f"daemon_summary_{date_str}.md"
        wss_pct = coverage_data["wss_coverage_pct"]
        md_content = f"""# Daemon Summary - {date_str}

## Overview
- **Date:** {stats.date}
- **Start:** {stats.start_time}
- **End:** {stats.end_time}
- **Total Ticks:** {stats.total_ticks}

## Snapshot Coverage
- **Total Snapshots:** {stats.total_snapshots:,}
- **Total Orderbooks:** {stats.total_orderbooks:,}

## Order Book Source Statistics
- **WSS Hits:** {stats.total_wss_hits:,}
- **REST Fallbacks:** {stats.total_rest_fallbacks:,}
- **WSS Coverage:** {wss_pct:.1f}%
- **Stale Count:** {stats.total_stale:,}
- **Missing Count:** {stats.total_missing:,}

## Errors
- **Total Errors:** {stats.total_errors}

---
*Generated by pmq ops daemon*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Exported summary to {md_path}")

    async def _finalize(self) -> None:
        """Finalize daemon: export current day stats and close connections.

        This method is idempotent and bounded - each cleanup step has a timeout
        to prevent hangs. Errors are logged but do not block other cleanup.
        """
        # Idempotency guard
        if self._finalized:
            return
        self._finalized = True

        logger.info("Finalizing daemon...")

        # Export current day's data if any (with timeout)
        if self._daily_stats and self._daily_stats.total_ticks > 0:
            try:
                await asyncio.wait_for(
                    self._export_daily_artifacts(self._daily_stats),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning("Daily export timed out after 5s")
            except Exception as e:
                logger.warning(f"Daily export failed: {e}")

        # Close WSS connection (with timeout)
        if self.wss_client:
            try:
                await asyncio.wait_for(self.wss_client.close(), timeout=2.0)
            except TimeoutError:
                logger.warning("WSS client close timed out after 2s")
            except Exception as e:
                logger.warning(f"WSS client close failed: {e}")

        # Close other resources (sync, but wrap in try/except)
        if self.ob_fetcher:
            try:
                self.ob_fetcher.close()
            except Exception as e:
                logger.warning(f"OrderBook fetcher close failed: {e}")

        try:
            self.gamma_client.close()
        except Exception as e:
            logger.warning(f"Gamma client close failed: {e}")

        logger.info("Daemon finalized")


# =============================================================================
# Signal Handlers
# =============================================================================


def setup_signal_handlers(runner: DaemonRunner) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        runner: Daemon runner instance
    """

    def handle_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, requesting shutdown")
        runner.request_shutdown()

    # Handle SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, handle_signal)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_signal)
