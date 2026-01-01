"""Continuous snapshot capture daemon.

Phase 5.1: Production-grade continuous data capture with:
- Resilient WSS connection with REST fallback
- Coverage tracking per tick
- Daily export artifacts (CSV, JSON, markdown)
- Clean shutdown handling

Phase 5.2: Extended with:
- Daily snapshot exports (gzip CSV, atomic writes)
- Optional retention cleanup (delete old snapshots after export)
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import gzip
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
    # Phase 5.3: Enhanced cache stats
    wss_fresh: int = 0  # Fresh WSS cache hits
    wss_stale: int = 0  # Stale WSS cache (triggered REST fallback)
    wss_missing: int = 0  # No WSS cache entry (triggered REST fallback)
    cache_age_median: float = 0.0  # Median cache age in seconds
    cache_age_max: float = 0.0  # Max cache age in seconds


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


def compute_adaptive_staleness(interval_seconds: int) -> float:
    """Compute adaptive staleness threshold based on daemon interval.

    Phase 5.3: Default staleness = max(3 * interval, 60 seconds).
    This prevents pathological staleness where cache is always considered stale.

    Args:
        interval_seconds: Daemon tick interval

    Returns:
        Recommended staleness threshold in seconds
    """
    return max(3.0 * interval_seconds, 60.0)


@dataclass
class DaemonConfig:
    """Configuration for daemon runner."""

    interval_seconds: int = 60
    limit: int = 200
    orderbook_source: str = "wss"  # "rest" or "wss"
    wss_staleness_seconds: float | None = None  # None = adaptive (Phase 5.3)
    max_hours: float | None = None
    export_dir: Path = field(default_factory=lambda: Path("exports"))
    with_orderbook: bool = True
    # Phase 5.2: Snapshot export settings
    snapshot_export: bool = True  # Export daily snapshots to gzip CSV
    snapshot_export_format: str = "csv_gz"  # "csv_gz" (parquet requires extra deps)
    retention_days: int | None = None  # Delete snapshots older than N days after export
    snapshot_export_timeout: float = 60.0  # Timeout for snapshot export (seconds)

    def get_effective_staleness(self) -> float:
        """Get effective staleness threshold (adaptive if not explicitly set)."""
        if self.wss_staleness_seconds is not None:
            return self.wss_staleness_seconds
        return compute_adaptive_staleness(self.interval_seconds)


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

        # Phase 5.3: Log effective staleness threshold
        effective_staleness = self.config.get_effective_staleness()
        staleness_type = "explicit" if self.config.wss_staleness_seconds else "adaptive"
        logger.info(
            f"Daemon starting: interval={self.config.interval_seconds}s, "
            f"source={self.config.orderbook_source}, "
            f"staleness={effective_staleness:.0f}s ({staleness_type}), "
            f"max_hours={self.config.max_hours or 'infinite'}"
        )

        # Connect WSS if enabled
        if self.wss_client and self.config.orderbook_source == "wss":
            # Phase 5.3: Update WSS client staleness to match daemon config
            self.wss_client.staleness_seconds = effective_staleness
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

            # Build token list for WSS operations
            token_ids = [m.yes_token_id for m in markets if m.yes_token_id]

            # Subscribe to WSS if enabled
            if self.wss_client and token_ids:
                await self.wss_client.subscribe(token_ids)

            # Phase 5.3: Get cache freshness stats before fetching orderbooks
            if self.wss_client and self.config.orderbook_source == "wss" and token_ids:
                staleness_threshold = self.config.get_effective_staleness()
                wss_fresh, wss_stale, wss_missing = self.wss_client.get_cache_freshness(
                    token_ids, staleness_threshold
                )
                tick_stats.wss_fresh = wss_fresh
                tick_stats.wss_stale = wss_stale
                tick_stats.wss_missing = wss_missing

                # Get cache age stats
                cache_ages = self.wss_client.get_cache_ages(token_ids)
                tick_stats.cache_age_median = cache_ages.median_age
                tick_stats.cache_age_max = cache_ages.max_age

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

                    # Fallback to REST if WSS returned None (stale or missing)
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

            # Phase 5.3: Compute stale_count from wss_stale + wss_missing for backwards compat
            tick_stats.stale_count = tick_stats.wss_stale + tick_stats.wss_missing

            # Save snapshots
            snapshot_time = now.isoformat()
            snapshot_count = self.dao.save_snapshots_bulk(markets, snapshot_time, orderbook_data)
            tick_stats.snapshots_saved = snapshot_count

            # Update runtime state
            self.dao.set_runtime_state("daemon_last_tick", snapshot_time)
            self.dao.set_runtime_state("daemon_total_ticks", str(self._total_ticks))

            # Phase 5.3: Enhanced logging with cache age stats
            if self.config.orderbook_source == "wss":
                wss_pct = (
                    (tick_stats.wss_hits / (tick_stats.wss_hits + tick_stats.rest_fallbacks) * 100)
                    if (tick_stats.wss_hits + tick_stats.rest_fallbacks) > 0
                    else 0.0
                )
                logger.info(
                    f"Tick {self._total_ticks}: {tick_stats.snapshots_saved} snapshots, "
                    f"{tick_stats.orderbooks_success} orderbooks | "
                    f"WSS:{tick_stats.wss_hits} ({wss_pct:.0f}%) REST:{tick_stats.rest_fallbacks} | "
                    f"cache: {tick_stats.cache_age_median:.1f}s median, {tick_stats.cache_age_max:.1f}s max"
                )
            else:
                logger.info(
                    f"Tick {self._total_ticks}: {tick_stats.snapshots_saved} snapshots, "
                    f"{tick_stats.orderbooks_success} orderbooks (REST mode)"
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
        """Check if day has changed and export previous day's data.

        On day rollover:
        1. Export daily artifacts (coverage JSON, ticks CSV, summary MD)
        2. Export snapshots for the completed day (gzip CSV)
        3. Run retention cleanup if configured
        """
        today = self.clock.now().strftime("%Y-%m-%d")

        if self._current_day and self._current_day != today and self._daily_stats:
            previous_day = self._current_day
            # Day changed - export previous day
            await self._export_daily_artifacts(self._daily_stats)
            self._daily_stats = None

            # Phase 5.2: Export snapshots with timeout
            if self.config.snapshot_export:
                try:
                    await asyncio.wait_for(
                        self._export_snapshots_for_day(previous_day),
                        timeout=self.config.snapshot_export_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        f"Snapshot export for {previous_day} timed out after {self.config.snapshot_export_timeout}s"
                    )
                except Exception as e:
                    logger.error(f"Snapshot export failed for {previous_day}: {e}")

            # Phase 5.2: Retention cleanup (only after successful export)
            if self.config.retention_days is not None:
                try:
                    await self._cleanup_retention(previous_day)
                except Exception as e:
                    logger.error(f"Retention cleanup failed: {e}")

    async def _export_snapshots_for_day(self, date_str: str) -> int:
        """Export all snapshots for a given date to gzip CSV.

        Uses atomic write (temp file -> rename) to ensure file integrity.
        Returns the number of rows exported.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Number of snapshot rows exported
        """
        if not self.config.snapshot_export:
            return 0

        export_dir = self.config.export_dir
        output_path = export_dir / f"snapshots_{date_str}.csv.gz"

        # Get snapshot count first
        snapshots = self.dao.get_snapshots_for_date(date_str)
        if not snapshots:
            logger.info(f"No snapshots to export for {date_str}")
            return 0

        # Atomic write: temp file in same directory, then rename
        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv.gz.tmp", dir=str(export_dir))
        try:
            os.close(temp_fd)  # Close the fd, we'll open with gzip
            temp_path_obj = Path(temp_path)

            # Column names from market_snapshots table
            fieldnames = [
                "id",
                "market_id",
                "yes_price",
                "no_price",
                "liquidity",
                "volume",
                "snapshot_time",
                "best_bid",
                "best_ask",
                "mid_price",
                "spread_bps",
                "top_depth_usd",
            ]

            # Stream write to gzip
            with gzip.open(temp_path_obj, "wt", encoding="utf-8", newline="") as gz:
                writer = csv.DictWriter(gz, fieldnames=fieldnames)
                writer.writeheader()

                for snapshot in snapshots:
                    writer.writerow(
                        {
                            "id": snapshot["id"],
                            "market_id": snapshot["market_id"],
                            "yes_price": snapshot["yes_price"],
                            "no_price": snapshot["no_price"],
                            "liquidity": snapshot.get("liquidity", 0.0),
                            "volume": snapshot.get("volume", 0.0),
                            "snapshot_time": snapshot["snapshot_time"],
                            "best_bid": snapshot.get("best_bid"),
                            "best_ask": snapshot.get("best_ask"),
                            "mid_price": snapshot.get("mid_price"),
                            "spread_bps": snapshot.get("spread_bps"),
                            "top_depth_usd": snapshot.get("top_depth_usd"),
                        }
                    )

            # Atomic rename (POSIX: atomic; Windows: may fail if exists)
            if sys.platform == "win32" and output_path.exists():
                output_path.unlink()
            temp_path_obj.rename(output_path)

            row_count = len(snapshots)
            logger.info(f"Exported {row_count:,} snapshots to {output_path}")
            return row_count

        except Exception:
            # Cleanup temp file on error
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)
            raise

    async def _cleanup_retention(self, exported_date: str) -> int:
        """Delete old snapshots if retention_days is configured.

        Only deletes snapshots OLDER than retention_days.
        The just-exported date is never deleted.

        Args:
            exported_date: The date that was just exported (YYYY-MM-DD)

        Returns:
            Number of rows deleted
        """
        if self.config.retention_days is None:
            return 0

        # Calculate cutoff: delete snapshots older than N days
        cutoff = self.clock.now() - timedelta(days=self.config.retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT00:00:00")

        # Safety check: cutoff must be before the exported date
        exported_start = f"{exported_date}T00:00:00"
        if cutoff_str >= exported_start:
            logger.warning(
                f"Retention cutoff {cutoff_str} is not older than exported date {exported_date}, skipping deletion"
            )
            return 0

        deleted = self.dao.delete_snapshots_before(cutoff_str)
        if deleted > 0:
            logger.info(f"Retention cleanup: deleted {deleted:,} snapshots older than {cutoff_str}")
        return deleted

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
        current_day = self._current_day
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

        # Phase 5.2: Export snapshots for current day on shutdown (with timeout)
        if self.config.snapshot_export and current_day:
            try:
                await asyncio.wait_for(
                    self._export_snapshots_for_day(current_day),
                    timeout=self.config.snapshot_export_timeout,
                )
            except TimeoutError:
                logger.warning(
                    f"Snapshot export for {current_day} timed out after {self.config.snapshot_export_timeout}s"
                )
            except Exception as e:
                logger.warning(f"Snapshot export failed for {current_day}: {e}")

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
