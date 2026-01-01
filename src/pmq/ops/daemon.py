"""Continuous snapshot capture daemon.

Phase 5.1: Production-grade continuous data capture with:
- Resilient WSS connection with REST fallback
- Coverage tracking per tick
- Daily export artifacts (CSV, JSON, markdown)
- Clean shutdown handling

Phase 5.2: Extended with:
- Daily snapshot exports (gzip CSV, atomic writes)
- Optional retention cleanup (delete old snapshots after export)

Phase 5.4: Health-gated fallback + REST reconciliation:
- WSS health tracking (connection-level, not per-market staleness)
- Quiet markets use cached data without REST fallback
- REST fallback only on: missing cache OR unhealthy WSS connection
- REST reconciliation sampler for drift detection
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
    # Phase 5.3: Enhanced cache stats (some deprecated in 5.4)
    wss_fresh: int = 0  # Fresh WSS cache hits (age <= 2*interval)
    wss_stale: int = 0  # Phase 5.3: Stale WSS cache (deprecated in 5.4 model)
    wss_missing: int = 0  # No WSS cache entry (triggered REST fallback)
    cache_age_median: float = 0.0  # Median cache age in seconds
    cache_age_max: float = 0.0  # Max cache age in seconds
    # Phase 5.4: Health-gated model stats
    wss_cache_used: int = 0  # Total WSS cache hits (regardless of age)
    wss_cache_quiet: int = 0  # Cache used but age > 2*interval (quiet market)
    wss_cache_very_old: int = 0  # Cache age > max_book_age (safety concern)
    wss_unhealthy_count: int = 0  # REST fallbacks due to unhealthy connection
    wss_healthy: bool = True  # Was WSS healthy during this tick?
    # Phase 5.4: Reconciliation stats
    reconciled_count: int = 0  # Number of tokens reconciled
    drift_count: int = 0  # Number with detected drift
    drift_max_spread: float = 0.0  # Max spread difference (bps)


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
    # Phase 5.4: Reconciliation aggregates
    total_reconciled: int = 0
    total_drift: int = 0


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
    wss_staleness_seconds: float | None = None  # Deprecated in Phase 5.4 (use health model)
    max_hours: float | None = None
    export_dir: Path = field(default_factory=lambda: Path("exports"))
    with_orderbook: bool = True
    # Phase 5.2: Snapshot export settings
    snapshot_export: bool = True  # Export daily snapshots to gzip CSV
    snapshot_export_format: str = "csv_gz"  # "csv_gz" (parquet requires extra deps)
    retention_days: int | None = None  # Delete snapshots older than N days after export
    snapshot_export_timeout: float = 60.0  # Timeout for snapshot export (seconds)
    # Phase 5.4: Health-gated fallback settings
    wss_health_timeout: float = 60.0  # Connection unhealthy if no message/pong in N seconds
    max_book_age: float = 1800.0  # Safety cap: 30 minutes max cache age
    reconcile_sample: int = 10  # Max tokens to reconcile per tick
    reconcile_min_age: float = 300.0  # Only reconcile caches older than 5 minutes
    reconcile_timeout: float = 5.0  # Timeout for reconciliation REST batch

    def get_effective_staleness(self) -> float:
        """Get effective staleness threshold (adaptive if not explicitly set).

        Note: Phase 5.4 deprecates staleness-based fallback in favor of health-gated model.
        This is kept for backwards compatibility with Phase 5.3 logging.
        """
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

        # Phase 5.4: Log config with health-based model
        logger.info(
            f"Daemon starting: interval={self.config.interval_seconds}s, "
            f"source={self.config.orderbook_source}, "
            f"health_timeout={self.config.wss_health_timeout:.0f}s, "
            f"max_book_age={self.config.max_book_age:.0f}s, "
            f"reconcile_sample={self.config.reconcile_sample}, "
            f"max_hours={self.config.max_hours or 'infinite'}"
        )

        # Connect WSS if enabled
        if self.wss_client and self.config.orderbook_source == "wss":
            # Phase 5.4: Set health timeout on WSS client
            self.wss_client.health_timeout_seconds = self.config.wss_health_timeout
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
        """Execute a single snapshot tick.

        Phase 5.4 health-gated fallback policy:
        - If cache missing: REST fetch (missing_count++)
        - Else if WSS unhealthy: REST fetch (wss_unhealthy_count++)
        - Else: use cached book (wss_cache_used++), classify as fresh/quiet/very_old
        """
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

            # Phase 5.4: Check WSS health once per tick (connection-level, not per-market)
            wss_healthy = True
            if self.wss_client and self.config.orderbook_source == "wss":
                wss_healthy = self.wss_client.is_healthy(self.config.wss_health_timeout)
                tick_stats.wss_healthy = wss_healthy
                if not wss_healthy:
                    logger.warning("WSS connection unhealthy - using REST fallback for all markets")

            # Get cache age stats for logging (Phase 5.3 compat)
            if self.wss_client and self.config.orderbook_source == "wss" and token_ids:
                cache_ages = self.wss_client.get_cache_ages(token_ids)
                tick_stats.cache_age_median = cache_ages.median_age
                tick_stats.cache_age_max = cache_ages.max_age

                # Phase 5.3 compat: compute freshness breakdown for logging
                staleness_threshold = self.config.get_effective_staleness()
                wss_fresh, wss_stale, wss_missing = self.wss_client.get_cache_freshness(
                    token_ids, staleness_threshold
                )
                tick_stats.wss_fresh = wss_fresh
                tick_stats.wss_stale = wss_stale
                tick_stats.wss_missing = wss_missing

            # Fetch order books with Phase 5.4 health-gated policy
            orderbook_data: dict[str, Any] | None = None
            if self.config.with_orderbook:
                orderbook_data = {}
                fresh_threshold = 2.0 * self.config.interval_seconds

                for market in markets:
                    token_id = market.yes_token_id
                    if not token_id:
                        tick_stats.missing_count += 1
                        continue

                    ob = None
                    used_rest = False

                    if self.wss_client and self.config.orderbook_source == "wss":
                        # Phase 5.4: Health-gated fallback logic
                        has_cache = self.wss_client.has_cached_book(token_id)

                        if not has_cache:
                            # A) Missing cache - must REST fetch
                            if self.ob_fetcher:
                                try:
                                    ob = self.ob_fetcher.fetch_order_book(token_id)
                                    used_rest = True
                                    tick_stats.wss_missing += 1
                                except Exception as e:
                                    logger.debug(f"REST fetch failed for {market.id}: {e}")
                                    tick_stats.missing_count += 1
                            else:
                                tick_stats.missing_count += 1

                        elif not wss_healthy:
                            # B) Unhealthy WSS - REST fallback
                            if self.ob_fetcher:
                                try:
                                    ob = self.ob_fetcher.fetch_order_book(token_id)
                                    used_rest = True
                                    tick_stats.wss_unhealthy_count += 1
                                except Exception as e:
                                    # Fall back to cached data if REST fails
                                    ob = self.wss_client.get_orderbook_if_healthy(
                                        token_id, max_book_age=self.config.max_book_age
                                    )
                                    logger.debug(
                                        f"REST fallback failed, using cache for {market.id}: {e}"
                                    )
                            else:
                                ob = self.wss_client.get_orderbook_if_healthy(
                                    token_id, max_book_age=self.config.max_book_age
                                )
                        else:
                            # C) Healthy WSS + has cache - use cached data
                            ob = self.wss_client.get_orderbook_if_healthy(
                                token_id, max_book_age=self.config.max_book_age
                            )
                            if ob:
                                tick_stats.wss_cache_used += 1
                                # Classify cache age
                                cache_entry_ages = self.wss_client.get_cache_ages([token_id])
                                if cache_entry_ages.count > 0:
                                    age = cache_entry_ages.max_age  # Single entry
                                    if age <= fresh_threshold:
                                        tick_stats.wss_fresh += 1
                                    elif age > self.config.max_book_age:
                                        tick_stats.wss_cache_very_old += 1
                                    else:
                                        tick_stats.wss_cache_quiet += 1
                            else:
                                # Cache exceeded max_book_age safety cap
                                tick_stats.wss_cache_very_old += 1
                                if self.ob_fetcher:
                                    try:
                                        ob = self.ob_fetcher.fetch_order_book(token_id)
                                        used_rest = True
                                    except Exception as e:
                                        logger.debug(f"REST fetch for very old cache failed: {e}")
                                        tick_stats.missing_count += 1

                    elif self.ob_fetcher:
                        # REST-only mode
                        try:
                            ob = self.ob_fetcher.fetch_order_book(token_id)
                            used_rest = True
                        except Exception as e:
                            logger.debug(f"REST fetch failed for {market.id}: {e}")
                            tick_stats.missing_count += 1

                    # Track stats
                    if ob and ob.has_valid_book:
                        orderbook_data[market.id] = ob.to_dict()
                        tick_stats.orderbooks_success += 1
                        if used_rest:
                            tick_stats.rest_fallbacks += 1
                        else:
                            tick_stats.wss_hits += 1

            # Phase 5.4: Run REST reconciliation sampler
            if (
                self.wss_client
                and self.ob_fetcher
                and self.config.orderbook_source == "wss"
                and wss_healthy
                and self.config.reconcile_sample > 0
            ):
                await self._run_reconciliation(token_ids, tick_stats)

            # Phase 5.3 compat: Compute stale_count
            tick_stats.stale_count = tick_stats.wss_stale + tick_stats.wss_missing

            # Save snapshots
            snapshot_time = now.isoformat()
            snapshot_count = self.dao.save_snapshots_bulk(markets, snapshot_time, orderbook_data)
            tick_stats.snapshots_saved = snapshot_count

            # Update runtime state
            self.dao.set_runtime_state("daemon_last_tick", snapshot_time)
            self.dao.set_runtime_state("daemon_total_ticks", str(self._total_ticks))

            # Phase 5.4: Enhanced logging
            if self.config.orderbook_source == "wss":
                wss_pct = (
                    (tick_stats.wss_hits / (tick_stats.wss_hits + tick_stats.rest_fallbacks) * 100)
                    if (tick_stats.wss_hits + tick_stats.rest_fallbacks) > 0
                    else 0.0
                )
                health_str = "healthy" if tick_stats.wss_healthy else "UNHEALTHY"
                logger.info(
                    f"Tick {self._total_ticks}: {tick_stats.snapshots_saved} snapshots, "
                    f"{tick_stats.orderbooks_success} orderbooks | "
                    f"WSS[{health_str}]:{tick_stats.wss_hits} ({wss_pct:.0f}%) "
                    f"REST:{tick_stats.rest_fallbacks} | "
                    f"cache: {tick_stats.cache_age_median:.1f}s med, {tick_stats.cache_age_max:.1f}s max | "
                    f"reconciled:{tick_stats.reconciled_count} drift:{tick_stats.drift_count}"
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

    async def _run_reconciliation(self, token_ids: list[str], tick_stats: TickStats) -> None:
        """Run REST reconciliation sampler to detect drift.

        Phase 5.4: Selects quiet caches (age >= reconcile_min_age) and
        compares WSS cache vs REST to detect drift.
        """
        if not self.wss_client or not self.ob_fetcher:
            return

        # Select tokens with old enough cache
        candidates: list[tuple[str, float]] = []  # (token_id, age)
        for token_id in token_ids:
            if self.wss_client.has_cached_book(token_id):
                ages = self.wss_client.get_cache_ages([token_id])
                if ages.count > 0 and ages.max_age >= self.config.reconcile_min_age:
                    candidates.append((token_id, ages.max_age))

        if not candidates:
            return

        # Sort by age descending, take up to reconcile_sample
        candidates.sort(key=lambda x: x[1], reverse=True)
        to_reconcile = candidates[: self.config.reconcile_sample]

        # Fetch REST with timeout
        drift_count = 0
        max_spread_diff = 0.0

        try:
            async with asyncio.timeout(self.config.reconcile_timeout):
                for token_id, _age in to_reconcile:
                    try:
                        # Get WSS cached data
                        wss_ob = self.wss_client.get_orderbook_if_healthy(token_id)
                        if not wss_ob:
                            continue

                        # Fetch REST
                        rest_ob = self.ob_fetcher.fetch_order_book(token_id)
                        if not rest_ob or not rest_ob.has_valid_book:
                            continue

                        tick_stats.reconciled_count += 1

                        # Compute drift
                        drift_detected = False
                        spread_diff = 0.0

                        if wss_ob.best_bid and rest_ob.best_bid:
                            bid_diff = abs(wss_ob.best_bid - rest_ob.best_bid)
                            if bid_diff > 0.001:  # > 0.1% difference
                                drift_detected = True
                                spread_diff = max(spread_diff, bid_diff * 10000)  # bps

                        if wss_ob.best_ask and rest_ob.best_ask:
                            ask_diff = abs(wss_ob.best_ask - rest_ob.best_ask)
                            if ask_diff > 0.001:
                                drift_detected = True
                                spread_diff = max(spread_diff, ask_diff * 10000)

                        if drift_detected:
                            drift_count += 1
                            max_spread_diff = max(max_spread_diff, spread_diff)
                            logger.debug(
                                f"Reconciliation drift for {token_id}: "
                                f"WSS bid={wss_ob.best_bid} ask={wss_ob.best_ask}, "
                                f"REST bid={rest_ob.best_bid} ask={rest_ob.best_ask}"
                            )

                    except Exception as e:
                        logger.debug(f"Reconciliation failed for {token_id}: {e}")

        except TimeoutError:
            logger.debug(f"Reconciliation timed out after {self.config.reconcile_timeout}s")

        tick_stats.drift_count = drift_count
        tick_stats.drift_max_spread = max_spread_diff

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
            # Phase 5.4: Reconciliation aggregates
            self._daily_stats.total_reconciled += tick_stats.reconciled_count
            self._daily_stats.total_drift += tick_stats.drift_count
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
        wss_coverage_pct = (
            (stats.total_wss_hits / (stats.total_wss_hits + stats.total_rest_fallbacks) * 100)
            if (stats.total_wss_hits + stats.total_rest_fallbacks) > 0
            else 0.0
        )
        drift_pct = (
            (stats.total_drift / stats.total_reconciled * 100)
            if stats.total_reconciled > 0
            else 0.0
        )
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
            "wss_coverage_pct": wss_coverage_pct,
            # Phase 5.4: Reconciliation stats
            "reconciled_count": stats.total_reconciled,
            "drift_count": stats.total_drift,
            "drift_pct": drift_pct,
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
                        # Phase 5.4 fields
                        "wss_healthy",
                        "wss_cache_used",
                        "wss_cache_quiet",
                        "reconciled_count",
                        "drift_count",
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
                            "wss_healthy": tick.wss_healthy,
                            "wss_cache_used": tick.wss_cache_used,
                            "wss_cache_quiet": tick.wss_cache_quiet,
                            "reconciled_count": tick.reconciled_count,
                            "drift_count": tick.drift_count,
                            "error": tick.error or "",
                        }
                    )
            logger.info(f"Exported ticks to {csv_path}")

        # Export markdown summary
        md_path = export_dir / f"daemon_summary_{date_str}.md"
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
- **WSS Coverage:** {wss_coverage_pct:.1f}%
- **Stale Count:** {stats.total_stale:,}
- **Missing Count:** {stats.total_missing:,}

## Reconciliation (Phase 5.4)
- **Reconciled Count:** {stats.total_reconciled:,}
- **Drift Detected:** {stats.total_drift:,}
- **Drift Percentage:** {drift_pct:.1f}%

## Errors
- **Total Errors:** {stats.total_errors}

---
*Generated by pmq ops daemon (Phase 5.4)*
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
