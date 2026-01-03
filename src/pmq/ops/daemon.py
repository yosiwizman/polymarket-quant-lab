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

Phase 5.5: Drift calibration + cache healing + metrics cleanup:
- Robust drift detection using mid/spread/depth thresholds (not strict equality)
- Cache healing: replace WSS cache with REST data when drift detected
- Explicit REST fallback buckets (missing, unhealthy, very_old, reconcile)
- Two coverage metrics: effective (excludes reconcile) and raw

Phase 5.7: Seed accounting + WSS health grace:
- rest_seed correctly counted in exports when seeding runs
- SeedResult dataclass for clear seeding metrics
- WSS health grace window to reduce rest_unhealthy at startup

Phase 5.8: REST resilience:
- Token bucket rate limiter for global REST request rate control
- Retry with exponential backoff + jitter for transient failures (429/5xx/timeout)
- New metrics: rest_retry_calls, seed_retry_calls, rest_429_count, rest_5xx_count

Phase 5.9: Seed eligibility + error taxonomy + WSS-first seeding:
- WSS-first seeding: Subscribe to WSS first, wait grace period, read from cache, REST fallback only for misses
- Seed outcome taxonomy: Classifies outcomes as ok, unseedable_*, or failed_unexpected
- New metrics: seed_from_wss, seed_from_rest, seed_unseedable, seed_failed_unexpected
- Optional skiplist for known-unseedable tokens

Phase 6.0: WSS seed bootstrap + persistent skiplist defaults:
- Fixed WSS-first seeding: Subscribe to WSS BEFORE grace wait so "book" messages populate cache
- Bootstrap wait polls for cache readiness with configurable interval (default 0.2s)
- Skiplist persistence ON by default (exports/seed_skiplist.json)
- New metric: seed_skipped_skiplist (tokens skipped due to skiplist)
- Atomic skiplist writes for safety
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
from collections.abc import Callable
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


# Phase 6.0: Default skiplist path
DEFAULT_SEED_SKIPLIST_FILENAME = "seed_skiplist.json"


@dataclass
class SeedResult:
    """Result of cache seeding operation (Phase 5.7/5.8/5.9/6.0)."""

    attempted: int = 0  # Number of tokens we tried to seed (after skiplist filter)
    succeeded: int = 0  # Number successfully cached
    rest_calls: int = 0  # REST API calls made (may be less than attempted if WSS-first)
    duration_seconds: float = 0.0  # Total seeding time
    errors: int = 0  # Number of failed fetches (legacy, kept for backward compat)
    # Phase 5.8: Retry metrics
    retry_calls: int = 0  # Total retry attempts during seeding
    http_429_count: int = 0  # 429 responses seen
    http_5xx_count: int = 0  # 5xx responses seen
    # Phase 5.9: Seed taxonomy fields
    seed_from_wss: int = 0  # Tokens seeded from WSS cache (no REST needed)
    seed_from_rest: int = 0  # Tokens seeded from REST calls
    seed_unseedable: int = 0  # Expected unseedable outcomes (empty books, 4xx)
    seed_failed_unexpected: int = 0  # True unexpected failures
    seed_unseedable_kinds: dict[str, int] = field(default_factory=dict)  # Breakdown by kind
    # Phase 6.0: Skiplist metric
    seed_skipped_skiplist: int = 0  # Tokens skipped due to skiplist (known-unseedable)
    seed_candidates_total: int = 0  # Total candidates before skiplist filter


@dataclass
class TickStats:
    """Statistics for a single tick."""

    timestamp: str
    markets_fetched: int = 0
    snapshots_saved: int = 0
    orderbooks_success: int = 0
    wss_hits: int = 0
    rest_fallbacks: int = 0
    stale_count: int = 0  # Deprecated in Phase 5.5 - kept for compat
    missing_count: int = 0
    error: str | None = None
    # Phase 5.3: Enhanced cache stats (some deprecated in 5.4/5.5)
    wss_fresh: int = 0  # Fresh WSS cache hits (age <= 2*interval)
    wss_stale: int = 0  # Deprecated - use explicit buckets
    wss_missing: int = 0  # No WSS cache entry (triggered REST fallback)
    cache_age_median: float = 0.0  # Median cache age in seconds
    cache_age_max: float = 0.0  # Max cache age in seconds
    # Phase 5.4/5.5: Health-gated model stats
    wss_cache_used: int = 0  # Total WSS cache hits (regardless of age)
    wss_cache_quiet: int = 0  # Cache used but age > 2*interval (quiet market)
    wss_cache_very_old: int = 0  # Cache age > max_book_age (safety concern)
    wss_unhealthy_count: int = 0  # REST fallbacks due to unhealthy connection
    wss_healthy: bool = True  # Was WSS healthy during this tick?
    # Phase 5.5: Explicit REST fallback buckets
    rest_missing: int = 0  # REST calls for missing cache
    rest_unhealthy: int = 0  # REST calls due to unhealthy connection
    rest_very_old: int = 0  # REST calls for very old cache
    rest_reconcile: int = 0  # REST calls for reconciliation sampling
    # Phase 5.6: Cache seeding bucket
    rest_seed: int = 0  # REST calls for initial cache seeding
    # Phase 5.5: Reconciliation stats (expanded from 5.4)
    reconciled_count: int = 0  # Number of tokens reconciled
    reconcile_ok_count: int = 0  # No drift detected
    reconcile_healed_count: int = 0  # Drift detected and cache healed
    drift_count: int = 0  # Number with detected drift (= healed if heal enabled)
    drift_max_mid_bps: float = 0.0  # Max mid price difference observed (bps)
    drift_max_spread: float = 0.0  # Deprecated - use drift_max_mid_bps
    # Phase 5.8: REST resilience metrics
    rest_retry_calls: int = 0  # Total retry attempts this tick
    rest_429_count: int = 0  # 429 responses this tick
    rest_5xx_count: int = 0  # 5xx responses this tick
    # Phase 6.1: Paper execution metrics
    paper_signals_found: int = 0  # Arbitrage signals found this tick
    paper_trades_executed: int = 0  # Paper trades executed this tick
    paper_blocked_by_risk: int = 0  # Trades blocked by risk gate
    paper_pnl_total: float = 0.0  # Total PnL (realized + unrealized)
    # Phase 6.2: Explain mode tracking
    paper_explain_candidates: int = 0  # Number of explain candidates this tick
    paper_explain_top_edge_bps: float = 0.0  # Top candidate edge this tick


@dataclass
class DailyStats:
    """Aggregated statistics for a day."""

    date: str
    total_ticks: int = 0
    total_snapshots: int = 0
    total_orderbooks: int = 0
    total_wss_hits: int = 0
    total_rest_fallbacks: int = 0
    total_stale: int = 0  # Deprecated - kept for compat
    total_missing: int = 0
    total_errors: int = 0
    start_time: str = ""
    end_time: str = ""
    tick_history: list[TickStats] = field(default_factory=list)
    # Phase 5.5: Explicit REST fallback buckets
    total_rest_missing: int = 0
    total_rest_unhealthy: int = 0
    total_rest_very_old: int = 0
    total_rest_reconcile: int = 0
    # Phase 5.6: Cache seeding bucket
    total_rest_seed: int = 0
    # Phase 5.5: WSS cache buckets
    total_wss_cache_fresh: int = 0
    total_wss_cache_quiet: int = 0
    # Phase 5.5: Reconciliation aggregates (expanded from 5.4)
    total_reconciled: int = 0
    total_reconcile_ok: int = 0
    total_reconcile_healed: int = 0
    total_drift: int = 0
    # Phase 5.8: REST resilience metrics
    total_rest_retry_calls: int = 0  # Total retry attempts across all REST calls
    seed_retry_calls: int = 0  # Retry attempts during seeding only
    total_rest_429: int = 0  # Total 429 responses
    total_rest_5xx: int = 0  # Total 5xx responses
    # Phase 6.1: Paper execution metrics (daily totals)
    paper_signals_found: int = 0  # Total arbitrage signals found
    paper_trades_executed: int = 0  # Total paper trades executed
    paper_blocked_by_risk: int = 0  # Total trades blocked by risk gate
    paper_pnl_realized: float = 0.0  # Realized PnL (end of day)
    paper_pnl_unrealized: float = 0.0  # Unrealized PnL (end of day)
    paper_pnl_total: float = 0.0  # Total PnL (end of day)
    # Phase 6.2: Explain mode daily aggregates
    paper_explain_rejection_counts: dict[str, int] = field(default_factory=dict)  # reason -> count
    paper_explain_ticks_with_candidates: int = 0  # Ticks with at least 1 candidate
    paper_explain_ticks_above_min_edge: int = 0  # Ticks with top edge >= min_edge
    paper_explain_avg_top_edge_bps: float = 0.0  # Average top edge across ticks
    paper_explain_max_top_edge_bps: float = 0.0  # Max top edge across ticks


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
    # Phase 5.5: Drift thresholds and healing
    reconcile_mid_bps: float = 25.0  # Mid price diff threshold (bps)
    reconcile_spread_bps: float = 25.0  # Spread diff threshold (bps)
    reconcile_depth_pct: float = 50.0  # Depth diff threshold (%)
    reconcile_depth_levels: int = 3  # Number of levels for depth comparison
    reconcile_heal: bool = True  # Replace cache with REST data on drift
    reconcile_max_heals: int = 25  # Max heals per tick (prevent storms)
    # Phase 5.6: Cache seeding settings
    seed_cache: bool = True  # Pre-populate cache at startup to reduce cold-start REST calls
    seed_max: int | None = None  # Max tokens to seed (None = use limit)
    seed_concurrency: int = 10  # Concurrent REST fetches during seeding
    seed_timeout: float = 30.0  # Total timeout for seeding phase
    # Phase 5.7: WSS health grace period
    wss_health_grace_seconds: float = (
        60.0  # Grace period after connect before enforcing health timeout
    )
    # Phase 5.8: REST resilience settings
    rest_rps: float = 8.0  # REST requests per second (rate limiter)
    rest_burst: int = 8  # REST burst capacity (token bucket size)
    rest_max_retries: int = 3  # Max retry attempts (not counting initial)
    rest_backoff_base: float = 0.25  # Base backoff delay in seconds
    rest_backoff_max: float = 3.0  # Maximum backoff delay in seconds
    # Phase 5.9/6.0: WSS-first seeding settings
    seed_mode: str = "wss_first"  # "rest" (legacy) or "wss_first" (Phase 5.9)
    seed_grace_seconds: float = 10.0  # Time to wait for WSS cache to populate (8-12s recommended)
    seed_skiplist_path: Path | None = (
        None  # Path to skiplist (None = use default: exports/seed_skiplist.json)
    )
    seed_skiplist_enabled: bool = True  # Phase 6.0: Enable skiplist persistence (default ON)
    # Phase 6.0: Bootstrap wait polling
    seed_bootstrap_poll_interval: float = 0.2  # Interval for polling cache readiness during grace
    # Phase 6.1: Paper execution settings
    paper_exec_enabled: bool = False  # Enable paper trading in daemon (default OFF for safety)
    paper_exec_max_trades_per_tick: int = 3  # Max paper trades per tick
    paper_exec_max_markets_scanned: int = 200  # Max markets to scan for signals
    paper_exec_min_edge_bps: float = 50.0  # Minimum edge in basis points
    paper_exec_trade_quantity: float = 10.0  # Quantity per paper trade
    paper_exec_require_approval: bool = True  # Require governance approval
    paper_exec_strategy_name: str = "paper_exec"  # Strategy name for governance
    # Phase 6.2: Paper exec explain mode settings
    paper_exec_explain: bool = False  # Enable explain mode (capture all candidates, disabled by default)
    paper_exec_explain_top_n: int = 10  # Number of top candidates to track per tick
    paper_exec_explain_export_path: Path | None = None  # JSONL export path (None = auto)

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
        # Phase 5.6/5.7: Seeding state
        self._seeded = False
        self._seed_result: SeedResult | None = None  # Phase 5.7: Full seeding metrics

        # Phase 5.8: REST resilience - rate limiter and config
        from pmq.ops.rest_resilience import (
            RestResilienceConfig,
            TokenBucketRateLimiter,
        )

        self._rest_resilience_config = RestResilienceConfig(
            rps=config.rest_rps,
            burst=config.rest_burst,
            max_retries=config.rest_max_retries,
            backoff_base=config.rest_backoff_base,
            backoff_max=config.rest_backoff_max,
        )
        self._rate_limiter = TokenBucketRateLimiter(
            rps=config.rest_rps,
            burst=config.rest_burst,
            clock=self.clock,
        )

        # Phase 6.1: Paper executor (lazy init to avoid import cycles)
        self._paper_executor: Any = None

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
        # Phase 6.0.1: Don't overwrite prior shutdown request (for testability)
        # This allows tests to call request_shutdown() before run()
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

        # Phase 6.0.1: Wrap entire body in try/finally to ensure _finalize() is always called
        try:
            # Phase 6.0.1: Check shutdown before any long operations
            if self._shutdown_requested:
                logger.info("Shutdown already requested, skipping startup")
                return

            # Connect WSS if enabled
            if self.wss_client and self.config.orderbook_source == "wss":
                # Phase 5.4: Set health timeout on WSS client
                self.wss_client.health_timeout_seconds = self.config.wss_health_timeout
                # Phase 5.7: Set health grace period
                self.wss_client.health_grace_seconds = self.config.wss_health_grace_seconds
                await self.wss_client.connect()
                connected = await self.wss_client.wait_connected(timeout=10.0)
                if not connected:
                    logger.warning("WSS connection timeout, will retry on first tick")

            # Phase 6.0.1: Check shutdown after WSS connect
            if self._shutdown_requested:
                logger.info("Shutdown requested during startup")
                return

            # Phase 5.6/5.7: Seed cache before first tick
            if (
                self.config.seed_cache
                and self.wss_client
                and self.ob_fetcher
                and self.config.orderbook_source == "wss"
            ):
                seed_result = await self._seed_cache()
                self._seed_result = seed_result
                # Phase 5.7/5.8: Initialize daily stats with seed metrics if needed
                if seed_result.rest_calls > 0:
                    self._ensure_daily_stats_initialized()
                    if self._daily_stats:
                        self._daily_stats.total_rest_seed += seed_result.rest_calls
                        # Phase 5.8: Propagate retry metrics from seeding
                        self._daily_stats.seed_retry_calls += seed_result.retry_calls
                        self._daily_stats.total_rest_retry_calls += seed_result.retry_calls
                        self._daily_stats.total_rest_429 += seed_result.http_429_count
                        self._daily_stats.total_rest_5xx += seed_result.http_5xx_count

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
            # Phase 6.0.1: _finalize() is always called, even on early return
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
                                    tick_stats.rest_missing += 1  # Phase 5.5
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
                                    tick_stats.rest_unhealthy += 1  # Phase 5.5
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
                                        tick_stats.rest_very_old += 1  # Phase 5.5
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

            # Phase 6.1: Run paper execution if enabled
            if self.config.paper_exec_enabled:
                await self._run_paper_execution(markets, tick_stats)

            # Update runtime state
            self.dao.set_runtime_state("daemon_last_tick", snapshot_time)
            self.dao.set_runtime_state("daemon_total_ticks", str(self._total_ticks))

            # Phase 5.5: Enhanced logging with healing info
            if self.config.orderbook_source == "wss":
                wss_pct = (
                    (tick_stats.wss_hits / (tick_stats.wss_hits + tick_stats.rest_fallbacks) * 100)
                    if (tick_stats.wss_hits + tick_stats.rest_fallbacks) > 0
                    else 0.0
                )
                health_str = "healthy" if tick_stats.wss_healthy else "UNHEALTHY"
                # Phase 5.5: Show reconcile stats including heals
                reconcile_str = (
                    f"reconciled:{tick_stats.reconciled_count} "
                    f"ok:{tick_stats.reconcile_ok_count} "
                    f"drift:{tick_stats.drift_count} "
                    f"healed:{tick_stats.reconcile_healed_count}"
                )
                logger.info(
                    f"Tick {self._total_ticks}: {tick_stats.snapshots_saved} snapshots, "
                    f"{tick_stats.orderbooks_success} orderbooks | "
                    f"WSS[{health_str}]:{tick_stats.wss_hits} ({wss_pct:.0f}%) "
                    f"REST:{tick_stats.rest_fallbacks} | "
                    f"cache: {tick_stats.cache_age_median:.1f}s med, {tick_stats.cache_age_max:.1f}s max | "
                    f"{reconcile_str}"
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
        """Run REST reconciliation sampler to detect and heal drift.

        Phase 5.5: Robust drift detection using configurable thresholds for
        mid price, spread, and depth differences. Optionally heals cache
        by replacing WSS data with REST data when drift is detected.
        """
        from pmq.ops.drift import DriftThresholds, compute_drift_metrics

        if not self.wss_client or not self.ob_fetcher:
            return

        # Build drift thresholds from config
        thresholds = DriftThresholds(
            mid_diff_bps=self.config.reconcile_mid_bps,
            spread_diff_bps=self.config.reconcile_spread_bps,
            depth_diff_pct=self.config.reconcile_depth_pct,
            depth_levels=self.config.reconcile_depth_levels,
        )

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

        # Track healing stats
        heals_this_tick = 0
        max_mid_diff_bps = 0.0

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
                        tick_stats.rest_reconcile += 1

                        if not rest_ob or not rest_ob.has_valid_book:
                            continue

                        tick_stats.reconciled_count += 1

                        # Phase 5.5: Use robust drift detection
                        metrics = compute_drift_metrics(wss_ob, rest_ob, thresholds)

                        # Track max mid diff observed
                        if metrics.mid_diff_bps is not None:
                            max_mid_diff_bps = max(max_mid_diff_bps, metrics.mid_diff_bps)

                        if metrics.has_drift:
                            tick_stats.drift_count += 1

                            # Phase 5.5: Cache healing
                            if (
                                self.config.reconcile_heal
                                and heals_this_tick < self.config.reconcile_max_heals
                            ):
                                self.wss_client.update_cache(token_id, rest_ob)
                                tick_stats.reconcile_healed_count += 1
                                heals_this_tick += 1
                                logger.debug(
                                    f"Healed cache for {token_id[:16]}...: {metrics.drift_reason}"
                                )
                            else:
                                logger.debug(
                                    f"Drift detected (not healed) for {token_id[:16]}...: {metrics.drift_reason}"
                                )
                        else:
                            tick_stats.reconcile_ok_count += 1

                    except Exception as e:
                        logger.debug(f"Reconciliation failed for {token_id[:16]}...: {e}")

        except TimeoutError:
            logger.debug(f"Reconciliation timed out after {self.config.reconcile_timeout}s")

        tick_stats.drift_max_mid_bps = max_mid_diff_bps
        # Deprecated field for compat
        tick_stats.drift_max_spread = max_mid_diff_bps

    async def _seed_cache(self) -> SeedResult:
        """Pre-populate WSS cache to reduce cold-start missing.

        Phase 5.9: WSS-first seeding with outcome taxonomy.
        Phase 6.0: Fixed bootstrap - subscribe to WSS BEFORE grace wait.

        - WSS-first mode: Subscribe to WSS first, wait grace period with polling,
          read from cache, REST fallback only for misses
        - REST mode (legacy): Use REST to seed all missing tokens
        - Seed outcomes classified as: ok, unseedable_*, or failed_unexpected

        Returns:
            SeedResult with detailed metrics including Phase 5.9/6.0 taxonomy fields
        """
        from pmq.ops.rest_resilience import RestCallStats, retry_rest_call
        from pmq.ops.seed_taxonomy import (
            SeedOutcome,
            SeedOutcomeKind,
            SeedOutcomeStats,
            SeedSkiplist,
            classify_orderbook_result,
        )

        # Check if seeding is enabled
        if not self.config.seed_cache:
            return SeedResult()

        if not self.wss_client or not self.ob_fetcher:
            return SeedResult()

        if self._seeded:
            return self._seed_result or SeedResult()  # Already seeded

        # Phase 6.0.1: Check shutdown before starting seeding
        if self._shutdown_requested:
            logger.info("Shutdown requested, skipping cache seeding")
            return SeedResult()

        # Capture in local vars for type narrowing (mypy doesn't track across nested functions)
        wss_client = self.wss_client
        ob_fetcher = self.ob_fetcher
        resilience_config = self._rest_resilience_config
        rate_limiter = self._rate_limiter
        seed_mode = self.config.seed_mode

        logger.info(f"Cache seeding started (mode={seed_mode})...")
        start_time = self.clock.monotonic()

        # Phase 6.0: Determine skiplist path (default to exports/seed_skiplist.json)
        skiplist: SeedSkiplist | None = None
        skiplist_path = self.config.seed_skiplist_path
        if skiplist_path is None and self.config.seed_skiplist_enabled:
            # Use default path
            skiplist_path = self.config.export_dir / DEFAULT_SEED_SKIPLIST_FILENAME

        if skiplist_path is not None and self.config.seed_skiplist_enabled:
            skiplist = SeedSkiplist(skiplist_path)
            skiplist.load()
            logger.info(f"Loaded seed skiplist with {len(skiplist)} entries from {skiplist_path}")

        # Get initial market list
        markets = self.gamma_client.list_markets(limit=self.config.limit)
        token_ids = [m.yes_token_id for m in markets if m.yes_token_id]

        if not token_ids:
            self._seeded = True
            return SeedResult()

        # Phase 6.0: Track candidates before skiplist filter
        seed_candidates_total = len(token_ids)
        seed_skipped_skiplist = 0

        # Phase 5.9/6.0: Filter out skiplisted tokens
        if skiplist:
            original_count = len(token_ids)
            token_ids = [tid for tid in token_ids if not skiplist.should_skip(tid)]
            seed_skipped_skiplist = original_count - len(token_ids)
            if seed_skipped_skiplist > 0:
                logger.info(f"Skipped {seed_skipped_skiplist} tokens from skiplist")

        # Apply seed_max cap
        seed_max = self.config.seed_max or self.config.limit
        to_seed = token_ids[:seed_max]

        # Phase 5.9/6.0: WSS-first seeding with bootstrap fix
        seed_from_wss = 0
        to_rest_seed: list[str] = []

        if seed_mode == "wss_first":
            # Phase 6.0 FIX: Subscribe to WSS BEFORE grace wait!
            # Polymarket emits "book" message on subscribe, so we need to subscribe first
            logger.info(f"WSS-first: subscribing to {len(to_seed)} tokens before grace wait...")
            await wss_client.subscribe(to_seed)

            # Phase 6.0.1: Check shutdown after subscribe
            if self._shutdown_requested:
                logger.info("Shutdown requested, aborting cache seeding")
                self._seeded = True
                return SeedResult()

            # Bootstrap wait with polling for cache readiness
            grace_seconds = self.config.seed_grace_seconds
            poll_interval = self.config.seed_bootstrap_poll_interval
            logger.info(
                f"WSS-first: waiting up to {grace_seconds:.1f}s for cache population (poll={poll_interval:.2f}s)..."
            )

            ready_tokens: set[str] = set()
            missing_tokens: set[str] = set(to_seed)

            # Phase 6.0.1: Use iteration-bounded loop instead of time-bounded
            # This ensures deterministic behavior under FakeClock/FakeSleep
            max_iters = int(grace_seconds / poll_interval) if poll_interval > 0 else 0
            iterations_done = 0

            for _ in range(max_iters):
                # Phase 6.0.1: Check shutdown each iteration
                if self._shutdown_requested:
                    logger.info("Shutdown requested during bootstrap grace wait")
                    break

                # Check which tokens became ready
                for tid in list(missing_tokens):
                    if wss_client.has_cached_book(tid):
                        cached_ob = wss_client.get_orderbook(tid, allow_stale=True)
                        if cached_ob and cached_ob.has_valid_book:
                            ready_tokens.add(tid)
                            missing_tokens.discard(tid)

                # All ready, exit early
                if not missing_tokens:
                    logger.debug(f"All {len(ready_tokens)} tokens ready, exiting grace wait early")
                    break

                # Poll interval sleep
                await self.sleep_fn(poll_interval)
                iterations_done += 1

            # Count results
            seed_from_wss = len(ready_tokens)
            to_rest_seed = list(missing_tokens)

            # Phase 6.0.1: If shutdown requested, skip REST seeding
            if self._shutdown_requested:
                self._seeded = True
                return SeedResult(
                    seed_from_wss=seed_from_wss,
                    seed_skipped_skiplist=seed_skipped_skiplist,
                    seed_candidates_total=seed_candidates_total,
                )

            logger.info(
                f"WSS-first: {seed_from_wss} from WSS cache, {len(to_rest_seed)} need REST fallback "
                f"(iterations={iterations_done})"
            )
        else:
            # Legacy REST mode: seed all missing tokens
            to_rest_seed = [tid for tid in to_seed if not wss_client.has_cached_book(tid)]
            logger.info(f"REST mode: {len(to_rest_seed)} tokens to seed")

        if not to_rest_seed:
            # All tokens seeded from WSS
            duration = self.clock.monotonic() - start_time
            self._seeded = True
            result = SeedResult(
                attempted=len(to_seed),
                succeeded=seed_from_wss,
                rest_calls=0,
                duration_seconds=duration,
                errors=0,
                seed_from_wss=seed_from_wss,
                seed_from_rest=0,
                seed_unseedable=0,
                seed_failed_unexpected=0,
                seed_skipped_skiplist=seed_skipped_skiplist,
                seed_candidates_total=seed_candidates_total,
            )
            logger.info(
                f"Cache seeding completed: {seed_from_wss}/{len(to_seed)} from WSS in {duration:.1f}s "
                f"(skiplist={seed_skipped_skiplist})"
            )
            return result

        # REST seeding for remaining tokens
        outcome_stats = SeedOutcomeStats()
        seed_from_rest = 0
        total_retries = 0
        total_429 = 0
        total_5xx = 0
        concurrency = self.config.seed_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        # Factory function to create typed callables for each token (for mypy)
        def make_fetch_fn(tid: str) -> Callable[[], Any]:
            return lambda: ob_fetcher.fetch_order_book(tid)

        async def seed_one(token_id: str) -> tuple[SeedOutcome, RestCallStats]:
            """Fetch and cache one token with retry, return outcome."""
            async with semaphore:
                try:
                    # Phase 5.8: Use retry_rest_call with rate limiting
                    # Note: retry_rest_call expects sync callable, wraps in executor
                    ob, stats = await retry_rest_call(
                        call_fn=make_fetch_fn(token_id),
                        config=resilience_config,
                        rate_limiter=rate_limiter,
                    )
                    # Classify outcome
                    outcome = classify_orderbook_result(ob, error=None)
                    outcome = SeedOutcome(
                        token_id=token_id,
                        kind=outcome.kind,
                        error_detail=outcome.error_detail,
                    )
                    if ob and ob.has_valid_book:
                        wss_client.update_cache(token_id, ob)
                    return outcome, stats
                except Exception as e:
                    # Classify exception
                    outcome = classify_orderbook_result(None, error=e)
                    outcome = SeedOutcome(
                        token_id=token_id,
                        kind=outcome.kind,
                        error_detail=outcome.error_detail,
                    )
                    return outcome, RestCallStats()

        try:
            async with asyncio.timeout(self.config.seed_timeout):
                tasks = [seed_one(tid) for tid in to_rest_seed]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        # Unexpected exception from gather itself
                        outcome = SeedOutcome(
                            token_id=to_rest_seed[i],
                            kind=SeedOutcomeKind.FAILED_UNEXPECTED,
                            error_detail=f"{type(r).__name__}: {r}",
                        )
                        outcome_stats.add(outcome)
                    elif isinstance(r, tuple):
                        outcome, stats = r
                        outcome_stats.add(outcome)
                        if outcome.is_success:
                            seed_from_rest += 1
                        # Update skiplist for unseedable tokens
                        if skiplist and outcome.is_unseedable:
                            skiplist.add(outcome)  # Pass full SeedOutcome object
                        # Aggregate retry stats
                        total_retries += stats.retry_count
                        total_429 += stats.http_429_count
                        total_5xx += stats.http_5xx_count
        except TimeoutError:
            logger.warning(f"Cache seeding timed out after {self.config.seed_timeout}s")
            # Mark remaining as failed_unexpected
            completed = outcome_stats.total
            for tid in to_rest_seed[completed:]:
                outcome = SeedOutcome(
                    token_id=tid,
                    kind=SeedOutcomeKind.FAILED_UNEXPECTED,
                    error_detail="Seeding timeout",
                )
                outcome_stats.add(outcome)

        # Save skiplist if updated
        if skiplist and skiplist.dirty:
            skiplist.save()
            logger.info(f"Updated seed skiplist with {len(skiplist)} entries")

        duration = self.clock.monotonic() - start_time
        self._seeded = True

        # Build result with Phase 5.9 fields
        total_succeeded = seed_from_wss + seed_from_rest
        # Legacy errors field = unseedable + failed_unexpected (for backward compat)
        legacy_errors = outcome_stats.unseedable + outcome_stats.failed_unexpected

        result = SeedResult(
            attempted=len(to_seed),
            succeeded=total_succeeded,
            rest_calls=len(to_rest_seed),  # Only REST calls, not WSS-seeded
            duration_seconds=duration,
            errors=legacy_errors,  # Kept for backward compat
            # Phase 5.8: Retry metrics
            retry_calls=total_retries,
            http_429_count=total_429,
            http_5xx_count=total_5xx,
            # Phase 5.9: Seed taxonomy fields
            seed_from_wss=seed_from_wss,
            seed_from_rest=seed_from_rest,
            seed_unseedable=outcome_stats.unseedable,
            seed_failed_unexpected=outcome_stats.failed_unexpected,
            seed_unseedable_kinds=dict(outcome_stats.kinds),
            # Phase 6.0: Skiplist metrics
            seed_skipped_skiplist=seed_skipped_skiplist,
            seed_candidates_total=seed_candidates_total,
        )

        logger.info(
            f"Cache seeding completed: {total_succeeded}/{len(to_seed)} "
            f"(wss={seed_from_wss}, rest={seed_from_rest}, unseedable={outcome_stats.unseedable}, "
            f"failed={outcome_stats.failed_unexpected}, skiplist={seed_skipped_skiplist}) in {duration:.1f}s"
        )
        return result

    async def _run_paper_execution(self, markets: list[Any], tick_stats: TickStats) -> None:
        """Run paper execution loop for this tick.

        Phase 6.1: Generates signals from market snapshots and executes
        paper trades through PaperLedger only.

        Phase 6.2: With explain mode, captures ALL candidates with rejection
        reasons and exports to JSONL for calibration.

        HARD RULE: No real order placement. All trades go to PaperLedger.

        Args:
            markets: List of market objects from this tick
            tick_stats: Tick stats to update with paper execution metrics
        """
        from pmq.ops.paper_exec import (
            PaperExecConfig,
            PaperExecutor,
            get_default_export_path,
            write_explain_tick,
        )

        # Lazy init paper executor
        if self._paper_executor is None:
            paper_config = PaperExecConfig(
                enabled=self.config.paper_exec_enabled,
                max_trades_per_tick=self.config.paper_exec_max_trades_per_tick,
                max_markets_scanned=self.config.paper_exec_max_markets_scanned,
                min_signal_edge_bps=self.config.paper_exec_min_edge_bps,
                trade_quantity=self.config.paper_exec_trade_quantity,
                require_approval=self.config.paper_exec_require_approval,
                # Phase 6.2: Explain mode settings
                explain_enabled=self.config.paper_exec_explain,
                explain_top_n=self.config.paper_exec_explain_top_n,
                explain_export_path=self.config.paper_exec_explain_export_path,
            )

            # Try to get risk gate (optional)
            risk_gate = None
            if self.config.paper_exec_require_approval:
                try:
                    from pmq.governance import RiskGate

                    risk_gate = RiskGate(dao=self.dao)
                except Exception as e:
                    logger.warning(f"Could not initialize RiskGate: {e}")

            self._paper_executor = PaperExecutor(
                config=paper_config,
                dao=self.dao,
                risk_gate=risk_gate,
            )
            logger.info(
                f"Paper executor initialized: enabled={paper_config.enabled}, "
                f"max_trades={paper_config.max_trades_per_tick}, "
                f"explain={paper_config.explain_enabled}"
            )

        # Convert markets to dict format for scanner
        markets_data = []
        for m in markets:
            # Build market dict compatible with scan_from_db
            markets_data.append(
                {
                    "id": m.id,
                    "question": getattr(m, "question", ""),
                    "active": getattr(m, "active", True),
                    "closed": getattr(m, "closed", False),
                    "last_price_yes": getattr(m, "yes_price", 0.0),
                    "last_price_no": getattr(m, "no_price", 0.0),
                    "liquidity": getattr(m, "liquidity", 0.0),
                }
            )

        # Execute paper trading tick
        try:
            result = self._paper_executor.execute_tick(
                markets_data=markets_data,
                strategy_name=self.config.paper_exec_strategy_name,
            )

            # Update tick stats with paper execution metrics
            tick_stats.paper_signals_found = result.signals_found
            tick_stats.paper_trades_executed = result.trades_executed
            tick_stats.paper_blocked_by_risk = result.blocked_by_risk + result.blocked_by_safety
            tick_stats.paper_pnl_total = result.total_pnl

            # Phase 6.2: Update explain mode tick stats
            if self.config.paper_exec_explain and result.explain_candidates:
                tick_stats.paper_explain_candidates = len(result.explain_candidates)
                tick_stats.paper_explain_top_edge_bps = result.explain_candidates[0].edge_bps

                # Export tick to JSONL
                export_path = (
                    self.config.paper_exec_explain_export_path
                    or get_default_export_path(self.config.export_dir)
                )
                write_explain_tick(
                    export_path=export_path,
                    tick_timestamp=tick_stats.timestamp,
                    result=result,
                )

            # Log paper execution if any activity
            if result.trades_executed > 0:
                logger.info(
                    f"Paper exec: {result.signals_found} signals, "
                    f"{result.trades_executed} trades, "
                    f"PnL=${result.total_pnl:.2f}"
                )

        except Exception as e:
            logger.error(f"Paper execution error: {e}")
            tick_stats.error = (tick_stats.error or "") + f" Paper exec error: {e}"

    def _ensure_daily_stats_initialized(self) -> None:
        """Ensure daily stats are initialized for today (Phase 5.7).

        Used to initialize stats before first tick when seeding runs.
        """
        today = self.clock.now().strftime("%Y-%m-%d")
        if self._current_day != today or self._daily_stats is None:
            self._current_day = today
            self._daily_stats = DailyStats(
                date=today,
                start_time=self.clock.now().isoformat(),
            )

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
            # Phase 5.5: Explicit REST fallback buckets
            self._daily_stats.total_rest_missing += tick_stats.rest_missing
            self._daily_stats.total_rest_unhealthy += tick_stats.rest_unhealthy
            self._daily_stats.total_rest_very_old += tick_stats.rest_very_old
            self._daily_stats.total_rest_reconcile += tick_stats.rest_reconcile
            # Phase 5.6: Cache seeding bucket
            self._daily_stats.total_rest_seed += tick_stats.rest_seed
            # Phase 5.5: WSS cache buckets
            self._daily_stats.total_wss_cache_fresh += tick_stats.wss_fresh
            self._daily_stats.total_wss_cache_quiet += tick_stats.wss_cache_quiet
            # Phase 5.5: Reconciliation aggregates (expanded from 5.4)
            self._daily_stats.total_reconciled += tick_stats.reconciled_count
            self._daily_stats.total_reconcile_ok += tick_stats.reconcile_ok_count
            self._daily_stats.total_reconcile_healed += tick_stats.reconcile_healed_count
            self._daily_stats.total_drift += tick_stats.drift_count
            # Phase 5.8: REST resilience metrics
            self._daily_stats.total_rest_retry_calls += tick_stats.rest_retry_calls
            self._daily_stats.total_rest_429 += tick_stats.rest_429_count
            self._daily_stats.total_rest_5xx += tick_stats.rest_5xx_count
            # Phase 6.1: Paper execution metrics
            self._daily_stats.paper_signals_found += tick_stats.paper_signals_found
            self._daily_stats.paper_trades_executed += tick_stats.paper_trades_executed
            self._daily_stats.paper_blocked_by_risk += tick_stats.paper_blocked_by_risk
            # Update PnL (latest snapshot, not cumulative)
            self._daily_stats.paper_pnl_total = tick_stats.paper_pnl_total
            # Phase 6.2: Explain mode aggregates
            if tick_stats.paper_explain_candidates > 0:
                self._daily_stats.paper_explain_ticks_with_candidates += 1
                top_edge = tick_stats.paper_explain_top_edge_bps
                if top_edge >= self.config.paper_exec_min_edge_bps:
                    self._daily_stats.paper_explain_ticks_above_min_edge += 1
                if top_edge > self._daily_stats.paper_explain_max_top_edge_bps:
                    self._daily_stats.paper_explain_max_top_edge_bps = top_edge
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
        """Export daily artifacts: CSV, JSON, markdown (Phase 5.7)."""
        date_str = stats.date
        export_dir = self.config.export_dir

        # Phase 5.6: Compute REST bucket total (invariant)
        rest_total = (
            stats.total_rest_missing
            + stats.total_rest_unhealthy
            + stats.total_rest_very_old
            + stats.total_rest_reconcile
            + stats.total_rest_seed
        )

        # Phase 5.6: Compute two coverage metrics
        # coverage_effective: excludes reconcile and seed REST calls from denominator
        wss_cache_total = stats.total_wss_cache_fresh + stats.total_wss_cache_quiet
        rest_fallback_real = (
            stats.total_rest_missing + stats.total_rest_unhealthy + stats.total_rest_very_old
        )
        effective_denom = wss_cache_total + rest_fallback_real
        coverage_effective_pct = (
            (wss_cache_total / effective_denom * 100) if effective_denom > 0 else 0.0
        )

        # coverage_raw: simple ratio of wss hits to total orderbooks
        wss_coverage_pct = (
            (stats.total_wss_hits / (stats.total_wss_hits + stats.total_rest_fallbacks) * 100)
            if (stats.total_wss_hits + stats.total_rest_fallbacks) > 0
            else 0.0
        )

        # Drift percentage
        drift_pct = (
            (stats.total_drift / stats.total_reconciled * 100)
            if stats.total_reconciled > 0
            else 0.0
        )
        heal_pct = (
            (stats.total_reconcile_healed / stats.total_reconciled * 100)
            if stats.total_reconciled > 0
            else 0.0
        )

        # Phase 5.7/5.9/6.0: Get seeding metrics if available
        seed_result = self._seed_result
        seed_attempted = seed_result.attempted if seed_result else 0
        seed_succeeded = seed_result.succeeded if seed_result else 0
        seed_duration = seed_result.duration_seconds if seed_result else 0.0
        seed_errors = seed_result.errors if seed_result else 0
        # Phase 5.9: Seed taxonomy fields
        seed_from_wss = seed_result.seed_from_wss if seed_result else 0
        seed_from_rest = seed_result.seed_from_rest if seed_result else 0
        seed_unseedable = seed_result.seed_unseedable if seed_result else 0
        seed_failed_unexpected = seed_result.seed_failed_unexpected if seed_result else 0
        seed_unseedable_kinds = seed_result.seed_unseedable_kinds if seed_result else {}
        # Phase 6.0: Skiplist metrics
        seed_skipped_skiplist = seed_result.seed_skipped_skiplist if seed_result else 0
        seed_candidates_total = seed_result.seed_candidates_total if seed_result else 0

        # Export coverage JSON (Phase 5.9 schema)
        coverage_path = export_dir / f"coverage_{date_str}.json"
        coverage_data = {
            "date": stats.date,
            "total_ticks": stats.total_ticks,
            "total_snapshots": stats.total_snapshots,
            "total_orderbooks": stats.total_orderbooks,
            # Legacy fields (kept for compat)
            "wss_hits": stats.total_wss_hits,
            "rest_fallbacks": stats.total_rest_fallbacks,
            "stale_count": stats.total_stale,
            "missing_count": stats.total_missing,
            "wss_coverage_pct": wss_coverage_pct,
            # Phase 5.5: Explicit buckets
            "wss_cache_fresh": stats.total_wss_cache_fresh,
            "wss_cache_quiet": stats.total_wss_cache_quiet,
            "rest_missing": stats.total_rest_missing,
            "rest_unhealthy": stats.total_rest_unhealthy,
            "rest_very_old": stats.total_rest_very_old,
            "rest_reconcile": stats.total_rest_reconcile,
            # Phase 5.6/5.7: Seeding and total
            "rest_seed": stats.total_rest_seed,
            "rest_total": rest_total,
            # Phase 5.7/5.9: Seeding details
            "seed_attempted": seed_attempted,
            "seed_succeeded": seed_succeeded,
            "seed_duration_seconds": seed_duration,
            "seed_errors": seed_errors,  # Legacy: unseedable + failed_unexpected
            # Phase 5.9: Seed taxonomy
            "seed_from_wss": seed_from_wss,
            "seed_from_rest": seed_from_rest,
            "seed_unseedable": seed_unseedable,
            "seed_failed_unexpected": seed_failed_unexpected,
            "seed_unseedable_kinds": seed_unseedable_kinds,
            # Phase 6.0: Skiplist metrics
            "seed_skipped_skiplist": seed_skipped_skiplist,
            "seed_candidates_total": seed_candidates_total,
            # Phase 5.6: Two coverage metrics
            "coverage_effective_pct": coverage_effective_pct,
            "coverage_raw_pct": wss_coverage_pct,
            # Phase 5.5: Reconciliation stats
            "reconciled_count": stats.total_reconciled,
            "reconcile_ok": stats.total_reconcile_ok,
            "reconcile_healed": stats.total_reconcile_healed,
            "drift_count": stats.total_drift,
            "drift_pct": drift_pct,
            "heal_pct": heal_pct,
            # Phase 5.8: REST resilience metrics
            "rest_retry_calls": stats.total_rest_retry_calls,
            "seed_retry_calls": stats.seed_retry_calls,
            "rest_429_count": stats.total_rest_429,
            "rest_5xx_count": stats.total_rest_5xx,
            # Phase 6.1: Paper execution metrics
            "paper_signals_found": stats.paper_signals_found,
            "paper_trades_executed": stats.paper_trades_executed,
            "paper_blocked_by_risk": stats.paper_blocked_by_risk,
            "paper_pnl_total": stats.paper_pnl_total,
            # Metadata
            "errors": stats.total_errors,
            "start_time": stats.start_time,
            "end_time": stats.end_time,
        }
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(coverage_data, f, indent=2)
        logger.info(f"Exported coverage to {coverage_path}")

        # Export tick history CSV (Phase 5.5 schema)
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
                        # Phase 5.5: Explicit REST buckets
                        "rest_missing",
                        "rest_unhealthy",
                        "rest_very_old",
                        "rest_reconcile",
                        "rest_seed",
                        # WSS stats
                        "wss_healthy",
                        "wss_cache_used",
                        "wss_cache_quiet",
                        "wss_fresh",
                        # Phase 5.5: Reconciliation
                        "reconciled_count",
                        "reconcile_ok",
                        "reconcile_healed",
                        "drift_count",
                        "drift_max_mid_bps",
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
                            "rest_missing": tick.rest_missing,
                            "rest_unhealthy": tick.rest_unhealthy,
                            "rest_very_old": tick.rest_very_old,
                            "rest_reconcile": tick.rest_reconcile,
                            "rest_seed": tick.rest_seed,
                            "wss_healthy": tick.wss_healthy,
                            "wss_cache_used": tick.wss_cache_used,
                            "wss_cache_quiet": tick.wss_cache_quiet,
                            "wss_fresh": tick.wss_fresh,
                            "reconciled_count": tick.reconciled_count,
                            "reconcile_ok": tick.reconcile_ok_count,
                            "reconcile_healed": tick.reconcile_healed_count,
                            "drift_count": tick.drift_count,
                            "drift_max_mid_bps": tick.drift_max_mid_bps,
                            "error": tick.error or "",
                        }
                    )
            logger.info(f"Exported ticks to {csv_path}")

        # Export markdown summary (Phase 5.7)
        md_path = export_dir / f"daemon_summary_{date_str}.md"

        # Phase 5.9/6.0: Cache seeding section with taxonomy
        seed_retries = seed_result.retry_calls if seed_result else 0
        seed_section = ""
        if seed_attempted > 0 or seed_skipped_skiplist > 0:
            # Build unseedable kinds breakdown
            kinds_breakdown = ""
            if seed_unseedable_kinds:
                kinds_lines = [f"  - {k}: {v}" for k, v in seed_unseedable_kinds.items() if v > 0]
                if kinds_lines:
                    kinds_breakdown = "\n" + "\n".join(kinds_lines)
            seed_section = f"""## Cache Seeding (Phase 6.0)
- **Candidates Total:** {seed_candidates_total:,}
- **Skipped (Skiplist):** {seed_skipped_skiplist:,}
- **Attempted:** {seed_attempted:,}
- **Succeeded:** {seed_succeeded:,}
  - From WSS: {seed_from_wss:,}
  - From REST: {seed_from_rest:,}
- **Duration:** {seed_duration:.1f}s
- **REST Calls:** {stats.total_rest_seed:,}
- **Retries:** {seed_retries:,}
- **Unseedable:** {seed_unseedable:,} (expected, non-retryable){kinds_breakdown}
- **Failed Unexpected:** {seed_failed_unexpected:,}
- **Legacy Errors:** {seed_errors:,}

"""

        # Phase 6.2: Paper Exec Diagnostics section (explain mode)
        explain_section = ""
        if self.config.paper_exec_explain and stats.paper_explain_ticks_with_candidates > 0:
            # Compute average top edge from tick history
            top_edges = [
                t.paper_explain_top_edge_bps
                for t in stats.tick_history
                if t.paper_explain_top_edge_bps > 0
            ]
            avg_top_edge = sum(top_edges) / len(top_edges) if top_edges else 0
            # Compute median
            if top_edges:
                sorted_edges = sorted(top_edges)
                mid_idx = len(sorted_edges) // 2
                if len(sorted_edges) % 2 == 0:
                    median_top_edge = (sorted_edges[mid_idx - 1] + sorted_edges[mid_idx]) / 2
                else:
                    median_top_edge = sorted_edges[mid_idx]
            else:
                median_top_edge = 0

            # Build rejection breakdown from accumulated counts
            rejection_lines = ""
            if stats.paper_explain_rejection_counts:
                sorted_rejections = sorted(
                    stats.paper_explain_rejection_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                rejection_lines = "\n".join(
                    f"  - {reason}: {count:,}"
                    for reason, count in sorted_rejections[:10]
                )
                rejection_lines = f"\n{rejection_lines}"

            min_edge = self.config.paper_exec_min_edge_bps
            explain_section = f"""## Paper Exec Diagnostics (Phase 6.2)
- **Ticks with Candidates:** {stats.paper_explain_ticks_with_candidates:,} / {stats.total_ticks}
- **Ticks with Edge >= {min_edge:.0f}bps:** {stats.paper_explain_ticks_above_min_edge:,}
- **Avg Top Edge:** {avg_top_edge:.1f} bps
- **Median Top Edge:** {median_top_edge:.1f} bps
- **Max Top Edge:** {stats.paper_explain_max_top_edge_bps:.1f} bps
- **Top Rejection Reasons:**{rejection_lines}

> **Calibration Tip:** If Ticks with Edge above min is low but Ticks with Candidates is high,
> consider lowering `--paper-exec-min-edge` (e.g., from 50 to 25 bps) for testing.

"""

        md_content = f"""# Daemon Summary - {date_str}

## Overview
- **Date:** {stats.date}
- **Start:** {stats.start_time}
- **End:** {stats.end_time}
- **Total Ticks:** {stats.total_ticks}

{seed_section}## Snapshot Coverage
- **Total Snapshots:** {stats.total_snapshots:,}
- **Total Orderbooks:** {stats.total_orderbooks:,}

## WSS Cache Statistics (Phase 5.5)
- **WSS Cache Fresh:** {stats.total_wss_cache_fresh:,}
- **WSS Cache Quiet:** {stats.total_wss_cache_quiet:,}
- **WSS Total Hits:** {stats.total_wss_hits:,}

## REST Fallback Breakdown (Phase 5.7)
- **REST Missing:** {stats.total_rest_missing:,} (no cached book)
- **REST Unhealthy:** {stats.total_rest_unhealthy:,} (WSS connection unhealthy)
- **REST Very Old:** {stats.total_rest_very_old:,} (cache > max_book_age)
- **REST Reconcile:** {stats.total_rest_reconcile:,} (reconciliation sampling)
- **REST Seed:** {stats.total_rest_seed:,} (initial cache seeding)
- **REST Total:** {rest_total:,} (sum of buckets)
- **Total REST Fallbacks (legacy):** {stats.total_rest_fallbacks:,}

## Coverage Metrics (Phase 5.6)
- **Coverage (Effective):** {coverage_effective_pct:.1f}% (excludes reconcile/seed REST calls)
- **Coverage (Raw):** {wss_coverage_pct:.1f}% (WSS hits / total)

## Reconciliation & Healing (Phase 5.5)
- **Tokens Reconciled:** {stats.total_reconciled:,}
- **OK (No Drift):** {stats.total_reconcile_ok:,}
- **Drift Detected:** {stats.total_drift:,}
- **Cache Healed:** {stats.total_reconcile_healed:,}
- **Drift Rate:** {drift_pct:.1f}%
- **Heal Rate:** {heal_pct:.1f}%

## REST Resilience (Phase 5.8)
- **Total Retries:** {stats.total_rest_retry_calls:,}
- **Seed Retries:** {stats.seed_retry_calls:,}
- **429 Responses:** {stats.total_rest_429:,}
- **5xx Responses:** {stats.total_rest_5xx:,}

## Paper Execution (Phase 6.1)
- **Signals Found:** {stats.paper_signals_found:,}
- **Trades Executed:** {stats.paper_trades_executed:,}
- **Blocked by Risk:** {stats.paper_blocked_by_risk:,}
- **Total PnL:** ${stats.paper_pnl_total:.2f}

{explain_section}## Errors
- **Total Errors:** {stats.total_errors}

---
*Generated by pmq ops daemon (Phase 6.2)*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Exported summary to {md_path}")

        # Phase 6.1: Export paper trades and positions
        if self.config.paper_exec_enabled:
            await self._export_paper_trades(date_str)
            await self._export_paper_positions(date_str)

    async def _export_paper_trades(self, date_str: str) -> int:
        """Export paper trades for a given date to gzip CSV.

        Phase 6.1: Exports all paper trades with atomic write.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Number of trades exported
        """
        export_dir = self.config.export_dir
        output_path = export_dir / f"paper_trades_{date_str}.csv.gz"

        # Get trades from DAO
        trades = self.dao.get_trades_for_export(limit=10000)
        if not trades:
            logger.info(f"No paper trades to export for {date_str}")
            return 0

        # Filter trades for this date (approximate - uses created_at)
        day_start = f"{date_str}T00:00:00"
        day_end = f"{date_str}T23:59:59"
        day_trades = [t for t in trades if day_start <= (t.get("created_at") or "9999") <= day_end]

        if not day_trades:
            logger.info(f"No paper trades to export for {date_str}")
            return 0

        # Atomic write: temp file in same directory, then rename
        temp_fd, temp_path = tempfile.mkstemp(suffix=".csv.gz.tmp", dir=str(export_dir))
        try:
            os.close(temp_fd)
            temp_path_obj = Path(temp_path)

            fieldnames = [
                "id",
                "strategy",
                "market_id",
                "side",
                "outcome",
                "price",
                "quantity",
                "notional",
                "created_at",
            ]

            with gzip.open(temp_path_obj, "wt", encoding="utf-8", newline="") as gz:
                writer = csv.DictWriter(gz, fieldnames=fieldnames)
                writer.writeheader()

                for trade in day_trades:
                    writer.writerow(
                        {
                            "id": trade.get("id"),
                            "strategy": trade.get("strategy"),
                            "market_id": trade.get("market_id"),
                            "side": trade.get("side"),
                            "outcome": trade.get("outcome"),
                            "price": trade.get("price"),
                            "quantity": trade.get("quantity"),
                            "notional": trade.get("notional"),
                            "created_at": trade.get("created_at"),
                        }
                    )

            # Atomic rename
            if sys.platform == "win32" and output_path.exists():
                output_path.unlink()
            temp_path_obj.rename(output_path)

            logger.info(f"Exported {len(day_trades):,} paper trades to {output_path}")
            return len(day_trades)

        except Exception:
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)
            raise

    async def _export_paper_positions(self, date_str: str) -> int:
        """Export paper positions snapshot for a given date to JSON.

        Phase 6.1: Exports positions snapshot with PnL summary.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Number of positions exported
        """
        export_dir = self.config.export_dir
        output_path = export_dir / f"paper_positions_{date_str}.json"

        # Get positions from DAO
        positions = self.dao.get_positions_for_export()

        # Calculate PnL summary
        total_realized = sum(p.get("realized_pnl", 0.0) for p in positions)
        total_yes_qty = sum(p.get("yes_quantity", 0.0) for p in positions)
        total_no_qty = sum(p.get("no_quantity", 0.0) for p in positions)

        positions_data = {
            "date": date_str,
            "snapshot_time": self.clock.now().isoformat(),
            "position_count": len(positions),
            "total_realized_pnl": total_realized,
            "total_yes_quantity": total_yes_qty,
            "total_no_quantity": total_no_qty,
            "positions": positions,
        }

        # Atomic write using temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json.tmp", dir=str(export_dir))
        try:
            os.close(temp_fd)
            temp_path_obj = Path(temp_path)

            with open(temp_path_obj, "w", encoding="utf-8") as f:
                json.dump(positions_data, f, indent=2, default=str)

            # Atomic rename
            if sys.platform == "win32" and output_path.exists():
                output_path.unlink()
            temp_path_obj.rename(output_path)

            logger.info(f"Exported {len(positions):,} paper positions to {output_path}")
            return len(positions)

        except Exception:
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)
            raise

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
        # Phase 5.7: Also export if seeding ran even without ticks
        current_day = self._current_day
        has_data = self._daily_stats and (
            self._daily_stats.total_ticks > 0 or self._daily_stats.total_rest_seed > 0
        )
        if has_data:
            assert self._daily_stats is not None  # mypy: validated by has_data
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

        # Phase 6.1: Export paper trades and positions on shutdown
        if self.config.paper_exec_enabled and current_day:
            try:
                await asyncio.wait_for(
                    self._export_paper_trades(current_day),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(f"Paper trades export for {current_day} timed out")
            except Exception as e:
                logger.warning(f"Paper trades export failed: {e}")

            try:
                await asyncio.wait_for(
                    self._export_paper_positions(current_day),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(f"Paper positions export for {current_day} timed out")
            except Exception as e:
                logger.warning(f"Paper positions export failed: {e}")

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
