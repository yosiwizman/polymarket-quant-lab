"""REST resilience utilities: rate limiting and retry with backoff.

Phase 5.8: Provides shared utilities for resilient REST calls across the daemon:
- Token bucket rate limiter for global request rate control
- Async retry wrapper with exponential backoff + jitter for transient failures

Retryable errors:
- HTTP 429 (rate limit)
- HTTP 5xx (server errors)
- Network/timeout exceptions

Non-retryable errors:
- HTTP 4xx (except 429) - client errors, fail fast
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import httpx

from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.ops.daemon import ClockProtocol

logger = get_logger("ops.rest_resilience")

T = TypeVar("T")


@dataclass
class RestResilienceConfig:
    """Configuration for REST resilience (Phase 5.8)."""

    rps: float = 8.0  # Requests per second (tokens refilled per second)
    burst: int = 8  # Maximum burst capacity (token bucket size)
    max_retries: int = 3  # Max retry attempts (not counting initial)
    backoff_base: float = 0.25  # Base backoff delay in seconds
    backoff_max: float = 3.0  # Maximum backoff delay in seconds
    jitter_factor: float = 0.25  # Random jitter as fraction of delay


@dataclass
class RestCallStats:
    """Statistics from a REST call attempt (Phase 5.8)."""

    success: bool = False
    retry_count: int = 0  # Number of retry attempts (not counting initial)
    http_429_count: int = 0  # Count of 429 responses seen
    http_5xx_count: int = 0  # Count of 5xx responses seen
    final_error: str | None = None  # Error message if failed


@dataclass
class RestResilienceMetrics:
    """Aggregate metrics for REST resilience (Phase 5.8)."""

    total_calls: int = 0
    total_retries: int = 0
    total_429: int = 0
    total_5xx: int = 0
    total_errors: int = 0

    def add_stats(self, stats: RestCallStats) -> None:
        """Add stats from a single call."""
        self.total_calls += 1
        self.total_retries += stats.retry_count
        self.total_429 += stats.http_429_count
        self.total_5xx += stats.http_5xx_count
        if not stats.success:
            self.total_errors += 1


class TokenBucketRateLimiter:
    """Async token bucket rate limiter.

    Provides deterministic rate limiting with configurable:
    - rps: tokens refilled per second
    - burst: maximum bucket capacity

    Thread-safe via asyncio.Lock. Testable via injectable clock.
    """

    def __init__(
        self,
        rps: float = 8.0,
        burst: int = 8,
        clock: ClockProtocol | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            rps: Requests per second (token refill rate)
            burst: Maximum tokens (bucket capacity)
            clock: Time provider for testability
        """
        self._rps = rps
        self._burst = burst
        self._tokens = float(burst)  # Start full
        self._last_refill: float | None = None  # Use None as sentinel
        self._lock = asyncio.Lock()
        self._clock = clock

    def _get_time(self) -> float:
        """Get current monotonic time."""
        if self._clock:
            return self._clock.monotonic()
        return time.monotonic()

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        if self._last_refill is None:
            self._last_refill = now
            return

        elapsed = now - self._last_refill
        refill = elapsed * self._rps
        self._tokens = min(self._burst, self._tokens + refill)
        self._last_refill = now

    async def acquire(
        self, sleep_fn: Callable[[float], Coroutine[Any, Any, None]] | None = None
    ) -> float:
        """Acquire a token, waiting if necessary.

        Args:
            sleep_fn: Optional async sleep function for testability

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = self._get_time()
            self._refill(now)

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # Calculate wait time for 1 token
            deficit = 1.0 - self._tokens
            wait_time = deficit / self._rps

            # Wait
            if sleep_fn:
                await sleep_fn(wait_time)
            else:
                await asyncio.sleep(wait_time)

            # Refill after wait
            now = self._get_time()
            self._refill(now)
            self._tokens -= 1.0

            return wait_time

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (for testing)."""
        now = self._get_time()
        self._refill(now)
        return self._tokens


def is_retryable_error(error: Exception) -> tuple[bool, str]:
    """Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        Tuple of (is_retryable, error_category)
        Categories: "429", "5xx", "timeout", "network", "4xx", "unknown"
    """
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 429:
            return True, "429"
        if 500 <= status < 600:
            return True, "5xx"
        if 400 <= status < 500:
            return False, "4xx"
        return False, f"http_{status}"

    if isinstance(error, httpx.TimeoutException | asyncio.TimeoutError):
        return True, "timeout"

    if isinstance(error, httpx.RequestError):
        return True, "network"

    return False, "unknown"


def compute_backoff(
    attempt: int,
    base: float = 0.25,
    max_delay: float = 3.0,
    jitter_factor: float = 0.25,
) -> float:
    """Compute exponential backoff with jitter.

    Args:
        attempt: Retry attempt number (0-indexed)
        base: Base delay in seconds
        max_delay: Maximum delay cap
        jitter_factor: Random jitter as fraction of delay

    Returns:
        Delay in seconds
    """
    # Exponential: base * 2^attempt
    delay = base * (2**attempt)
    delay = min(delay, max_delay)

    # Add jitter: ±jitter_factor * delay
    jitter = random.uniform(-jitter_factor, jitter_factor) * delay
    delay = max(0.0, delay + jitter)

    return float(delay)


async def retry_rest_call(  # noqa: UP047
    call_fn: Callable[[], T],
    config: RestResilienceConfig,
    rate_limiter: TokenBucketRateLimiter | None = None,
    sleep_fn: Callable[[float], Coroutine[Any, Any, None]] | None = None,
) -> tuple[T | None, RestCallStats]:
    """Execute a REST call with retry and rate limiting.

    Args:
        call_fn: Sync callable that performs the REST call
        config: Resilience configuration
        rate_limiter: Optional rate limiter (skipped if None)
        sleep_fn: Optional async sleep function for testability

    Returns:
        Tuple of (result or None, RestCallStats)
    """
    stats = RestCallStats()
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):  # +1 for initial attempt
        # Rate limit
        if rate_limiter:
            await rate_limiter.acquire(sleep_fn)

        try:
            # Run sync call in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call_fn)
            stats.success = True
            return result, stats

        except Exception as e:
            last_error = e
            is_retry, category = is_retryable_error(e)

            # Track error categories
            if category == "429":
                stats.http_429_count += 1
            elif category == "5xx":
                stats.http_5xx_count += 1

            # Non-retryable: fail fast
            if not is_retry:
                stats.final_error = f"{category}: {e}"
                logger.debug(f"Non-retryable error (attempt {attempt + 1}): {category}")
                return None, stats

            # Last attempt: no more retries
            if attempt >= config.max_retries:
                stats.final_error = f"{category}: {e}"
                logger.debug(f"Max retries ({config.max_retries}) exceeded: {category}")
                return None, stats

            # Compute backoff and wait
            stats.retry_count += 1
            delay = compute_backoff(
                attempt,
                base=config.backoff_base,
                max_delay=config.backoff_max,
                jitter_factor=config.jitter_factor,
            )

            logger.debug(
                f"Retryable error (attempt {attempt + 1}): {category}, retrying in {delay:.2f}s"
            )

            if sleep_fn:
                await sleep_fn(delay)
            else:
                await asyncio.sleep(delay)

    # Should not reach here, but handle gracefully
    stats.final_error = f"unexpected: {last_error}"
    return None, stats


# Global metrics instance for daemon-wide tracking
_global_metrics: RestResilienceMetrics = field(default_factory=RestResilienceMetrics)


def get_global_metrics() -> RestResilienceMetrics:
    """Get the global REST resilience metrics instance."""
    global _global_metrics
    if not isinstance(_global_metrics, RestResilienceMetrics):
        _global_metrics = RestResilienceMetrics()
    return _global_metrics


def reset_global_metrics() -> None:
    """Reset global metrics (for testing)."""
    global _global_metrics
    _global_metrics = RestResilienceMetrics()
