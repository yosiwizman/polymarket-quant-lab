"""Tests for REST resilience utilities (Phase 5.8).

Tests cover:
- Token bucket rate limiter behavior
- Retry with exponential backoff
- Error classification (retryable vs non-retryable)
- Integration with deterministic clock/sleep
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import pytest

from pmq.ops.rest_resilience import (
    RestCallStats,
    RestResilienceConfig,
    TokenBucketRateLimiter,
    compute_backoff,
    is_retryable_error,
    retry_rest_call,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class FakeClock:
    """Fake clock for deterministic testing."""

    _monotonic: float = 0.0

    def now(self) -> datetime:
        """Return current time (not used in rate limiter)."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return monotonic time."""
        return self._monotonic

    def advance(self, seconds: float) -> None:
        """Advance monotonic time."""
        self._monotonic += seconds


class FakeSleep:
    """Fake async sleep that tracks calls and advances clock."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.total_sleep = 0.0
        self.call_count = 0
        self.sleep_durations: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """Record sleep call and advance clock."""
        self.total_sleep += seconds
        self.call_count += 1
        self.sleep_durations.append(seconds)
        if self.clock:
            self.clock.advance(seconds)
        await asyncio.sleep(0)  # Yield to event loop


class FakeHTTPStatusError(httpx.HTTPStatusError):
    """Fake HTTP status error for testing."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        # Create a minimal response object
        request = httpx.Request("GET", "http://test.com")
        response = httpx.Response(status_code, request=request)
        super().__init__(
            message=f"HTTP {status_code}",
            request=request,
            response=response,
        )


# =============================================================================
# is_retryable_error Tests
# =============================================================================


class TestIsRetryableError:
    """Tests for error classification."""

    def test_429_is_retryable(self) -> None:
        """HTTP 429 should be retryable."""
        error = FakeHTTPStatusError(429)
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "429"

    def test_500_is_retryable(self) -> None:
        """HTTP 500 should be retryable."""
        error = FakeHTTPStatusError(500)
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "5xx"

    def test_502_is_retryable(self) -> None:
        """HTTP 502 should be retryable."""
        error = FakeHTTPStatusError(502)
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "5xx"

    def test_503_is_retryable(self) -> None:
        """HTTP 503 should be retryable."""
        error = FakeHTTPStatusError(503)
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "5xx"

    def test_400_not_retryable(self) -> None:
        """HTTP 400 should NOT be retryable."""
        error = FakeHTTPStatusError(400)
        is_retry, category = is_retryable_error(error)
        assert is_retry is False
        assert category == "4xx"

    def test_404_not_retryable(self) -> None:
        """HTTP 404 should NOT be retryable."""
        error = FakeHTTPStatusError(404)
        is_retry, category = is_retryable_error(error)
        assert is_retry is False
        assert category == "4xx"

    def test_timeout_is_retryable(self) -> None:
        """Timeout exceptions should be retryable."""
        error = httpx.TimeoutException("timeout")
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "timeout"

    def test_asyncio_timeout_is_retryable(self) -> None:
        """asyncio.TimeoutError should be retryable."""
        error = TimeoutError()
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "timeout"

    def test_network_error_is_retryable(self) -> None:
        """Network errors should be retryable."""
        error = httpx.ConnectError("connection refused")
        is_retry, category = is_retryable_error(error)
        assert is_retry is True
        assert category == "network"

    def test_unknown_error_not_retryable(self) -> None:
        """Unknown exceptions should NOT be retryable."""
        error = ValueError("unknown")
        is_retry, category = is_retryable_error(error)
        assert is_retry is False
        assert category == "unknown"


# =============================================================================
# compute_backoff Tests
# =============================================================================


class TestComputeBackoff:
    """Tests for backoff calculation."""

    def test_base_backoff_attempt_0(self) -> None:
        """First attempt should use base backoff."""
        # With jitter_factor=0, should be exactly base
        delay = compute_backoff(attempt=0, base=0.25, max_delay=3.0, jitter_factor=0.0)
        assert delay == 0.25

    def test_exponential_growth(self) -> None:
        """Backoff should grow exponentially."""
        # With jitter_factor=0
        delay_0 = compute_backoff(attempt=0, base=0.25, max_delay=10.0, jitter_factor=0.0)
        delay_1 = compute_backoff(attempt=1, base=0.25, max_delay=10.0, jitter_factor=0.0)
        delay_2 = compute_backoff(attempt=2, base=0.25, max_delay=10.0, jitter_factor=0.0)

        assert delay_0 == 0.25  # 0.25 * 2^0 = 0.25
        assert delay_1 == 0.5  # 0.25 * 2^1 = 0.5
        assert delay_2 == 1.0  # 0.25 * 2^2 = 1.0

    def test_max_delay_cap(self) -> None:
        """Backoff should be capped at max_delay."""
        delay = compute_backoff(attempt=10, base=0.25, max_delay=3.0, jitter_factor=0.0)
        assert delay == 3.0

    def test_jitter_range(self) -> None:
        """Jitter should add variability within expected range."""
        delays = [
            compute_backoff(attempt=0, base=1.0, max_delay=10.0, jitter_factor=0.25)
            for _ in range(100)
        ]
        # With base=1.0 and jitter_factor=0.25, delays should be in [0.75, 1.25]
        assert all(0.75 <= d <= 1.25 for d in delays)
        # Should have some variation (not all the same)
        assert len(set(delays)) > 1


# =============================================================================
# TokenBucketRateLimiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_immediate_acquire_with_full_bucket(self) -> None:
        """Should acquire immediately when bucket is full."""
        clock = FakeClock()
        limiter = TokenBucketRateLimiter(rps=1.0, burst=1, clock=clock)

        wait_time = await limiter.acquire()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_wait_when_bucket_empty(self) -> None:
        """Should wait when bucket is empty."""
        clock = FakeClock()
        sleep_fn = FakeSleep(clock)
        limiter = TokenBucketRateLimiter(rps=1.0, burst=1, clock=clock)

        # First acquire: immediate
        await limiter.acquire(sleep_fn)
        assert sleep_fn.call_count == 0

        # Second acquire: must wait for token refill
        wait_time = await limiter.acquire(sleep_fn)
        assert wait_time > 0.0
        assert sleep_fn.call_count == 1
        # With rps=1, wait should be ~1 second
        assert 0.9 <= sleep_fn.total_sleep <= 1.1

    @pytest.mark.asyncio
    async def test_burst_capacity(self) -> None:
        """Should allow burst requests up to capacity."""
        clock = FakeClock()
        sleep_fn = FakeSleep(clock)
        limiter = TokenBucketRateLimiter(rps=1.0, burst=3, clock=clock)

        # Should allow 3 immediate requests
        for _ in range(3):
            await limiter.acquire(sleep_fn)

        assert sleep_fn.call_count == 0

        # 4th request should wait
        await limiter.acquire(sleep_fn)
        assert sleep_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self) -> None:
        """Tokens should refill over time."""
        clock = FakeClock()
        sleep_fn = FakeSleep(clock)
        limiter = TokenBucketRateLimiter(rps=1.0, burst=1, clock=clock)

        # Use the token
        await limiter.acquire(sleep_fn)

        # Advance time by 1 second (should refill 1 token)
        clock.advance(1.0)

        # Should be able to acquire immediately
        wait_time = await limiter.acquire(sleep_fn)
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_available_tokens_property(self) -> None:
        """available_tokens should reflect current state."""
        clock = FakeClock()
        limiter = TokenBucketRateLimiter(rps=1.0, burst=2, clock=clock)

        # Initial: full bucket
        assert limiter.available_tokens == 2.0

        # After acquire
        await limiter.acquire()
        assert limiter.available_tokens == 1.0


# =============================================================================
# retry_rest_call Tests
# =============================================================================


class TestRetryRestCall:
    """Tests for retry wrapper."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Should return result without retries on success."""

        def success_fn() -> str:
            return "success"

        config = RestResilienceConfig(max_retries=3)
        result, stats = await retry_rest_call(success_fn, config)

        assert result == "success"
        assert stats.success is True
        assert stats.retry_count == 0
        assert stats.http_429_count == 0
        assert stats.http_5xx_count == 0
        assert stats.final_error is None

    @pytest.mark.asyncio
    async def test_retry_on_429_then_success(self) -> None:
        """Should retry on 429 and succeed on retry."""
        call_count = 0

        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FakeHTTPStatusError(429)
            return "success"

        config = RestResilienceConfig(max_retries=3, backoff_base=0.1, backoff_max=0.5)
        sleep_fn = FakeSleep()
        result, stats = await retry_rest_call(fail_then_succeed, config, sleep_fn=sleep_fn)

        assert result == "success"
        assert stats.success is True
        assert stats.retry_count == 1
        assert stats.http_429_count == 1
        assert call_count == 2
        assert sleep_fn.call_count >= 1  # At least one backoff sleep

    @pytest.mark.asyncio
    async def test_retry_on_5xx_then_success(self) -> None:
        """Should retry on 5xx and succeed on retry."""
        call_count = 0

        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise FakeHTTPStatusError(502)
            return "success"

        config = RestResilienceConfig(max_retries=3, backoff_base=0.1, backoff_max=0.5)
        sleep_fn = FakeSleep()
        result, stats = await retry_rest_call(fail_then_succeed, config, sleep_fn=sleep_fn)

        assert result == "success"
        assert stats.success is True
        assert stats.retry_count == 2
        assert stats.http_5xx_count == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self) -> None:
        """Should NOT retry on 4xx (except 429) - fail fast."""
        call_count = 0

        def fail_404() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeHTTPStatusError(404)

        config = RestResilienceConfig(max_retries=3, backoff_base=0.1)
        sleep_fn = FakeSleep()
        result, stats = await retry_rest_call(fail_404, config, sleep_fn=sleep_fn)

        assert result is None
        assert stats.success is False
        assert stats.retry_count == 0
        assert call_count == 1  # Only one attempt
        assert sleep_fn.call_count == 0  # No backoff sleep

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        """Should fail after exhausting retries."""
        call_count = 0

        def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeHTTPStatusError(500)

        config = RestResilienceConfig(max_retries=2, backoff_base=0.1, backoff_max=0.5)
        sleep_fn = FakeSleep()
        result, stats = await retry_rest_call(always_fail, config, sleep_fn=sleep_fn)

        assert result is None
        assert stats.success is False
        assert stats.retry_count == 2  # Two retries after initial attempt
        assert stats.http_5xx_count == 3  # Initial + 2 retries
        assert call_count == 3  # 1 initial + 2 retries
        assert "5xx" in (stats.final_error or "")

    @pytest.mark.asyncio
    async def test_with_rate_limiter(self) -> None:
        """Should respect rate limiter."""
        clock = FakeClock()
        sleep_fn = FakeSleep(clock)
        limiter = TokenBucketRateLimiter(rps=1.0, burst=1, clock=clock)

        call_count = 0

        def success_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        config = RestResilienceConfig(max_retries=0)

        # First call: immediate
        result1, _ = await retry_rest_call(
            success_fn, config, rate_limiter=limiter, sleep_fn=sleep_fn
        )
        assert result1 == "success"
        assert sleep_fn.call_count == 0

        # Second call: should wait for rate limiter
        result2, _ = await retry_rest_call(
            success_fn, config, rate_limiter=limiter, sleep_fn=sleep_fn
        )
        assert result2 == "success"
        assert sleep_fn.call_count == 1  # Rate limiter caused sleep

    @pytest.mark.asyncio
    async def test_timeout_error_retry(self) -> None:
        """Should retry on timeout exceptions."""
        call_count = 0

        def timeout_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timeout")
            return "success"

        config = RestResilienceConfig(max_retries=3, backoff_base=0.1)
        sleep_fn = FakeSleep()
        result, stats = await retry_rest_call(timeout_then_succeed, config, sleep_fn=sleep_fn)

        assert result == "success"
        assert stats.success is True
        assert stats.retry_count == 1


# =============================================================================
# RestCallStats Tests
# =============================================================================


class TestRestCallStats:
    """Tests for RestCallStats dataclass."""

    def test_default_values(self) -> None:
        """RestCallStats should initialize with defaults."""
        stats = RestCallStats()
        assert stats.success is False
        assert stats.retry_count == 0
        assert stats.http_429_count == 0
        assert stats.http_5xx_count == 0
        assert stats.final_error is None

    def test_custom_values(self) -> None:
        """RestCallStats should accept custom values."""
        stats = RestCallStats(
            success=True,
            retry_count=2,
            http_429_count=1,
            http_5xx_count=1,
            final_error=None,
        )
        assert stats.success is True
        assert stats.retry_count == 2
        assert stats.http_429_count == 1
        assert stats.http_5xx_count == 1


# =============================================================================
# RestResilienceConfig Tests
# =============================================================================


class TestRestResilienceConfig:
    """Tests for RestResilienceConfig dataclass."""

    def test_default_values(self) -> None:
        """RestResilienceConfig should have sensible defaults."""
        config = RestResilienceConfig()
        assert config.rps == 8.0
        assert config.burst == 8
        assert config.max_retries == 3
        assert config.backoff_base == 0.25
        assert config.backoff_max == 3.0
        assert config.jitter_factor == 0.25

    def test_custom_values(self) -> None:
        """RestResilienceConfig should accept custom values."""
        config = RestResilienceConfig(
            rps=10.0,
            burst=10,
            max_retries=5,
            backoff_base=0.5,
            backoff_max=5.0,
        )
        assert config.rps == 10.0
        assert config.burst == 10
        assert config.max_retries == 5
