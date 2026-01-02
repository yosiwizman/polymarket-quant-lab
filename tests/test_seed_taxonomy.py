"""Tests for seed taxonomy and WSS-first seeding (Phase 5.9).

Tests cover:
- SeedOutcomeKind classification
- classify_orderbook_result behavior
- SeedSkiplist load/save/expiry
- classify_http_error for various status codes
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from pmq.ops.seed_taxonomy import (
    SeedOutcome,
    SeedOutcomeKind,
    SeedOutcomeStats,
    SeedSkiplist,
    classify_http_error,
    classify_orderbook_result,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class FakeHTTPStatusError(httpx.HTTPStatusError):
    """Fake HTTP status error for testing."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        request = httpx.Request("GET", "http://test.com")
        response = httpx.Response(status_code, request=request)
        super().__init__(
            message=f"HTTP {status_code}",
            request=request,
            response=response,
        )


def make_mock_orderbook(
    token_id: str = "test_token",
    has_valid_book: bool = True,
    error: str | None = None,
) -> MagicMock:
    """Create a mock OrderBookData for testing."""
    mock = MagicMock()
    mock.token_id = token_id
    mock.has_valid_book = has_valid_book
    mock.error = error
    return mock


# =============================================================================
# SeedOutcomeKind Tests
# =============================================================================


class TestSeedOutcomeKind:
    """Tests for SeedOutcomeKind enum."""

    def test_ok_is_success(self) -> None:
        """OK should be marked as success."""
        assert SeedOutcomeKind.OK.is_success is True
        assert SeedOutcomeKind.OK.is_unseedable is False

    def test_unseedable_empty_book(self) -> None:
        """UNSEEDABLE_EMPTY_BOOK should be unseedable, not success."""
        kind = SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK
        assert kind.is_success is False
        assert kind.is_unseedable is True

    def test_unseedable_not_found(self) -> None:
        """UNSEEDABLE_NOT_FOUND should be unseedable, not success."""
        kind = SeedOutcomeKind.UNSEEDABLE_NOT_FOUND
        assert kind.is_success is False
        assert kind.is_unseedable is True

    def test_unseedable_bad_request(self) -> None:
        """UNSEEDABLE_BAD_REQUEST should be unseedable, not success."""
        kind = SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST
        assert kind.is_success is False
        assert kind.is_unseedable is True

    def test_unseedable_other_4xx(self) -> None:
        """UNSEEDABLE_OTHER_4XX should be unseedable, not success."""
        kind = SeedOutcomeKind.UNSEEDABLE_OTHER_4XX
        assert kind.is_success is False
        assert kind.is_unseedable is True

    def test_failed_unexpected(self) -> None:
        """FAILED_UNEXPECTED should not be success or unseedable."""
        kind = SeedOutcomeKind.FAILED_UNEXPECTED
        assert kind.is_success is False
        assert kind.is_unseedable is False


# =============================================================================
# classify_http_error Tests
# =============================================================================


class TestClassifyHttpError:
    """Tests for HTTP error classification."""

    def test_404_is_not_found(self) -> None:
        """HTTP 404 should be classified as UNSEEDABLE_NOT_FOUND."""
        kind = classify_http_error(404)
        assert kind == SeedOutcomeKind.UNSEEDABLE_NOT_FOUND

    def test_400_is_bad_request(self) -> None:
        """HTTP 400 should be classified as UNSEEDABLE_BAD_REQUEST."""
        kind = classify_http_error(400)
        assert kind == SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST

    def test_401_is_other_4xx(self) -> None:
        """HTTP 401 should be classified as UNSEEDABLE_OTHER_4XX."""
        kind = classify_http_error(401)
        assert kind == SeedOutcomeKind.UNSEEDABLE_OTHER_4XX

    def test_403_is_other_4xx(self) -> None:
        """HTTP 403 should be classified as UNSEEDABLE_OTHER_4XX."""
        kind = classify_http_error(403)
        assert kind == SeedOutcomeKind.UNSEEDABLE_OTHER_4XX

    def test_422_is_other_4xx(self) -> None:
        """HTTP 422 should be classified as UNSEEDABLE_OTHER_4XX."""
        kind = classify_http_error(422)
        assert kind == SeedOutcomeKind.UNSEEDABLE_OTHER_4XX

    def test_429_is_failed_unexpected(self) -> None:
        """HTTP 429 should be classified as FAILED_UNEXPECTED (retryable in retry wrapper)."""
        kind = classify_http_error(429)
        assert kind == SeedOutcomeKind.FAILED_UNEXPECTED

    def test_500_is_failed_unexpected(self) -> None:
        """HTTP 500 should be classified as FAILED_UNEXPECTED (retryable in retry wrapper)."""
        kind = classify_http_error(500)
        assert kind == SeedOutcomeKind.FAILED_UNEXPECTED

    def test_502_is_failed_unexpected(self) -> None:
        """HTTP 502 should be classified as FAILED_UNEXPECTED."""
        kind = classify_http_error(502)
        assert kind == SeedOutcomeKind.FAILED_UNEXPECTED


# =============================================================================
# classify_orderbook_result Tests
# =============================================================================


class TestClassifyOrderbookResult:
    """Tests for orderbook result classification."""

    def test_success_with_valid_book(self) -> None:
        """Valid orderbook should be classified as OK."""
        ob = make_mock_orderbook(has_valid_book=True, error=None)
        outcome = classify_orderbook_result(ob)
        assert outcome.kind == SeedOutcomeKind.OK
        assert outcome.is_success is True

    def test_empty_book(self) -> None:
        """Empty book (no bids/asks) should be classified as UNSEEDABLE_EMPTY_BOOK."""
        ob = make_mock_orderbook(has_valid_book=False, error=None)
        outcome = classify_orderbook_result(ob)
        assert outcome.kind == SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK
        assert outcome.is_unseedable is True

    def test_error_404(self) -> None:
        """HTTP 404 error should be classified as UNSEEDABLE_NOT_FOUND."""
        error = FakeHTTPStatusError(404)
        outcome = classify_orderbook_result(None, error=error)
        assert outcome.kind == SeedOutcomeKind.UNSEEDABLE_NOT_FOUND

    def test_error_400(self) -> None:
        """HTTP 400 error should be classified as UNSEEDABLE_BAD_REQUEST."""
        error = FakeHTTPStatusError(400)
        outcome = classify_orderbook_result(None, error=error)
        assert outcome.kind == SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST

    def test_error_500(self) -> None:
        """HTTP 500 error should be classified as FAILED_UNEXPECTED."""
        error = FakeHTTPStatusError(500)
        outcome = classify_orderbook_result(None, error=error)
        assert outcome.kind == SeedOutcomeKind.FAILED_UNEXPECTED

    def test_error_in_orderbook_404(self) -> None:
        """OrderBook with 404 error field should be UNSEEDABLE_NOT_FOUND."""
        ob = make_mock_orderbook(error="http_404: not_found")
        outcome = classify_orderbook_result(ob)
        assert outcome.kind == SeedOutcomeKind.UNSEEDABLE_NOT_FOUND

    def test_error_in_orderbook_400(self) -> None:
        """OrderBook with 400 error field should be UNSEEDABLE_BAD_REQUEST."""
        ob = make_mock_orderbook(error="http_400: bad_request")
        outcome = classify_orderbook_result(ob)
        assert outcome.kind == SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST

    def test_none_orderbook(self) -> None:
        """None orderbook with no error should be FAILED_UNEXPECTED."""
        outcome = classify_orderbook_result(None)
        assert outcome.kind == SeedOutcomeKind.FAILED_UNEXPECTED

    def test_network_error(self) -> None:
        """Network errors should be FAILED_UNEXPECTED."""
        error = httpx.ConnectError("connection refused")
        outcome = classify_orderbook_result(None, error=error)
        assert outcome.kind == SeedOutcomeKind.FAILED_UNEXPECTED


# =============================================================================
# SeedOutcomeStats Tests
# =============================================================================


class TestSeedOutcomeStats:
    """Tests for outcome statistics aggregation."""

    def test_empty_stats(self) -> None:
        """Empty stats should have zero values."""
        stats = SeedOutcomeStats()
        assert stats.total == 0
        assert stats.ok == 0
        assert stats.unseedable == 0
        assert stats.failed_unexpected == 0
        assert stats.kinds == {}

    def test_add_ok(self) -> None:
        """Adding OK outcome should increment ok count."""
        stats = SeedOutcomeStats()
        outcome = SeedOutcome(token_id="test", kind=SeedOutcomeKind.OK)
        stats.add(outcome)
        assert stats.total == 1
        assert stats.ok == 1
        assert stats.unseedable == 0
        assert stats.failed_unexpected == 0

    def test_add_unseedable(self) -> None:
        """Adding unseedable outcome should increment unseedable count."""
        stats = SeedOutcomeStats()
        outcome = SeedOutcome(token_id="test", kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND)
        stats.add(outcome)
        assert stats.total == 1
        assert stats.ok == 0
        assert stats.unseedable == 1
        assert stats.failed_unexpected == 0

    def test_add_failed_unexpected(self) -> None:
        """Adding failed_unexpected should increment that count."""
        stats = SeedOutcomeStats()
        outcome = SeedOutcome(token_id="test", kind=SeedOutcomeKind.FAILED_UNEXPECTED)
        stats.add(outcome)
        assert stats.total == 1
        assert stats.ok == 0
        assert stats.unseedable == 0
        assert stats.failed_unexpected == 1

    def test_kinds_breakdown(self) -> None:
        """Should track breakdown by specific kind."""
        stats = SeedOutcomeStats()
        stats.add(SeedOutcome(token_id="t1", kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND))
        stats.add(SeedOutcome(token_id="t2", kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND))
        stats.add(SeedOutcome(token_id="t3", kind=SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK))
        assert stats.kinds["unseedable_not_found"] == 2
        assert stats.kinds["unseedable_empty_book"] == 1


# =============================================================================
# SeedSkiplist Tests
# =============================================================================


class TestSeedSkiplist:
    """Tests for skiplist functionality."""

    def test_empty_skiplist(self) -> None:
        """Empty skiplist should have no entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            skiplist.load()
            assert len(skiplist) == 0
            assert skiplist.should_skip("any_token") is False

    def test_add_and_skip(self) -> None:
        """Adding an entry should make it skippable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            outcome = SeedOutcome(
                token_id="test_token",
                kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND,
            )
            skiplist.add(outcome)
            assert skiplist.should_skip("test_token") is True
            assert len(skiplist) == 1

    def test_only_unseedable_added(self) -> None:
        """Only unseedable outcomes should be added to skiplist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            # OK outcome should NOT be added
            ok_outcome = SeedOutcome(token_id="ok_token", kind=SeedOutcomeKind.OK)
            skiplist.add(ok_outcome)
            assert skiplist.should_skip("ok_token") is False
            # FAILED_UNEXPECTED should NOT be added
            failed_outcome = SeedOutcome(
                token_id="failed_token",
                kind=SeedOutcomeKind.FAILED_UNEXPECTED,
            )
            skiplist.add(failed_outcome)
            assert skiplist.should_skip("failed_token") is False

    def test_save_and_load(self) -> None:
        """Skiplist should persist and reload correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"

            # Create and save
            skiplist1 = SeedSkiplist(path)
            outcome = SeedOutcome(
                token_id="persist_token",
                kind=SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK,
            )
            skiplist1.add(outcome)
            skiplist1.save()

            # Reload in new instance
            skiplist2 = SeedSkiplist(path)
            skiplist2.load()
            assert skiplist2.should_skip("persist_token") is True
            assert len(skiplist2) == 1

    def test_dirty_flag(self) -> None:
        """Dirty flag should be set when entries are added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            assert skiplist.dirty is False
            outcome = SeedOutcome(
                token_id="test_token",
                kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND,
            )
            skiplist.add(outcome)
            assert skiplist.dirty is True

    def test_expiry(self) -> None:
        """Expired entries should not be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"

            # Create expired entry manually
            now = datetime.now(UTC)
            past = now - timedelta(hours=48)  # 48 hours ago
            data = {
                "version": 1,
                "entries": [
                    {
                        "token_id": "expired_token",
                        "kind": "unseedable_not_found",
                        "added_at": past.isoformat(),
                        "expires_at": (past + timedelta(hours=24)).isoformat(),  # Expired
                    }
                ],
            }
            path.write_text(json.dumps(data))

            skiplist = SeedSkiplist(path)
            skiplist.load()
            # Should not skip expired entry
            assert skiplist.should_skip("expired_token") is False

    def test_add_token_method(self) -> None:
        """add_token method should work with token ID and kind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            skiplist.add_token("direct_token", SeedOutcomeKind.UNSEEDABLE_NOT_FOUND)
            assert skiplist.should_skip("direct_token") is True
            assert skiplist.dirty is True

    def test_get_skipped_tokens(self) -> None:
        """get_skipped_tokens should return all non-expired tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skiplist.json"
            skiplist = SeedSkiplist(path)
            skiplist.add(SeedOutcome("t1", SeedOutcomeKind.UNSEEDABLE_NOT_FOUND))
            skiplist.add(SeedOutcome("t2", SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK))
            tokens = skiplist.get_skipped_tokens()
            assert tokens == {"t1", "t2"}
