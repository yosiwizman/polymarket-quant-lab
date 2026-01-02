"""Seed outcome taxonomy for classifying cache seeding results.

Phase 5.9: Provides explicit classification of seed outcomes to distinguish between:
- Expected unseedable markets (empty books, not found, closed)
- Unexpected failures (network errors, parsing issues)

This makes seed_errors interpretable and reduces false alarm from "expected" failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.markets.orderbook import OrderBookData

logger = get_logger("ops.seed_taxonomy")


class SeedOutcomeKind(str, Enum):
    """Classification of seed operation outcomes.

    Categories:
    - ok: Successfully seeded with valid book data
    - unseedable_*: Expected non-retryable outcomes (not errors in the traditional sense)
    - failed_unexpected: True unexpected errors that warrant investigation
    """

    OK = "ok"
    UNSEEDABLE_EMPTY_BOOK = "unseedable_empty_book"  # API returned 200 but no bids/asks
    UNSEEDABLE_NOT_FOUND = "unseedable_not_found"  # HTTP 404
    UNSEEDABLE_BAD_REQUEST = "unseedable_bad_request"  # HTTP 400
    UNSEEDABLE_OTHER_4XX = "unseedable_other_4xx"  # Other 4xx (except 429)
    FAILED_UNEXPECTED = "failed_unexpected"  # Network errors, parsing issues, etc.

    @property
    def is_unseedable(self) -> bool:
        """Check if this outcome represents an expected unseedable market."""
        return self.value.startswith("unseedable_")

    @property
    def is_success(self) -> bool:
        """Check if seeding was successful."""
        return self == SeedOutcomeKind.OK


@dataclass
class SeedOutcome:
    """Result of a single seed attempt with classification."""

    token_id: str
    kind: SeedOutcomeKind
    error_detail: str | None = None  # Additional error info for debugging

    @property
    def is_success(self) -> bool:
        """Check if seeding was successful."""
        return self.kind.is_success

    @property
    def is_unseedable(self) -> bool:
        """Check if market is unseedable (expected failure)."""
        return self.kind.is_unseedable


def classify_http_error(status_code: int) -> SeedOutcomeKind:
    """Classify HTTP status code into seed outcome kind.

    Args:
        status_code: HTTP response status code

    Returns:
        SeedOutcomeKind for the error
    """
    if status_code == 404:
        return SeedOutcomeKind.UNSEEDABLE_NOT_FOUND
    if status_code == 400:
        return SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST
    if 400 <= status_code < 500 and status_code != 429:
        return SeedOutcomeKind.UNSEEDABLE_OTHER_4XX
    # 429 and 5xx are retryable - handled by retry wrapper, shouldn't reach here
    return SeedOutcomeKind.FAILED_UNEXPECTED


def classify_orderbook_result(
    ob: OrderBookData | None, error: Exception | None = None
) -> SeedOutcome:
    """Classify an orderbook fetch result into a seed outcome.

    Args:
        ob: The OrderBookData result (may be None on error)
        error: Exception if the fetch failed

    Returns:
        SeedOutcome with classification
    """
    if error is not None:
        # Check for HTTP status errors
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            kind = classify_http_error(status)
            return SeedOutcome(
                token_id=ob.token_id if ob else "unknown",
                kind=kind,
                error_detail=f"HTTP {status}",
            )

        # Network/timeout errors that got through retry wrapper are unexpected
        return SeedOutcome(
            token_id=ob.token_id if ob else "unknown",
            kind=SeedOutcomeKind.FAILED_UNEXPECTED,
            error_detail=f"{type(error).__name__}: {error}",
        )

    if ob is None:
        return SeedOutcome(
            token_id="unknown",
            kind=SeedOutcomeKind.FAILED_UNEXPECTED,
            error_detail="No orderbook returned",
        )

    # Check for error field set in OrderBookData
    if ob.error:
        error_str = ob.error.lower()
        if "404" in error_str or "not_found" in error_str:
            return SeedOutcome(
                token_id=ob.token_id,
                kind=SeedOutcomeKind.UNSEEDABLE_NOT_FOUND,
                error_detail=ob.error,
            )
        if "400" in error_str or "bad_request" in error_str:
            return SeedOutcome(
                token_id=ob.token_id,
                kind=SeedOutcomeKind.UNSEEDABLE_BAD_REQUEST,
                error_detail=ob.error,
            )
        if "http_4" in error_str:
            # Parse status code if available
            for code in ["401", "403", "422"]:
                if code in error_str:
                    return SeedOutcome(
                        token_id=ob.token_id,
                        kind=SeedOutcomeKind.UNSEEDABLE_OTHER_4XX,
                        error_detail=ob.error,
                    )
            return SeedOutcome(
                token_id=ob.token_id,
                kind=SeedOutcomeKind.UNSEEDABLE_OTHER_4XX,
                error_detail=ob.error,
            )
        # Other errors are unexpected
        return SeedOutcome(
            token_id=ob.token_id,
            kind=SeedOutcomeKind.FAILED_UNEXPECTED,
            error_detail=ob.error,
        )

    # Check for empty book (valid response but no data)
    if not ob.has_valid_book:
        return SeedOutcome(
            token_id=ob.token_id,
            kind=SeedOutcomeKind.UNSEEDABLE_EMPTY_BOOK,
            error_detail="No bids or asks",
        )

    # Success!
    return SeedOutcome(
        token_id=ob.token_id,
        kind=SeedOutcomeKind.OK,
    )


@dataclass
class SeedOutcomeStats:
    """Aggregated statistics for seed outcomes (Phase 5.9)."""

    total: int = 0
    ok: int = 0
    unseedable: int = 0
    failed_unexpected: int = 0
    # Breakdown by kind
    kinds: dict[str, int] = field(default_factory=dict)

    def add(self, outcome: SeedOutcome) -> None:
        """Add an outcome to the stats."""
        self.total += 1
        kind_key = outcome.kind.value
        self.kinds[kind_key] = self.kinds.get(kind_key, 0) + 1

        if outcome.is_success:
            self.ok += 1
        elif outcome.is_unseedable:
            self.unseedable += 1
        else:
            self.failed_unexpected += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "total": self.total,
            "ok": self.ok,
            "unseedable": self.unseedable,
            "failed_unexpected": self.failed_unexpected,
            "kinds": self.kinds,
        }


# =============================================================================
# Skiplist for Unseedable Tokens (Optional Feature)
# =============================================================================


@dataclass
class SkiplistEntry:
    """Entry in the seed skiplist."""

    token_id: str
    kind: str  # SeedOutcomeKind value
    added_at: str  # ISO timestamp
    expires_at: str  # ISO timestamp


@dataclass
class SeedSkiplist:
    """Simple file-backed skiplist for unseedable tokens.

    Stores token IDs that are known to be unseedable, with TTL expiry.
    Uses atomic writes for safety.
    """

    path: Path
    ttl_hours: float = 24.0  # Default: skip for 24 hours

    _entries: dict[str, SkiplistEntry] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)
    _dirty: bool = field(default=False, init=False)  # Track if entries changed

    @property
    def dirty(self) -> bool:
        """Check if skiplist has unsaved changes."""
        return self._dirty

    def load(self) -> None:
        """Load skiplist from file if it exists (public API)."""
        self._load()

    def _load(self) -> None:
        """Load skiplist from file if it exists."""
        if self._loaded:
            return

        self._loaded = True
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for entry_data in data.get("entries", []):
                entry = SkiplistEntry(
                    token_id=entry_data["token_id"],
                    kind=entry_data["kind"],
                    added_at=entry_data["added_at"],
                    expires_at=entry_data["expires_at"],
                )
                self._entries[entry.token_id] = entry
            logger.debug(f"Loaded {len(self._entries)} skiplist entries from {self.path}")
        except Exception as e:
            logger.warning(f"Failed to load skiplist from {self.path}: {e}")
            self._entries = {}

    def _save(self) -> None:
        """Save skiplist to file with atomic write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Build data
        data = {
            "version": 1,
            "entries": [
                {
                    "token_id": e.token_id,
                    "kind": e.kind,
                    "added_at": e.added_at,
                    "expires_at": e.expires_at,
                }
                for e in self._entries.values()
            ],
        }

        # Atomic write: write to temp, then rename
        temp_path = self.path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_path.replace(self.path)
            logger.debug(f"Saved {len(self._entries)} skiplist entries to {self.path}")
        except Exception as e:
            logger.warning(f"Failed to save skiplist to {self.path}: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def is_skipped(self, token_id: str) -> bool:
        """Check if a token is in the skiplist and not expired.

        Args:
            token_id: Token ID to check

        Returns:
            True if token should be skipped
        """
        self._load()

        entry = self._entries.get(token_id)
        if entry is None:
            return False

        # Check expiry
        now = datetime.now(UTC)
        try:
            expires_at = datetime.fromisoformat(entry.expires_at.replace("Z", "+00:00"))
            if now >= expires_at:
                # Expired - remove entry
                del self._entries[token_id]
                return False
        except ValueError:
            # Invalid timestamp - remove entry
            del self._entries[token_id]
            return False

        return True

    def should_skip(self, token_id: str) -> bool:
        """Alias for is_skipped (public API)."""
        return self.is_skipped(token_id)

    def add_token(self, token_id: str, kind: SeedOutcomeKind) -> None:
        """Add a token to the skiplist by ID.

        Args:
            token_id: Token ID to add
            kind: The outcome kind (must be unseedable)
        """
        if not kind.is_unseedable:
            return

        self._load()

        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.ttl_hours)

        self._entries[token_id] = SkiplistEntry(
            token_id=token_id,
            kind=kind.value,
            added_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        self._dirty = True

    def add(self, outcome: SeedOutcome) -> None:
        """Add an unseedable outcome to the skiplist.

        Only adds if outcome is unseedable (not ok, not failed_unexpected).

        Args:
            outcome: The seed outcome to add
        """
        if not outcome.is_unseedable:
            return  # Only skip unseedable markets

        self._load()

        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.ttl_hours)

        self._entries[outcome.token_id] = SkiplistEntry(
            token_id=outcome.token_id,
            kind=outcome.kind.value,
            added_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        self._dirty = True

    def save(self) -> None:
        """Persist skiplist to disk."""
        self._load()  # Ensure loaded first
        self._prune_expired()
        self._save()

    def _prune_expired(self) -> None:
        """Remove expired entries."""
        now = datetime.now(UTC)
        expired = []
        for token_id, entry in self._entries.items():
            try:
                expires_at = datetime.fromisoformat(entry.expires_at.replace("Z", "+00:00"))
                if now >= expires_at:
                    expired.append(token_id)
            except ValueError:
                expired.append(token_id)

        for token_id in expired:
            del self._entries[token_id]

    def get_skipped_tokens(self) -> set[str]:
        """Get all currently skipped (non-expired) token IDs.

        Returns:
            Set of token IDs to skip
        """
        self._load()
        self._prune_expired()
        return set(self._entries.keys())

    def __len__(self) -> int:
        """Get number of entries in skiplist."""
        self._load()
        return len(self._entries)
