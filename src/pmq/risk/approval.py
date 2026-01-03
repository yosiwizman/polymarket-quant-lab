"""Approval with TTL for safe paper execution.

Phase 9: Provides explicit, expiring approvals for paper_exec so that
execution can be enabled safely without removing guardrails permanently.

DESIGN PRINCIPLES:
- Safe by default: No approval = NOT approved
- Explicit TTL: Approvals expire automatically
- File-based storage: Simple JSON files under exports/approvals/
- Scoped: Different scopes can have independent approvals

USAGE:
    # Approve paper_exec for 60 minutes
    approve("paper_exec", ttl_seconds=3600, reason="calibration run")

    # Check if approved
    if is_approved("paper_exec"):
        # Execute paper trades
        ...

    # Revoke early
    revoke("paper_exec")
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pmq.logging import get_logger

logger = get_logger("risk.approval")

# Default approvals directory
DEFAULT_APPROVALS_DIR = Path("exports/approvals")


@dataclass
class Approval:
    """An approval record with TTL.

    Attributes:
        scope: The approval scope (e.g., "paper_exec")
        approved_at: ISO timestamp when approval was granted
        expires_at: ISO timestamp when approval expires
        approved_by: Best-effort identifier of who approved (optional)
        reason: Optional reason for the approval
    """

    scope: str
    approved_at: str
    expires_at: str
    approved_by: str | None = None
    reason: str | None = None

    def is_valid(self, now: datetime | None = None) -> bool:
        """Check if approval is still valid (not expired).

        Args:
            now: Current time (defaults to UTC now)

        Returns:
            True if approval has not expired
        """
        if now is None:
            now = datetime.now(UTC)
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return now < expires
        except (ValueError, AttributeError):
            return False

    def time_remaining(self, now: datetime | None = None) -> timedelta:
        """Get time remaining until expiration.

        Args:
            now: Current time (defaults to UTC now)

        Returns:
            Time remaining (may be negative if expired)
        """
        if now is None:
            now = datetime.now(UTC)
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return expires - now
        except (ValueError, AttributeError):
            return timedelta(seconds=0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Approval:
        """Create Approval from dictionary."""
        return cls(
            scope=data["scope"],
            approved_at=data["approved_at"],
            expires_at=data["expires_at"],
            approved_by=data.get("approved_by"),
            reason=data.get("reason"),
        )


class ApprovalStore:
    """File-based storage for approvals.

    Stores each scope's approval as a separate JSON file under
    the approvals directory.
    """

    def __init__(self, approvals_dir: Path | None = None) -> None:
        """Initialize approval store.

        Args:
            approvals_dir: Directory for approval files (default: exports/approvals/)
        """
        self.approvals_dir = approvals_dir or DEFAULT_APPROVALS_DIR
        self.approvals_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, scope: str) -> Path:
        """Get file path for a scope's approval."""
        # Sanitize scope name for filesystem
        safe_scope = scope.replace("/", "_").replace("\\", "_")
        return self.approvals_dir / f"{safe_scope}.json"

    def save(self, approval: Approval) -> None:
        """Save an approval to disk.

        Args:
            approval: Approval to save
        """
        path = self._get_path(approval.scope)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(approval.to_dict(), f, indent=2)
            logger.info(f"Saved approval for scope '{approval.scope}' to {path}")
        except OSError as e:
            logger.error(f"Failed to save approval for '{approval.scope}': {e}")
            raise

    def load(self, scope: str) -> Approval | None:
        """Load an approval from disk.

        Args:
            scope: The approval scope to load

        Returns:
            Approval if found and valid JSON, None otherwise
        """
        path = self._get_path(scope)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Approval.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load approval for '{scope}': {e}")
            return None

    def delete(self, scope: str) -> bool:
        """Delete an approval file.

        Args:
            scope: The approval scope to delete

        Returns:
            True if file was deleted, False if it didn't exist
        """
        path = self._get_path(scope)
        if path.exists():
            try:
                path.unlink()
                logger.info(f"Deleted approval for scope '{scope}'")
                return True
            except OSError as e:
                logger.error(f"Failed to delete approval for '{scope}': {e}")
                return False
        return False

    def list_all(self) -> list[Approval]:
        """List all stored approvals.

        Returns:
            List of all approvals (including expired ones)
        """
        approvals: list[Approval] = []
        if not self.approvals_dir.exists():
            return approvals

        for path in self.approvals_dir.glob("*.json"):
            scope = path.stem
            approval = self.load(scope)
            if approval:
                approvals.append(approval)
        return approvals


# Default store instance
_default_store: ApprovalStore | None = None


def _get_store(approvals_dir: Path | None = None) -> ApprovalStore:
    """Get or create the default approval store."""
    global _default_store
    if approvals_dir is not None:
        return ApprovalStore(approvals_dir)
    if _default_store is None:
        _default_store = ApprovalStore()
    return _default_store


def approve(
    scope: str,
    ttl_seconds: int,
    reason: str | None = None,
    approved_by: str | None = None,
    approvals_dir: Path | None = None,
) -> Approval:
    """Grant approval for a scope with TTL.

    Args:
        scope: The approval scope (e.g., "paper_exec")
        ttl_seconds: Time-to-live in seconds
        reason: Optional reason for the approval
        approved_by: Optional identifier for who approved
        approvals_dir: Optional custom directory for approval files

    Returns:
        The created Approval object
    """
    store = _get_store(approvals_dir)
    now = datetime.now(UTC)
    expires = now + timedelta(seconds=ttl_seconds)

    approval = Approval(
        scope=scope,
        approved_at=now.isoformat(),
        expires_at=expires.isoformat(),
        approved_by=approved_by or _get_current_user(),
        reason=reason,
    )

    store.save(approval)
    logger.info(f"Approved '{scope}' for {ttl_seconds}s (until {expires.isoformat()[:19]})")
    return approval


def revoke(scope: str, approvals_dir: Path | None = None) -> bool:
    """Revoke an approval.

    Args:
        scope: The approval scope to revoke
        approvals_dir: Optional custom directory for approval files

    Returns:
        True if approval was revoked, False if it didn't exist
    """
    store = _get_store(approvals_dir)
    deleted = store.delete(scope)
    if deleted:
        logger.info(f"Revoked approval for '{scope}'")
    else:
        logger.info(f"No approval to revoke for '{scope}'")
    return deleted


def is_approved(
    scope: str,
    now: datetime | None = None,
    approvals_dir: Path | None = None,
) -> bool:
    """Check if a scope is currently approved.

    SAFE BY DEFAULT: Returns False if:
    - No approval file exists
    - Approval file is corrupted
    - Approval has expired

    Args:
        scope: The approval scope to check
        now: Current time (defaults to UTC now)
        approvals_dir: Optional custom directory for approval files

    Returns:
        True only if a valid, non-expired approval exists
    """
    store = _get_store(approvals_dir)
    approval = store.load(scope)
    if approval is None:
        return False
    return approval.is_valid(now)


def get_approval(
    scope: str,
    approvals_dir: Path | None = None,
) -> Approval | None:
    """Get the approval for a scope (even if expired).

    Args:
        scope: The approval scope
        approvals_dir: Optional custom directory for approval files

    Returns:
        Approval if found, None otherwise
    """
    store = _get_store(approvals_dir)
    return store.load(scope)


def list_approvals(approvals_dir: Path | None = None) -> list[Approval]:
    """List all approvals with their status.

    Args:
        approvals_dir: Optional custom directory for approval files

    Returns:
        List of all approvals (including expired ones)
    """
    store = _get_store(approvals_dir)
    return store.list_all()


def _get_current_user() -> str:
    """Best-effort detection of current user."""
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
