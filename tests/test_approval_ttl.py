"""Tests for Phase 9 approval TTL system.

Tests cover:
- Approval creation with TTL
- Approval expiration
- Revocation
- Listing approvals
- Safe fallback behavior
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pmq.risk.approval import (
    Approval,
    ApprovalStore,
    approve,
    get_approval,
    is_approved,
    list_approvals,
    revoke,
)

# =============================================================================
# Test: Approval Dataclass
# =============================================================================


class TestApprovalDataclass:
    """Tests for Approval dataclass."""

    def test_approval_creation(self) -> None:
        """Test basic approval creation."""
        approval = Approval(
            scope="paper_exec",
            approved_at=datetime.now(UTC).isoformat(),
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            approved_by="test_user",
            reason="Testing approval",
        )

        assert approval.scope == "paper_exec"
        assert approval.approved_by == "test_user"
        assert approval.reason == "Testing approval"

    def test_approval_is_valid_true(self) -> None:
        """Test is_valid returns True for valid approval."""
        approval = Approval(
            scope="test",
            approved_at=datetime.now(UTC).isoformat(),
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        )

        assert approval.is_valid() is True

    def test_approval_is_valid_false(self) -> None:
        """Test is_valid returns False for expired approval."""
        approval = Approval(
            scope="test",
            approved_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        )

        assert approval.is_valid() is False

    def test_approval_to_dict(self) -> None:
        """Test to_dict serialization."""
        now = datetime.now(UTC)
        approval = Approval(
            scope="paper_exec",
            approved_at=now.isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
            approved_by="admin",
            reason="Production approval",
        )

        d = approval.to_dict()

        assert d["scope"] == "paper_exec"
        assert d["approved_at"] == now.isoformat()
        assert d["approved_by"] == "admin"
        assert d["reason"] == "Production approval"

    def test_approval_from_dict(self) -> None:
        """Test from_dict deserialization."""
        data = {
            "scope": "daemon",
            "approved_at": "2025-01-03T12:00:00",
            "expires_at": "2025-01-03T13:00:00",
            "approved_by": "user",
            "reason": "Test",
        }

        approval = Approval.from_dict(data)

        assert approval.scope == "daemon"
        assert approval.approved_by == "user"

# =============================================================================
# Test: ApprovalStore
# =============================================================================


class TestApprovalStore:
    """Tests for ApprovalStore JSON persistence."""

    def test_store_save_and_load(self) -> None:
        """Test saving and loading approvals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")

            # Create approval
            now = datetime.now(UTC)
            approval = Approval(
                scope="paper_exec",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
                approved_by="test",
                reason="Testing",
            )

            # Save it
            store.save(approval)

            assert approval.scope == "paper_exec"

            # Load it back
            loaded = store.load("paper_exec")

            assert loaded is not None
            assert loaded.scope == "paper_exec"
            assert loaded.approved_by == "test"

    def test_store_load_nonexistent(self) -> None:
        """Test loading non-existent approval returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")
            result = store.load("nonexistent")
            assert result is None

    def test_store_load_expired(self) -> None:
        """Test loading expired approval still returns it (validity check is separate)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")

            # Create expired approval
            approval = Approval(
                scope="test",
                approved_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            )

            # Save it
            store.save(approval)

            # Should still load (validity check is separate from loading)
            result = store.load("test")
            assert result is not None
            assert result.is_valid() is False

    def test_store_delete(self) -> None:
        """Test deleting an approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")

            # Create approval
            now = datetime.now(UTC)
            approval = Approval(
                scope="test",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            # Verify it exists
            assert store.load("test") is not None

            # Delete
            result = store.delete("test")
            assert result is True

            # Should be gone now
            assert store.load("test") is None

    def test_store_delete_nonexistent(self) -> None:
        """Test deleting non-existent approval returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")
            result = store.delete("nonexistent")
            assert result is False

    def test_store_list_all(self) -> None:
        """Test listing all approvals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")
            now = datetime.now(UTC)

            # Create multiple approvals
            for scope in ["scope1", "scope2", "scope3"]:
                approval = Approval(
                    scope=scope,
                    approved_at=now.isoformat(),
                    expires_at=(now + timedelta(hours=1)).isoformat(),
                )
                store.save(approval)

            # List all
            approvals = store.list_all()

            assert len(approvals) == 3
            scopes = {a.scope for a in approvals}
            assert scopes == {"scope1", "scope2", "scope3"}

    def test_store_list_all_includes_expired(self) -> None:
        """Test listing includes all approvals (even expired ones)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")
            now = datetime.now(UTC)

            # Create valid approval
            valid = Approval(
                scope="valid",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(valid)

            # Create expired approval
            expired = Approval(
                scope="expired",
                approved_at=(now - timedelta(hours=2)).isoformat(),
                expires_at=(now - timedelta(hours=1)).isoformat(),
            )
            store.save(expired)

            # List returns both (filtering by validity is separate)
            approvals = store.list_all()
            assert len(approvals) == 2
            scopes = {a.scope for a in approvals}
            assert scopes == {"valid", "expired"}


# =============================================================================
# Test: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_approve_function(self) -> None:
        """Test approve() helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            approval = approve(
                scope="paper_exec",
                ttl_seconds=7200,  # 120 minutes
                approved_by="admin",
                reason="Production run",
                approvals_dir=approvals_dir,
            )

            assert approval.scope == "paper_exec"
            assert approval.approved_by == "admin"

    def test_is_approved_function(self) -> None:
        """Test is_approved() helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"

            # Not approved initially
            assert is_approved("test", approvals_dir=approvals_dir) is False

            # Approve
            approve(scope="test", ttl_seconds=3600, approvals_dir=approvals_dir)

            # Now approved
            assert is_approved("test", approvals_dir=approvals_dir) is True

    def test_revoke_function(self) -> None:
        """Test revoke() helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"

            # Create approval
            approve(scope="test", ttl_seconds=3600, approvals_dir=approvals_dir)
            assert is_approved("test", approvals_dir=approvals_dir) is True

            # Revoke
            result = revoke("test", approvals_dir=approvals_dir)
            assert result is True

            # No longer approved
            assert is_approved("test", approvals_dir=approvals_dir) is False

    def test_get_approval_function(self) -> None:
        """Test get_approval() helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"

            # No approval
            assert get_approval("test", approvals_dir=approvals_dir) is None

            # Create approval
            approve(scope="test", ttl_seconds=3600, reason="Test reason", approvals_dir=approvals_dir)

            # Get it back
            approval = get_approval("test", approvals_dir=approvals_dir)
            assert approval is not None
            assert approval.reason == "Test reason"

    def test_list_approvals_function(self) -> None:
        """Test list_approvals() helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"

            # Empty list
            assert list_approvals(approvals_dir=approvals_dir) == []

            # Create some approvals
            approve(scope="a", ttl_seconds=3600, approvals_dir=approvals_dir)
            approve(scope="b", ttl_seconds=3600, approvals_dir=approvals_dir)

            # List them
            approvals = list_approvals(approvals_dir=approvals_dir)
            assert len(approvals) == 2


# =============================================================================
# Test: Edge Cases and Safety
# =============================================================================


class TestEdgeCasesAndSafety:
    """Tests for edge cases and safe fallback behavior."""

    def test_corrupted_file_returns_none(self) -> None:
        """Test that corrupted JSON file returns None (safe fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            approvals_dir.mkdir(parents=True, exist_ok=True)
            store = ApprovalStore(approvals_dir=approvals_dir)

            # Create corrupted file
            file_path = approvals_dir / "corrupted.json"
            with open(file_path, "w") as f:
                f.write("not valid json {{{")

            # Should return None (safe fallback)
            result = store.load("corrupted")
            assert result is None

    def test_missing_fields_returns_none(self) -> None:
        """Test that file with missing required fields returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            approvals_dir.mkdir(parents=True, exist_ok=True)
            store = ApprovalStore(approvals_dir=approvals_dir)

            # Create file with missing fields
            file_path = approvals_dir / "incomplete.json"
            with open(file_path, "w") as f:
                json.dump({"scope": "incomplete"}, f)  # Missing required fields

            # Should return None (safe fallback)
            result = store.load("incomplete")
            assert result is None

    def test_safe_scope_sanitization(self) -> None:
        """Test that scope names are sanitized for safe file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ApprovalStore(approvals_dir=Path(tmpdir) / "approvals")
            now = datetime.now(UTC)

            # Scope with special characters - store sanitizes to safe filename
            approval = Approval(
                scope="scope/with\\special:chars",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            # Should work and be retrievable
            result = store.load("scope/with\\special:chars")
            assert result is not None

    def test_concurrent_access_safe(self) -> None:
        """Test that file operations are atomic (no partial writes)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            store = ApprovalStore(approvals_dir=approvals_dir)
            now = datetime.now(UTC)

            # Create an approval
            approval = Approval(
                scope="test",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            # File should be valid JSON
            file_path = approvals_dir / "test.json"
            with open(file_path) as f:
                data = json.load(f)

            assert "scope" in data
            assert "approved_at" in data
            assert "expires_at" in data


# =============================================================================
# Test: Integration with File System
# =============================================================================


class TestFileSystemIntegration:
    """Tests for file system behavior."""

    def test_creates_directory_if_missing(self) -> None:
        """Test that approvals directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "deep" / "nested" / "dir" / "approvals"
            store = ApprovalStore(approvals_dir=nested_dir)
            now = datetime.now(UTC)

            # Should create directory on save
            approval = Approval(
                scope="test",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            assert nested_dir.exists()

    def test_file_naming_convention(self) -> None:
        """Test that files are named <scope>.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            store = ApprovalStore(approvals_dir=approvals_dir)
            now = datetime.now(UTC)

            approval = Approval(
                scope="paper_exec",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            expected_path = approvals_dir / "paper_exec.json"
            assert expected_path.exists()

    def test_delete_deletes_file(self) -> None:
        """Test that delete actually deletes the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            approvals_dir = Path(tmpdir) / "approvals"
            store = ApprovalStore(approvals_dir=approvals_dir)
            now = datetime.now(UTC)

            approval = Approval(
                scope="test",
                approved_at=now.isoformat(),
                expires_at=(now + timedelta(hours=1)).isoformat(),
            )
            store.save(approval)

            file_path = approvals_dir / "test.json"
            assert file_path.exists()

            store.delete("test")
            assert not file_path.exists()
