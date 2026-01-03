"""Live execution ledger for tracking real order placements.

Phase 12: Provides a secure, auditable ledger for live order tracking.

DESIGN PRINCIPLES:
- Full redaction: No secrets (API keys, private keys) ever stored
- Reproducible: All parameters needed to recreate order are logged
- Audit trail: Every order attempt is recorded with status and timestamps
- File-based: JSON files in exports/live_ledger/ for easy inspection

USAGE:
    from pmq.ops.live_ledger import LiveLedger, LiveOrderRecord

    ledger = LiveLedger()
    record = LiveOrderRecord(
        token_id="0x...",
        side="BUY",
        price=0.45,
        size=10.0,
        order_id="abc123",
        status="POSTED",
    )
    ledger.record_order(record)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pmq.auth.redact import mask_string, redact_secrets
from pmq.logging import get_logger

logger = get_logger("ops.live_ledger")

# Default ledger directory
DEFAULT_LEDGER_DIR = Path("exports/live_ledger")


@dataclass
class LiveOrderRecord:
    """Record of a live order attempt.

    All fields are safe for logging (no secrets).
    Token IDs are masked for privacy but identifiable.
    """

    # Order identification
    order_id: str | None  # CLOB order ID (if posted successfully)
    timestamp: str  # ISO timestamp of the attempt

    # Market/token info (masked for privacy)
    market_id: str
    token_id: str  # Masked token ID
    outcome: str  # "YES" or "NO"

    # Order parameters (reproducible)
    side: str  # "BUY" or "SELL"
    price: float  # Limit price
    size: float  # Quantity in shares
    notional_usd: float  # price * size

    # Execution result
    status: str  # "POSTED", "REJECTED", "DRY_RUN", "ERROR"
    error_message: str | None = None

    # Context
    arb_side: str = "NONE"  # BUY_BOTH, SELL_BOTH, NONE
    edge_bps: float = 0.0  # Edge at time of order
    dry_run: bool = False  # Was this a dry run?

    # Timestamps
    created_at: str = ""  # When record was created

    def __post_init__(self) -> None:
        """Set created_at if not set."""
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveOrderRecord:
        """Create from dictionary."""
        return cls(
            order_id=data.get("order_id"),
            timestamp=data["timestamp"],
            market_id=data["market_id"],
            token_id=data["token_id"],
            outcome=data["outcome"],
            side=data["side"],
            price=data["price"],
            size=data["size"],
            notional_usd=data["notional_usd"],
            status=data["status"],
            error_message=data.get("error_message"),
            arb_side=data.get("arb_side", "NONE"),
            edge_bps=data.get("edge_bps", 0.0),
            dry_run=data.get("dry_run", False),
            created_at=data.get("created_at", ""),
        )


@dataclass
class LiveLedgerStats:
    """Statistics from the live ledger."""

    total_orders: int = 0
    posted_orders: int = 0
    rejected_orders: int = 0
    dry_run_orders: int = 0
    error_orders: int = 0
    total_notional_usd: float = 0.0
    orders_last_hour: int = 0


class LiveLedger:
    """Ledger for tracking live order placements.

    Stores records in JSON files for auditability.
    Uses atomic writes for file integrity.
    """

    def __init__(self, ledger_dir: Path | None = None) -> None:
        """Initialize live ledger.

        Args:
            ledger_dir: Directory for ledger files (default: exports/live_ledger/)
        """
        self.ledger_dir = ledger_dir or DEFAULT_LEDGER_DIR
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[LiveOrderRecord] = []
        self._load_today()

    def _get_today_file(self) -> Path:
        """Get path to today's ledger file."""
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.ledger_dir / f"live_orders_{date_str}.json"

    def _load_today(self) -> None:
        """Load today's records from file."""
        file_path = self._get_today_file()
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._records = [LiveOrderRecord.from_dict(r) for r in data]
                logger.debug(f"Loaded {len(self._records)} records from {file_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load ledger from {file_path}: {e}")
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        """Save records to file atomically."""
        file_path = self._get_today_file()

        # Atomic write: temp file then rename
        fd, temp_path = tempfile.mkstemp(suffix=".json.tmp", dir=str(self.ledger_dir))
        try:
            os.close(fd)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    [r.to_dict() for r in self._records],
                    f,
                    indent=2,
                )
            # Atomic rename
            Path(temp_path).replace(file_path)
        except Exception:
            # Clean up temp file on error
            Path(temp_path).unlink(missing_ok=True)
            raise

    def record_order(self, record: LiveOrderRecord) -> None:
        """Record a live order attempt.

        Args:
            record: Order record to save
        """
        # Mask token ID for privacy
        if len(record.token_id) > 16:
            record.token_id = mask_string(record.token_id, 8)

        # Redact any secrets from error message
        if record.error_message:
            record.error_message = redact_secrets(record.error_message)

        self._records.append(record)
        self._save()

        status_emoji = {
            "POSTED": "✓",
            "DRY_RUN": "○",
            "REJECTED": "✗",
            "ERROR": "⚠",
        }.get(record.status, "?")

        logger.info(
            f"{status_emoji} Live order: {record.side} {record.size:.2f} @ {record.price:.4f} "
            f"(${record.notional_usd:.2f}) → {record.status}"
        )

    def get_stats(self) -> LiveLedgerStats:
        """Get statistics from today's ledger.

        Returns:
            LiveLedgerStats with counts and totals
        """
        stats = LiveLedgerStats()
        now = datetime.now(UTC)
        one_hour_ago = now.timestamp() - 3600

        for record in self._records:
            stats.total_orders += 1
            if record.status == "POSTED":
                stats.posted_orders += 1
                stats.total_notional_usd += record.notional_usd
            elif record.status == "DRY_RUN":
                stats.dry_run_orders += 1
            elif record.status == "REJECTED":
                stats.rejected_orders += 1
            else:
                stats.error_orders += 1

            # Count orders in last hour
            try:
                order_ts = datetime.fromisoformat(
                    record.timestamp.replace("Z", "+00:00")
                ).timestamp()
                if order_ts >= one_hour_ago:
                    stats.orders_last_hour += 1
            except (ValueError, AttributeError):
                # Timestamp parsing failed - skip this record for rate limiting
                pass

        return stats

    def get_orders_last_hour(self) -> int:
        """Get count of orders posted in the last hour.

        Returns:
            Number of POSTED orders in the last 60 minutes
        """
        now = datetime.now(UTC)
        one_hour_ago = now.timestamp() - 3600
        count = 0

        for record in self._records:
            if record.status != "POSTED":
                continue
            try:
                order_ts = datetime.fromisoformat(
                    record.timestamp.replace("Z", "+00:00")
                ).timestamp()
                if order_ts >= one_hour_ago:
                    count += 1
            except (ValueError, AttributeError):
                # Timestamp parsing failed - skip this record for rate limiting
                pass

        return count

    def get_recent_orders(self, limit: int = 20) -> list[LiveOrderRecord]:
        """Get most recent orders.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of most recent orders (newest first)
        """
        return list(reversed(self._records[-limit:]))

    def clear_records(self) -> None:
        """Clear all records (for testing only)."""
        self._records = []
        self._save()
