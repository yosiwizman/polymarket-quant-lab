"""Quality checks for snapshot data.

Provides gap detection, duplicate detection, and stale market detection
to ensure snapshot data is suitable for backtesting.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pmq.logging import get_logger
from pmq.storage.dao import DAO

logger = get_logger("quality.checks")


def _parse_iso_datetime(dt_str: str) -> datetime:
    """Parse ISO datetime string to timezone-aware datetime.

    Handles various ISO formats:
    - 2025-01-01T00:00:00Z
    - 2025-01-01T00:00:00+00:00
    - 2025-01-01T00:00:00 (naive, assumed UTC)

    Args:
        dt_str: ISO datetime string

    Returns:
        Timezone-aware datetime (UTC)
    """
    # Normalize Z to +00:00
    normalized = dt_str.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(normalized)
        # If naive (no timezone), assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        # Fallback: try basic parsing and assume UTC
        try:
            dt = datetime.fromisoformat(dt_str.split("+")[0].replace("Z", ""))
            return dt.replace(tzinfo=UTC)
        except ValueError:
            # Last resort: return current time (shouldn't happen)
            logger.warning(f"Could not parse datetime: {dt_str}")
            return datetime.now(UTC)


# Minimum distinct snapshot times required for meaningful analysis
MIN_SNAPSHOTS_FOR_QUALITY = 30


class QualityStatus:
    """Quality status constants."""

    SUFFICIENT = "SUFFICIENT"  # Enough data for reliable analysis
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough snapshots
    DEGRADED = "DEGRADED"  # Data available but quality issues
    UNKNOWN = "UNKNOWN"  # Cannot determine


@dataclass
class GapInfo:
    """Information about a gap in snapshot data."""

    gap_start: str
    gap_end: str
    gap_seconds: float
    expected_seconds: int


@dataclass
class QualityResult:
    """Result of quality checks on a time window."""

    window_from: str
    window_to: str
    expected_interval_seconds: int
    markets_seen: int = 0
    snapshots_written: int = 0
    missing_intervals: int = 0
    largest_gap_seconds: float = 0.0
    duplicate_count: int = 0
    stale_market_count: int = 0
    coverage_pct: float = 0.0
    status: str = QualityStatus.UNKNOWN  # Quality status
    gaps: list[GapInfo] = field(default_factory=list)
    top_gap_markets: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sufficient(self) -> bool:
        """Check if data is sufficient for analysis."""
        return self.status == QualityStatus.SUFFICIENT

    @property
    def distinct_times(self) -> int:
        """Get distinct snapshot times from notes."""
        value = self.notes.get("distinct_snapshot_times", 0)
        return int(value) if value is not None else 0


class QualityChecker:
    """Checks snapshot data quality.

    Analyzes snapshot data for:
    - Gaps: Missing intervals between snapshots
    - Duplicates: Same market+time entries
    - Stale markets: Markets not updated recently
    - Coverage: Percentage of expected data present
    """

    def __init__(self, dao: DAO | None = None) -> None:
        """Initialize checker.

        Args:
            dao: Data access object
        """
        self._dao = dao or DAO()

    def check_window(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int = 60,
    ) -> QualityResult:
        """Run all quality checks on a time window.

        Args:
            start_time: Start of window (ISO format)
            end_time: End of window (ISO format)
            expected_interval_seconds: Expected interval between snapshots

        Returns:
            QualityResult with all check results
        """
        result = QualityResult(
            window_from=start_time,
            window_to=end_time,
            expected_interval_seconds=expected_interval_seconds,
        )

        # Get basic coverage stats
        coverage = self._dao.get_snapshot_coverage(start_time, end_time)
        result.snapshots_written = coverage["total_snapshots"]
        result.markets_seen = coverage["markets_covered"]

        # Check for gaps
        gaps = self._check_gaps(start_time, end_time, expected_interval_seconds)
        result.gaps = gaps
        result.missing_intervals = len(gaps)
        if gaps:
            result.largest_gap_seconds = max(g.gap_seconds for g in gaps)

        # Check for duplicates
        duplicates = self._dao.get_duplicate_snapshots(start_time, end_time)
        result.duplicate_count = sum(d["dup_count"] - 1 for d in duplicates)

        # Calculate coverage percentage
        result.coverage_pct = self._calculate_coverage(
            start_time, end_time, expected_interval_seconds, coverage["distinct_times"]
        )

        # Find markets with most gaps (top offenders)
        result.top_gap_markets = self._find_gap_markets(
            start_time, end_time, expected_interval_seconds
        )

        # Check for stale markets
        result.stale_market_count = self._count_stale_markets(end_time)

        # Add notes
        distinct_times = coverage["distinct_times"]
        result.notes = {
            "distinct_snapshot_times": distinct_times,
            "duplicate_entries": len(duplicates),
            "min_snapshots_required": MIN_SNAPSHOTS_FOR_QUALITY,
        }

        # Determine quality status
        if distinct_times < MIN_SNAPSHOTS_FOR_QUALITY:
            result.status = QualityStatus.INSUFFICIENT_DATA
            # Adjust coverage to reflect insufficient data
            # Coverage is technically correct but misleading without enough samples
            result.notes["coverage_note"] = (
                f"Only {distinct_times} snapshot times found, "
                f"need at least {MIN_SNAPSHOTS_FOR_QUALITY} for reliable analysis"
            )
        elif result.coverage_pct >= 80 and result.duplicate_count == 0:
            result.status = QualityStatus.SUFFICIENT
        elif result.coverage_pct >= 50:
            result.status = QualityStatus.DEGRADED
        else:
            result.status = QualityStatus.INSUFFICIENT_DATA

        return result

    def _check_gaps(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int,
    ) -> list[GapInfo]:
        """Find gaps in snapshot times.

        Args:
            start_time: Start of window
            end_time: End of window
            expected_interval_seconds: Expected interval

        Returns:
            List of GapInfo objects
        """
        raw_gaps = self._dao.get_snapshot_gaps(start_time, end_time, expected_interval_seconds)
        return [
            GapInfo(
                gap_start=g["gap_start"],
                gap_end=g["gap_end"],
                gap_seconds=g["gap_seconds"],
                expected_seconds=g["expected_seconds"],
            )
            for g in raw_gaps
        ]

    def _calculate_coverage(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int,
        actual_intervals: int,
    ) -> float:
        """Calculate coverage percentage.

        Args:
            start_time: Start of window
            end_time: End of window
            expected_interval_seconds: Expected interval
            actual_intervals: Number of actual distinct snapshot times

        Returns:
            Coverage percentage (0-100)
        """
        try:
            start = _parse_iso_datetime(start_time)
            end = _parse_iso_datetime(end_time)
            window_seconds = (end - start).total_seconds()

            if window_seconds <= 0:
                return 0.0

            expected_intervals = int(window_seconds / expected_interval_seconds) + 1
            if expected_intervals <= 0:
                return 0.0

            coverage = (actual_intervals / expected_intervals) * 100
            return min(100.0, coverage)  # Cap at 100%
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _find_gap_markets(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int,
    ) -> list[dict[str, Any]]:
        """Find markets with the most gaps.

        Args:
            start_time: Start of window
            end_time: End of window
            expected_interval_seconds: Expected interval

        Returns:
            List of markets with gap info
        """
        # This is a simplified version - per-market gap analysis
        # For full implementation, would need per-market snapshot times
        coverage = self._dao.get_snapshot_coverage(start_time, end_time)

        # Find markets with fewer snapshots than expected
        try:
            start = _parse_iso_datetime(start_time)
            end = _parse_iso_datetime(end_time)
            window_seconds = (end - start).total_seconds()
            expected_count = int(window_seconds / expected_interval_seconds) + 1
        except ValueError:
            return []

        gap_markets = []
        for market in coverage.get("markets", [])[:20]:
            actual = market["snapshot_count"]
            if actual < expected_count * 0.8:  # Less than 80% coverage
                gap_markets.append(
                    {
                        "market_id": market["market_id"],
                        "snapshot_count": actual,
                        "expected_count": expected_count,
                        "coverage_pct": (actual / expected_count * 100)
                        if expected_count > 0
                        else 0,
                    }
                )

        return sorted(gap_markets, key=lambda x: x["coverage_pct"])[:10]

    def _count_stale_markets(self, end_time: str) -> int:
        """Count markets not updated recently.

        Args:
            end_time: Reference time

        Returns:
            Number of stale markets
        """
        # A market is stale if it has snapshots but none in the last portion of the window
        # This is a heuristic - markets that stopped updating
        summary = self._dao.get_snapshot_summary()
        last_snapshot = summary.get("last_snapshot")

        if not last_snapshot:
            return 0

        try:
            last = _parse_iso_datetime(last_snapshot)
            end = _parse_iso_datetime(end_time)

            # If last snapshot is more than 1 hour before end_time, consider stale
            if (end - last).total_seconds() > 3600:
                return 1  # At least some staleness
            return 0
        except ValueError:
            return 0
