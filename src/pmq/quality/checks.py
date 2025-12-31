"""Quality checks for snapshot data.

Provides gap detection, duplicate detection, and stale market detection
to ensure snapshot data is suitable for backtesting.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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

# Maturity thresholds
MATURITY_READY_THRESHOLD = 70  # Minimum maturity score to be ready for scorecard
MATURITY_IDEAL_COVERAGE = 90  # Coverage % for 100% maturity
MATURITY_MIN_TIMES = 30  # Minimum distinct times for any maturity


class WindowMode:
    """Quality window mode constants."""

    EXPLICIT = "explicit"  # Explicit --from/--to range
    LAST_MINUTES = "last_minutes"  # Rolling window: last N minutes
    LAST_TIMES = "last_times"  # Last K distinct snapshot times


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
    window_mode: str = WindowMode.EXPLICIT  # How the window was computed
    markets_seen: int = 0
    snapshots_written: int = 0
    missing_intervals: int = 0
    largest_gap_seconds: float = 0.0
    duplicate_count: int = 0
    stale_market_count: int = 0
    coverage_pct: float = 0.0
    status: str = QualityStatus.UNKNOWN  # Quality status
    maturity_score: int = 0  # 0-100 maturity score
    ready_for_scorecard: bool = False  # True if data is ready for scorecard
    gaps: list[GapInfo] = field(default_factory=list)
    top_gap_markets: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    # Contiguous window fields (Phase 4.5)
    contiguous: bool = False  # Whether contiguous filtering was applied
    gap_cutoff_time: str | None = None  # First time excluded due to gap

    @property
    def is_sufficient(self) -> bool:
        """Check if data is sufficient for analysis."""
        return self.status == QualityStatus.SUFFICIENT

    @property
    def distinct_times(self) -> int:
        """Get distinct snapshot times from notes."""
        value = self.notes.get("distinct_snapshot_times", 0)
        return int(value) if value is not None else 0

    @property
    def expected_times(self) -> int:
        """Get expected snapshot times from notes."""
        value = self.notes.get("expected_times", 0)
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

        # Calculate expected times
        expected_times = self._calculate_expected_times(
            start_time, end_time, expected_interval_seconds
        )

        # Add notes
        distinct_times = coverage["distinct_times"]
        result.notes = {
            "distinct_snapshot_times": distinct_times,
            "expected_times": expected_times,
            "duplicate_entries": len(duplicates),
            "min_snapshots_required": MIN_SNAPSHOTS_FOR_QUALITY,
        }

        # Determine quality status
        if distinct_times < MIN_SNAPSHOTS_FOR_QUALITY:
            result.status = QualityStatus.INSUFFICIENT_DATA
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

        # Calculate maturity score and readiness
        result.maturity_score = self._calculate_maturity(
            distinct_times, result.coverage_pct, result.duplicate_count
        )
        result.ready_for_scorecard = result.maturity_score >= MATURITY_READY_THRESHOLD

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

    def _calculate_expected_times(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int,
    ) -> int:
        """Calculate expected number of snapshot times in window.

        Args:
            start_time: Start of window
            end_time: End of window
            expected_interval_seconds: Expected interval

        Returns:
            Expected number of distinct snapshot times
        """
        try:
            start = _parse_iso_datetime(start_time)
            end = _parse_iso_datetime(end_time)
            window_seconds = (end - start).total_seconds()

            if window_seconds <= 0:
                return 0

            return int(window_seconds / expected_interval_seconds) + 1
        except ValueError:
            return 0

    def _calculate_maturity(
        self,
        distinct_times: int,
        coverage_pct: float,
        duplicate_count: int,
    ) -> int:
        """Calculate data maturity score (0-100).

        Maturity combines:
        - Data volume: Do we have enough distinct snapshot times?
        - Coverage: Are we capturing data at the expected rate?
        - Quality: Are there duplicates or other issues?

        Args:
            distinct_times: Number of distinct snapshot times
            coverage_pct: Coverage percentage (0-100)
            duplicate_count: Number of duplicate entries

        Returns:
            Maturity score 0-100
        """
        # Minimum data threshold - no maturity if too few samples
        if distinct_times < MATURITY_MIN_TIMES:
            # Partial credit based on how close to minimum
            return int((distinct_times / MATURITY_MIN_TIMES) * 40)  # Max 40 if under threshold

        # Base maturity from coverage (60% weight)
        coverage_score = min(coverage_pct / MATURITY_IDEAL_COVERAGE, 1.0) * 60

        # Volume bonus for exceeding minimum (20% weight)
        volume_ratio = min(distinct_times / (MATURITY_MIN_TIMES * 2), 1.0)
        volume_score = volume_ratio * 20

        # Quality score - penalize duplicates (20% weight)
        quality_score = 20.0
        if duplicate_count > 0:
            # Each duplicate reduces quality score
            duplicate_penalty = min(duplicate_count * 2, 20)
            quality_score = max(0, quality_score - duplicate_penalty)

        total = int(coverage_score + volume_score + quality_score)
        return min(100, max(0, total))

    def check_last_minutes(
        self,
        minutes: int,
        expected_interval_seconds: int = 60,
    ) -> QualityResult:
        """Check quality of data from the last N minutes.

        Uses a rolling window from (now - minutes) to now.

        Args:
            minutes: Number of minutes to look back
            expected_interval_seconds: Expected interval between snapshots

        Returns:
            QualityResult with rolling window data
        """
        now = datetime.now(UTC)
        start = now - timedelta(minutes=minutes)

        start_str = start.isoformat()
        end_str = now.isoformat()

        result = self.check_window(start_str, end_str, expected_interval_seconds)
        result.window_mode = WindowMode.LAST_MINUTES
        result.notes["rolling_minutes"] = minutes

        return result

    def check_explicit_window(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int = 60,
    ) -> QualityResult:
        """Check quality for an explicit time window (Phase 4.7).

        Computes expected points based on the time range:
        - expected_points = floor((end - start) / interval) + 1
        - observed_points = actual distinct snapshot times in window
        - quality_pct = observed_points / expected_points * 100

        This is used to evaluate quality on the exact effective window
        used by walk-forward evaluation.

        Args:
            start_time: Start of window (ISO format)
            end_time: End of window (ISO format)
            expected_interval_seconds: Expected interval between snapshots

        Returns:
            QualityResult with explicit window quality metrics
        """
        # Calculate expected points for this window
        expected_points = self._calculate_expected_times(
            start_time, end_time, expected_interval_seconds
        )

        # Get coverage info
        coverage = self._dao.get_snapshot_coverage(start_time, end_time)
        observed_points = coverage["distinct_times"]

        # Calculate coverage percentage based on expected vs observed
        if expected_points > 0:
            coverage_pct = (observed_points / expected_points) * 100
            coverage_pct = min(100.0, coverage_pct)
        else:
            coverage_pct = 0.0

        # Check for gaps
        gaps = self._check_gaps(start_time, end_time, expected_interval_seconds)

        # Check for duplicates
        duplicates = self._dao.get_duplicate_snapshots(start_time, end_time)
        duplicate_count = sum(d["dup_count"] - 1 for d in duplicates)

        # Determine status
        if observed_points < MIN_SNAPSHOTS_FOR_QUALITY:
            status = QualityStatus.INSUFFICIENT_DATA
        elif coverage_pct >= 80 and duplicate_count == 0:
            status = QualityStatus.SUFFICIENT
        elif coverage_pct >= 50:
            status = QualityStatus.DEGRADED
        else:
            status = QualityStatus.INSUFFICIENT_DATA

        # Calculate maturity
        maturity = self._calculate_maturity(observed_points, coverage_pct, duplicate_count)

        return QualityResult(
            window_from=start_time,
            window_to=end_time,
            expected_interval_seconds=expected_interval_seconds,
            window_mode=WindowMode.EXPLICIT,
            markets_seen=coverage["markets_covered"],
            snapshots_written=coverage["total_snapshots"],
            missing_intervals=len(gaps),
            largest_gap_seconds=max((g.gap_seconds for g in gaps), default=0.0),
            duplicate_count=duplicate_count,
            stale_market_count=self._count_stale_markets(end_time),
            coverage_pct=coverage_pct,
            status=status,
            maturity_score=maturity,
            ready_for_scorecard=maturity >= MATURITY_READY_THRESHOLD,
            gaps=gaps,
            top_gap_markets=self._find_gap_markets(start_time, end_time, expected_interval_seconds),
            notes={
                "distinct_snapshot_times": observed_points,
                "expected_times": expected_points,
                "observed_times": observed_points,
                "duplicate_entries": len(duplicates),
                "min_snapshots_required": MIN_SNAPSHOTS_FOR_QUALITY,
                "window_mode": "explicit",
                "quality_note": f"Quality evaluated on explicit window: {start_time[:19]} to {end_time[:19]}",
            },
        )

    def check_last_times(
        self,
        limit: int = 30,
        expected_interval_seconds: int = 60,
        contiguous: bool = True,
        gap_factor: float = 2.5,
    ) -> QualityResult:
        """Check quality based on the last K distinct snapshot times.

        Unlike check_last_minutes which uses a time-based window,
        this method uses the actual captured snapshot times to define
        the window. This avoids penalizing for gaps before data capture.

        Args:
            limit: Number of distinct snapshot times to analyze
            expected_interval_seconds: Expected interval between snapshots
            contiguous: If True, stop at gaps (default True for last-times mode)
            gap_factor: Multiplier for gap detection (gap > interval * factor)

        Returns:
            QualityResult based on actual captured data
        """
        # Get snapshot times (contiguous or not)
        gap_cutoff_time: str | None = None
        total_available: int = 0

        if contiguous:
            result = self._dao.get_recent_snapshot_times_contiguous(
                limit=limit,
                interval_seconds=expected_interval_seconds,
                gap_factor=gap_factor,
            )
            times = result.times
            gap_cutoff_time = result.gap_cutoff_time
            total_available = result.total_available
        else:
            times = self._dao.get_recent_snapshot_times(limit)
            total_available = len(times)

        if not times:
            # No data at all
            return QualityResult(
                window_from="",
                window_to="",
                expected_interval_seconds=expected_interval_seconds,
                window_mode=WindowMode.LAST_TIMES,
                status=QualityStatus.INSUFFICIENT_DATA,
                maturity_score=0,
                ready_for_scorecard=False,
                contiguous=contiguous,
                gap_cutoff_time=gap_cutoff_time,
                notes={
                    "distinct_snapshot_times": 0,
                    "expected_times": limit,
                    "requested_times": limit,
                    "total_available": total_available,
                    "contiguous": contiguous,
                    "error": "No snapshot data available",
                },
            )

        # Window spans from first to last of the retrieved times
        start_time = times[0]
        end_time = times[-1]

        # Get coverage info for this window
        coverage = self._dao.get_snapshot_coverage(start_time, end_time)
        distinct_times = len(times)

        # For last_times mode, coverage is based on contiguous capture
        # We check gaps within the actual captured window
        gaps = self._check_gaps(start_time, end_time, expected_interval_seconds)

        # Check for duplicates
        duplicates = self._dao.get_duplicate_snapshots(start_time, end_time)
        duplicate_count = sum(d["dup_count"] - 1 for d in duplicates)

        # Coverage calculation: actual times vs what was requested
        # This gives high coverage if we got what we asked for
        coverage_pct = (distinct_times / limit) * 100 if limit > 0 else 0
        coverage_pct = min(100.0, coverage_pct)

        # Determine status
        if distinct_times < MIN_SNAPSHOTS_FOR_QUALITY:
            status = QualityStatus.INSUFFICIENT_DATA
        elif coverage_pct >= 80 and duplicate_count == 0:
            status = QualityStatus.SUFFICIENT
        elif coverage_pct >= 50:
            status = QualityStatus.DEGRADED
        else:
            status = QualityStatus.INSUFFICIENT_DATA

        # Calculate maturity
        maturity = self._calculate_maturity(distinct_times, coverage_pct, duplicate_count)

        # Build notes
        notes: dict[str, Any] = {
            "distinct_snapshot_times": distinct_times,
            "expected_times": limit,
            "requested_times": limit,
            "total_available": total_available,
            "duplicate_entries": len(duplicates),
            "min_snapshots_required": MIN_SNAPSHOTS_FOR_QUALITY,
            "contiguous": contiguous,
        }

        # Add gap info if contiguous mode detected a gap
        if contiguous and gap_cutoff_time:
            notes["gap_cutoff_time"] = gap_cutoff_time
            notes["contiguous_note"] = (
                f"Stopped at gap before {gap_cutoff_time[:19]}. "
                f"Using {distinct_times} of {total_available} available times."
            )

        return QualityResult(
            window_from=start_time,
            window_to=end_time,
            expected_interval_seconds=expected_interval_seconds,
            window_mode=WindowMode.LAST_TIMES,
            markets_seen=coverage["markets_covered"],
            snapshots_written=coverage["total_snapshots"],
            missing_intervals=len(gaps),
            largest_gap_seconds=max((g.gap_seconds for g in gaps), default=0.0),
            duplicate_count=duplicate_count,
            stale_market_count=self._count_stale_markets(end_time),
            coverage_pct=coverage_pct,
            status=status,
            maturity_score=maturity,
            ready_for_scorecard=maturity >= MATURITY_READY_THRESHOLD,
            gaps=gaps,
            top_gap_markets=self._find_gap_markets(start_time, end_time, expected_interval_seconds),
            notes=notes,
            contiguous=contiguous,
            gap_cutoff_time=gap_cutoff_time,
        )
