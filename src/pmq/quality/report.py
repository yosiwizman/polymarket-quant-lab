"""Quality reporting for snapshot data.

Generates and stores quality reports for snapshot data windows.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from pmq.logging import get_logger
from pmq.quality.checks import QualityChecker, QualityResult
from pmq.storage.dao import DAO

logger = get_logger("quality.report")


class QualityReporter:
    """Generates and stores quality reports.

    Creates quality reports for snapshot data and saves them
    to the database for tracking over time.
    """

    def __init__(self, dao: DAO | None = None) -> None:
        """Initialize reporter.

        Args:
            dao: Data access object
        """
        self._dao = dao or DAO()
        self._checker = QualityChecker(dao=self._dao)

    def generate_report(
        self,
        start_time: str,
        end_time: str,
        expected_interval_seconds: int = 60,
        save: bool = True,
    ) -> QualityResult:
        """Generate a quality report for a time window.

        Args:
            start_time: Start of window (ISO format or YYYY-MM-DD)
            end_time: End of window (ISO format or YYYY-MM-DD)
            expected_interval_seconds: Expected snapshot interval
            save: Whether to save the report to database

        Returns:
            QualityResult with all findings
        """
        # Normalize times
        start_time = self._normalize_time(start_time, is_start=True)
        end_time = self._normalize_time(end_time, is_start=False)

        logger.info(f"Generating quality report: {start_time} to {end_time}")

        # Run checks
        result = self._checker.check_window(
            start_time, end_time, expected_interval_seconds
        )

        # Save to database if requested
        if save:
            self._dao.save_quality_report(
                window_from=result.window_from,
                window_to=result.window_to,
                expected_interval_seconds=result.expected_interval_seconds,
                markets_seen=result.markets_seen,
                snapshots_written=result.snapshots_written,
                missing_intervals=result.missing_intervals,
                largest_gap_seconds=result.largest_gap_seconds,
                duplicate_count=result.duplicate_count,
                stale_market_count=result.stale_market_count,
                coverage_pct=result.coverage_pct,
                notes=result.notes,
            )
            logger.info(f"Saved quality report (coverage: {result.coverage_pct:.1f}%)")

        return result

    def generate_last_24h_report(
        self,
        expected_interval_seconds: int = 60,
        save: bool = True,
    ) -> QualityResult:
        """Generate report for the last 24 hours.

        Args:
            expected_interval_seconds: Expected snapshot interval
            save: Whether to save the report

        Returns:
            QualityResult
        """
        now = datetime.now(UTC)
        start_time = (now - timedelta(hours=24)).isoformat()
        end_time = now.isoformat()

        return self.generate_report(
            start_time, end_time, expected_interval_seconds, save
        )

    def get_coverage_summary(
        self,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        """Get a coverage summary for a time window.

        Args:
            start_time: Start of window
            end_time: End of window

        Returns:
            Summary dict with coverage stats
        """
        start_time = self._normalize_time(start_time, is_start=True)
        end_time = self._normalize_time(end_time, is_start=False)

        coverage = self._dao.get_snapshot_coverage(start_time, end_time)
        summary = self._dao.get_snapshot_summary()

        return {
            "window_from": start_time,
            "window_to": end_time,
            "total_snapshots": coverage["total_snapshots"],
            "distinct_times": coverage["distinct_times"],
            "markets_covered": coverage["markets_covered"],
            "overall_stats": summary,
            "top_markets": coverage.get("markets", [])[:10],
        }

    def get_status_badge(self, result: QualityResult | None = None) -> str:
        """Get a status badge based on quality metrics.

        Args:
            result: QualityResult to evaluate (or uses latest report)

        Returns:
            Status string: "healthy", "degraded", or "unhealthy"
        """
        if result is None:
            # Get latest report from DB
            latest = self._dao.get_latest_quality_report()
            if not latest:
                return "unknown"

            coverage_pct = latest.get("coverage_pct", 0)
            missing_intervals = latest.get("missing_intervals", 0)
            duplicate_count = latest.get("duplicate_count", 0)
        else:
            coverage_pct = result.coverage_pct
            missing_intervals = result.missing_intervals
            duplicate_count = result.duplicate_count

        # Determine status
        if coverage_pct >= 95 and missing_intervals <= 5 and duplicate_count == 0:
            return "healthy"
        elif coverage_pct >= 80 and missing_intervals <= 20:
            return "degraded"
        else:
            return "unhealthy"

    def _normalize_time(self, time_str: str, is_start: bool) -> str:
        """Normalize time string to ISO format.

        Args:
            time_str: Input time string
            is_start: Whether this is a start time (use 00:00) or end (23:59)

        Returns:
            ISO formatted time string
        """
        if "T" in time_str:
            return time_str

        # YYYY-MM-DD format
        if is_start:
            return f"{time_str}T00:00:00+00:00"
        else:
            return f"{time_str}T23:59:59+00:00"
