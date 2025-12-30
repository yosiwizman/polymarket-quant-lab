"""Tests for snapshot quality and manifest functionality.

Tests gap detection, coverage calculation, quality reports,
and backtest manifest storage.
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pmq.quality.checks import (
    MIN_SNAPSHOTS_FOR_QUALITY,
    QualityChecker,
    QualityResult,
    QualityStatus,
    _parse_iso_datetime,
)
from pmq.quality.report import QualityReporter
from pmq.storage.dao import DAO
from pmq.storage.db import Database


@pytest.fixture
def temp_db() -> Database:
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path=db_path)
        db.initialize()
        yield db
        db.close()


@pytest.fixture
def dao_with_db(temp_db: Database) -> DAO:
    """Create a DAO with temp database."""
    return DAO(db=temp_db)


def _create_test_market(dao: DAO, market_id: str) -> None:
    """Helper to create a test market for foreign key constraints."""
    from pmq.models import GammaMarket

    market = GammaMarket(
        id=market_id,
        slug=f"test-{market_id}",
        question=f"Test market {market_id}?",
        condition_id="test_condition",
        yes_price=0.5,
        no_price=0.5,
        liquidity=1000.0,
        volume=10000.0,
        volume24hr=500.0,
        active=True,
        closed=False,
    )
    dao.upsert_market(market)


class TestQualityChecker:
    """Tests for QualityChecker class."""

    def test_check_window_empty(self, dao_with_db: DAO) -> None:
        """Check window with no snapshots returns zero coverage."""
        checker = QualityChecker(dao=dao_with_db)

        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T01:00:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.snapshots_written == 0
        assert result.markets_seen == 0
        assert result.coverage_pct == 0.0
        assert result.missing_intervals == 0
        assert result.duplicate_count == 0

    def test_check_window_with_data(self, dao_with_db: DAO) -> None:
        """Check window with snapshots calculates coverage."""
        # Create market first (FK constraint)
        _create_test_market(dao_with_db, "market_1")

        # Insert test snapshots
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(10):
            snapshot_time = (base_time + timedelta(minutes=i)).isoformat()
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        checker = QualityChecker(dao=dao_with_db)

        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:09:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.snapshots_written == 10
        assert result.markets_seen == 1
        assert result.coverage_pct > 0
        assert result.duplicate_count == 0

    def test_gap_detection(self, dao_with_db: DAO) -> None:
        """Detect gaps in snapshot data."""
        _create_test_market(dao_with_db, "market_1")

        # Insert snapshots with a gap (0, 1, 2, then 5 - gap at 3, 4)
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in [0, 1, 2, 5, 6, 7]:
            snapshot_time = (base_time + timedelta(minutes=i)).isoformat()
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        checker = QualityChecker(dao=dao_with_db)

        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:07:00+00:00",
            expected_interval_seconds=60,
        )

        # Should detect the gap between minute 2 and 5
        assert result.missing_intervals >= 1
        assert result.largest_gap_seconds >= 180  # 3 minutes

    def test_duplicate_detection(self, dao_with_db: DAO) -> None:
        """Detect duplicate snapshots."""
        _create_test_market(dao_with_db, "market_1")

        # Insert duplicate snapshots (same market + time)
        snapshot_time = "2024-01-01T00:00:00+00:00"
        for _ in range(3):  # 3 duplicates
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        checker = QualityChecker(dao=dao_with_db)

        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T01:00:00+00:00",
            expected_interval_seconds=60,
        )

        # Should count 2 duplicates (3 entries - 1 original = 2 dupes)
        assert result.duplicate_count == 2


class TestQualityReporter:
    """Tests for QualityReporter class."""

    def test_generate_report_saves_to_db(self, dao_with_db: DAO) -> None:
        """Generate report saves to database."""
        _create_test_market(dao_with_db, "market_1")

        # Insert some test data
        dao_with_db.save_snapshot(
            market_id="market_1",
            yes_price=0.5,
            no_price=0.5,
            liquidity=1000.0,
            volume=500.0,
            snapshot_time="2024-01-01T00:00:00+00:00",
        )

        reporter = QualityReporter(dao=dao_with_db)

        reporter.generate_report(
            start_time="2024-01-01",
            end_time="2024-01-01",
            expected_interval_seconds=60,
            save=True,
        )

        # Should save to DB
        latest = dao_with_db.get_latest_quality_report()
        assert latest is not None
        assert latest["snapshots_written"] == 1

    def test_status_badge_healthy(self, dao_with_db: DAO) -> None:
        """Status badge returns healthy for good data."""
        reporter = QualityReporter(dao=dao_with_db)

        result = QualityResult(
            window_from="2024-01-01",
            window_to="2024-01-01",
            expected_interval_seconds=60,
            coverage_pct=98.0,
            missing_intervals=2,
            duplicate_count=0,
        )

        status = reporter.get_status_badge(result)
        assert status == "healthy"

    def test_status_badge_degraded(self, dao_with_db: DAO) -> None:
        """Status badge returns degraded for moderate issues."""
        reporter = QualityReporter(dao=dao_with_db)

        result = QualityResult(
            window_from="2024-01-01",
            window_to="2024-01-01",
            expected_interval_seconds=60,
            coverage_pct=85.0,
            missing_intervals=10,
            duplicate_count=0,
        )

        status = reporter.get_status_badge(result)
        assert status == "degraded"

    def test_status_badge_unhealthy(self, dao_with_db: DAO) -> None:
        """Status badge returns unhealthy for bad data."""
        reporter = QualityReporter(dao=dao_with_db)

        result = QualityResult(
            window_from="2024-01-01",
            window_to="2024-01-01",
            expected_interval_seconds=60,
            coverage_pct=50.0,
            missing_intervals=100,
            duplicate_count=50,
        )

        status = reporter.get_status_badge(result)
        assert status == "unhealthy"

    def test_coverage_summary(self, dao_with_db: DAO) -> None:
        """Get coverage summary for window."""
        # Create markets first
        for i in range(5):
            _create_test_market(dao_with_db, f"market_{i}")

        # Insert test snapshots
        for i in range(5):
            dao_with_db.save_snapshot(
                market_id=f"market_{i}",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time="2024-01-01T00:00:00+00:00",
            )

        reporter = QualityReporter(dao=dao_with_db)
        summary = reporter.get_coverage_summary("2024-01-01", "2024-01-01")

        assert summary["total_snapshots"] == 5
        assert summary["markets_covered"] == 5


class TestBacktestManifest:
    """Tests for backtest manifest functionality."""

    def test_save_manifest(self, dao_with_db: DAO) -> None:
        """Save and retrieve a manifest."""
        # Create a backtest run first
        dao_with_db.create_backtest_run(
            run_id="test_run_123",
            strategy="arb",
            start_date="2024-01-01",
            end_date="2024-01-07",
            initial_balance=10000.0,
        )

        # Save manifest
        dao_with_db.save_backtest_manifest(
            run_id="test_run_123",
            strategy="arb",
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-07T23:59:59+00:00",
            config_hash="abc123def456",
            code_git_sha="1234567890abcdef",
            snapshot_count=1000,
            first_snapshot_time="2024-01-01T00:00:00+00:00",
            last_snapshot_time="2024-01-07T23:59:00+00:00",
        )

        # Retrieve manifest
        manifest = dao_with_db.get_backtest_manifest("test_run_123")

        assert manifest is not None
        assert manifest["strategy"] == "arb"
        assert manifest["config_hash"] == "abc123def456"
        assert manifest["code_git_sha"] == "1234567890abcdef"
        assert manifest["snapshot_count"] == 1000

    def test_manifest_with_market_filter(self, dao_with_db: DAO) -> None:
        """Manifest with market filter list."""
        dao_with_db.create_backtest_run(
            run_id="test_run_456",
            strategy="statarb",
            start_date="2024-01-01",
            end_date="2024-01-07",
            initial_balance=10000.0,
        )

        market_ids = ["market_a", "market_b", "market_c"]

        dao_with_db.save_backtest_manifest(
            run_id="test_run_456",
            strategy="statarb",
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-07T23:59:59+00:00",
            config_hash="xyz789",
            market_filter=market_ids,
            snapshot_count=500,
        )

        manifest = dao_with_db.get_backtest_manifest("test_run_456")

        assert manifest is not None
        assert manifest["market_filter"] == market_ids


class TestSnapshotCoverageDAO:
    """Tests for snapshot coverage DAO methods."""

    def test_get_snapshot_summary(self, dao_with_db: DAO) -> None:
        """Get overall snapshot summary."""
        # Create markets first
        for i in range(3):
            _create_test_market(dao_with_db, f"market_{i}")

        # Insert test data
        for i in range(10):
            dao_with_db.save_snapshot(
                market_id=f"market_{i % 3}",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=datetime(2024, 1, 1, i, 0, 0, tzinfo=UTC).isoformat(),
            )

        summary = dao_with_db.get_snapshot_summary()

        assert summary["total_snapshots"] == 10
        assert summary["unique_markets"] == 3

    def test_get_snapshot_coverage(self, dao_with_db: DAO) -> None:
        """Get coverage for a time window."""
        # Create markets first
        for i in range(3):
            _create_test_market(dao_with_db, f"market_{i}")

        # Insert snapshots for different markets
        for market_num in range(3):
            for hour in range(5):
                dao_with_db.save_snapshot(
                    market_id=f"market_{market_num}",
                    yes_price=0.5,
                    no_price=0.5,
                    liquidity=1000.0,
                    volume=500.0,
                    snapshot_time=datetime(2024, 1, 1, hour, 0, 0, tzinfo=UTC).isoformat(),
                )

        coverage = dao_with_db.get_snapshot_coverage(
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T04:00:00+00:00",
        )

        assert coverage["total_snapshots"] == 15
        assert coverage["markets_covered"] == 3
        assert coverage["distinct_times"] == 5

    def test_get_snapshot_gaps(self, dao_with_db: DAO) -> None:
        """Find gaps in snapshot data."""
        _create_test_market(dao_with_db, "market_1")

        # Insert with gaps
        times = [0, 1, 2, 5, 6, 10]  # Gap at 3-4, 7-9
        for hour in times:
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=datetime(2024, 1, 1, hour, 0, 0, tzinfo=UTC).isoformat(),
            )

        gaps = dao_with_db.get_snapshot_gaps(
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T10:00:00+00:00",
            expected_interval_seconds=3600,  # 1 hour
        )

        # Should find 2 gaps
        assert len(gaps) == 2

    def test_get_duplicate_snapshots(self, dao_with_db: DAO) -> None:
        """Find duplicate snapshots."""
        _create_test_market(dao_with_db, "market_1")

        snapshot_time = "2024-01-01T00:00:00+00:00"

        # Insert 3 duplicates
        for _ in range(3):
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        duplicates = dao_with_db.get_duplicate_snapshots(
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
        )

        assert len(duplicates) == 1
        assert duplicates[0]["dup_count"] == 3


class TestTimezoneHandling:
    """Tests for timezone-aware datetime parsing (Phase 3.1 fix)."""

    def test_parse_iso_datetime_with_z(self) -> None:
        """Parse datetime with Z suffix."""
        dt = _parse_iso_datetime("2024-01-01T12:00:00Z")
        assert dt.tzinfo is not None
        assert dt.hour == 12

    def test_parse_iso_datetime_with_offset(self) -> None:
        """Parse datetime with +00:00 offset."""
        dt = _parse_iso_datetime("2024-01-01T12:00:00+00:00")
        assert dt.tzinfo is not None
        assert dt.hour == 12

    def test_parse_iso_datetime_naive(self) -> None:
        """Parse naive datetime (no timezone) - assumes UTC."""
        dt = _parse_iso_datetime("2024-01-01T12:00:00")
        assert dt.tzinfo is not None  # Should be converted to UTC
        assert dt.hour == 12

    def test_parse_iso_datetime_comparison(self) -> None:
        """Compare datetimes from different formats."""
        dt1 = _parse_iso_datetime("2024-01-01T12:00:00Z")
        dt2 = _parse_iso_datetime("2024-01-01T12:00:00+00:00")
        dt3 = _parse_iso_datetime("2024-01-01T12:00:00")

        # All should be equal (same moment in time)
        assert dt1 == dt2
        assert dt2 == dt3

        # Subtraction should work without TypeError
        diff = (dt1 - dt2).total_seconds()
        assert diff == 0.0


class TestQualityStatus:
    """Tests for QualityStatus and INSUFFICIENT_DATA handling."""

    def test_insufficient_data_when_few_snapshots(self, dao_with_db: DAO) -> None:
        """Status is INSUFFICIENT_DATA when < 30 distinct snapshot times."""
        _create_test_market(dao_with_db, "market_1")

        # Insert only 10 snapshot times (< MIN_SNAPSHOTS_FOR_QUALITY)
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(10):
            snapshot_time = (base_time + timedelta(minutes=i)).isoformat()
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        checker = QualityChecker(dao=dao_with_db)
        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:09:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.status == QualityStatus.INSUFFICIENT_DATA
        assert not result.is_sufficient
        assert "coverage_note" in result.notes

    def test_sufficient_data_when_many_snapshots(self, dao_with_db: DAO) -> None:
        """Status is SUFFICIENT when >= 30 distinct snapshot times and good coverage."""
        _create_test_market(dao_with_db, "market_1")

        # Insert 40 snapshot times
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(40):
            snapshot_time = (base_time + timedelta(minutes=i)).isoformat()
            dao_with_db.save_snapshot(
                market_id="market_1",
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=500.0,
                snapshot_time=snapshot_time,
            )

        checker = QualityChecker(dao=dao_with_db)
        result = checker.check_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:39:00+00:00",
            expected_interval_seconds=60,
        )

        # With 40 snapshots and 100% coverage, status should be SUFFICIENT
        assert result.status == QualityStatus.SUFFICIENT
        assert result.is_sufficient

    def test_min_snapshots_constant(self) -> None:
        """MIN_SNAPSHOTS_FOR_QUALITY is 30."""
        assert MIN_SNAPSHOTS_FOR_QUALITY == 30

    def test_quality_result_distinct_times_property(self) -> None:
        """QualityResult.distinct_times returns notes value."""
        result = QualityResult(
            window_from="2024-01-01",
            window_to="2024-01-01",
            expected_interval_seconds=60,
            notes={"distinct_snapshot_times": 42},
        )
        assert result.distinct_times == 42
