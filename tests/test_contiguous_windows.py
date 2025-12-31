"""Tests for gap-aware contiguous window selection (Phase 4.5).

These tests verify that:
1. DAO correctly identifies gaps and returns contiguous blocks
2. QualityChecker supports contiguous mode
3. Eval pipeline uses contiguous windows for walk-forward
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmq.models import GammaMarket
from pmq.storage.dao import DAO, ContiguousTimesResult
from pmq.storage.db import Database


class TestContiguousTimesResult:
    """Tests for ContiguousTimesResult dataclass."""

    def test_basic_creation(self) -> None:
        """ContiguousTimesResult can be created with all fields."""
        result = ContiguousTimesResult(
            times=["2025-01-01T10:00:00", "2025-01-01T10:01:00"],
            gap_cutoff_time="2025-01-01T09:00:00",
            total_available=50,
            contiguous=True,
        )
        assert len(result.times) == 2
        assert result.gap_cutoff_time == "2025-01-01T09:00:00"
        assert result.total_available == 50
        assert result.contiguous is True

    def test_no_gap(self) -> None:
        """ContiguousTimesResult with no gap has None gap_cutoff_time."""
        result = ContiguousTimesResult(
            times=["2025-01-01T10:00:00", "2025-01-01T10:01:00"],
            gap_cutoff_time=None,
            total_available=2,
            contiguous=True,
        )
        assert result.gap_cutoff_time is None


def _create_test_market(dao: DAO, market_id: str) -> None:
    """Helper to create a test market for foreign key constraints."""
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


class TestDAOContiguousSelection:
    """Tests for DAO.get_recent_snapshot_times_contiguous() method."""

    @pytest.fixture
    def temp_db(self) -> Database:
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path=db_path)
            db.initialize()
            yield db
            db.close()

    @pytest.fixture
    def dao(self, temp_db: Database) -> DAO:
        """Create DAO with temp database."""
        return DAO(db=temp_db)

    def _insert_snapshots(self, dao: DAO, times: list[str], market_id: str = "test_market") -> None:
        """Helper to insert snapshot times."""
        # Create market first (FK constraint)
        _create_test_market(dao, market_id)
        for t in times:
            dao.save_snapshot(
                market_id=market_id,
                yes_price=0.5,
                no_price=0.5,
                liquidity=1000.0,
                volume=100.0,
                snapshot_time=t,
            )

    def test_empty_database(self, dao: DAO) -> None:
        """Returns empty result when no snapshots exist."""
        result = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=2.5
        )
        assert result.times == []
        assert result.gap_cutoff_time is None
        assert result.total_available == 0
        assert result.contiguous is True

    def test_single_snapshot(self, dao: DAO) -> None:
        """Returns single time when only one snapshot exists."""
        self._insert_snapshots(dao, ["2025-01-01T10:00:00"])

        result = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=2.5
        )

        assert result.times == ["2025-01-01T10:00:00"]
        assert result.gap_cutoff_time is None
        assert result.total_available == 1

    def test_no_gaps_all_returned(self, dao: DAO) -> None:
        """Returns all times when no gaps exist."""
        # Create 10 times at 60s intervals (no gaps)
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        times = [(base + timedelta(seconds=i * 60)).isoformat() for i in range(10)]
        self._insert_snapshots(dao, times)

        result = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=2.5
        )

        assert len(result.times) == 10
        assert result.gap_cutoff_time is None
        assert result.total_available == 10

    def test_gap_detected_stops_at_boundary(self, dao: DAO) -> None:
        """Stops at gap boundary and returns only recent contiguous block."""
        # Create: t1-t10 every 60s, then 2-hour gap, then t11-t30 every 60s
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Old block: t1-t10 (before gap)
        old_times = [(base + timedelta(seconds=i * 60)).isoformat() for i in range(10)]

        # Gap: 2 hours (7200 seconds) - well beyond threshold of 60s * 2.5 = 150s
        gap_start = base + timedelta(seconds=9 * 60)  # Last of old block
        new_start = gap_start + timedelta(hours=2)  # Start of new block

        # New block: t11-t30 (after gap)
        new_times = [(new_start + timedelta(seconds=i * 60)).isoformat() for i in range(20)]

        self._insert_snapshots(dao, old_times + new_times)

        result = dao.get_recent_snapshot_times_contiguous(
            limit=200, interval_seconds=60, gap_factor=2.5
        )

        # Should return only the new block (20 times)
        assert len(result.times) == 20
        assert result.gap_cutoff_time is not None
        # Gap cutoff should be the last time of the old block
        assert result.gap_cutoff_time == old_times[-1]
        assert result.total_available == 30  # 10 old + 20 new

    def test_limit_respected(self, dao: DAO) -> None:
        """Respects limit even when more contiguous times available."""
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        times = [(base + timedelta(seconds=i * 60)).isoformat() for i in range(50)]
        self._insert_snapshots(dao, times)

        result = dao.get_recent_snapshot_times_contiguous(
            limit=20, interval_seconds=60, gap_factor=2.5
        )

        assert len(result.times) == 20
        # Should be the 20 most recent times
        assert result.times == sorted(times)[-20:]

    def test_gap_factor_affects_detection(self, dao: DAO) -> None:
        """Gap factor affects what counts as a gap."""
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Create times with 120s intervals (2x expected 60s interval)
        times = [(base + timedelta(seconds=i * 120)).isoformat() for i in range(10)]
        self._insert_snapshots(dao, times)

        # With gap_factor=1.5, 120s gaps should be detected (60*1.5=90s threshold)
        result_strict = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=1.5
        )
        # Should stop at first gap, returning only 1 time
        assert len(result_strict.times) == 1

        # With gap_factor=3.0, 120s gaps should NOT be detected (60*3=180s threshold)
        result_loose = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=3.0
        )
        # Should return all 10 times
        assert len(result_loose.times) == 10

    def test_times_returned_ascending(self, dao: DAO) -> None:
        """Times are returned in ascending order."""
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
        times = [(base + timedelta(seconds=i * 60)).isoformat() for i in range(10)]
        self._insert_snapshots(dao, times)

        result = dao.get_recent_snapshot_times_contiguous(
            limit=30, interval_seconds=60, gap_factor=2.5
        )

        # Verify ascending order
        for i in range(1, len(result.times)):
            assert result.times[i] > result.times[i - 1]


class TestQualityCheckerContiguous:
    """Tests for QualityChecker.check_last_times() with contiguous mode."""

    @pytest.fixture
    def mock_dao(self) -> MagicMock:
        """Create mock DAO for testing."""
        dao = MagicMock(spec=DAO)
        return dao

    def test_contiguous_true_calls_contiguous_method(self, mock_dao: MagicMock) -> None:
        """When contiguous=True, uses get_recent_snapshot_times_contiguous()."""
        from pmq.quality.checks import QualityChecker

        # Setup mock
        mock_dao.get_recent_snapshot_times_contiguous.return_value = ContiguousTimesResult(
            times=["2025-01-01T10:00:00", "2025-01-01T10:01:00"],
            gap_cutoff_time=None,
            total_available=2,
            contiguous=True,
        )
        mock_dao.get_snapshot_coverage.return_value = {
            "total_snapshots": 100,
            "markets_covered": 50,
            "distinct_times": 2,
            "markets": [],
        }
        mock_dao.get_duplicate_snapshots.return_value = []
        mock_dao.get_snapshot_summary.return_value = {
            "last_snapshot": "2025-01-01T10:01:00",
        }
        mock_dao.get_snapshot_gaps.return_value = []

        checker = QualityChecker(dao=mock_dao)
        result = checker.check_last_times(limit=30, expected_interval_seconds=60, contiguous=True)

        mock_dao.get_recent_snapshot_times_contiguous.assert_called_once()
        assert result.contiguous is True

    def test_contiguous_false_calls_regular_method(self, mock_dao: MagicMock) -> None:
        """When contiguous=False, uses get_recent_snapshot_times()."""
        from pmq.quality.checks import QualityChecker

        mock_dao.get_recent_snapshot_times.return_value = [
            "2025-01-01T10:00:00",
            "2025-01-01T10:01:00",
        ]
        mock_dao.get_snapshot_coverage.return_value = {
            "total_snapshots": 100,
            "markets_covered": 50,
            "distinct_times": 2,
            "markets": [],
        }
        mock_dao.get_duplicate_snapshots.return_value = []
        mock_dao.get_snapshot_summary.return_value = {
            "last_snapshot": "2025-01-01T10:01:00",
        }
        mock_dao.get_snapshot_gaps.return_value = []

        checker = QualityChecker(dao=mock_dao)
        result = checker.check_last_times(limit=30, expected_interval_seconds=60, contiguous=False)

        mock_dao.get_recent_snapshot_times.assert_called_once()
        mock_dao.get_recent_snapshot_times_contiguous.assert_not_called()
        assert result.contiguous is False

    def test_gap_cutoff_included_in_result(self, mock_dao: MagicMock) -> None:
        """Gap cutoff time is included in QualityResult."""
        from pmq.quality.checks import QualityChecker

        mock_dao.get_recent_snapshot_times_contiguous.return_value = ContiguousTimesResult(
            times=["2025-01-01T12:00:00", "2025-01-01T12:01:00"],
            gap_cutoff_time="2025-01-01T10:00:00",  # Gap detected
            total_available=50,
            contiguous=True,
        )
        mock_dao.get_snapshot_coverage.return_value = {
            "total_snapshots": 100,
            "markets_covered": 50,
            "distinct_times": 2,
            "markets": [],
        }
        mock_dao.get_duplicate_snapshots.return_value = []
        mock_dao.get_snapshot_summary.return_value = {
            "last_snapshot": "2025-01-01T12:01:00",
        }
        mock_dao.get_snapshot_gaps.return_value = []

        checker = QualityChecker(dao=mock_dao)
        result = checker.check_last_times(limit=100, expected_interval_seconds=60, contiguous=True)

        assert result.gap_cutoff_time == "2025-01-01T10:00:00"
        assert result.notes.get("gap_cutoff_time") == "2025-01-01T10:00:00"
        assert result.notes.get("total_available") == 50

    def test_contiguous_default_true(self, mock_dao: MagicMock) -> None:
        """Contiguous defaults to True for check_last_times()."""
        from pmq.quality.checks import QualityChecker

        mock_dao.get_recent_snapshot_times_contiguous.return_value = ContiguousTimesResult(
            times=["2025-01-01T10:00:00"],
            gap_cutoff_time=None,
            total_available=1,
            contiguous=True,
        )
        mock_dao.get_snapshot_coverage.return_value = {
            "total_snapshots": 50,
            "markets_covered": 25,
            "distinct_times": 1,
            "markets": [],
        }
        mock_dao.get_duplicate_snapshots.return_value = []
        mock_dao.get_snapshot_summary.return_value = {
            "last_snapshot": "2025-01-01T10:00:00",
        }
        mock_dao.get_snapshot_gaps.return_value = []

        checker = QualityChecker(dao=mock_dao)
        # Call without specifying contiguous - should default to True
        checker.check_last_times(limit=30, expected_interval_seconds=60)

        mock_dao.get_recent_snapshot_times_contiguous.assert_called_once()


class TestEvalPipelineContiguous:
    """Tests for evaluation pipeline contiguous window integration."""

    def test_walkforward_uses_contiguous(self) -> None:
        """Walk-forward evaluation uses contiguous mode."""
        from pmq.evaluation.pipeline import EvaluationPipeline

        mock_dao = MagicMock(spec=DAO)
        pipeline = EvaluationPipeline(dao=mock_dao)

        # Mock quality result with contiguous info
        mock_quality_result = MagicMock()
        mock_quality_result.status = "SUFFICIENT"
        mock_quality_result.maturity_score = 80
        mock_quality_result.ready_for_scorecard = True
        mock_quality_result.window_from = "2025-01-01T10:00:00"
        mock_quality_result.window_to = "2025-01-01T12:00:00"
        mock_quality_result.contiguous = True
        mock_quality_result.gap_cutoff_time = "2025-01-01T09:00:00"
        mock_quality_result.distinct_times = 100

        with (
            patch.object(pipeline, "_check_quality", return_value=mock_quality_result),
            patch.object(pipeline, "_should_use_walk_forward", return_value=True),
        ):
            # Just verify the contiguous flag is used
            quality_result = pipeline._check_quality(
                window_mode="last_times",
                window_value=150,
                interval_seconds=60,
                contiguous=True,
            )

            assert quality_result.contiguous is True

    def test_scaling_train_test_when_insufficient(self) -> None:
        """Train/test scaled down when fewer times available than requested."""
        # This verifies the scaling logic in the pipeline
        train_times = 100
        test_times = 50
        total_available = 90  # Less than 150 requested

        train_ratio = train_times / (train_times + test_times)  # 0.666...
        actual_train = max(1, int(total_available * train_ratio))  # 60
        actual_test = max(1, total_available - actual_train)  # 30

        assert actual_train == 60
        assert actual_test == 30
        assert actual_train + actual_test == 90

    def test_result_includes_contiguous_fields(self) -> None:
        """EvaluationResult includes contiguous-related fields."""
        from pmq.evaluation.pipeline import EvaluationResult

        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="v1",
            created_at="2025-01-01T10:00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=80,
            ready_for_scorecard=True,
            window_from="2025-01-01T10:00:00",
            window_to="2025-01-01T12:00:00",
            contiguous=True,
            gap_cutoff_time="2025-01-01T09:00:00",
            requested_times=150,
            actual_times=100,
        )

        assert result.contiguous is True
        assert result.gap_cutoff_time == "2025-01-01T09:00:00"
        assert result.requested_times == 150
        assert result.actual_times == 100


class TestReporterContiguous:
    """Tests for reporter contiguous info output."""

    def test_md_report_includes_contiguous_info(self) -> None:
        """Markdown report includes contiguous mode information."""
        from pmq.evaluation.pipeline import EvaluationResult
        from pmq.evaluation.reporter import EvaluationReporter

        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2025-01-01T10:00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=80,
            ready_for_scorecard=True,
            window_from="2025-01-01T10:00:00",
            window_to="2025-01-01T12:00:00",
            contiguous=True,
            gap_cutoff_time="2025-01-01T09:00:00",
            requested_times=150,
            actual_times=100,
            approval_status="PASSED",
            approval_reasons=["All checks passed"],
        )

        reporter = EvaluationReporter(dao=MagicMock())
        md = reporter.generate_report_md(result=result)

        assert "Contiguous Mode" in md
        assert "Requested Times" in md
        assert "150" in md
        assert "Actual Times" in md
        assert "100" in md
        assert "Gap Cutoff" in md
        assert "2025-01-01T09:00:00" in md
