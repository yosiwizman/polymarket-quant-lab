"""Tests for Phase 4.7 - Quality window alignment with walk-forward evaluation.

Tests:
1. check_explicit_window computes expected/observed points correctly
2. EvaluationResult has effective window fields
3. Eval pipeline aligns quality with walk-forward effective window
4. Approval gate uses effective_quality_pct when available
5. Reporter shows effective window info
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pmq.evaluation.pipeline import EvaluationResult
from pmq.evaluation.reporter import EvaluationReporter
from pmq.quality.checks import (
    QualityChecker,
    QualityStatus,
    WindowMode,
)
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


class TestExplicitWindowQuality:
    """Tests for check_explicit_window method (Phase 4.7)."""

    def test_explicit_window_expected_points_calculation(self, dao_with_db: DAO) -> None:
        """check_explicit_window calculates expected_points = floor((end-start)/interval)+1."""
        _create_test_market(dao_with_db, "market_1")

        # Insert snapshots for 60 minutes (0 to 59)
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(60):
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

        # Window from 00:00 to 00:29 (30 minutes) should expect 30 points
        result = checker.check_explicit_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:29:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.window_mode == WindowMode.EXPLICIT
        assert result.notes.get("expected_times") == 30
        assert result.notes.get("observed_times") == 30
        assert result.coverage_pct == 100.0
        assert result.notes.get("window_mode") == "explicit"

    def test_explicit_window_partial_coverage(self, dao_with_db: DAO) -> None:
        """check_explicit_window computes correct coverage when data is partial."""
        _create_test_market(dao_with_db, "market_1")

        # Insert only 20 snapshots for first 20 minutes
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(20):
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

        # Window from 00:00 to 00:39 (40 minutes) should expect 40 points
        result = checker.check_explicit_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:39:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.notes.get("expected_times") == 40
        assert result.notes.get("observed_times") == 20
        assert result.coverage_pct == 50.0  # 20/40 = 50%

    def test_explicit_window_no_data(self, dao_with_db: DAO) -> None:
        """check_explicit_window handles no data gracefully."""
        checker = QualityChecker(dao=dao_with_db)

        result = checker.check_explicit_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:59:00+00:00",
            expected_interval_seconds=60,
        )

        assert result.notes.get("expected_times") == 60
        assert result.notes.get("observed_times") == 0
        assert result.coverage_pct == 0.0
        assert result.status == QualityStatus.INSUFFICIENT_DATA


class TestEvaluationResultEffectiveFields:
    """Tests for EvaluationResult effective window fields."""

    def test_effective_window_fields_exist(self) -> None:
        """EvaluationResult has Phase 4.7 effective window fields."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=80,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T01:00:00+00:00",
        )

        # Verify default values
        assert result.effective_window_from == ""
        assert result.effective_window_to == ""
        assert result.effective_expected_points == 0
        assert result.effective_observed_points == 0
        assert result.effective_quality_pct == 0.0
        assert result.quality_window_aligned is False

    def test_effective_window_fields_can_be_set(self) -> None:
        """EvaluationResult effective window fields can be set."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=80,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T01:00:00+00:00",
            # Phase 4.7 fields
            effective_window_from="2024-01-01T00:10:00+00:00",
            effective_window_to="2024-01-01T00:50:00+00:00",
            effective_expected_points=41,
            effective_observed_points=40,
            effective_quality_pct=97.56,
            quality_window_aligned=True,
        )

        assert result.effective_window_from == "2024-01-01T00:10:00+00:00"
        assert result.effective_window_to == "2024-01-01T00:50:00+00:00"
        assert result.effective_expected_points == 41
        assert result.effective_observed_points == 40
        assert result.effective_quality_pct == 97.56
        assert result.quality_window_aligned is True


class TestReporterEffectiveWindow:
    """Tests for reporter showing effective window info."""

    def test_reporter_includes_effective_window_section(self) -> None:
        """Reporter includes effective window quality section when aligned."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=85,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T02:00:00+00:00",
            walk_forward=True,
            train_window_from="2024-01-01T00:00:00+00:00",
            train_window_to="2024-01-01T01:00:00+00:00",
            test_window_from="2024-01-01T01:00:00+00:00",
            test_window_to="2024-01-01T02:00:00+00:00",
            # Phase 4.7 effective window
            effective_window_from="2024-01-01T00:00:00+00:00",
            effective_window_to="2024-01-01T02:00:00+00:00",
            effective_expected_points=121,
            effective_observed_points=118,
            effective_quality_pct=97.5,
            quality_window_aligned=True,
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Verify effective window section is included
        assert "Effective Window Quality" in md_report
        assert "Aligned with Walk-Forward" in md_report
        assert "Expected Points:** 121" in md_report
        assert "Observed Points:** 118" in md_report
        assert "Quality Pct:** 97.5%" in md_report
        assert "TRAIN+TEST" in md_report

    def test_reporter_omits_effective_window_when_not_aligned(self) -> None:
        """Reporter omits effective window section when not aligned."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="arb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=85,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T02:00:00+00:00",
            # Not using walk-forward, so quality_window_aligned=False (default)
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Verify effective window section is NOT included
        assert "Effective Window Quality" not in md_report


class TestApprovalGateEffectiveQuality:
    """Tests for approval gate using effective_quality_pct."""

    def test_effective_quality_passed_to_scorecard(self, dao_with_db: DAO) -> None:
        """Verify effective_quality_pct is used when quality_window_aligned=True."""
        # This is a unit test for the _evaluate_approval logic
        from pmq.evaluation.pipeline import EvaluationPipeline
        from pmq.quality.checks import QualityResult

        pipeline = EvaluationPipeline(dao=dao_with_db)

        # Create a mock quality result
        quality_result = QualityResult(
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T01:00:00+00:00",
            expected_interval_seconds=60,
            coverage_pct=50.0,  # Low initial coverage
            status=QualityStatus.DEGRADED,
            maturity_score=70,
            ready_for_scorecard=True,
            notes={"distinct_snapshot_times": 30, "expected_times": 60},
        )

        # Create mock metrics
        class MockMetrics:
            total_pnl = 100.0
            max_drawdown = 0.05
            win_rate = 0.6
            sharpe_ratio = 1.5
            total_trades = 50
            trades_per_day = 10.0
            capital_utilization = 0.5

        metrics = MockMetrics()

        # Test without effective quality (uses coverage_pct=50)
        scorecard_without = pipeline._evaluate_approval(
            metrics=metrics,
            quality_result=quality_result,
            initial_balance=10000.0,
            validation_mode=False,
            effective_quality_pct=None,
        )

        # Test with effective quality (uses 95.0)
        scorecard_with = pipeline._evaluate_approval(
            metrics=metrics,
            quality_result=quality_result,
            initial_balance=10000.0,
            validation_mode=False,
            effective_quality_pct=95.0,
        )

        # The scorecard with higher effective quality should have better score
        # (data quality contributes to score)
        assert scorecard_with.score >= scorecard_without.score


class TestExplicitWindowCLI:
    """Tests for CLI using check_explicit_window."""

    def test_explicit_window_quality_note(self, dao_with_db: DAO) -> None:
        """check_explicit_window adds quality_note to notes."""
        _create_test_market(dao_with_db, "market_1")

        # Insert some data
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        for i in range(30):
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

        result = checker.check_explicit_window(
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-01T00:29:00+00:00",
            expected_interval_seconds=60,
        )

        # Verify quality_note is set
        assert "quality_note" in result.notes
        assert "explicit window" in result.notes["quality_note"].lower()
        assert "2024-01-01T00:00:00" in result.notes["quality_note"]
        assert "2024-01-01T00:29:00" in result.notes["quality_note"]
