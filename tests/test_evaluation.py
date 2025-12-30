"""Tests for evaluation pipeline (Phase 4).

Tests cover:
- EvaluationPipeline orchestration
- EvaluationReporter output formats
- Artifact persistence
- Fail-fast behavior when data not ready
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmq.evaluation import EvaluationPipeline, EvaluationReporter, EvaluationResult
from pmq.quality import QualityResult
from pmq.storage import DAO
from pmq.storage.db import Database


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating an evaluation result."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="arb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00",
            final_status="PASSED",
            quality_status="HEALTHY",
            maturity_score=90,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            backtest_run_id="bt-123",
            backtest_pnl=100.0,
            backtest_score=75.0,
            approval_status="PASSED",
            approval_reasons=["PASS: PnL positive"],
        )

        assert result.eval_id == "test-123"
        assert result.final_status == "PASSED"
        assert result.maturity_score == 90
        assert result.ready_for_scorecard is True

    def test_result_defaults(self) -> None:
        """Test EvaluationResult has reasonable defaults."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="arb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00",
            final_status="PENDING",
            quality_status="UNKNOWN",
            maturity_score=0,
            ready_for_scorecard=False,
            window_from="",
            window_to="",
        )

        assert result.backtest_run_id is None
        assert result.backtest_pnl == 0.0
        assert result.approval_status == "PENDING"
        assert result.approval_reasons == []
        assert result.commands == []


class TestEvaluationReporter:
    """Tests for EvaluationReporter."""

    @pytest.fixture
    def temp_db(self) -> Database:
        """Create a temporary database for testing."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        db = Database(db_path=Path(tmp.name))
        db.initialize()
        return db

    @pytest.fixture
    def dao(self, temp_db: Database) -> DAO:
        """Create a DAO with test database."""
        return DAO(db=temp_db)

    def test_generate_report_md_from_result(self, dao: DAO) -> None:
        """Test generating markdown report from EvaluationResult."""
        reporter = EvaluationReporter(dao=dao)

        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="arb",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00",
            final_status="PASSED",
            quality_status="HEALTHY",
            maturity_score=90,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            backtest_run_id="bt-123",
            backtest_pnl=100.0,
            backtest_score=75.0,
            approval_status="PASSED",
            approval_reasons=["PASS: PnL positive", "PASS: Win rate above threshold"],
            summary="Evaluation PASSED. Score: 75.0/100",
            commands=["pmq snapshots quality --last-times 30 --interval 60"],
        )

        md = reporter.generate_report_md(result=result)

        assert "# Evaluation Report" in md
        assert "test-123" in md
        assert "arb" in md
        assert "PASSED" in md
        assert "Maturity Score" in md or "maturity" in md.lower()
        assert "Step 1: Data Quality" in md
        assert "Step 2: Backtest" in md
        assert "Step 3: Approval" in md

    def test_generate_report_json_from_result(self, dao: DAO) -> None:
        """Test generating JSON report from EvaluationResult."""
        reporter = EvaluationReporter(dao=dao)

        result = EvaluationResult(
            eval_id="test-456",
            strategy_name="observer",
            strategy_version="v1",
            created_at="2024-01-01T00:00:00",
            final_status="FAILED",
            quality_status="DEGRADED",
            maturity_score=50,
            ready_for_scorecard=False,
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            summary="Data not ready for evaluation.",
        )

        json_str = reporter.generate_report_json(result=result)
        data = json.loads(json_str)

        assert data["eval_id"] == "test-456"
        assert data["final_status"] == "FAILED"
        assert data["maturity_score"] == 50
        assert data["ready_for_scorecard"] is False

    def test_generate_report_csv(self, dao: DAO) -> None:
        """Test generating CSV report."""
        reporter = EvaluationReporter(dao=dao)

        # Create a mock evaluation in DB
        dao.create_evaluation_run(
            eval_id="csv-test-123",
            strategy_name="arb",
            strategy_version="v1",
            window_mode="last_times",
            interval_seconds=60,
            window_value=30,
        )
        dao.complete_evaluation(
            eval_id="csv-test-123",
            final_status="PASSED",
            summary="Test evaluation",
            commands=["pmq eval run"],
        )

        csv = reporter.generate_report_csv(eval_id="csv-test-123")

        # Should have header row and data row
        lines = csv.strip().split("\n")
        assert len(lines) == 2
        assert "id" in lines[0]
        assert "final_status" in lines[0]
        assert "csv-test-123" in lines[1]

    def test_list_evaluations(self, dao: DAO) -> None:
        """Test listing evaluations."""
        reporter = EvaluationReporter(dao=dao)

        # Create some evaluations
        for i in range(5):
            dao.create_evaluation_run(
                eval_id=f"list-test-{i}",
                strategy_name="arb" if i % 2 == 0 else "observer",
                strategy_version="v1",
                window_mode="last_times",
                interval_seconds=60,
            )
            dao.complete_evaluation(
                eval_id=f"list-test-{i}",
                final_status="PASSED" if i < 3 else "FAILED",
                summary=f"Test {i}",
                commands=[],
            )

        # List all
        all_evals = reporter.list_evaluations(limit=10)
        assert len(all_evals) == 5

        # Filter by strategy
        arb_evals = reporter.list_evaluations(strategy_name="arb")
        assert all(ev["strategy_name"] == "arb" for ev in arb_evals)

        # Filter by status
        passed_evals = reporter.list_evaluations(status="PASSED")
        assert all(ev["final_status"] == "PASSED" for ev in passed_evals)


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline."""

    @pytest.fixture
    def temp_db(self) -> Database:
        """Create a temporary database for testing."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        db = Database(db_path=Path(tmp.name))
        db.initialize()
        return db

    @pytest.fixture
    def dao(self, temp_db: Database) -> DAO:
        """Create a DAO with test database."""
        return DAO(db=temp_db)

    def test_pipeline_fails_when_not_ready(self, dao: DAO) -> None:
        """Test pipeline fails early when data quality is not ready."""
        pipeline = EvaluationPipeline(dao=dao)

        # Mock the quality checker to return not ready
        mock_quality = QualityResult(
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            expected_interval_seconds=60,
            window_mode="last_times",
            status="DEGRADED",
            coverage_pct=50.0,
            missing_intervals=15,
            duplicate_count=0,
            largest_gap_seconds=300.0,
            markets_seen=100,
            snapshots_written=1500,
            gaps=[],
            maturity_score=40,  # Below 70 threshold
            ready_for_scorecard=False,
            notes={"distinct_snapshot_times": 15, "expected_times": 30},
        )

        with patch.object(pipeline._quality_checker, "check_last_times", return_value=mock_quality):
            result = pipeline.run(
                strategy_name="arb",
                strategy_version="v1",
                window_mode="last_times",
                window_value=30,
                interval_seconds=60,
            )

        assert result.final_status == "FAILED"
        assert result.ready_for_scorecard is False
        assert "not ready" in result.summary.lower() or "maturity" in result.summary.lower()
        # Backtest should not have been run
        assert result.backtest_run_id is None

    def test_pipeline_creates_quality_artifact(self, dao: DAO) -> None:
        """Test pipeline saves quality artifact when data is not ready."""
        pipeline = EvaluationPipeline(dao=dao)

        # Mock quality as not ready - this will fail early and still create artifacts
        mock_quality = QualityResult(
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            expected_interval_seconds=60,
            window_mode="last_times",
            status="DEGRADED",
            coverage_pct=50.0,
            missing_intervals=15,
            duplicate_count=0,
            largest_gap_seconds=300.0,
            markets_seen=100,
            snapshots_written=1500,
            gaps=[],
            maturity_score=40,
            ready_for_scorecard=False,
            notes={"distinct_snapshot_times": 15, "expected_times": 30},
        )

        with patch.object(pipeline._quality_checker, "check_last_times", return_value=mock_quality):
            result = pipeline.run(
                strategy_name="arb",
                strategy_version="v1",
                window_mode="last_times",
                window_value=30,
                interval_seconds=60,
            )

        # Check quality artifact was saved even though evaluation failed early
        artifacts = dao.get_evaluation_artifacts(result.eval_id)
        artifact_kinds = {a["kind"] for a in artifacts}

        assert "QUALITY_JSON" in artifact_kinds

        # Verify quality artifact content
        quality_artifact = next(a for a in artifacts if a["kind"] == "QUALITY_JSON")
        quality_data = json.loads(quality_artifact["content"])
        assert quality_data["coverage_pct"] == 50.0
        assert quality_data["maturity_score"] == 40

    def test_pipeline_records_commands(self, dao: DAO) -> None:
        """Test pipeline records executed commands."""
        pipeline = EvaluationPipeline(dao=dao)

        mock_quality = QualityResult(
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
            expected_interval_seconds=60,
            window_mode="last_times",
            status="DEGRADED",
            coverage_pct=50.0,
            missing_intervals=15,
            duplicate_count=0,
            largest_gap_seconds=300.0,
            markets_seen=100,
            snapshots_written=1500,
            gaps=[],
            maturity_score=40,
            ready_for_scorecard=False,
            notes={"distinct_snapshot_times": 15, "expected_times": 30},
        )

        with patch.object(pipeline._quality_checker, "check_last_times", return_value=mock_quality):
            result = pipeline.run(
                strategy_name="arb",
                strategy_version="v1",
                window_mode="last_times",
                window_value=30,
                interval_seconds=60,
            )

        # Commands should be recorded
        assert len(result.commands) > 0
        assert any("quality" in cmd for cmd in result.commands)


class TestEvaluationDAO:
    """Tests for evaluation DAO methods."""

    @pytest.fixture
    def temp_db(self) -> Database:
        """Create a temporary database for testing."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        db = Database(db_path=Path(tmp.name))
        db.initialize()
        return db

    @pytest.fixture
    def dao(self, temp_db: Database) -> DAO:
        """Create a DAO with test database."""
        return DAO(db=temp_db)

    def test_create_and_get_evaluation(self, dao: DAO) -> None:
        """Test creating and retrieving an evaluation run."""
        eval_id = "dao-test-123"

        dao.create_evaluation_run(
            eval_id=eval_id,
            strategy_name="arb",
            strategy_version="v1",
            window_mode="last_times",
            interval_seconds=60,
            window_value=30,
            git_sha="abc123",
        )

        # Retrieve
        ev = dao.get_evaluation_run(eval_id)
        assert ev is not None
        assert ev["id"] == eval_id
        assert ev["strategy_name"] == "arb"
        assert ev["window_mode"] == "last_times"
        assert ev["git_sha"] == "abc123"
        assert ev["final_status"] == "PENDING"

    def test_update_evaluation_quality(self, dao: DAO) -> None:
        """Test updating evaluation with quality results."""
        eval_id = "quality-update-test"

        dao.create_evaluation_run(
            eval_id=eval_id,
            strategy_name="arb",
            strategy_version="v1",
            window_mode="last_times",
            interval_seconds=60,
        )

        dao.update_evaluation_quality(
            eval_id=eval_id,
            quality_status="HEALTHY",
            maturity_score=90,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00",
            window_to="2024-01-01T01:00:00",
        )

        ev = dao.get_evaluation_run(eval_id)
        assert ev["quality_status"] == "HEALTHY"
        assert ev["maturity_score"] == 90
        assert ev["ready_for_scorecard"] == 1

    def test_save_and_get_artifacts(self, dao: DAO) -> None:
        """Test saving and retrieving artifacts."""
        eval_id = "artifact-test"

        dao.create_evaluation_run(
            eval_id=eval_id,
            strategy_name="arb",
            strategy_version="v1",
            window_mode="last_times",
            interval_seconds=60,
        )

        # Save artifacts
        dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="QUALITY_JSON",
            content='{"coverage": 100}',
        )
        dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="REPORT_MD",
            content="# Report",
        )

        # Retrieve
        artifacts = dao.get_evaluation_artifacts(eval_id)
        assert len(artifacts) == 2

        kinds = {a["kind"] for a in artifacts}
        assert "QUALITY_JSON" in kinds
        assert "REPORT_MD" in kinds

    def test_complete_evaluation(self, dao: DAO) -> None:
        """Test completing an evaluation."""
        eval_id = "complete-test"

        dao.create_evaluation_run(
            eval_id=eval_id,
            strategy_name="arb",
            strategy_version="v1",
            window_mode="last_times",
            interval_seconds=60,
        )

        dao.complete_evaluation(
            eval_id=eval_id,
            final_status="PASSED",
            summary="All checks passed",
            commands=["pmq eval run", "pmq eval report"],
        )

        ev = dao.get_evaluation_run(eval_id)
        assert ev["final_status"] == "PASSED"
        assert ev["summary"] == "All checks passed"

        # Commands should be JSON
        commands = json.loads(ev["commands_json"])
        assert len(commands) == 2

    def test_filter_evaluations_by_status(self, dao: DAO) -> None:
        """Test filtering evaluations by status."""
        # Create evaluations with different statuses
        for i, status in enumerate(["PASSED", "PASSED", "FAILED", "PENDING"]):
            eval_id = f"filter-test-{i}"
            dao.create_evaluation_run(
                eval_id=eval_id,
                strategy_name="arb",
                strategy_version="v1",
                window_mode="last_times",
                interval_seconds=60,
            )
            if status != "PENDING":
                dao.complete_evaluation(
                    eval_id=eval_id,
                    final_status=status,
                    summary=f"Test {status}",
                    commands=[],
                )

        passed = dao.get_evaluation_runs(final_status="PASSED")
        assert len(passed) == 2

        failed = dao.get_evaluation_runs(final_status="FAILED")
        assert len(failed) == 1
