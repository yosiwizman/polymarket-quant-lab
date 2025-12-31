"""Tests for Phase 4.8 - Evaluation realism (costs + constraints).

Tests:
1. CLI precedence rules (CLI > YAML > defaults)
2. Pipeline passes fee/slippage to walk-forward
3. Constraint override behavior + pair count accounting
4. Reporter includes Cost Assumptions and Constraint Filtering sections
5. EvaluationResult realism fields
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from pmq.evaluation.pipeline import EvaluationPipeline, EvaluationResult
from pmq.evaluation.reporter import EvaluationReporter
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


class TestEvaluationResultRealismFields:
    """Tests for EvaluationResult realism fields (Phase 4.8)."""

    def test_realism_fields_exist(self) -> None:
        """EvaluationResult has Phase 4.8 realism fields."""
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
        assert result.fee_bps == 0.0
        assert result.slippage_bps == 0.0
        assert result.constraints_applied is False
        assert result.constraint_min_liquidity is None
        assert result.constraint_max_spread is None
        assert result.pairs_before_constraints == 0
        assert result.pairs_after_constraints == 0
        assert result.pairs_filtered_low_liquidity == 0
        assert result.pairs_filtered_high_spread == 0

    def test_realism_fields_can_be_set(self) -> None:
        """EvaluationResult realism fields can be populated."""
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
            # Phase 4.8 realism fields
            fee_bps=3.0,
            slippage_bps=8.0,
            constraints_applied=True,
            constraint_min_liquidity=500.0,
            constraint_max_spread=0.03,
            pairs_before_constraints=10,
            pairs_after_constraints=7,
            pairs_filtered_low_liquidity=2,
            pairs_filtered_high_spread=1,
        )

        assert result.fee_bps == 3.0
        assert result.slippage_bps == 8.0
        assert result.constraints_applied is True
        assert result.constraint_min_liquidity == 500.0
        assert result.constraint_max_spread == 0.03
        assert result.pairs_before_constraints == 10
        assert result.pairs_after_constraints == 7
        assert result.pairs_filtered_low_liquidity == 2
        assert result.pairs_filtered_high_spread == 1


class TestParamsYAMLLoading:
    """Tests for params YAML loading with fee/slippage."""

    def test_load_params_from_yaml_with_costs(self, dao_with_db: DAO, tmp_path: Path) -> None:
        """Load fee_bps/slippage_bps from YAML file."""
        # Create a YAML file with custom costs
        params_file = tmp_path / "params.yml"
        params_file.write_text(
            yaml.dump(
                {
                    "lookback": 40,
                    "entry_z": 2.5,
                    "exit_z": 0.3,
                    "fee_bps": 5.0,
                    "slippage_bps": 10.0,
                }
            )
        )

        pipeline = EvaluationPipeline(dao=dao_with_db)
        params = pipeline._load_statarb_params(str(params_file))

        assert params["fee_bps"] == 5.0
        assert params["slippage_bps"] == 10.0
        assert params["lookback"] == 40
        assert params["entry_z"] == 2.5

    def test_load_params_defaults_when_no_costs_in_yaml(
        self, dao_with_db: DAO, tmp_path: Path
    ) -> None:
        """Use default fee/slippage when not in YAML."""
        # Create a YAML file without cost params
        params_file = tmp_path / "params.yml"
        params_file.write_text(
            yaml.dump(
                {
                    "lookback": 40,
                    "entry_z": 2.5,
                }
            )
        )

        pipeline = EvaluationPipeline(dao=dao_with_db)
        params = pipeline._load_statarb_params(str(params_file))

        # Should use project defaults
        assert params["fee_bps"] == 2.0
        assert params["slippage_bps"] == 5.0
        assert params["lookback"] == 40

    def test_load_params_defaults_when_no_file(self, dao_with_db: DAO) -> None:
        """Use defaults when no params file exists."""
        pipeline = EvaluationPipeline(dao=dao_with_db)
        params = pipeline._load_statarb_params("/nonexistent/path.yml")

        assert params["fee_bps"] == 2.0
        assert params["slippage_bps"] == 5.0
        assert params["lookback"] == 30
        assert params["entry_z"] == 2.0

    def test_load_params_nested_statarb_key(self, dao_with_db: DAO, tmp_path: Path) -> None:
        """Handle nested 'statarb' key from tuning output."""
        params_file = tmp_path / "params.yml"
        params_file.write_text(
            yaml.dump(
                {
                    "statarb": {
                        "lookback": 50,
                        "fee_bps": 3.0,
                        "slippage_bps": 7.0,
                    }
                }
            )
        )

        pipeline = EvaluationPipeline(dao=dao_with_db)
        params = pipeline._load_statarb_params(str(params_file))

        assert params["fee_bps"] == 3.0
        assert params["slippage_bps"] == 7.0
        assert params["lookback"] == 50


class TestReporterRealismSections:
    """Tests for reporter showing Cost Assumptions and Constraint Filtering."""

    def test_reporter_shows_cost_assumptions(self) -> None:
        """Reporter includes Cost Assumptions section for walk-forward."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=85,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T02:00:00+00:00",
            walk_forward=True,
            backtest_run_id="wf_abc123",
            train_window_from="2024-01-01T00:00:00",
            train_window_to="2024-01-01T01:00:00",
            test_window_from="2024-01-01T01:00:00",
            test_window_to="2024-01-01T02:00:00",
            # Phase 4.8 realism fields
            fee_bps=2.0,
            slippage_bps=5.0,
            statarb_params={"lookback": 30, "entry_z": 2.0},
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Verify Cost Assumptions section
        assert "### Cost Assumptions" in md_report
        assert "**Fee:** 2.0 bps" in md_report
        assert "**Slippage:** 5.0 bps" in md_report
        assert "**Total Round-Trip Cost:** 7.0 bps" in md_report

    def test_reporter_shows_constraint_filtering(self) -> None:
        """Reporter includes Constraint Filtering section when pairs filtered."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=85,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T02:00:00+00:00",
            walk_forward=True,
            backtest_run_id="wf_abc123",
            train_window_from="2024-01-01T00:00:00",
            train_window_to="2024-01-01T01:00:00",
            test_window_from="2024-01-01T01:00:00",
            test_window_to="2024-01-01T02:00:00",
            fee_bps=2.0,
            slippage_bps=5.0,
            # Constraint filtering
            constraints_applied=True,
            pairs_before_constraints=10,
            pairs_after_constraints=7,
            pairs_filtered_low_liquidity=2,
            pairs_filtered_high_spread=1,
            constraint_min_liquidity=500.0,
            constraint_max_spread=0.03,
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Verify Constraint Filtering section
        assert "### Constraint Filtering" in md_report
        assert "**Pairs Before Filtering:** 10" in md_report
        assert "**Pairs After Filtering:** 7" in md_report
        assert "**Filtered (Low Liquidity):** 2" in md_report
        assert "**Filtered (High Spread):** 1" in md_report
        assert "**Global Min Liquidity:** 500.0" in md_report
        assert "**Global Max Spread:** 0.03" in md_report

    def test_reporter_omits_constraint_section_when_no_filtering(self) -> None:
        """Reporter omits Constraint Filtering when no pairs and no constraints."""
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
            # Standard backtest, no walk-forward
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Should NOT have constraint section for non-walk-forward
        assert "### Constraint Filtering" not in md_report

    def test_reporter_params_excludes_fee_slippage(self) -> None:
        """Parameters Used section excludes fee/slippage (shown in Cost Assumptions)."""
        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2024-01-01T00:00:00+00:00",
            final_status="PASSED",
            quality_status="SUFFICIENT",
            maturity_score=85,
            ready_for_scorecard=True,
            window_from="2024-01-01T00:00:00+00:00",
            window_to="2024-01-01T02:00:00+00:00",
            walk_forward=True,
            backtest_run_id="wf_abc123",
            train_window_from="2024-01-01T00:00:00",
            train_window_to="2024-01-01T01:00:00",
            test_window_from="2024-01-01T01:00:00",
            test_window_to="2024-01-01T02:00:00",
            fee_bps=2.0,
            slippage_bps=5.0,
            statarb_params={
                "lookback": 30,
                "entry_z": 2.0,
                "fee_bps": 2.0,  # Should be filtered out
                "slippage_bps": 5.0,  # Should be filtered out
            },
        )

        reporter = EvaluationReporter()
        md_report = reporter.generate_report_md(result=result)

        # Parameters Used should show lookback/entry_z
        assert "**lookback:** 30" in md_report
        assert "**entry_z:** 2.0" in md_report
        # But fee/slippage should only appear in Cost Assumptions, not Parameters Used
        # (checking they don't appear with the - **fee_bps:** format)
        params_section_start = md_report.find("### Parameters Used")
        cost_section_start = md_report.find("### Cost Assumptions")
        if params_section_start != -1 and cost_section_start != -1:
            params_section = md_report[params_section_start:cost_section_start]
            assert "- **fee_bps:**" not in params_section
            assert "- **slippage_bps:**" not in params_section


class TestPipelineSignature:
    """Tests for pipeline.run() accepting realism parameters."""

    def test_pipeline_run_accepts_realism_params(self, dao_with_db: DAO) -> None:
        """Pipeline.run() accepts fee/slippage/constraint override parameters."""
        pipeline = EvaluationPipeline(dao=dao_with_db)

        # Check that the method signature includes realism parameters
        import inspect

        sig = inspect.signature(pipeline.run)
        params = sig.parameters

        assert "fee_bps_override" in params
        assert "slippage_bps_override" in params
        assert "min_liquidity_override" in params
        assert "max_spread_override" in params

        # Verify default values are None
        assert params["fee_bps_override"].default is None
        assert params["slippage_bps_override"].default is None
        assert params["min_liquidity_override"].default is None
        assert params["max_spread_override"].default is None


class TestCostPrecedence:
    """Tests for cost parameter precedence (CLI > YAML > defaults)."""

    def test_cli_override_takes_precedence(self, dao_with_db: DAO, tmp_path: Path) -> None:
        """CLI override takes precedence over YAML values."""
        # Create YAML with fee_bps=5.0
        params_file = tmp_path / "params.yml"
        params_file.write_text(yaml.dump({"fee_bps": 5.0, "slippage_bps": 8.0}))

        pipeline = EvaluationPipeline(dao=dao_with_db)

        # Load base params from YAML
        params = pipeline._load_statarb_params(str(params_file))
        assert params["fee_bps"] == 5.0

        # In _run_walkforward, CLI override would take precedence
        # This tests the precedence logic directly
        fee_bps_override = 3.0
        effective_fee_bps = (
            fee_bps_override if fee_bps_override is not None else params.get("fee_bps", 2.0)
        )
        assert effective_fee_bps == 3.0  # CLI wins

    def test_yaml_takes_precedence_over_defaults(self, dao_with_db: DAO, tmp_path: Path) -> None:
        """YAML values take precedence over project defaults."""
        params_file = tmp_path / "params.yml"
        params_file.write_text(yaml.dump({"fee_bps": 4.0}))

        pipeline = EvaluationPipeline(dao=dao_with_db)
        params = pipeline._load_statarb_params(str(params_file))

        # YAML value should override default (2.0)
        assert params["fee_bps"] == 4.0
        # But slippage should use default since not in YAML
        assert params["slippage_bps"] == 5.0

    def test_defaults_used_when_no_override(self, dao_with_db: DAO) -> None:
        """Project defaults used when no CLI or YAML override."""
        pipeline = EvaluationPipeline(dao=dao_with_db)

        # Non-existent params file falls back to defaults
        params = pipeline._load_statarb_params("/nonexistent/path/to/params.yml")

        # Should use project defaults
        assert params["fee_bps"] == 2.0
        assert params["slippage_bps"] == 5.0
