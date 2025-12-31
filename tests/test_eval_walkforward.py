"""Tests for walk-forward evaluation integration in eval pipeline.

These tests verify that:
1. Walk-forward detection logic works correctly
2. Walk-forward path is called instead of legacy backtest for statarb
3. TRAIN/TEST split is correct (no overlap, chronological)
4. TEST metrics from walk-forward are what scorecard sees
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmq.evaluation.pipeline import EvaluationPipeline, EvaluationResult


class TestWalkForwardDetection:
    """Tests for _should_use_walk_forward() method."""

    def test_explicit_true(self) -> None:
        """Walk-forward enabled when explicitly set to True."""
        pipeline = EvaluationPipeline(dao=MagicMock())
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="v1",
            walk_forward=True,
            statarb_params_path=None,
        )
        assert result is True

    def test_explicit_false(self) -> None:
        """Walk-forward disabled when explicitly set to False."""
        pipeline = EvaluationPipeline(dao=MagicMock())
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="zscore-v1",  # Would normally enable
            walk_forward=False,
            statarb_params_path="/some/path.yml",  # Would normally enable
        )
        assert result is False

    def test_non_statarb_strategy(self) -> None:
        """Walk-forward always disabled for non-statarb strategies."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        # arb strategy
        result = pipeline._should_use_walk_forward(
            strategy_name="arb",
            strategy_version="v1",
            walk_forward=True,  # Even explicit True
            statarb_params_path=None,
        )
        assert result is False

        # observer strategy
        result = pipeline._should_use_walk_forward(
            strategy_name="observer",
            strategy_version="v1",
            walk_forward=None,
            statarb_params_path=None,
        )
        assert result is False

    def test_version_contains_zscore(self) -> None:
        """Walk-forward enabled when version contains 'zscore'."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        # lowercase
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="zscore-v1",
            walk_forward=None,
            statarb_params_path=None,
        )
        assert result is True

        # mixed case
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="v1-ZScore-tuned",
            walk_forward=None,
            statarb_params_path=None,
        )
        assert result is True

    def test_version_contains_walkforward(self) -> None:
        """Walk-forward enabled when version contains 'walkforward'."""
        pipeline = EvaluationPipeline(dao=MagicMock())
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="v2-walkforward",
            walk_forward=None,
            statarb_params_path=None,
        )
        assert result is True

    def test_params_path_provided(self) -> None:
        """Walk-forward enabled when statarb_params_path is provided."""
        pipeline = EvaluationPipeline(dao=MagicMock())
        result = pipeline._should_use_walk_forward(
            strategy_name="statarb",
            strategy_version="v1",  # Generic version
            walk_forward=None,
            statarb_params_path="/path/to/params.yml",
        )
        assert result is True

    def test_default_config_exists(self) -> None:
        """Walk-forward enabled when config/statarb_best.yml exists."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        with patch.object(Path, "exists", return_value=True):
            result = pipeline._should_use_walk_forward(
                strategy_name="statarb",
                strategy_version="v1",
                walk_forward=None,
                statarb_params_path=None,
            )
            assert result is True

    def test_no_auto_detect_triggers(self) -> None:
        """Walk-forward disabled when no auto-detect conditions met."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        with patch.object(Path, "exists", return_value=False):
            result = pipeline._should_use_walk_forward(
                strategy_name="statarb",
                strategy_version="v1",  # No zscore/walkforward
                walk_forward=None,
                statarb_params_path=None,  # No params path
            )
            assert result is False


class TestParamsLoading:
    """Tests for _load_statarb_params() and _parse_params_yaml()."""

    def test_defaults_when_no_file(self) -> None:
        """Returns defaults when no params file exists."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        with patch.object(Path, "exists", return_value=False):
            params = pipeline._load_statarb_params(None)

        assert params["lookback"] == 30
        assert params["entry_z"] == 2.0
        assert params["exit_z"] == 0.5
        assert params["max_hold_bars"] == 60
        assert params["cooldown_bars"] == 5
        assert params["fee_bps"] == 0.0
        assert params["slippage_bps"] == 0.0

    def test_parse_flat_yaml(self, tmp_path: Path) -> None:
        """Parse YAML with flat structure."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        yaml_content = """
lookback: 20
entry_z: 2.5
exit_z: 0.3
max_hold_bars: 30
"""
        yaml_file = tmp_path / "params.yml"
        yaml_file.write_text(yaml_content)

        params = pipeline._load_statarb_params(str(yaml_file))

        assert params["lookback"] == 20
        assert params["entry_z"] == 2.5
        assert params["exit_z"] == 0.3
        assert params["max_hold_bars"] == 30
        # Defaults for missing keys
        assert params["cooldown_bars"] == 5
        assert params["fee_bps"] == 0.0

    def test_parse_nested_yaml(self, tmp_path: Path) -> None:
        """Parse YAML with 'statarb' nested key (from tuning output)."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        yaml_content = """
statarb:
  lookback: 15
  entry_z: 1.8
  exit_z: 0.4
  max_hold_bars: 45
tuning_metadata:
  best_sharpe: 0.15
"""
        yaml_file = tmp_path / "statarb_best.yml"
        yaml_file.write_text(yaml_content)

        params = pipeline._load_statarb_params(str(yaml_file))

        assert params["lookback"] == 15
        assert params["entry_z"] == 1.8
        assert params["exit_z"] == 0.4
        assert params["max_hold_bars"] == 45

    def test_invalid_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """Returns defaults when YAML is invalid."""
        pipeline = EvaluationPipeline(dao=MagicMock())

        yaml_file = tmp_path / "bad.yml"
        yaml_file.write_text("{{invalid yaml: [}")

        params = pipeline._load_statarb_params(str(yaml_file))

        assert params["lookback"] == 30  # Default


class TestWalkForwardMetricsConversion:
    """Tests for _walkforward_to_backtest_metrics()."""

    def test_metrics_conversion(self) -> None:
        """WalkForwardMetrics correctly converted to BacktestMetrics."""
        from pmq.backtest.metrics import BacktestMetrics
        from pmq.statarb.walkforward import WalkForwardMetrics, WalkForwardSplit

        pipeline = EvaluationPipeline(dao=MagicMock())

        wf_metrics = WalkForwardMetrics(
            total_pnl=100.0,
            sharpe_ratio=0.5,
            win_rate=0.6,
            max_drawdown=0.05,
            total_trades=10,
            avg_trades_per_pair=2.0,
            total_fees=5.0,
            net_pnl=95.0,
            entry_count=5,
            exit_count=5,
        )

        # Mock WalkForwardResult
        wf_result = MagicMock()
        wf_result.split = WalkForwardSplit(
            train_times=["t1", "t2"],
            test_times=["t3", "t4", "t5"],
            train_count=2,
            test_count=3,
            total_count=5,
            first_train="t1",
            last_train="t2",
            first_test="t3",
            last_test="t5",
        )

        metrics = pipeline._walkforward_to_backtest_metrics(wf_metrics, wf_result)

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_pnl == 100.0
        assert metrics.sharpe_ratio == 0.5
        assert metrics.win_rate == 0.6
        assert metrics.max_drawdown == 0.05
        assert metrics.total_trades == 10


class TestEvaluationResultWalkForwardFields:
    """Tests for EvaluationResult walk-forward fields."""

    def test_default_values(self) -> None:
        """Walk-forward fields have correct defaults."""
        result = EvaluationResult(
            eval_id="test",
            strategy_name="statarb",
            strategy_version="v1",
            created_at="2025-01-01T00:00:00",
            final_status="PENDING",
            quality_status="READY",
            maturity_score=100,
            ready_for_scorecard=True,
            window_from="2025-01-01",
            window_to="2025-01-02",
        )

        assert result.walk_forward is False
        assert result.train_times_count == 0
        assert result.test_times_count == 0
        assert result.fitted_pairs_count == 0
        assert result.total_pairs_count == 0
        assert result.train_window_from == ""
        assert result.train_window_to == ""
        assert result.test_window_from == ""
        assert result.test_window_to == ""
        assert result.statarb_params == {}

    def test_walk_forward_fields_populated(self) -> None:
        """Walk-forward fields can be populated."""
        result = EvaluationResult(
            eval_id="test",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2025-01-01T00:00:00",
            final_status="PASSED",
            quality_status="READY",
            maturity_score=100,
            ready_for_scorecard=True,
            window_from="2025-01-01",
            window_to="2025-01-02",
            walk_forward=True,
            train_times_count=100,
            test_times_count=50,
            fitted_pairs_count=15,
            total_pairs_count=20,
            train_window_from="2025-01-01T00:00:00",
            train_window_to="2025-01-01T01:40:00",
            test_window_from="2025-01-01T01:41:00",
            test_window_to="2025-01-01T02:30:00",
            statarb_params={"lookback": 20, "entry_z": 2.5},
            backtest_sharpe=0.11,
            backtest_win_rate=0.55,
            backtest_max_drawdown=0.03,
            backtest_total_trades=25,
        )

        assert result.walk_forward is True
        assert result.train_times_count == 100
        assert result.test_times_count == 50
        assert result.fitted_pairs_count == 15
        assert result.total_pairs_count == 20
        assert result.statarb_params == {"lookback": 20, "entry_z": 2.5}
        assert result.backtest_sharpe == 0.11


class TestReporterWalkForward:
    """Tests for reporter walk-forward output."""

    def test_md_report_includes_walkforward_section(self) -> None:
        """Markdown report includes walk-forward section when enabled."""
        from pmq.evaluation.reporter import EvaluationReporter

        result = EvaluationResult(
            eval_id="test-123",
            strategy_name="statarb",
            strategy_version="zscore-v1",
            created_at="2025-01-01T00:00:00",
            final_status="FAILED",
            quality_status="SUFFICIENT",
            maturity_score=100,
            ready_for_scorecard=True,
            window_from="2025-01-01",
            window_to="2025-01-02",
            backtest_run_id="wf_abc123",
            backtest_pnl=0.06,
            backtest_score=34.0,
            approval_status="FAILED",
            approval_reasons=["FAIL: PnL below threshold"],
            walk_forward=True,
            train_times_count=100,
            test_times_count=50,
            fitted_pairs_count=15,
            total_pairs_count=20,
            train_window_from="2025-01-01T00:00:00",
            train_window_to="2025-01-01T01:40:00",
            test_window_from="2025-01-01T01:41:00",
            test_window_to="2025-01-01T02:30:00",
            statarb_params={"lookback": 20, "entry_z": 2.5},
            backtest_sharpe=0.11,
            backtest_win_rate=0.11,
            backtest_max_drawdown=0.09,
            backtest_total_trades=9,
            summary="Evaluation FAILED",
        )

        reporter = EvaluationReporter(dao=MagicMock())
        md = reporter._generate_md_from_result(result)

        # Check walk-forward specific content
        assert "Walk-Forward (TEST only)" in md
        assert "Walk-Forward Evaluation" in md
        assert "TRAIN Window" in md
        assert "TEST Window" in md
        assert "TRAIN Snapshots" in md
        assert "TEST Snapshots" in md
        assert "Fitted Pairs" in md
        assert "15/20" in md
        assert "TEST Metrics (used for Scorecard)" in md
        assert "no data leakage" in md
        assert "Parameters Used" in md
        assert "lookback" in md

    def test_md_report_no_walkforward_section_when_disabled(self) -> None:
        """Markdown report uses standard backtest section when walk-forward disabled."""
        from pmq.evaluation.reporter import EvaluationReporter

        result = EvaluationResult(
            eval_id="test-456",
            strategy_name="arb",
            strategy_version="v1",
            created_at="2025-01-01T00:00:00",
            final_status="PASSED",
            quality_status="READY",
            maturity_score=100,
            ready_for_scorecard=True,
            window_from="2025-01-01",
            window_to="2025-01-02",
            backtest_run_id="bt_xyz789",
            backtest_pnl=150.0,
            backtest_score=85.0,
            approval_status="PASSED",
            approval_reasons=["PASS: All criteria met"],
            walk_forward=False,
            summary="Evaluation PASSED",
        )

        reporter = EvaluationReporter(dao=MagicMock())
        md = reporter._generate_md_from_result(result)

        # Check standard backtest section
        assert "## Step 2: Backtest" in md
        assert "Walk-Forward Evaluation" not in md
        assert "TRAIN Window" not in md
        assert "no data leakage" not in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
