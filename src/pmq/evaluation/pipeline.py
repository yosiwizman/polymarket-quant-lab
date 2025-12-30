"""Evaluation pipeline orchestrates end-to-end strategy validation.

Executes:
1. Quality check (assert data is ready)
2. Backtest run
3. Approval evaluation
4. Optional paper trading smoke test
5. Final report generation
"""

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pmq.backtest.runner import BacktestRunner
from pmq.governance.scorecard import compute_scorecard
from pmq.logging import get_logger
from pmq.quality import QualityChecker, QualityResult
from pmq.storage.dao import DAO

logger = get_logger("evaluation.pipeline")


def _get_git_sha() -> str | None:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


@dataclass
class EvaluationResult:
    """Result of an evaluation pipeline run."""

    eval_id: str
    strategy_name: str
    strategy_version: str
    created_at: str
    final_status: str  # PASSED, FAILED, PENDING

    # Quality results
    quality_status: str
    maturity_score: int
    ready_for_scorecard: bool
    window_from: str
    window_to: str

    # Backtest results
    backtest_run_id: str | None = None
    backtest_pnl: float = 0.0
    backtest_score: float = 0.0

    # Approval results
    approval_status: str = "PENDING"
    approval_reasons: list[str] = field(default_factory=list)

    # Paper run results
    paper_run_id: str | None = None
    paper_trades_count: int = 0
    paper_errors_count: int = 0

    # Summary
    summary: str = ""
    commands: list[str] = field(default_factory=list)


class EvaluationPipeline:
    """Orchestrates end-to-end strategy evaluation.

    Steps:
    1. Check data quality (must be READY)
    2. Run deterministic backtest
    3. Evaluate for approval (standard mode)
    4. Optional: Run paper trading smoke test
    5. Generate final report
    """

    def __init__(self, dao: DAO | None = None) -> None:
        """Initialize pipeline.

        Args:
            dao: Data access object
        """
        self._dao = dao or DAO()
        self._quality_checker = QualityChecker(dao=self._dao)
        self._commands: list[str] = []

    def run(
        self,
        strategy_name: str,
        strategy_version: str,
        window_mode: str = "last_times",
        window_value: int = 30,
        interval_seconds: int = 60,
        paper_minutes: int | None = None,
        quantity: float = 10.0,
        initial_balance: float = 10000.0,
        window_from: str | None = None,
        window_to: str | None = None,
        pairs_config: str | None = None,
    ) -> EvaluationResult:
        """Run the full evaluation pipeline.

        Args:
            strategy_name: Strategy to evaluate (arb, statarb, observer)
            strategy_version: Version string (e.g., v1)
            window_mode: Quality window mode (last_times, last_minutes, explicit)
            window_value: N for last_times/last_minutes modes
            interval_seconds: Expected snapshot interval
            paper_minutes: Optional paper trading duration (0 to skip)
            quantity: Trade quantity for backtest
            initial_balance: Starting balance for backtest
            window_from: Explicit window start (for explicit mode)
            window_to: Explicit window end (for explicit mode)
            pairs_config: Path to pairs config file (required for statarb)

        Returns:
            EvaluationResult with all step outcomes
        """
        eval_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        git_sha = _get_git_sha()

        logger.info(f"Starting evaluation {eval_id}: {strategy_name} v{strategy_version}")

        # Early check: statarb requires pairs config
        pairs_config_result = None
        if strategy_name == "statarb":
            if not pairs_config:
                raise ValueError(
                    "statarb strategy requires --pairs config. "
                    "Run 'pmq statarb pairs suggest' to generate one, or "
                    "provide an existing config with --pairs config/pairs.yml"
                )
            # Validate pairs config
            from pathlib import Path

            from pmq.statarb import PairsConfigError, load_validated_pairs_config

            try:
                pairs_config_result = load_validated_pairs_config(Path(pairs_config))
            except PairsConfigError as e:
                raise ValueError(f"Invalid pairs config: {e}") from e

            if not pairs_config_result.has_enabled_pairs:
                raise ValueError(
                    f"No enabled pairs in {pairs_config}. "
                    "Enable at least one pair or run 'pmq statarb pairs suggest'."
                )

        # Create evaluation record
        self._dao.create_evaluation_run(
            eval_id=eval_id,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            window_mode=window_mode,
            interval_seconds=interval_seconds,
            window_value=window_value if window_mode != "explicit" else None,
            window_from=window_from,
            window_to=window_to,
            git_sha=git_sha,
        )

        # Initialize result
        result = EvaluationResult(
            eval_id=eval_id,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            created_at=created_at,
            final_status="PENDING",
            quality_status="UNKNOWN",
            maturity_score=0,
            ready_for_scorecard=False,
            window_from="",
            window_to="",
        )

        # Step 1: Quality check
        self._log_command(
            f"pmq snapshots quality --{window_mode.replace('_', '-')} {window_value} "
            f"--interval {interval_seconds}"
        )

        quality_result = self._check_quality(
            window_mode=window_mode,
            window_value=window_value,
            interval_seconds=interval_seconds,
            window_from=window_from,
            window_to=window_to,
        )

        result.quality_status = quality_result.status
        result.maturity_score = quality_result.maturity_score
        result.ready_for_scorecard = quality_result.ready_for_scorecard
        result.window_from = quality_result.window_from
        result.window_to = quality_result.window_to

        # Update DB with quality results
        self._dao.update_evaluation_quality(
            eval_id=eval_id,
            quality_status=quality_result.status,
            maturity_score=quality_result.maturity_score,
            ready_for_scorecard=quality_result.ready_for_scorecard,
            window_from=quality_result.window_from,
            window_to=quality_result.window_to,
        )

        # Save quality artifact
        self._dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="QUALITY_JSON",
            content=json.dumps(self._quality_to_dict(quality_result)),
        )

        # Check if ready
        if not quality_result.ready_for_scorecard:
            result.final_status = "FAILED"
            result.summary = (
                f"Data not ready for evaluation. "
                f"Maturity: {quality_result.maturity_score}/100 (need 70+). "
                f"Status: {quality_result.status}"
            )
            result.commands = self._commands
            self._dao.complete_evaluation(
                eval_id=eval_id,
                final_status="FAILED",
                summary=result.summary,
                commands=self._commands,
            )
            logger.warning(f"Evaluation {eval_id} failed: data not ready")
            return result

        # Step 2: Run backtest
        backtest_cmd = (
            f"pmq backtest run --strategy {strategy_name} "
            f"--from {quality_result.window_from[:10]} --to {quality_result.window_to[:10]} "
            f"--balance {initial_balance} --quantity {quantity}"
        )
        if pairs_config:
            backtest_cmd += f" --pairs {pairs_config}"
        self._log_command(backtest_cmd)

        backtest_result = self._run_backtest(
            strategy_name=strategy_name,
            start_date=quality_result.window_from,
            end_date=quality_result.window_to,
            quantity=quantity,
            initial_balance=initial_balance,
            pairs_config=pairs_config,
        )

        # Save pairs config artifact if used
        if pairs_config_result:
            self._dao.save_evaluation_artifact(
                evaluation_id=eval_id,
                kind="PAIRS_CONFIG_JSON",
                content=json.dumps({
                    "path": pairs_config_result.config_path,
                    "hash": pairs_config_result.config_hash,
                    "enabled_pairs": len(pairs_config_result.enabled_pairs),
                    "disabled_pairs": len(pairs_config_result.disabled_pairs),
                    "pairs": [p.to_dict() for p in pairs_config_result.enabled_pairs],
                }),
            )

        result.backtest_run_id = backtest_result["run_id"]
        result.backtest_pnl = backtest_result["metrics"].total_pnl
        result.backtest_score = 0.0  # Will be set by scorecard

        # Update DB with backtest results
        self._dao.update_evaluation_backtest(
            eval_id=eval_id,
            backtest_run_id=backtest_result["run_id"],
            backtest_pnl=backtest_result["metrics"].total_pnl,
            backtest_score=0.0,
        )

        # Save backtest artifact
        self._dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="BACKTEST_JSON",
            content=json.dumps(
                {
                    "run_id": backtest_result["run_id"],
                    "metrics": backtest_result["metrics"].__dict__,
                }
            ),
        )

        # Step 3: Approval evaluation (standard mode)
        self._log_command(f"pmq approve evaluate --run-id {backtest_result['run_id']}")

        scorecard = self._evaluate_approval(
            metrics=backtest_result["metrics"],
            quality_result=quality_result,
            initial_balance=initial_balance,
            validation_mode=(strategy_name == "observer"),
        )

        result.backtest_score = scorecard.score
        result.approval_status = "PASSED" if scorecard.passed else "FAILED"
        result.approval_reasons = scorecard.reasons

        # Update DB with approval results
        self._dao.update_evaluation_backtest(
            eval_id=eval_id,
            backtest_run_id=backtest_result["run_id"],
            backtest_pnl=backtest_result["metrics"].total_pnl,
            backtest_score=scorecard.score,
        )
        self._dao.update_evaluation_approval(
            eval_id=eval_id,
            approval_status=result.approval_status,
            approval_reasons=scorecard.reasons,
        )

        # Save scorecard artifact
        scorecard_txt = self._format_scorecard_txt(scorecard)
        self._dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="SCORECARD_TXT",
            content=scorecard_txt,
        )

        # Check if approval passed
        if not scorecard.passed:
            result.final_status = "FAILED"
            result.summary = (
                f"Approval failed. Score: {scorecard.score:.1f}/100. "
                f"Reasons: {'; '.join(r for r in scorecard.reasons if r.startswith('FAIL'))}"
            )
            result.commands = self._commands
            self._dao.complete_evaluation(
                eval_id=eval_id,
                final_status="FAILED",
                summary=result.summary,
                commands=self._commands,
            )
            logger.warning(f"Evaluation {eval_id} failed: approval not passed")
            return result

        # Step 4: Optional paper trading smoke test
        if paper_minutes and paper_minutes > 0:
            self._log_command(
                f"pmq paper run --strategy {strategy_name} --minutes {paper_minutes} "
                f"--quantity {quantity}"
            )

            paper_result = self._run_paper_smoke_test(
                strategy_name=strategy_name,
                minutes=paper_minutes,
                quantity=quantity,
            )

            result.paper_run_id = paper_result.get("run_id")
            result.paper_trades_count = paper_result.get("trades_count", 0)
            result.paper_errors_count = paper_result.get("errors_count", 0)

            # Update DB with paper results
            self._dao.update_evaluation_paper(
                eval_id=eval_id,
                paper_run_id=result.paper_run_id,
                paper_trades_count=result.paper_trades_count,
                paper_errors_count=result.paper_errors_count,
            )

            # Save paper log artifact
            self._dao.save_evaluation_artifact(
                evaluation_id=eval_id,
                kind="PAPER_LOG",
                content=json.dumps(paper_result),
            )

        # Step 5: Final status
        result.final_status = "PASSED"
        result.summary = (
            f"Evaluation PASSED. "
            f"Score: {scorecard.score:.1f}/100, PnL: ${result.backtest_pnl:.2f}, "
            f"Maturity: {result.maturity_score}/100"
        )
        if paper_minutes:
            result.summary += f", Paper trades: {result.paper_trades_count}"

        result.commands = self._commands

        # Complete evaluation
        self._dao.complete_evaluation(
            eval_id=eval_id,
            final_status="PASSED",
            summary=result.summary,
            commands=self._commands,
        )

        logger.info(f"Evaluation {eval_id} completed: PASSED")
        return result

    def _log_command(self, cmd: str) -> None:
        """Log a command that would be executed."""
        self._commands.append(cmd)

    def _check_quality(
        self,
        window_mode: str,
        window_value: int,
        interval_seconds: int,
        window_from: str | None = None,
        window_to: str | None = None,
    ) -> QualityResult:
        """Check data quality based on window mode."""
        if window_mode == "last_times":
            return self._quality_checker.check_last_times(
                limit=window_value,
                expected_interval_seconds=interval_seconds,
            )
        elif window_mode == "last_minutes":
            return self._quality_checker.check_last_minutes(
                minutes=window_value,
                expected_interval_seconds=interval_seconds,
            )
        else:
            # Explicit mode
            if not window_from or not window_to:
                raise ValueError("window_from and window_to required for explicit mode")
            return self._quality_checker.check_window(
                start_time=window_from,
                end_time=window_to,
                expected_interval_seconds=interval_seconds,
            )

    def _run_backtest(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        quantity: float,
        initial_balance: float,
        pairs_config: str | None = None,
    ) -> dict[str, Any]:
        """Run backtest for the strategy."""
        runner = BacktestRunner(dao=self._dao, initial_balance=initial_balance)

        if strategy_name == "arb":
            run_id, metrics = runner.run_arb_backtest(
                start_date=start_date,
                end_date=end_date,
                quantity=quantity,
            )
        elif strategy_name == "statarb":
            run_id, metrics = runner.run_statarb_backtest(
                start_date=start_date,
                end_date=end_date,
                pairs_config=pairs_config,
                quantity=quantity,
            )
        elif strategy_name == "observer":
            # Observer doesn't trade, just returns empty metrics
            run_id, metrics = runner.run_arb_backtest(
                start_date=start_date,
                end_date=end_date,
                quantity=0,  # No trades
                threshold=0.0,  # Impossible threshold
            )
        else:
            # Default to arb for unknown strategies
            run_id, metrics = runner.run_arb_backtest(
                start_date=start_date,
                end_date=end_date,
                quantity=quantity,
            )

        return {"run_id": run_id, "metrics": metrics}

    def _evaluate_approval(
        self,
        metrics: Any,
        quality_result: QualityResult,
        initial_balance: float,
        validation_mode: bool = False,
    ) -> Any:
        """Evaluate strategy for approval."""
        return compute_scorecard(
            total_pnl=metrics.total_pnl,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            sharpe_ratio=metrics.sharpe_ratio,
            total_trades=metrics.total_trades,
            trades_per_day=metrics.trades_per_day,
            capital_utilization=metrics.capital_utilization,
            initial_balance=initial_balance,
            data_quality_pct=quality_result.coverage_pct,
            validation_mode=validation_mode,
            data_quality_status=quality_result.status,
            maturity_score=quality_result.maturity_score,
            ready_for_scorecard=quality_result.ready_for_scorecard,
        )

    def _run_paper_smoke_test(
        self,
        strategy_name: str,
        minutes: int,
        quantity: float,
    ) -> dict[str, Any]:
        """Run a short paper trading smoke test.

        Note: This is a simplified version that just records intent.
        Full paper trading integration would require the paper module.
        """
        # For Phase 4, we just record that a paper run was requested
        # Full integration would call the paper trading loop
        run_id = f"paper_{uuid.uuid4().hex[:8]}"

        return {
            "run_id": run_id,
            "strategy": strategy_name,
            "minutes": minutes,
            "quantity": quantity,
            "trades_count": 0,  # Placeholder
            "errors_count": 0,
            "note": "Paper smoke test placeholder - full integration in future phase",
        }

    def _quality_to_dict(self, result: QualityResult) -> dict[str, Any]:
        """Convert QualityResult to dict for JSON serialization."""
        return {
            "window_from": result.window_from,
            "window_to": result.window_to,
            "window_mode": result.window_mode,
            "status": result.status,
            "maturity_score": result.maturity_score,
            "ready_for_scorecard": result.ready_for_scorecard,
            "coverage_pct": result.coverage_pct,
            "distinct_times": result.distinct_times,
            "expected_times": result.expected_times,
            "missing_intervals": result.missing_intervals,
            "duplicate_count": result.duplicate_count,
            "largest_gap_seconds": result.largest_gap_seconds,
        }

    def _format_scorecard_txt(self, scorecard: Any) -> str:
        """Format scorecard as human-readable text."""
        lines = [
            "=" * 60,
            "STRATEGY SCORECARD",
            "=" * 60,
            f"Score: {scorecard.score:.1f}/100",
            f"Status: {'PASSED' if scorecard.passed else 'FAILED'}",
            "",
            "Reasons:",
        ]
        for reason in scorecard.reasons:
            lines.append(f"  - {reason}")

        if scorecard.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in scorecard.warnings:
                lines.append(f"  - {warning}")

        lines.append("")
        lines.append("Metrics Used:")
        for key, value in scorecard.metrics_used.items():
            if value is not None:
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)
