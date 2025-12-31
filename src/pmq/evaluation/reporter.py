"""Evaluation reporter generates deterministic go/no-go reports."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from pmq.evaluation.pipeline import EvaluationResult
from pmq.storage.dao import DAO


class EvaluationReporter:
    """Generates evaluation reports in various formats."""

    def __init__(self, dao: DAO | None = None) -> None:
        """Initialize reporter.

        Args:
            dao: Data access object
        """
        self._dao = dao or DAO()

    def get_evaluation(self, eval_id: str) -> dict[str, Any] | None:
        """Get evaluation run from database.

        Args:
            eval_id: Evaluation ID

        Returns:
            Evaluation data dict or None
        """
        return self._dao.get_evaluation_run(eval_id)

    def get_artifacts(self, eval_id: str) -> list[dict[str, Any]]:
        """Get evaluation artifacts from database.

        Args:
            eval_id: Evaluation ID

        Returns:
            List of artifact dicts
        """
        return self._dao.get_evaluation_artifacts(eval_id)

    def generate_report_md(
        self,
        eval_data: dict[str, Any] | None = None,
        eval_id: str | None = None,
        result: EvaluationResult | None = None,
    ) -> str:
        """Generate markdown report for an evaluation.

        Args:
            eval_data: Evaluation data dict from DB
            eval_id: Evaluation ID (will fetch from DB)
            result: EvaluationResult object

        Returns:
            Markdown report string
        """
        if result:
            return self._generate_md_from_result(result)

        if eval_id and not eval_data:
            eval_data = self.get_evaluation(eval_id)

        if not eval_data:
            return "# Evaluation Not Found\n\nNo evaluation data available."

        return self._generate_md_from_db(eval_data)

    def generate_report_json(
        self,
        eval_data: dict[str, Any] | None = None,
        eval_id: str | None = None,
        result: EvaluationResult | None = None,
    ) -> str:
        """Generate JSON report for an evaluation.

        Args:
            eval_data: Evaluation data dict from DB
            eval_id: Evaluation ID (will fetch from DB)
            result: EvaluationResult object

        Returns:
            JSON report string
        """
        if result:
            return json.dumps(asdict(result), indent=2, default=str)

        if eval_id and not eval_data:
            eval_data = self.get_evaluation(eval_id)

        if not eval_data:
            return json.dumps({"error": "Evaluation not found"})

        # Include artifacts
        artifacts = self.get_artifacts(eval_data["id"])
        report = {
            "evaluation": eval_data,
            "artifacts": artifacts,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        return json.dumps(report, indent=2, default=str)

    def generate_report_csv(
        self,
        eval_data: dict[str, Any] | None = None,
        eval_id: str | None = None,
    ) -> str:
        """Generate CSV report for an evaluation.

        Args:
            eval_data: Evaluation data dict from DB
            eval_id: Evaluation ID (will fetch from DB)

        Returns:
            CSV report string
        """
        if eval_id and not eval_data:
            eval_data = self.get_evaluation(eval_id)

        if not eval_data:
            return "error,Evaluation not found"

        # CSV header and single row
        headers = [
            "id",
            "created_at",
            "strategy_name",
            "strategy_version",
            "final_status",
            "quality_status",
            "maturity_score",
            "backtest_pnl",
            "backtest_score",
            "approval_status",
            "paper_trades_count",
            "summary",
        ]

        values = [str(eval_data.get(h, "")) for h in headers]

        return ",".join(headers) + "\n" + ",".join(values)

    def list_evaluations(
        self,
        limit: int = 20,
        strategy_name: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent evaluations.

        Args:
            limit: Max number to return
            strategy_name: Filter by strategy
            status: Filter by final_status

        Returns:
            List of evaluation dicts
        """
        return self._dao.get_evaluation_runs(
            limit=limit,
            strategy_name=strategy_name,
            final_status=status,
        )

    def _generate_md_from_result(self, result: EvaluationResult) -> str:
        """Generate markdown from EvaluationResult object."""
        status_emoji = "✅" if result.final_status == "PASSED" else "❌"

        lines = [
            f"# Evaluation Report {status_emoji}",
            "",
            f"**ID:** `{result.eval_id}`",
            f"**Strategy:** {result.strategy_name} v{result.strategy_version}",
            f"**Created:** {result.created_at}",
            f"**Final Status:** **{result.final_status}**",
        ]

        if result.walk_forward:
            lines.append("**Mode:** Walk-Forward (TEST only)")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Summary",
                "",
                result.summary,
                "",
                "---",
                "",
                "## Step 1: Data Quality",
                "",
                f"- **Status:** {result.quality_status}",
                f"- **Maturity Score:** {result.maturity_score}/100",
                f"- **Ready for Scorecard:** {'Yes' if result.ready_for_scorecard else 'No'}",
                f"- **Window:** {result.window_from} → {result.window_to}",
            ]
        )

        # Add contiguous info (Phase 4.5)
        if result.contiguous:
            lines.append("- **Contiguous Mode:** Yes")
            lines.append(f"- **Requested Times:** {result.requested_times}")
            lines.append(f"- **Actual Times:** {result.actual_times}")
            if result.gap_cutoff_time:
                lines.append(f"- **Gap Cutoff:** {result.gap_cutoff_time}")

        # Add effective window quality info (Phase 4.7)
        if result.quality_window_aligned:
            lines.extend(
                [
                    "",
                    "### Effective Window Quality (Aligned with Walk-Forward)",
                    "",
                    f"- **Effective Window:** {result.effective_window_from[:19]} → {result.effective_window_to[:19]}",
                    f"- **Expected Points:** {result.effective_expected_points}",
                    f"- **Observed Points:** {result.effective_observed_points}",
                    f"- **Quality Pct:** {result.effective_quality_pct:.1f}%",
                    "",
                    "*Quality re-evaluated on the exact window used by walk-forward (TRAIN+TEST)*",
                ]
            )

        lines.append("")

        if result.backtest_run_id:
            if result.walk_forward:
                lines.extend(
                    [
                        "## Step 2: Walk-Forward Evaluation",
                        "",
                        f"- **Run ID:** `{result.backtest_run_id}`",
                        f"- **TRAIN Window:** {result.train_window_from} → {result.train_window_to}",
                        f"- **TEST Window:** {result.test_window_from} → {result.test_window_to}",
                        f"- **TRAIN Snapshots:** {result.train_times_count}",
                        f"- **TEST Snapshots:** {result.test_times_count}",
                        f"- **Fitted Pairs:** {result.fitted_pairs_count}/{result.total_pairs_count}",
                        "",
                        "### TEST Metrics (used for Scorecard)",
                        "",
                        f"- **PnL:** ${result.backtest_pnl:.2f}",
                        f"- **Sharpe:** {result.backtest_sharpe:.3f}",
                        f"- **Win Rate:** {result.backtest_win_rate:.1%}",
                        f"- **Max Drawdown:** {result.backtest_max_drawdown:.2%}",
                        f"- **Total Trades:** {result.backtest_total_trades}",
                        f"- **Score:** {result.backtest_score:.1f}/100",
                        "",
                    ]
                )
                if result.statarb_params:
                    lines.extend(
                        [
                            "### Parameters Used",
                            "",
                        ]
                    )
                    for key, value in result.statarb_params.items():
                        # Skip fee/slippage in params since we show them in Cost Assumptions
                        if key not in ("fee_bps", "slippage_bps"):
                            lines.append(f"- **{key}:** {value}")
                    lines.append("")

                # Phase 4.8: Enhanced Cost Assumptions section
                lines.extend(
                    [
                        "### Cost Assumptions",
                        "",
                        f"- **Fee:** {result.fee_bps} bps",
                        f"- **Slippage:** {result.slippage_bps} bps",
                        f"- **Total Round-Trip Cost:** {result.fee_bps + result.slippage_bps} bps",
                        "",
                    ]
                )

                # Phase 4.8: Constraint Filtering section
                if result.constraints_applied or result.pairs_before_constraints > 0:
                    lines.extend(
                        [
                            "### Constraint Filtering",
                            "",
                        ]
                    )
                    lines.append(f"- **Pairs Before Filtering:** {result.pairs_before_constraints}")
                    lines.append(f"- **Pairs After Filtering:** {result.pairs_after_constraints}")
                    if result.pairs_filtered_low_liquidity > 0:
                        lines.append(
                            f"- **Filtered (Low Liquidity):** {result.pairs_filtered_low_liquidity}"
                        )
                    if result.pairs_filtered_high_spread > 0:
                        lines.append(
                            f"- **Filtered (High Spread):** {result.pairs_filtered_high_spread}"
                        )
                    if result.constraint_min_liquidity is not None:
                        lines.append(
                            f"- **Global Min Liquidity:** {result.constraint_min_liquidity}"
                        )
                    if result.constraint_max_spread is not None:
                        lines.append(f"- **Global Max Spread:** {result.constraint_max_spread}")
                    if not result.constraints_applied:
                        lines.append("- **Note:** No constraints applied (all pairs passed)")
                    lines.append("")

                lines.append(
                    "*Note: Scorecard evaluated on TEST only (walk-forward, no data leakage)*"
                )
                lines.append("")
            else:
                lines.extend(
                    [
                        "## Step 2: Backtest",
                        "",
                        f"- **Run ID:** `{result.backtest_run_id}`",
                        f"- **PnL:** ${result.backtest_pnl:.2f}",
                        f"- **Score:** {result.backtest_score:.1f}/100",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Step 3: Approval",
                "",
                f"- **Status:** {result.approval_status}",
                "",
                "**Reasons:**",
                "",
            ]
        )

        for reason in result.approval_reasons:
            lines.append(f"- {reason}")

        if result.paper_run_id:
            lines.extend(
                [
                    "",
                    "## Step 4: Paper Trading",
                    "",
                    f"- **Run ID:** `{result.paper_run_id}`",
                    f"- **Trades:** {result.paper_trades_count}",
                    f"- **Errors:** {result.paper_errors_count}",
                    "",
                ]
            )

        if result.commands:
            lines.extend(
                [
                    "---",
                    "",
                    "## Commands Executed",
                    "",
                    "```bash",
                ]
            )
            for cmd in result.commands:
                lines.append(cmd)
            lines.extend(
                [
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "---",
                "",
                f"*Report generated at {datetime.now(UTC).isoformat()}*",
            ]
        )

        return "\n".join(lines)

    def _generate_md_from_db(self, eval_data: dict[str, Any]) -> str:
        """Generate markdown from database evaluation record."""
        final_status = eval_data.get("final_status", "UNKNOWN")
        status_emoji = "✅" if final_status == "PASSED" else "❌"

        lines = [
            f"# Evaluation Report {status_emoji}",
            "",
            f"**ID:** `{eval_data.get('id', 'N/A')}`",
            f"**Strategy:** {eval_data.get('strategy_name', 'N/A')} v{eval_data.get('strategy_version', 'N/A')}",
            f"**Created:** {eval_data.get('created_at', 'N/A')}",
            f"**Git SHA:** `{eval_data.get('git_sha', 'N/A')}`",
            f"**Final Status:** **{final_status}**",
            "",
            "---",
            "",
            "## Summary",
            "",
            eval_data.get("summary", "No summary available."),
            "",
            "---",
            "",
            "## Step 1: Data Quality",
            "",
            f"- **Window Mode:** {eval_data.get('window_mode', 'N/A')}",
            f"- **Status:** {eval_data.get('quality_status', 'N/A')}",
            f"- **Maturity Score:** {eval_data.get('maturity_score', 0)}/100",
            f"- **Ready for Scorecard:** {'Yes' if eval_data.get('ready_for_scorecard') else 'No'}",
            f"- **Window:** {eval_data.get('window_from', 'N/A')} → {eval_data.get('window_to', 'N/A')}",
            "",
        ]

        backtest_run_id = eval_data.get("backtest_run_id")
        if backtest_run_id:
            lines.extend(
                [
                    "## Step 2: Backtest",
                    "",
                    f"- **Run ID:** `{backtest_run_id}`",
                    f"- **PnL:** ${eval_data.get('backtest_pnl', 0):.2f}",
                    f"- **Score:** {eval_data.get('backtest_score', 0):.1f}/100",
                    "",
                ]
            )

        approval_status = eval_data.get("approval_status", "PENDING")
        lines.extend(
            [
                "## Step 3: Approval",
                "",
                f"- **Status:** {approval_status}",
                "",
            ]
        )

        approval_reasons = eval_data.get("approval_reasons_json")
        if approval_reasons:
            try:
                reasons = json.loads(approval_reasons)
                if reasons:
                    lines.append("**Reasons:**")
                    lines.append("")
                    for reason in reasons:
                        lines.append(f"- {reason}")
                    lines.append("")
            except json.JSONDecodeError:
                pass

        paper_run_id = eval_data.get("paper_run_id")
        if paper_run_id:
            lines.extend(
                [
                    "## Step 4: Paper Trading",
                    "",
                    f"- **Run ID:** `{paper_run_id}`",
                    f"- **Trades:** {eval_data.get('paper_trades_count', 0)}",
                    f"- **Errors:** {eval_data.get('paper_errors_count', 0)}",
                    "",
                ]
            )

        commands_json = eval_data.get("commands_json")
        if commands_json:
            try:
                commands = json.loads(commands_json)
                if commands:
                    lines.extend(
                        [
                            "---",
                            "",
                            "## Commands Executed",
                            "",
                            "```bash",
                        ]
                    )
                    for cmd in commands:
                        lines.append(cmd)
                    lines.extend(
                        [
                            "```",
                            "",
                        ]
                    )
            except json.JSONDecodeError:
                pass

        lines.extend(
            [
                "---",
                "",
                f"*Report generated at {datetime.now(UTC).isoformat()}*",
            ]
        )

        return "\n".join(lines)

    def save_report_to_db(
        self,
        eval_id: str,
        result: EvaluationResult | None = None,
    ) -> None:
        """Save MD and JSON reports as artifacts.

        Args:
            eval_id: Evaluation ID
            result: Optional EvaluationResult object
        """
        eval_data = self.get_evaluation(eval_id) if not result else None

        # Generate and save MD report
        md_report = self.generate_report_md(eval_data=eval_data, result=result)
        self._dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="REPORT_MD",
            content=md_report,
        )

        # Generate and save JSON report
        json_report = self.generate_report_json(eval_data=eval_data, result=result)
        self._dao.save_evaluation_artifact(
            evaluation_id=eval_id,
            kind="REPORT_JSON",
            content=json_report,
        )
