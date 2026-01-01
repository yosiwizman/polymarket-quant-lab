"""Tests for Phase 4.6: Realistic Costs + Market Constraints.

Tests cover:
1. Cost defaults changed (fee_bps=2.0, slippage_bps=5.0)
2. PnL reduction with costs vs no-cost baseline
3. Constraint filtering logic
4. YAML params loading with costs
5. Reporter includes cost info
"""

from pmq.statarb.constraints import (
    ConstraintResult,
    FilterReason,
    apply_market_constraints,
    compute_pair_liquidity,
    compute_pair_spread,
    constraint_result_to_dict,
)
from pmq.statarb.pairs_config import PairConfig
from pmq.statarb.tuning import GridConfig
from pmq.statarb.walkforward import WalkForwardMetrics, compute_metrics_from_signals
from pmq.statarb.zscore import SignalAction, ZScoreSignal

# -----------------------------------------------------------------------------
# Test Cost Defaults
# -----------------------------------------------------------------------------


class TestCostDefaults:
    """Test that cost defaults are realistic (Phase 4.6)."""

    def test_grid_config_defaults_have_costs(self) -> None:
        """GridConfig defaults should have fee_bps=2.0 and slippage_bps=5.0."""
        grid = GridConfig()
        assert grid.fee_bps == [2.0], "Default fee_bps should be [2.0]"
        assert grid.slippage_bps == [5.0], "Default slippage_bps should be [5.0]"

    def test_walkforward_defaults_have_costs(self) -> None:
        """run_walk_forward defaults should have fee_bps=2.0 and slippage_bps=5.0."""
        import inspect

        from pmq.statarb.walkforward import run_walk_forward

        sig = inspect.signature(run_walk_forward)
        assert sig.parameters["fee_bps"].default == 2.0
        assert sig.parameters["slippage_bps"].default == 5.0


# -----------------------------------------------------------------------------
# Test PnL Reduction with Costs
# -----------------------------------------------------------------------------


class TestPnLWithCosts:
    """Test that costs reduce PnL compared to zero-cost baseline."""

    def _create_mock_signals(self) -> list[ZScoreSignal]:
        """Create mock signals for a profitable trade."""
        return [
            ZScoreSignal(
                time="2024-12-01T10:00:00",
                pair_name="TestPair",
                market_a_id="market_a",
                market_b_id="market_b",
                action=SignalAction.ENTER_LONG,
                z_score=2.5,
                spread=0.10,
                price_a=0.50,
                price_b=0.52,
                beta=1.0,
                reason="Entry signal",
            ),
            ZScoreSignal(
                time="2024-12-01T10:05:00",
                pair_name="TestPair",
                market_a_id="market_a",
                market_b_id="market_b",
                action=SignalAction.EXIT,
                z_score=0.3,
                spread=0.15,  # Spread increased (profit for long)
                price_a=0.52,
                price_b=0.54,
                beta=1.0,
                reason="Exit signal",
            ),
        ]

    def test_zero_cost_baseline(self) -> None:
        """Zero cost should give raw PnL."""
        signals = self._create_mock_signals()
        metrics = compute_metrics_from_signals(
            signals=signals,
            fee_bps=0.0,
            slippage_bps=0.0,
            quantity_per_trade=10.0,
        )
        assert metrics.total_trades == 1
        assert metrics.total_fees == 0.0
        # Spread change: 0.15 - 0.10 = 0.05 profit per unit for long
        expected_pnl = 0.05 * 10.0
        assert abs(metrics.total_pnl - expected_pnl) < 0.01

    def test_with_costs_reduces_pnl(self) -> None:
        """Non-zero costs should reduce PnL."""
        signals = self._create_mock_signals()

        # No costs
        metrics_no_cost = compute_metrics_from_signals(
            signals=signals,
            fee_bps=0.0,
            slippage_bps=0.0,
            quantity_per_trade=10.0,
        )

        # With realistic costs
        metrics_with_cost = compute_metrics_from_signals(
            signals=signals,
            fee_bps=2.0,
            slippage_bps=5.0,
            quantity_per_trade=10.0,
        )

        assert metrics_with_cost.total_pnl < metrics_no_cost.total_pnl
        assert metrics_with_cost.total_fees > 0
        assert metrics_with_cost.net_pnl < metrics_no_cost.net_pnl

    def test_higher_costs_reduce_pnl_more(self) -> None:
        """Higher costs should reduce PnL more."""
        signals = self._create_mock_signals()

        metrics_low_cost = compute_metrics_from_signals(
            signals=signals,
            fee_bps=1.0,
            slippage_bps=2.0,
            quantity_per_trade=10.0,
        )

        metrics_high_cost = compute_metrics_from_signals(
            signals=signals,
            fee_bps=10.0,
            slippage_bps=20.0,
            quantity_per_trade=10.0,
        )

        assert metrics_high_cost.total_pnl < metrics_low_cost.total_pnl


# -----------------------------------------------------------------------------
# Test Constraint Filtering
# -----------------------------------------------------------------------------


class TestConstraintFiltering:
    """Test market constraint filtering logic."""

    def _create_test_pair(
        self,
        name: str = "TestPair",
        min_liquidity: float | None = None,
        max_spread: float | None = None,
    ) -> PairConfig:
        """Create a test pair config."""
        return PairConfig(
            market_a_id="market_a",
            market_b_id="market_b",
            name=name,
            min_liquidity=min_liquidity,
            max_spread=max_spread,
        )

    def _create_snapshots_with_liquidity(
        self,
        liquidity: float,
        spread: float,
    ) -> list[dict]:
        """Create snapshots with specified liquidity and spread."""
        times = ["2024-12-01T10:00:00", "2024-12-01T10:01:00"]
        snapshots = []
        for t in times:
            for market_id in ["market_a", "market_b"]:
                mid_price = 0.50
                half_spread = spread * mid_price / 2
                snapshots.append(
                    {
                        "market_id": market_id,
                        "snapshot_time": t,
                        "yes_price": mid_price,
                        "yes_bid": mid_price - half_spread,
                        "yes_ask": mid_price + half_spread,
                        "yes_bid_amount": liquidity / 2,
                        "yes_ask_amount": liquidity / 2,
                    }
                )
        return snapshots

    def test_no_constraints_passes_all(self) -> None:
        """Pairs without constraints should all pass."""
        pairs = [self._create_test_pair("Pair1"), self._create_test_pair("Pair2")]
        snapshots = self._create_snapshots_with_liquidity(1000.0, 0.02)
        times = ["2024-12-01T10:00:00"]

        result = apply_market_constraints(pairs, snapshots, times)

        assert result.eligible_count == 2
        assert len(result.filtered_pairs) == 0
        assert result.constraints_applied

    def test_low_liquidity_filters_pair(self) -> None:
        """Pair with liquidity below threshold should be filtered."""
        pair = self._create_test_pair("LowLiqPair", min_liquidity=5000.0)
        snapshots = self._create_snapshots_with_liquidity(100.0, 0.02)  # Low liquidity
        times = ["2024-12-01T10:00:00"]

        result = apply_market_constraints([pair], snapshots, times)

        assert result.eligible_count == 0
        assert result.filtered_low_liquidity == 1
        assert len(result.filter_reasons) == 1
        assert result.filter_reasons[0].reason == "low_liquidity"

    def test_high_spread_filters_pair(self) -> None:
        """Pair with spread above threshold should be filtered."""
        pair = self._create_test_pair("HighSpreadPair", max_spread=0.01)
        snapshots = self._create_snapshots_with_liquidity(1000.0, 0.05)  # High spread
        times = ["2024-12-01T10:00:00"]

        result = apply_market_constraints([pair], snapshots, times)

        assert result.eligible_count == 0
        assert result.filtered_high_spread == 1
        assert len(result.filter_reasons) == 1
        assert result.filter_reasons[0].reason == "high_spread"

    def test_enforce_constraints_false_skips_filtering(self) -> None:
        """When enforce_constraints=False, all pairs should pass."""
        pair = self._create_test_pair("Pair", min_liquidity=1000000.0)  # Impossible threshold
        snapshots = self._create_snapshots_with_liquidity(100.0, 0.02)
        times = ["2024-12-01T10:00:00"]

        result = apply_market_constraints([pair], snapshots, times, enforce_constraints=False)

        assert result.eligible_count == 1
        assert not result.constraints_applied

    def test_missing_data_passes_pair(self) -> None:
        """Pair should pass if constraint data is not available."""
        pair = self._create_test_pair("Pair", min_liquidity=1000.0)
        # Snapshots without liquidity data
        snapshots = [
            {
                "market_id": "market_a",
                "snapshot_time": "2024-12-01T10:00:00",
                "yes_price": 0.50,
            }
        ]
        times = ["2024-12-01T10:00:00"]

        result = apply_market_constraints([pair], snapshots, times)

        # Should pass because liquidity data not available
        assert result.eligible_count == 1

    def test_constraint_result_to_dict(self) -> None:
        """Test serialization of constraint result."""
        result = ConstraintResult(
            total_pairs=3,
            eligible_count=2,
            constraints_applied=True,
            filtered_low_liquidity=1,
            filter_reasons=[
                FilterReason(
                    pair_name="BadPair",
                    reason="low_liquidity",
                    threshold=1000.0,
                    actual_value=100.0,
                )
            ],
        )

        d = constraint_result_to_dict(result)

        assert d["total_pairs"] == 3
        assert d["eligible_count"] == 2
        assert d["filtered_low_liquidity"] == 1
        assert len(d["filter_reasons"]) == 1
        assert d["filter_reasons"][0]["pair_name"] == "BadPair"


# -----------------------------------------------------------------------------
# Test YAML Params Loading
# -----------------------------------------------------------------------------


class TestYAMLParamsLoading:
    """Test loading statarb params from YAML with cost fields."""

    def test_parse_params_with_costs(self, tmp_path) -> None:
        """YAML with cost fields should be parsed correctly."""
        from pmq.evaluation.pipeline import EvaluationPipeline

        yaml_content = """
statarb:
  lookback: 25
  entry_z: 1.8
  exit_z: 0.4
  max_hold_bars: 45
  cooldown_bars: 3
  fee_bps: 3.5
  slippage_bps: 8.0
"""
        yaml_path = tmp_path / "params.yml"
        yaml_path.write_text(yaml_content)

        defaults = {"fee_bps": 2.0, "slippage_bps": 5.0, "lookback": 30}
        pipeline = EvaluationPipeline()
        params = pipeline._parse_params_yaml(yaml_path, defaults)

        assert params["fee_bps"] == 3.5
        assert params["slippage_bps"] == 8.0
        assert params["lookback"] == 25

    def test_parse_params_defaults_costs(self, tmp_path) -> None:
        """YAML without cost fields should use defaults."""
        from pmq.evaluation.pipeline import EvaluationPipeline

        yaml_content = """
statarb:
  lookback: 25
  entry_z: 1.8
"""
        yaml_path = tmp_path / "params.yml"
        yaml_path.write_text(yaml_content)

        defaults = {"fee_bps": 2.0, "slippage_bps": 5.0, "lookback": 30}
        pipeline = EvaluationPipeline()
        params = pipeline._parse_params_yaml(yaml_path, defaults)

        assert params["fee_bps"] == 2.0  # Default
        assert params["slippage_bps"] == 5.0  # Default
        assert params["lookback"] == 25  # From YAML


# -----------------------------------------------------------------------------
# Test Liquidity and Spread Computation
# -----------------------------------------------------------------------------


class TestLiquiditySpreadComputation:
    """Test liquidity and spread computation helpers."""

    def test_compute_pair_liquidity(self) -> None:
        """Test liquidity computation."""
        snapshots = [
            {
                "market_id": "A",
                "snapshot_time": "t1",
                "yes_bid_amount": 100.0,
                "yes_ask_amount": 150.0,
            },
            {
                "market_id": "B",
                "snapshot_time": "t1",
                "yes_bid_amount": 200.0,
                "yes_ask_amount": 250.0,
            },
        ]
        times = ["t1"]

        # Phase 4.9: compute_pair_liquidity returns (liquidity, used_microstructure)
        liquidity, used_micro = compute_pair_liquidity(snapshots, "A", "B", times)

        # Average of (100+150) and (200+250) = (250 + 450) / 2 = 350
        assert liquidity is not None
        assert abs(liquidity - 350.0) < 0.01
        assert used_micro is False  # No microstructure data in legacy test

    def test_compute_pair_spread(self) -> None:
        """Test spread computation."""
        snapshots = [
            {
                "market_id": "A",
                "snapshot_time": "t1",
                "yes_bid": 0.48,
                "yes_ask": 0.52,
            },
        ]
        times = ["t1"]

        # Phase 4.9: compute_pair_spread returns (spread, used_microstructure)
        spread, used_micro = compute_pair_spread(snapshots, "A", "B", times)

        # spread = (0.52 - 0.48) / 0.50 = 0.08
        assert spread is not None
        assert abs(spread - 0.08) < 0.01
        assert used_micro is False  # No microstructure data in legacy test

    def test_compute_liquidity_no_data(self) -> None:
        """Should return None if no liquidity data."""
        snapshots = [{"market_id": "A", "snapshot_time": "t1", "yes_price": 0.50}]
        times = ["t1"]

        # Phase 4.9: returns (None, False) when no data
        liquidity, used_micro = compute_pair_liquidity(snapshots, "A", "B", times)
        assert liquidity is None
        assert used_micro is False


# -----------------------------------------------------------------------------
# Test WalkForward Integration
# -----------------------------------------------------------------------------


class TestWalkForwardConstraintIntegration:
    """Test constraints are integrated into walk-forward."""

    def test_walkforward_result_has_constraint_field(self) -> None:
        """WalkForwardResult should have constraint_result field."""
        import dataclasses

        from pmq.statarb.walkforward import WalkForwardResult

        fields = {f.name for f in dataclasses.fields(WalkForwardResult)}
        assert "constraint_result" in fields

    def test_result_to_dict_includes_constraints(self) -> None:
        """result_to_dict should include constraint info if present."""
        from pmq.statarb.constraints import ConstraintResult
        from pmq.statarb.walkforward import (
            WalkForwardResult,
            WalkForwardSplit,
            result_to_dict,
        )

        result = WalkForwardResult(
            split=WalkForwardSplit(
                train_times=[],
                test_times=[],
                train_count=0,
                test_count=0,
                total_count=0,
                first_train="",
                last_train="",
                first_test="",
                last_test="",
            ),
            fitted_params={},
            signals=[],
            train_metrics=None,
            test_metrics=WalkForwardMetrics(
                total_pnl=0,
                sharpe_ratio=0,
                win_rate=0,
                max_drawdown=0,
                total_trades=0,
                avg_trades_per_pair=0,
                total_fees=0,
                net_pnl=0,
                entry_count=0,
                exit_count=0,
            ),
            pair_summaries=[],
            config_used={},
            constraint_result=ConstraintResult(
                total_pairs=5,
                eligible_count=3,
                constraints_applied=True,
                filtered_low_liquidity=2,
            ),
        )

        d = result_to_dict(result)

        assert "constraints" in d
        assert d["constraints"]["total_pairs"] == 5
        assert d["constraints"]["eligible_count"] == 3
        assert d["constraints"]["filtered_low_liquidity"] == 2
