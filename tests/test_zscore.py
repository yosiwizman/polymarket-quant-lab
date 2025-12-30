"""Tests for z-score engine, walk-forward, and tuning modules.

All tests use deterministic fixture data to ensure reproducibility.
"""

import pytest

from pmq.statarb.pairs_config import PairConfig
from pmq.statarb.tuning import GridConfig, generate_param_combinations
from pmq.statarb.walkforward import (
    WalkForwardMetrics,
    compute_metrics_from_signals,
    extract_pair_prices,
    split_times,
)
from pmq.statarb.zscore import (
    FittedParams,
    SignalAction,
    ZScoreSignal,
    fit_pair_params,
    generate_signals,
    ols_beta,
    rolling_mean_std,
    spread_series,
    zscore_series,
)


# =============================================================================
# OLS Beta Tests
# =============================================================================


class TestOlsBeta:
    """Tests for OLS beta (hedge ratio) computation."""

    def test_perfect_correlation(self) -> None:
        """Beta should be 1.0 for perfectly correlated prices."""
        prices_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        prices_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        beta = ols_beta(prices_a, prices_b)
        assert abs(beta - 1.0) < 1e-10

    def test_double_prices(self) -> None:
        """Beta should be 2.0 when A = 2*B."""
        prices_a = [2.0, 4.0, 6.0, 8.0, 10.0]
        prices_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        beta = ols_beta(prices_a, prices_b)
        assert abs(beta - 2.0) < 1e-10

    def test_half_prices(self) -> None:
        """Beta should be 0.5 when A = 0.5*B."""
        prices_a = [0.5, 1.0, 1.5, 2.0, 2.5]
        prices_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        beta = ols_beta(prices_a, prices_b)
        assert abs(beta - 0.5) < 1e-10

    def test_constant_prices_b(self) -> None:
        """Beta should be 1.0 when B is constant (zero variance)."""
        prices_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        prices_b = [2.0, 2.0, 2.0, 2.0, 2.0]
        beta = ols_beta(prices_a, prices_b)
        assert beta == 1.0

    def test_insufficient_data(self) -> None:
        """Beta should be 1.0 with less than 3 data points."""
        assert ols_beta([1.0, 2.0], [1.0, 2.0]) == 1.0
        assert ols_beta([1.0], [1.0]) == 1.0
        assert ols_beta([], []) == 1.0

    def test_mismatched_lengths(self) -> None:
        """Beta should be 1.0 with mismatched lengths."""
        assert ols_beta([1.0, 2.0, 3.0], [1.0, 2.0]) == 1.0

    def test_deterministic(self) -> None:
        """OLS beta should be deterministic."""
        prices_a = [0.5, 0.52, 0.48, 0.55, 0.51, 0.49, 0.53]
        prices_b = [0.45, 0.47, 0.44, 0.50, 0.46, 0.44, 0.48]
        beta1 = ols_beta(prices_a, prices_b)
        beta2 = ols_beta(prices_a, prices_b)
        assert beta1 == beta2


# =============================================================================
# Spread Series Tests
# =============================================================================


class TestSpreadSeries:
    """Tests for spread calculation."""

    def test_basic_spread(self) -> None:
        """Test basic spread calculation."""
        prices_a = [0.5, 0.6, 0.55]
        prices_b = [0.4, 0.5, 0.45]
        spreads = spread_series(prices_a, prices_b, beta=1.0)
        assert len(spreads) == 3
        assert abs(spreads[0] - 0.1) < 1e-10
        assert abs(spreads[1] - 0.1) < 1e-10
        assert abs(spreads[2] - 0.1) < 1e-10

    def test_spread_with_beta(self) -> None:
        """Test spread with non-1.0 beta."""
        prices_a = [1.0, 2.0, 3.0]
        prices_b = [0.5, 1.0, 1.5]
        spreads = spread_series(prices_a, prices_b, beta=2.0)
        # spread = a - 2*b = 0 for all
        assert all(abs(s) < 1e-10 for s in spreads)

    def test_mismatched_lengths_raises(self) -> None:
        """Spread should raise on mismatched lengths."""
        with pytest.raises(ValueError):
            spread_series([1.0, 2.0], [1.0], beta=1.0)


# =============================================================================
# Rolling Mean/Std and Z-Score Tests
# =============================================================================


class TestRollingMeanStd:
    """Tests for rolling mean and standard deviation."""

    def test_rolling_mean(self) -> None:
        """Test rolling mean calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        means, stds = rolling_mean_std(values, lookback=3)
        
        # First few values use expanding window
        assert abs(means[0] - 1.0) < 1e-10  # Just first value
        assert abs(means[1] - 1.5) < 1e-10  # Mean of [1, 2]
        assert abs(means[2] - 2.0) < 1e-10  # Mean of [1, 2, 3]
        assert abs(means[3] - 3.0) < 1e-10  # Mean of [2, 3, 4]
        assert abs(means[4] - 4.0) < 1e-10  # Mean of [3, 4, 5]

    def test_empty_list(self) -> None:
        """Empty list should return empty results."""
        means, stds = rolling_mean_std([], lookback=3)
        assert means == []
        assert stds == []


class TestZScoreSeries:
    """Tests for z-score calculation."""

    def test_zscore_values(self) -> None:
        """Test z-score calculation."""
        # Create data that will have known z-scores
        spreads = [0.0, 0.0, 0.0, 3.0, 0.0]  # Spike at index 3
        zscores = zscore_series(spreads, lookback=3)
        
        # Z-scores should be 0 for first few, spike at index 3, then revert
        assert len(zscores) == 5
        # First value has no prior data for std, so z=0
        assert zscores[0] == 0.0
        
    def test_constant_spreads(self) -> None:
        """Constant spreads should give z-score of 0."""
        spreads = [0.5, 0.5, 0.5, 0.5, 0.5]
        zscores = zscore_series(spreads, lookback=3)
        # All z-scores should be 0 (or near-zero due to zero std)
        assert all(abs(z) < 1e-10 for z in zscores)


# =============================================================================
# Fit Pair Params Tests
# =============================================================================


class TestFitPairParams:
    """Tests for fitting pair parameters."""

    def test_valid_fit(self) -> None:
        """Test successful parameter fitting."""
        prices_a = [0.5, 0.52, 0.48, 0.55, 0.51, 0.49, 0.53, 0.50, 0.52, 0.48]
        prices_b = [0.45, 0.47, 0.44, 0.50, 0.46, 0.44, 0.48, 0.45, 0.47, 0.43]
        
        params = fit_pair_params(
            prices_a, prices_b,
            pair_name="Test Pair",
            market_a_id="market_a",
            market_b_id="market_b",
        )
        
        assert params.is_valid
        assert params.pair_name == "Test Pair"
        assert params.train_count == 10
        assert params.beta > 0
        assert params.spread_std > 0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        params = fit_pair_params(
            [0.5, 0.52], [0.45, 0.47],
            "Test", "a", "b"
        )
        assert not params.is_valid
        assert "Insufficient TRAIN data" in params.error_msg


# =============================================================================
# Generate Signals Tests
# =============================================================================


class TestGenerateSignals:
    """Tests for signal generation."""

    def test_entry_long_signal(self) -> None:
        """Test entry long signal when z <= -entry_z."""
        params = FittedParams(
            pair_name="Test",
            market_a_id="a",
            market_b_id="b",
            beta=1.0,
            spread_mean=0.0,
            spread_std=0.1,
            train_count=20,
            is_valid=True,
        )
        
        times = ["t1", "t2", "t3"]
        # Create spreads that give z-score <= -2.0
        prices_a = [0.3, 0.3, 0.3]  # Spread = 0.3 - 0.5 = -0.2
        prices_b = [0.5, 0.5, 0.5]  # z = (-0.2 - 0) / 0.1 = -2.0
        
        signals = generate_signals(
            times, prices_a, prices_b, params,
            lookback=10, entry_z=2.0, exit_z=0.5, max_hold_bars=10
        )
        
        # Should have entry signal
        assert len(signals) >= 1
        entry_signals = [s for s in signals if s.action == SignalAction.ENTER_LONG]
        assert len(entry_signals) >= 1

    def test_entry_short_signal(self) -> None:
        """Test entry short signal when z >= entry_z."""
        params = FittedParams(
            pair_name="Test",
            market_a_id="a",
            market_b_id="b",
            beta=1.0,
            spread_mean=0.0,
            spread_std=0.1,
            train_count=20,
            is_valid=True,
        )

        times = ["t1", "t2", "t3"]
        # Create spreads that give z-score > 2.0 (clearly above threshold)
        prices_a = [0.75, 0.75, 0.75]  # Spread = 0.75 - 0.5 = 0.25
        prices_b = [0.5, 0.5, 0.5]  # z = (0.25 - 0) / 0.1 = 2.5

        signals = generate_signals(
            times, prices_a, prices_b, params,
            lookback=10, entry_z=2.0, exit_z=0.5, max_hold_bars=10
        )

        entry_signals = [s for s in signals if s.action == SignalAction.ENTER_SHORT]
        assert len(entry_signals) >= 1

    def test_invalid_params_no_signals(self) -> None:
        """Invalid params should produce no signals."""
        params = FittedParams(
            pair_name="Test",
            market_a_id="a",
            market_b_id="b",
            beta=1.0,
            spread_mean=0.0,
            spread_std=0.1,
            train_count=5,
            is_valid=False,
            error_msg="Invalid",
        )
        
        signals = generate_signals(
            ["t1"], [0.5], [0.5], params,
            lookback=10, entry_z=2.0, exit_z=0.5, max_hold_bars=10
        )
        assert signals == []


# =============================================================================
# Walk-Forward Split Tests
# =============================================================================


class TestWalkForwardSplit:
    """Tests for walk-forward time splitting."""

    def test_basic_split(self) -> None:
        """Test basic TRAIN/TEST split."""
        times = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
        split = split_times(times, train_count=6, test_count=4)
        
        assert split.train_count == 6
        assert split.test_count == 4
        assert split.train_times == ["t1", "t2", "t3", "t4", "t5", "t6"]
        assert split.test_times == ["t7", "t8", "t9", "t10"]

    def test_no_overlap(self) -> None:
        """TRAIN and TEST should not overlap (no data leakage)."""
        times = [f"t{i}" for i in range(100)]
        split = split_times(times, train_count=70, test_count=30)
        
        train_set = set(split.train_times)
        test_set = set(split.test_times)
        
        # Ensure no overlap
        assert train_set.isdisjoint(test_set), "TRAIN and TEST must not overlap"
        
        # Ensure chronological ordering
        if split.train_times and split.test_times:
            assert split.last_train < split.first_test, "TRAIN must come before TEST"

    def test_insufficient_data_scales(self) -> None:
        """When insufficient data, should scale proportionally."""
        times = ["t1", "t2", "t3", "t4", "t5"]
        split = split_times(times, train_count=100, test_count=50)
        
        # Should scale down proportionally
        assert split.train_count + split.test_count <= 5
        assert split.train_count > 0 or split.test_count > 0

    def test_empty_times(self) -> None:
        """Empty times should return empty split."""
        split = split_times([], train_count=10, test_count=5)
        assert split.train_count == 0
        assert split.test_count == 0


class TestExtractPairPrices:
    """Tests for extracting pair prices from snapshots."""

    def test_aligned_prices(self) -> None:
        """Test extracting aligned price series."""
        pair = PairConfig(
            market_a_id="market_a",
            market_b_id="market_b",
            name="Test Pair",
        )
        
        snapshots = [
            {"market_id": "market_a", "snapshot_time": "t1", "yes_price": 0.5},
            {"market_id": "market_b", "snapshot_time": "t1", "yes_price": 0.4},
            {"market_id": "market_a", "snapshot_time": "t2", "yes_price": 0.55},
            {"market_id": "market_b", "snapshot_time": "t2", "yes_price": 0.45},
            {"market_id": "market_a", "snapshot_time": "t3", "yes_price": 0.52},
            # Missing market_b at t3
        ]
        
        times, prices_a, prices_b = extract_pair_prices(
            snapshots, pair, ["t1", "t2", "t3"]
        )
        
        # Should only have times where both markets have data
        assert len(times) == 2  # t1 and t2 only
        assert len(prices_a) == 2
        assert len(prices_b) == 2
        assert times == ["t1", "t2"]


# =============================================================================
# Grid Search Tests
# =============================================================================


class TestGridSearch:
    """Tests for grid search parameter generation."""

    def test_param_combinations_count(self) -> None:
        """Test total combinations count."""
        grid = GridConfig(
            lookback=[20, 30],
            entry_z=[1.5, 2.0],
            exit_z=[0.5],
            max_hold_bars=[60],
            cooldown_bars=[5],
            fee_bps=[0.0],
            slippage_bps=[0.0],
        )
        
        combos = generate_param_combinations(grid)
        assert len(combos) == 2 * 2 * 1 * 1 * 1 * 1 * 1  # 4 combinations

    def test_deterministic_ordering(self) -> None:
        """Grid combinations should be deterministically ordered."""
        grid = GridConfig()
        combos1 = generate_param_combinations(grid)
        combos2 = generate_param_combinations(grid)
        
        assert combos1 == combos2, "Grid combinations must be deterministic"

    def test_sorted_params(self) -> None:
        """Parameters should be sorted within each dimension."""
        grid = GridConfig(
            lookback=[50, 20, 30],  # Unsorted input
            entry_z=[2.5, 1.5, 2.0],
        )
        
        combos = generate_param_combinations(grid)
        
        # First combo should have smallest values
        assert combos[0]["lookback"] == 20
        assert combos[0]["entry_z"] == 1.5


# =============================================================================
# Metrics Computation Tests
# =============================================================================


class TestMetricsComputation:
    """Tests for metrics computation from signals."""

    def test_empty_signals(self) -> None:
        """Empty signals should produce zero metrics."""
        metrics = compute_metrics_from_signals(
            [], fee_bps=0.0, slippage_bps=0.0, quantity_per_trade=10.0
        )
        
        assert metrics.total_pnl == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0

    def test_winning_trade(self) -> None:
        """Test metrics with a winning trade."""
        params = FittedParams(
            pair_name="Test", market_a_id="a", market_b_id="b",
            beta=1.0, spread_mean=0.0, spread_std=0.1, train_count=20, is_valid=True
        )
        
        # Entry at high z-score (spread = 0.2), exit at low z-score (spread = 0.05)
        signals = [
            ZScoreSignal(
                time="t1", pair_name="Test", market_a_id="a", market_b_id="b",
                action=SignalAction.ENTER_SHORT,  # Short spread expecting it to decrease
                z_score=2.0, spread=0.2, price_a=0.6, price_b=0.4, beta=1.0
            ),
            ZScoreSignal(
                time="t2", pair_name="Test", market_a_id="a", market_b_id="b",
                action=SignalAction.EXIT,
                z_score=0.5, spread=0.05, price_a=0.5, price_b=0.45, beta=1.0
            ),
        ]
        
        metrics = compute_metrics_from_signals(
            signals, fee_bps=0.0, slippage_bps=0.0, quantity_per_trade=10.0
        )
        
        # Short spread: profit when spread decreases
        # Spread went from 0.2 to 0.05, change = -0.15
        # Short profit = -(-0.15) * 10 = 1.5
        assert metrics.total_trades == 1
        assert metrics.total_pnl > 0  # Should be profitable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
