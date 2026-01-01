"""Tests for drift detection module.

Phase 5.5: Unit tests for drift calculation, threshold detection,
and edge cases (empty sides, crossed book, tiny prices, rounding).
"""

from pmq.markets.orderbook import OrderBookData, OrderBookLevel
from pmq.ops.drift import (
    DEFAULT_DEPTH_DIFF_PCT,
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_MID_DIFF_BPS,
    DEFAULT_SPREAD_DIFF_BPS,
    DriftMetrics,
    DriftThresholds,
    compute_drift_metrics,
)

# =============================================================================
# DriftThresholds Tests
# =============================================================================


class TestDriftThresholds:
    """Tests for DriftThresholds dataclass."""

    def test_default_values(self) -> None:
        """DriftThresholds should have sensible defaults."""
        thresholds = DriftThresholds()
        assert thresholds.mid_diff_bps == DEFAULT_MID_DIFF_BPS
        assert thresholds.spread_diff_bps == DEFAULT_SPREAD_DIFF_BPS
        assert thresholds.depth_diff_pct == DEFAULT_DEPTH_DIFF_PCT
        assert thresholds.depth_levels == DEFAULT_DEPTH_LEVELS

    def test_custom_values(self) -> None:
        """DriftThresholds should accept custom values."""
        thresholds = DriftThresholds(
            mid_diff_bps=50.0,
            spread_diff_bps=100.0,
            depth_diff_pct=75.0,
            depth_levels=5,
        )
        assert thresholds.mid_diff_bps == 50.0
        assert thresholds.spread_diff_bps == 100.0
        assert thresholds.depth_diff_pct == 75.0
        assert thresholds.depth_levels == 5


# =============================================================================
# DriftMetrics Tests
# =============================================================================


class TestDriftMetrics:
    """Tests for DriftMetrics dataclass."""

    def test_default_values(self) -> None:
        """DriftMetrics should initialize with None/False values."""
        metrics = DriftMetrics()
        assert metrics.mid_diff_bps is None
        assert metrics.spread_diff_bps is None
        assert metrics.depth_diff_pct is None
        assert metrics.has_drift is False
        assert metrics.drift_reason is None

    def test_drift_detected(self) -> None:
        """DriftMetrics should track drift detection."""
        metrics = DriftMetrics(
            mid_diff_bps=30.0,
            has_drift=True,
            drift_reason="mid=30.0bps",
        )
        assert metrics.has_drift is True
        assert metrics.drift_reason == "mid=30.0bps"


# =============================================================================
# compute_drift_metrics Tests
# =============================================================================


def _make_orderbook(
    token_id: str = "test",
    best_bid: float | None = None,
    best_ask: float | None = None,
    spread_bps: float | None = None,
    mid_price: float | None = None,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> OrderBookData:
    """Helper to create OrderBookData for testing."""
    ob = OrderBookData(
        token_id=token_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread_bps=spread_bps,
        mid_price=mid_price,
    )
    if bids:
        ob.bids = [OrderBookLevel(price=p, size=s) for p, s in bids]
    if asks:
        ob.asks = [OrderBookLevel(price=p, size=s) for p, s in asks]
    return ob


class TestComputeDriftMetrics:
    """Tests for compute_drift_metrics function."""

    def test_identical_orderbooks_no_drift(self) -> None:
        """Identical orderbooks should have no drift."""
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=392.16,
            bids=[(0.50, 100.0), (0.49, 50.0)],
            asks=[(0.52, 100.0), (0.53, 50.0)],
        )
        rest_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=392.16,
            bids=[(0.50, 100.0), (0.49, 50.0)],
            asks=[(0.52, 100.0), (0.53, 50.0)],
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps == 0.0
        assert metrics.spread_diff_bps == 0.0
        assert metrics.has_drift is False

    def test_mid_price_drift_triggers_drift(self) -> None:
        """Mid price difference above threshold should trigger drift."""
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=392.16,
        )
        # REST has different mid (0.53 vs 0.51 = ~3.9% = 390 bps)
        rest_ob = _make_orderbook(
            best_bid=0.52,
            best_ask=0.54,
            mid_price=0.53,
            spread_bps=377.36,
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps is not None
        assert metrics.mid_diff_bps > DEFAULT_MID_DIFF_BPS
        assert metrics.has_drift is True
        assert "mid=" in (metrics.drift_reason or "")

    def test_spread_drift_triggers_drift(self) -> None:
        """Spread difference above threshold should trigger drift."""
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=100.0,  # 100 bps spread
        )
        rest_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=150.0,  # 150 bps spread (50 bps diff > 25 bps threshold)
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.spread_diff_bps == 50.0
        assert metrics.has_drift is True
        assert "spread=" in (metrics.drift_reason or "")

    def test_depth_drift_triggers_drift(self) -> None:
        """Depth difference above threshold should trigger drift."""
        # WSS has larger depth
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=392.16,
            bids=[(0.50, 200.0), (0.49, 100.0)],
            asks=[(0.52, 200.0), (0.53, 100.0)],
        )
        # REST has much smaller depth (>50% difference)
        rest_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=392.16,
            bids=[(0.50, 50.0), (0.49, 25.0)],
            asks=[(0.52, 50.0), (0.53, 25.0)],
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.depth_diff_pct is not None
        assert metrics.depth_diff_pct > DEFAULT_DEPTH_DIFF_PCT
        assert metrics.has_drift is True
        assert "depth=" in (metrics.drift_reason or "")

    def test_below_threshold_no_drift(self) -> None:
        """Differences below thresholds should not trigger drift."""
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=100.0,
            bids=[(0.50, 100.0)],
            asks=[(0.52, 100.0)],
        )
        # Tiny differences (within thresholds)
        rest_ob = _make_orderbook(
            best_bid=0.501,
            best_ask=0.521,
            mid_price=0.511,  # ~0.2% diff = 20 bps (< 25 bps)
            spread_bps=110.0,  # 10 bps diff (< 25 bps)
            bids=[(0.501, 90.0)],  # ~10% depth diff (< 50%)
            asks=[(0.521, 90.0)],
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.has_drift is False
        assert metrics.drift_reason is None

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        wss_ob = _make_orderbook(
            best_bid=0.50,
            best_ask=0.52,
            mid_price=0.51,
            spread_bps=100.0,
        )
        rest_ob = _make_orderbook(
            best_bid=0.501,
            best_ask=0.521,
            mid_price=0.511,
            spread_bps=110.0,
        )

        # With tight thresholds (10 bps), should detect drift
        tight_thresholds = DriftThresholds(
            mid_diff_bps=10.0,
            spread_diff_bps=5.0,
        )
        metrics = compute_drift_metrics(wss_ob, rest_ob, tight_thresholds)

        assert metrics.has_drift is True


class TestDriftEdgeCases:
    """Tests for edge cases in drift detection."""

    def test_missing_mid_prices(self) -> None:
        """Missing mid prices should result in None mid_diff."""
        wss_ob = _make_orderbook(best_bid=0.50, best_ask=None, mid_price=None)
        rest_ob = _make_orderbook(best_bid=0.50, best_ask=None, mid_price=None)

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps is None
        assert metrics.has_drift is False

    def test_missing_spreads(self) -> None:
        """Missing spreads should result in None spread_diff."""
        wss_ob = _make_orderbook(best_bid=0.50, best_ask=0.52, spread_bps=None)
        rest_ob = _make_orderbook(best_bid=0.50, best_ask=0.52, spread_bps=None)

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.spread_diff_bps is None

    def test_empty_depth(self) -> None:
        """Empty bids/asks should result in None depth_diff."""
        wss_ob = _make_orderbook(best_bid=0.50, best_ask=0.52)
        rest_ob = _make_orderbook(best_bid=0.50, best_ask=0.52)

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        # Empty depth = 0, so can't compute percentage diff
        # Actually the computation will be 0 - 0 = 0, but denom is 0 -> None
        assert metrics.depth_diff_pct is None

    def test_zero_rest_mid_price(self) -> None:
        """Zero REST mid price should result in None mid_diff."""
        wss_ob = _make_orderbook(mid_price=0.50)
        rest_ob = _make_orderbook(mid_price=0.0)

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps is None

    def test_tiny_prices(self) -> None:
        """Tiny prices (like 0.001) should work correctly."""
        wss_ob = _make_orderbook(
            best_bid=0.001,
            best_ask=0.002,
            mid_price=0.0015,
            spread_bps=6666.67,
        )
        rest_ob = _make_orderbook(
            best_bid=0.001,
            best_ask=0.002,
            mid_price=0.0015,
            spread_bps=6666.67,
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps == 0.0
        assert metrics.has_drift is False

    def test_crossed_book_prices(self) -> None:
        """Crossed book (bid > ask) should still compute metrics."""
        wss_ob = _make_orderbook(
            best_bid=0.52,
            best_ask=0.50,  # Crossed!
            mid_price=0.51,
            spread_bps=-392.16,  # Negative spread
        )
        rest_ob = _make_orderbook(
            best_bid=0.52,
            best_ask=0.50,
            mid_price=0.51,
            spread_bps=-392.16,
        )

        # Should still work (just comparing values)
        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps == 0.0
        assert metrics.spread_diff_bps == 0.0
        assert metrics.has_drift is False

    def test_large_price_differences(self) -> None:
        """Large price differences should produce large bps values."""
        wss_ob = _make_orderbook(mid_price=0.10)
        rest_ob = _make_orderbook(mid_price=0.20)  # 100% difference = 10000 bps

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.mid_diff_bps is not None
        assert metrics.mid_diff_bps == 5000.0  # |0.10 - 0.20| / 0.20 * 10000
        assert metrics.has_drift is True

    def test_decimal_precision(self) -> None:
        """Decimal calculations should avoid floating-point errors."""
        # These values could cause floating-point precision issues
        wss_ob = _make_orderbook(mid_price=0.3333333333333333)
        rest_ob = _make_orderbook(mid_price=0.3333333333333334)

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        # Should be essentially zero (within rounding)
        assert metrics.mid_diff_bps is not None
        assert metrics.mid_diff_bps < 0.01  # Less than 0.01 bps

    def test_multiple_drift_reasons(self) -> None:
        """Multiple threshold violations should all be listed."""
        wss_ob = _make_orderbook(
            mid_price=0.50,
            spread_bps=100.0,
            bids=[(0.50, 200.0)],
            asks=[(0.52, 200.0)],
        )
        rest_ob = _make_orderbook(
            mid_price=0.55,  # 10% diff = 1000 bps > 25
            spread_bps=150.0,  # 50 bps diff > 25
            bids=[(0.55, 50.0)],  # Large depth diff > 50%
            asks=[(0.57, 50.0)],
        )

        metrics = compute_drift_metrics(wss_ob, rest_ob)

        assert metrics.has_drift is True
        assert metrics.drift_reason is not None
        # Should contain multiple reasons
        assert "mid=" in metrics.drift_reason
        assert "spread=" in metrics.drift_reason
        assert "depth=" in metrics.drift_reason
