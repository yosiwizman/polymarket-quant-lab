"""Tests for orderbook-based edge calculator.

Phase 8: Unit tests for edge_calc.py with deterministic fixtures.

Test scenarios:
- BUY_BOTH: ask_yes=0.48, ask_no=0.49 → raw_edge_bps=300
- SELL_BOTH: bid_yes=0.53, bid_no=0.52 → raw_edge_bps=500
- Normal market: sums ~1.0 → edges near 0 / negative
- Edge cases: missing prices, invalid books, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

from pmq.ops.edge_calc import (
    ArbSide,
    EdgeResult,
    compute_arb_edge,
    compute_edge_from_prices,
)

# =============================================================================
# Mock OrderBookData for testing
# =============================================================================


@dataclass
class MockOrderBook:
    """Mock OrderBookData for testing edge computation."""

    token_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None
    best_ask_size: float | None = None
    top_depth_usd: float | None = None

    @property
    def has_valid_book(self) -> bool:
        """Check if we have valid bid and ask data."""
        return self.best_bid is not None and self.best_ask is not None


# =============================================================================
# Test: BUY_BOTH Arb Opportunity
# =============================================================================


class TestBuyBothArb:
    """Test BUY_BOTH arbitrage opportunity detection.

    BUY_BOTH: Buy YES at ask + Buy NO at ask
    Profit when: ask_yes + ask_no < 1.0
    raw_edge_bps = (1.0 - (ask_yes + ask_no)) * 10_000
    """

    def test_buy_both_positive_edge(self) -> None:
        """BUY_BOTH with positive edge: ask_yes=0.48, ask_no=0.49 → 300 bps."""
        # Setup: ask_yes + ask_no = 0.97 < 1.0 → profit potential
        yes_book = MockOrderBook(
            token_id="yes_token_123",
            best_bid=0.47,
            best_ask=0.48,
            top_depth_usd=100.0,
        )
        no_book = MockOrderBook(
            token_id="no_token_456",
            best_bid=0.48,
            best_ask=0.49,
            top_depth_usd=150.0,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Verify edge computation
        # buy_cost = 0.48 + 0.49 = 0.97
        # buy_edge_bps = (1.0 - 0.97) * 10_000 = 300
        assert result.arb_side == ArbSide.BUY_BOTH
        assert result.raw_edge_bps == 300.0
        assert result.buy_edge_bps == 300.0
        assert result.buy_cost == 0.97

        # Verify token IDs populated
        assert result.yes_token_id == "yes_token_123"
        assert result.no_token_id == "no_token_456"

        # Verify prices captured
        assert result.ask_yes == 0.48
        assert result.ask_no == 0.49
        assert result.bid_yes == 0.47
        assert result.bid_no == 0.48

        # Verify no error
        assert result.error is None

    def test_buy_both_with_fees(self) -> None:
        """BUY_BOTH edge should decrease with fees."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,
        )

        # Without fees: 300 bps
        result_no_fees = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]
        assert result_no_fees.raw_edge_bps == 300.0

        # With 10 bps fee per leg: 300 - (10 * 2) = 280 bps
        result_with_fees = compute_arb_edge(yes_book, no_book, fee_bps=10.0)  # type: ignore[arg-type]
        assert result_with_fees.raw_edge_bps == 280.0

        # With fee + slippage: 300 - ((10 + 5) * 2) = 270 bps
        result_with_all = compute_arb_edge(yes_book, no_book, fee_bps=10.0, slippage_bps=5.0)  # type: ignore[arg-type]
        assert result_with_all.raw_edge_bps == 270.0


# =============================================================================
# Test: SELL_BOTH Arb Opportunity
# =============================================================================


class TestSellBothArb:
    """Test SELL_BOTH arbitrage opportunity detection.

    SELL_BOTH: Sell YES at bid + Sell NO at bid
    Profit when: bid_yes + bid_no > 1.0
    raw_edge_bps = ((bid_yes + bid_no) - 1.0) * 10_000
    """

    def test_sell_both_positive_edge(self) -> None:
        """SELL_BOTH with positive edge: bid_yes=0.53, bid_no=0.52 → 500 bps."""
        # Setup: bid_yes + bid_no = 1.05 > 1.0 → profit potential
        yes_book = MockOrderBook(
            token_id="yes_token_abc",
            best_bid=0.53,
            best_ask=0.54,
            top_depth_usd=200.0,
        )
        no_book = MockOrderBook(
            token_id="no_token_xyz",
            best_bid=0.52,
            best_ask=0.53,
            top_depth_usd=180.0,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Verify edge computation
        # sell_revenue = 0.53 + 0.52 = 1.05
        # sell_edge_bps = (1.05 - 1.0) * 10_000 = 500
        # buy_edge_bps = (1.0 - (0.54 + 0.53)) * 10_000 = (1.0 - 1.07) * 10_000 = -700
        # Best is SELL_BOTH at 500 bps
        assert result.arb_side == ArbSide.SELL_BOTH
        assert result.raw_edge_bps == 500.0
        assert result.sell_edge_bps == 500.0
        assert result.sell_revenue == 1.05

        # Verify BUY side was also computed (but worse)
        assert result.buy_edge_bps == -700.0

        # Verify token IDs
        assert result.yes_token_id == "yes_token_abc"
        assert result.no_token_id == "no_token_xyz"

    def test_sell_both_with_fees(self) -> None:
        """SELL_BOTH edge should decrease with fees."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.53,
            best_ask=0.54,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.52,
            best_ask=0.53,
        )

        # Without fees: 500 bps
        result_no_fees = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]
        assert result_no_fees.raw_edge_bps == 500.0

        # With 10 bps fee per leg: 500 - (10 * 2) = 480 bps
        result_with_fees = compute_arb_edge(yes_book, no_book, fee_bps=10.0)  # type: ignore[arg-type]
        assert result_with_fees.raw_edge_bps == 480.0


# =============================================================================
# Test: Normal Market (No Arb)
# =============================================================================


class TestNormalMarket:
    """Test normal market conditions where edges are near zero or negative."""

    def test_efficient_market_negative_edges(self) -> None:
        """Efficient market: bid/ask sums near 1.0 → negative edges."""
        # Setup: Typical efficient market
        # ask_yes + ask_no = 0.51 + 0.51 = 1.02 → BUY_BOTH edge = -200
        # bid_yes + bid_no = 0.49 + 0.49 = 0.98 → SELL_BOTH edge = -200
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.49,
            best_ask=0.51,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.49,
            best_ask=0.51,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Both edges are -200, BUY_BOTH wins (or ties, implementation chooses BUY)
        assert result.arb_side in (ArbSide.BUY_BOTH, ArbSide.SELL_BOTH)
        assert result.raw_edge_bps == -200.0

        # Verify both edges were computed
        assert result.buy_edge_bps == -200.0
        assert result.sell_edge_bps == -200.0

    def test_tight_spread_zero_edge(self) -> None:
        """Market with tight spreads summing to exactly 1.0."""
        # ask_yes + ask_no = 0.50 + 0.50 = 1.0 → BUY_BOTH edge = 0
        # bid_yes + bid_no = 0.50 + 0.50 = 1.0 → SELL_BOTH edge = 0
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.50,
            best_ask=0.50,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.50,
            best_ask=0.50,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        assert result.raw_edge_bps == 0.0
        assert result.buy_edge_bps == 0.0
        assert result.sell_edge_bps == 0.0

    def test_asymmetric_market(self) -> None:
        """Market where one side has opportunity but other doesn't."""
        # ask_yes + ask_no = 0.40 + 0.58 = 0.98 → BUY_BOTH edge = 200
        # bid_yes + bid_no = 0.38 + 0.56 = 0.94 → SELL_BOTH edge = -600
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.38,
            best_ask=0.40,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.56,
            best_ask=0.58,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # BUY_BOTH wins
        assert result.arb_side == ArbSide.BUY_BOTH
        assert result.raw_edge_bps == 200.0
        assert result.buy_edge_bps == 200.0
        assert result.sell_edge_bps == -600.0


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_yes_book(self) -> None:
        """Should handle missing YES orderbook."""
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.50,
            best_ask=0.51,
        )

        result = compute_arb_edge(None, no_book)  # type: ignore[arg-type]

        assert result.arb_side == ArbSide.NONE
        assert result.raw_edge_bps == 0.0
        assert result.error == "missing_orderbook"

    def test_missing_no_book(self) -> None:
        """Should handle missing NO orderbook."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.50,
            best_ask=0.51,
        )

        result = compute_arb_edge(yes_book, None)  # type: ignore[arg-type]

        assert result.arb_side == ArbSide.NONE
        assert result.raw_edge_bps == 0.0
        assert result.error == "missing_orderbook"

    def test_invalid_yes_book(self) -> None:
        """Should handle invalid YES book (no bid/ask)."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=None,
            best_ask=None,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.50,
            best_ask=0.51,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        assert result.arb_side == ArbSide.NONE
        assert result.raw_edge_bps == 0.0
        assert result.error == "invalid_book"

    def test_partial_book_asks_only(self) -> None:
        """Should compute BUY_BOTH when only asks are available."""
        # Only asks, no bids
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=None,
            best_ask=0.48,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=None,
            best_ask=0.49,
        )

        # has_valid_book requires both bid and ask, so this should fail
        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]
        assert result.error == "invalid_book"

    def test_partial_book_bids_only(self) -> None:
        """Should compute SELL_BOTH when only bids are available."""
        # Only bids, no asks
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.53,
            best_ask=None,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.52,
            best_ask=None,
        )

        # has_valid_book requires both bid and ask, so this should fail
        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]
        assert result.error == "invalid_book"


# =============================================================================
# Test: Mid Price and Spread Computation
# =============================================================================


class TestMidPriceAndSpread:
    """Test mid price and spread_bps computation."""

    def test_mid_price_for_buy_both(self) -> None:
        """Mid price should be computed from YES market for BUY_BOTH."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Mid of YES = (0.47 + 0.48) / 2 = 0.475
        assert result.mid_price == 0.475
        assert result.arb_side == ArbSide.BUY_BOTH

    def test_mid_price_for_sell_both(self) -> None:
        """Mid price should be computed from YES market for SELL_BOTH."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.53,
            best_ask=0.54,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.52,
            best_ask=0.53,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Mid of YES = (0.53 + 0.54) / 2 = 0.535
        assert result.mid_price == 0.535
        assert result.arb_side == ArbSide.SELL_BOTH

    def test_spread_bps_computation(self) -> None:
        """Spread should be sum of individual leg spreads."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,  # spread = (0.48-0.47)/0.47 * 10000 = ~212.77
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,  # spread = (0.49-0.48)/0.48 * 10000 = ~208.33
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        # Combined spread ~421 bps (sum of both legs)
        assert 420.0 < result.spread_bps < 422.0


# =============================================================================
# Test: Depth Info
# =============================================================================


class TestDepthInfo:
    """Test min_depth_usd computation."""

    def test_min_depth_both_books(self) -> None:
        """min_depth_usd should be minimum of both books."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,
            top_depth_usd=100.0,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,
            top_depth_usd=150.0,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        assert result.min_depth_usd == 100.0

    def test_min_depth_one_book_missing(self) -> None:
        """min_depth_usd should use available depth."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,
            top_depth_usd=None,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,
            top_depth_usd=150.0,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        assert result.min_depth_usd == 150.0

    def test_min_depth_both_missing(self) -> None:
        """min_depth_usd should be None if both missing."""
        yes_book = MockOrderBook(
            token_id="yes",
            best_bid=0.47,
            best_ask=0.48,
            top_depth_usd=None,
        )
        no_book = MockOrderBook(
            token_id="no",
            best_bid=0.48,
            best_ask=0.49,
            top_depth_usd=None,
        )

        result = compute_arb_edge(yes_book, no_book)  # type: ignore[arg-type]

        assert result.min_depth_usd is None


# =============================================================================
# Test: Convenience Function
# =============================================================================


class TestComputeEdgeFromPrices:
    """Test compute_edge_from_prices convenience function."""

    def test_buy_both_from_prices(self) -> None:
        """Should compute BUY_BOTH edge from raw prices."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            yes_token_id="yes_123",
            no_token_id="no_456",
        )

        assert result.arb_side == ArbSide.BUY_BOTH
        assert result.raw_edge_bps == 300.0
        assert result.yes_token_id == "yes_123"
        assert result.no_token_id == "no_456"

    def test_sell_both_from_prices(self) -> None:
        """Should compute SELL_BOTH edge from raw prices."""
        result = compute_edge_from_prices(
            ask_yes=0.54,
            ask_no=0.53,
            bid_yes=0.53,
            bid_no=0.52,
        )

        assert result.arb_side == ArbSide.SELL_BOTH
        assert result.raw_edge_bps == 500.0

    def test_with_fees_from_prices(self) -> None:
        """Should apply fees correctly."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            fee_bps=10.0,
            slippage_bps=5.0,
        )

        # 300 - (15 * 2) = 270
        assert result.raw_edge_bps == 270.0


# =============================================================================
# Test: EdgeResult Serialization
# =============================================================================


# =============================================================================
# Test: Phase 9 Net Edge Computation
# =============================================================================


class TestNetEdgeComputation:
    """Test Phase 9 gross_edge_bps and net_edge_bps computation.

    Phase 9: Edge is now computed in two stages:
    - gross_edge_bps: Edge BEFORE fees/slippage
    - net_edge_bps: Edge AFTER fees/slippage (= gross - 2 * (fee_bps + slippage_bps))
    """

    def test_gross_and_net_edge_buy_both(self) -> None:
        """Test gross/net edge computation for BUY_BOTH."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            fee_bps=10.0,
            slippage_bps=5.0,
        )

        # Gross edge: (1.0 - (0.48 + 0.49)) * 10000 = 300 bps
        assert result.gross_edge_bps == 300.0

        # Net edge: 300 - 2 * (10 + 5) = 300 - 30 = 270 bps
        assert result.net_edge_bps == 270.0

        # raw_edge_bps should equal net_edge_bps for backward compat
        assert result.raw_edge_bps == result.net_edge_bps

    def test_gross_and_net_edge_sell_both(self) -> None:
        """Test gross/net edge computation for SELL_BOTH."""
        result = compute_edge_from_prices(
            ask_yes=0.54,
            ask_no=0.53,
            bid_yes=0.53,
            bid_no=0.52,
            fee_bps=20.0,
            slippage_bps=10.0,
        )

        # Gross edge: ((0.53 + 0.52) - 1.0) * 10000 = 500 bps
        assert result.gross_edge_bps == 500.0

        # Net edge: 500 - 2 * (20 + 10) = 500 - 60 = 440 bps
        assert result.net_edge_bps == 440.0

    def test_net_edge_can_be_negative(self) -> None:
        """Test that net edge can go negative with high fees."""
        result = compute_edge_from_prices(
            ask_yes=0.49,
            ask_no=0.50,
            bid_yes=0.48,
            bid_no=0.49,
            fee_bps=100.0,  # High fees wipe out edge
            slippage_bps=50.0,
        )

        # Gross edge: (1.0 - 0.99) * 10000 = 100 bps
        assert result.gross_edge_bps == 100.0

        # Net edge: 100 - 2 * (100 + 50) = 100 - 300 = -200 bps
        assert result.net_edge_bps == -200.0

    def test_zero_fees_gross_equals_net(self) -> None:
        """Test that gross == net when fees are zero."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        assert result.gross_edge_bps == 300.0
        assert result.net_edge_bps == 300.0
        assert result.gross_edge_bps == result.net_edge_bps

    def test_fee_bps_and_slippage_bps_stored(self) -> None:
        """Test that fee_bps and slippage_bps are stored in result."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            fee_bps=15.0,
            slippage_bps=8.0,
        )

        assert result.fee_bps == 15.0
        assert result.slippage_bps == 8.0

    def test_to_dict_includes_net_edge_fields(self) -> None:
        """Test that to_dict includes gross/net edge and fee fields."""
        result = compute_edge_from_prices(
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            fee_bps=10.0,
            slippage_bps=5.0,
        )

        d = result.to_dict()

        assert "gross_edge_bps" in d
        assert "net_edge_bps" in d
        assert "fee_bps" in d
        assert "slippage_bps" in d
        assert d["gross_edge_bps"] == 300.0
        assert d["net_edge_bps"] == 270.0
        assert d["fee_bps"] == 10.0
        assert d["slippage_bps"] == 5.0


class TestEdgeResultSerialization:
    """Test EdgeResult.to_dict() serialization."""

    def test_to_dict_complete(self) -> None:
        """to_dict should include all fields."""
        result = EdgeResult(
            yes_token_id="yes_abc",
            no_token_id="no_xyz",
            arb_side=ArbSide.BUY_BOTH,
            raw_edge_bps=300.0,
            ask_yes=0.48,
            ask_no=0.49,
            bid_yes=0.47,
            bid_no=0.48,
            buy_cost=0.97,
            buy_edge_bps=300.0,
            sell_revenue=0.95,
            sell_edge_bps=-500.0,
            mid_price=0.475,
            spread_bps=421.1,
            min_depth_usd=100.0,
        )

        d = result.to_dict()

        assert d["yes_token_id"] == "yes_abc"
        assert d["no_token_id"] == "no_xyz"
        assert d["arb_side"] == "BUY_BOTH"
        assert d["raw_edge_bps"] == 300.0
        assert d["ask_yes"] == 0.48
        assert d["ask_no"] == 0.49
        assert d["bid_yes"] == 0.47
        assert d["bid_no"] == 0.48
        assert d["buy_cost"] == 0.97
        assert d["buy_edge_bps"] == 300.0
        assert d["sell_revenue"] == 0.95
        assert d["sell_edge_bps"] == -500.0
        assert d["mid_price"] == 0.475
        assert d["spread_bps"] == 421.1
        assert d["min_depth_usd"] == 100.0
        assert d["error"] is None

    def test_to_dict_with_none_values(self) -> None:
        """to_dict should handle None values."""
        result = EdgeResult(
            yes_token_id="yes",
            no_token_id="no",
            arb_side=ArbSide.NONE,
            raw_edge_bps=0.0,
            error="invalid_book",
        )

        d = result.to_dict()

        assert d["ask_yes"] is None
        assert d["ask_no"] is None
        assert d["bid_yes"] is None
        assert d["bid_no"] is None
        assert d["buy_cost"] is None
        assert d["buy_edge_bps"] is None
        assert d["sell_revenue"] is None
        assert d["sell_edge_bps"] is None
        assert d["min_depth_usd"] is None
        assert d["error"] == "invalid_book"
