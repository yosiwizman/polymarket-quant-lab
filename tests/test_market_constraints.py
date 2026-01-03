"""Tests for market_constraints module (Phase 13).

Tests tick_size/min_order_size quantization and validation logic.
"""

from decimal import Decimal

import pytest

from pmq.ops.market_constraints import (
    DEFAULT_MIN_ORDER_SIZE,
    DEFAULT_TICK_SIZE,
    MAX_VALID_PRICE,
    MIN_VALID_PRICE,
    MarketConstraints,
    compute_non_marketable_buy_price,
    compute_non_marketable_sell_price,
    is_price_marketable,
    quantize_price,
    quantize_price_for_buy,
    quantize_price_for_sell,
    quantize_size,
    validate_probe_order,
)


class TestQuantizePrice:
    """Tests for price quantization."""

    def test_quantize_price_round_down(self) -> None:
        """Test price rounds down when round_down=True."""
        result = quantize_price(0.4567, Decimal("0.01"), round_down=True)
        assert result == Decimal("0.45")

    def test_quantize_price_round_up(self) -> None:
        """Test price rounds up when round_down=False."""
        result = quantize_price(0.4567, Decimal("0.01"), round_down=False)
        assert result == Decimal("0.46")

    def test_quantize_price_already_aligned(self) -> None:
        """Test aligned price stays unchanged."""
        result = quantize_price(0.45, Decimal("0.01"))
        assert result == Decimal("0.45")

    def test_quantize_price_string_input(self) -> None:
        """Test string input works."""
        result = quantize_price("0.4567", Decimal("0.01"))
        assert result == Decimal("0.45")

    def test_quantize_price_decimal_input(self) -> None:
        """Test Decimal input works."""
        result = quantize_price(Decimal("0.4567"), Decimal("0.01"))
        assert result == Decimal("0.45")

    def test_quantize_price_enforces_min_bound(self) -> None:
        """Test price is bounded to MIN_VALID_PRICE."""
        result = quantize_price(0.001, Decimal("0.01"), round_down=True)
        assert result == MIN_VALID_PRICE

    def test_quantize_price_enforces_max_bound(self) -> None:
        """Test price is bounded to MAX_VALID_PRICE."""
        result = quantize_price(0.999, Decimal("0.01"), round_down=False)
        assert result == MAX_VALID_PRICE

    def test_quantize_price_for_buy(self) -> None:
        """Test buy price always rounds down."""
        result = quantize_price_for_buy(0.4567)
        assert result == Decimal("0.45")

    def test_quantize_price_for_sell(self) -> None:
        """Test sell price always rounds up."""
        result = quantize_price_for_sell(0.4567)
        assert result == Decimal("0.46")


class TestQuantizeSize:
    """Tests for size quantization."""

    def test_quantize_size_below_minimum(self) -> None:
        """Test size below minimum returns minimum."""
        result = quantize_size(0.5, Decimal("1.0"))
        assert result == Decimal("1.0")

    def test_quantize_size_round_down(self) -> None:
        """Test size rounds down when round_down=True."""
        result = quantize_size(5.7, Decimal("1.0"), round_down=True)
        assert result == Decimal("5")

    def test_quantize_size_round_up(self) -> None:
        """Test size rounds up when round_down=False."""
        result = quantize_size(5.3, Decimal("1.0"), round_down=False)
        assert result == Decimal("6")

    def test_quantize_size_exactly_minimum(self) -> None:
        """Test size at exactly minimum stays unchanged."""
        result = quantize_size(1.0, Decimal("1.0"))
        assert result == Decimal("1")

    def test_quantize_size_with_small_min(self) -> None:
        """Test with small minimum order size."""
        result = quantize_size(0.15, Decimal("0.1"), round_down=True)
        assert result == Decimal("0.1")


class TestNonMarketablePrices:
    """Tests for computing non-marketable prices."""

    def test_non_marketable_buy_price(self) -> None:
        """Test non-marketable buy price is below best bid."""
        result = compute_non_marketable_buy_price(Decimal("0.45"), Decimal("0.01"))
        assert result == Decimal("0.44")

    def test_non_marketable_buy_price_at_min(self) -> None:
        """Test non-marketable buy price doesn't go below MIN_VALID_PRICE."""
        result = compute_non_marketable_buy_price(Decimal("0.01"), Decimal("0.01"))
        assert result == MIN_VALID_PRICE

    def test_non_marketable_sell_price(self) -> None:
        """Test non-marketable sell price is above best ask."""
        result = compute_non_marketable_sell_price(Decimal("0.55"), Decimal("0.01"))
        assert result == Decimal("0.56")

    def test_non_marketable_sell_price_at_max(self) -> None:
        """Test non-marketable sell price doesn't go above MAX_VALID_PRICE."""
        result = compute_non_marketable_sell_price(Decimal("0.99"), Decimal("0.01"))
        assert result == MAX_VALID_PRICE


class TestIsPriceMarketable:
    """Tests for marketability checking."""

    def test_buy_at_ask_is_marketable(self) -> None:
        """Test BUY at ask price is marketable."""
        result = is_price_marketable(Decimal("0.50"), "BUY", Decimal("0.45"), Decimal("0.50"))
        assert result is True

    def test_buy_above_ask_is_marketable(self) -> None:
        """Test BUY above ask is marketable."""
        result = is_price_marketable(Decimal("0.51"), "BUY", Decimal("0.45"), Decimal("0.50"))
        assert result is True

    def test_buy_below_ask_is_not_marketable(self) -> None:
        """Test BUY below ask is not marketable."""
        result = is_price_marketable(Decimal("0.49"), "BUY", Decimal("0.45"), Decimal("0.50"))
        assert result is False

    def test_sell_at_bid_is_marketable(self) -> None:
        """Test SELL at bid price is marketable."""
        result = is_price_marketable(Decimal("0.45"), "SELL", Decimal("0.45"), Decimal("0.50"))
        assert result is True

    def test_sell_below_bid_is_marketable(self) -> None:
        """Test SELL below bid is marketable."""
        result = is_price_marketable(Decimal("0.44"), "SELL", Decimal("0.45"), Decimal("0.50"))
        assert result is True

    def test_sell_above_bid_is_not_marketable(self) -> None:
        """Test SELL above bid is not marketable."""
        result = is_price_marketable(Decimal("0.46"), "SELL", Decimal("0.45"), Decimal("0.50"))
        assert result is False

    def test_marketable_with_no_best_ask(self) -> None:
        """Test BUY with no best ask is not marketable."""
        result = is_price_marketable(Decimal("0.50"), "BUY", Decimal("0.45"), None)
        assert result is False

    def test_marketable_with_no_best_bid(self) -> None:
        """Test SELL with no best bid is not marketable."""
        result = is_price_marketable(Decimal("0.45"), "SELL", None, Decimal("0.50"))
        assert result is False


class TestValidateProbeOrder:
    """Tests for probe order validation."""

    @pytest.fixture
    def constraints(self) -> MarketConstraints:
        """Create test constraints."""
        return MarketConstraints(
            token_id="test_token",
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("1.0"),
            best_bid=Decimal("0.45"),
            best_ask=Decimal("0.50"),
        )

    def test_valid_buy_probe(self, constraints: MarketConstraints) -> None:
        """Test valid non-marketable buy probe order."""
        is_valid, error = validate_probe_order(Decimal("0.44"), Decimal("1.0"), "BUY", constraints)
        assert is_valid is True
        assert error == "OK"

    def test_valid_sell_probe(self, constraints: MarketConstraints) -> None:
        """Test valid non-marketable sell probe order."""
        is_valid, error = validate_probe_order(Decimal("0.51"), Decimal("1.0"), "SELL", constraints)
        assert is_valid is True
        assert error == "OK"

    def test_invalid_price_out_of_bounds(self, constraints: MarketConstraints) -> None:
        """Test price outside valid bounds fails."""
        is_valid, error = validate_probe_order(Decimal("1.01"), Decimal("1.0"), "BUY", constraints)
        assert is_valid is False
        assert "outside valid range" in error

    def test_invalid_price_not_aligned(self, constraints: MarketConstraints) -> None:
        """Test price not aligned to tick fails."""
        is_valid, error = validate_probe_order(Decimal("0.445"), Decimal("1.0"), "BUY", constraints)
        assert is_valid is False
        assert "not aligned" in error

    def test_invalid_size_below_minimum(self, constraints: MarketConstraints) -> None:
        """Test size below minimum fails."""
        is_valid, error = validate_probe_order(Decimal("0.44"), Decimal("0.5"), "BUY", constraints)
        assert is_valid is False
        assert "below min_order_size" in error

    def test_invalid_marketable_buy(self, constraints: MarketConstraints) -> None:
        """Test marketable buy price fails validation."""
        is_valid, error = validate_probe_order(Decimal("0.50"), Decimal("1.0"), "BUY", constraints)
        assert is_valid is False
        assert "marketable" in error

    def test_invalid_marketable_sell(self, constraints: MarketConstraints) -> None:
        """Test marketable sell price fails validation."""
        is_valid, error = validate_probe_order(Decimal("0.45"), Decimal("1.0"), "SELL", constraints)
        assert is_valid is False
        assert "marketable" in error


class TestMarketConstraints:
    """Tests for MarketConstraints class."""

    def test_has_valid_book_both_prices(self) -> None:
        """Test has_valid_book is True when both prices exist."""
        constraints = MarketConstraints(
            token_id="test",
            best_bid=Decimal("0.45"),
            best_ask=Decimal("0.50"),
        )
        assert constraints.has_valid_book is True

    def test_has_valid_book_missing_bid(self) -> None:
        """Test has_valid_book is False when bid is missing."""
        constraints = MarketConstraints(
            token_id="test",
            best_bid=None,
            best_ask=Decimal("0.50"),
        )
        assert constraints.has_valid_book is False

    def test_has_valid_book_missing_ask(self) -> None:
        """Test has_valid_book is False when ask is missing."""
        constraints = MarketConstraints(
            token_id="test",
            best_bid=Decimal("0.45"),
            best_ask=None,
        )
        assert constraints.has_valid_book is False

    def test_default_values(self) -> None:
        """Test default tick_size and min_order_size."""
        constraints = MarketConstraints(token_id="test")
        assert constraints.tick_size == DEFAULT_TICK_SIZE
        assert constraints.min_order_size == DEFAULT_MIN_ORDER_SIZE

    def test_fetch_with_empty_token_id(self) -> None:
        """Test fetch with empty token ID returns error."""
        constraints = MarketConstraints.fetch_for_token("")
        assert constraints.error == "empty_token_id"
