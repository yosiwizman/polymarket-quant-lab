"""Tests for exchange types and mappers (Phase 13)."""

from __future__ import annotations

import pytest

from pmq.exchange.types import (
    AccountBalance,
    Exchange,
    MarketRef,
    Orderbook,
    OrderAction,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class TestExchangeEnum:
    """Tests for Exchange enum."""

    def test_polymarket_value(self) -> None:
        assert Exchange.POLYMARKET.value == "polymarket"

    def test_kalshi_value(self) -> None:
        assert Exchange.KALSHI.value == "kalshi"


class TestMarketRef:
    """Tests for MarketRef dataclass."""

    def test_create_minimal(self) -> None:
        ref = MarketRef(
            exchange=Exchange.KALSHI,
            market_id="TEST-001",
        )
        assert ref.exchange == Exchange.KALSHI
        assert ref.market_id == "TEST-001"
        assert ref.active is True
        assert ref.closed is False

    def test_combined_price(self) -> None:
        ref = MarketRef(
            exchange=Exchange.KALSHI,
            market_id="TEST-001",
            yes_price=0.60,
            no_price=0.42,
        )
        assert abs(ref.combined_price - 1.02) < 0.01

    def test_to_dict(self) -> None:
        ref = MarketRef(
            exchange=Exchange.POLYMARKET,
            market_id="abc123",
            ticker="BTC-YES",
        )
        d = ref.to_dict()
        assert d["exchange"] == "polymarket"
        assert d["market_id"] == "abc123"
        assert d["ticker"] == "BTC-YES"


class TestOrderbook:
    """Tests for Orderbook dataclass."""

    def test_has_valid_book_empty(self) -> None:
        ob = Orderbook(market_id="TEST")
        assert ob.has_valid_book is False

    def test_has_valid_book_yes_side(self) -> None:
        ob = Orderbook(
            market_id="TEST",
            best_yes_bid=0.55,
            best_yes_ask=0.60,
        )
        assert ob.has_valid_book is True

    def test_combined_ask(self) -> None:
        ob = Orderbook(
            market_id="TEST",
            best_yes_ask=0.55,
            best_no_ask=0.48,
        )
        assert abs(ob.combined_ask - 1.03) < 0.01  # type: ignore[operator]

    def test_combined_bid(self) -> None:
        ob = Orderbook(
            market_id="TEST",
            best_yes_bid=0.52,
            best_no_bid=0.45,
        )
        assert abs(ob.combined_bid - 0.97) < 0.01  # type: ignore[operator]


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_create(self) -> None:
        req = OrderRequest(
            market_id="MARKET-001",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            price=0.55,
            quantity=100.0,
        )
        assert req.side == OrderSide.YES
        assert req.action == OrderAction.BUY
        assert req.quantity == 100.0

    def test_to_dict(self) -> None:
        req = OrderRequest(
            market_id="M1",
            side=OrderSide.NO,
            action=OrderAction.SELL,
            order_type=OrderType.MARKET,
            price=0.45,
            quantity=50.0,
        )
        d = req.to_dict()
        assert d["side"] == "NO"
        assert d["action"] == "SELL"


class TestOrderResponse:
    """Tests for OrderResponse dataclass."""

    def test_is_open(self) -> None:
        resp = OrderResponse(
            order_id="ORD-001",
            market_id="M1",
            status=OrderStatus.OPEN,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            price=0.50,
            quantity=100.0,
        )
        assert resp.is_open is True
        assert resp.is_filled is False

    def test_is_filled(self) -> None:
        resp = OrderResponse(
            order_id="ORD-002",
            market_id="M1",
            status=OrderStatus.FILLED,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            price=0.50,
            quantity=100.0,
            filled_quantity=100.0,
        )
        assert resp.is_open is False
        assert resp.is_filled is True


class TestPosition:
    """Tests for Position dataclass."""

    def test_has_position_empty(self) -> None:
        pos = Position(market_id="M1")
        assert pos.has_position is False

    def test_has_position_yes(self) -> None:
        pos = Position(market_id="M1", yes_quantity=50.0)
        assert pos.has_position is True

    def test_total_pnl(self) -> None:
        pos = Position(
            market_id="M1",
            realized_pnl=10.0,
            unrealized_pnl=5.0,
        )
        assert abs(pos.total_pnl - 15.0) < 0.01


class TestAccountBalance:
    """Tests for AccountBalance dataclass."""

    def test_create(self) -> None:
        bal = AccountBalance(
            exchange=Exchange.KALSHI,
            available_balance=100.0,
            total_balance=150.0,
            locked_balance=50.0,
        )
        assert bal.available_balance == 100.0
        assert bal.currency == "USD"

    def test_to_dict(self) -> None:
        bal = AccountBalance(
            exchange=Exchange.POLYMARKET,
            available_balance=250.0,
            total_balance=250.0,
            currency="USDC",
        )
        d = bal.to_dict()
        assert d["exchange"] == "polymarket"
        assert d["currency"] == "USDC"
