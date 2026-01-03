"""Tests for Kalshi mappers (Phase 13)."""

from __future__ import annotations

import pytest

from pmq.exchange.kalshi.mappers import (
    map_kalshi_balance,
    map_kalshi_market,
    map_kalshi_orderbook,
    map_kalshi_order_response,
    map_kalshi_position,
    to_kalshi_order_params,
)
from pmq.exchange.types import (
    Exchange,
    OrderAction,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestMapKalshiMarket:
    """Tests for map_kalshi_market."""

    def test_basic_mapping(self) -> None:
        market_data = {
            "ticker": "BTCUSD-24JAN",
            "title": "Will BTC exceed $50k?",
            "status": "open",
            "yes_bid": 55,
            "yes_ask": 58,
            "volume": 100000,
            "liquidity": 50000,
        }
        ref = map_kalshi_market(market_data)

        assert ref.exchange == Exchange.KALSHI
        assert ref.market_id == "BTCUSD-24JAN"
        assert ref.ticker == "BTCUSD-24JAN"
        assert ref.active is True
        assert ref.closed is False
        assert ref.yes_price == 0.58  # yes_ask / 100

    def test_closed_market(self) -> None:
        market_data = {
            "ticker": "TEST-001",
            "status": "settled",
        }
        ref = map_kalshi_market(market_data)

        assert ref.active is False
        assert ref.closed is True


class TestMapKalshiOrderbook:
    """Tests for map_kalshi_orderbook."""

    def test_empty_orderbook(self) -> None:
        ob = map_kalshi_orderbook("TEST", {"yes": [], "no": []})

        assert ob.market_id == "TEST"
        assert ob.exchange == Exchange.KALSHI
        assert ob.best_yes_bid is None
        assert ob.best_yes_ask is None

    def test_with_levels(self) -> None:
        orderbook = {
            "yes": [[55, 100], [54, 200]],  # [price_cents, count]
            "no": [[47, 50], [46, 75]],
        }
        ob = map_kalshi_orderbook("MARKET-001", orderbook)

        assert ob.best_yes_bid == 0.55  # 55 / 100
        assert ob.best_no_bid == 0.47

        # YES ask = 1 - NO bid = 1 - 0.47 = 0.53
        assert ob.best_yes_ask == 0.53


class TestMapKalshiPosition:
    """Tests for map_kalshi_position."""

    def test_yes_position(self) -> None:
        pos_data = {
            "ticker": "TEST-001",
            "position": 50,  # Positive = YES
            "realized_pnl": 500,  # In cents
        }
        pos = map_kalshi_position(pos_data)

        assert pos.market_id == "TEST-001"
        assert pos.yes_quantity == 50.0
        assert pos.no_quantity == 0.0
        assert pos.realized_pnl == 5.0  # 500 cents = $5

    def test_no_position(self) -> None:
        pos_data = {
            "ticker": "TEST-001",
            "position": -30,  # Negative = NO
        }
        pos = map_kalshi_position(pos_data)

        assert pos.yes_quantity == 0.0
        assert pos.no_quantity == 30.0


class TestMapKalshiBalance:
    """Tests for map_kalshi_balance."""

    def test_basic_balance(self) -> None:
        response = {"balance": 10050}  # In cents
        bal = map_kalshi_balance(response)

        assert bal.exchange == Exchange.KALSHI
        assert bal.available_balance == 100.50  # $100.50
        assert bal.total_balance == 100.50
        assert bal.currency == "USD"


class TestMapKalshiOrderResponse:
    """Tests for map_kalshi_order_response."""

    def test_resting_order(self) -> None:
        order = {
            "order_id": "ORD-123",
            "ticker": "BTCUSD-24JAN",
            "side": "yes",
            "action": "buy",
            "type": "limit",
            "yes_price": 55,
            "count": 100,
            "remaining_count": 50,
            "status": "resting",
            "created_time": "2024-01-15T10:30:00Z",
        }
        resp = map_kalshi_order_response(order)

        assert resp.order_id == "ORD-123"
        assert resp.status == OrderStatus.OPEN
        assert resp.side == OrderSide.YES
        assert resp.action == OrderAction.BUY
        assert resp.quantity == 100.0
        assert resp.filled_quantity == 50.0

    def test_executed_order(self) -> None:
        order = {
            "order_id": "ORD-456",
            "ticker": "TEST-001",
            "side": "no",
            "action": "sell",
            "type": "limit",
            "no_price": 45,
            "count": 200,
            "remaining_count": 0,
            "status": "executed",
        }
        resp = map_kalshi_order_response(order)

        assert resp.status == OrderStatus.FILLED
        assert resp.side == OrderSide.NO
        assert resp.action == OrderAction.SELL


class TestToKalshiOrderParams:
    """Tests for to_kalshi_order_params."""

    def test_yes_buy_order(self) -> None:
        request = OrderRequest(
            market_id="BTCUSD-24JAN",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            price=0.55,
            quantity=100.0,
        )
        params = to_kalshi_order_params(request)

        assert params["ticker"] == "BTCUSD-24JAN"
        assert params["side"] == "yes"  # Kalshi expects lowercase
        assert params["action"] == "buy"
        assert params["count"] == 100
        assert params["yes_price"] == 55  # 0.55 * 100

    def test_no_sell_order(self) -> None:
        request = OrderRequest(
            market_id="TEST-001",
            side=OrderSide.NO,
            action=OrderAction.SELL,
            order_type=OrderType.LIMIT,
            price=0.40,
            quantity=50.0,
        )
        params = to_kalshi_order_params(request)

        assert params["side"] == "no"  # Kalshi expects lowercase
        assert params["action"] == "sell"
        assert params["no_price"] == 40
