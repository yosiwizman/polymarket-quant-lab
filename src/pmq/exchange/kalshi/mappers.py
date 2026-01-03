"""Mappers to convert Kalshi API responses to normalized exchange types.

Phase 13: Converts Kalshi-specific data structures to the generic exchange types
defined in pmq.exchange.types.
"""

from __future__ import annotations

from typing import Any

from pmq.exchange.types import (
    AccountBalance,
    Exchange,
    MarketRef,
    OrderAction,
    Orderbook,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


def map_kalshi_market(market: dict[str, Any]) -> MarketRef:
    """Convert Kalshi market response to MarketRef.

    Kalshi market structure (relevant fields):
    - ticker: Market ticker (e.g., "BTCUSD-24JAN")
    - title: Market title/question
    - status: "open", "closed", "settled"
    - yes_bid: Best yes bid price (cents)
    - yes_ask: Best yes ask price (cents)
    - volume: Total volume traded
    - liquidity: Available liquidity

    Args:
        market: Kalshi market dict

    Returns:
        Normalized MarketRef
    """
    ticker = market.get("ticker", "")

    # Kalshi uses cents (1-99) for prices
    yes_bid = market.get("yes_bid", 0)
    yes_ask = market.get("yes_ask", 0)

    # Calculate best prices (convert cents to 0-1 range)
    yes_price = yes_ask / 100 if yes_ask else 0.0
    no_price = (100 - yes_bid) / 100 if yes_bid else 0.0

    # Liquidity in cents -> dollars
    liquidity = market.get("liquidity", 0)
    volume = market.get("volume", 0)
    status = market.get("status", "open")

    return MarketRef(
        exchange=Exchange.KALSHI,
        market_id=ticker,
        ticker=ticker,
        yes_token_id=f"{ticker}:yes",
        no_token_id=f"{ticker}:no",
        question=market.get("title", ""),
        active=status == "open",
        closed=status in ("closed", "settled"),
        yes_price=yes_price,
        no_price=no_price,
        liquidity=liquidity / 100 if liquidity else 0.0,
        volume_24h=volume / 100 if volume else 0.0,
    )


def map_kalshi_orderbook(ticker: str, orderbook: dict[str, Any]) -> Orderbook:
    """Convert Kalshi orderbook response to normalized Orderbook.

    Kalshi orderbook structure:
    - yes: [[price, count], ...] - sorted by price desc (bids for YES)
    - no: [[price, count], ...] - sorted by price desc (bids for NO)

    Prices are in cents (1-99). Each level is [price_cents, contract_count].

    Args:
        ticker: Market ticker
        orderbook: Kalshi orderbook dict

    Returns:
        Normalized Orderbook
    """
    yes_levels = orderbook.get("yes", [])
    no_levels = orderbook.get("no", [])

    # Extract best bid/ask from orderbook
    best_yes_bid: float | None = None
    best_yes_bid_size: float | None = None
    best_no_bid: float | None = None
    best_no_bid_size: float | None = None

    if yes_levels and len(yes_levels[0]) >= 2:
        best_yes_bid = yes_levels[0][0] / 100
        best_yes_bid_size = float(yes_levels[0][1])

    if no_levels and len(no_levels[0]) >= 2:
        best_no_bid = no_levels[0][0] / 100
        best_no_bid_size = float(no_levels[0][1])

    # YES ask = implied from NO bid (buy NO at X = sell YES at 1-X)
    best_yes_ask: float | None = None
    best_yes_ask_size: float | None = None
    if best_no_bid is not None:
        best_yes_ask = 1.0 - best_no_bid
        best_yes_ask_size = best_no_bid_size

    # NO ask = implied from YES bid
    best_no_ask: float | None = None
    best_no_ask_size: float | None = None
    if best_yes_bid is not None:
        best_no_ask = 1.0 - best_yes_bid
        best_no_ask_size = best_yes_bid_size

    # Compute mid prices and spreads
    mid_yes: float | None = None
    spread_yes_bps: float | None = None
    if best_yes_bid is not None and best_yes_ask is not None:
        mid_yes = (best_yes_bid + best_yes_ask) / 2
        spread_yes_bps = (best_yes_ask - best_yes_bid) * 10000

    mid_no: float | None = None
    spread_no_bps: float | None = None
    if best_no_bid is not None and best_no_ask is not None:
        mid_no = (best_no_bid + best_no_ask) / 2
        spread_no_bps = (best_no_ask - best_no_bid) * 10000

    return Orderbook(
        market_id=ticker,
        exchange=Exchange.KALSHI,
        best_yes_bid=best_yes_bid,
        best_yes_ask=best_yes_ask,
        best_no_bid=best_no_bid,
        best_no_ask=best_no_ask,
        best_yes_bid_size=best_yes_bid_size,
        best_yes_ask_size=best_yes_ask_size,
        best_no_bid_size=best_no_bid_size,
        best_no_ask_size=best_no_ask_size,
        mid_yes=mid_yes,
        mid_no=mid_no,
        spread_yes_bps=spread_yes_bps,
        spread_no_bps=spread_no_bps,
    )


def map_kalshi_position(position: dict[str, Any]) -> Position:
    """Convert Kalshi position response to normalized Position.

    Kalshi position structure:
    - ticker: Market ticker
    - position: Number of contracts (positive = YES, negative = NO)
    - market_exposure: Maximum loss
    - realized_pnl: Realized P&L in cents

    Args:
        position: Kalshi position dict

    Returns:
        Normalized Position
    """
    ticker = position.get("ticker", "")
    contracts = position.get("position", 0)

    # Positive position = YES, negative = NO
    yes_qty = float(contracts) if contracts > 0 else 0.0
    no_qty = float(abs(contracts)) if contracts < 0 else 0.0

    # P&L in cents -> dollars
    realized_pnl_cents = position.get("realized_pnl", 0)

    return Position(
        market_id=ticker,
        exchange=Exchange.KALSHI,
        yes_quantity=yes_qty,
        no_quantity=no_qty,
        avg_yes_price=0.0,
        avg_no_price=0.0,
        realized_pnl=realized_pnl_cents / 100 if realized_pnl_cents else 0.0,
        unrealized_pnl=0.0,
    )


def map_kalshi_balance(balance_response: dict[str, Any]) -> AccountBalance:
    """Convert Kalshi balance response to normalized AccountBalance.

    Kalshi balance structure:
    - balance: Available balance in cents

    Args:
        balance_response: Kalshi balance dict

    Returns:
        Normalized AccountBalance
    """
    balance_cents = balance_response.get("balance", 0)
    balance_usd = balance_cents / 100

    return AccountBalance(
        exchange=Exchange.KALSHI,
        available_balance=balance_usd,
        total_balance=balance_usd,
        locked_balance=0.0,
        currency="USD",
    )


def map_kalshi_order_response(order: dict[str, Any]) -> OrderResponse:
    """Convert Kalshi order response to normalized OrderResponse.

    Kalshi order structure:
    - order_id: Order ID
    - ticker: Market ticker
    - side: "yes" or "no"
    - action: "buy" or "sell"
    - type: "limit" or "market"
    - yes_price / no_price: Price in cents
    - count: Total contracts
    - remaining_count: Unfilled contracts
    - status: "resting", "pending", "canceled", "executed"
    - created_time: ISO timestamp

    Args:
        order: Kalshi order dict

    Returns:
        Normalized OrderResponse
    """
    ticker = order.get("ticker", "")
    side_str = order.get("side", "yes").lower()
    side = OrderSide.YES if side_str == "yes" else OrderSide.NO

    # Get action - buy or sell
    action_str = order.get("action", "buy").lower()
    action = OrderAction.BUY if action_str == "buy" else OrderAction.SELL

    # Get order type
    type_str = order.get("type", "limit").lower()
    order_type = OrderType.LIMIT if type_str == "limit" else OrderType.MARKET

    # Get price based on side (cents -> 0-1)
    price_cents = order.get("yes_price") or order.get("no_price") or 0
    price = price_cents / 100

    # Map status - note: CANCELLED has two L's in our enum
    status_str = order.get("status", "").lower()
    status_map = {
        "resting": OrderStatus.OPEN,
        "pending": OrderStatus.PENDING,
        "canceled": OrderStatus.CANCELLED,
        "executed": OrderStatus.FILLED,
    }
    status = status_map.get(status_str, OrderStatus.PENDING)

    total = float(order.get("count", 0))
    remaining = float(order.get("remaining_count", 0))
    filled = total - remaining

    return OrderResponse(
        order_id=order.get("order_id", ""),
        market_id=ticker,
        status=status,
        side=side,
        action=action,
        order_type=order_type,
        price=price,
        quantity=total,
        filled_quantity=filled,
        remaining_quantity=remaining,
        client_order_id=order.get("client_order_id"),
        created_at=order.get("created_time", ""),
    )


def to_kalshi_order_params(request: OrderRequest) -> dict[str, Any]:
    """Convert normalized OrderRequest to Kalshi API parameters.

    Args:
        request: Normalized order request

    Returns:
        Dict of Kalshi API parameters
    """
    # Extract side - Kalshi expects lowercase
    side = request.side.value.lower()  # "yes" or "no"

    # Price in cents (1-99)
    price_cents = int(request.price * 100)

    # Map action - Kalshi expects lowercase
    action = request.action.value.lower()  # "buy" or "sell"

    # Map order type - Kalshi expects lowercase
    order_type = request.order_type.value.lower()  # "limit" or "market"

    params: dict[str, Any] = {
        "ticker": request.market_id,
        "side": side,
        "action": action,
        "count": int(request.quantity),
        "type": order_type,
    }

    # Set price based on side
    if side == "yes":
        params["yes_price"] = price_cents
    else:
        params["no_price"] = price_cents

    if request.client_order_id:
        params["client_order_id"] = request.client_order_id

    return params
