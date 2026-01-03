"""Type mappers for converting Polymarket types to normalized types.

Phase 13: Converts GammaMarket, OrderBookData, and CLOB responses to
normalized exchange types (MarketRef, Orderbook, OrderResponse, etc.).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pmq.exchange.types import (
    AccountBalance,
    Exchange,
    MarketRef,
    OrderAction,
    Orderbook,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

if TYPE_CHECKING:
    from pmq.markets.orderbook import OrderBookData
    from pmq.models import GammaMarket


def gamma_market_to_market_ref(market: GammaMarket) -> MarketRef:
    """Convert GammaMarket to normalized MarketRef.

    Args:
        market: GammaMarket from Gamma API

    Returns:
        Normalized MarketRef
    """
    return MarketRef(
        exchange=Exchange.POLYMARKET,
        market_id=market.id,
        ticker=market.slug or market.id[:16],
        question=market.question,
        active=market.active and not market.closed,
        closed=market.closed,
        yes_token_id=market.yes_token_id,
        no_token_id=market.no_token_id,
        liquidity=market.liquidity,
        volume_24h=market.volume24hr,
        yes_price=market.yes_price,
        no_price=market.no_price,
        end_date=market.end_date_iso,
        metadata={
            "condition_id": market.condition_id,
            "accepting_orders": market.accepting_orders,
            "enable_order_book": market.enable_order_book,
            "best_bid": market.best_bid,
            "best_ask": market.best_ask,
        },
    )


def orderbook_data_to_orderbook(
    ob: OrderBookData,
    market_id: str,
) -> Orderbook:
    """Convert OrderBookData to normalized Orderbook.

    Note: Polymarket orderbooks are per-token (YES or NO), so a full market
    orderbook requires fetching both YES and NO token orderbooks separately.
    This function converts a single token's orderbook, typically the YES token.

    For YES token orderbook:
    - best_bid = YES bid, best_ask = YES ask
    - NO prices are computed as 1 - YES prices

    Args:
        ob: OrderBookData from orderbook fetcher
        market_id: Market identifier

    Returns:
        Normalized Orderbook
    """
    # Compute NO prices as inverse of YES prices (for binary markets)
    best_no_bid = 1.0 - ob.best_ask if ob.best_ask is not None else None
    best_no_ask = 1.0 - ob.best_bid if ob.best_bid is not None else None

    # Compute mid prices
    mid_yes = ob.mid_price
    mid_no = 1.0 - mid_yes if mid_yes is not None else None

    # Compute spreads
    spread_yes_bps = ob.spread_bps
    spread_no_bps = spread_yes_bps  # Same spread for binary complement

    return Orderbook(
        market_id=market_id,
        exchange=Exchange.POLYMARKET,
        best_yes_bid=ob.best_bid,
        best_yes_ask=ob.best_ask,
        best_no_bid=best_no_bid,
        best_no_ask=best_no_ask,
        best_yes_bid_size=ob.best_bid_size,
        best_yes_ask_size=ob.best_ask_size,
        best_no_bid_size=ob.best_ask_size,  # Inverse liquidity
        best_no_ask_size=ob.best_bid_size,  # Inverse liquidity
        mid_yes=mid_yes,
        mid_no=mid_no,
        spread_yes_bps=spread_yes_bps,
        spread_no_bps=spread_no_bps,
        timestamp=datetime.now(UTC).isoformat(),
        depth_usd=ob.top_depth_usd,
        error=ob.error,
    )


def clob_order_to_order_response(
    clob_response: dict[str, Any],
    order_request_side: OrderSide,
    order_request_action: OrderAction,
) -> OrderResponse:
    """Convert CLOB API order response to normalized OrderResponse.

    Args:
        clob_response: Response from py_clob_client order methods
        order_request_side: Side from original order request
        order_request_action: Action from original order request

    Returns:
        Normalized OrderResponse
    """
    # Extract order ID
    order_id = clob_response.get("orderID") or clob_response.get("order_id") or ""

    # Map status
    status_str = clob_response.get("status", "").upper()
    status_map = {
        "LIVE": OrderStatus.OPEN,
        "OPEN": OrderStatus.OPEN,
        "FILLED": OrderStatus.FILLED,
        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        "CANCELED": OrderStatus.CANCELLED,
        "CANCELLED": OrderStatus.CANCELLED,
        "REJECTED": OrderStatus.REJECTED,
        "EXPIRED": OrderStatus.EXPIRED,
    }
    status = status_map.get(status_str, OrderStatus.PENDING)

    # Extract quantities
    original_size = float(clob_response.get("original_size", 0) or clob_response.get("size", 0))
    remaining_size = float(clob_response.get("size_matched", 0) or 0)
    filled_size = original_size - remaining_size if original_size > 0 else 0

    return OrderResponse(
        order_id=order_id,
        market_id=clob_response.get("asset_id", "") or clob_response.get("token_id", ""),
        status=status,
        side=order_request_side,
        action=order_request_action,
        order_type=OrderType.LIMIT,  # Polymarket CLOB only supports limit orders
        price=float(clob_response.get("price", 0) or 0),
        quantity=original_size,
        filled_quantity=filled_size,
        remaining_quantity=remaining_size,
        avg_fill_price=float(clob_response.get("avg_fill_price", 0) or 0) or None,
        created_at=clob_response.get("created_at", ""),
        updated_at=clob_response.get("updated_at", ""),
        error=clob_response.get("error"),
    )


def clob_orders_to_order_responses(
    clob_orders: list[dict[str, Any]],
) -> list[OrderResponse]:
    """Convert list of CLOB orders to OrderResponses.

    Args:
        clob_orders: List of order dicts from CLOB API

    Returns:
        List of normalized OrderResponses
    """
    responses = []
    for order in clob_orders:
        # Determine side and action from CLOB order
        side_str = order.get("side", "").upper()
        # In Polymarket, we're always buying contracts
        # Side indicates YES vs NO
        if "YES" in side_str or order.get("outcome", "").upper() == "YES":
            side = OrderSide.YES
        else:
            side = OrderSide.NO

        # Action is always BUY for open positions
        action = OrderAction.BUY

        responses.append(clob_order_to_order_response(order, side, action))

    return responses


def clob_balance_to_account_balance(
    balance_response: dict[str, Any],
) -> AccountBalance:
    """Convert CLOB balance response to AccountBalance.

    Note: Polymarket uses USDC for balances.

    Args:
        balance_response: Response from balance API

    Returns:
        Normalized AccountBalance
    """
    available = float(balance_response.get("available", 0) or 0)
    total = float(balance_response.get("total", 0) or balance_response.get("balance", 0) or 0)

    return AccountBalance(
        exchange=Exchange.POLYMARKET,
        available_balance=available,
        total_balance=total if total > 0 else available,
        currency="USDC",
        locked_balance=total - available if total > available else 0,
    )


def clob_positions_to_positions(
    positions_response: list[dict[str, Any]],
) -> list[Position]:
    """Convert CLOB positions response to Position list.

    Args:
        positions_response: List of position dicts from CLOB API

    Returns:
        List of normalized Positions
    """
    positions = []

    # Group by market_id (Polymarket positions are per-token, need to combine YES/NO)
    by_market: dict[str, dict[str, Any]] = {}

    for pos in positions_response:
        market_id = pos.get("market_id", "") or pos.get("condition_id", "")
        if market_id not in by_market:
            by_market[market_id] = {
                "yes_quantity": 0.0,
                "no_quantity": 0.0,
                "avg_yes_price": 0.0,
                "avg_no_price": 0.0,
            }

        outcome = pos.get("outcome", "").upper()
        quantity = float(pos.get("size", 0) or pos.get("quantity", 0) or 0)
        avg_price = float(pos.get("avg_price", 0) or 0)

        if outcome == "YES":
            by_market[market_id]["yes_quantity"] += quantity
            by_market[market_id]["avg_yes_price"] = avg_price
        else:
            by_market[market_id]["no_quantity"] += quantity
            by_market[market_id]["avg_no_price"] = avg_price

    for market_id, data in by_market.items():
        positions.append(
            Position(
                market_id=market_id,
                exchange=Exchange.POLYMARKET,
                yes_quantity=data["yes_quantity"],
                no_quantity=data["no_quantity"],
                avg_yes_price=data["avg_yes_price"],
                avg_no_price=data["avg_no_price"],
            )
        )

    return positions
