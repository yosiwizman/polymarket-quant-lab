"""Exchange abstraction layer for multi-exchange support.

Phase 13: Provides a unified interface for interacting with prediction market
exchanges (Polymarket, Kalshi, etc.) through normalized protocols and types.

DESIGN PRINCIPLES:
- Protocol-based: Use typing.Protocol for interface definitions
- Normalized types: Exchange-agnostic dataclasses for cross-exchange compatibility
- Factory pattern: Single entry point for client creation
- Safe by default: No trading without explicit configuration

USAGE:
    from pmq.exchange import get_exchange_client, Exchange, MarketRef, Orderbook

    # Get client for specific exchange
    client = get_exchange_client("kalshi")

    # List markets
    markets = client.list_markets(limit=100)

    # Get orderbook
    ob = client.get_orderbook(market_id)
"""

from pmq.exchange.factory import (
    get_exchange_client,
    get_market_data_client,
    get_trading_client,
)
from pmq.exchange.protocols import (
    ExchangeClientProtocol,
    MarketDataClientProtocol,
    TradingClientProtocol,
)
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

__all__ = [
    # Protocols
    "ExchangeClientProtocol",
    "MarketDataClientProtocol",
    "TradingClientProtocol",
    # Types
    "Exchange",
    "MarketRef",
    "Orderbook",
    "OrderSide",
    "OrderAction",
    "OrderType",
    "OrderStatus",
    "OrderRequest",
    "OrderResponse",
    "Position",
    "AccountBalance",
    # Factory
    "get_exchange_client",
    "get_market_data_client",
    "get_trading_client",
]
