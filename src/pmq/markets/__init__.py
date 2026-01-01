"""Market data adapters for Polymarket microstructure data.

This module provides:
- Gamma API helpers for market/token ID resolution
- Order book fetching from CLOB public endpoints
- WebSocket streaming for real-time order book updates
- Microstructure computation (spread, liquidity, mid_price)

All functions use public endpoints only - no authentication required.
"""

from pmq.markets.orderbook import (
    OrderBookData,
    OrderBookFetcher,
    OrderBookLevel,
)
from pmq.markets.wss_market import (
    MarketWssClient,
    WssStats,
)

__all__ = [
    "MarketWssClient",
    "OrderBookData",
    "OrderBookFetcher",
    "OrderBookLevel",
    "WssStats",
]
