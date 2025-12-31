"""Market data adapters for Polymarket microstructure data.

This module provides:
- Gamma API helpers for market/token ID resolution
- Order book fetching from CLOB public endpoints
- Microstructure computation (spread, liquidity, mid_price)

All functions use public endpoints only - no authentication required.
"""

from pmq.markets.orderbook import (
    OrderBookData,
    OrderBookFetcher,
    OrderBookLevel,
)

__all__ = [
    "OrderBookData",
    "OrderBookFetcher",
    "OrderBookLevel",
]
