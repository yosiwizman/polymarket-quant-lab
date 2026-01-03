"""Polymarket exchange client implementation.

Phase 13: Wraps existing Polymarket modules (gamma_client, orderbook, py-clob-client)
to implement the normalized exchange protocols.

This module provides:
- PolymarketMarketDataClient: Read-only market data
- PolymarketTradingClient: Trading operations (requires auth)
- PolymarketExchangeClient: Combined client
"""

from pmq.exchange.polymarket.client import (
    PolymarketExchangeClient,
    PolymarketMarketDataClient,
    PolymarketTradingClient,
)

__all__ = [
    "PolymarketMarketDataClient",
    "PolymarketTradingClient",
    "PolymarketExchangeClient",
]
