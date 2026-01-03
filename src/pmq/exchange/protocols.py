"""Protocol definitions for exchange client interfaces.

Phase 13: Defines typing.Protocol interfaces for exchange clients to enable
duck-typing and testability across different exchange implementations.

DESIGN PRINCIPLES:
- Protocol-based: No inheritance required, structural subtyping
- Minimal: Only methods actually used by the application
- Async-ready: Methods can be sync or async (implementation-dependent)
- Safe by default: Trading methods clearly separated from market data

USAGE:
    from pmq.exchange.protocols import MarketDataClientProtocol

    def process_markets(client: MarketDataClientProtocol) -> None:
        markets = client.list_markets(limit=100)
        for market in markets:
            ob = client.get_orderbook(market.market_id)
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pmq.exchange.types import (
        AccountBalance,
        Exchange,
        MarketRef,
        Orderbook,
        OrderRequest,
        OrderResponse,
        Position,
    )


@runtime_checkable
class MarketDataClientProtocol(Protocol):
    """Protocol for market data operations.

    Provides read-only access to market information and orderbooks.
    No trading or account operations.
    """

    @property
    def exchange(self) -> Exchange:
        """Return the exchange this client connects to."""
        ...

    def list_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[MarketRef]:
        """List available markets.

        Args:
            limit: Maximum number of markets to return
            active_only: If True, only return active (tradeable) markets

        Returns:
            List of MarketRef objects
        """
        ...

    def get_market(self, market_id: str) -> MarketRef | None:
        """Get a specific market by ID.

        Args:
            market_id: Exchange-specific market identifier

        Returns:
            MarketRef if found, None otherwise
        """
        ...

    def get_orderbook(self, market_id: str) -> Orderbook:
        """Get current orderbook for a market.

        Args:
            market_id: Exchange-specific market identifier

        Returns:
            Orderbook snapshot (may have error field set if fetch failed)
        """
        ...


@runtime_checkable
class TradingClientProtocol(Protocol):
    """Protocol for trading operations.

    Provides account and order management.
    REQUIRES AUTHENTICATION.
    """

    @property
    def exchange(self) -> Exchange:
        """Return the exchange this client connects to."""
        ...

    def get_balance(self) -> AccountBalance:
        """Get account balance.

        Returns:
            AccountBalance with available/total balance
        """
        ...

    def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of Position objects
        """
        ...

    def get_open_orders(self, market_id: str | None = None) -> list[OrderResponse]:
        """Get open orders.

        Args:
            market_id: Optional market filter

        Returns:
            List of open OrderResponse objects
        """
        ...

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order.

        Args:
            order: OrderRequest with trade details

        Returns:
            OrderResponse with order status

        Raises:
            Exception: If order placement fails
        """
        ...

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an existing order.

        Args:
            order_id: Exchange-assigned order ID

        Returns:
            OrderResponse with cancellation status

        Raises:
            Exception: If cancellation fails
        """
        ...


@runtime_checkable
class ExchangeClientProtocol(MarketDataClientProtocol, TradingClientProtocol, Protocol):
    """Combined protocol for full exchange client functionality.

    Extends both MarketDataClientProtocol and TradingClientProtocol.
    Use this when you need both market data and trading capabilities.
    """

    def close(self) -> None:
        """Close the client and release resources.

        Should be called when done with the client.
        """
        ...

    def is_authenticated(self) -> bool:
        """Check if client has valid authentication.

        Returns:
            True if client can perform authenticated operations
        """
        ...
