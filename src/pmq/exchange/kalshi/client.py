"""Kalshi exchange client implementing normalized exchange protocols.

Phase 13: Provides KalshiMarketDataClient, KalshiTradingClient, and
KalshiExchangeClient implementing the exchange abstraction protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pmq.auth.kalshi_creds import load_kalshi_creds
from pmq.exchange.kalshi.api import KalshiApi
from pmq.exchange.kalshi.mappers import (
    map_kalshi_balance,
    map_kalshi_market,
    map_kalshi_order_response,
    map_kalshi_orderbook,
    map_kalshi_position,
    to_kalshi_order_params,
)
from pmq.exchange.types import (
    AccountBalance,
    Exchange,
    MarketRef,
    Orderbook,
    OrderRequest,
    OrderResponse,
    Position,
)
from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.auth.kalshi_creds import KalshiCredentials

logger = get_logger("exchange.kalshi.client")


class KalshiMarketDataClient:
    """Kalshi market data client.

    Implements MarketDataClientProtocol for Kalshi exchange.
    Public endpoints only - no credentials required.
    """

    def __init__(self, api: KalshiApi | None = None) -> None:
        """Initialize market data client.

        Args:
            api: KalshiApi instance (creates new one if not provided)
        """
        self._api = api or KalshiApi()
        self._owns_api = api is None

    @property
    def exchange(self) -> Exchange:
        """Get exchange identifier."""
        return Exchange.KALSHI

    def get_market(self, market_id: str) -> MarketRef:
        """Get market reference by ticker.

        Args:
            market_id: Market ticker (e.g., "BTCUSD-24JAN")

        Returns:
            Normalized market reference
        """
        response = self._api.get_market(market_id)
        market_data = response.get("market", response)
        return map_kalshi_market(market_data)

    def get_markets(
        self,
        limit: int = 100,
        status: str | None = "open",
        series_ticker: str | None = None,
    ) -> list[MarketRef]:
        """Get list of markets.

        Args:
            limit: Maximum markets to return
            status: Filter by status (open, closed, settled)
            series_ticker: Filter by series

        Returns:
            List of normalized market references
        """
        response = self._api.get_markets(
            limit=limit,
            status=status,
            series_ticker=series_ticker,
        )
        markets = response.get("markets", [])
        return [map_kalshi_market(m) for m in markets]

    def list_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[MarketRef]:
        """List markets (protocol method).

        Args:
            limit: Maximum markets to return
            active_only: If True, only return open markets

        Returns:
            List of normalized market references
        """
        status = "open" if active_only else None
        return self.get_markets(limit=limit, status=status)

    def get_orderbook(self, market_id: str, depth: int = 10) -> Orderbook:
        """Get orderbook for a market.

        Args:
            market_id: Market ticker
            depth: Number of price levels

        Returns:
            Normalized orderbook
        """
        response = self._api.get_market_orderbook(market_id, depth=depth)
        orderbook_data = response.get("orderbook", response)
        return map_kalshi_orderbook(market_id, orderbook_data)

    def close(self) -> None:
        """Close client resources."""
        if self._owns_api and self._api:
            self._api.close()


class KalshiTradingClient:
    """Kalshi trading client.

    Implements TradingClientProtocol for Kalshi exchange.
    Requires authentication for all operations.
    """

    def __init__(
        self,
        creds: KalshiCredentials | None = None,
        api: KalshiApi | None = None,
    ) -> None:
        """Initialize trading client.

        Args:
            creds: Kalshi credentials (loads from disk if not provided)
            api: KalshiApi instance (creates authenticated one if not provided)
        """
        self._creds = creds or load_kalshi_creds()
        self._api = api or KalshiApi(creds=self._creds)
        self._owns_api = api is None

    @property
    def exchange(self) -> Exchange:
        """Get exchange identifier."""
        return Exchange.KALSHI

    def get_balance(self) -> AccountBalance:
        """Get account balance.

        Returns:
            Normalized account balance
        """
        response = self._api.get_balance()
        return map_kalshi_balance(response)

    def get_positions(self) -> list[Position]:
        """Get account positions.

        Returns:
            List of normalized positions
        """
        positions: list[Position] = []
        cursor: str | None = None

        while True:
            response = self._api.get_positions(limit=100, cursor=cursor)
            position_list = response.get("market_positions", [])

            for pos in position_list:
                # Only include non-zero positions
                if pos.get("position", 0) != 0:
                    positions.append(map_kalshi_position(pos))

            cursor = response.get("cursor")
            if not cursor or not position_list:
                break

        return positions

    def get_open_orders(self, market_id: str | None = None) -> list[OrderResponse]:
        """Get open orders.

        Args:
            market_id: Optional market ticker to filter

        Returns:
            List of normalized order responses
        """
        response = self._api.get_orders(
            ticker=market_id,
            status="resting",
        )
        orders = response.get("orders", [])
        return [map_kalshi_order_response(o) for o in orders]

    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a new order.

        Args:
            request: Normalized order request

        Returns:
            Normalized order response
        """
        params = to_kalshi_order_params(request)
        response = self._api.create_order(**params)
        order_data = response.get("order", response)
        return map_kalshi_order_response(order_data)

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResponse with cancellation status
        """
        from pmq.exchange.types import OrderAction, OrderSide, OrderStatus, OrderType

        try:
            response = self._api.cancel_order(order_id)
            if response and "order" in response:
                return map_kalshi_order_response(response["order"])
            # Return a minimal cancelled response
            return OrderResponse(
                order_id=order_id,
                market_id="",
                status=OrderStatus.CANCELLED,
                side=OrderSide.YES,
                action=OrderAction.BUY,
                order_type=OrderType.LIMIT,
                price=0.0,
                quantity=0.0,
            )
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return OrderResponse(
                order_id=order_id,
                market_id="",
                status=OrderStatus.REJECTED,
                side=OrderSide.YES,
                action=OrderAction.BUY,
                order_type=OrderType.LIMIT,
                price=0.0,
                quantity=0.0,
                error=str(e),
            )

    def close(self) -> None:
        """Close client resources."""
        if self._owns_api and self._api:
            self._api.close()


class KalshiExchangeClient:
    """Combined Kalshi exchange client.

    Implements ExchangeClientProtocol, providing both market data
    and trading functionality.
    """

    def __init__(
        self,
        creds: KalshiCredentials | None = None,
    ) -> None:
        """Initialize exchange client.

        Args:
            creds: Kalshi credentials (loads from disk if not provided)
        """
        self._creds = creds or load_kalshi_creds()
        self._api = KalshiApi(creds=self._creds)

        # Create sub-clients sharing the same API instance
        self._market_data = KalshiMarketDataClient(api=self._api)
        self._trading = KalshiTradingClient(creds=self._creds, api=self._api)

    @property
    def exchange(self) -> Exchange:
        """Get exchange identifier."""
        return Exchange.KALSHI

    @property
    def market_data(self) -> KalshiMarketDataClient:
        """Get market data client."""
        return self._market_data

    @property
    def trading(self) -> KalshiTradingClient:
        """Get trading client."""
        return self._trading

    # ==========================================================================
    # MarketDataClientProtocol methods
    # ==========================================================================

    def get_market(self, market_id: str) -> MarketRef:
        """Get market reference by ticker."""
        return self._market_data.get_market(market_id)

    def get_markets(
        self,
        limit: int = 100,
        status: str | None = "open",
        series_ticker: str | None = None,
    ) -> list[MarketRef]:
        """Get list of markets."""
        return self._market_data.get_markets(
            limit=limit,
            status=status,
            series_ticker=series_ticker,
        )

    def list_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[MarketRef]:
        """List markets (protocol method)."""
        return self._market_data.list_markets(limit=limit, active_only=active_only)

    def get_orderbook(self, market_id: str, depth: int = 10) -> Orderbook:
        """Get orderbook for a market."""
        return self._market_data.get_orderbook(market_id, depth=depth)

    # ==========================================================================
    # TradingClientProtocol methods
    # ==========================================================================

    def get_balance(self) -> AccountBalance:
        """Get account balance."""
        return self._trading.get_balance()

    def get_positions(self) -> list[Position]:
        """Get account positions."""
        return self._trading.get_positions()

    def get_open_orders(self, market_id: str | None = None) -> list[OrderResponse]:
        """Get open orders."""
        return self._trading.get_open_orders(market_id)

    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a new order."""
        return self._trading.place_order(request)

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        return self._trading.cancel_order(order_id)

    # ==========================================================================
    # ExchangeClientProtocol methods
    # ==========================================================================

    def is_authenticated(self) -> bool:
        """Check if client has valid authentication."""
        return self._creds is not None

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    def close(self) -> None:
        """Close all client resources."""
        self._api.close()

    def __enter__(self) -> KalshiExchangeClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
