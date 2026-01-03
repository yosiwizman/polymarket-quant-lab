"""Polymarket exchange client implementations.

Phase 13: Wraps existing Polymarket modules to implement normalized
exchange protocols. Composes GammaClient, OrderBookFetcher, and
py_clob_client to provide a unified interface.

DESIGN PRINCIPLES:
- Composition: Wraps existing modules without modification
- Lazy initialization: Clients created on first use
- Resource management: Proper cleanup via close()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pmq.exchange.polymarket.mappers import (
    clob_order_to_order_response,
    clob_orders_to_order_responses,
    gamma_market_to_market_ref,
    orderbook_data_to_orderbook,
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
from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.auth.creds import ClobCredentials
    from pmq.gamma_client import GammaClient
    from pmq.markets.orderbook import OrderBookFetcher

logger = get_logger("exchange.polymarket")


class PolymarketMarketDataClient:
    """Polymarket market data client (read-only, no auth required).

    Wraps GammaClient for market listing and OrderBookFetcher for orderbooks.
    """

    def __init__(
        self,
        gamma_client: GammaClient | None = None,
        orderbook_fetcher: OrderBookFetcher | None = None,
    ) -> None:
        """Initialize market data client.

        Args:
            gamma_client: Optional pre-configured GammaClient
            orderbook_fetcher: Optional pre-configured OrderBookFetcher
        """
        self._gamma_client = gamma_client
        self._ob_fetcher = orderbook_fetcher
        self._owns_gamma = gamma_client is None
        self._owns_ob = orderbook_fetcher is None

        # Cache for market lookups
        self._market_cache: dict[str, MarketRef] = {}

    def _get_gamma_client(self) -> GammaClient:
        """Get or create Gamma client."""
        if self._gamma_client is None:
            from pmq.gamma_client import GammaClient

            self._gamma_client = GammaClient()
        return self._gamma_client

    def _get_ob_fetcher(self) -> OrderBookFetcher:
        """Get or create orderbook fetcher."""
        if self._ob_fetcher is None:
            from pmq.markets.orderbook import OrderBookFetcher

            self._ob_fetcher = OrderBookFetcher()
        return self._ob_fetcher

    @property
    def exchange(self) -> Exchange:
        """Return exchange identifier."""
        return Exchange.POLYMARKET

    def list_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[MarketRef]:
        """List available markets.

        Args:
            limit: Maximum number of markets
            active_only: Only return active markets

        Returns:
            List of MarketRef
        """
        client = self._get_gamma_client()
        gamma_markets = client.list_markets(limit=limit)

        markets = []
        for gm in gamma_markets:
            market_ref = gamma_market_to_market_ref(gm)
            if active_only and not market_ref.active:
                continue
            markets.append(market_ref)
            # Cache for later lookup
            self._market_cache[market_ref.market_id] = market_ref

        return markets

    def get_market(self, market_id: str) -> MarketRef | None:
        """Get a specific market by ID.

        Args:
            market_id: Market identifier

        Returns:
            MarketRef if found
        """
        # Check cache first
        if market_id in self._market_cache:
            return self._market_cache[market_id]

        # Fetch from API
        client = self._get_gamma_client()
        try:
            gamma_market = client.get_market(market_id)
            if gamma_market:
                market_ref = gamma_market_to_market_ref(gamma_market)
                self._market_cache[market_id] = market_ref
                return market_ref
        except Exception as e:
            logger.debug(f"Failed to get market {market_id}: {e}")

        return None

    def get_orderbook(self, market_id: str) -> Orderbook:
        """Get orderbook for a market.

        Args:
            market_id: Market identifier

        Returns:
            Orderbook snapshot
        """
        # Need token ID for orderbook fetch
        market = self.get_market(market_id)
        if market is None or market.yes_token_id is None:
            return Orderbook(
                market_id=market_id,
                exchange=Exchange.POLYMARKET,
                error="market_not_found_or_no_token_id",
            )

        fetcher = self._get_ob_fetcher()
        ob_data = fetcher.fetch_order_book(market.yes_token_id)
        return orderbook_data_to_orderbook(ob_data, market_id)

    def close(self) -> None:
        """Close resources."""
        if self._owns_gamma and self._gamma_client is not None:
            self._gamma_client.close()
            self._gamma_client = None
        if self._owns_ob and self._ob_fetcher is not None:
            self._ob_fetcher.close()
            self._ob_fetcher = None


class PolymarketTradingClient:
    """Polymarket trading client (requires auth).

    Wraps py_clob_client for order management.
    """

    def __init__(
        self,
        creds: ClobCredentials,
        clob_client: Any | None = None,
    ) -> None:
        """Initialize trading client.

        Args:
            creds: Polymarket CLOB credentials
            clob_client: Optional pre-configured ClobClient
        """
        self._creds = creds
        self._clob_client = clob_client
        self._owns_clob = clob_client is None

    def _get_clob_client(self) -> Any:
        """Get or create CLOB client."""
        if self._clob_client is None:
            from pmq.auth.client_factory import create_clob_client

            self._clob_client = create_clob_client(creds=self._creds)
        return self._clob_client

    @property
    def exchange(self) -> Exchange:
        """Return exchange identifier."""
        return Exchange.POLYMARKET

    def get_balance(self) -> AccountBalance:
        """Get account balance.

        Returns:
            AccountBalance
        """
        # Note: py_clob_client doesn't have a direct balance API
        # This would need the Polymarket API or on-chain balance check
        # For now, return a placeholder
        logger.warning("Polymarket balance API not directly supported, returning placeholder")
        return AccountBalance(
            exchange=Exchange.POLYMARKET,
            available_balance=0.0,
            total_balance=0.0,
            currency="USDC",
        )

    def get_positions(self) -> list[Position]:
        """Get open positions.

        Returns:
            List of Position
        """
        # Note: Would need Polymarket positions API
        logger.warning("Polymarket positions API not directly supported, returning empty")
        return []

    def get_open_orders(self, market_id: str | None = None) -> list[OrderResponse]:
        """Get open orders.

        Args:
            market_id: Optional market filter

        Returns:
            List of open orders
        """
        client = self._get_clob_client()
        try:
            params = {}
            if market_id:
                params["asset_id"] = market_id
            orders = client.get_orders(params) if params else client.get_orders()
            if isinstance(orders, list):
                return clob_orders_to_order_responses(orders)
            return []
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order.

        Args:
            order: OrderRequest with trade details

        Returns:
            OrderResponse with status
        """
        try:
            from py_clob_client.order import (
                BUY,
                SELL,
                OrderArgs,
            )

            # Map action to CLOB side
            clob_side = BUY if order.action == OrderAction.BUY else SELL

            # Need token_id, not market_id for Polymarket
            # The market_id should be the token_id for YES/NO
            order_args = OrderArgs(
                token_id=order.market_id,  # In Polymarket, we use token_id
                price=order.price,
                size=order.quantity,
                side=clob_side,
            )

            client = self._get_clob_client()
            response = client.create_and_post_order(order_args)

            return clob_order_to_order_response(response, order.side, order.action)

        except Exception as e:
            from pmq.auth.redact import redact_secrets

            safe_error = redact_secrets(str(e))
            logger.error(f"Order placement failed: {safe_error}")
            return OrderResponse(
                order_id="",
                market_id=order.market_id,
                status=OrderStatus.REJECTED,
                side=order.side,
                action=order.action,
                order_type=order.order_type,
                price=order.price,
                quantity=order.quantity,
                error=safe_error,
            )

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResponse with status
        """
        try:
            client = self._get_clob_client()
            response = client.cancel_order(order_id)

            return OrderResponse(
                order_id=order_id,
                market_id="",
                status=OrderStatus.CANCELLED,
                side=OrderSide.YES,  # Unknown, use default
                action=OrderAction.BUY,  # Unknown, use default
                order_type=OrderType.LIMIT,
                price=0.0,
                quantity=0.0,
                error=response.get("error") if isinstance(response, dict) else None,
            )

        except Exception as e:
            from pmq.auth.redact import redact_secrets

            safe_error = redact_secrets(str(e))
            logger.error(f"Order cancellation failed: {safe_error}")
            return OrderResponse(
                order_id=order_id,
                market_id="",
                status=OrderStatus.REJECTED,
                side=OrderSide.YES,
                action=OrderAction.BUY,
                order_type=OrderType.LIMIT,
                price=0.0,
                quantity=0.0,
                error=safe_error,
            )

    def close(self) -> None:
        """Close resources."""
        # ClobClient doesn't have a close method
        self._clob_client = None


class PolymarketExchangeClient(PolymarketMarketDataClient):
    """Full Polymarket client with market data and trading.

    Combines PolymarketMarketDataClient and PolymarketTradingClient.
    """

    def __init__(
        self,
        creds: ClobCredentials,
        gamma_client: GammaClient | None = None,
        orderbook_fetcher: OrderBookFetcher | None = None,
        clob_client: Any | None = None,
    ) -> None:
        """Initialize full client.

        Args:
            creds: Polymarket CLOB credentials
            gamma_client: Optional pre-configured GammaClient
            orderbook_fetcher: Optional pre-configured OrderBookFetcher
            clob_client: Optional pre-configured ClobClient
        """
        super().__init__(gamma_client, orderbook_fetcher)
        self._trading_client = PolymarketTradingClient(creds, clob_client)
        self._creds = creds

    def get_balance(self) -> AccountBalance:
        """Get account balance."""
        return self._trading_client.get_balance()

    def get_positions(self) -> list[Position]:
        """Get open positions."""
        return self._trading_client.get_positions()

    def get_open_orders(self, market_id: str | None = None) -> list[OrderResponse]:
        """Get open orders."""
        return self._trading_client.get_open_orders(market_id)

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order."""
        return self._trading_client.place_order(order)

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an existing order."""
        return self._trading_client.cancel_order(order_id)

    def is_authenticated(self) -> bool:
        """Check if client has valid authentication."""
        return self._creds is not None

    def close(self) -> None:
        """Close all resources."""
        super().close()
        self._trading_client.close()
