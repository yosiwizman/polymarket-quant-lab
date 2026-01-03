"""Factory functions for creating exchange clients.

Phase 13: Provides a single entry point for creating exchange clients
based on exchange name, handling credential loading and validation.

DESIGN PRINCIPLES:
- Single responsibility: Factory only creates clients
- Validation: Checks exchange name and credentials
- Safe by default: No trading without explicit authentication

USAGE:
    from pmq.exchange import get_exchange_client, get_market_data_client

    # Market data only (no auth required)
    market_client = get_market_data_client("kalshi")
    markets = market_client.list_markets()

    # Full client with trading (auth required)
    client = get_exchange_client("kalshi")
    balance = client.get_balance()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pmq.exchange.types import Exchange
from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.exchange.protocols import (
        ExchangeClientProtocol,
        MarketDataClientProtocol,
        TradingClientProtocol,
    )

logger = get_logger("exchange.factory")


class ExchangeNotSupportedError(Exception):
    """Raised when an unsupported exchange is requested."""

    pass


class ExchangeAuthError(Exception):
    """Raised when authentication fails or credentials are missing."""

    pass


def _validate_exchange(exchange: str) -> Exchange:
    """Validate and normalize exchange name.

    Args:
        exchange: Exchange name (case-insensitive)

    Returns:
        Exchange enum value

    Raises:
        ExchangeNotSupportedError: If exchange is not supported
    """
    exchange_lower = exchange.lower()
    try:
        return Exchange(exchange_lower)
    except ValueError:
        supported = ", ".join(e.value for e in Exchange)
        raise ExchangeNotSupportedError(
            f"Exchange '{exchange}' is not supported. Supported exchanges: {supported}"
        ) from None


def get_market_data_client(
    exchange: str,
    **kwargs: Any,
) -> MarketDataClientProtocol:
    """Get a market data client for the specified exchange.

    Market data clients provide read-only access to market information
    and orderbooks. No authentication required for most exchanges.

    Args:
        exchange: Exchange name ("polymarket" or "kalshi")
        **kwargs: Exchange-specific options

    Returns:
        MarketDataClientProtocol implementation

    Raises:
        ExchangeNotSupportedError: If exchange is not supported
    """
    exchange_enum = _validate_exchange(exchange)

    if exchange_enum == Exchange.POLYMARKET:
        from pmq.exchange.polymarket.client import PolymarketMarketDataClient

        return PolymarketMarketDataClient(**kwargs)

    elif exchange_enum == Exchange.KALSHI:
        from pmq.exchange.kalshi.client import KalshiMarketDataClient

        return KalshiMarketDataClient(**kwargs)

    raise ExchangeNotSupportedError(f"Exchange '{exchange}' is not yet implemented")


def get_trading_client(
    exchange: str,
    creds_dir: Path | None = None,
    **kwargs: Any,
) -> TradingClientProtocol:
    """Get a trading client for the specified exchange.

    Trading clients require authentication and can place/cancel orders.

    Args:
        exchange: Exchange name ("polymarket" or "kalshi")
        creds_dir: Optional custom credentials directory
        **kwargs: Exchange-specific options

    Returns:
        TradingClientProtocol implementation

    Raises:
        ExchangeNotSupportedError: If exchange is not supported
        ExchangeAuthError: If credentials are missing or invalid
    """
    exchange_enum = _validate_exchange(exchange)

    if exchange_enum == Exchange.POLYMARKET:
        from pmq.auth.creds import load_creds
        from pmq.exchange.polymarket.client import PolymarketTradingClient

        creds = load_creds(creds_dir)
        if creds is None:
            raise ExchangeAuthError(
                "Polymarket credentials not found. Run 'pmq ops auth init' to create credentials."
            )

        return PolymarketTradingClient(creds=creds, **kwargs)

    elif exchange_enum == Exchange.KALSHI:
        from pmq.auth.kalshi_creds import load_kalshi_creds
        from pmq.exchange.kalshi.client import KalshiTradingClient

        kalshi_creds = load_kalshi_creds(creds_dir)
        if kalshi_creds is None:
            raise ExchangeAuthError(
                "Kalshi credentials not found. Run 'pmq ops auth init --exchange kalshi' "
                "to configure credentials."
            )

        return KalshiTradingClient(creds=kalshi_creds, **kwargs)

    raise ExchangeNotSupportedError(f"Exchange '{exchange}' is not yet implemented")


def get_exchange_client(
    exchange: str,
    creds_dir: Path | None = None,
    **kwargs: Any,
) -> ExchangeClientProtocol:
    """Get a full exchange client with market data and trading.

    This is a convenience function that returns a client implementing
    both MarketDataClientProtocol and TradingClientProtocol.

    Args:
        exchange: Exchange name ("polymarket" or "kalshi")
        creds_dir: Optional custom credentials directory
        **kwargs: Exchange-specific options

    Returns:
        ExchangeClientProtocol implementation

    Raises:
        ExchangeNotSupportedError: If exchange is not supported
        ExchangeAuthError: If credentials are missing or invalid
    """
    exchange_enum = _validate_exchange(exchange)

    if exchange_enum == Exchange.POLYMARKET:
        from pmq.auth.creds import load_creds
        from pmq.exchange.polymarket.client import PolymarketExchangeClient

        creds = load_creds(creds_dir)
        if creds is None:
            raise ExchangeAuthError(
                "Polymarket credentials not found. Run 'pmq ops auth init' to create credentials."
            )

        return PolymarketExchangeClient(creds=creds, **kwargs)

    elif exchange_enum == Exchange.KALSHI:
        from pmq.auth.kalshi_creds import load_kalshi_creds
        from pmq.exchange.kalshi.client import KalshiExchangeClient

        kalshi_creds = load_kalshi_creds(creds_dir)
        if kalshi_creds is None:
            raise ExchangeAuthError(
                "Kalshi credentials not found. Run 'pmq ops auth init --exchange kalshi' "
                "to configure credentials."
            )

        return KalshiExchangeClient(creds=kalshi_creds, **kwargs)

    raise ExchangeNotSupportedError(f"Exchange '{exchange}' is not yet implemented")
