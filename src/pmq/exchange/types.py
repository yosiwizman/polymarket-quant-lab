"""Normalized types for cross-exchange compatibility.

Phase 13: Exchange-agnostic dataclasses that provide a unified representation
of market data, orders, and positions across different prediction markets.

All exchanges (Polymarket, Kalshi, etc.) map their native types to these
normalized types for consistent handling throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Exchange(str, Enum):
    """Supported prediction market exchanges."""

    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class OrderSide(str, Enum):
    """Side of a prediction market outcome."""

    YES = "YES"
    NO = "NO"


class OrderAction(str, Enum):
    """Trading action."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    """Order execution status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class MarketRef:
    """Normalized market reference.

    Represents a single prediction market across exchanges.
    For binary markets, YES and NO are the two outcomes.

    Attributes:
        exchange: Source exchange
        market_id: Exchange-specific market identifier
        ticker: Human-readable ticker (e.g., "BTCUSD-24JAN" for Kalshi)
        question: Market question or title
        active: Whether market is accepting orders
        closed: Whether market has resolved
        yes_token_id: Token ID for YES outcome (Polymarket-specific)
        no_token_id: Token ID for NO outcome (Polymarket-specific)
        liquidity: Total liquidity in USD
        volume_24h: 24-hour trading volume
        yes_price: Last YES price (0-1)
        no_price: Last NO price (0-1)
        end_date: Market end/resolution date
        metadata: Exchange-specific additional data
    """

    exchange: Exchange
    market_id: str
    ticker: str = ""
    question: str = ""
    active: bool = True
    closed: bool = False
    yes_token_id: str | None = None
    no_token_id: str | None = None
    liquidity: float = 0.0
    volume_24h: float = 0.0
    yes_price: float = 0.0
    no_price: float = 0.0
    end_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_price(self) -> float:
        """Sum of YES and NO prices (should be ~1.0 for efficient markets)."""
        return self.yes_price + self.no_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange": self.exchange.value,
            "market_id": self.market_id,
            "ticker": self.ticker,
            "question": self.question,
            "active": self.active,
            "closed": self.closed,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "liquidity": self.liquidity,
            "volume_24h": self.volume_24h,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "end_date": self.end_date,
            "metadata": self.metadata,
        }


@dataclass
class Orderbook:
    """Normalized orderbook snapshot.

    Represents the current state of bids and asks for a market.
    For binary markets, we track YES and NO sides separately.

    Attributes:
        market_id: Market identifier
        exchange: Source exchange
        best_yes_bid: Best bid price for YES outcome
        best_yes_ask: Best ask price for YES outcome
        best_no_bid: Best bid price for NO outcome
        best_no_ask: Best ask price for NO outcome
        best_yes_bid_size: Size at best YES bid
        best_yes_ask_size: Size at best YES ask
        best_no_bid_size: Size at best NO bid
        best_no_ask_size: Size at best NO ask
        mid_yes: Mid price for YES ((bid + ask) / 2)
        mid_no: Mid price for NO
        spread_yes_bps: YES spread in basis points
        spread_no_bps: NO spread in basis points
        timestamp: When orderbook was captured
        depth_usd: Estimated fillable depth in USD
        error: Error message if fetch failed
    """

    market_id: str
    exchange: Exchange = Exchange.POLYMARKET
    best_yes_bid: float | None = None
    best_yes_ask: float | None = None
    best_no_bid: float | None = None
    best_no_ask: float | None = None
    best_yes_bid_size: float | None = None
    best_yes_ask_size: float | None = None
    best_no_bid_size: float | None = None
    best_no_ask_size: float | None = None
    mid_yes: float | None = None
    mid_no: float | None = None
    spread_yes_bps: float | None = None
    spread_no_bps: float | None = None
    timestamp: str = ""
    depth_usd: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def has_valid_book(self) -> bool:
        """Check if we have valid bid/ask data for at least one side."""
        yes_valid = self.best_yes_bid is not None and self.best_yes_ask is not None
        no_valid = self.best_no_bid is not None and self.best_no_ask is not None
        return yes_valid or no_valid

    @property
    def combined_ask(self) -> float | None:
        """Sum of YES ask and NO ask (cost to buy both outcomes)."""
        if self.best_yes_ask is not None and self.best_no_ask is not None:
            return self.best_yes_ask + self.best_no_ask
        return None

    @property
    def combined_bid(self) -> float | None:
        """Sum of YES bid and NO bid (revenue from selling both outcomes)."""
        if self.best_yes_bid is not None and self.best_no_bid is not None:
            return self.best_yes_bid + self.best_no_bid
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "exchange": self.exchange.value,
            "best_yes_bid": self.best_yes_bid,
            "best_yes_ask": self.best_yes_ask,
            "best_no_bid": self.best_no_bid,
            "best_no_ask": self.best_no_ask,
            "best_yes_bid_size": self.best_yes_bid_size,
            "best_yes_ask_size": self.best_yes_ask_size,
            "best_no_bid_size": self.best_no_bid_size,
            "best_no_ask_size": self.best_no_ask_size,
            "mid_yes": self.mid_yes,
            "mid_no": self.mid_no,
            "spread_yes_bps": self.spread_yes_bps,
            "spread_no_bps": self.spread_no_bps,
            "timestamp": self.timestamp,
            "depth_usd": self.depth_usd,
            "error": self.error,
        }


@dataclass
class OrderRequest:
    """Request to place an order.

    Attributes:
        market_id: Market to trade
        side: YES or NO outcome
        action: BUY or SELL
        order_type: LIMIT or MARKET
        price: Limit price (0-1 for prediction markets)
        quantity: Number of contracts/shares
        client_order_id: Optional client-assigned order ID
        time_in_force: Order validity (e.g., "GTC", "IOC")
    """

    market_id: str
    side: OrderSide
    action: OrderAction
    order_type: OrderType
    price: float
    quantity: float
    client_order_id: str | None = None
    time_in_force: str = "GTC"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "side": self.side.value,
            "action": self.action.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "quantity": self.quantity,
            "client_order_id": self.client_order_id,
            "time_in_force": self.time_in_force,
        }


@dataclass
class OrderResponse:
    """Response from order placement or query.

    Attributes:
        order_id: Exchange-assigned order ID
        market_id: Market the order is for
        status: Current order status
        side: YES or NO outcome
        action: BUY or SELL
        order_type: LIMIT or MARKET
        price: Order price
        quantity: Original order quantity
        filled_quantity: Quantity that has been filled
        remaining_quantity: Quantity still open
        avg_fill_price: Average fill price
        created_at: When order was created
        updated_at: When order was last updated
        error: Error message if order failed
        client_order_id: Client-assigned order ID if provided
    """

    order_id: str
    market_id: str
    status: OrderStatus
    side: OrderSide
    action: OrderAction
    order_type: OrderType
    price: float
    quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: float | None = None
    created_at: str = ""
    updated_at: str = ""
    error: str | None = None
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity - self.filled_quantity

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "market_id": self.market_id,
            "status": self.status.value,
            "side": self.side.value,
            "action": self.action.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "client_order_id": self.client_order_id,
        }


@dataclass
class Position:
    """Current position in a market.

    Attributes:
        market_id: Market identifier
        exchange: Source exchange
        yes_quantity: Quantity of YES contracts held
        no_quantity: Quantity of NO contracts held
        avg_yes_price: Average entry price for YES
        avg_no_price: Average entry price for NO
        unrealized_pnl: Unrealized PnL at current prices
        realized_pnl: Realized PnL from closed positions
        market_value: Current market value of position
        updated_at: When position was last updated
    """

    market_id: str
    exchange: Exchange = Exchange.POLYMARKET
    yes_quantity: float = 0.0
    no_quantity: float = 0.0
    avg_yes_price: float = 0.0
    avg_no_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = datetime.now(UTC).isoformat()

    @property
    def has_position(self) -> bool:
        """Check if there's any non-zero position."""
        return self.yes_quantity != 0.0 or self.no_quantity != 0.0

    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "exchange": self.exchange.value,
            "yes_quantity": self.yes_quantity,
            "no_quantity": self.no_quantity,
            "avg_yes_price": self.avg_yes_price,
            "avg_no_price": self.avg_no_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "market_value": self.market_value,
            "updated_at": self.updated_at,
        }


@dataclass
class AccountBalance:
    """Account balance information.

    Attributes:
        exchange: Source exchange
        available_balance: Balance available for trading
        total_balance: Total account balance (including locked)
        currency: Currency code (e.g., "USD", "USDC")
        locked_balance: Balance locked in open orders
        updated_at: When balance was last updated
    """

    exchange: Exchange
    available_balance: float
    total_balance: float
    currency: str = "USD"
    locked_balance: float = 0.0
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange": self.exchange.value,
            "available_balance": self.available_balance,
            "total_balance": self.total_balance,
            "currency": self.currency,
            "locked_balance": self.locked_balance,
            "updated_at": self.updated_at,
        }
