"""Pydantic v2 data models for Polymarket data structures."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Side(str, Enum):
    """Trade side."""

    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    """Market outcome type."""

    YES = "YES"
    NO = "NO"


class SignalType(str, Enum):
    """Signal types for trading strategies."""

    ARBITRAGE = "ARBITRAGE"
    STAT_ARB = "STAT_ARB"


class MarketStatus(str, Enum):
    """Market status."""

    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


# =============================================================================
# Gamma API Models
# =============================================================================


class GammaToken(BaseModel):
    """Token information from Gamma API."""

    model_config = ConfigDict(extra="ignore")

    token_id: str = Field(alias="token_id")
    outcome: str
    price: float = Field(default=0.0)
    winner: bool = Field(default=False)


class GammaMarket(BaseModel):
    """Market data from Gamma API.

    This represents a single market/condition within an event.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    question: str = ""
    condition_id: str = Field(default="", alias="conditionId")
    slug: str = ""
    description: str = ""
    end_date_iso: str | None = Field(default=None, alias="endDateIso")
    game_start_time: str | None = Field(default=None, alias="gameStartTime")
    active: bool = True
    closed: bool = False
    archived: bool = False
    accepting_orders: bool = Field(default=True, alias="acceptingOrders")
    enable_order_book: bool = Field(default=True, alias="enableOrderBook")

    # Pricing
    outcome_prices: str | None = Field(default=None, alias="outcomePrices")
    best_bid: float = Field(default=0.0, alias="bestBid")
    best_ask: float = Field(default=0.0, alias="bestAsk")
    last_trade_price: float = Field(default=0.0, alias="lastTradePrice")
    volume: float = Field(default=0.0)
    volume24hr: float = Field(default=0.0)
    liquidity: float = Field(default=0.0)

    # Token IDs for YES/NO outcomes
    clob_token_ids: str | None = Field(default=None, alias="clobTokenIds")

    # Grouping
    group_item_title: str | None = Field(default=None, alias="groupItemTitle")
    market_maker_address: str | None = Field(default=None, alias="marketMakerAddress")

    # Raw tokens if available
    tokens: list[GammaToken] = Field(default_factory=list)

    @property
    def yes_price(self) -> float:
        """Get YES token price."""
        if self.outcome_prices:
            try:
                prices = [float(p) for p in self.outcome_prices.strip("[]").split(",")]
                return prices[0] if prices else 0.0
            except (ValueError, IndexError):
                pass
        # Try from tokens
        for token in self.tokens:
            if token.outcome.upper() == "YES":
                return token.price
        return self.best_bid if self.best_bid > 0 else 0.5

    @property
    def no_price(self) -> float:
        """Get NO token price."""
        if self.outcome_prices:
            try:
                prices = [float(p) for p in self.outcome_prices.strip("[]").split(",")]
                return prices[1] if len(prices) > 1 else 0.0
            except (ValueError, IndexError):
                pass
        # Try from tokens
        for token in self.tokens:
            if token.outcome.upper() == "NO":
                return token.price
        return 1.0 - self.yes_price

    @property
    def yes_token_id(self) -> str | None:
        """Get YES token ID."""
        if self.clob_token_ids:
            try:
                ids = self.clob_token_ids.strip("[]").replace('"', "").split(",")
                return ids[0].strip() if ids else None
            except (ValueError, IndexError):
                pass
        return None

    @property
    def no_token_id(self) -> str | None:
        """Get NO token ID."""
        if self.clob_token_ids:
            try:
                ids = self.clob_token_ids.strip("[]").replace('"', "").split(",")
                return ids[1].strip() if len(ids) > 1 else None
            except (ValueError, IndexError):
                pass
        return None


class GammaEvent(BaseModel):
    """Event from Gamma API containing multiple markets."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    title: str = ""
    slug: str = ""
    description: str = ""
    start_date: str | None = Field(default=None, alias="startDate")
    end_date: str | None = Field(default=None, alias="endDate")
    active: bool = True
    closed: bool = False
    archived: bool = False
    liquidity: float = Field(default=0.0)
    volume: float = Field(default=0.0)
    volume24hr: float = Field(default=0.0)

    markets: list[GammaMarket] = Field(default_factory=list)


# =============================================================================
# Internal Models for Signal Detection
# =============================================================================


class ArbitrageSignal(BaseModel):
    """Arbitrage opportunity signal."""

    model_config = ConfigDict(frozen=True)

    market_id: str
    market_question: str
    yes_price: float
    no_price: float
    combined_price: float
    profit_potential: float = Field(description="1 - combined_price")
    liquidity: float
    detected_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    @property
    def is_valid(self) -> bool:
        """Check if arbitrage is still valid (combined < 1)."""
        return self.combined_price < 1.0


class StatArbSignal(BaseModel):
    """Statistical arbitrage signal between correlated markets."""

    model_config = ConfigDict(frozen=True)

    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    price_a: float
    price_b: float
    spread: float
    entry_threshold: float
    exit_threshold: float
    direction: str = Field(description="LONG_A_SHORT_B or LONG_B_SHORT_A")
    detected_at: datetime = Field(default_factory=lambda: datetime.utcnow())


# =============================================================================
# Paper Trading Models
# =============================================================================


class PaperTrade(BaseModel):
    """A simulated paper trade."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    strategy: str
    market_id: str
    market_question: str = ""
    side: Side
    outcome: Outcome
    price: float
    quantity: float
    notional: float = Field(description="price * quantity")
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())


class PaperPosition(BaseModel):
    """Current paper position in a market."""

    market_id: str
    market_question: str = ""
    yes_quantity: float = 0.0
    no_quantity: float = 0.0
    avg_price_yes: float = 0.0
    avg_price_no: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    @property
    def has_position(self) -> bool:
        """Check if there's any position."""
        return self.yes_quantity != 0.0 or self.no_quantity != 0.0

    def unrealized_pnl(self, current_yes_price: float, current_no_price: float) -> float:
        """Calculate unrealized PnL at current prices."""
        yes_pnl = (current_yes_price - self.avg_price_yes) * self.yes_quantity
        no_pnl = (current_no_price - self.avg_price_no) * self.no_quantity
        return yes_pnl + no_pnl

    def total_pnl(self, current_yes_price: float, current_no_price: float) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(current_yes_price, current_no_price)


# =============================================================================
# API Response Models
# =============================================================================


class MarketsResponse(BaseModel):
    """Response from Gamma markets endpoint."""

    model_config = ConfigDict(extra="ignore")

    data: list[GammaMarket] = Field(default_factory=list)
    next_cursor: str | None = None


class EventsResponse(BaseModel):
    """Response from Gamma events endpoint."""

    model_config = ConfigDict(extra="ignore")

    data: list[GammaEvent] = Field(default_factory=list)
    next_cursor: str | None = None
