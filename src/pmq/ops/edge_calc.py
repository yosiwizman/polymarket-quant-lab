"""Orderbook-based arbitrage edge calculator.

Phase 8: Computes non-zero raw_edge_bps from actual CLOB orderbook data
to enable meaningful calibration and diagnostics.

Arbitrage edge in prediction markets:
- BUY_BOTH (taker): Buy YES at ask + Buy NO at ask → profit if sum < 1.0
- SELL_BOTH (taker): Sell YES at bid + Sell NO at bid → profit if sum > 1.0

Uses Decimal for precise price/size math to avoid floating-point issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmq.markets.orderbook import OrderBookData


class ArbSide(str, Enum):
    """Arbitrage execution side."""

    BUY_BOTH = "BUY_BOTH"  # Buy YES at ask + Buy NO at ask
    SELL_BOTH = "SELL_BOTH"  # Sell YES at bid + Sell NO at bid
    NONE = "NONE"  # No valid arb opportunity


@dataclass
class EdgeResult:
    """Result of orderbook-based edge computation.

    Phase 8: All fields populated from real orderbook data.
    """

    # Token identifiers
    yes_token_id: str
    no_token_id: str

    # Best arb opportunity
    arb_side: ArbSide
    raw_edge_bps: float  # Edge in basis points (can be negative)

    # Price data from orderbooks
    ask_yes: float | None = None  # Best ask for YES token
    ask_no: float | None = None  # Best ask for NO token
    bid_yes: float | None = None  # Best bid for YES token
    bid_no: float | None = None  # Best bid for NO token

    # BUY_BOTH specific
    buy_cost: float | None = None  # ask_yes + ask_no
    buy_edge_bps: float | None = None  # (1.0 - buy_cost) * 10_000

    # SELL_BOTH specific
    sell_revenue: float | None = None  # bid_yes + bid_no
    sell_edge_bps: float | None = None  # (sell_revenue - 1.0) * 10_000

    # Computed metrics
    mid_price: float = 0.5  # Meaningful mid price
    spread_bps: float = 0.0  # Combined spread in bps

    # Depth info (for sizing)
    min_depth_usd: float | None = None  # Min of all relevant depths

    # Error tracking
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "arb_side": self.arb_side.value,
            "raw_edge_bps": round(self.raw_edge_bps, 2),
            "ask_yes": round(self.ask_yes, 6) if self.ask_yes is not None else None,
            "ask_no": round(self.ask_no, 6) if self.ask_no is not None else None,
            "bid_yes": round(self.bid_yes, 6) if self.bid_yes is not None else None,
            "bid_no": round(self.bid_no, 6) if self.bid_no is not None else None,
            "buy_cost": round(self.buy_cost, 6) if self.buy_cost is not None else None,
            "buy_edge_bps": round(self.buy_edge_bps, 2) if self.buy_edge_bps is not None else None,
            "sell_revenue": round(self.sell_revenue, 6) if self.sell_revenue is not None else None,
            "sell_edge_bps": round(self.sell_edge_bps, 2) if self.sell_edge_bps is not None else None,
            "mid_price": round(self.mid_price, 6),
            "spread_bps": round(self.spread_bps, 2),
            "min_depth_usd": round(self.min_depth_usd, 2) if self.min_depth_usd is not None else None,
            "error": self.error,
        }


def _to_decimal(value: float | str | None, default: Decimal | None = None) -> Decimal | None:
    """Safely convert to Decimal."""
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except Exception:
        return default


def compute_arb_edge(
    yes_book: OrderBookData | None,
    no_book: OrderBookData | None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> EdgeResult:
    """Compute arbitrage edge from YES and NO token orderbooks.

    Phase 8: Uses actual bid/ask levels from CLOB to compute real arb edge.

    Two arb strategies:
    1. BUY_BOTH: Buy YES at ask + Buy NO at ask
       - Profit if ask_yes + ask_no < 1.0
       - raw_edge_bps = (1.0 - (ask_yes + ask_no)) * 10_000

    2. SELL_BOTH: Sell YES at bid + Sell NO at bid
       - Profit if bid_yes + bid_no > 1.0
       - raw_edge_bps = ((bid_yes + bid_no) - 1.0) * 10_000

    Args:
        yes_book: OrderBookData for YES token
        no_book: OrderBookData for NO token
        fee_bps: Trading fee in basis points (default: 0)
        slippage_bps: Expected slippage in basis points (default: 0)

    Returns:
        EdgeResult with computed edge, choosing the better of BUY_BOTH/SELL_BOTH
    """
    # Default result for error cases
    yes_token_id = yes_book.token_id if yes_book else ""
    no_token_id = no_book.token_id if no_book else ""

    default_result = EdgeResult(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        arb_side=ArbSide.NONE,
        raw_edge_bps=0.0,
    )

    # Validate inputs
    if not yes_book or not no_book:
        default_result.error = "missing_orderbook"
        return default_result

    if not yes_book.has_valid_book or not no_book.has_valid_book:
        default_result.error = "invalid_book"
        return default_result

    # Extract prices using Decimal for precision
    ask_yes = _to_decimal(yes_book.best_ask)
    ask_no = _to_decimal(no_book.best_ask)
    bid_yes = _to_decimal(yes_book.best_bid)
    bid_no = _to_decimal(no_book.best_bid)

    # Need at least asks for BUY_BOTH or bids for SELL_BOTH
    has_asks = ask_yes is not None and ask_no is not None
    has_bids = bid_yes is not None and bid_no is not None

    if not has_asks and not has_bids:
        default_result.error = "no_prices"
        return default_result

    # Constants
    ONE = Decimal("1.0")
    BPS = Decimal("10000")
    fee_dec = Decimal(str(fee_bps))
    slip_dec = Decimal(str(slippage_bps))
    total_cost_bps = fee_dec + slip_dec  # Applied per leg

    # Compute BUY_BOTH edge (if we have asks)
    buy_edge_bps: Decimal | None = None
    buy_cost: Decimal | None = None
    if has_asks and ask_yes is not None and ask_no is not None:
        buy_cost = ask_yes + ask_no
        # Raw edge before fees
        buy_edge_raw = (ONE - buy_cost) * BPS
        # Apply round-trip cost (2 legs × fee+slippage)
        buy_edge_bps = buy_edge_raw - (total_cost_bps * 2)

    # Compute SELL_BOTH edge (if we have bids)
    sell_edge_bps: Decimal | None = None
    sell_revenue: Decimal | None = None
    if has_bids and bid_yes is not None and bid_no is not None:
        sell_revenue = bid_yes + bid_no
        # Raw edge before fees
        sell_edge_raw = (sell_revenue - ONE) * BPS
        # Apply round-trip cost (2 legs × fee+slippage)
        sell_edge_bps = sell_edge_raw - (total_cost_bps * 2)

    # Choose best side
    if buy_edge_bps is not None and sell_edge_bps is not None:
        if buy_edge_bps >= sell_edge_bps:
            best_side = ArbSide.BUY_BOTH
            best_edge = buy_edge_bps
        else:
            best_side = ArbSide.SELL_BOTH
            best_edge = sell_edge_bps
    elif buy_edge_bps is not None:
        best_side = ArbSide.BUY_BOTH
        best_edge = buy_edge_bps
    elif sell_edge_bps is not None:
        best_side = ArbSide.SELL_BOTH
        best_edge = sell_edge_bps
    else:
        # Shouldn't happen given earlier checks
        default_result.error = "no_edge_computed"
        return default_result

    # Compute mid_price based on best side
    if best_side == ArbSide.BUY_BOTH and ask_yes is not None and ask_no is not None:
        # Mid of YES market (what we're primarily buying)
        mid = (ask_yes + bid_yes) / 2 if bid_yes is not None else ask_yes
    elif best_side == ArbSide.SELL_BOTH and bid_yes is not None and bid_no is not None:
        # Mid of YES market (what we're primarily selling)
        mid = (ask_yes + bid_yes) / 2 if ask_yes is not None else bid_yes
    else:
        mid = Decimal("0.5")

    # Compute combined spread
    # Spread = sum of individual spreads for both legs
    spread_bps_total = Decimal("0")
    if ask_yes is not None and bid_yes is not None and bid_yes > 0:
        yes_spread = ((ask_yes - bid_yes) / bid_yes) * BPS
        spread_bps_total += yes_spread
    if ask_no is not None and bid_no is not None and bid_no > 0:
        no_spread = ((ask_no - bid_no) / bid_no) * BPS
        spread_bps_total += no_spread

    # Compute minimum depth across relevant prices
    depths = []
    if yes_book.top_depth_usd is not None:
        depths.append(yes_book.top_depth_usd)
    if no_book.top_depth_usd is not None:
        depths.append(no_book.top_depth_usd)
    min_depth = min(depths) if depths else None

    # Build result
    return EdgeResult(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        arb_side=best_side,
        raw_edge_bps=float(best_edge.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        ask_yes=float(ask_yes) if ask_yes is not None else None,
        ask_no=float(ask_no) if ask_no is not None else None,
        bid_yes=float(bid_yes) if bid_yes is not None else None,
        bid_no=float(bid_no) if bid_no is not None else None,
        buy_cost=float(buy_cost) if buy_cost is not None else None,
        buy_edge_bps=float(buy_edge_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) if buy_edge_bps is not None else None,
        sell_revenue=float(sell_revenue) if sell_revenue is not None else None,
        sell_edge_bps=float(sell_edge_bps.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) if sell_edge_bps is not None else None,
        mid_price=float(mid.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)),
        spread_bps=float(spread_bps_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        min_depth_usd=min_depth,
    )


def compute_edge_from_prices(
    ask_yes: float | None,
    ask_no: float | None,
    bid_yes: float | None,
    bid_no: float | None,
    yes_token_id: str | None = None,
    no_token_id: str | None = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> EdgeResult:
    """Compute arb edge from raw price values.

    Convenience function when OrderBookData objects aren't available.

    Args:
        ask_yes: Best ask price for YES token
        ask_no: Best ask price for NO token
        bid_yes: Best bid price for YES token
        bid_no: Best bid price for NO token
        yes_token_id: YES token identifier (optional)
        no_token_id: NO token identifier (optional)
        fee_bps: Trading fee in basis points
        slippage_bps: Expected slippage in basis points

    Returns:
        EdgeResult with computed edge
    """
    # Create mock OrderBookData for the compute function
    from dataclasses import dataclass as dc

    @dc
    class MockBook:
        token_id: str
        best_ask: float | None
        best_bid: float | None
        top_depth_usd: float | None = None

        @property
        def has_valid_book(self) -> bool:
            return self.best_bid is not None or self.best_ask is not None

    yes_book = MockBook(
        token_id=yes_token_id or "",
        best_ask=ask_yes,
        best_bid=bid_yes,
    )
    no_book = MockBook(
        token_id=no_token_id or "",
        best_ask=ask_no,
        best_bid=bid_no,
    )

    return compute_arb_edge(
        yes_book,  # type: ignore[arg-type]
        no_book,  # type: ignore[arg-type]
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
