"""Market constraints and price/size quantization utilities.

Phase 13: Enforces exchange tick_size and min_order_size constraints
to ensure all submitted orders are valid and comply with CLOB rules.

DESIGN PRINCIPLES:
- Use Decimal for all price/size calculations to avoid floating-point errors
- Never submit prices that cross the book in probe mode
- Fail-safe: if constraints unknown, use conservative defaults
- All public - no authentication required to fetch constraints

POLYMARKET CLOB DEFAULTS (as of 2024):
- tick_size: 0.01 (1 cent price increments)
- min_order_size: ~0.1-1.0 shares (varies by market)

USAGE:
    from pmq.ops.market_constraints import MarketConstraints, quantize_price, quantize_size

    constraints = MarketConstraints.fetch_for_token(token_id)
    safe_price = quantize_price(0.4567, constraints.tick_size, round_down=True)
    safe_size = quantize_size(5.0, constraints.min_order_size)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import TYPE_CHECKING

import httpx

from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.markets.orderbook import OrderBookData

logger = get_logger("ops.market_constraints")

# Default CLOB API base URL
CLOB_BASE_URL = "https://clob.polymarket.com"

# Conservative defaults for Polymarket CLOB
DEFAULT_TICK_SIZE = Decimal("0.01")  # 1 cent
DEFAULT_MIN_ORDER_SIZE = Decimal("1.0")  # 1 share

# Price bounds (Polymarket prices are 0 < price < 1)
MIN_VALID_PRICE = Decimal("0.01")
MAX_VALID_PRICE = Decimal("0.99")


@dataclass
class MarketConstraints:
    """Exchange constraints for a specific token/market.

    Contains tick_size and min_order_size needed for valid order construction.
    """

    token_id: str
    tick_size: Decimal = DEFAULT_TICK_SIZE
    min_order_size: Decimal = DEFAULT_MIN_ORDER_SIZE

    # Top-of-book for non-marketable order construction
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None

    # Error tracking
    error: str | None = None

    @classmethod
    def fetch_for_token(
        cls,
        token_id: str,
        base_url: str = CLOB_BASE_URL,
        timeout: float = 10.0,
    ) -> MarketConstraints:
        """Fetch market constraints from CLOB book endpoint.

        Args:
            token_id: Token ID to fetch constraints for
            base_url: CLOB API base URL
            timeout: HTTP timeout in seconds

        Returns:
            MarketConstraints with tick_size and min_order_size populated
        """
        result = cls(token_id=token_id)

        if not token_id:
            result.error = "empty_token_id"
            return result

        try:
            with httpx.Client(timeout=timeout) as client:
                # Fetch book data to get tick_size and min_order_size from metadata
                url = f"{base_url.rstrip('/')}/book"
                response = client.get(url, params={"token_id": token_id})
                response.raise_for_status()
                data = response.json()

                # Parse tick_size from response (if present)
                # CLOB may return this as "minimum_tick_size" or similar
                tick_raw = data.get("minimum_tick_size") or data.get("tick_size")
                if tick_raw is not None:
                    try:
                        result.tick_size = Decimal(str(tick_raw))
                        logger.debug(f"Got tick_size={result.tick_size} from API")
                    except Exception:
                        pass  # Keep default

                # Parse min_order_size from response (if present)
                min_size_raw = data.get("minimum_order_size") or data.get("min_order_size")
                if min_size_raw is not None:
                    try:
                        result.min_order_size = Decimal(str(min_size_raw))
                        logger.debug(f"Got min_order_size={result.min_order_size} from API")
                    except Exception:
                        pass  # Keep default

                # Extract best bid/ask for non-marketable order construction
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if bids:
                    # Bids are sorted by price descending
                    best_bid_raw = bids[0].get("price")
                    if best_bid_raw is not None:
                        result.best_bid = Decimal(str(best_bid_raw))

                if asks:
                    # Asks are sorted by price ascending
                    best_ask_raw = asks[0].get("price")
                    if best_ask_raw is not None:
                        result.best_ask = Decimal(str(best_ask_raw))

                logger.info(
                    f"Constraints for {token_id[:16]}...: "
                    f"tick={result.tick_size}, min_size={result.min_order_size}, "
                    f"bid={result.best_bid}, ask={result.best_ask}"
                )

        except httpx.HTTPStatusError as e:
            result.error = f"http_{e.response.status_code}"
            logger.warning(f"HTTP error fetching constraints for {token_id[:16]}...: {e}")
        except httpx.RequestError as e:
            result.error = f"request_error: {type(e).__name__}"
            logger.warning(f"Request error fetching constraints for {token_id[:16]}...: {e}")
        except Exception as e:
            result.error = f"error: {type(e).__name__}"
            logger.warning(f"Error fetching constraints for {token_id[:16]}...: {e}")

        return result

    @classmethod
    def from_orderbook(cls, book: OrderBookData) -> MarketConstraints:
        """Create constraints from existing OrderBookData.

        Uses default tick_size/min_order_size but extracts bid/ask from book.

        Args:
            book: OrderBookData to extract prices from

        Returns:
            MarketConstraints with top-of-book populated
        """
        result = cls(token_id=book.token_id)

        if book.best_bid is not None:
            result.best_bid = Decimal(str(book.best_bid))
        if book.best_ask is not None:
            result.best_ask = Decimal(str(book.best_ask))

        return result

    @property
    def has_valid_book(self) -> bool:
        """Check if we have valid bid and ask prices."""
        return self.best_bid is not None and self.best_ask is not None


def quantize_price(
    price: float | Decimal | str,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
    round_down: bool = True,
) -> Decimal:
    """Quantize a price to the nearest valid tick.

    Args:
        price: Price to quantize
        tick_size: Tick size (price increment)
        round_down: If True, round down (safer for buys); if False, round up

    Returns:
        Price aligned to tick_size

    Examples:
        >>> quantize_price(0.4567, Decimal("0.01"), round_down=True)
        Decimal('0.45')
        >>> quantize_price(0.4567, Decimal("0.01"), round_down=False)
        Decimal('0.46')
    """
    price_dec = Decimal(str(price))

    # Determine rounding mode
    rounding = ROUND_DOWN if round_down else ROUND_UP

    # Quantize to tick size: divide, round, multiply
    ticks = (price_dec / tick_size).to_integral_value(rounding=rounding)
    quantized = ticks * tick_size

    # Enforce bounds
    if quantized < MIN_VALID_PRICE:
        quantized = MIN_VALID_PRICE
    elif quantized > MAX_VALID_PRICE:
        quantized = MAX_VALID_PRICE

    return quantized


def quantize_price_for_buy(
    price: float | Decimal | str,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
) -> Decimal:
    """Quantize price for a BUY order (round down for safety).

    For BUY orders, we want to round DOWN so we don't pay more than intended.

    Args:
        price: Price to quantize
        tick_size: Tick size

    Returns:
        Price aligned to tick_size, rounded down
    """
    return quantize_price(price, tick_size, round_down=True)


def quantize_price_for_sell(
    price: float | Decimal | str,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
) -> Decimal:
    """Quantize price for a SELL order (round up for safety).

    For SELL orders, we want to round UP so we don't sell for less than intended.

    Args:
        price: Price to quantize
        tick_size: Tick size

    Returns:
        Price aligned to tick_size, rounded up
    """
    return quantize_price(price, tick_size, round_down=False)


def quantize_size(
    size: float | Decimal | str,
    min_order_size: Decimal = DEFAULT_MIN_ORDER_SIZE,
    round_down: bool = True,
) -> Decimal:
    """Quantize order size to meet minimum requirements.

    Args:
        size: Desired order size
        min_order_size: Minimum allowed order size
        round_down: If True, round down; if False, round up

    Returns:
        Size that meets minimum requirements

    Examples:
        >>> quantize_size(0.5, Decimal("1.0"))
        Decimal('1.0')
        >>> quantize_size(5.7, Decimal("1.0"), round_down=True)
        Decimal('5')
    """
    size_dec = Decimal(str(size))

    # Ensure at least minimum size
    if size_dec < min_order_size:
        return min_order_size

    # Round to whole units of min_order_size
    rounding = ROUND_DOWN if round_down else ROUND_UP
    units = (size_dec / min_order_size).to_integral_value(rounding=rounding)
    quantized = units * min_order_size

    # Ensure at least minimum after rounding
    if quantized < min_order_size:
        quantized = min_order_size

    return quantized


def compute_non_marketable_buy_price(
    best_bid: Decimal,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
) -> Decimal:
    """Compute a non-marketable BUY price (below best bid).

    For probe orders, we want a price that WON'T fill:
    - BUY order: price below best bid
    - This price will sit on the book without matching

    Args:
        best_bid: Current best bid price
        tick_size: Tick size for quantization

    Returns:
        Non-marketable buy price (best_bid - tick_size), bounded

    Examples:
        >>> compute_non_marketable_buy_price(Decimal("0.45"), Decimal("0.01"))
        Decimal('0.44')
    """
    # Price one tick below best bid
    target = best_bid - tick_size

    # Quantize down for safety
    quantized = quantize_price(target, tick_size, round_down=True)

    # Ensure valid bounds
    if quantized < MIN_VALID_PRICE:
        quantized = MIN_VALID_PRICE

    return quantized


def compute_non_marketable_sell_price(
    best_ask: Decimal,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
) -> Decimal:
    """Compute a non-marketable SELL price (above best ask).

    For probe orders, we want a price that WON'T fill:
    - SELL order: price above best ask
    - This price will sit on the book without matching

    Args:
        best_ask: Current best ask price
        tick_size: Tick size for quantization

    Returns:
        Non-marketable sell price (best_ask + tick_size), bounded

    Examples:
        >>> compute_non_marketable_sell_price(Decimal("0.55"), Decimal("0.01"))
        Decimal('0.56')
    """
    # Price one tick above best ask
    target = best_ask + tick_size

    # Quantize up for safety
    quantized = quantize_price(target, tick_size, round_down=False)

    # Ensure valid bounds
    if quantized > MAX_VALID_PRICE:
        quantized = MAX_VALID_PRICE

    return quantized


def is_price_marketable(
    price: Decimal,
    side: str,
    best_bid: Decimal | None,
    best_ask: Decimal | None,
) -> bool:
    """Check if a price would immediately match (marketable).

    Args:
        price: Order price
        side: "BUY" or "SELL"
        best_bid: Current best bid
        best_ask: Current best ask

    Returns:
        True if order would match immediately, False if it would rest

    Examples:
        >>> is_price_marketable(Decimal("0.50"), "BUY", Decimal("0.45"), Decimal("0.50"))
        True  # BUY at 0.50 crosses the ask at 0.50
        >>> is_price_marketable(Decimal("0.44"), "BUY", Decimal("0.45"), Decimal("0.50"))
        False  # BUY at 0.44 is below best bid, will rest
    """
    side_upper = side.upper()

    if side_upper == "BUY":
        return best_ask is not None and price >= best_ask
    if side_upper == "SELL":
        return best_bid is not None and price <= best_bid
    return False


def validate_probe_order(
    price: Decimal,
    size: Decimal,
    side: str,
    constraints: MarketConstraints,
) -> tuple[bool, str]:
    """Validate a probe order against constraints.

    Checks:
    1. Price is properly quantized to tick_size
    2. Size meets min_order_size
    3. Price is within valid bounds
    4. Price is NON-MARKETABLE (won't fill)

    Args:
        price: Order price
        size: Order size
        side: "BUY" or "SELL"
        constraints: Market constraints

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check price bounds
    if price < MIN_VALID_PRICE or price > MAX_VALID_PRICE:
        return False, f"Price {price} outside valid range [{MIN_VALID_PRICE}, {MAX_VALID_PRICE}]"

    # Check tick alignment
    expected_price = quantize_price(price, constraints.tick_size)
    if price != expected_price:
        return False, f"Price {price} not aligned to tick_size {constraints.tick_size}"

    # Check minimum size
    if size < constraints.min_order_size:
        return False, f"Size {size} below min_order_size {constraints.min_order_size}"

    # Check marketability (probe orders MUST be non-marketable)
    if is_price_marketable(price, side, constraints.best_bid, constraints.best_ask):
        return False, f"Price {price} is marketable (would fill immediately)"

    return True, "OK"
