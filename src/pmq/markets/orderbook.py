"""Order book fetching and microstructure computation.

Fetches order books from Polymarket CLOB public endpoints and computes:
- best_bid, best_ask
- mid_price (if both sides exist)
- spread_bps = ((ask - bid) / mid) * 10_000
- top_depth_usd (conservative: min of bid/ask size × price)

All public endpoints - no authentication required.
"""

from dataclasses import dataclass, field

import httpx

from pmq.logging import get_logger

logger = get_logger("markets.orderbook")

# Polymarket CLOB public endpoint
CLOB_BASE_URL = "https://clob.polymarket.com"


@dataclass
class OrderBookLevel:
    """Single level in an order book."""

    price: float
    size: float

    @property
    def notional(self) -> float:
        """Notional value at this level (price × size)."""
        return self.price * self.size


@dataclass
class OrderBookData:
    """Order book data with computed microstructure metrics.

    All fields are optional to support graceful degradation when
    order book data is unavailable or incomplete.
    """

    token_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    best_bid_size: float | None = None
    best_ask_size: float | None = None
    mid_price: float | None = None
    spread_bps: float | None = None
    top_depth_usd: float | None = None
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    error: str | None = None

    @property
    def has_valid_book(self) -> bool:
        """Check if we have valid bid and ask data."""
        return self.best_bid is not None and self.best_ask is not None

    def to_dict(self) -> dict[str, object]:
        """Convert to dict for storage/serialization."""
        return {
            "token_id": self.token_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "best_bid_size": self.best_bid_size,
            "best_ask_size": self.best_ask_size,
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "top_depth_usd": self.top_depth_usd,
            "error": self.error,
        }


def compute_microstructure(
    best_bid: float | None,
    best_ask: float | None,
    best_bid_size: float | None,
    best_ask_size: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Compute microstructure metrics from top-of-book.

    Args:
        best_bid: Best bid price
        best_ask: Best ask price
        best_bid_size: Size at best bid
        best_ask_size: Size at best ask

    Returns:
        Tuple of (mid_price, spread_bps, top_depth_usd)
    """
    mid_price: float | None = None
    spread_bps: float | None = None
    top_depth_usd: float | None = None

    # Mid price requires both bid and ask
    if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
        mid_price = (best_bid + best_ask) / 2.0

        # Spread in basis points: ((ask - bid) / mid) * 10_000
        if mid_price > 0:
            spread_bps = ((best_ask - best_bid) / mid_price) * 10_000

    # Top depth: conservative estimate using min of bid/ask notional
    # This represents the "guaranteed" fillable depth
    if best_bid is not None and best_bid_size is not None:
        bid_notional = best_bid * best_bid_size
    else:
        bid_notional = None

    if best_ask is not None and best_ask_size is not None:
        ask_notional = best_ask * best_ask_size
    else:
        ask_notional = None

    if bid_notional is not None and ask_notional is not None:
        top_depth_usd = min(bid_notional, ask_notional)
    elif bid_notional is not None:
        top_depth_usd = bid_notional
    elif ask_notional is not None:
        top_depth_usd = ask_notional

    return mid_price, spread_bps, top_depth_usd


class OrderBookFetcher:
    """Fetches order books from Polymarket CLOB public endpoints.

    Uses the public order book endpoint that doesn't require authentication.
    Handles errors gracefully - returns OrderBookData with error field set.
    """

    def __init__(
        self,
        base_url: str = CLOB_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the fetcher.

        Args:
            base_url: CLOB API base URL
            timeout: HTTP timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout)
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OrderBookFetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch_order_book(self, token_id: str) -> OrderBookData:
        """Fetch order book for a token ID.

        Args:
            token_id: The CLOB token ID (from GammaMarket.yes_token_id or no_token_id)

        Returns:
            OrderBookData with microstructure metrics (or error if fetch failed)
        """
        result = OrderBookData(token_id=token_id)

        if not token_id:
            result.error = "empty_token_id"
            return result

        try:
            client = self._get_client()
            # Public order book endpoint
            url = f"{self._base_url}/book"
            params = {"token_id": token_id}

            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse bids and asks
            bids_data = data.get("bids", [])
            asks_data = data.get("asks", [])

            # Parse into OrderBookLevel objects
            for bid in bids_data:
                try:
                    level = OrderBookLevel(
                        price=float(bid.get("price", 0)),
                        size=float(bid.get("size", 0)),
                    )
                    if level.price > 0 and level.size > 0:
                        result.bids.append(level)
                except (ValueError, TypeError):
                    continue

            for ask in asks_data:
                try:
                    level = OrderBookLevel(
                        price=float(ask.get("price", 0)),
                        size=float(ask.get("size", 0)),
                    )
                    if level.price > 0 and level.size > 0:
                        result.asks.append(level)
                except (ValueError, TypeError):
                    continue

            # Sort: bids descending by price, asks ascending by price
            result.bids.sort(key=lambda x: x.price, reverse=True)
            result.asks.sort(key=lambda x: x.price)

            # Extract top of book
            if result.bids:
                result.best_bid = result.bids[0].price
                result.best_bid_size = result.bids[0].size

            if result.asks:
                result.best_ask = result.asks[0].price
                result.best_ask_size = result.asks[0].size

            # Compute microstructure
            mid, spread, depth = compute_microstructure(
                result.best_bid,
                result.best_ask,
                result.best_bid_size,
                result.best_ask_size,
            )
            result.mid_price = mid
            result.spread_bps = spread
            result.top_depth_usd = depth

            logger.debug(
                f"Fetched order book for {token_id[:16]}...: "
                f"bid={result.best_bid}, ask={result.best_ask}, "
                f"spread_bps={result.spread_bps:.1f}"
                if result.spread_bps
                else f"Fetched order book for {token_id[:16]}...: no spread"
            )

        except httpx.HTTPStatusError as e:
            result.error = f"http_{e.response.status_code}"
            logger.warning(f"HTTP error fetching order book for {token_id[:16]}...: {e}")
        except httpx.RequestError as e:
            result.error = f"request_error: {type(e).__name__}"
            logger.warning(f"Request error fetching order book for {token_id[:16]}...: {e}")
        except Exception as e:
            result.error = f"error: {type(e).__name__}"
            logger.warning(f"Error fetching order book for {token_id[:16]}...: {e}")

        return result

    def fetch_order_books_batch(
        self,
        token_ids: list[str],
        continue_on_error: bool = True,
    ) -> dict[str, OrderBookData]:
        """Fetch order books for multiple tokens.

        Args:
            token_ids: List of token IDs to fetch
            continue_on_error: If True, continue fetching even if some fail

        Returns:
            Dict mapping token_id to OrderBookData
        """
        results: dict[str, OrderBookData] = {}

        for token_id in token_ids:
            if not token_id:
                continue

            try:
                result = self.fetch_order_book(token_id)
                results[token_id] = result
            except Exception as e:
                if continue_on_error:
                    results[token_id] = OrderBookData(
                        token_id=token_id,
                        error=f"batch_error: {type(e).__name__}",
                    )
                else:
                    raise

        return results
