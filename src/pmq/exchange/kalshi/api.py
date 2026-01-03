"""Kalshi REST API wrapper with authentication and rate limiting.

Phase 13: Implements a lightweight HTTP client for Kalshi's Trading API v2.

AUTHENTICATION:
Kalshi uses RSA signatures for API authentication. Each request requires:
1. Timestamp header
2. Signature header: RSA signature of (timestamp + method + path)

RATE LIMITING:
- Public endpoints: 10 requests/second
- Private endpoints: 10 requests/second
- Built-in rate limiting with token bucket

API DOCUMENTATION:
https://trading-api.readme.io/reference/getting-started
"""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from pmq.auth.redact import redact_secrets
from pmq.logging import get_logger

if TYPE_CHECKING:
    from pmq.auth.kalshi_creds import KalshiCredentials

logger = get_logger("exchange.kalshi.api")

# Default timeouts
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 10.0

# Rate limit settings (Kalshi allows ~10 req/sec)
DEFAULT_RPS = 8.0
DEFAULT_BURST = 8


class KalshiApiError(Exception):
    """Base exception for Kalshi API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KalshiAuthError(KalshiApiError):
    """Authentication error (401/403)."""

    pass


class KalshiRateLimitError(KalshiApiError):
    """Rate limit exceeded (429)."""

    pass


class KalshiApi:
    """Low-level Kalshi API wrapper.

    Handles authentication, rate limiting, and HTTP request/response.
    """

    def __init__(
        self,
        creds: KalshiCredentials | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        rps: float = DEFAULT_RPS,
        burst: int = DEFAULT_BURST,
    ) -> None:
        """Initialize API client.

        Args:
            creds: Kalshi credentials (optional for public endpoints)
            timeout: Request timeout in seconds
            rps: Requests per second limit
            burst: Burst capacity
        """
        self._creds = creds
        self._timeout = httpx.Timeout(timeout, connect=DEFAULT_CONNECT_TIMEOUT)
        self._rps = rps
        self._burst = burst

        # Token bucket for rate limiting
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

        # HTTP client (lazy init)
        self._client: httpx.Client | None = None

        # RSA key (lazy init from creds)
        self._private_key: Any = None

    def _get_base_url(self) -> str:
        """Get API base URL from credentials or default to production."""
        if self._creds:
            return self._creds.api_base
        from pmq.auth.kalshi_creds import KALSHI_API_BASE_PROD

        return KALSHI_API_BASE_PROD

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def _load_private_key(self) -> Any:
        """Load RSA private key from credentials.

        Returns:
            RSA private key object
        """
        if self._private_key is not None:
            return self._private_key

        if self._creds is None:
            raise KalshiAuthError("No credentials provided")

        try:
            from cryptography.hazmat.primitives import serialization

            # Try to load as PEM
            key_bytes = self._creds.api_secret.encode()
            self._private_key = serialization.load_pem_private_key(
                key_bytes,
                password=None,
            )
            return self._private_key
        except ImportError:
            raise KalshiAuthError(
                "cryptography package required for Kalshi auth. "
                "Install with: pip install cryptography"
            ) from None
        except Exception as exc:
            raise KalshiAuthError(f"Failed to load private key: {exc}") from exc

    def _sign_request(
        self,
        method: str,
        path: str,
        timestamp: str,
    ) -> str:
        """Create RSA signature for request.

        Kalshi signature: RSA-SHA256(timestamp + method + path)

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g., /trade-api/v2/markets)
            timestamp: ISO timestamp

        Returns:
            Base64-encoded signature
        """
        private_key = self._load_private_key()

        # Message to sign: timestamp + method + path
        message = f"{timestamp}{method}{path}".encode()

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            signature = private_key.sign(
                message,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            raise KalshiAuthError(f"Failed to sign request: {e}") from e

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Refill tokens
        self._tokens = min(
            self._burst,
            self._tokens + elapsed * self._rps,
        )
        self._last_refill = now

        # Wait if no tokens
        if self._tokens < 1.0:
            sleep_time = (1.0 - self._tokens) / self._rps
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self._tokens = 1.0
            self._last_refill = time.monotonic()

        self._tokens -= 1.0

    def _make_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth_required: bool = False,
    ) -> dict[str, Any]:
        """Make HTTP request to Kalshi API.

        Args:
            method: HTTP method
            path: API path (without base URL)
            params: Query parameters
            json_body: JSON body for POST/PUT
            auth_required: Whether authentication is required

        Returns:
            JSON response as dict

        Raises:
            KalshiApiError: On API error
            KalshiAuthError: On authentication error
            KalshiRateLimitError: On rate limit exceeded
        """
        self._wait_for_rate_limit()

        base_url = self._get_base_url()
        url = f"{base_url}{path}"

        headers: dict[str, str] = {}

        # Add auth headers if required
        if auth_required:
            if self._creds is None:
                raise KalshiAuthError("Authentication required but no credentials provided")

            timestamp = datetime.now(UTC).isoformat()
            signature = self._sign_request(method.upper(), path, timestamp)

            headers["KALSHI-ACCESS-KEY"] = self._creds.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        try:
            client = self._get_client()
            response = client.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers=headers,
            )

            # Check for errors
            if response.status_code == 401 or response.status_code == 403:
                raise KalshiAuthError(
                    f"Authentication failed: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            elif response.status_code == 429:
                raise KalshiRateLimitError(
                    "Rate limit exceeded",
                    status_code=429,
                    response_body=response.text,
                )
            elif response.status_code >= 400:
                raise KalshiApiError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            result: dict[str, Any] = response.json()
            return result

        except httpx.RequestError as e:
            safe_error = redact_secrets(str(e))
            raise KalshiApiError(f"Request failed: {safe_error}") from e

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ==========================================================================
    # Public Market Data Endpoints
    # ==========================================================================

    def get_markets(
        self,
        limit: int = 100,
        status: str | None = "open",
        cursor: str | None = None,
        series_ticker: str | None = None,
    ) -> dict[str, Any]:
        """Get list of markets.

        Args:
            limit: Max markets to return (max 200)
            status: Filter by status (open, closed, settled)
            cursor: Pagination cursor
            series_ticker: Filter by series

        Returns:
            Markets response with 'markets' list and 'cursor'
        """
        params: dict[str, Any] = {"limit": min(limit, 200)}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker

        return self._make_request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict[str, Any]:
        """Get a single market by ticker.

        Args:
            ticker: Market ticker (e.g., "BTCUSD-24JAN")

        Returns:
            Market data
        """
        return self._make_request("GET", f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        """Get orderbook for a market.

        Args:
            ticker: Market ticker
            depth: Number of levels (default 10)

        Returns:
            Orderbook with 'yes' and 'no' sides
        """
        params = {"depth": depth}
        return self._make_request("GET", f"/markets/{ticker}/orderbook", params=params)

    # ==========================================================================
    # Authenticated Trading Endpoints
    # ==========================================================================

    def get_balance(self) -> dict[str, Any]:
        """Get account balance.

        Returns:
            Balance response with 'balance' field (in cents)
        """
        return self._make_request("GET", "/portfolio/balance", auth_required=True)

    def get_positions(
        self,
        limit: int = 100,
        cursor: str | None = None,
        settlement_status: str | None = None,
    ) -> dict[str, Any]:
        """Get account positions.

        Args:
            limit: Max positions to return
            cursor: Pagination cursor
            settlement_status: Filter by status (unsettled, settled)

        Returns:
            Positions response
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if settlement_status:
            params["settlement_status"] = settlement_status

        return self._make_request("GET", "/portfolio/positions", params=params, auth_required=True)

    def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = "resting",
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Get orders.

        Args:
            ticker: Filter by market ticker
            status: Filter by status (resting, pending, canceled, executed)
            limit: Max orders to return
            cursor: Pagination cursor

        Returns:
            Orders response
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        return self._make_request("GET", "/portfolio/orders", params=params, auth_required=True)

    def create_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # Number of contracts
        type_: str = "limit",
        yes_price: int | None = None,  # Price in cents (1-99)
        no_price: int | None = None,
        client_order_id: str | None = None,
        expiration_ts: int | None = None,
    ) -> dict[str, Any]:
        """Create a new order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type_: Order type ("limit" or "market")
            yes_price: Price in cents for YES (1-99)
            no_price: Price in cents for NO (1-99)
            client_order_id: Optional client order ID
            expiration_ts: Optional expiration timestamp

        Returns:
            Order response
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": count,
            "type": type_.lower(),
        }

        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        if expiration_ts:
            body["expiration_ts"] = expiration_ts

        return self._make_request("POST", "/portfolio/orders", json_body=body, auth_required=True)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        return self._make_request("DELETE", f"/portfolio/orders/{order_id}", auth_required=True)

    def get_fills(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Get order fills.

        Args:
            ticker: Filter by market ticker
            limit: Max fills to return
            cursor: Pagination cursor

        Returns:
            Fills response
        """
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor

        return self._make_request("GET", "/portfolio/fills", params=params, auth_required=True)
