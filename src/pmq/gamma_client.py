"""Gamma API client for fetching public market data."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from pmq.config import Settings, get_settings
from pmq.logging import get_logger
from pmq.models import GammaEvent, GammaMarket

logger = get_logger("gamma_client")


class GammaClientError(Exception):
    """Base exception for Gamma client errors."""


class GammaClient:
    """Client for Polymarket Gamma API (public endpoints only).

    The Gamma API provides market metadata and pricing information.
    Base URL: https://gamma-api.polymarket.com

    This client implements caching to reduce API calls.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        """Initialize the Gamma client.

        Args:
            settings: Application settings (uses global if not provided)
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self._settings = settings or get_settings()
        self._base_url = self._settings.gamma_base_url.rstrip("/")
        self._cache_ttl = cache_ttl_seconds
        self._cache_dir = self._settings.cache_dir / "gamma"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP client configuration
        self._timeout = httpx.Timeout(self._settings.http_timeout)
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "GammaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self._cache_dir / f"{safe_key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | list[Any] | None:
        """Read from cache if valid."""
        cache_file = self._cache_path(key)
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            age = (datetime.now(timezone.utc) - cached_at.replace(tzinfo=timezone.utc)).seconds

            if age < self._cache_ttl:
                logger.debug(f"Cache hit for {key} (age={age}s)")
                return data.get("payload")
            logger.debug(f"Cache expired for {key} (age={age}s)")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")

        return None

    def _write_cache(self, key: str, payload: dict[str, Any] | list[Any]) -> None:
        """Write to cache."""
        cache_file = self._cache_path(key)
        data = {
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        logger.debug(f"Cached {key}")

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any] | list[Any]:
        """Make a GET request to the Gamma API.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_cache: Whether to use caching

        Returns:
            Response JSON data

        Raises:
            GammaClientError: If request fails
        """
        cache_key = endpoint
        if params:
            cache_key += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))

        # Check cache first
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        # Make request
        client = self._get_client()
        try:
            logger.info(f"GET {endpoint}", extra={"params": params})
            response = client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            if use_cache:
                self._write_cache(cache_key, data)

            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise GammaClientError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise GammaClientError(f"Request failed: {e}") from e

    def list_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[GammaMarket]:
        """Fetch list of markets from Gamma API.

        Args:
            active: Filter by active status
            closed: Filter by closed status
            limit: Maximum number of results
            offset: Pagination offset
            order: Field to order by
            ascending: Sort direction

        Returns:
            List of market objects
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }

        data = self._request("/markets", params=params)

        # Handle both array response and object with data field
        if isinstance(data, list):
            markets_data = data
        elif isinstance(data, dict):
            markets_data = data.get("data", data.get("markets", []))
        else:
            markets_data = []

        markets = []
        for item in markets_data:
            try:
                market = GammaMarket.model_validate(item)
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")

        logger.info(f"Fetched {len(markets)} markets")
        return markets

    def get_market(self, market_id: str) -> GammaMarket | None:
        """Fetch a specific market by ID.

        Args:
            market_id: Market identifier

        Returns:
            Market object or None if not found
        """
        try:
            data = self._request(f"/markets/{market_id}")
            if isinstance(data, dict):
                return GammaMarket.model_validate(data)
        except GammaClientError as e:
            logger.warning(f"Failed to fetch market {market_id}: {e}")
        return None

    def list_events(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GammaEvent]:
        """Fetch list of events from Gamma API.

        Events contain multiple related markets.

        Args:
            active: Filter by active status
            closed: Filter by closed status
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of event objects
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }

        data = self._request("/events", params=params)

        if isinstance(data, list):
            events_data = data
        elif isinstance(data, dict):
            events_data = data.get("data", data.get("events", []))
        else:
            events_data = []

        events = []
        for item in events_data:
            try:
                event = GammaEvent.model_validate(item)
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")

        logger.info(f"Fetched {len(events)} events")
        return events

    def search_markets(self, query: str, limit: int = 20) -> list[GammaMarket]:
        """Search for markets by text query.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching markets
        """
        params = {"_q": query, "_limit": limit}
        data = self._request("/markets", params=params, use_cache=False)

        if isinstance(data, list):
            markets_data = data
        elif isinstance(data, dict):
            markets_data = data.get("data", data.get("markets", []))
        else:
            markets_data = []

        markets = []
        for item in markets_data:
            try:
                market = GammaMarket.model_validate(item)
                markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")

        return markets

    def clear_cache(self) -> int:
        """Clear all cached data.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self._cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cache files")
        return count
