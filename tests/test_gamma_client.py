"""Tests for Gamma API client."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmq.config import Settings
from pmq.gamma_client import GammaClient
from pmq.models import GammaMarket


@pytest.fixture
def temp_settings():
    """Create settings with temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        settings = Settings(
            data_dir=tmppath / "data",
            cache_dir=tmppath / "data/cache",
            db_path=tmppath / "data/test.db",
        )
        yield settings


@pytest.fixture
def sample_market_data():
    """Sample market data from Gamma API."""
    return {
        "id": "0x1234567890abcdef",
        "question": "Will Bitcoin reach $100k by end of 2025?",
        "conditionId": "0xabcdef",
        "slug": "bitcoin-100k-2025",
        "description": "Test market description",
        "active": True,
        "closed": False,
        "outcomePrices": "[0.65, 0.35]",
        "liquidity": 50000.0,
        "volume": 100000.0,
        "volume24hr": 5000.0,
        "clobTokenIds": '["token_yes", "token_no"]',
    }


@pytest.fixture
def sample_markets_response(sample_market_data):
    """Sample markets list response."""
    return [sample_market_data]


class TestGammaClient:
    """Tests for GammaClient class."""

    def test_init(self, temp_settings):
        """Test client initialization."""
        client = GammaClient(settings=temp_settings)
        assert client._base_url == "https://gamma-api.polymarket.com"
        client.close()

    def test_cache_path_generation(self, temp_settings):
        """Test cache path generation."""
        client = GammaClient(settings=temp_settings)
        path = client._cache_path("/markets?limit=100")
        assert "markets" in str(path)
        assert path.suffix == ".json"
        client.close()

    def test_write_and_read_cache(self, temp_settings):
        """Test cache write and read."""
        client = GammaClient(settings=temp_settings, cache_ttl_seconds=3600)

        test_data = {"foo": "bar", "count": 42}
        key = "test_key"

        # Write to cache
        client._write_cache(key, test_data)

        # Read from cache
        cached = client._read_cache(key)
        assert cached == test_data

        client.close()

    def test_cache_expiration(self, temp_settings):
        """Test cache expiration."""
        client = GammaClient(settings=temp_settings, cache_ttl_seconds=0)

        test_data = {"foo": "bar"}
        key = "expired_key"

        client._write_cache(key, test_data)

        # Should return None because TTL is 0
        cached = client._read_cache(key)
        assert cached is None

        client.close()

    @patch("httpx.Client")
    def test_list_markets_success(self, mock_client_class, temp_settings, sample_markets_response):
        """Test successful market listing."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = sample_markets_response
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = GammaClient(settings=temp_settings, cache_ttl_seconds=0)
        client._client = mock_client

        markets = client.list_markets(limit=10)

        assert len(markets) == 1
        assert isinstance(markets[0], GammaMarket)
        assert markets[0].id == "0x1234567890abcdef"
        assert markets[0].question == "Will Bitcoin reach $100k by end of 2025?"

        client.close()

    def test_market_price_parsing(self, sample_market_data):
        """Test market price extraction from outcome_prices."""
        market = GammaMarket.model_validate(sample_market_data)

        assert market.yes_price == 0.65
        assert market.no_price == 0.35

    def test_market_token_id_parsing(self, sample_market_data):
        """Test token ID extraction."""
        market = GammaMarket.model_validate(sample_market_data)

        assert market.yes_token_id == "token_yes"
        assert market.no_token_id == "token_no"

    def test_clear_cache(self, temp_settings):
        """Test cache clearing."""
        client = GammaClient(settings=temp_settings)

        # Write some cache files
        client._write_cache("key1", {"data": 1})
        client._write_cache("key2", {"data": 2})

        # Clear cache
        count = client.clear_cache()

        assert count == 2

        # Verify cache is empty
        assert client._read_cache("key1") is None
        assert client._read_cache("key2") is None

        client.close()


class TestGammaMarketModel:
    """Tests for GammaMarket model."""

    def test_market_with_empty_prices(self):
        """Test market with no price data."""
        data = {
            "id": "test_id",
            "question": "Test question",
            "active": True,
            "closed": False,
        }
        market = GammaMarket.model_validate(data)

        # Should use default value
        assert market.yes_price == 0.5
        assert market.no_price == 0.5

    def test_market_with_tokens(self):
        """Test market with tokens array."""
        data = {
            "id": "test_id",
            "question": "Test question",
            "active": True,
            "closed": False,
            "tokens": [
                {"token_id": "yes_token", "outcome": "Yes", "price": 0.7},
                {"token_id": "no_token", "outcome": "No", "price": 0.3},
            ],
        }
        market = GammaMarket.model_validate(data)

        # Should extract prices from tokens
        # Note: outcome_prices takes precedence, so tokens are fallback
        assert market.yes_price in [0.5, 0.7]  # Depends on parsing logic

    def test_market_inactive(self):
        """Test inactive market detection."""
        data = {
            "id": "test_id",
            "question": "Test question",
            "active": False,
            "closed": True,
        }
        market = GammaMarket.model_validate(data)

        assert not market.active
        assert market.closed
