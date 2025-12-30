"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SafetyConfig(BaseSettings):
    """Safety guardrails for paper trading."""

    model_config = SettingsConfigDict(env_prefix="PMQ_SAFETY_")

    kill_switch: bool = Field(default=False, description="Emergency stop all operations")
    max_positions: int = Field(default=100, description="Maximum number of open positions")
    max_notional_per_market: float = Field(
        default=1000.0, description="Maximum notional value per market in USD"
    )
    max_trades_per_hour: int = Field(default=50, description="Rate limit on trades per hour")


class ArbitrageConfig(BaseSettings):
    """Configuration for deterministic arbitrage detection."""

    model_config = SettingsConfigDict(env_prefix="PMQ_ARB_")

    threshold: float = Field(
        default=0.99, description="YES + NO price threshold for arbitrage signal"
    )
    min_liquidity: float = Field(
        default=100.0, description="Minimum liquidity required in USD"
    )


class StatArbConfig(BaseSettings):
    """Configuration for statistical arbitrage detection."""

    model_config = SettingsConfigDict(env_prefix="PMQ_STATARB_")

    entry_threshold: float = Field(
        default=0.10, description="Spread threshold to enter position"
    )
    exit_threshold: float = Field(
        default=0.02, description="Spread threshold to exit position"
    )
    pairs_file: Path = Field(
        default=Path("config/pairs.yml"), description="Path to pairs configuration file"
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PMQ_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API endpoints
    gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Gamma API base URL",
    )
    clob_base_url: str = Field(
        default="https://clob.polymarket.com",
        description="CLOB API base URL",
    )

    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    db_path: Path = Field(default=Path("data/pmq.db"), description="SQLite database path")

    # HTTP client settings
    http_timeout: float = Field(default=30.0, description="HTTP request timeout in seconds")
    http_retries: int = Field(default=3, description="Number of HTTP retries")

    # Nested configs
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    statarb: StatArbConfig = Field(default_factory=StatArbConfig)

    @field_validator("data_dir", "cache_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        """Ensure directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("db_path", mode="after")
    @classmethod
    def ensure_db_parent_exists(cls, v: Path) -> Path:
        """Ensure database parent directory exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


def load_pairs_config(path: Path) -> list[dict[str, Any]]:
    """Load pairs configuration from YAML file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("pairs", []) if data else []


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global _settings
    _settings = None
