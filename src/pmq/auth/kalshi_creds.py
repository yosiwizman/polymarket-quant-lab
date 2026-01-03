"""Secure credential storage for Kalshi API.

Phase 13: Provides secure local storage for Kalshi API credentials
(email, api_key, api_secret) with proper file permissions.

DESIGN PRINCIPLES:
- Safe by default: Credentials never logged or printed in full
- Secure storage: File permissions 0600 (owner read/write only)
- No git: ~/.pmq/kalshi_creds.json is outside repo
- Compatible: Same patterns as Polymarket creds module

USAGE:
    # Initialize credentials (interactive or from env)
    creds = KalshiCredentials(
        email="user@example.com",
        api_key="...",
        api_secret="...",
    )
    save_kalshi_creds(creds)

    # Load and use credentials
    creds = load_kalshi_creds()
    if creds:
        client = KalshiExchangeClient(creds=creds)
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pmq.logging import get_logger

logger = get_logger("auth.kalshi_creds")

# Default credentials directory and file
DEFAULT_CREDS_DIR = Path.home() / ".pmq"
DEFAULT_KALSHI_CREDS_FILE = DEFAULT_CREDS_DIR / "kalshi_creds.json"

# Required file permissions (owner read/write only)
SECURE_PERMS = stat.S_IRUSR | stat.S_IWUSR  # 0o600

# Kalshi API base URLs
KALSHI_API_BASE_PROD = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_API_BASE_DEMO = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class KalshiCredentials:
    """Kalshi API credentials.

    Attributes:
        email: Kalshi account email
        api_key: API key (also called "Key ID" in Kalshi dashboard)
        api_secret: API secret (private key content for RSA auth)
        api_base: API base URL (production or demo)
        member_id: Optional member/user ID (obtained after auth)
    """

    email: str
    api_key: str
    api_secret: str
    api_base: str = KALSHI_API_BASE_PROD
    member_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KalshiCredentials:
        """Create credentials from dictionary."""
        return cls(
            email=data["email"],
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            api_base=data.get("api_base", KALSHI_API_BASE_PROD),
            member_id=data.get("member_id", ""),
        )

    def masked_api_key(self) -> str:
        """Return masked API key showing first and last 4 characters.

        Example: "abc1...xyz9"
        """
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def masked_email(self) -> str:
        """Return masked email showing first 2 characters before @.

        Example: "ab...@example.com"
        """
        if "@" not in self.email:
            return "****"
        local, domain = self.email.split("@", 1)
        if len(local) <= 2:
            return f"**@{domain}"
        return f"{local[:2]}...@{domain}"

    @property
    def is_demo(self) -> bool:
        """Check if using demo API."""
        return "demo" in self.api_base.lower()


def get_kalshi_creds_path(creds_dir: Path | None = None) -> Path:
    """Get the path to the Kalshi credentials file.

    Args:
        creds_dir: Optional custom directory

    Returns:
        Path to kalshi_creds.json
    """
    if creds_dir is not None:
        return creds_dir / "kalshi_creds.json"
    return DEFAULT_KALSHI_CREDS_FILE


def save_kalshi_creds(
    creds: KalshiCredentials,
    creds_dir: Path | None = None,
) -> Path:
    """Save Kalshi credentials to disk with secure permissions.

    Args:
        creds: Credentials to save
        creds_dir: Optional custom directory

    Returns:
        Path to saved file

    Raises:
        OSError: If file cannot be created or permissions set
    """
    creds_path = get_kalshi_creds_path(creds_dir)
    creds_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(creds_path, "w", encoding="utf-8") as f:
        json.dump(creds.to_dict(), f, indent=2)

    # Set secure permissions (owner read/write only)
    try:
        os.chmod(creds_path, SECURE_PERMS)
        logger.info(f"Saved Kalshi credentials to {creds_path} with secure permissions")
    except OSError as e:
        logger.warning(f"Could not set secure permissions on {creds_path}: {e}")
        # On Windows, chmod may not work fully but we continue

    return creds_path


def load_kalshi_creds(creds_dir: Path | None = None) -> KalshiCredentials | None:
    """Load Kalshi credentials from disk.

    Args:
        creds_dir: Optional custom directory

    Returns:
        KalshiCredentials if found and valid, None otherwise
    """
    creds_path = get_kalshi_creds_path(creds_dir)

    if not creds_path.exists():
        logger.debug(f"No Kalshi credentials file at {creds_path}")
        return None

    try:
        with open(creds_path, encoding="utf-8") as f:
            data = json.load(f)
        return KalshiCredentials.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load Kalshi credentials from {creds_path}: {e}")
        return None


def delete_kalshi_creds(creds_dir: Path | None = None) -> bool:
    """Delete Kalshi credentials file.

    Args:
        creds_dir: Optional custom directory

    Returns:
        True if deleted, False if didn't exist
    """
    creds_path = get_kalshi_creds_path(creds_dir)

    if creds_path.exists():
        try:
            creds_path.unlink()
            logger.info(f"Deleted Kalshi credentials at {creds_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete Kalshi credentials: {e}")
            return False
    return False


def kalshi_creds_exist(creds_dir: Path | None = None) -> bool:
    """Check if Kalshi credentials file exists.

    Args:
        creds_dir: Optional custom directory

    Returns:
        True if file exists
    """
    return get_kalshi_creds_path(creds_dir).exists()


def get_kalshi_creds_from_env() -> KalshiCredentials | None:
    """Load Kalshi credentials from environment variables.

    Environment variables:
        KALSHI_EMAIL: Account email
        KALSHI_API_KEY: API key
        KALSHI_API_SECRET: API secret (or path to private key file)
        KALSHI_API_BASE: Optional API base URL (default: production)

    Returns:
        KalshiCredentials if all required env vars are set, None otherwise
    """
    email = os.environ.get("KALSHI_EMAIL")
    api_key = os.environ.get("KALSHI_API_KEY")
    api_secret = os.environ.get("KALSHI_API_SECRET")

    if not email or not api_key or not api_secret:
        return None

    # Check if api_secret is a file path
    if os.path.isfile(api_secret):
        try:
            with open(api_secret, encoding="utf-8") as f:
                api_secret = f.read().strip()
        except OSError as e:
            logger.warning(f"Failed to read API secret from file {api_secret}: {e}")
            return None

    api_base = os.environ.get("KALSHI_API_BASE", KALSHI_API_BASE_PROD)

    return KalshiCredentials(
        email=email,
        api_key=api_key,
        api_secret=api_secret,
        api_base=api_base,
    )
