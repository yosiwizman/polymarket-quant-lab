"""Secure credential storage for Polymarket CLOB API.

Phase 11: Provides secure local storage for CLOB API credentials
(apiKey, secret, passphrase) with proper file permissions.

DESIGN PRINCIPLES:
- Safe by default: Credentials never logged or printed in full
- Secure storage: File permissions 0600 (owner read/write only)
- No git: ~/.pmq/creds.json is outside repo
- Clear warnings: Users informed about key invalidation risks

USAGE:
    # Initialize credentials (requires PRIVATE_KEY env var)
    creds = derive_or_create_api_creds(private_key, funder_address)
    save_creds(creds)

    # Load and use credentials
    creds = load_creds()
    if creds:
        client = ClobClient(..., creds=creds.to_clob_creds(), ...)
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pmq.logging import get_logger

logger = get_logger("auth.creds")

# Default credentials directory and file
DEFAULT_CREDS_DIR = Path.home() / ".pmq"
DEFAULT_CREDS_FILE = DEFAULT_CREDS_DIR / "creds.json"

# Required file permissions (owner read/write only)
SECURE_PERMS = stat.S_IRUSR | stat.S_IWUSR  # 0o600


@dataclass
class ClobCredentials:
    """CLOB API credentials.

    Attributes:
        api_key: The API key (UUID format)
        api_secret: Base64-encoded secret
        api_passphrase: Passphrase string
        funder_address: The funder/proxy wallet address
        signature_type: Signature type (0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE)
    """

    api_key: str
    api_secret: str
    api_passphrase: str
    funder_address: str
    signature_type: int = 1  # Default to POLY_PROXY for most users

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClobCredentials:
        """Create credentials from dictionary."""
        return cls(
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            api_passphrase=data["api_passphrase"],
            funder_address=data["funder_address"],
            signature_type=data.get("signature_type", 1),
        )

    def to_clob_creds(self) -> dict[str, str]:
        """Convert to py_clob_client ApiCreds format.

        Returns dict with keys: apiKey, secret, passphrase
        """
        return {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "passphrase": self.api_passphrase,
        }

    def masked_api_key(self) -> str:
        """Return masked API key showing first and last 4 characters.

        Example: "550e...0000"
        """
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


def get_creds_path(creds_dir: Path | None = None) -> Path:
    """Get the path to the credentials file.

    Args:
        creds_dir: Optional custom directory

    Returns:
        Path to creds.json
    """
    if creds_dir is not None:
        return creds_dir / "creds.json"
    return DEFAULT_CREDS_FILE


def save_creds(
    creds: ClobCredentials,
    creds_dir: Path | None = None,
) -> Path:
    """Save credentials to disk with secure permissions.

    Args:
        creds: Credentials to save
        creds_dir: Optional custom directory

    Returns:
        Path to saved file

    Raises:
        OSError: If file cannot be created or permissions set
    """
    creds_path = get_creds_path(creds_dir)
    creds_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(creds_path, "w", encoding="utf-8") as f:
        json.dump(creds.to_dict(), f, indent=2)

    # Set secure permissions (owner read/write only)
    try:
        os.chmod(creds_path, SECURE_PERMS)
        logger.info(f"Saved credentials to {creds_path} with secure permissions")
    except OSError as e:
        logger.warning(f"Could not set secure permissions on {creds_path}: {e}")
        # On Windows, chmod may not work fully but we continue

    return creds_path


def load_creds(creds_dir: Path | None = None) -> ClobCredentials | None:
    """Load credentials from disk.

    Args:
        creds_dir: Optional custom directory

    Returns:
        ClobCredentials if found and valid, None otherwise
    """
    creds_path = get_creds_path(creds_dir)

    if not creds_path.exists():
        logger.debug(f"No credentials file at {creds_path}")
        return None

    try:
        with open(creds_path, encoding="utf-8") as f:
            data = json.load(f)
        return ClobCredentials.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load credentials from {creds_path}: {e}")
        return None


def delete_creds(creds_dir: Path | None = None) -> bool:
    """Delete credentials file.

    Args:
        creds_dir: Optional custom directory

    Returns:
        True if deleted, False if didn't exist
    """
    creds_path = get_creds_path(creds_dir)

    if creds_path.exists():
        try:
            creds_path.unlink()
            logger.info(f"Deleted credentials at {creds_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False
    return False


def creds_exist(creds_dir: Path | None = None) -> bool:
    """Check if credentials file exists.

    Args:
        creds_dir: Optional custom directory

    Returns:
        True if file exists
    """
    return get_creds_path(creds_dir).exists()


def derive_or_create_api_creds(
    private_key: str,
    funder_address: str,
    host: str = "https://clob.polymarket.com",
    chain_id: int = 137,
) -> ClobCredentials:
    """Derive or create API credentials using py_clob_client.

    This performs L1 authentication to derive/create L2 API credentials.

    WARNING: Creating new credentials may invalidate existing ones!

    Args:
        private_key: Ethereum private key (hex, with or without 0x prefix)
        funder_address: Funder/proxy wallet address
        host: CLOB API host
        chain_id: Blockchain chain ID (137 = Polygon mainnet)

    Returns:
        ClobCredentials with apiKey, secret, passphrase

    Raises:
        ImportError: If py_clob_client not installed
        Exception: If API call fails
    """
    try:
        from py_clob_client.client import ClobClient
    except ImportError as e:
        raise ImportError("py_clob_client not installed. Run: poetry add py-clob-client") from e

    # Ensure private key has 0x prefix
    if not private_key.startswith("0x"):
        private_key = f"0x{private_key}"

    # Create L1 client (no creds needed for L1)
    client = ClobClient(host=host, chain_id=chain_id, key=private_key)

    # Derive or create API credentials
    # This calls POST /auth/api-key or GET /auth/derive-api-key
    api_creds = client.create_or_derive_api_creds()

    return ClobCredentials(
        api_key=api_creds["apiKey"],
        api_secret=api_creds["secret"],
        api_passphrase=api_creds["passphrase"],
        funder_address=funder_address,
        signature_type=1,  # POLY_PROXY - most common for external integrations
    )


def create_l2_client(
    creds: ClobCredentials,
    private_key: str,
    host: str = "https://clob.polymarket.com",
    chain_id: int = 137,
) -> Any:
    """Create an L2-authenticated CLOB client.

    Args:
        creds: API credentials
        private_key: Ethereum private key
        host: CLOB API host
        chain_id: Blockchain chain ID

    Returns:
        Authenticated ClobClient ready for L2 operations

    Raises:
        ImportError: If py_clob_client not installed
    """
    try:
        from py_clob_client.client import ClobClient
    except ImportError as e:
        raise ImportError("py_clob_client not installed. Run: poetry add py-clob-client") from e

    # Ensure private key has 0x prefix
    if not private_key.startswith("0x"):
        private_key = f"0x{private_key}"

    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=private_key,
        creds=creds.to_clob_creds(),
        signature_type=creds.signature_type,
        funder=creds.funder_address,
    )
