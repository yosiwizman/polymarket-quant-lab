"""Centralized CLOB client factory for authenticated API access.

Phase 12: Provides a single factory for creating authenticated CLOB clients
using credentials from the credential store.

DESIGN PRINCIPLES:
- Safe by default: Never log or expose credentials
- Centralized: One place to configure host, chain, signature type
- Testable: Supports mock client injection for testing
- No private key storage: Key passed in, not stored

USAGE:
    from pmq.auth.client_factory import create_clob_client, check_auth_sanity

    # Create authenticated client
    client = create_clob_client(private_key="0x...")

    # Check auth sanity (can we connect and auth?)
    result = check_auth_sanity(client)
    if result.success:
        print("Auth OK")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pmq.auth.creds import ClobCredentials, load_creds
from pmq.auth.redact import REDACTED, mask_string
from pmq.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger("auth.client_factory")

# Default CLOB API configuration
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_CHAIN_ID = 137  # Polygon mainnet


class ClobClientProtocol(Protocol):
    """Protocol for CLOB client interface.

    This allows for mock clients in tests without importing py_clob_client.
    """

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        ...

    def get_orders(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Get open orders."""
        ...

    def create_and_post_order(
        self,
        order: Any,  # OrderArgs from py_clob_client
    ) -> dict[str, Any]:
        """Create and post a new order."""
        ...

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order."""
        ...


@dataclass
class AuthSanityResult:
    """Result of authentication sanity check."""

    success: bool
    message: str
    masked_api_key: str = ""
    funder_address: str = ""
    open_orders_count: int | None = None
    error: str | None = None


def get_private_key_from_env() -> str | None:
    """Get private key from environment variable.

    Checks PMQ_PRIVATE_KEY and PRIVATE_KEY env vars.

    Returns:
        Private key string or None if not set
    """
    return os.environ.get("PMQ_PRIVATE_KEY") or os.environ.get("PRIVATE_KEY")


def create_clob_client(
    private_key: str | None = None,
    creds: ClobCredentials | None = None,
    creds_dir: Path | None = None,
    host: str = DEFAULT_CLOB_HOST,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> Any:
    """Create an authenticated CLOB client.

    Order of credential resolution:
    1. Explicit creds parameter
    2. Load from creds_dir (default ~/.pmq/creds.json)

    Private key resolution:
    1. Explicit private_key parameter
    2. PMQ_PRIVATE_KEY env var
    3. PRIVATE_KEY env var

    Args:
        private_key: Ethereum private key (optional, will try env vars)
        creds: Pre-loaded credentials (optional)
        creds_dir: Directory to load credentials from
        host: CLOB API host
        chain_id: Blockchain chain ID

    Returns:
        Authenticated ClobClient

    Raises:
        ImportError: If py_clob_client not installed
        ValueError: If credentials or private key not available
    """
    try:
        from py_clob_client.client import ClobClient  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("py_clob_client not installed. Run: poetry add py-clob-client") from e

    # Resolve private key
    pk = private_key or get_private_key_from_env()
    if not pk:
        raise ValueError(
            "Private key required. Set PMQ_PRIVATE_KEY env var or pass private_key parameter."
        )

    # Ensure 0x prefix
    if not pk.startswith("0x"):
        pk = f"0x{pk}"

    # Resolve credentials
    if creds is None:
        creds = load_creds(creds_dir)
        if creds is None:
            raise ValueError("No credentials found. Run 'pmq ops auth init' to create credentials.")

    logger.info(
        f"Creating CLOB client: host={host}, chain_id={chain_id}, "
        f"api_key={creds.masked_api_key()}, funder={mask_string(creds.funder_address, 6)}"
    )

    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=pk,
        creds=creds.to_clob_creds(),
        signature_type=creds.signature_type,
        funder=creds.funder_address,
    )


def check_auth_sanity(
    client: ClobClientProtocol,
    creds: ClobCredentials | None = None,
) -> AuthSanityResult:
    """Check that authentication is working.

    Performs a harmless API call (get_orders) to verify:
    1. Credentials are valid
    2. Network connectivity is good
    3. API key has not been revoked

    Args:
        client: CLOB client to check
        creds: Optional credentials for masked key display

    Returns:
        AuthSanityResult with success status and details
    """
    masked_key = creds.masked_api_key() if creds else REDACTED
    funder = creds.funder_address if creds else REDACTED

    try:
        # Get open orders - harmless read operation
        orders = client.get_orders()
        open_count = len(orders) if isinstance(orders, list) else 0

        logger.info(f"Auth sanity check passed: {open_count} open orders found")

        return AuthSanityResult(
            success=True,
            message="Authentication successful",
            masked_api_key=masked_key,
            funder_address=funder,
            open_orders_count=open_count,
        )

    except Exception as e:
        error_str = str(e)
        # Redact any secrets that might be in error messages
        from pmq.auth.redact import redact_secrets

        safe_error = redact_secrets(error_str)

        logger.error(f"Auth sanity check failed: {safe_error}")

        return AuthSanityResult(
            success=False,
            message="Authentication failed",
            masked_api_key=masked_key,
            funder_address=funder,
            error=safe_error,
        )


def check_allowances_hint(client: ClobClientProtocol) -> str | None:  # noqa: ARG001
    """Check if allowances might be missing and return hint if so.

    NOTE: This does NOT set allowances - it only detects the issue
    and provides guidance. Client arg reserved for future implementation.

    Args:
        client: CLOB client (reserved for future allowance checking)

    Returns:
        Hint string if allowances seem to be missing, None otherwise
    """
    # For now, we don't have a reliable way to check allowances via API
    # The user will see an error when trying to place an order if allowances
    # are missing. We return a generic hint.
    return (
        "If you see 'insufficient allowance' errors when placing orders, "
        "you may need to set USDC and conditional token allowances. "
        "See Polymarket documentation for approve() transactions."
    )
