"""Authentication module for Polymarket CLOB API.

Phase 11: Secure credential management and secret redaction.
"""

from pmq.auth.creds import (
    ClobCredentials,
    create_l2_client,
    creds_exist,
    delete_creds,
    derive_or_create_api_creds,
    get_creds_path,
    load_creds,
    save_creds,
)
from pmq.auth.redact import (
    REDACTED,
    looks_like_secret,
    mask_string,
    redact_env_value,
    redact_secrets,
    safe_dict_for_logging,
)

__all__ = [
    # Credentials
    "ClobCredentials",
    "creds_exist",
    "create_l2_client",
    "delete_creds",
    "derive_or_create_api_creds",
    "get_creds_path",
    "load_creds",
    "save_creds",
    # Redaction
    "REDACTED",
    "looks_like_secret",
    "mask_string",
    "redact_env_value",
    "redact_secrets",
    "safe_dict_for_logging",
]
