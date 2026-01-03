"""Redaction utilities to prevent secrets from appearing in logs.

Phase 11: Provides functions to redact sensitive information from
strings before logging or printing.

USAGE:
    from pmq.auth.redact import redact_secrets, REDACTED

    # Redact known secret patterns
    safe_msg = redact_secrets(f"Key is {api_key}")

    # Check if string looks like a secret
    if looks_like_secret(value):
        log(REDACTED)
"""

from __future__ import annotations

import re
from re import Pattern

# Placeholder for redacted content
REDACTED = "***REDACTED***"

# Patterns for common secret formats
SECRET_PATTERNS: list[tuple[str, Pattern[str]]] = [
    # Private keys (64 hex chars with optional 0x prefix)
    ("private_key", re.compile(r"0x[a-fA-F0-9]{64}\b")),
    ("private_key", re.compile(r"\b[a-fA-F0-9]{64}\b")),
    # API keys (UUID format)
    (
        "api_key",
        re.compile(
            r"\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b"
        ),
    ),
    # Base64 secrets (common lengths)
    ("secret", re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")),
    # Ethereum addresses (40 hex chars with 0x prefix)
    ("address", re.compile(r"0x[a-fA-F0-9]{40}\b")),
]

# Environment variable names that contain secrets
SECRET_ENV_VARS = frozenset(
    {
        "PRIVATE_KEY",
        "PMQ_PRIVATE_KEY",
        "POLY_API_KEY",
        "POLY_API_SECRET",
        "POLY_API_PASSPHRASE",
        "CLOB_API_KEY",
        "CLOB_API_SECRET",
        "CLOB_API_PASSPHRASE",
    }
)


def redact_secrets(text: str, replacement: str = REDACTED) -> str:
    """Redact known secret patterns from text.

    Args:
        text: Input text that may contain secrets
        replacement: String to replace secrets with

    Returns:
        Text with secrets replaced

    Example:
        >>> redact_secrets("Key: 0x1234...64chars")
        "Key: ***REDACTED***"
    """
    result = text
    for _name, pattern in SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def looks_like_secret(value: str) -> bool:
    """Check if a string looks like a secret.

    Args:
        value: String to check

    Returns:
        True if the string matches common secret patterns
    """
    return any(pattern.fullmatch(value) for _name, pattern in SECRET_PATTERNS)


def mask_string(value: str, visible_chars: int = 4) -> str:
    """Mask a string showing only first and last N characters.

    Args:
        value: String to mask
        visible_chars: Number of chars to show at start and end

    Returns:
        Masked string like "abcd...wxyz"

    Example:
        >>> mask_string("1234567890abcdef", visible_chars=4)
        "1234...cdef"
    """
    if len(value) <= visible_chars * 2:
        return "*" * len(value)
    return f"{value[:visible_chars]}...{value[-visible_chars:]}"


def redact_env_value(env_name: str, env_value: str) -> str:
    """Redact an environment variable value if it's a known secret.

    Args:
        env_name: Name of the environment variable
        env_value: Value of the environment variable

    Returns:
        The value or REDACTED if it's a secret
    """
    # Check if env var name is in known secrets list
    if env_name.upper() in SECRET_ENV_VARS:
        return REDACTED

    # Also check if the value looks like a secret
    if looks_like_secret(env_value):
        return REDACTED

    return env_value


def safe_dict_for_logging(
    data: dict[str, object], redact_keys: set[str] | None = None
) -> dict[str, object]:
    """Create a copy of a dict safe for logging by redacting secrets.

    Args:
        data: Dictionary that may contain secrets
        redact_keys: Additional keys to redact (case-insensitive)

    Returns:
        Copy of dict with secret values replaced
    """
    if redact_keys is None:
        redact_keys = set()

    # Default keys to redact
    default_redact = {
        "api_key",
        "apikey",
        "api-key",
        "api_secret",
        "apisecret",
        "api-secret",
        "secret",
        "api_passphrase",
        "apipassphrase",
        "api-passphrase",
        "passphrase",
        "private_key",
        "privatekey",
        "private-key",
        "key",
        "password",
        "passwd",
        "token",
    }
    all_redact_keys = default_redact | {k.lower() for k in redact_keys}

    result: dict[str, object] = {}
    for k, v in data.items():
        if k.lower() in all_redact_keys or (isinstance(v, str) and looks_like_secret(v)):
            result[k] = REDACTED
        elif isinstance(v, dict):
            result[k] = safe_dict_for_logging(dict(v), redact_keys)
        else:
            result[k] = v
    return result
