"""Unit tests for auth module (Phase 11).

Tests credential storage, masking, and redaction utilities.
"""

import json
from pathlib import Path

from pmq.auth import (
    ClobCredentials,
    creds_exist,
    delete_creds,
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

# =============================================================================
# ClobCredentials Tests
# =============================================================================


class TestClobCredentials:
    """Tests for ClobCredentials dataclass."""

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        creds = ClobCredentials(
            api_key="550e8400-e29b-41d4-a716-446655440000",
            api_secret="base64EncodedSecretString==",
            api_passphrase="randomPassphraseString",
            funder_address="0x1234567890123456789012345678901234567890",
            signature_type=1,
        )
        data = creds.to_dict()
        restored = ClobCredentials.from_dict(data)

        assert restored.api_key == creds.api_key
        assert restored.api_secret == creds.api_secret
        assert restored.api_passphrase == creds.api_passphrase
        assert restored.funder_address == creds.funder_address
        assert restored.signature_type == creds.signature_type

    def test_to_clob_creds(self) -> None:
        """Test conversion to py_clob_client format."""
        creds = ClobCredentials(
            api_key="550e8400-e29b-41d4-a716-446655440000",
            api_secret="base64Secret==",
            api_passphrase="mypassphrase",
            funder_address="0x123",
        )
        clob_creds = creds.to_clob_creds()

        assert clob_creds["apiKey"] == creds.api_key
        assert clob_creds["secret"] == creds.api_secret
        assert clob_creds["passphrase"] == creds.api_passphrase

    def test_masked_api_key(self) -> None:
        """Test API key masking."""
        creds = ClobCredentials(
            api_key="550e8400-e29b-41d4-a716-446655440000",
            api_secret="secret",
            api_passphrase="pass",
            funder_address="0x123",
        )
        masked = creds.masked_api_key()

        assert masked == "550e...0000"
        assert creds.api_key not in masked

    def test_masked_api_key_short(self) -> None:
        """Test masking of very short API key."""
        creds = ClobCredentials(
            api_key="short",
            api_secret="secret",
            api_passphrase="pass",
            funder_address="0x123",
        )
        masked = creds.masked_api_key()

        assert masked == "****"


# =============================================================================
# Credential Storage Tests
# =============================================================================


class TestCredentialStorage:
    """Tests for credential file operations."""

    def test_save_and_load_creds(self, tmp_path: Path) -> None:
        """Test saving and loading credentials."""
        creds = ClobCredentials(
            api_key="550e8400-e29b-41d4-a716-446655440000",
            api_secret="base64EncodedSecretString==",
            api_passphrase="randomPassphraseString",
            funder_address="0x1234567890123456789012345678901234567890",
            signature_type=1,
        )

        # Save
        saved_path = save_creds(creds, creds_dir=tmp_path)
        assert saved_path.exists()

        # Load
        loaded = load_creds(creds_dir=tmp_path)
        assert loaded is not None
        assert loaded.api_key == creds.api_key
        assert loaded.api_secret == creds.api_secret
        assert loaded.api_passphrase == creds.api_passphrase
        assert loaded.funder_address == creds.funder_address
        assert loaded.signature_type == creds.signature_type

    def test_creds_exist(self, tmp_path: Path) -> None:
        """Test credential existence check."""
        assert not creds_exist(creds_dir=tmp_path)

        creds = ClobCredentials(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            funder_address="0x123",
        )
        save_creds(creds, creds_dir=tmp_path)

        assert creds_exist(creds_dir=tmp_path)

    def test_delete_creds(self, tmp_path: Path) -> None:
        """Test credential deletion."""
        creds = ClobCredentials(
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
            funder_address="0x123",
        )
        save_creds(creds, creds_dir=tmp_path)
        assert creds_exist(creds_dir=tmp_path)

        # Delete
        result = delete_creds(creds_dir=tmp_path)
        assert result is True
        assert not creds_exist(creds_dir=tmp_path)

        # Delete again (should return False)
        result = delete_creds(creds_dir=tmp_path)
        assert result is False

    def test_load_creds_missing(self, tmp_path: Path) -> None:
        """Test loading when no credentials exist."""
        loaded = load_creds(creds_dir=tmp_path)
        assert loaded is None

    def test_load_creds_corrupted(self, tmp_path: Path) -> None:
        """Test loading corrupted credentials file."""
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("not valid json")

        loaded = load_creds(creds_dir=tmp_path)
        assert loaded is None

    def test_get_creds_path_default(self) -> None:
        """Test default credentials path."""
        path = get_creds_path()
        assert path == Path.home() / ".pmq" / "creds.json"

    def test_get_creds_path_custom(self, tmp_path: Path) -> None:
        """Test custom credentials path."""
        path = get_creds_path(creds_dir=tmp_path)
        assert path == tmp_path / "creds.json"


# =============================================================================
# Redaction Tests
# =============================================================================


class TestRedaction:
    """Tests for secret redaction utilities."""

    def test_redact_private_key_with_prefix(self) -> None:
        """Test redacting private key with 0x prefix."""
        # 64 hex chars after 0x
        private_key = "0x" + "a" * 64
        text = f"Private key is {private_key}"
        result = redact_secrets(text)

        assert private_key not in result
        assert REDACTED in result

    def test_redact_private_key_without_prefix(self) -> None:
        """Test redacting private key without 0x prefix."""
        # 64 hex chars
        private_key = "a" * 64
        text = f"Key: {private_key}"
        result = redact_secrets(text)

        assert private_key not in result
        assert REDACTED in result

    def test_redact_uuid(self) -> None:
        """Test redacting UUID-format API key."""
        api_key = "550e8400-e29b-41d4-a716-446655440000"
        text = f"API key: {api_key}"
        result = redact_secrets(text)

        assert api_key not in result
        assert REDACTED in result

    def test_redact_ethereum_address(self) -> None:
        """Test redacting Ethereum address."""
        address = "0x" + "1234567890" * 4
        text = f"Address: {address}"
        result = redact_secrets(text)

        assert address not in result
        assert REDACTED in result

    def test_redact_preserves_safe_text(self) -> None:
        """Test that safe text is preserved."""
        safe_text = "Hello world, this is safe text."
        result = redact_secrets(safe_text)
        assert result == safe_text

    def test_looks_like_secret_uuid(self) -> None:
        """Test UUID detection."""
        assert looks_like_secret("550e8400-e29b-41d4-a716-446655440000") is True

    def test_looks_like_secret_private_key(self) -> None:
        """Test private key detection."""
        assert looks_like_secret("0x" + "a" * 64) is True
        assert looks_like_secret("a" * 64) is True

    def test_looks_like_secret_normal_text(self) -> None:
        """Test normal text is not flagged."""
        assert looks_like_secret("hello world") is False
        assert looks_like_secret("12345") is False

    def test_mask_string(self) -> None:
        """Test string masking."""
        assert mask_string("1234567890abcdef", visible_chars=4) == "1234...cdef"
        assert mask_string("short", visible_chars=4) == "*****"

    def test_redact_env_value_known_secret(self) -> None:
        """Test redacting known secret env vars."""
        assert redact_env_value("PRIVATE_KEY", "some_value") == REDACTED
        assert redact_env_value("PMQ_PRIVATE_KEY", "some_value") == REDACTED
        assert redact_env_value("POLY_API_KEY", "some_value") == REDACTED

    def test_redact_env_value_safe(self) -> None:
        """Test preserving safe env vars."""
        assert redact_env_value("HOME", "/home/user") == "/home/user"
        assert redact_env_value("PATH", "/usr/bin") == "/usr/bin"

    def test_redact_env_value_secret_looking_value(self) -> None:
        """Test redacting value that looks like a secret."""
        # Even if env name is not in the known list, if value looks like a secret
        uuid_value = "550e8400-e29b-41d4-a716-446655440000"
        assert redact_env_value("UNKNOWN_VAR", uuid_value) == REDACTED

    def test_safe_dict_for_logging(self) -> None:
        """Test dict redaction for logging."""
        data = {
            "api_key": "550e8400-e29b-41d4-a716-446655440000",
            "secret": "my_secret_value",
            "passphrase": "my_passphrase",
            "normal_field": "normal_value",
            "nested": {
                "password": "nested_password",
                "ok": "nested_ok",
            },
        }
        safe = safe_dict_for_logging(data)

        assert safe["api_key"] == REDACTED
        assert safe["secret"] == REDACTED
        assert safe["passphrase"] == REDACTED
        assert safe["normal_field"] == "normal_value"
        assert safe["nested"]["password"] == REDACTED
        assert safe["nested"]["ok"] == "nested_ok"

    def test_safe_dict_for_logging_custom_keys(self) -> None:
        """Test dict redaction with custom keys."""
        data = {
            "custom_secret": "sensitive_data",
            "normal": "ok",
        }
        safe = safe_dict_for_logging(data, redact_keys={"custom_secret"})

        assert safe["custom_secret"] == REDACTED
        assert safe["normal"] == "ok"


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuthIntegration:
    """Integration tests for auth workflow."""

    def test_full_creds_workflow(self, tmp_path: Path) -> None:
        """Test complete credentials workflow."""
        # Initially no creds
        assert not creds_exist(creds_dir=tmp_path)
        assert load_creds(creds_dir=tmp_path) is None

        # Create and save creds
        creds = ClobCredentials(
            api_key="550e8400-e29b-41d4-a716-446655440000",
            api_secret="base64EncodedSecretString==",
            api_passphrase="randomPassphraseString",
            funder_address="0x1234567890123456789012345678901234567890",
            signature_type=1,
        )
        save_creds(creds, creds_dir=tmp_path)

        # Verify exists and can be loaded
        assert creds_exist(creds_dir=tmp_path)
        loaded = load_creds(creds_dir=tmp_path)
        assert loaded is not None
        assert loaded.api_key == creds.api_key

        # Delete creds
        delete_creds(creds_dir=tmp_path)
        assert not creds_exist(creds_dir=tmp_path)

    def test_creds_file_content(self, tmp_path: Path) -> None:
        """Test that credentials file is valid JSON."""
        creds = ClobCredentials(
            api_key="test-key",
            api_secret="test-secret",
            api_passphrase="test-pass",
            funder_address="0xtest",
        )
        saved_path = save_creds(creds, creds_dir=tmp_path)

        # Read and verify JSON
        with open(saved_path) as f:
            data = json.load(f)

        assert data["api_key"] == "test-key"
        assert data["api_secret"] == "test-secret"
        assert data["api_passphrase"] == "test-pass"
        assert data["funder_address"] == "0xtest"
