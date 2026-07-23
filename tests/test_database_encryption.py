import pytest

from idea import database


def test_encryption_key_has_no_known_fallback(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ENCRYPTION_KEY is required"):
        database._get_encryption_key()


def test_api_key_encryption_round_trips_with_explicit_key(monkeypatch):
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        "unit-test-only-explicit-master-key-with-sufficient-entropy",
    )

    encrypted = database.encrypt_api_key("gemini-secret")

    assert encrypted != "gemini-secret"
    assert database.decrypt_api_key(encrypted) == "gemini-secret"


def test_encryption_key_rejects_short_secret(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "too-short")

    with pytest.raises(database.EncryptionConfigurationError, match="at least 32 bytes"):
        database._get_encryption_key()


def test_get_user_api_key_does_not_hide_configuration_error(monkeypatch):
    async def fake_settings(_user_id):
        return {"api_key_encrypted": "ciphertext"}

    monkeypatch.setattr(database, "get_user_settings", fake_settings)
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)

    import asyncio

    with pytest.raises(database.EncryptionConfigurationError):
        asyncio.run(database.get_user_api_key("user-1"))


def test_get_user_api_key_reports_invalid_ciphertext(monkeypatch):
    async def fake_settings(_user_id):
        return {"api_key_encrypted": "not-a-fernet-token"}

    monkeypatch.setattr(database, "get_user_settings", fake_settings)
    monkeypatch.setenv(
        "ENCRYPTION_KEY",
        "unit-test-only-explicit-master-key-with-sufficient-entropy",
    )

    import asyncio

    with pytest.raises(database.APIKeyDecryptionError, match="cannot be decrypted"):
        asyncio.run(database.get_user_api_key("user-1"))
