import pytest

from app.core.config import Settings


def test_sync_database_url_conversion() -> None:
    s = Settings(
        secret_key="a" * 32,
        database_url="postgresql+asyncpg://u:p@localhost/db",
    )
    assert s.sync_database_url == "postgresql+psycopg2://u:p@localhost/db"


def test_max_upload_size_bytes() -> None:
    s = Settings(secret_key="a" * 32, database_url="postgresql+asyncpg://u:p@h/db")
    assert s.max_upload_size_bytes == s.max_upload_size_mb * 1024 * 1024


def test_langfuse_disabled_when_keys_missing() -> None:
    s = Settings(secret_key="a" * 32, database_url="postgresql+asyncpg://u:p@h/db")
    assert s.langfuse_enabled is False


def test_langfuse_enabled_when_keys_set() -> None:
    s = Settings(
        secret_key="a" * 32,
        database_url="postgresql+asyncpg://u:p@h/db",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    assert s.langfuse_enabled is True


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        Settings(
            environment="production",
            secret_key="a-safe-production-secret-key-32c",
            database_url="postgresql+asyncpg://u:p@h/db",
            cors_origins=["*"],  # type: ignore[list-item]
        )
