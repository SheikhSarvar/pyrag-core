"""
API Key authentication — T54.
Validates X-API-Key header against configured keys.
Supports multiple keys (comma-separated in env) for key rotation.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import Request
from fastapi.security import APIKeyHeader

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError

settings = get_settings()

_api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

# Paths that don't require authentication
_PUBLIC_PATHS = frozenset({
    "/", "/health", "/docs", "/redoc", "/openapi.json",
})


def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-attack-resistant string comparison."""
    return hmac.compare_digest(a.encode(), b.encode())


def _get_valid_keys() -> set[str]:
    """
    Read valid API keys from settings.
    In production this should come from a secrets manager or DB table,
    not raw env vars — this is the V1 implementation.
    """
    import os
    raw = os.getenv("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(request: Request) -> str | None:
    """
    FastAPI dependency — validates the API key header.
    Raises AuthenticationError if invalid or missing (when keys are configured).
    Returns the validated key, or None if auth is disabled (no keys configured).
    """
    valid_keys = _get_valid_keys()

    # If no keys are configured, auth is effectively disabled (dev mode)
    if not valid_keys:
        return None

    if request.url.path in _PUBLIC_PATHS:
        return None

    provided = request.headers.get(settings.api_key_header)
    if not provided:
        raise AuthenticationError(f"Missing {settings.api_key_header} header")

    for valid_key in valid_keys:
        if _constant_time_compare(provided, valid_key):
            return provided

    raise AuthenticationError("Invalid API key")


def hash_api_key(key: str) -> str:
    """Hash an API key for storage (never store raw keys)."""
    return hashlib.sha256(key.encode()).hexdigest()
