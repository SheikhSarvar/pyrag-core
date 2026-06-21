"""
Request validation — T56.
Shared Pydantic validators and sanitization helpers used across schemas.
Most validation lives in Pydantic field constraints (see app/schemas/*.py);
this module covers cross-cutting sanitization that field types can't express.
"""
from __future__ import annotations

import re

from app.core.exceptions import ValidationError

# ── Filename sanitization ─────────────────────────────────────────────────────

_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w\-. ]")
_MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """Strip path traversal and unsafe characters from an uploaded filename."""
    # Strip directory components — never trust client-supplied paths
    name = filename.replace("\\", "/").split("/")[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    name = name.strip(". ")
    if not name:
        raise ValidationError("Filename is empty after sanitization")
    if len(name) > _MAX_FILENAME_LENGTH:
        # Preserve extension when truncating
        if "." in name:
            base, ext = name.rsplit(".", 1)
            name = base[: _MAX_FILENAME_LENGTH - len(ext) - 1] + "." + ext
        else:
            name = name[:_MAX_FILENAME_LENGTH]
    return name


# ── Dataset / identifier validation ───────────────────────────────────────────

_SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\- ]{1,255}$")


def validate_dataset_name(name: str) -> str:
    """Ensure dataset names are safe for use as vector-collection name components."""
    name = name.strip()
    if not _SAFE_NAME_PATTERN.match(name):
        raise ValidationError(
            "Dataset name may only contain letters, numbers, spaces, hyphens, and underscores"
        )
    return name


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate a string looks like a UUID. Raises ValidationError if not."""
    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )
    if not _UUID_RE.match(value):
        raise ValidationError(f"Invalid {field_name}: not a valid UUID")
    return value


# ── Query sanitization ─────────────────────────────────────────────────────────

_MAX_QUERY_LENGTH = 2000


def sanitize_query(query: str) -> str:
    """Trim and bound user-supplied search/chat query text."""
    query = query.strip()
    if not query:
        raise ValidationError("Query cannot be empty")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValidationError(f"Query exceeds maximum length of {_MAX_QUERY_LENGTH} characters")
    return query


# ── URL validation (for web ingestion) ────────────────────────────────────────

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"})


def validate_ingestion_url(url: str) -> str:
    """
    Validate a URL before allowing the scraper to fetch it.
    Blocks localhost/metadata-endpoint SSRF targets.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValidationError(f"URL scheme must be one of {_ALLOWED_SCHEMES}")
    hostname = (parsed.hostname or "").lower()
    if hostname in _BLOCKED_HOSTS:
        raise ValidationError("URL host is not allowed")
    if hostname.startswith("169.254.") or hostname.startswith("10.") or hostname.startswith("192.168."):
        raise ValidationError("URL targets a private/internal network range")
    return url
