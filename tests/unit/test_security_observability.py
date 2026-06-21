"""
Tests for Phase 8/9: auth, validation, tracking, middleware.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from app.core.exceptions import AuthenticationError, ValidationError
from app.core.validation import (
    sanitize_filename,
    sanitize_query,
    validate_dataset_name,
    validate_ingestion_url,
    validate_uuid,
)


# ── Validation ────────────────────────────────────────────────────────────────

def test_sanitize_filename_strips_path_traversal() -> None:
    result = sanitize_filename("../../etc/passwd")
    assert "/" not in result
    assert ".." not in result or result == "passwd"


def test_sanitize_filename_removes_unsafe_chars() -> None:
    result = sanitize_filename("my<file>:name?.pdf")
    assert "<" not in result
    assert ">" not in result
    assert ":" not in result
    assert "?" not in result


def test_sanitize_filename_preserves_extension_on_truncate() -> None:
    long_name = "a" * 300 + ".pdf"
    result = sanitize_filename(long_name)
    assert result.endswith(".pdf")
    assert len(result) <= 255


def test_sanitize_filename_empty_raises() -> None:
    with pytest.raises(ValidationError):
        sanitize_filename("...")


def test_validate_dataset_name_accepts_valid() -> None:
    assert validate_dataset_name("my-dataset_v2") == "my-dataset_v2"


def test_validate_dataset_name_rejects_special_chars() -> None:
    with pytest.raises(ValidationError):
        validate_dataset_name("dataset; DROP TABLE users;")


def test_validate_uuid_accepts_valid() -> None:
    valid = "550e8400-e29b-41d4-a716-446655440000"
    assert validate_uuid(valid) == valid


def test_validate_uuid_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        validate_uuid("not-a-uuid")


def test_sanitize_query_strips_whitespace() -> None:
    assert sanitize_query("  hello world  ") == "hello world"


def test_sanitize_query_empty_raises() -> None:
    with pytest.raises(ValidationError):
        sanitize_query("   ")


def test_sanitize_query_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        sanitize_query("a" * 3000)


def test_validate_ingestion_url_blocks_localhost() -> None:
    with pytest.raises(ValidationError):
        validate_ingestion_url("http://localhost:8000/admin")


def test_validate_ingestion_url_blocks_metadata_endpoint() -> None:
    with pytest.raises(ValidationError):
        validate_ingestion_url("http://169.254.169.254/latest/meta-data/")


def test_validate_ingestion_url_blocks_private_network() -> None:
    with pytest.raises(ValidationError):
        validate_ingestion_url("http://192.168.1.1/internal")


def test_validate_ingestion_url_allows_public_https() -> None:
    url = "https://example.com/article"
    assert validate_ingestion_url(url) == url


def test_validate_ingestion_url_rejects_bad_scheme() -> None:
    with pytest.raises(ValidationError):
        validate_ingestion_url("ftp://example.com/file")


# ── API Key Auth ──────────────────────────────────────────────────────────────

def _make_request(path: str = "/api/v1/search", headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 12345),
        "query_string": b"",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_verify_api_key_disabled_when_no_keys_set() -> None:
    from app.core.middleware.auth import verify_api_key
    with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
        request = _make_request()
        result = await verify_api_key(request)
        assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_missing_header_raises() -> None:
    from app.core.middleware.auth import verify_api_key
    with patch.dict(os.environ, {"API_KEYS": "secret-key-123"}, clear=False):
        request = _make_request()
        with pytest.raises(AuthenticationError, match="Missing"):
            await verify_api_key(request)


@pytest.mark.asyncio
async def test_verify_api_key_valid_key_passes() -> None:
    from app.core.middleware.auth import verify_api_key
    with patch.dict(os.environ, {"API_KEYS": "secret-key-123"}, clear=False):
        request = _make_request(headers={"X-API-Key": "secret-key-123"})
        result = await verify_api_key(request)
        assert result == "secret-key-123"


@pytest.mark.asyncio
async def test_verify_api_key_invalid_key_raises() -> None:
    from app.core.middleware.auth import verify_api_key
    with patch.dict(os.environ, {"API_KEYS": "secret-key-123"}, clear=False):
        request = _make_request(headers={"X-API-Key": "wrong-key"})
        with pytest.raises(AuthenticationError, match="Invalid"):
            await verify_api_key(request)


@pytest.mark.asyncio
async def test_verify_api_key_public_path_skips_auth() -> None:
    from app.core.middleware.auth import verify_api_key
    with patch.dict(os.environ, {"API_KEYS": "secret-key-123"}, clear=False):
        request = _make_request(path="/health")
        result = await verify_api_key(request)
        assert result is None


def test_hash_api_key_is_deterministic() -> None:
    from app.core.middleware.auth import hash_api_key
    h1 = hash_api_key("my-secret-key")
    h2 = hash_api_key("my-secret-key")
    assert h1 == h2
    assert h1 != "my-secret-key"


# ── Usage tracking ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_record_from_llm_response() -> None:
    from app.core.tracking import UsageRecord
    from app.services.llm.base import LLMResponse

    response = LLMResponse(
        content="answer",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost_usd=0.001,
        latency_ms=200,
    )
    record = UsageRecord.from_llm_response(response, request_type="chat", dataset_id="ds-1")
    assert record.request_type == "chat"
    assert record.provider == "openai"
    assert record.total_tokens == 150
    assert record.cost_usd == 0.001


@pytest.mark.asyncio
async def test_track_usage_calls_repository() -> None:
    from app.core.tracking import UsageRecord, track_usage

    record = UsageRecord(request_type="search", dataset_id="ds-1", total_tokens=10)
    mock_session = MagicMock()

    with patch("app.db.repositories.AnalyticsRepository") as MockRepo:
        MockRepo.return_value.create = AsyncMock()
        await track_usage(record, mock_session)
        MockRepo.return_value.create.assert_called_once()


@pytest.mark.asyncio
async def test_track_usage_never_raises_on_failure() -> None:
    from app.core.tracking import UsageRecord, track_usage

    record = UsageRecord(request_type="search")
    mock_session = MagicMock()

    with patch("app.db.repositories.AnalyticsRepository") as MockRepo:
        MockRepo.return_value.create = AsyncMock(side_effect=Exception("DB down"))
        # Should not raise
        await track_usage(record, mock_session)


# ── Observability (Langfuse) ──────────────────────────────────────────────────

def test_get_langfuse_returns_none_when_disabled() -> None:
    from app.core.observability import get_langfuse
    with patch("app.core.observability.get_settings") as ms:
        ms.return_value.langfuse_enabled = False
        result = get_langfuse()
    assert result is None


def test_rag_trace_start_noop_when_langfuse_disabled() -> None:
    from app.core.observability import RAGTrace
    with patch("app.core.observability.get_langfuse", return_value=None):
        trace = RAGTrace(name="test-trace")
        result = trace.start()
        assert result is trace  # returns self for chaining
        # Should not raise even with no client
        trace.end(output="done")


def test_rag_trace_log_retrieval_noop_when_no_trace() -> None:
    from app.core.observability import RAGTrace
    with patch("app.core.observability.get_langfuse", return_value=None):
        trace = RAGTrace(name="test").start()
        # Should not raise
        trace.log_retrieval(query="test", chunks=[], latency_ms=10)
