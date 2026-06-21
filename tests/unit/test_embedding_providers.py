"""
Tests for embedding providers — OpenAI call is mocked, no API key needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding.providers import OpenAIEmbedding, SentenceTransformerEmbedding


# ── OpenAI ────────────────────────────────────────────────────────────────────

@pytest.fixture
def openai_embedding() -> OpenAIEmbedding:
    with patch("app.services.embedding.providers.get_settings") as mock_settings:
        mock_settings.return_value.openai_api_key = "sk-test"
        mock_settings.return_value.embedding_model = "text-embedding-3-small"
        with patch("app.services.embedding.providers.AsyncOpenAI"):
            provider = OpenAIEmbedding(model="text-embedding-3-small")
    return provider


@pytest.mark.asyncio
async def test_openai_embed_texts_returns_vectors(openai_embedding: OpenAIEmbedding) -> None:
    fake_embedding = [0.1] * 1536
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_embedding)]
    openai_embedding._client = AsyncMock()
    openai_embedding._client.embeddings.create = AsyncMock(return_value=mock_resp)

    result = await openai_embedding.embed_texts(["hello world"])
    assert len(result) == 1
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_openai_embed_empty_returns_empty(openai_embedding: OpenAIEmbedding) -> None:
    result = await openai_embedding.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_openai_embed_query_returns_single_vector(openai_embedding: OpenAIEmbedding) -> None:
    fake_embedding = [0.5] * 1536
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_embedding)]
    openai_embedding._client = AsyncMock()
    openai_embedding._client.embeddings.create = AsyncMock(return_value=mock_resp)

    result = await openai_embedding.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) == 1536


def test_openai_dimensions() -> None:
    with patch("app.services.embedding.providers.get_settings") as ms:
        ms.return_value.openai_api_key = "sk-test"
        with patch("app.services.embedding.providers.AsyncOpenAI"):
            p = OpenAIEmbedding(model="text-embedding-3-large")
    assert p.dimensions == 3072


def test_openai_model_name() -> None:
    with patch("app.services.embedding.providers.get_settings") as ms:
        ms.return_value.openai_api_key = "sk-test"
        with patch("app.services.embedding.providers.AsyncOpenAI"):
            p = OpenAIEmbedding(model="text-embedding-3-small")
    assert p.model_name == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_openai_batches_large_input(openai_embedding: OpenAIEmbedding) -> None:
    """Verify that 150 texts triggers 2 API calls (batch_size=100)."""
    fake_embedding = [0.1] * 1536
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=fake_embedding)] * 100
    openai_embedding._client = AsyncMock()
    openai_embedding._client.embeddings.create = AsyncMock(return_value=mock_resp)
    openai_embedding._batch_size = 100

    texts = ["text"] * 150
    # Patch second call to return 50 items
    mock_resp2 = MagicMock()
    mock_resp2.data = [MagicMock(embedding=fake_embedding)] * 50
    openai_embedding._client.embeddings.create.side_effect = [mock_resp, mock_resp2]

    result = await openai_embedding.embed_texts(texts)
    assert len(result) == 150
    assert openai_embedding._client.embeddings.create.call_count == 2
