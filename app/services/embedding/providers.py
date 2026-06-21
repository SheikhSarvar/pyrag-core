"""
Embedding service — T19.
Supports OpenAI, sentence-transformers, and Ollama.
All providers implement the same interface: embed_texts(list[str]) -> list[list[float]]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIEmbedding(EmbeddingProvider):

    _DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100) -> None:
        self._model = model
        self._batch_size = batch_size
        settings = get_settings()
        if not settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                resp = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                results.extend([r.embedding for r in resp.data])
                logger.debug("Embedded batch", provider="openai", count=len(batch))
            return results
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]


# ── Sentence Transformers (local) ─────────────────────────────────────────────

class SentenceTransformerEmbedding(EmbeddingProvider):

    def __init__(self, model: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        self._model_name = model
        self._batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model)
            self._dims = self._model.get_sentence_embedding_dimension() or 384
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        if not texts:
            return []
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False),
            )
            return [e.tolist() for e in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"SentenceTransformer embedding failed: {exc}") from exc

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaEmbedding(EmbeddingProvider):

    _DEFAULT_DIMS: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(self, model: str = "nomic-embed-text") -> None:
        self._model = model
        settings = get_settings()
        self._base_url = settings.ollama_base_url

    @property
    def dimensions(self) -> int:
        return self._DEFAULT_DIMS.get(self._model, 768)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import httpx
        results: list[list[float]] = []
        async with httpx.AsyncClient(base_url=self._base_url, timeout=60) as client:
            for text in texts:
                try:
                    resp = await client.post("/api/embeddings", json={"model": self._model, "prompt": text})
                    resp.raise_for_status()
                    results.append(resp.json()["embedding"])
                except Exception as exc:
                    raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc
        return results

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]


# ── Factory ───────────────────────────────────────────────────────────────────

@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    provider = settings.embedding_provider
    model = settings.embedding_model

    if provider == "openai":
        return OpenAIEmbedding(model=model)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedding(model=model)
    if provider == "ollama":
        return OllamaEmbedding(model=model)

    raise EmbeddingError(f"Unknown embedding provider: {provider!r}")
