"""
Tests for the indexer — uses in-memory vector store and a stub embedder.
No real Qdrant or OpenAI calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Dataset, Document  # noqa: F401 — register models
from app.db.repositories import DatasetRepository, DocumentRepository, ChunkRepository
from app.services.ingestion.chunkers import RecursiveChunker
from app.services.ingestion.indexer import (
    collection_name,
    delete_document_vectors,
    ensure_collection,
    index_chunks,
)
from tests.unit.test_vector_base import InMemoryVectorStore

TEST_DB = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def seeded(session: AsyncSession):
    ds_repo = DatasetRepository(session)
    doc_repo = DocumentRepository(session)
    ds = await ds_repo.create(
        name="test-ds",
        chunk_strategy="recursive",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    doc = await doc_repo.create(
        dataset_id=ds.id,
        name="file.txt",
        original_name="file.txt",
        file_type="txt",
        file_size=100,
    )
    return ds, doc


class StubEmbedder:
    dimensions = 4
    model_name = "stub"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_collection_name_format() -> None:
    name = collection_name("abc-123")
    assert name.startswith("pyrag_")
    assert "-" not in name


@pytest.mark.asyncio
async def test_ensure_collection_creates_if_missing() -> None:
    vs = InMemoryVectorStore()
    await ensure_collection(vs, "ds-1", dimensions=4)
    assert await vs.collection_exists("pyrag_ds_1")


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent() -> None:
    vs = InMemoryVectorStore()
    await ensure_collection(vs, "ds-1", dimensions=4)
    await ensure_collection(vs, "ds-1", dimensions=4)  # Should not raise
    assert await vs.collection_exists("pyrag_ds_1")


@pytest.mark.asyncio
async def test_index_chunks_writes_to_vector_store_and_db(
    session: AsyncSession, seeded: tuple
) -> None:
    ds, doc = seeded
    vs = InMemoryVectorStore()
    embedder = StubEmbedder()

    chunker = RecursiveChunker(chunk_size=50, overlap=5)
    text = "This is a test document. It has several sentences. Each one contributes content."
    chunks = chunker.chunk(text)

    total = await index_chunks(
        session=session,
        vector_store=vs,
        embedder=embedder,
        dataset_id=ds.id,
        document_id=doc.id,
        doc_metadata={"title": "Test", "filename": "file.txt", "file_type": "txt", "source_url": ""},
        chunks=chunks,
    )

    assert total == len(chunks)

    # Vector store has the right count
    col = collection_name(ds.id)
    assert await vs.count(col) == total

    # DB has chunk records
    chunk_repo = ChunkRepository(session)
    db_chunks = await chunk_repo.list_by_document(doc.id)
    assert len(db_chunks) == total


@pytest.mark.asyncio
async def test_index_chunks_empty_input(session: AsyncSession, seeded: tuple) -> None:
    ds, doc = seeded
    vs = InMemoryVectorStore()
    total = await index_chunks(
        session=session,
        vector_store=vs,
        embedder=StubEmbedder(),
        dataset_id=ds.id,
        document_id=doc.id,
        doc_metadata={},
        chunks=[],
    )
    assert total == 0


@pytest.mark.asyncio
async def test_index_chunks_chunk_metadata_stored(session: AsyncSession, seeded: tuple) -> None:
    ds, doc = seeded
    vs = InMemoryVectorStore()

    chunker = RecursiveChunker(chunk_size=200, overlap=10)
    chunks = chunker.chunk("Hello world. This is a test sentence for indexing.")

    await index_chunks(
        session=session,
        vector_store=vs,
        embedder=StubEmbedder(),
        dataset_id=ds.id,
        document_id=doc.id,
        doc_metadata={"title": "Doc", "filename": "f.txt", "file_type": "txt", "source_url": ""},
        chunks=chunks,
    )

    chunk_repo = ChunkRepository(session)
    db_chunks = await chunk_repo.list_by_document(doc.id)
    assert db_chunks[0].chunk_metadata is not None
    assert "chunk_index" in db_chunks[0].chunk_metadata


@pytest.mark.asyncio
async def test_delete_document_vectors(session: AsyncSession, seeded: tuple) -> None:
    ds, doc = seeded
    vs = InMemoryVectorStore()

    chunker = RecursiveChunker(chunk_size=200, overlap=10)
    chunks = chunker.chunk("Some content for deletion test.")
    col = collection_name(ds.id)

    await index_chunks(
        session=session,
        vector_store=vs,
        embedder=StubEmbedder(),
        dataset_id=ds.id,
        document_id=doc.id,
        doc_metadata={"title": "D", "filename": "f.txt", "file_type": "txt", "source_url": ""},
        chunks=chunks,
    )

    assert await vs.count(col) > 0
    await delete_document_vectors(vs, ds.id, doc.id)
    assert await vs.count(col) == 0
