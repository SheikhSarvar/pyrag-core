"""
Unit tests for repository layer.
Uses an in-memory SQLite DB (via aiosqlite) so no real Postgres needed.
"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.repositories import (
    AnalyticsRepository,
    ChunkRepository,
    DatasetRepository,
    DocumentRepository,
    JobRepository,
)
from app.core.exceptions import NotFoundError

# Import models to register them with metadata
import app.db.models  # noqa: F401

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── Dataset ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_get_dataset(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    ds = await repo.create(name="test-ds", chunk_strategy="recursive", embedding_model="text-embedding-3-small", embedding_dimensions=1536)
    assert ds.id is not None
    fetched = await repo.get(ds.id)
    assert fetched is not None
    assert fetched.name == "test-ds"


@pytest.mark.asyncio
async def test_dataset_not_found_raises(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    with pytest.raises(NotFoundError):
        await repo.get_or_raise("nonexistent-id")


@pytest.mark.asyncio
async def test_dataset_get_by_name(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    await repo.create(name="unique-name", chunk_strategy="fixed", embedding_model="text-embedding-3-small", embedding_dimensions=1536)
    found = await repo.get_by_name("unique-name")
    assert found is not None
    assert found.name == "unique-name"
    missing = await repo.get_by_name("does-not-exist")
    assert missing is None


@pytest.mark.asyncio
async def test_dataset_update(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    ds = await repo.create(name="old-name", chunk_strategy="fixed", embedding_model="text-embedding-3-small", embedding_dimensions=1536)
    updated = await repo.update(ds.id, name="new-name")
    assert updated.name == "new-name"


@pytest.mark.asyncio
async def test_dataset_delete(session: AsyncSession) -> None:
    repo = DatasetRepository(session)
    ds = await repo.create(name="to-delete", chunk_strategy="fixed", embedding_model="text-embedding-3-small", embedding_dimensions=1536)
    deleted = await repo.delete(ds.id)
    assert deleted is True
    assert await repo.get(ds.id) is None


# ── Document ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_document_and_set_status(session: AsyncSession) -> None:
    ds_repo = DatasetRepository(session)
    doc_repo = DocumentRepository(session)

    ds = await ds_repo.create(name="ds-for-doc", chunk_strategy="fixed", embedding_model="text-embedding-3-small", embedding_dimensions=1536)
    doc = await doc_repo.create(
        dataset_id=ds.id,
        name="test.pdf",
        original_name="test.pdf",
        file_type="pdf",
        file_size=1024,
    )
    assert doc.status == "pending"

    await doc_repo.set_status(doc.id, "indexed")
    await session.refresh(doc)
    assert doc.status == "indexed"


# ── Job ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_lifecycle(session: AsyncSession) -> None:
    job_repo = JobRepository(session)
    job = await job_repo.create(job_type="ingest", payload={"doc_id": "abc"})
    assert job.status == "queued"

    await job_repo.mark_started(job.id, celery_task_id="celery-123")
    await session.refresh(job)
    assert job.status == "processing"
    assert job.celery_task_id == "celery-123"

    await job_repo.mark_completed(job.id, result={"chunks": 42})
    await session.refresh(job)
    assert job.status == "completed"
    assert job.progress == 100


# ── Analytics ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analytics_aggregation(session: AsyncSession) -> None:
    repo = AnalyticsRepository(session)
    await repo.create(request_id="r1", request_type="chat", provider="openai", total_tokens=100, cost_usd=0.001, latency_ms=300)
    await repo.create(request_id="r2", request_type="chat", provider="openai", total_tokens=200, cost_usd=0.002, latency_ms=400)
    await repo.create(request_id="r3", request_type="search", provider="anthropic", total_tokens=50, cost_usd=0.0005, latency_ms=150)

    total_cost = await repo.total_cost()
    assert abs(total_cost - 0.0035) < 1e-6

    total_tokens = await repo.total_tokens()
    assert total_tokens == 350

    by_provider = await repo.cost_by_provider()
    providers = {r["provider"] for r in by_provider}
    assert "openai" in providers
    assert "anthropic" in providers
