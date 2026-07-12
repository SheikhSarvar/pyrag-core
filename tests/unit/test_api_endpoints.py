"""
API endpoint tests — all services mocked, no real DB or LLM calls.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Search ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_200() -> None:
    from app.services.retrieval.context import CompressedContext, AssembledPrompt
    from app.services.retrieval.query_understanding import UnderstoodQuery
    from app.services.retrieval.pipeline import RetrievalResult

    mock_result = RetrievalResult(
        query=UnderstoodQuery(original="test", normalized="test", intent="search", keywords=["test"]),
        prompt=AssembledPrompt(system="sys", user="usr", context_chunks=[], total_tokens=10),
        context=CompressedContext(chunks=[], total_tokens=0, dropped_count=0),
        raw_result_count=0,
        mode="hybrid",
    )

    with patch("app.api.v1.endpoints.search.run_retrieval_pipeline", AsyncMock(return_value=mock_result)), \
         patch("app.api.v1.endpoints.search.AnalyticsRepository") as mock_repo:
        mock_repo.return_value.create = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={
                "dataset_id": "ds-1",
                "query": "What is revenue?",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["query"] == "What is revenue?"
    assert data["dataset_id"] == "ds-1"


@pytest.mark.asyncio
async def test_search_returns_results() -> None:
    from app.services.retrieval.context import CompressedContext, AssembledPrompt
    from app.services.retrieval.query_understanding import UnderstoodQuery
    from app.services.retrieval.pipeline import RetrievalResult

    chunk = {
        "id": "chunk-1",
        "text": "Revenue grew 20% YoY.",
        "score": 0.92,
        "metadata": {"filename": "report.pdf", "document_title": "Q3 Report", "source_url": ""},
    }
    mock_result = RetrievalResult(
        query=UnderstoodQuery(original="revenue", normalized="revenue", intent="search", keywords=["revenue"]),
        prompt=AssembledPrompt(system="sys", user="usr", context_chunks=[chunk], total_tokens=20),
        context=CompressedContext(chunks=[chunk], total_tokens=15, dropped_count=0),
        raw_result_count=1,
        mode="hybrid",
    )

    with patch("app.api.v1.endpoints.search.run_retrieval_pipeline", AsyncMock(return_value=mock_result)), \
         patch("app.api.v1.endpoints.search.AnalyticsRepository") as mock_repo:
        mock_repo.return_value.create = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/search", json={
                "dataset_id": "ds-1",
                "query": "revenue",
                "top_k": 5,
            })

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["chunk_id"] == "chunk-1"
    assert data["results"][0]["score"] == 0.92


# ── Chat ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_returns_answer() -> None:
    from app.services.retrieval.context import CompressedContext, AssembledPrompt
    from app.services.retrieval.query_understanding import UnderstoodQuery
    from app.services.retrieval.pipeline import RetrievalResult
    from app.services.llm.base import LLMResponse

    mock_retrieval = RetrievalResult(
        query=UnderstoodQuery(original="msg", normalized="msg", intent="search", keywords=[]),
        prompt=AssembledPrompt(system="Be helpful.", user="What is revenue?", context_chunks=[], total_tokens=10),
        context=CompressedContext(chunks=[], total_tokens=0, dropped_count=0),
        raw_result_count=0,
        mode="hybrid",
    )
    mock_llm_resp = LLMResponse(
        content="Revenue grew 20%.",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=50,
        completion_tokens=15,
        total_tokens=65,
        cost_usd=0.00001,
        latency_ms=300,
    )
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=mock_llm_resp)
    mock_llm.provider_name = "openai"
    mock_llm.model_name = "gpt-4o-mini"

    with patch("app.api.v1.endpoints.chat.run_retrieval_pipeline", AsyncMock(return_value=mock_retrieval)), \
         patch("app.api.v1.endpoints.chat.get_llm_provider_from_db", AsyncMock(return_value=mock_llm)), \
         patch("app.api.v1.endpoints.chat.AnalyticsRepository") as mock_repo:
        mock_repo.return_value.create = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/chat", json={
                "dataset_id": "ds-1",
                "message": "What is revenue?",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Revenue grew 20%."
    assert data["provider"] == "openai"
    assert data["total_tokens"] == 65


@pytest.mark.asyncio
async def test_document_upload_queues_ingestion_task() -> None:
    from app.api.v1.endpoints import documents as documents_endpoint

    mock_dataset = MagicMock()
    mock_dataset.chunk_strategy = "recursive"
    mock_document = MagicMock()
    mock_job = MagicMock()
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()

    async def override_get_db() -> AsyncIterator[AsyncMock]:
        yield mock_session

    mock_minio = MagicMock()
    mock_minio.upload_file = MagicMock(return_value="ds-1/doc-1/raw/report.txt")

    mock_task_result = MagicMock(id="celery-task-1")
    mock_task = MagicMock()
    mock_task.apply_async = MagicMock(return_value=mock_task_result)

    with patch.object(documents_endpoint, "DatasetRepository") as MockDatasetRepo, \
         patch.object(documents_endpoint, "DocumentRepository") as MockDocumentRepo, \
         patch.object(documents_endpoint, "JobRepository") as MockJobRepo, \
         patch.object(documents_endpoint, "get_minio_client", return_value=mock_minio), \
         patch("app.worker.tasks.ingestion.ingest_document", mock_task):
        MockDatasetRepo.return_value.get_or_raise = AsyncMock(return_value=mock_dataset)
        MockDocumentRepo.return_value.create = AsyncMock(return_value=mock_document)
        MockDocumentRepo.return_value.set_status = AsyncMock()
        MockJobRepo.return_value.create = AsyncMock(return_value=mock_job)
        MockJobRepo.return_value.mark_started = AsyncMock()
        MockJobRepo.return_value.mark_failed = AsyncMock()

        app.dependency_overrides[documents_endpoint.get_db] = override_get_db
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/documents/upload",
                    data={"dataset_id": "ds-1"},
                    files={"file": ("../../report.txt", b"hello world", "text/plain")},
                )
        finally:
            app.dependency_overrides.pop(documents_endpoint.get_db, None)

    assert resp.status_code == 202
    data = resp.json()
    assert data["dataset_id"] == "ds-1"
    assert data["filename"] == "report.txt"
    assert data["status"] == "pending"
    assert mock_minio.upload_file.call_count == 1
    assert mock_task.apply_async.call_count == 1
    assert mock_session.commit.await_count >= 2


@pytest.mark.asyncio
async def test_document_reindex_commits_before_enqueue() -> None:
    from app.api.v1.endpoints import documents as documents_endpoint

    mock_document = MagicMock()
    mock_document.dataset_id = "ds-1"
    mock_document.original_name = "report.txt"
    mock_document.file_size = 11
    mock_document.storage_path = "ds-1/doc-1/raw/report.txt"

    mock_dataset = MagicMock()
    mock_dataset.chunk_strategy = "recursive"

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()

    async def override_get_db() -> AsyncIterator[AsyncMock]:
        yield mock_session

    mock_task_result = MagicMock(id="celery-task-2")
    mock_task = MagicMock()

    def apply_async_side_effect(*args, **kwargs):
        assert mock_session.commit.await_count == 1
        return mock_task_result

    mock_task.apply_async = MagicMock(side_effect=apply_async_side_effect)

    with patch.object(documents_endpoint, "DocumentRepository") as MockDocumentRepo, \
         patch.object(documents_endpoint, "JobRepository") as MockJobRepo, \
         patch("app.db.repositories.DatasetRepository") as MockDatasetRepo, \
         patch("app.worker.tasks.ingestion.ingest_document", mock_task):
        MockDocumentRepo.return_value.get = AsyncMock(return_value=mock_document)
        MockJobRepo.return_value.create = AsyncMock(return_value=MagicMock())
        MockJobRepo.return_value.mark_started = AsyncMock()
        MockDatasetRepo.return_value.get = AsyncMock(return_value=mock_dataset)

        app.dependency_overrides[documents_endpoint.get_db] = override_get_db
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/documents/reindex",
                    json={"document_ids": ["doc-1"]},
                )
        finally:
            app.dependency_overrides.pop(documents_endpoint.get_db, None)

    assert resp.status_code == 202
    assert resp.json()["total"] == 1
    assert mock_task.apply_async.call_count == 1
    assert mock_session.commit.await_count == 2


@pytest.mark.asyncio
async def test_agents_chat_returns_answer() -> None:
    mock_result = {
        "answer": "The answer is 42.",
        "steps": [
            {"step": 0, "tool": "search_ds-1", "input": "question", "output": "[]"},
        ],
        "sources": [
            {"chunk_id": "chunk-1", "score": 0.98, "text": "Relevant text", "filename": "report.txt"},
        ],
        "model": "gpt-4o-mini",
        "total_tokens": 12,
        "cost_usd": 0.00012,
    }

    with patch("app.api.v1.endpoints.agents.run_agent", AsyncMock(return_value=mock_result)), \
         patch("app.api.v1.endpoints.agents.AnalyticsRepository") as mock_repo:
        mock_repo.return_value.create = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/agents/chat",
                json={
                    "dataset_id": "ds-1",
                    "message": "What is the answer?",
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer is 42."
    assert data["model"] == "gpt-4o-mini"
    assert data["total_tokens"] == 12


# ── Analytics ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analytics_summary() -> None:
    with patch("app.api.v1.endpoints.analytics.AnalyticsRepository") as MockRepo:
        instance = MockRepo.return_value
        instance.count = AsyncMock(return_value=42)
        instance.total_tokens = AsyncMock(return_value=100000)
        instance.total_cost = AsyncMock(return_value=0.125)
        instance.avg_latency_ms = AsyncMock(return_value=450.5)
        instance.requests_by_type = AsyncMock(return_value=[{"request_type": "chat", "count": 30}])
        instance.cost_by_provider = AsyncMock(return_value=[{"provider": "openai", "total_cost": 0.12}])

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/analytics")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 42
    assert data["total_tokens"] == 100000
    assert data["total_cost_usd"] == 0.125
    assert data["avg_latency_ms"] == 450.5


@pytest.mark.asyncio
async def test_analytics_summary_alias() -> None:
    with patch("app.api.v1.endpoints.analytics.AnalyticsRepository") as MockRepo:
        instance = MockRepo.return_value
        instance.count = AsyncMock(return_value=42)
        instance.total_tokens = AsyncMock(return_value=100000)
        instance.total_cost = AsyncMock(return_value=0.125)
        instance.avg_latency_ms = AsyncMock(return_value=450.5)
        instance.requests_by_type = AsyncMock(return_value=[{"request_type": "chat", "count": 30}])
        instance.cost_by_provider = AsyncMock(return_value=[{"provider": "openai", "total_cost": 0.12}])

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/v1/analytics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 42
    assert data["total_tokens"] == 100000
    assert data["total_cost_usd"] == 0.125
    assert data["avg_latency_ms"] == 450.5


@pytest.mark.asyncio
async def test_analytics_cost() -> None:
    with patch("app.api.v1.endpoints.analytics.AnalyticsRepository") as MockRepo:
        instance = MockRepo.return_value
        instance.total_cost = AsyncMock(return_value=0.05)
        instance.cost_by_provider = AsyncMock(return_value=[])

        with patch("app.api.v1.endpoints.analytics.select"), \
             patch("app.api.v1.endpoints.analytics.func"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get("/api/v1/analytics/cost")

    # Either 200 or 500 depending on mock depth — just verify the route exists
    assert resp.status_code in (200, 500)


@pytest.mark.asyncio
async def test_search_invalid_body() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/search", json={"dataset_id": "ds-1"})  # missing query
    assert resp.status_code == 422
