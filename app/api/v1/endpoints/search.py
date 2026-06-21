"""
Search API — T40.
POST /api/v1/search
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.db.repositories import AnalyticsRepository
from app.schemas.search import ChunkResult, SearchRequest, SearchResponse
from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    session: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Retrieve relevant chunks for a query against a dataset.
    Supports standard (dense only) and hybrid (dense + BM25) modes.
    """
    start = time.monotonic()
    request_id = str(uuid.uuid4())

    config = RetrievalConfig(
        mode=body.mode,
        top_k=body.top_k,
        candidate_k=body.top_k * 3,
        rerank=body.rerank,
        rerank_top_k=body.top_k,
        expand_query=body.expand_query,
        score_threshold=body.score_threshold,
    )

    result = await run_retrieval_pipeline(
        dataset_id=body.dataset_id,
        query=body.query,
        config=config,
        session=session,
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    chunks = [
        ChunkResult(
            chunk_id=c["id"],
            score=c["score"],
            text=c["text"],
            metadata=c["metadata"],
            document_title=c["metadata"].get("document_title", ""),
            filename=c["metadata"].get("filename", ""),
            source_url=c["metadata"].get("source_url", ""),
        )
        for c in result.context.chunks
    ]

    # Track analytics
    try:
        analytics_repo = AnalyticsRepository(session)
        await analytics_repo.create(
            request_id=request_id,
            request_type="search",
            dataset_id=body.dataset_id,
            latency_ms=latency_ms,
            status="success",
        )
    except Exception:
        pass

    return SearchResponse(
        query=body.query,
        dataset_id=body.dataset_id,
        mode=body.mode,
        results=chunks,
        total_results=len(chunks),
        latency_ms=latency_ms,
        reranked=body.rerank,
    )
