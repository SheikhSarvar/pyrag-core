"""
Search API — T40.
POST /api/v1/search
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.db.repositories import AnalyticsRepository
from app.schemas.search import (
    ChunkResult,
    RetrievalTraceChunk,
    SearchDebugResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.retrieval.cache import (
    build_search_cache_key,
    get_cached_search,
    get_dataset_cache_version,
    set_cached_search,
)
from app.services.retrieval.metadata_filter import resolve_metadata_chunk_ids
from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline

router = APIRouter()
logger = get_logger(__name__)

_SEARCH_CACHE_TTL_SECONDS = 300


def _sanitize_filters(filters: dict | None) -> dict | None:
    if not filters:
        return None

    cleaned: dict[str, Any] = {}
    for key, value in filters.items():
        if key.startswith("additionalProp"):
            continue
        if value in (None, "", [], {}):
            continue
        cleaned[key] = value

    return cleaned or None


def _trace_chunk(candidate: Any) -> RetrievalTraceChunk:
    data = {
        "chunk_id": getattr(candidate, "chunk_id", getattr(candidate, "id", "")),
        "text": getattr(candidate, "chunk_text", ""),
        "metadata": getattr(candidate, "metadata", {}) or {},
        "score": getattr(candidate, "score", None),
        "rerank_score": getattr(candidate, "rerank_score", None),
        "rrf_score": getattr(candidate, "rrf_score", None),
        "dense_score": getattr(candidate, "dense_score", None),
        "sparse_score": getattr(candidate, "sparse_score", None),
    }
    return RetrievalTraceChunk(**data)


async def _search_impl(
    body: SearchRequest,
    session: AsyncSession,
    *,
    debug: bool = False,
) -> tuple[SearchResponse, dict[str, Any]]:
    start = time.monotonic()
    request_id = str(uuid.uuid4())

    cache_version = await get_dataset_cache_version(body.dataset_id)
    filters = _sanitize_filters(body.filters)
    cache_key = build_search_cache_key(
        dataset_id=body.dataset_id,
        query=body.query,
        mode=body.mode,
        top_k=body.top_k,
        rerank=body.rerank,
        expand_query=body.expand_query,
        score_threshold=body.score_threshold,
        filters=filters,
        cache_version=cache_version,
    )

    if not debug:
        cached = await get_cached_search(cache_key)
        if cached is not None and cached.get("total_results", 0) > 0:
            latency_ms = int((time.monotonic() - start) * 1000)
            cached_response = SearchResponse(
                query=cached["query"],
                dataset_id=cached["dataset_id"],
                mode=cached["mode"],
                results=[ChunkResult.model_validate(item) for item in cached["results"]],
                total_results=cached["total_results"],
                latency_ms=latency_ms,
                reranked=cached["reranked"],
                cache_hit=True,
                rerank_status=cached.get("rerank_status", "not_requested"),
                rerank_error=cached.get("rerank_error"),
            )
            return cached_response, {
                "cache_hit": True,
                "cache_key": cache_key,
                "metadata_filtered_chunk_ids": cached.get("metadata_filtered_chunk_ids", []),
                "raw_candidates": cached.get("raw_candidates", []),
                "reranked_candidates": cached.get("reranked_candidates", []),
            }

    metadata_filtered_chunk_ids: list[str] = []
    effective_filters = filters
    if filters:
        metadata_filtered_chunk_ids = await resolve_metadata_chunk_ids(session, body.dataset_id, filters)
        if not metadata_filtered_chunk_ids:
            latency_ms = int((time.monotonic() - start) * 1000)
            empty_response = SearchResponse(
                query=body.query,
                dataset_id=body.dataset_id,
                mode=body.mode,
                results=[],
                total_results=0,
                latency_ms=latency_ms,
                reranked=body.rerank,
                cache_hit=False,
                rerank_status="empty",
            )
            return empty_response, {
                "cache_hit": False,
                "cache_key": cache_key,
                "metadata_filtered_chunk_ids": [],
                "raw_candidates": [],
                "reranked_candidates": [],
            }
        effective_filters = dict(filters)
        effective_filters["chunk_id"] = metadata_filtered_chunk_ids

    config = RetrievalConfig(
        mode=body.mode,
        top_k=body.top_k,
        candidate_k=body.top_k * 3,
        rerank=body.rerank,
        rerank_top_k=body.top_k,
        expand_query=body.expand_query,
        score_threshold=body.score_threshold,
        filters=effective_filters,
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

    response = SearchResponse(
        query=body.query,
        dataset_id=body.dataset_id,
        mode=body.mode,
        results=chunks,
        total_results=len(chunks),
        latency_ms=latency_ms,
        reranked=body.rerank,
        cache_hit=False,
        rerank_status=result.rerank_status,
        rerank_error=result.rerank_error,
    )

    if not debug:
        if response.total_results > 0:
            await set_cached_search(
                cache_key,
                {
                    "query": response.query,
                    "dataset_id": response.dataset_id,
                    "mode": response.mode,
                    "results": [item.model_dump() for item in response.results],
                    "total_results": response.total_results,
                    "reranked": response.reranked,
                    "rerank_status": response.rerank_status,
                    "rerank_error": response.rerank_error,
                    "metadata_filtered_chunk_ids": metadata_filtered_chunk_ids,
                    "raw_candidates": [_trace_chunk(candidate).model_dump() for candidate in result.raw_candidates],
                    "reranked_candidates": [_trace_chunk(candidate).model_dump() for candidate in result.reranked_candidates],
                },
                ttl_seconds=_SEARCH_CACHE_TTL_SECONDS,
            )

    trace = {
        "cache_hit": False,
        "cache_key": cache_key,
        "metadata_filtered_chunk_ids": metadata_filtered_chunk_ids,
        "raw_candidates": [_trace_chunk(candidate).model_dump() for candidate in result.raw_candidates],
        "reranked_candidates": [_trace_chunk(candidate).model_dump() for candidate in result.reranked_candidates],
        "rerank_status": result.rerank_status,
        "rerank_error": result.rerank_error,
    }
    return response, trace


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    session: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Retrieve relevant chunks for a query against a dataset.
    Supports standard (dense only) and hybrid (dense + BM25) modes.
    """
    request_id = str(uuid.uuid4())
    response, _trace = await _search_impl(body, session, debug=False)

    # Track analytics
    try:
        analytics_repo = AnalyticsRepository(session)
        await analytics_repo.create(
            request_id=request_id,
            request_type="search",
            dataset_id=body.dataset_id,
            latency_ms=response.latency_ms,
            status="success",
        )
    except Exception:
        pass

    return response


@router.post("/debug", response_model=SearchDebugResponse)
async def search_debug(
    body: SearchRequest,
    session: AsyncSession = Depends(get_db),
) -> SearchDebugResponse:
    response, trace = await _search_impl(body, session, debug=True)
    return SearchDebugResponse(
        query=response.query,
        dataset_id=response.dataset_id,
        mode=response.mode,
        cache_hit=trace["cache_hit"],
        cache_key=trace["cache_key"],
        metadata_filtered_chunk_ids=trace["metadata_filtered_chunk_ids"],
        raw_candidates=[RetrievalTraceChunk.model_validate(item) for item in trace["raw_candidates"]],
        reranked_candidates=[RetrievalTraceChunk.model_validate(item) for item in trace["reranked_candidates"]],
        final_chunks=response.results,
        total_results=response.total_results,
        latency_ms=response.latency_ms,
        reranked=response.reranked,
        rerank_status=trace["rerank_status"],
        rerank_error=trace["rerank_error"],
    )
