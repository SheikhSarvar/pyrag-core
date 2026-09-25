"""
Retrieval pipeline orchestrator.
Wires together: understand -> expand -> retrieve (dense|sparse|hybrid) -> rerank -> compress -> assemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.services.retrieval.context import (
    AssembledPrompt,
    CompressedContext,
    assemble_prompt,
    compress_context,
)
from app.services.retrieval.query_expansion import expand_query
from app.services.retrieval.query_understanding import UnderstoodQuery, understand_query

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"  # standard | hybrid | agentic
    top_k: int = 10
    candidate_k: int = 30
    rerank: bool = True
    rerank_backend: str = "local"  # local | cohere
    rerank_top_k: int = 5
    expand_query: bool = False
    use_llm_expansion: bool = False
    max_context_tokens: int = 4000
    dedup_threshold: float = 0.85
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    score_threshold: float | None = None
    filters: dict | None = None


@dataclass
class RetrievalResult:
    query: UnderstoodQuery
    prompt: AssembledPrompt
    context: CompressedContext
    raw_result_count: int
    mode: str
    raw_candidates: list[Any] = field(default_factory=list)
    reranked_candidates: list[Any] = field(default_factory=list)
    rerank_status: str = "not_requested"
    rerank_error: str | None = None


async def run_retrieval_pipeline(
    dataset_id: str,
    query: str,
    config: RetrievalConfig | None = None,
    system_prompt: str | None = None,
    session=None,
) -> RetrievalResult:
    """
    Full retrieval pipeline from raw query to assembled LLM prompt.
    """
    cfg = config or RetrievalConfig()

    understood = understand_query(query)
    search_query = understood.normalized
    logger.debug("Query understood", intent=understood.intent, is_question=understood.is_question)

    search_queries = [search_query]
    if cfg.expand_query:
        expanded = await expand_query(search_query, use_llm=cfg.use_llm_expansion)
        search_queries = expanded.all_queries
        logger.debug("Query expanded", variants=len(expanded.variants))

    raw_results: list = []
    raw_candidates: list[Any] = []
    rerank_outcome = None

    if cfg.mode == "standard":
        from app.services.retrieval.dense import dense_search

        for search_term in search_queries:
            raw_results.extend(
                await dense_search(
                    dataset_id,
                    search_term,
                    top_k=cfg.candidate_k,
                    score_threshold=cfg.score_threshold,
                    filters=cfg.filters,
                )
            )
        raw_candidates = list(raw_results)

    elif cfg.mode == "hybrid":
        from app.services.retrieval.hybrid import hybrid_search

        for search_term in search_queries:
            raw_results.extend(
                await hybrid_search(
                    dataset_id,
                    search_term,
                    top_k=cfg.candidate_k,
                    candidate_k=cfg.candidate_k,
                    dense_weight=cfg.dense_weight,
                    sparse_weight=cfg.sparse_weight,
                    score_threshold=cfg.score_threshold,
                    filters=cfg.filters,
                    session=session,
                )
            )
        raw_candidates = list(raw_results)

    else:
        raw_results = []
        raw_candidates = []

    logger.debug("Retrieved candidates", count=len(raw_results), mode=cfg.mode)

    if cfg.rerank and raw_results:
        from app.services.retrieval.reranker import rerank_results

        rerank_outcome = await rerank_results(
            search_query,
            raw_results,
            top_k=cfg.rerank_top_k,
            backend=cfg.rerank_backend,
        )
        raw_results = rerank_outcome.results
        logger.debug("Reranked", kept=len(raw_results), status=rerank_outcome.status)
    elif not cfg.rerank:
        raw_results = raw_results[: cfg.top_k]

    raw_count = len(raw_results)
    rerank_status = "not_requested"
    rerank_error: str | None = None
    if cfg.rerank:
        if rerank_outcome is not None:
            rerank_status = rerank_outcome.status
            rerank_error = rerank_outcome.error
        elif not raw_results:
            rerank_status = "empty"
    else:
        rerank_status = "disabled"

    context = compress_context(
        raw_results,
        max_tokens=cfg.max_context_tokens,
        dedup_threshold=cfg.dedup_threshold,
    )

    prompt = assemble_prompt(
        query=understood.original,
        context=context,
        system_prompt=system_prompt,
        intent=understood.intent,
    )

    return RetrievalResult(
        query=understood,
        prompt=prompt,
        context=context,
        raw_result_count=raw_count,
        mode=cfg.mode,
        raw_candidates=raw_candidates,
        reranked_candidates=list(raw_results),
        rerank_status=rerank_status,
        rerank_error=rerank_error,
    )
