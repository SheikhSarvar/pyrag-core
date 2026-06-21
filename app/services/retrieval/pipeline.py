"""
Retrieval pipeline orchestrator.
Wires together: understand → expand → retrieve (dense|sparse|hybrid) → rerank → compress → assemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.retrieval.context import AssembledPrompt, CompressedContext, assemble_prompt, compress_context
from app.services.retrieval.query_expansion import expand_query
from app.services.retrieval.query_understanding import UnderstoodQuery, understand_query

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"             # standard | hybrid | agentic
    top_k: int = 10
    candidate_k: int = 30
    rerank: bool = True
    rerank_backend: str = "local"    # local | cohere
    rerank_top_k: int = 5
    expand_query: bool = False
    use_llm_expansion: bool = False
    max_context_tokens: int = 4000
    dedup_threshold: float = 0.85
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    score_threshold: float | None = None


@dataclass
class RetrievalResult:
    query: UnderstoodQuery
    prompt: AssembledPrompt
    context: CompressedContext
    raw_result_count: int
    mode: str


async def run_retrieval_pipeline(
    dataset_id: str,
    query: str,
    config: RetrievalConfig | None = None,
    system_prompt: str | None = None,
    session=None,
) -> RetrievalResult:
    """
    Full retrieval pipeline from raw query to assembled LLM prompt.

    Args:
        dataset_id:    Which dataset to search.
        query:         Raw user query.
        config:        Retrieval parameters (defaults to hybrid + rerank).
        system_prompt: Override system prompt for prompt assembly.
        session:       AsyncSession (required for sparse/hybrid retrieval).

    Returns:
        RetrievalResult with assembled prompt ready for an LLM call.
    """
    cfg = config or RetrievalConfig()

    # T22 — Query understanding
    understood = understand_query(query)
    search_query = understood.normalized
    logger.debug("Query understood", intent=understood.intent, is_question=understood.is_question)

    # T23 — Optional query expansion
    if cfg.expand_query:
        expanded = await expand_query(search_query, use_llm=cfg.use_llm_expansion)
        search_query = expanded.original  # Use original; variants used for multi-query below
        logger.debug("Query expanded", variants=len(expanded.variants))

    # T24 / T25 / T26 — Retrieval
    raw_results: list = []

    if cfg.mode == "standard":
        from app.services.retrieval.dense import dense_search
        raw_results = await dense_search(
            dataset_id, search_query,
            top_k=cfg.candidate_k,
            score_threshold=cfg.score_threshold,
        )

    elif cfg.mode == "hybrid":
        from app.services.retrieval.hybrid import hybrid_search
        raw_results = await hybrid_search(
            dataset_id, search_query,
            top_k=cfg.candidate_k,
            candidate_k=cfg.candidate_k,
            dense_weight=cfg.dense_weight,
            sparse_weight=cfg.sparse_weight,
            session=session,
        )

    else:  # agentic — caller manages retrieval via tools
        raw_results = []

    logger.debug("Retrieved candidates", count=len(raw_results), mode=cfg.mode)

    # T27 — Reranking
    if cfg.rerank and raw_results:
        from app.services.retrieval.reranker import rerank_results
        raw_results = await rerank_results(
            search_query,
            raw_results,
            top_k=cfg.rerank_top_k,
            backend=cfg.rerank_backend,
        )
        logger.debug("Reranked", kept=len(raw_results))

    elif not cfg.rerank:
        raw_results = raw_results[: cfg.top_k]

    raw_count = len(raw_results)

    # T28 — Context compression
    context = compress_context(
        raw_results,
        max_tokens=cfg.max_context_tokens,
        dedup_threshold=cfg.dedup_threshold,
    )

    # T29 — Prompt assembly
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
    )
