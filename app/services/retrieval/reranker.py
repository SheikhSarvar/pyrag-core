"""
Reranker — T27.
Re-scores retrieved candidates with a cross-encoder model.
Two backends:
  1. sentence-transformers CrossEncoder (local, no API cost)
  2. Cohere Rerank API (cloud, higher quality)

Falls back to original order if neither is available.
"""
from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Sequence

from app.core.logging import get_logger

logger = get_logger(__name__)

MIN_RERANK_CANDIDATES = 20
MAX_RERANK_CANDIDATES = 50


@dataclass
class RerankedResult:
    chunk_id: str
    rerank_score: float
    original_score: float
    chunk_text: str
    metadata: dict


@dataclass
class RerankOutcome:
    results: list[RerankedResult]
    status: str
    error: str | None = None


class CrossEncoderReranker:
    """
    Local cross-encoder using sentence-transformers.
    Default model: ms-marco-MiniLM-L-6-v2 — fast and accurate for English.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model
        self._model: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is None:
            with self._load_lock:
                if self._model is not None:
                    return
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self._model_name)
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers required for local reranking. "
                        "Run: pip install sentence-transformers"
                    ) from exc

    def warmup(self) -> None:
        self._load()

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str, float, dict]],  # (id, text, score, meta)
        top_k: int = 5,
    ) -> RerankOutcome:
        if not candidates:
            return RerankOutcome(results=[], status="empty")
        try:
            self._load()
            logger.info(
                "Local rerank input",
                candidate_count=len(candidates),
                top_k=top_k,
                model=self._model_name,
            )
            pairs = [(query, text) for _, text, _, _ in candidates]
            predicted = self._model.predict(pairs)
            scores: list[float] = predicted.tolist() if hasattr(predicted, "tolist") else list(predicted)
            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            results = [
                RerankedResult(
                    chunk_id=cid,
                    rerank_score=float(score),
                    original_score=orig_score,
                    chunk_text=text,
                    metadata=meta,
                )
                for score, (cid, text, orig_score, meta) in ranked[:top_k]
            ]
            logger.info(
                "Local rerank output",
                result_count=len(results),
                top_candidate_ids=[item.chunk_id for item in results],
            )
            return RerankOutcome(
                results=results,
                status="ok",
            )
        except Exception as exc:
            logger.exception("Local rerank failed", error=str(exc))
            # Fail open: if the model cannot be loaded or run, keep original order.
            return RerankOutcome(
                results=[
                    RerankedResult(
                        chunk_id=cid,
                        rerank_score=orig_score,
                        original_score=orig_score,
                        chunk_text=text,
                        metadata=meta,
                    )
                    for cid, text, orig_score, meta in candidates[:top_k]
                ],
                status="fallback",
                error=str(exc),
            )


class CohereReranker:
    """
    Cohere Rerank API — cloud-based, language-agnostic, high quality.
    Requires COHERE_API_KEY in environment.
    """

    def __init__(self, model: str = "rerank-english-v3.0") -> None:
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import cohere
            import os

            self._client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY", ""))
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str, float, dict]],
        top_k: int = 5,
    ) -> RerankOutcome:
        if not candidates:
            return RerankOutcome(results=[], status="empty")
        try:
            co = self._get_client()
            logger.info(
                "Cohere rerank input",
                candidate_count=len(candidates),
                top_k=top_k,
                model=self._model,
            )
            docs = [text for _, text, _, _ in candidates]
            resp = await co.rerank(
                model=self._model,
                query=query,
                documents=docs,
                top_n=top_k,
            )
            results: list[RerankedResult] = []
            for r in resp.results:
                cid, text, orig_score, meta = candidates[r.index]
                results.append(
                    RerankedResult(
                        chunk_id=cid,
                        rerank_score=r.relevance_score,
                        original_score=orig_score,
                        chunk_text=text,
                        metadata=meta,
                    )
                )
            logger.info(
                "Cohere rerank output",
                result_count=len(results),
                top_candidate_ids=[item.chunk_id for item in results],
            )
            return RerankOutcome(results=results, status="ok")
        except Exception as exc:
            logger.exception("Cohere rerank failed", error=str(exc))
            # Graceful degradation — return original order
            return RerankOutcome(
                results=[
                    RerankedResult(
                        chunk_id=cid,
                        rerank_score=orig_score,
                        original_score=orig_score,
                        chunk_text=text,
                        metadata=meta,
                    )
                    for cid, text, orig_score, meta in candidates[:top_k]
                ],
                status="fallback",
                error=str(exc),
            )


@lru_cache(maxsize=1)
def get_local_reranker(model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    return CrossEncoderReranker(model=model)


@lru_cache(maxsize=1)
def get_cohere_reranker(model: str = "rerank-english-v3.0") -> CohereReranker:
    return CohereReranker(model=model)


def warmup_rerankers() -> None:
    """Load any local reranker models so the first request stays fast."""
    try:
        logger.info("Warming local reranker")
        get_local_reranker().warmup()
        logger.info("Local reranker warmed up")
    except Exception as exc:
        logger.warning("Local reranker warmup failed", error=str(exc))


def _limit_candidates(
    candidates: Sequence[Any],
    top_k: int,
) -> list[Any]:
    if not candidates:
        return []
    window = min(
        len(candidates),
        max(min(top_k, MAX_RERANK_CANDIDATES), MIN_RERANK_CANDIDATES),
    )
    return list(candidates[:window])


async def rerank_results(
    query: str,
    candidates: list[Any],  # DenseResult | HybridResult | SparseResult
    top_k: int = 5,
    backend: str = "local",
) -> RerankOutcome:
    """
    Rerank a list of retrieval results.

    Args:
        query:      The search query.
        candidates: Any retrieval result objects with .chunk_id, .chunk_text,
                    .metadata, and a score attribute.
        top_k:      Number of final results to return.
        backend:    'local' (CrossEncoder) or 'cohere' (Cohere API).

    Returns:
        Reranked list of RerankedResult.
    """
    def _score(r: Any) -> float:
        for attr in ("rrf_score", "score", "rerank_score"):
            if hasattr(r, attr):
                return float(getattr(r, attr))
        return 0.0

    limited_candidates = _limit_candidates(candidates, top_k)
    logger.info(
        "Rerank orchestration input",
        backend=backend,
        query=query,
        candidate_count=len(candidates),
        limited_count=len(limited_candidates),
        top_k=top_k,
    )
    tuples = [
        (r.chunk_id, r.chunk_text, _score(r), r.metadata)
        for r in limited_candidates
    ]

    if backend == "cohere":
        try:
            reranker = get_cohere_reranker()
            return await reranker.rerank(query, tuples, top_k=top_k)
        except Exception as exc:
            logger.exception("Cohere rerank orchestration failed", error=str(exc))
            return RerankOutcome(
                results=[
                    RerankedResult(
                        chunk_id=cid,
                        rerank_score=orig_score,
                        original_score=orig_score,
                        chunk_text=text,
                        metadata=meta,
                    )
                    for cid, text, orig_score, meta in tuples[:top_k]
                ],
                status="fallback",
                error=str(exc),
            )

    # Default: local cross-encoder (sync, run in executor)
    try:
        reranker_local = get_local_reranker()
        return await asyncio.to_thread(reranker_local.rerank, query, tuples, top_k=top_k)
    except Exception as exc:
        logger.exception("Local rerank orchestration failed", error=str(exc))
        return RerankOutcome(
            results=[
                RerankedResult(
                    chunk_id=cid,
                    rerank_score=orig_score,
                    original_score=orig_score,
                    chunk_text=text,
                    metadata=meta,
                )
                for cid, text, orig_score, meta in tuples[:top_k]
            ],
            status="fallback",
            error=str(exc),
        )
