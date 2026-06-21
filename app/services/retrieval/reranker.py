"""
Reranker — T27.
Re-scores retrieved candidates with a cross-encoder model.
Two backends:
  1. sentence-transformers CrossEncoder (local, no API cost)
  2. Cohere Rerank API (cloud, higher quality)

Falls back to original order if neither is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankedResult:
    chunk_id: str
    rerank_score: float
    original_score: float
    chunk_text: str
    metadata: dict


class CrossEncoderReranker:
    """
    Local cross-encoder using sentence-transformers.
    Default model: ms-marco-MiniLM-L-6-v2 — fast and accurate for English.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers required for local reranking. "
                    "Run: pip install sentence-transformers"
                ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float, dict]],  # (id, text, score, meta)
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not candidates:
            return []
        self._load()
        pairs = [(query, text) for _, text, _, _ in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            RerankedResult(
                chunk_id=cid,
                rerank_score=float(score),
                original_score=orig_score,
                chunk_text=text,
                metadata=meta,
            )
            for score, (cid, text, orig_score, meta) in ranked[:top_k]
        ]


class CohereReranker:
    """
    Cohere Rerank API — cloud-based, language-agnostic, high quality.
    Requires COHERE_API_KEY in environment.
    """

    def __init__(self, model: str = "rerank-english-v3.0") -> None:
        self._model = model

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float, dict]],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not candidates:
            return []
        try:
            import cohere
            import os
            co = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY", ""))
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
            return results
        except Exception:
            # Graceful degradation — return original order
            return [
                RerankedResult(
                    chunk_id=cid,
                    rerank_score=orig_score,
                    original_score=orig_score,
                    chunk_text=text,
                    metadata=meta,
                )
                for cid, text, orig_score, meta in candidates[:top_k]
            ]


async def rerank_results(
    query: str,
    candidates: list[Any],  # DenseResult | HybridResult | SparseResult
    top_k: int = 5,
    backend: str = "local",
) -> list[RerankedResult]:
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

    tuples = [
        (r.chunk_id, r.chunk_text, _score(r), r.metadata)
        for r in candidates
    ]

    if backend == "cohere":
        reranker = CohereReranker()
        return await reranker.rerank(query, tuples, top_k=top_k)

    # Default: local cross-encoder (sync, run in executor)
    import asyncio
    reranker_local = CrossEncoderReranker()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: reranker_local.rerank(query, tuples, top_k=top_k),
    )
