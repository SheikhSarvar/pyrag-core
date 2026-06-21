"""
Hybrid retrieval fusion — T26.
Combines dense (vector) and sparse (BM25) results using
Reciprocal Rank Fusion (RRF).

RRF score = Σ 1/(k + rank_i)   where k=60 is the standard constant.
This is rank-order fusion — no score normalisation needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.retrieval.dense import DenseResult, dense_search
from app.services.retrieval.sparse import SparseResult, sparse_search


@dataclass
class HybridResult:
    chunk_id: str
    rrf_score: float
    dense_score: float | None
    sparse_score: float | None
    chunk_text: str
    metadata: dict


def _rrf_score(ranks: list[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


def fuse_results(
    dense_results: list[DenseResult],
    sparse_results: list[SparseResult],
    top_k: int = 10,
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[HybridResult]:
    """
    Merge dense and sparse result lists using weighted RRF.

    Args:
        dense_results:  Ordered list of dense search results.
        sparse_results: Ordered list of sparse search results.
        top_k:          Number of final results to return.
        k:              RRF constant (60 is empirically optimal).
        dense_weight:   Weight multiplier for dense RRF scores.
        sparse_weight:  Weight multiplier for sparse RRF scores.

    Returns:
        Fused list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    dense_score_map: dict[str, float] = {}
    sparse_score_map: dict[str, float] = {}
    text_map: dict[str, str] = {}
    meta_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, 1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (k + rank))
        dense_score_map[cid] = result.score
        text_map[cid] = result.chunk_text
        meta_map[cid] = result.metadata

    for rank, result in enumerate(sparse_results, 1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (k + rank))
        sparse_score_map[cid] = result.score
        if cid not in text_map:
            text_map[cid] = result.chunk_text
            meta_map[cid] = result.metadata

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        HybridResult(
            chunk_id=cid,
            rrf_score=score,
            dense_score=dense_score_map.get(cid),
            sparse_score=sparse_score_map.get(cid),
            chunk_text=text_map.get(cid, ""),
            metadata=meta_map.get(cid, {}),
        )
        for cid, score in ranked[:top_k]
    ]


async def hybrid_search(
    dataset_id: str,
    query: str,
    top_k: int = 10,
    candidate_k: int = 30,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    session=None,
) -> list[HybridResult]:
    """
    Run dense + sparse in parallel then fuse with RRF.

    Args:
        dataset_id:    Target dataset.
        query:         Query string.
        top_k:         Final results after fusion.
        candidate_k:   Candidates retrieved from each source before fusion.
        dense_weight:  RRF weight for dense results (higher = more semantic).
        sparse_weight: RRF weight for sparse results (higher = more keyword-exact).
        session:       AsyncSession for sparse BM25 chunk fetch.
    """
    import asyncio

    dense_task = asyncio.create_task(
        dense_search(dataset_id, query, top_k=candidate_k)
    )
    sparse_task = asyncio.create_task(
        sparse_search(dataset_id, query, top_k=candidate_k, session=session)
    )
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    return fuse_results(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
