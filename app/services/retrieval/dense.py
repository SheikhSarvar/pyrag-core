"""
Dense retrieval — T24.
Embeds the query and runs a nearest-neighbour search against the vector store.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DenseResult:
    chunk_id: str
    score: float
    chunk_text: str
    metadata: dict


async def dense_search(
    dataset_id: str,
    query: str,
    top_k: int = 20,
    score_threshold: float | None = None,
    filters: dict | None = None,
) -> list[DenseResult]:
    """
    Embed `query` and retrieve the top-k nearest vectors for `dataset_id`.

    Args:
        dataset_id:       Target dataset.
        query:            Raw query string (will be embedded).
        top_k:            Number of candidates to retrieve.
        score_threshold:  Minimum cosine similarity to include a result.
        filters:          Additional payload filters forwarded to the vector store.

    Returns:
        List of DenseResult sorted by descending score.
    """
    from app.services.embedding.providers import get_embedding_provider
    from app.services.vector.factory import get_vector_store
    from app.services.vector.base import SearchQuery
    from app.services.ingestion.indexer import collection_name

    embedder = get_embedding_provider()
    vector_store = get_vector_store()

    query_vector = await embedder.embed_query(query)

    col = collection_name(dataset_id)
    if not await vector_store.collection_exists(col):
        return []

    search_filters = {"dataset_id": dataset_id}
    if filters:
        search_filters.update(filters)

    results = await vector_store.search(
        col,
        SearchQuery(
            vector=query_vector,
            top_k=top_k,
            filters=search_filters,
            score_threshold=score_threshold,
        ),
    )

    return [
        DenseResult(
            chunk_id=r.id,
            score=r.score,
            chunk_text=r.payload.get("chunk_text", ""),
            metadata=r.payload,
        )
        for r in results
    ]
