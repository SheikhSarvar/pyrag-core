"""
Weaviate v4 adapter.
Install extra: pip install pyrag-core[weaviate]
"""
from __future__ import annotations

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.services.vector.base import (
    SearchQuery,
    SearchResult,
    VectorPoint,
    VectorStore,
)

logger = get_logger(__name__)


class WeaviateAdapter(VectorStore):
    """Weaviate v4 vector store adapter."""

    def __init__(self, client: "weaviate.WeaviateClient") -> None:  # type: ignore[name-defined]  # noqa: F821
        self._client = client

    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        try:
            import weaviate.classes.config as wc
            if self._client.collections.exists(collection_name):
                return
            distance_map = {
                "cosine": wc.VectorDistances.COSINE,
                "dot": wc.VectorDistances.DOT,
                "euclidean": wc.VectorDistances.L2_SQUARED,
            }
            self._client.collections.create(
                name=collection_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=distance_map.get(distance, wc.VectorDistances.COSINE)
                ),
            )
            logger.info("Created Weaviate collection", collection=collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Weaviate create_collection failed: {exc}") from exc

    async def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.collections.delete(collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Weaviate delete_collection failed: {exc}") from exc

    async def collection_exists(self, collection_name: str) -> bool:
        return bool(self._client.collections.exists(collection_name))

    async def collection_info(self, collection_name: str) -> dict:
        col = self._client.collections.get(collection_name)
        agg = col.aggregate.over_all(total_count=True)
        return {"name": collection_name, "count": agg.total_count}

    async def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        try:
            col = self._client.collections.get(collection_name)
            with col.batch.dynamic() as batch:
                for p in points:
                    batch.add_object(properties=p.payload, uuid=p.id, vector=p.vector)
        except Exception as exc:
            raise VectorStoreError(f"Weaviate upsert failed: {exc}") from exc

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        try:
            col = self._client.collections.get(collection_name)
            for id_ in ids:
                col.data.delete_by_id(id_)
        except Exception as exc:
            raise VectorStoreError(f"Weaviate delete failed: {exc}") from exc

    async def delete_by_filter(self, collection_name: str, filters: dict) -> None:
        try:
            import weaviate.classes.query as wq
            col = self._client.collections.get(collection_name)
            for key, value in filters.items():
                col.data.delete_many(where=wq.Filter.by_property(key).equal(value))
        except Exception as exc:
            raise VectorStoreError(f"Weaviate delete_by_filter failed: {exc}") from exc

    async def search(self, collection_name: str, query: SearchQuery) -> list[SearchResult]:
        try:
            col = self._client.collections.get(collection_name)
            results = col.query.near_vector(
                near_vector=query.vector,
                limit=query.top_k,
                return_metadata=["certainty"],
            )
            return [
                SearchResult(id=str(o.uuid), score=o.metadata.certainty or 0.0, payload=o.properties)
                for o in results.objects
            ]
        except Exception as exc:
            raise VectorStoreError(f"Weaviate search failed: {exc}") from exc

    async def get(self, collection_name: str, ids: list[str]) -> list[VectorPoint]:
        try:
            col = self._client.collections.get(collection_name)
            return [
                VectorPoint(id=str(obj.uuid), vector=[], payload=obj.properties)
                for id_ in ids
                if (obj := col.query.fetch_object_by_id(id_)) is not None
            ]
        except Exception as exc:
            raise VectorStoreError(f"Weaviate get failed: {exc}") from exc

    async def count(self, collection_name: str, filters: dict | None = None) -> int:
        col = self._client.collections.get(collection_name)
        agg = col.aggregate.over_all(total_count=True)
        return agg.total_count or 0
