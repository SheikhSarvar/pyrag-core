from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.services.vector.base import (
    SearchQuery,
    SearchResult,
    VectorPoint,
    VectorStore,
)

logger = get_logger(__name__)

_DISTANCE_MAP = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclidean": qmodels.Distance.EUCLID,
}


class QdrantAdapter(VectorStore):

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    # ── Collection management ─────────────────────────────────────────────────

    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        try:
            if await self.collection_exists(collection_name):
                logger.debug("Collection already exists", collection=collection_name)
                return
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dimensions,
                    distance=_DISTANCE_MAP.get(distance, qmodels.Distance.COSINE),
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )
            logger.info("Created collection", collection=collection_name, dimensions=dimensions)
        except Exception as exc:
            raise VectorStoreError(f"Failed to create collection: {exc}") from exc

    async def delete_collection(self, collection_name: str) -> None:
        try:
            await self._client.delete_collection(collection_name)
            logger.info("Deleted collection", collection=collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete collection: {exc}") from exc

    async def collection_exists(self, collection_name: str) -> bool:
        try:
            await self._client.get_collection(collection_name)
            return True
        except (UnexpectedResponse, Exception):
            return False

    async def collection_info(self, collection_name: str) -> dict:
        try:
            info = await self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": info.points_count,
                "status": str(info.status),
                "dimensions": info.config.params.vectors.size if info.config.params.vectors else None,  # type: ignore[union-attr]
            }
        except Exception as exc:
            raise VectorStoreError(f"Failed to get collection info: {exc}") from exc

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert(
        self, collection_name: str, points: list[VectorPoint]
    ) -> None:
        if not points:
            return
        try:
            qdrant_points = [
                qmodels.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload,
                )
                for p in points
            ]
            await self._client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=True,
            )
            logger.debug("Upserted vectors", collection=collection_name, count=len(points))
        except Exception as exc:
            raise VectorStoreError(f"Upsert failed: {exc}") from exc

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=qmodels.PointIdsList(points=ids),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc

    async def delete_by_filter(
        self, collection_name: str, filters: dict
    ) -> None:
        try:
            conditions = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v),
                )
                for k, v in filters.items()
            ]
            await self._client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(must=conditions)
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Delete by filter failed: {exc}") from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self, collection_name: str, query: SearchQuery
    ) -> list[SearchResult]:
        try:
            qdrant_filter = None
            if query.filters:
                qdrant_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key=k,
                            match=qmodels.MatchValue(value=v),
                        )
                        for k, v in query.filters.items()
                    ]
                )
            results = await self._client.search(
                collection_name=collection_name,
                query_vector=query.vector,
                limit=query.top_k,
                query_filter=qdrant_filter,
                score_threshold=query.score_threshold,
                with_payload=True,
            )
            return [
                SearchResult(id=str(r.id), score=r.score, payload=r.payload or {})
                for r in results
            ]
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

    async def get(
        self, collection_name: str, ids: list[str]
    ) -> list[VectorPoint]:
        try:
            records = await self._client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_vectors=True,
                with_payload=True,
            )
            return [
                VectorPoint(
                    id=str(r.id),
                    vector=r.vector if isinstance(r.vector, list) else [],  # type: ignore[arg-type]
                    payload=r.payload or {},
                )
                for r in records
            ]
        except Exception as exc:
            raise VectorStoreError(f"Get failed: {exc}") from exc

    async def count(
        self, collection_name: str, filters: dict | None = None
    ) -> int:
        try:
            qdrant_filter = None
            if filters:
                qdrant_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                        for k, v in filters.items()
                    ]
                )
            result = await self._client.count(
                collection_name=collection_name,
                count_filter=qdrant_filter,
                exact=True,
            )
            return result.count
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
