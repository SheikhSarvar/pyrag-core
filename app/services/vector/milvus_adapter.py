"""
Milvus adapter.
Install extra: pip install pyrag-core[milvus]
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

_METRIC_MAP = {"cosine": "COSINE", "dot": "IP", "euclidean": "L2"}


class MilvusAdapter(VectorStore):
    """Milvus 2.x vector store adapter (uses pymilvus)."""

    def __init__(self, uri: str = "http://localhost:19530") -> None:
        self._uri = uri

    def _col(self, name: str) -> "MilvusClient":  # type: ignore[name-defined]  # noqa: F821
        from pymilvus import MilvusClient
        return MilvusClient(uri=self._uri)

    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        try:
            from pymilvus import MilvusClient, DataType
            client = MilvusClient(uri=self._uri)
            if client.has_collection(collection_name):
                return
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, max_length=255, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimensions)
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type=_METRIC_MAP.get(distance, "COSINE"),
                params={"M": 16, "efConstruction": 100},
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("Created Milvus collection", collection=collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Milvus create_collection failed: {exc}") from exc

    async def delete_collection(self, collection_name: str) -> None:
        try:
            from pymilvus import MilvusClient
            MilvusClient(uri=self._uri).drop_collection(collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Milvus delete_collection failed: {exc}") from exc

    async def collection_exists(self, collection_name: str) -> bool:
        from pymilvus import MilvusClient
        return bool(MilvusClient(uri=self._uri).has_collection(collection_name))

    async def collection_info(self, collection_name: str) -> dict:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=self._uri)
        stats = client.get_collection_stats(collection_name)
        return {"name": collection_name, "count": int(stats.get("row_count", 0))}

    async def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self._uri)
            data = [{"id": p.id, "vector": p.vector, **p.payload} for p in points]
            client.upsert(collection_name=collection_name, data=data)
        except Exception as exc:
            raise VectorStoreError(f"Milvus upsert failed: {exc}") from exc

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        try:
            from pymilvus import MilvusClient
            MilvusClient(uri=self._uri).delete(collection_name=collection_name, ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Milvus delete failed: {exc}") from exc

    async def delete_by_filter(self, collection_name: str, filters: dict) -> None:
        try:
            from pymilvus import MilvusClient
            expr = " && ".join(f'{k} == "{v}"' for k, v in filters.items())
            MilvusClient(uri=self._uri).delete(collection_name=collection_name, filter=expr)
        except Exception as exc:
            raise VectorStoreError(f"Milvus delete_by_filter failed: {exc}") from exc

    async def search(self, collection_name: str, query: SearchQuery) -> list[SearchResult]:
        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=self._uri)
            results = client.search(
                collection_name=collection_name,
                data=[query.vector],
                limit=query.top_k,
                output_fields=["*"],
            )
            return [
                SearchResult(id=str(r["id"]), score=float(r["distance"]), payload=r.get("entity", {}))
                for r in results[0]
            ]
        except Exception as exc:
            raise VectorStoreError(f"Milvus search failed: {exc}") from exc

    async def get(self, collection_name: str, ids: list[str]) -> list[VectorPoint]:
        try:
            from pymilvus import MilvusClient
            records = MilvusClient(uri=self._uri).get(collection_name=collection_name, ids=ids, output_fields=["*"])
            return [VectorPoint(id=str(r["id"]), vector=r.get("vector", []), payload=r) for r in records]
        except Exception as exc:
            raise VectorStoreError(f"Milvus get failed: {exc}") from exc

    async def count(self, collection_name: str, filters: dict | None = None) -> int:
        from pymilvus import MilvusClient
        stats = MilvusClient(uri=self._uri).get_collection_stats(collection_name)
        return int(stats.get("row_count", 0))
