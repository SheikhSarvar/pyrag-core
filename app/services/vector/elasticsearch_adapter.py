"""
Elasticsearch adapter using dense_vector + kNN search (ES 8+).
Install extra: pip install pyrag-core[elasticsearch]
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


class ElasticsearchAdapter(VectorStore):

    def __init__(self, hosts: list[str] | None = None, **kwargs) -> None:  # type: ignore[type-arg]
        self._hosts = hosts or ["http://localhost:9200"]
        self._kwargs = kwargs

    def _client(self) -> "Elasticsearch":  # type: ignore[name-defined]  # noqa: F821
        from elasticsearch import Elasticsearch
        return Elasticsearch(self._hosts, **self._kwargs)

    def _index(self, collection_name: str) -> str:
        return collection_name.lower().replace(" ", "_")

    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        index = self._index(collection_name)
        similarity_map = {"cosine": "cosine", "dot": "dot_product", "euclidean": "l2_norm"}
        try:
            es = self._client()
            if es.indices.exists(index=index):
                return
            es.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "id": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dimensions,
                            "index": True,
                            "similarity": similarity_map.get(distance, "cosine"),
                        },
                        "payload": {"type": "object", "dynamic": True},
                    }
                },
            )
            logger.info("Created ES index", index=index)
        except Exception as exc:
            raise VectorStoreError(f"ES create_collection failed: {exc}") from exc

    async def delete_collection(self, collection_name: str) -> None:
        es = self._client()
        es.indices.delete(index=self._index(collection_name), ignore_unavailable=True)

    async def collection_exists(self, collection_name: str) -> bool:
        return bool(self._client().indices.exists(index=self._index(collection_name)))

    async def collection_info(self, collection_name: str) -> dict:
        index = self._index(collection_name)
        es = self._client()
        stats = es.indices.stats(index=index)
        count = stats["indices"][index]["primaries"]["docs"]["count"]
        return {"name": collection_name, "count": count}

    async def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        from elasticsearch.helpers import bulk
        index = self._index(collection_name)
        actions = [
            {
                "_index": index,
                "_id": p.id,
                "_source": {"id": p.id, "embedding": p.vector, "payload": p.payload},
            }
            for p in points
        ]
        try:
            bulk(self._client(), actions)
        except Exception as exc:
            raise VectorStoreError(f"ES upsert failed: {exc}") from exc

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        from elasticsearch.helpers import bulk
        index = self._index(collection_name)
        actions = [{"_op_type": "delete", "_index": index, "_id": id_} for id_ in ids]
        bulk(self._client(), actions, ignore_status=404)

    async def delete_by_filter(self, collection_name: str, filters: dict) -> None:
        index = self._index(collection_name)
        must = [{"term": {f"payload.{k}": v}} for k, v in filters.items()]
        self._client().delete_by_query(index=index, body={"query": {"bool": {"must": must}}})

    async def search(self, collection_name: str, query: SearchQuery) -> list[SearchResult]:
        try:
            index = self._index(collection_name)
            body: dict = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query.vector,
                    "k": query.top_k,
                    "num_candidates": query.top_k * 5,
                }
            }
            resp = self._client().search(index=index, body=body, size=query.top_k)
            return [
                SearchResult(
                    id=hit["_source"]["id"],
                    score=hit["_score"],
                    payload=hit["_source"].get("payload", {}),
                )
                for hit in resp["hits"]["hits"]
            ]
        except Exception as exc:
            raise VectorStoreError(f"ES search failed: {exc}") from exc

    async def get(self, collection_name: str, ids: list[str]) -> list[VectorPoint]:
        index = self._index(collection_name)
        resp = self._client().mget(index=index, body={"ids": ids})
        return [
            VectorPoint(id=doc["_id"], vector=[], payload=doc["_source"].get("payload", {}))
            for doc in resp["docs"]
            if doc.get("found")
        ]

    async def count(self, collection_name: str, filters: dict | None = None) -> int:
        index = self._index(collection_name)
        resp = self._client().count(index=index)
        return resp["count"]
