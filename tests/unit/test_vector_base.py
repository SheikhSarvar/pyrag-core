"""
Tests for VectorStore interface contract.
Uses an in-memory stub — no real Qdrant/Weaviate needed.
"""
import pytest

from app.services.vector.base import SearchQuery, SearchResult, VectorPoint, VectorStore


class InMemoryVectorStore(VectorStore):
    """Minimal in-memory implementation for testing the interface contract."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorPoint]] = {}

    async def create_collection(self, collection_name, dimensions, distance="cosine"):
        self._collections.setdefault(collection_name, {})

    async def delete_collection(self, collection_name):
        self._collections.pop(collection_name, None)

    async def collection_exists(self, collection_name):
        return collection_name in self._collections

    async def collection_info(self, collection_name):
        return {"name": collection_name, "count": len(self._collections.get(collection_name, {}))}

    async def upsert(self, collection_name, points):
        col = self._collections.setdefault(collection_name, {})
        for p in points:
            col[p.id] = p

    async def delete(self, collection_name, ids):
        col = self._collections.get(collection_name, {})
        for id_ in ids:
            col.pop(id_, None)

    async def delete_by_filter(self, collection_name, filters):
        col = self._collections.get(collection_name, {})
        to_delete = [
            id_ for id_, p in col.items()
            if all(p.payload.get(k) == v for k, v in filters.items())
        ]
        for id_ in to_delete:
            col.pop(id_)

    async def search(self, collection_name, query):
        col = self._collections.get(collection_name, {})
        # Fake score: 1.0 for all results
        results = [SearchResult(id=p.id, score=1.0, payload=p.payload) for p in col.values()]
        return results[: query.top_k]

    async def get(self, collection_name, ids):
        col = self._collections.get(collection_name, {})
        return [col[id_] for id_ in ids if id_ in col]

    async def count(self, collection_name, filters=None):
        return len(self._collections.get(collection_name, {}))


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_create_and_check_collection(store: InMemoryVectorStore) -> None:
    assert not await store.collection_exists("test")
    await store.create_collection("test", dimensions=4)
    assert await store.collection_exists("test")


@pytest.mark.asyncio
async def test_upsert_and_count(store: InMemoryVectorStore) -> None:
    await store.create_collection("col", dimensions=4)
    points = [
        VectorPoint(id="a", vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "hello"}),
        VectorPoint(id="b", vector=[0.5, 0.6, 0.7, 0.8], payload={"text": "world"}),
    ]
    await store.upsert("col", points)
    assert await store.count("col") == 2


@pytest.mark.asyncio
async def test_upsert_is_idempotent(store: InMemoryVectorStore) -> None:
    await store.create_collection("col", dimensions=4)
    p = VectorPoint(id="x", vector=[1.0, 0.0, 0.0, 0.0], payload={"v": 1})
    await store.upsert("col", [p])
    p2 = VectorPoint(id="x", vector=[1.0, 0.0, 0.0, 0.0], payload={"v": 2})
    await store.upsert("col", [p2])
    assert await store.count("col") == 1
    fetched = await store.get("col", ["x"])
    assert fetched[0].payload["v"] == 2


@pytest.mark.asyncio
async def test_delete_by_id(store: InMemoryVectorStore) -> None:
    await store.create_collection("col", dimensions=4)
    await store.upsert("col", [VectorPoint(id="del", vector=[0.0] * 4, payload={})])
    await store.delete("col", ["del"])
    assert await store.count("col") == 0


@pytest.mark.asyncio
async def test_delete_by_filter(store: InMemoryVectorStore) -> None:
    await store.create_collection("col", dimensions=4)
    await store.upsert("col", [
        VectorPoint(id="1", vector=[0.0] * 4, payload={"doc": "abc"}),
        VectorPoint(id="2", vector=[0.0] * 4, payload={"doc": "xyz"}),
    ])
    await store.delete_by_filter("col", {"doc": "abc"})
    assert await store.count("col") == 1


@pytest.mark.asyncio
async def test_search_returns_top_k(store: InMemoryVectorStore) -> None:
    await store.create_collection("col", dimensions=4)
    for i in range(10):
        await store.upsert("col", [VectorPoint(id=str(i), vector=[float(i)] * 4, payload={})])
    results = await store.search("col", SearchQuery(vector=[0.0] * 4, top_k=3))
    assert len(results) == 3


@pytest.mark.asyncio
async def test_delete_collection(store: InMemoryVectorStore) -> None:
    await store.create_collection("to_drop", dimensions=4)
    await store.delete_collection("to_drop")
    assert not await store.collection_exists("to_drop")


@pytest.mark.asyncio
async def test_collection_info(store: InMemoryVectorStore) -> None:
    await store.create_collection("info_col", dimensions=8)
    await store.upsert("info_col", [VectorPoint(id="z", vector=[0.0] * 8, payload={})])
    info = await store.collection_info("info_col")
    assert info["count"] == 1
    assert info["name"] == "info_col"
