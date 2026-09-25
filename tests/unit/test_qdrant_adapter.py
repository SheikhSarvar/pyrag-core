from types import SimpleNamespace

import pytest

from app.services.vector.base import SearchQuery
from app.services.vector import qdrant_adapter as qdrant_module
from app.services.vector.qdrant_adapter import QdrantAdapter


class QueryPointsOnlyClient:
    def __init__(self) -> None:
        self.captured = None

    async def query_points(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="chunk-1",
                    score=0.87,
                    payload={"chunk_text": "hello world"},
                )
            ]
        )


class FailingPayloadIndexClient(QueryPointsOnlyClient):
    async def create_payload_index(self, **kwargs):
        raise RuntimeError("payload index failed")


@pytest.mark.asyncio
async def test_search_uses_query_points_when_search_is_missing() -> None:
    client = QueryPointsOnlyClient()
    adapter = QdrantAdapter(client)  # type: ignore[arg-type]

    results = await adapter.search(
        "dataset-1",
        SearchQuery(vector=[0.1, 0.2, 0.3], top_k=5, filters={"dataset_id": "dataset-1"}),
    )

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].score == 0.87
    assert results[0].payload["chunk_text"] == "hello world"
    assert client.captured is not None
    assert client.captured["collection_name"] == "dataset-1"
    assert client.captured["limit"] == 5


@pytest.mark.asyncio
async def test_payload_index_failure_does_not_cache_collection() -> None:
    qdrant_module._indexed_collections.discard("dataset-1")
    client = FailingPayloadIndexClient()
    adapter = QdrantAdapter(client)  # type: ignore[arg-type]

    await adapter._ensure_payload_indexes("dataset-1")

    assert "dataset-1" not in qdrant_module._indexed_collections


def test_build_filter_supports_multi_value_filters() -> None:
    flt = QdrantAdapter._build_filter({"chunk_id": ["chunk-1", "chunk-2"], "dataset_id": "ds-1"})

    assert flt is not None
    assert len(flt.must or []) == 2
    assert any(cond.key == "chunk_id" for cond in flt.must or [])
    assert any(cond.key == "dataset_id" for cond in flt.must or [])
