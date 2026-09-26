from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.vector.base import SearchQuery
from app.services.vector.qdrant_adapter import QdrantAdapter


class _QueryPointsOnlyClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(id="chunk-1", score=0.91, payload={"chunk_text": "hello"}),
                SimpleNamespace(id="chunk-2", score=0.81, payload={"chunk_text": "world"}),
            ]
        )


@pytest.mark.asyncio
async def test_qdrant_search_uses_query_points_when_search_missing() -> None:
    client = _QueryPointsOnlyClient()
    adapter = QdrantAdapter(client)  # type: ignore[arg-type]

    results = await adapter.search(
        "test-collection",
        SearchQuery(vector=[0.1, 0.2, 0.3], top_k=2, filters={"dataset_id": "ds-1"}),
    )

    assert len(results) == 2
    assert results[0].id == "chunk-1"
    assert client.calls[0]["collection_name"] == "test-collection"
    assert client.calls[0]["limit"] == 2
    assert client.calls[0]["query"] == [0.1, 0.2, 0.3]

