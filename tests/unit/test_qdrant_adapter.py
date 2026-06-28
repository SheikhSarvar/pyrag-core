from types import SimpleNamespace

import pytest

from app.services.vector.base import SearchQuery
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
