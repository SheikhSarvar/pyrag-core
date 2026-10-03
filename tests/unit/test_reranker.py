from __future__ import annotations

import sys
import types

import pytest

from app.services.retrieval.reranker import (
    CohereReranker,
    CrossEncoderReranker,
    RerankOutcome,
    RerankedResult,
    rerank_results,
)


class _Candidate:
    def __init__(self, chunk_id: str, score: float) -> None:
        self.chunk_id = chunk_id
        self.chunk_text = f"text {chunk_id}"
        self.metadata = {"idx": chunk_id}
        self.score = score


def test_local_reranker_returns_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.2, 0.9, 0.5]

    monkeypatch.setattr(CrossEncoderReranker, "_load", lambda self: setattr(self, "_model", _Model()))

    outcome = CrossEncoderReranker().rerank(
        "query",
        [
            ("a", "text a", 0.1, {}),
            ("b", "text b", 0.2, {}),
            ("c", "text c", 0.3, {}),
        ],
        top_k=2,
    )

    assert isinstance(outcome, RerankOutcome)
    assert outcome.status == "ok"
    assert [item.chunk_id for item in outcome.results] == ["b", "c"]


@pytest.mark.asyncio
async def test_cohere_reranker_returns_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        def __init__(self) -> None:
            self.results = [
                types.SimpleNamespace(index=1, relevance_score=0.9),
                types.SimpleNamespace(index=0, relevance_score=0.7),
            ]

    class _Client:
        async def rerank(self, **kwargs):
            return _Response()

    fake_module = types.SimpleNamespace(AsyncClient=lambda api_key: _Client())
    monkeypatch.setitem(sys.modules, "cohere", fake_module)

    outcome = await CohereReranker().rerank(
        "query",
        [
            ("a", "text a", 0.1, {}),
            ("b", "text b", 0.2, {}),
        ],
        top_k=2,
    )

    assert isinstance(outcome, RerankOutcome)
    assert outcome.status == "ok"
    assert [item.chunk_id for item in outcome.results] == ["b", "a"]


@pytest.mark.asyncio
async def test_rerank_results_caps_candidate_window(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    class _FakeReranker:
        def rerank(self, query: str, candidates, top_k: int = 5) -> RerankOutcome:
            seen["count"] = len(candidates)
            return RerankOutcome(
                results=[
                    RerankedResult(
                        chunk_id=cid,
                        rerank_score=score,
                        original_score=score,
                        chunk_text=text,
                        metadata=meta,
                    )
                    for cid, text, score, meta in candidates[:top_k]
                ],
                status="ok",
            )

    monkeypatch.setattr("app.services.retrieval.reranker.get_local_reranker", lambda: _FakeReranker())

    outcome = await rerank_results(
        "query",
        [_Candidate(str(i), float(100 - i)) for i in range(100)],
        top_k=5,
        backend="local",
    )

    assert seen["count"] == 20
    assert outcome.status == "ok"

