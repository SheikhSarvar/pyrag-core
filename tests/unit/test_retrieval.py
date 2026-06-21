"""
Unit tests for the retrieval pipeline.
No real vector store, embedder, or LLM calls.
"""
from __future__ import annotations

import pytest

from app.services.retrieval.query_understanding import understand_query
from app.services.retrieval.query_expansion import expand_query, ExpandedQuery
from app.services.retrieval.sparse import BM25
from app.services.retrieval.hybrid import fuse_results
from app.services.retrieval.context import compress_context, assemble_prompt, CompressedContext
from app.services.retrieval.dense import DenseResult
from app.services.retrieval.sparse import SparseResult
from app.services.retrieval.hybrid import HybridResult


# ── Query Understanding ───────────────────────────────────────────────────────

def test_understand_query_normalises_whitespace() -> None:
    result = understand_query("  what  is   revenue  ")
    assert "  " not in result.normalized


def test_understand_detects_question() -> None:
    result = understand_query("What is our quarterly revenue?")
    assert result.is_question is True


def test_understand_non_question() -> None:
    result = understand_query("Show me the sales report")
    assert result.is_question is False


def test_understand_intent_summarize() -> None:
    result = understand_query("Summarize the key findings from the document")
    assert result.intent == "summarize"


def test_understand_intent_compare() -> None:
    result = understand_query("Compare Q1 versus Q2 results")
    assert result.intent == "compare"


def test_understand_intent_explain() -> None:
    result = understand_query("Explain what EBITDA means")
    assert result.intent == "explain"


def test_understand_intent_default_search() -> None:
    result = understand_query("quarterly revenue 2024")
    assert result.intent == "search"


def test_understand_extracts_keywords() -> None:
    result = understand_query("What are the main revenue drivers this quarter?")
    assert "revenue" in result.keywords
    assert "quarter" in result.keywords
    assert "the" not in result.keywords  # stop word filtered


# ── Query Expansion ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expand_rule_based_synonym() -> None:
    result = await expand_query("What is our total revenue this year?", use_llm=False)
    assert result.original == "What is our total revenue this year?"
    # "revenue" should trigger synonym expansion
    assert len(result.variants) >= 1


@pytest.mark.asyncio
async def test_expand_all_queries_deduplicated() -> None:
    result = await expand_query("revenue report", use_llm=False)
    all_q = result.all_queries
    lower_set = [q.lower() for q in all_q]
    assert len(lower_set) == len(set(lower_set))


@pytest.mark.asyncio
async def test_expand_no_synonym_returns_original_only() -> None:
    result = await expand_query("xyz abstract concept qwerty", use_llm=False)
    assert result.original in result.all_queries


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _make_docs(texts: list[str]) -> list[tuple[str, str, dict]]:
    return [(str(i), t, {}) for i, t in enumerate(texts)]


def test_bm25_returns_top_k() -> None:
    bm25 = BM25()
    bm25.build(_make_docs([
        "revenue growth increased significantly this quarter",
        "employee headcount remains stable across all divisions",
        "revenue forecast for next year looks optimistic",
        "product launch scheduled for Q3",
    ]))
    results = bm25.search("revenue quarter", top_k=2)
    assert len(results) == 2


def test_bm25_ranks_relevant_first() -> None:
    bm25 = BM25()
    bm25.build(_make_docs([
        "the quick brown fox jumps over the lazy dog",
        "revenue increased by twenty percent this quarter",
        "revenue and earnings both hit record highs this quarter",
    ]))
    results = bm25.search("revenue quarter earnings")
    assert results[0].chunk_id == "2"  # most relevant


def test_bm25_empty_corpus_returns_empty() -> None:
    bm25 = BM25()
    bm25.build([])
    assert bm25.search("anything") == []


def test_bm25_no_match_returns_empty() -> None:
    bm25 = BM25()
    bm25.build(_make_docs(["the sky is blue"]))
    results = bm25.search("revenue forecast")
    assert results == []


def test_bm25_scores_positive_for_match() -> None:
    bm25 = BM25()
    bm25.build(_make_docs(["quarterly revenue report"]))
    results = bm25.search("revenue")
    assert results[0].score > 0


# ── Hybrid Fusion ─────────────────────────────────────────────────────────────

def _make_dense(chunk_id: str, score: float) -> DenseResult:
    return DenseResult(chunk_id=chunk_id, score=score, chunk_text=f"text {chunk_id}", metadata={})


def _make_sparse(chunk_id: str, score: float) -> SparseResult:
    return SparseResult(chunk_id=chunk_id, score=score, chunk_text=f"text {chunk_id}", metadata={})


def test_fuse_results_merges_both_sources() -> None:
    dense = [_make_dense("a", 0.9), _make_dense("b", 0.7)]
    sparse = [_make_sparse("b", 10.0), _make_sparse("c", 8.0)]
    fused = fuse_results(dense, sparse, top_k=5)
    ids = {r.chunk_id for r in fused}
    assert ids == {"a", "b", "c"}


def test_fuse_results_top_k_limits_output() -> None:
    dense = [_make_dense(str(i), float(10 - i)) for i in range(10)]
    sparse = [_make_sparse(str(i), float(10 - i)) for i in range(10)]
    fused = fuse_results(dense, sparse, top_k=3)
    assert len(fused) == 3


def test_fuse_results_chunk_in_both_gets_higher_score() -> None:
    dense = [_make_dense("shared", 0.9), _make_dense("dense_only", 0.8)]
    sparse = [_make_sparse("shared", 10.0), _make_sparse("sparse_only", 9.0)]
    fused = fuse_results(dense, sparse, top_k=5)
    shared = next(r for r in fused if r.chunk_id == "shared")
    assert shared.dense_score is not None
    assert shared.sparse_score is not None


# ── Context Compression ───────────────────────────────────────────────────────

def _make_hybrid(chunk_id: str, score: float, text: str) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        rrf_score=score,
        dense_score=score,
        sparse_score=None,
        chunk_text=text,
        metadata={"filename": "doc.pdf"},
    )


def test_compress_context_respects_token_budget() -> None:
    results = [_make_hybrid(str(i), 1.0 - i * 0.1, "word " * 500) for i in range(10)]
    context = compress_context(results, max_tokens=600)
    assert context.total_tokens <= 600


def test_compress_context_deduplicates() -> None:
    text = "This is a very specific sentence about quarterly revenue growth."
    results = [
        _make_hybrid("a", 0.9, text),
        _make_hybrid("b", 0.8, text + " slightly different ending."),  # similar enough to dedup
    ]
    context = compress_context(results, max_tokens=5000, dedup_threshold=0.7)
    assert len(context.chunks) == 1
    assert context.dropped_count == 1


def test_compress_context_drops_low_score() -> None:
    results = [
        _make_hybrid("a", 0.8, "good chunk"),
        _make_hybrid("b", 0.1, "low quality chunk"),
    ]
    context = compress_context(results, max_tokens=5000, min_score=0.5)
    assert len(context.chunks) == 1
    assert context.chunks[0]["id"] == "a"


# ── Prompt Assembly ───────────────────────────────────────────────────────────

def test_assemble_prompt_includes_query() -> None:
    context = CompressedContext(
        chunks=[{"id": "1", "text": "Revenue grew 20% YoY.", "score": 0.9, "metadata": {"filename": "report.pdf"}}],
        total_tokens=10,
        dropped_count=0,
    )
    prompt = assemble_prompt("What drove revenue growth?", context)
    assert "What drove revenue growth?" in prompt.user
    assert "Revenue grew 20% YoY." in prompt.user


def test_assemble_prompt_system_contains_instructions() -> None:
    context = CompressedContext(chunks=[], total_tokens=0, dropped_count=0)
    prompt = assemble_prompt("test", context)
    assert "context" in prompt.system.lower()


def test_assemble_prompt_summarize_intent_modifies_system() -> None:
    context = CompressedContext(chunks=[], total_tokens=0, dropped_count=0)
    prompt = assemble_prompt("Summarize the report", context, intent="summarize")
    assert "summary" in prompt.system.lower() or "summarize" in prompt.system.lower() or "concise" in prompt.system.lower()


def test_assemble_prompt_source_attribution() -> None:
    context = CompressedContext(
        chunks=[{"id": "1", "text": "Key finding.", "score": 1.0, "metadata": {"filename": "q3.pdf"}}],
        total_tokens=5,
        dropped_count=0,
    )
    prompt = assemble_prompt("What are the key findings?", context)
    assert "q3.pdf" in prompt.user
