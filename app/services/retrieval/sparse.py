"""
Sparse retrieval — T25.
BM25 over chunk text stored in PostgreSQL.
No external index needed — suitable for datasets up to ~100k chunks.
For larger datasets, plug in Elasticsearch with BM25 field queries.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class SparseResult:
    chunk_id: str
    score: float
    chunk_text: str
    metadata: dict


class BM25:
    """
    In-process BM25 implementation.
    Call `build(corpus)` then `search(query, top_k)`.

    k1=1.5, b=0.75 — standard defaults that work well across domains.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._doc_metadata: list[dict] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def build(
        self,
        documents: list[tuple[str, str, dict]],  # (id, text, metadata)
    ) -> None:
        """Build BM25 index from a list of (id, text, metadata) tuples."""
        self._doc_ids = [d[0] for d in documents]
        self._doc_texts = [d[1] for d in documents]
        self._doc_metadata = [d[2] for d in documents]
        self._corpus = [self._tokenize(d[1]) for d in documents]

        n = len(self._corpus)
        if n == 0:
            return

        self._avgdl = sum(len(doc) for doc in self._corpus) / n

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df: Counter[str] = Counter()
        for doc_tokens in self._corpus:
            for token in set(doc_tokens):
                df[token] += 1

        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SparseResult]:
        if not self._corpus:
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = []

        for doc_tokens in self._corpus:
            tf_map: Counter[str] = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for token in query_tokens:
                if token not in self._idf:
                    continue
                tf = tf_map[token]
                idf = self._idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * (numerator / denominator)
            scores.append(score)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            SparseResult(
                chunk_id=self._doc_ids[i],
                score=scores[i],
                chunk_text=self._doc_texts[i],
                metadata=self._doc_metadata[i],
            )
            for i in ranked[:top_k]
            if scores[i] > 0
        ]


async def sparse_search(
    dataset_id: str,
    query: str,
    top_k: int = 20,
    session=None,  # AsyncSession — injected at call site
) -> list[SparseResult]:
    """
    Build a BM25 index on-the-fly from chunks in PostgreSQL and search it.

    For production-scale deployments (>100k chunks), replace with an
    Elasticsearch BM25 query against `ElasticsearchAdapter`.
    """
    if session is None:
        return []

    from sqlalchemy import select, text
    from app.db.models.chunk import Chunk

    result = await session.execute(
        select(Chunk.id, Chunk.chunk_text, Chunk.chunk_metadata)
        .where(Chunk.dataset_id == dataset_id)
        .order_by(Chunk.created_at.desc())
        .limit(50_000)  # safety cap
    )
    rows = result.all()

    if not rows:
        return []

    docs = [(row.id, row.chunk_text, row.chunk_metadata or {}) for row in rows]
    bm25 = BM25()
    bm25.build(docs)
    return bm25.search(query, top_k=top_k)
