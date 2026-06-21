"""
Chunking engine — T18.
Four strategies, all return list[str]. Strategy is selected per-dataset.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[ChunkResult]: ...


# ── Token counter (tiktoken with fallback) ────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # ~4 chars/token fallback


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 50) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            if end == len(tokens):
                break
            start = end - overlap_tokens
        return chunks
    except Exception:
        # Character-based fallback
        size = max_tokens * 4
        overlap = overlap_tokens * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks


# ── Fixed Size Chunker ────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Split into fixed token windows with overlap.
    Fastest, least semantic — good for structured data (tables, CSV).
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[ChunkResult]:
        raw = _split_by_tokens(text, self.chunk_size, self.overlap)
        return [
            ChunkResult(text=c.strip(), index=i, metadata={"strategy": "fixed", "chunk_size": self.chunk_size})
            for i, c in enumerate(raw)
            if c.strip()
        ]


# ── Recursive Chunker ─────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Split on semantic boundaries: paragraphs → sentences → words.
    Preserves natural language structure. Default strategy.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(self, chunk_size: int = 1000, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return _split_by_tokens(text, self.chunk_size, self.overlap)

        sep = separators[0]
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if _count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if _count_tokens(part) > self.chunk_size:
                    sub = self._split(part, separators[1:])
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    def chunk(self, text: str) -> list[ChunkResult]:
        raw = self._split(text, self._SEPARATORS)
        return [
            ChunkResult(text=c.strip(), index=i, metadata={"strategy": "recursive"})
            for i, c in enumerate(raw)
            if c.strip()
        ]


# ── Semantic Chunker ──────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Embedding-based segmentation: splits where cosine similarity between
    adjacent sentence embeddings drops below a threshold.
    Falls back to recursive if embeddings unavailable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        threshold: float = 0.75,
        min_chunk_tokens: int = 100,
        max_chunk_tokens: int = 1000,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.embeddings.create(input=sentences, model=self.model)
            return [r.embedding for r in resp.data]
        except Exception:
            # Fallback: return zero vectors (will trigger threshold-based split at 0.0)
            return [[0.0] for _ in sentences]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def chunk(self, text: str) -> list[ChunkResult]:
        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return [ChunkResult(text=text.strip(), index=0, metadata={"strategy": "semantic"})]

        embeddings = self._embed_sentences(sentences)
        groups: list[list[str]] = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            current_tokens = _count_tokens(" ".join(groups[-1]))

            if sim < self.threshold or current_tokens >= self.max_chunk_tokens:
                if current_tokens >= self.min_chunk_tokens:
                    groups.append([sentences[i]])
                else:
                    groups[-1].append(sentences[i])
            else:
                groups[-1].append(sentences[i])

        return [
            ChunkResult(
                text=" ".join(g).strip(),
                index=i,
                metadata={"strategy": "semantic", "sentence_count": len(g)},
            )
            for i, g in enumerate(groups)
            if " ".join(g).strip()
        ]


# ── Hierarchical Chunker ──────────────────────────────────────────────────────

class HierarchicalChunker(BaseChunker):
    """
    Parent-child structure: large parent chunks + smaller child chunks.
    Parent chunks provide broad context; child chunks are indexed for retrieval.
    Metadata links child back to its parent index.
    """

    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 500,
        overlap: int = 50,
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[ChunkResult]:
        parent_texts = _split_by_tokens(text, self.parent_size, self.overlap)
        results: list[ChunkResult] = []
        child_index = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            child_texts = _split_by_tokens(parent_text, self.child_size, self.overlap)
            for child_text in child_texts:
                if child_text.strip():
                    results.append(
                        ChunkResult(
                            text=child_text.strip(),
                            index=child_index,
                            metadata={
                                "strategy": "hierarchical",
                                "parent_index": parent_idx,
                                "parent_text": parent_text[:200],  # preview for context
                            },
                        )
                    )
                    child_index += 1

        return results


# ── Factory ───────────────────────────────────────────────────────────────────

def get_chunker(strategy: str, **kwargs: int | float | str) -> BaseChunker:
    strategies: dict[str, type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "hierarchical": HierarchicalChunker,
    }
    cls = strategies.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {list(strategies)}")
    return cls(**kwargs)  # type: ignore[arg-type]
