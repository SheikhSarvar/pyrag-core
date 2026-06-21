"""
Context compression (T28) and prompt assembly (T29).
Deduplicates, trims low-value chunks, and builds the final prompt
that goes to the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── T28 — Context Compression ─────────────────────────────────────────────────

@dataclass
class CompressedContext:
    chunks: list[dict]          # {id, text, score, metadata}
    total_tokens: int
    dropped_count: int


def _approx_tokens(text: str) -> int:
    """~4 chars per token is accurate enough for context budgeting."""
    return max(1, len(text) // 4)


def _similarity_ratio(a: str, b: str) -> float:
    """Fast character-level Jaccard similarity for dedup."""
    a_set = set(re.findall(r"\b\w{4,}\b", a.lower()))
    b_set = set(re.findall(r"\b\w{4,}\b", b.lower()))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def compress_context(
    results: list[Any],
    max_tokens: int = 4000,
    dedup_threshold: float = 0.85,
    min_score: float = 0.0,
) -> CompressedContext:
    """
    Deduplicate and trim retrieval results to fit within a token budget.

    Args:
        results:         Any retrieval result objects with .chunk_text and a score attr.
        max_tokens:      Hard token budget for all context chunks combined.
        dedup_threshold: Jaccard similarity above which a chunk is considered duplicate.
        min_score:       Drop chunks below this score (0.0 = keep all).

    Returns:
        CompressedContext with the surviving chunks and token count.
    """
    def _score(r: Any) -> float:
        for attr in ("rerank_score", "rrf_score", "score"):
            if hasattr(r, attr):
                return float(getattr(r, attr))
        return 0.0

    # Sort descending by score
    candidates = sorted(results, key=_score, reverse=True)
    kept: list[dict] = []
    seen_texts: list[str] = []
    total_tokens = 0
    dropped = 0

    for result in candidates:
        score = _score(result)
        text = getattr(result, "chunk_text", "")

        # Drop below score threshold
        if score < min_score:
            dropped += 1
            continue

        # Dedup: skip if too similar to an already-kept chunk
        is_dup = any(
            _similarity_ratio(text, seen) >= dedup_threshold
            for seen in seen_texts
        )
        if is_dup:
            dropped += 1
            continue

        chunk_tokens = _approx_tokens(text)
        if total_tokens + chunk_tokens > max_tokens:
            dropped += 1
            continue

        kept.append({
            "id": getattr(result, "chunk_id", ""),
            "text": text,
            "score": score,
            "metadata": getattr(result, "metadata", {}),
        })
        seen_texts.append(text)
        total_tokens += chunk_tokens

    return CompressedContext(chunks=kept, total_tokens=total_tokens, dropped_count=dropped)


# ── T29 — Prompt Assembly ─────────────────────────────────────────────────────

@dataclass
class AssembledPrompt:
    system: str
    user: str
    context_chunks: list[dict]
    total_tokens: int


_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain enough information "
    "to answer, say so clearly. Do not make up information."
)

_CONTEXT_TEMPLATE = """
<context>
{chunks}
</context>

{query}
""".strip()

_CHUNK_TEMPLATE = """[{index}] (source: {source})
{text}""".strip()


def assemble_prompt(
    query: str,
    context: CompressedContext,
    system_prompt: str | None = None,
    intent: str = "search",
) -> AssembledPrompt:
    """
    Build a system + user prompt pair from the compressed context.

    Args:
        query:         The user's original query.
        context:       Output of compress_context().
        system_prompt: Override the default system prompt.
        intent:        Query intent (search | summarize | compare | explain)
                       used to tailor instructions.

    Returns:
        AssembledPrompt with system and user fields ready for the LLM.
    """
    intent_addendum = {
        "summarize": " Provide a concise, structured summary of the key points.",
        "compare":   " Explicitly compare and contrast the relevant items mentioned in the context.",
        "explain":   " Explain the concept clearly, step by step if needed.",
        "search":    "",
    }.get(intent, "")

    system = (system_prompt or _DEFAULT_SYSTEM) + intent_addendum

    chunk_blocks = "\n\n".join(
        _CHUNK_TEMPLATE.format(
            index=i + 1,
            source=chunk["metadata"].get("filename", chunk["metadata"].get("source_url", "unknown")),
            text=chunk["text"].strip(),
        )
        for i, chunk in enumerate(context.chunks)
    )

    user = _CONTEXT_TEMPLATE.format(chunks=chunk_blocks, query=query)

    return AssembledPrompt(
        system=system,
        user=user,
        context_chunks=context.chunks,
        total_tokens=context.total_tokens + _approx_tokens(system) + _approx_tokens(query),
    )
