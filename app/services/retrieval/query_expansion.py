"""
Query expansion — T23.
Generates alternative phrasings to broaden recall before retrieval.
Two strategies:
  1. Rule-based synonym expansion (no API, always available)
  2. LLM-based expansion (optional, higher quality)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExpandedQuery:
    original: str
    variants: list[str] = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        """Original + all variants, deduplicated."""
        seen: set[str] = set()
        result: list[str] = []
        for q in [self.original] + self.variants:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(q)
        return result


# ── Rule-based synonym map ─────────────────────────────────────────────────────
# Domain-agnostic terms likely to appear in enterprise knowledge bases.

_SYNONYMS: dict[str, list[str]] = {
    "revenue":     ["sales", "income", "earnings", "turnover"],
    "cost":        ["expense", "expenditure", "price", "spend"],
    "customer":    ["client", "user", "buyer", "consumer"],
    "employee":    ["staff", "worker", "team member", "headcount"],
    "product":     ["item", "offering", "solution", "service"],
    "increase":    ["grow", "rise", "improve", "boost", "uptick"],
    "decrease":    ["decline", "drop", "reduce", "fall", "downturn"],
    "report":      ["document", "summary", "analysis", "review"],
    "meeting":     ["call", "sync", "session", "discussion"],
    "issue":       ["problem", "bug", "defect", "error", "failure"],
    "policy":      ["guideline", "rule", "procedure", "protocol"],
    "quarter":     ["Q1", "Q2", "Q3", "Q4", "quarterly"],
    "performance": ["metrics", "KPI", "results", "outcomes"],
}

_COMPILED_SYNONYMS = {re.compile(rf"\b{k}\b", re.IGNORECASE): v for k, v in _SYNONYMS.items()}


def _rule_based_expand(query: str, max_variants: int = 3) -> list[str]:
    variants: list[str] = []
    for pattern, synonyms in _COMPILED_SYNONYMS.items():
        if pattern.search(query):
            for syn in synonyms[:2]:
                variant = pattern.sub(syn, query)
                if variant.lower() != query.lower():
                    variants.append(variant)
                if len(variants) >= max_variants:
                    return variants
    return variants


async def _llm_expand(query: str, n_variants: int = 3) -> list[str]:
    """
    Use an LLM to generate semantically equivalent reformulations.
    Returns empty list if LLM is unavailable — callers must handle gracefully.
    """
    try:
        from openai import AsyncOpenAI
        from app.core.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            return []

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        prompt = (
            f"Generate {n_variants} alternative phrasings of the following search query. "
            f"Return only the phrasings, one per line, no numbering or explanation.\n\n"
            f"Query: {query}"
        )
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        content = resp.choices[0].message.content or ""
        return [line.strip() for line in content.splitlines() if line.strip()][:n_variants]
    except Exception:
        return []


async def expand_query(
    query: str,
    *,
    use_llm: bool = False,
    max_variants: int = 3,
) -> ExpandedQuery:
    """
    Expand a query with synonym substitutions and optionally LLM reformulations.

    Args:
        query: The normalised query string.
        use_llm: Whether to call an LLM for semantic variants (slower, better).
        max_variants: Maximum number of variants to return.

    Returns:
        ExpandedQuery with original + variant strings.
    """
    variants = _rule_based_expand(query, max_variants=max_variants)

    if use_llm and len(variants) < max_variants:
        llm_variants = await _llm_expand(query, n_variants=max_variants - len(variants))
        variants.extend(llm_variants)

    return ExpandedQuery(original=query, variants=variants[:max_variants])
