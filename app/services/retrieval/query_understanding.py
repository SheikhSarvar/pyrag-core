"""
Query understanding — T22.
Normalises the raw query string before retrieval.
Dependency-free: no external API calls in the hot path.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class UnderstoodQuery:
    original: str
    normalized: str
    language: str = "en"
    is_question: bool = False
    intent: str = "search"          # search | summarize | compare | explain
    keywords: list[str] = field(default_factory=list)


# ── Normalisation ─────────────────────────────────────────────────────────────

_EXCESS_WS   = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s\-?!.,]")

_QUESTION_WORDS = frozenset({
    "what", "who", "when", "where", "why", "how",
    "which", "is", "are", "was", "were", "can", "could",
    "should", "would", "do", "does", "did",
})

_SUMMARIZE_SIGNALS = frozenset({"summarize", "summarise", "summary", "overview", "tldr", "tl;dr"})
_COMPARE_SIGNALS   = frozenset({"compare", "difference", "vs", "versus", "contrast", "between"})
_EXPLAIN_SIGNALS   = frozenset({"explain", "what is", "what are", "define", "definition", "meaning"})


def _detect_intent(text: str) -> str:
    lower = text.lower()
    if any(s in lower for s in _SUMMARIZE_SIGNALS):
        return "summarize"
    if any(s in lower for s in _COMPARE_SIGNALS):
        return "compare"
    if any(s in lower for s in _EXPLAIN_SIGNALS):
        return "explain"
    return "search"


def _is_question(text: str) -> bool:
    if text.strip().endswith("?"):
        return True
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    return first_word in _QUESTION_WORDS


def _extract_keywords(text: str, stop_words: frozenset[str] | None = None) -> list[str]:
    _STOP = stop_words or frozenset({
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "with", "from", "by",
        "this", "that", "these", "those", "it", "its",
    })
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return [w for w in words if w not in _STOP]


def _detect_language(text: str) -> str:
    """
    Lightweight heuristic language detection.
    Returns ISO 639-1 code. Falls back to 'en'.
    Uses langdetect if installed, otherwise returns 'en'.
    """
    try:
        from langdetect import detect
        return detect(text) or "en"
    except Exception:
        return "en"


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _EXCESS_WS.sub(" ", text)
    return text.strip()


def understand_query(raw_query: str) -> UnderstoodQuery:
    """
    Parse and enrich a raw query string.
    Returns an UnderstoodQuery with normalised text, language, intent, and keywords.
    """
    normalized = normalize_query(raw_query)
    return UnderstoodQuery(
        original=raw_query,
        normalized=normalized,
        language=_detect_language(normalized),
        is_question=_is_question(normalized),
        intent=_detect_intent(normalized),
        keywords=_extract_keywords(normalized),
    )
