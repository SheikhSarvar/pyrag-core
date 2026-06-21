"""
Text cleaner — T16.
Removes noise, normalises whitespace, strips boilerplate.
Designed to be fast and dependency-free.
"""
from __future__ import annotations

import re
import unicodedata


# ── Patterns compiled once at import time ─────────────────────────────────────

_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
_EXCESSIVE_SPACES   = re.compile(r"[ \t]{2,}")
_NULL_BYTES         = re.compile(r"\x00")
_CONTROL_CHARS      = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_NUMBERS       = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)
_REPEATED_CHARS     = re.compile(r"(.)\1{4,}")   # 5+ same chars in a row
_URL_PATTERN        = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN      = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)

# Lines that are almost certainly headers/footers/noise
_NOISE_PATTERNS = re.compile(
    r"^(confidential|all rights reserved|copyright|©|page \d+|\.{5,}|_{5,}|-{5,})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(
    text: str,
    *,
    remove_urls: bool = False,
    remove_emails: bool = False,
    normalize_unicode: bool = True,
    min_line_length: int = 3,
) -> str:
    """
    Clean extracted document text.

    Args:
        text: Raw extracted text.
        remove_urls: Strip HTTP/HTTPS URLs.
        remove_emails: Strip email addresses.
        normalize_unicode: NFKC-normalise unicode (ligatures, full-width chars, etc.).
        min_line_length: Lines shorter than this are dropped.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Unicode normalisation
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    # Remove null bytes and control characters
    text = _NULL_BYTES.sub("", text)
    text = _CONTROL_CHARS.sub("", text)

    # Optionally strip URLs and emails
    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)
    if remove_emails:
        text = _EMAIL_PATTERN.sub(" ", text)

    # Remove obvious noise lines
    text = _NOISE_PATTERNS.sub("", text)
    text = _PAGE_NUMBERS.sub("", text)

    # Normalise whitespace
    text = _EXCESSIVE_SPACES.sub(" ", text)

    # Filter short / empty lines
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if len(line.strip()) >= min_line_length
    ]
    text = "\n".join(lines)

    # Collapse excessive blank lines to max 2
    text = _EXCESSIVE_NEWLINES.sub("\n\n", text)

    return text.strip()


def extract_metadata_hints(text: str) -> dict:
    """
    Heuristically extract title and section hints from cleaned text.
    Used by the metadata extractor as a fallback.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ""
    if lines:
        # First non-empty line that's not too long is likely a title
        for line in lines[:5]:
            if 5 < len(line) < 150:
                title = line
                break
    return {"inferred_title": title, "word_count": len(text.split())}
