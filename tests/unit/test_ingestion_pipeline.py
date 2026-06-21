"""
Unit tests for the ingestion pipeline components.
No real files, vector store, or Celery needed — everything is mocked or in-memory.
"""
from __future__ import annotations

import pytest

from app.services.ingestion.cleaner import clean_text, extract_metadata_hints
from app.services.ingestion.chunkers import (
    FixedSizeChunker,
    HierarchicalChunker,
    RecursiveChunker,
    get_chunker,
)
from app.services.ingestion.metadata import build_chunk_metadata, extract_metadata
from app.services.ingestion.parsers import (
    CSVParser,
    HTMLParser,
    TextParser,
    get_parser,
    parse_document,
)


# ── Cleaner ───────────────────────────────────────────────────────────────────

def test_clean_removes_excessive_newlines() -> None:
    dirty = "Hello\n\n\n\n\nWorld"
    result = clean_text(dirty)
    assert "\n\n\n" not in result


def test_clean_removes_null_bytes() -> None:
    dirty = "Hello\x00World"
    assert "\x00" not in clean_text(dirty)


def test_clean_removes_control_chars() -> None:
    dirty = "Good\x01text\x1fhere"
    assert "\x01" not in clean_text(dirty)
    assert "\x1f" not in clean_text(dirty)


def test_clean_strips_short_lines() -> None:
    text = "Good paragraph here.\n\nOk\nAnother good sentence follows."
    result = clean_text(text, min_line_length=5)
    assert "Ok" not in result


def test_clean_removes_urls() -> None:
    text = "Visit https://example.com for more info."
    result = clean_text(text, remove_urls=True)
    assert "https://" not in result


def test_clean_empty_input() -> None:
    assert clean_text("") == ""


def test_extract_metadata_hints_infers_title() -> None:
    text = "Annual Report 2024\n\nThis document covers our yearly performance."
    hints = extract_metadata_hints(text)
    assert hints["inferred_title"] == "Annual Report 2024"
    assert hints["word_count"] > 0


# ── Parsers ───────────────────────────────────────────────────────────────────

def test_text_parser() -> None:
    parser = TextParser()
    result = parser.parse(b"Hello World", "test.txt")
    assert result.text == "Hello World"
    assert result.metadata["filename"] == "test.txt"


def test_csv_parser() -> None:
    parser = CSVParser()
    data = b"name,age\nAlice,30\nBob,25"
    result = parser.parse(data, "data.csv")
    assert "Alice" in result.text
    assert "Bob" in result.text
    assert "|" in result.text


def test_html_parser_strips_script_tags() -> None:
    parser = HTMLParser()
    html = b"<html><body><script>alert('xss')</script><p>Hello World</p></body></html>"
    result = parser.parse(html, "page.html")
    assert "alert" not in result.text
    assert "Hello World" in result.text


def test_get_parser_raises_for_unsupported() -> None:
    from app.core.exceptions import UnsupportedFileTypeError
    with pytest.raises(UnsupportedFileTypeError):
        get_parser("exe")


def test_parse_document_dispatches_by_extension() -> None:
    result = parse_document(b"# Heading\n\nSome content", "notes.md")
    assert "Heading" in result.text


# ── Chunkers ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Introduction

This is the first paragraph of the document. It contains several sentences
that form a coherent unit of thought about the topic at hand.

Background

The second section provides context. It explains why this topic matters and
what previous work has been done in the field. Researchers have studied this
for decades without reaching a consensus.

Conclusion

Finally, we summarise the key findings. The main takeaway is that context
matters enormously when evaluating any claim.
""".strip()


def test_fixed_chunker_produces_chunks() -> None:
    chunker = FixedSizeChunker(chunk_size=100, overlap=10)
    results = chunker.chunk(SAMPLE_TEXT)
    assert len(results) >= 1
    for r in results:
        assert r.text.strip()
        assert r.metadata["strategy"] == "fixed"


def test_recursive_chunker_preserves_structure() -> None:
    chunker = RecursiveChunker(chunk_size=100, overlap=10)
    results = chunker.chunk(SAMPLE_TEXT)
    assert len(results) >= 1
    all_text = " ".join(r.text for r in results)
    assert "Introduction" in all_text or "paragraph" in all_text


def test_recursive_chunker_indices_are_sequential() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=5)
    results = chunker.chunk(SAMPLE_TEXT)
    indices = [r.index for r in results]
    assert indices == list(range(len(results)))


def test_hierarchical_chunker_sets_parent_index() -> None:
    chunker = HierarchicalChunker(parent_size=200, child_size=50, overlap=10)
    results = chunker.chunk(SAMPLE_TEXT)
    assert len(results) >= 1
    for r in results:
        assert "parent_index" in r.metadata
        assert isinstance(r.metadata["parent_index"], int)


def test_get_chunker_factory() -> None:
    for strategy in ("fixed", "recursive", "hierarchical"):
        chunker = get_chunker(strategy)
        assert callable(chunker.chunk)


def test_get_chunker_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("magic")


def test_chunker_handles_empty_text() -> None:
    for strategy in ("fixed", "recursive", "hierarchical"):
        chunker = get_chunker(strategy)
        results = chunker.chunk("")
        assert results == []


# ── Metadata ──────────────────────────────────────────────────────────────────

def test_extract_metadata_builds_full_record() -> None:
    meta = extract_metadata(
        filename="report.pdf",
        file_size=1024,
        parser_metadata={"title": "Q3 Report", "author": "Jane", "pages": 10},
        cleaned_text="Some clean text here.",
    )
    assert meta["title"] == "Q3 Report"
    assert meta["author"] == "Jane"
    assert meta["pages"] == 10
    assert meta["file_type"] == "pdf"
    assert meta["file_size"] == 1024
    assert "indexed_at" in meta


def test_extract_metadata_falls_back_to_filename() -> None:
    meta = extract_metadata(
        filename="quarterly_review.pdf",
        file_size=512,
        parser_metadata={},
        cleaned_text="Some text",
    )
    assert "Quarterly" in meta["title"]


def test_build_chunk_metadata() -> None:
    doc_meta = {"title": "Doc", "filename": "doc.pdf", "file_type": "pdf", "source_url": ""}
    chunk_meta = build_chunk_metadata(
        doc_metadata=doc_meta,
        chunk_index=3,
        chunk_text="This is a chunk of text.",
        page_hint=2,
    )
    assert chunk_meta["chunk_index"] == 3
    assert chunk_meta["page"] == 2
    assert chunk_meta["word_count"] == 6
    assert chunk_meta["document_title"] == "Doc"
