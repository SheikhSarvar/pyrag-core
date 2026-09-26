"""
Unit tests for the ingestion pipeline components.
No real files, vector store, or Celery needed — everything is mocked or in-memory.
"""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fitz
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
    PDFParser,
    TextParser,
    UnstructuredParser,
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


def _make_minimal_pdf(text: str = "Hello PyMuPDF4LLM") -> bytes:
    """Build a real single-page PDF in memory using fitz."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_parser_returns_markdown_and_closes_document() -> None:
    """PDFParser now uses PyMuPDF4LLM — output is markdown, not plain text."""
    raw = _make_minimal_pdf("Hello PyMuPDF4LLM")
    result = PDFParser().parse(raw, "report.pdf")
    assert result.pages == 1
    assert result.metadata["pages"] == 1
    # PyMuPDF4LLM returns markdown — content is present but format differs from fitz plain text
    assert "Hello" in result.text
    assert result.elements is None  # native PDF parser leaves elements unpopulated
    assert result.metadata["parser"] == "PDFParser"
    assert "pymupdf4llm" in result.metadata["parser_implementation"]
    # extraction_strategy is stamped by parse_document(), not by PDFParser().parse() directly


def test_pdf_parser_metadata_captured() -> None:
    """PDFParser metadata carries parser provenance fields."""
    raw = _make_minimal_pdf("Content")
    result = PDFParser().parse(raw, "doc.pdf")
    assert "parser" in result.metadata
    assert "parser_implementation" in result.metadata
    assert result.metadata["filename"] == "doc.pdf"


def test_pdf_parser_image_only_returns_empty_or_string() -> None:
    """On an image-only (blank text) PDF, PyMuPDF4LLM returns a string (possibly empty)."""
    # Build a page with no extractable text (empty text insertion)
    doc = fitz.open()
    doc.new_page()
    raw = doc.tobytes()
    doc.close()
    result = PDFParser().parse(raw, "scanned.pdf")
    assert isinstance(result.text, str)  # always a string, even if empty


def test_parse_document_stamps_native_strategy_on_pdf() -> None:
    """parse_document() stamps extraction_strategy='native' on the returned metadata."""
    raw = _make_minimal_pdf("Strategy stamp check")
    result = parse_document(raw, "check.pdf", strategy="native")
    assert result.metadata["extraction_strategy"] == "native"


# ── Registry / strategy dispatch ──────────────────────────────────────────────

def test_get_parser_returns_native_by_default() -> None:
    """get_parser with no strategy arg defaults to the 'native' parser."""
    parser = get_parser("pdf")
    assert isinstance(parser, PDFParser)


def test_get_parser_native_explicit() -> None:
    """Explicitly requesting strategy='native' also returns the native parser."""
    parser = get_parser("pdf", strategy="native")
    assert isinstance(parser, PDFParser)


def test_get_parser_unstructured_returns_unstructured_parser() -> None:
    """get_parser(strategy='unstructured') returns UnstructuredParser for every format."""
    for ext in ("pdf", "docx", "pptx", "xlsx", "csv", "txt", "html"):
        parser = get_parser(ext, strategy="unstructured")
        assert isinstance(parser, UnstructuredParser), (
            f"Expected UnstructuredParser for ext={ext!r}, got {type(parser)}"
        )


def test_get_parser_raises_for_invalid_strategy() -> None:
    """An unknown strategy name raises UnsupportedFileTypeError."""
    from app.core.exceptions import UnsupportedFileTypeError
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported extraction strategy"):
        get_parser("pdf", strategy="magical_llm")


def test_parse_document_unstructured_strategy_mocked() -> None:
    """
    parse_document(strategy='unstructured') calls UnstructuredParser.parse.
    We mock the partition call so no real Unstructured heavy deps are needed.
    """
    fake_element = MagicMock()
    fake_element.text = "Mocked unstructured content"
    fake_element.category = "NarrativeText"
    fake_element.__class__.__name__ = "NarrativeText"

    with patch(
        "unstructured.partition.auto.partition",
        return_value=[fake_element],
    ):
        result = parse_document(b"any bytes", "test.txt", strategy="unstructured")

    assert "Mocked unstructured content" in result.text
    assert result.elements is not None
    assert len(result.elements) == 1
    assert result.metadata["extraction_strategy"] == "unstructured"
    assert result.metadata["parser_implementation"].startswith("unstructured")


def test_unstructured_parser_mocked_on_pdf() -> None:
    """UnstructuredParser on PDF populates both .text and .elements."""
    fake_el = MagicMock()
    fake_el.text = "PDF extracted via Unstructured"
    fake_el.category = "NarrativeText"
    fake_el.__class__.__name__ = "NarrativeText"

    raw = _make_minimal_pdf("PDF Unstructured test")
    with patch(
        "unstructured.partition.auto.partition",
        return_value=[fake_el],
    ):
        result = UnstructuredParser().parse(raw, "sample.pdf")

    assert "PDF extracted via Unstructured" in result.text
    assert result.elements is not None


def test_unstructured_parser_excludes_non_text_elements() -> None:
    """UnstructuredParser skips elements whose text is empty/whitespace."""
    el_with_text = MagicMock()
    el_with_text.text = "Real content"
    el_with_text.category = "NarrativeText"
    el_with_text.__class__.__name__ = "NarrativeText"

    el_empty = MagicMock()
    el_empty.text = "   "
    el_empty.category = "Header"
    el_empty.__class__.__name__ = "Header"

    with patch(
        "unstructured.partition.auto.partition",
        return_value=[el_with_text, el_empty],
    ):
        result = UnstructuredParser().parse(b"some content", "doc.docx")

    # Only the non-empty element should appear in the concatenated text
    assert "Real content" in result.text
    assert "   " not in result.text


# ── Metadata provenance ───────────────────────────────────────────────────────

def test_extract_metadata_records_extraction_strategy() -> None:
    """extract_metadata stores extraction_strategy when provided in parser_metadata."""
    from app.services.ingestion.metadata import extract_metadata

    meta = extract_metadata(
        filename="report.pdf",
        file_size=2048,
        parser_metadata={
            "pages": 4,
            "extraction_strategy": "unstructured",
            "parser_implementation": "unstructured/0.16.0",
        },
        cleaned_text="Some clean text.",
    )
    assert meta["extraction_strategy"] == "unstructured"
    assert meta["parser_implementation"] == "unstructured/0.16.0"


def test_extract_metadata_defaults_native_strategy() -> None:
    """extract_metadata defaults extraction_strategy to 'native' when not in parser_metadata."""
    from app.services.ingestion.metadata import extract_metadata

    meta = extract_metadata(
        filename="notes.txt",
        file_size=128,
        parser_metadata={},
        cleaned_text="Some text.",
    )
    assert meta["extraction_strategy"] == "native"


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
