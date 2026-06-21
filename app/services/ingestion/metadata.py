"""
Metadata extractor — T17.
Merges parser-supplied metadata with heuristic and file-level metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def extract_metadata(
    filename: str,
    file_size: int,
    parser_metadata: dict,
    cleaned_text: str,
    source_url: str | None = None,
) -> dict:
    """
    Build the final metadata dict stored in Chunk.chunk_metadata and Document.

    Priority: parser_metadata > heuristic hints > defaults.
    """
    from app.services.ingestion.cleaner import extract_metadata_hints

    hints = extract_metadata_hints(cleaned_text)
    suffix = Path(filename).suffix.lstrip(".").lower()

    # Title resolution: parser > filename stem > heuristic
    title = (
        parser_metadata.get("title")
        or Path(filename).stem.replace("_", " ").replace("-", " ").title()
        or hints.get("inferred_title", "")
    )

    return {
        "filename": filename,
        "file_type": suffix,
        "file_size": file_size,
        "title": title,
        "author": parser_metadata.get("author", ""),
        "pages": parser_metadata.get("pages", 1),
        "slides": parser_metadata.get("slides"),
        "sheets": parser_metadata.get("sheets"),
        "word_count": hints.get("word_count", 0),
        "source_url": source_url or parser_metadata.get("source_url", ""),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


def build_chunk_metadata(
    doc_metadata: dict,
    chunk_index: int,
    chunk_text: str,
    page_hint: int | None = None,
    section_hint: str | None = None,
) -> dict:
    """
    Build per-chunk metadata payload stored alongside vectors.
    Kept in both PostgreSQL (chunks.chunk_metadata) and the vector store payload.
    """
    return {
        # Document-level context
        "document_title": doc_metadata.get("title", ""),
        "filename": doc_metadata.get("filename", ""),
        "file_type": doc_metadata.get("file_type", ""),
        "source_url": doc_metadata.get("source_url", ""),
        # Chunk-level context
        "chunk_index": chunk_index,
        "page": page_hint,
        "section": section_hint,
        "word_count": len(chunk_text.split()),
        "char_count": len(chunk_text),
    }
