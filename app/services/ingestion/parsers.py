"""
Document parsers — T14 + T15.
Each parser receives raw bytes and returns plain text + basic metadata.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ParsedDocument:
    text: str
    metadata: dict = field(default_factory=dict)
    pages: int = 1
    elements: list[Any] | None = None


class Parser(Protocol):
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...


# ── PDF ───────────────────────────────────────────────────────────────────────

class PDFParser:
    """Native PDF parser backed by PyMuPDF4LLM for markdown text extraction."""

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        import fitz  # pymupdf
        import pymupdf4llm

        doc = fitz.open(stream=data, filetype="pdf")
        page_count = doc.page_count
        metadata = {
            "title": doc.metadata.get("title", "") if doc.metadata else "",
            "author": doc.metadata.get("author", "") if doc.metadata else "",
            "subject": doc.metadata.get("subject", "") if doc.metadata else "",
            "pages": page_count,
            "filename": filename,
            "parser": "PDFParser",
            "parser_implementation": f"pymupdf4llm/{getattr(pymupdf4llm, '__version__', 'unknown')}",
        }
        # Built-in OCR triggers automatically on pages with no extractable text
        text = pymupdf4llm.to_markdown(doc) or ""
        doc.close()
        return ParsedDocument(text=text, metadata=metadata, pages=page_count, elements=None)


# ── DOCX ──────────────────────────────────────────────────────────────────────

class DOCXParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Include text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        text = "\n\n".join(paragraphs)
        props = doc.core_properties
        metadata = {
            "title": props.title or "",
            "author": props.author or "",
            "filename": filename,
        }
        return ParsedDocument(text=text, metadata=metadata)


# ── PPTX ──────────────────────────────────────────────────────────────────────

class PPTXParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from pptx import Presentation

        prs = Presentation(io.BytesIO(data))
        slides_text: list[str] = []
        for i, slide in enumerate(prs.slides, 1):
            slide_lines: list[str] = [f"[Slide {i}]"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            slide_lines.append(line)
            slides_text.append("\n".join(slide_lines))
        text = "\n\n".join(slides_text)
        metadata = {"filename": filename, "slides": len(prs.slides)}
        return ParsedDocument(text=text, metadata=metadata, pages=len(prs.slides))


# ── XLSX ──────────────────────────────────────────────────────────────────────

class XLSXParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheet_count = len(wb.sheetnames)
        sheets_text: list[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[str] = [f"[Sheet: {sheet_name}]"]
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join(str(v) for v in row if v is not None)
                if row_text.strip():
                    rows.append(row_text)
            sheets_text.append("\n".join(rows))
        wb.close()
        return ParsedDocument(
            text="\n\n".join(sheets_text),
            metadata={"filename": filename, "sheets": sheet_count},
        )


# ── CSV ───────────────────────────────────────────────────────────────────────

class CSVParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text_data = data.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text_data))
        rows = [" | ".join(row) for row in reader if any(cell.strip() for cell in row)]
        return ParsedDocument(
            text="\n".join(rows),
            metadata={"filename": filename, "rows": len(rows)},
        )


# ── TXT / Markdown ────────────────────────────────────────────────────────────

class TextParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text = data.decode("utf-8", errors="replace")
        return ParsedDocument(text=text, metadata={"filename": filename})


# ── HTML ──────────────────────────────────────────────────────────────────────

class HTMLParser:
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator="\n", strip=True)
        return ParsedDocument(text=text, metadata={"filename": filename, "title": title})


# ── Web Scraper (T15) ─────────────────────────────────────────────────────────

class WebScraper:
    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def scrape(self, url: str) -> ParsedDocument:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(url, timeout=self._timeout, follow_redirects=True, headers={
            "User-Agent": "PyRAG-Core/1.0 (+https://github.com/pyrag-core)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text(separator="\n", strip=True)
        return ParsedDocument(
            text=text,
            metadata={"source_url": url, "title": title, "content_type": resp.headers.get("content-type", "")},
        )


# ── Unstructured (Advanced Extraction) ───────────────────────────────────────

class UnstructuredParser:
    """
    Advanced multi-format parser powered by Unstructured.
    Extracts typed structural elements (Title, NarrativeText, Table, ListItem, etc.)
    and produces both a flattened text representation (bridged with markdown headings)
    and the structured elements list.
    """

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        import unstructured
        from unstructured.partition.auto import partition

        elements = partition(file=io.BytesIO(data), metadata_filename=filename)

        # Deliberate short-term bridge: Mark titles/headings with markdown syntax (e.g. '## <title>')
        # in the flattened text so downstream text-based cleaner and chunkers can recognize
        # structural boundaries without touching cleaner.py or chunker.py yet.
        # Future extension: An element-aware chunker will consume `ParsedDocument.elements`
        # directly instead of re-deriving structure from flattened text.
        text_blocks: list[str] = []
        for el in elements:
            category = getattr(el, "category", "") or type(el).__name__
            el_text = getattr(el, "text", None)
            if el_text is None:
                el_text = str(el)
            el_text = str(el_text).strip()
            if not el_text:
                continue

            if category in ("Title", "Header", "Subheadline"):
                if not el_text.startswith("#"):
                    text_blocks.append(f"## {el_text}")
                else:
                    text_blocks.append(el_text)
            elif category == "ListItem":
                if not (el_text.startswith("- ") or el_text.startswith("* ")):
                    text_blocks.append(f"- {el_text}")
                else:
                    text_blocks.append(el_text)
            elif category == "Table":
                meta = getattr(el, "metadata", None)
                table_html = getattr(meta, "text_as_html", None) if meta else None
                text_blocks.append(table_html if table_html else el_text)
            else:
                text_blocks.append(el_text)

        text = "\n\n".join(text_blocks)

        page_numbers: list[int] = []
        for el in elements:
            meta = getattr(el, "metadata", None)
            if meta is not None:
                pn = getattr(meta, "page_number", None)
                if isinstance(pn, int):
                    page_numbers.append(pn)
        pages = max(page_numbers) if page_numbers else 1

        title = ""
        for el in elements:
            if getattr(el, "category", "") == "Title":
                t = getattr(el, "text", None)
                if t is None:
                    t = str(el)
                t = str(t).strip()
                if t:
                    title = t
                    break

        try:
            from unstructured.__version__ import __version__ as unstructured_version
        except Exception:
            ver_obj = getattr(unstructured, "__version__", "unknown")
            unstructured_version = getattr(ver_obj, "__version__", str(ver_obj))

        metadata = {
            "filename": filename,
            "pages": pages,
            "parser": "UnstructuredParser",
            "parser_implementation": f"unstructured/{unstructured_version}",
            "element_count": len(elements),
        }
        if title:
            metadata["title"] = title

        return ParsedDocument(text=text, metadata=metadata, pages=pages, elements=elements)


# ── Registry ──────────────────────────────────────────────────────────────────

DEFAULT_EXTRACTION_STRATEGY = "native"
SUPPORTED_EXTRACTION_STRATEGIES = {"native", "unstructured"}

_pdf_native = PDFParser()
_docx_native = DOCXParser()
_pptx_native = PPTXParser()
_xlsx_native = XLSXParser()
_csv_native = CSVParser()
_text_native = TextParser()
_html_native = HTMLParser()
_unstructured = UnstructuredParser()

_PARSER_REGISTRY: dict[str, dict[str, Parser]] = {
    "pdf":      {"native": _pdf_native, "unstructured": _unstructured},
    "docx":     {"native": _docx_native, "unstructured": _unstructured},
    "pptx":     {"native": _pptx_native, "unstructured": _unstructured},
    "xlsx":     {"native": _xlsx_native, "unstructured": _unstructured},
    "xls":      {"native": _xlsx_native, "unstructured": _unstructured},
    "csv":      {"native": _csv_native, "unstructured": _unstructured},
    "txt":      {"native": _text_native, "unstructured": _unstructured},
    "md":       {"native": _text_native, "unstructured": _unstructured},
    "markdown": {"native": _text_native, "unstructured": _unstructured},
    "html":     {"native": _html_native, "unstructured": _unstructured},
    "htm":      {"native": _html_native, "unstructured": _unstructured},
}

SUPPORTED_EXTENSIONS = set(_PARSER_REGISTRY.keys())
_PARSERS = _PARSER_REGISTRY  # Backwards compatibility alias


def get_supported_strategies(extension: str | None = None) -> set[str]:
    """Return supported extraction strategies, optionally filtered by extension."""
    if extension is None:
        return set(SUPPORTED_EXTRACTION_STRATEGIES)
    ext = extension.lower().lstrip(".")
    ext_parsers = _PARSER_REGISTRY.get(ext)
    if ext_parsers is None:
        return set()
    return set(ext_parsers.keys())


def get_parser(extension: str, strategy: str = DEFAULT_EXTRACTION_STRATEGY) -> Parser:
    from app.core.exceptions import UnsupportedFileTypeError

    ext = extension.lower().lstrip(".")
    ext_parsers = _PARSER_REGISTRY.get(ext)
    if ext_parsers is None:
        raise UnsupportedFileTypeError(f"Unsupported file type: .{ext}")

    parser = ext_parsers.get(strategy.lower())
    if parser is None:
        raise UnsupportedFileTypeError(
            f"Unsupported extraction strategy '{strategy}' for file type: .{ext}"
        )
    return parser


def parse_document(
    data: bytes,
    filename: str,
    strategy: str = DEFAULT_EXTRACTION_STRATEGY,
) -> ParsedDocument:
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    parser = get_parser(ext, strategy=strategy)
    doc = parser.parse(data, filename)
    doc.metadata["extraction_strategy"] = strategy.lower()
    return doc
