"""
Ingestion pipeline orchestrator.
Wires together: parse → clean → metadata → chunk → embed → index.
Called by the Celery task (T13) and the reindex endpoint (T39).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories import DocumentRepository, JobRepository
from app.services.embedding.providers import EmbeddingProvider, get_embedding_provider
from app.services.ingestion.chunkers import get_chunker
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.indexer import index_chunks
from app.services.ingestion.metadata import extract_metadata
from app.services.ingestion.parsers import parse_document
from app.services.storage.minio_client import MinIOClient, get_minio_client
from app.services.vector.base import VectorStore
from app.services.vector.factory import get_vector_store

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    dataset_id: str
    chunks_indexed: int
    success: bool
    error: str | None = None


async def run_ingestion_pipeline(
    *,
    session: AsyncSession,
    dataset_id: str,
    document_id: str,
    filename: str,
    file_size: int,
    storage_path: str,
    chunk_strategy: str = "recursive",
    source_url: str | None = None,
    job_id: str | None = None,
    reindex: bool = False,
    # Injectable for testing
    minio_client: MinIOClient | None = None,
    vector_store: VectorStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> IngestionResult:

    doc_repo = DocumentRepository(session)
    job_repo = JobRepository(session) if job_id else None

    async def _update_progress(pct: int) -> None:
        if job_repo and job_id:
            await job_repo.set_progress(job_id, pct)

    try:
        await doc_repo.set_status(document_id, "processing")
        await _update_progress(5)

        # ── Step 1: Download ──────────────────────────────────────────────────
        client = minio_client or get_minio_client()
        from app.core.config import get_settings
        settings = get_settings()

        # On reindex: short-circuit the expensive raw-download + parse by
        # reading previously extracted text from the processed bucket.
        cleaned: str | None = None
        parsed_metadata: dict = {}
        if reindex:
            processed_key = MinIOClient.processed_path(dataset_id, document_id, "extracted.txt")
            if client.object_exists(settings.minio_bucket_processed, processed_key):
                try:
                    cleaned = client.download_bytes(
                        settings.minio_bucket_processed, processed_key
                    ).decode("utf-8")
                    logger.info(
                        "Reindex: loaded extracted text from processed bucket — skipping raw parse",
                        document_id=document_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Reindex: failed to read processed text, falling back to raw parse",
                        document_id=document_id,
                        error=str(exc),
                    )
                    cleaned = None

        if cleaned is None:
            # Full path: download raw file and parse it
            raw_data = client.download_bytes(settings.minio_bucket_raw, storage_path)
            logger.info("Downloaded document", document_id=document_id, size=len(raw_data))
            await _update_progress(15)

            # ── Step 2: Parse ─────────────────────────────────────────────────
            parsed = parse_document(raw_data, filename)
            parsed_metadata = parsed.metadata
            await _update_progress(30)

            # ── Step 3: Clean ─────────────────────────────────────────────────
            cleaned = clean_text(parsed.text)
            if not cleaned:
                raise ValueError("Document produced no extractable text after cleaning")

            # Persist extracted text so future reindex flows skip raw re-parse.
            try:
                processed_key = MinIOClient.processed_path(dataset_id, document_id, "extracted.txt")
                client.upload_bytes(
                    bucket=settings.minio_bucket_processed,
                    object_name=processed_key,
                    data=cleaned.encode("utf-8"),
                    content_type="text/plain; charset=utf-8",
                    metadata={"dataset_id": dataset_id, "document_id": document_id},
                )
                logger.info("Saved extracted text", document_id=document_id, object=processed_key)
            except Exception as exc:
                logger.warning(
                    "Failed to save extracted text",
                    document_id=document_id,
                    error=str(exc),
                )
        else:
            # Came from processed bucket — skip steps 1-3 progress markers
            await _update_progress(30)

        await _update_progress(40)

        # ── Step 4: Extract metadata ──────────────────────────────────────────
        doc_metadata = extract_metadata(
            filename=filename,
            file_size=file_size,
            parser_metadata=parsed_metadata,
            cleaned_text=cleaned,
            source_url=source_url,
        )
        await _update_progress(50)

        # ── Step 5: Chunk ─────────────────────────────────────────────────────
        chunker = get_chunker(chunk_strategy)
        chunk_results = chunker.chunk(cleaned)
        logger.info("Chunked document", document_id=document_id, chunks=len(chunk_results), strategy=chunk_strategy)
        await _update_progress(65)

        # ── Step 6: Embed + Index ─────────────────────────────────────────────
        emb = embedder or get_embedding_provider()
        vs = vector_store or get_vector_store()

        total = await index_chunks(
            session=session,
            vector_store=vs,
            embedder=emb,
            dataset_id=dataset_id,
            document_id=document_id,
            doc_metadata=doc_metadata,
            chunks=chunk_results,
        )
        await _update_progress(95)

        # ── Step 7: Mark complete ─────────────────────────────────────────────
        await doc_repo.set_status(document_id, "indexed")
        if job_repo and job_id:
            await job_repo.mark_completed(job_id, result={"chunks_indexed": total})
        await _update_progress(100)

        logger.info("Ingestion complete", document_id=document_id, chunks=total)
        return IngestionResult(
            document_id=document_id,
            dataset_id=dataset_id,
            chunks_indexed=total,
            success=True,
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Ingestion failed", document_id=document_id, error=error_msg)
        await doc_repo.set_status(document_id, "failed", error_message=error_msg)
        if job_repo and job_id:
            await job_repo.mark_failed(job_id, error_message=error_msg)
        return IngestionResult(
            document_id=document_id,
            dataset_id=dataset_id,
            chunks_indexed=0,
            success=False,
            error=error_msg,
        )
