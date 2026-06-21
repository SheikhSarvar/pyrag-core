"""
Celery ingestion tasks — T13.
The task is thin: it sets up the async context and delegates to pipeline.py.
"""
from __future__ import annotations

import asyncio

from celery import Task
from celery.utils.log import get_task_logger

from app.worker.celery_app import celery_app

logger = get_task_logger(__name__)


class IngestionTask(Task):
    """Base class with retry config for ingestion tasks."""
    abstract = True
    max_retries = 3
    default_retry_delay = 30  # seconds


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="app.worker.tasks.ingestion.ingest_document",
    queue="ingestion",
)
def ingest_document(
    self: IngestionTask,
    *,
    dataset_id: str,
    document_id: str,
    filename: str,
    file_size: int,
    storage_path: str,
    chunk_strategy: str = "recursive",
    source_url: str | None = None,
    job_id: str | None = None,
) -> dict:
    """
    Celery task: run the full ingestion pipeline for one document.
    Returns a result dict with success flag and chunk count.
    """
    logger.info(
        "Starting ingestion task",
        extra={"document_id": document_id, "dataset_id": dataset_id},
    )

    async def _run() -> dict:
        from app.db.session import AsyncSessionLocal
        from app.services.ingestion.pipeline import run_ingestion_pipeline

        async with AsyncSessionLocal() as session:
            result = await run_ingestion_pipeline(
                session=session,
                dataset_id=dataset_id,
                document_id=document_id,
                filename=filename,
                file_size=file_size,
                storage_path=storage_path,
                chunk_strategy=chunk_strategy,
                source_url=source_url,
                job_id=job_id,
            )
            await session.commit()
            return {
                "success": result.success,
                "chunks_indexed": result.chunks_indexed,
                "error": result.error,
                "document_id": result.document_id,
            }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.exception("Ingestion task failed, retrying", exc_info=exc)
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="app.worker.tasks.ingestion.ingest_url",
    queue="ingestion",
)
def ingest_url(
    self: IngestionTask,
    *,
    dataset_id: str,
    document_id: str,
    url: str,
    job_id: str | None = None,
) -> dict:
    """Celery task: scrape a URL and ingest as a document."""

    async def _run() -> dict:
        from app.db.session import AsyncSessionLocal
        from app.db.repositories import DocumentRepository
        from app.services.ingestion.parsers import WebScraper
        from app.services.ingestion.cleaner import clean_text
        from app.services.ingestion.metadata import extract_metadata
        from app.services.ingestion.chunkers import get_chunker
        from app.services.ingestion.indexer import index_chunks
        from app.services.embedding.providers import get_embedding_provider
        from app.services.vector.factory import get_vector_store

        async with AsyncSessionLocal() as session:
            doc_repo = DocumentRepository(session)
            try:
                await doc_repo.set_status(document_id, "processing")

                scraper = WebScraper()
                parsed = scraper.scrape(url)
                cleaned = clean_text(parsed.text)

                doc_metadata = extract_metadata(
                    filename=url,
                    file_size=len(cleaned.encode()),
                    parser_metadata=parsed.metadata,
                    cleaned_text=cleaned,
                    source_url=url,
                )
                chunker = get_chunker("recursive")
                chunks = chunker.chunk(cleaned)

                total = await index_chunks(
                    session=session,
                    vector_store=get_vector_store(),
                    embedder=get_embedding_provider(),
                    dataset_id=dataset_id,
                    document_id=document_id,
                    doc_metadata=doc_metadata,
                    chunks=chunks,
                )
                await doc_repo.set_status(document_id, "indexed")
                await session.commit()
                return {"success": True, "chunks_indexed": total, "document_id": document_id}

            except Exception as exc:
                await doc_repo.set_status(document_id, "failed", error_message=str(exc))
                await session.commit()
                raise

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
