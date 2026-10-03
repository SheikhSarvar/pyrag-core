"""
Document API endpoints — T12 (upload + job creation) + T39.
POST   /documents/upload
GET    /documents
DELETE /documents/{id}
POST   /documents/reindex
GET    /documents/{id}/status
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, NotFoundError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.core.validation import sanitize_filename
from app.db.session import get_db
from app.db.repositories import ChunkRepository, DatasetRepository, DocumentRepository, JobRepository
from app.schemas.chunk import ChunkDetailResponse, ChunkListResponse
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    JobStatusResponse,
    ReindexRequest,
)
from app.services.ingestion.parsers import SUPPORTED_EXTENSIONS
from app.services.storage.minio_client import MinIOClient, get_minio_client
from app.services.retrieval.cache import bump_dataset_cache_version

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


def _chunk_quality_flags(chunk, source_url: str | None) -> list[str]:
    flags: list[str] = []
    text = chunk.chunk_text or ""
    metadata = chunk.chunk_metadata or {}

    if len(text.strip()) < 50:
        flags.append("very_short")
    if len(text) > 6000:
        flags.append("very_long")
    if not metadata.get("filename"):
        flags.append("missing_filename")
    if not metadata.get("document_title"):
        flags.append("missing_document_title")
    if not metadata.get("source_url") and not source_url:
        flags.append("missing_source_url")
    if chunk.token_count <= 0:
        flags.append("missing_token_count")
    return flags


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    minio: MinIOClient = Depends(get_minio_client),
) -> DocumentUploadResponse:
    """Upload a document file to a dataset and queue it for ingestion."""

    # Validate dataset exists
    ds_repo = DatasetRepository(session)
    dataset = await ds_repo.get_or_raise(dataset_id)

    # Validate file type
    filename = sanitize_filename(file.filename or "upload.bin")
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"File type '.{ext}' is not supported")

    # Validate file size (avoid buffering large uploads in memory)
    file_size: int
    file_data: bytes | None = None
    try:
        file.file.seek(0, io.SEEK_END)
        file_size = int(file.file.tell())
        file.file.seek(0)
    except Exception:
        file_data = await file.read()
        file_size = len(file_data)

    if file_size > settings.max_upload_size_bytes:
        raise FileTooLargeError(f"File exceeds {settings.max_upload_size_mb}MB limit")

    # Generate IDs
    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    storage_path = MinIOClient.raw_path(dataset_id, document_id, filename)

    # Upload to MinIO
    if file_data is not None:
        minio.upload_bytes(
            bucket=settings.minio_bucket_raw,
            object_name=storage_path,
            data=file_data,
            content_type=file.content_type or "application/octet-stream",
            metadata={"dataset_id": dataset_id, "document_id": document_id},
        )
    else:
        file.file.seek(0)
        minio.upload_file(
            bucket=settings.minio_bucket_raw,
            object_name=storage_path,
            data=file.file,
            length=file_size,
            content_type=file.content_type or "application/octet-stream",
            metadata={"dataset_id": dataset_id, "document_id": document_id},
        )

    # Create DB records
    doc_repo = DocumentRepository(session)
    job_repo = JobRepository(session)

    await doc_repo.create(
        id=document_id,
        dataset_id=dataset_id,
        name=filename,
        original_name=filename,
        file_type=ext,
        file_size=file_size,
        storage_path=storage_path,
        status="pending",
    )

    await job_repo.create(
        id=job_id,
        job_type="ingest",
        dataset_id=dataset_id,
        document_id=document_id,
        payload={
            "filename": filename,
            "file_size": file_size,
            "storage_path": storage_path,
            "chunk_strategy": dataset.chunk_strategy,
        },
    )

    # Persist the records before handing work to Celery so the worker can read
    # them immediately even if the task starts before the request finishes.
    await session.commit()

    # Dispatch Celery task
    from app.worker.tasks.ingestion import ingest_document

    try:
        task = ingest_document.apply_async(
            kwargs={
                "dataset_id": dataset_id,
                "document_id": document_id,
                "filename": filename,
                "file_size": file_size,
                "storage_path": storage_path,
                "chunk_strategy": dataset.chunk_strategy,
                "job_id": job_id,
            },
            task_id=job_id,
        )
    except Exception as exc:
        await doc_repo.set_status(document_id, "failed", error_message=str(exc))
        await job_repo.mark_failed(job_id, error_message=str(exc))
        await session.commit()
        raise

    await job_repo.mark_started(job_id, celery_task_id=task.id)
    await session.commit()

    logger.info("Document upload queued", document_id=document_id, job_id=job_id)

    return DocumentUploadResponse(
        document_id=document_id,
        job_id=job_id,
        dataset_id=dataset_id,
        filename=filename,
        file_size=file_size,  # file_data may be None on the streaming path
        status="pending",
        message="Document uploaded and queued for ingestion",
    )


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    dataset_id: str = Query(...),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    repo = DocumentRepository(session)
    docs = await repo.list_by_dataset(dataset_id, offset=offset, limit=limit)
    total = await repo.count(filters={"dataset_id": dataset_id})
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        offset=offset,
        limit=limit,
    )


# ── Get status ────────────────────────────────────────────────────────────────

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    repo = DocumentRepository(session)
    doc = await repo.get_or_raise(document_id)
    return DocumentResponse.model_validate(doc)


@router.get("/{document_id}/chunks", response_model=ChunkListResponse)
async def list_document_chunks(
    document_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
) -> ChunkListResponse:
    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)

    doc = await doc_repo.get_or_raise(document_id)
    chunks = await chunk_repo.list_by_document(document_id)
    total = len(chunks)
    page = chunks[offset : offset + limit]

    items: list[ChunkDetailResponse] = []
    for chunk in page:
        items.append(
            ChunkDetailResponse(
                id=chunk.id,
                dataset_id=chunk.dataset_id,
                document_id=chunk.document_id,
                document_name=doc.original_name,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                vector_reference=chunk.vector_reference,
                chunk_metadata=chunk.chunk_metadata,
                word_count=len(chunk.chunk_text.split()),
                char_count=len(chunk.chunk_text),
                approx_token_count=max(1, len(chunk.chunk_text) // 4),
                quality_flags=_chunk_quality_flags(chunk, doc.source_url),
                created_at=chunk.created_at,
                updated_at=chunk.updated_at,
            )
        )

    return ChunkListResponse(items=items, total=total, offset=offset, limit=limit)


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_db),
    minio: MinIOClient = Depends(get_minio_client),
) -> None:
    doc_repo = DocumentRepository(session)
    doc = await doc_repo.get_or_raise(document_id)

    # Remove from MinIO
    if doc.storage_path:
        try:
            minio.delete_object(settings.minio_bucket_raw, doc.storage_path)
        except Exception:
            pass  # Best-effort — don't fail delete if storage removal fails

    # Remove vectors
    try:
        from app.services.vector.factory import get_vector_store
        from app.services.ingestion.indexer import delete_document_vectors
        vs = get_vector_store()
        await delete_document_vectors(vs, doc.dataset_id, document_id)
    except Exception:
        pass

    await doc_repo.delete(document_id)
    await bump_dataset_cache_version(doc.dataset_id)


# ── Reindex ───────────────────────────────────────────────────────────────────

@router.post("/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_documents(
    body: ReindexRequest,
    session: AsyncSession = Depends(get_db),
) -> dict:
    doc_repo = DocumentRepository(session)
    job_repo = JobRepository(session)
    queued = []

    # Cache dataset lookups to avoid one DB query per document when many
    # documents in the batch share the same dataset.
    from app.db.repositories import DatasetRepository
    ds_repo = DatasetRepository(session)
    _dataset_cache: dict[str, object] = {}

    for doc_id in body.document_ids:
        doc = await doc_repo.get(doc_id)
        if doc is None:
            continue

        if doc.dataset_id not in _dataset_cache:
            _dataset_cache[doc.dataset_id] = await ds_repo.get(doc.dataset_id)
        dataset = _dataset_cache[doc.dataset_id]
        chunk_strategy = dataset.chunk_strategy if dataset else "recursive"

        job_id = str(uuid.uuid4())
        await job_repo.create(
            id=job_id,
            job_type="reindex",
            dataset_id=doc.dataset_id,
            document_id=doc_id,
            payload={"storage_path": doc.storage_path, "chunk_strategy": chunk_strategy},
        )
        await session.commit()

        from app.worker.tasks.ingestion import ingest_document
        task = ingest_document.apply_async(
            kwargs={
                "dataset_id": doc.dataset_id,
                "document_id": doc_id,
                "filename": doc.original_name,
                "file_size": doc.file_size,
                "storage_path": doc.storage_path or "",
                "chunk_strategy": chunk_strategy,
                "job_id": job_id,
                "reindex": True,  # signals pipeline to skip raw parse if processed text exists
            },
            task_id=job_id,
        )
        await job_repo.mark_started(job_id, celery_task_id=task.id)
        await session.commit()
        await bump_dataset_cache_version(doc.dataset_id)
        queued.append({"document_id": doc_id, "job_id": job_id})

    return {"queued": queued, "total": len(queued)}


# ── Job status ────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    session: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    repo = JobRepository(session)
    job = await repo.get_or_raise(job_id)
    return JobStatusResponse.model_validate(job)
