"""
Indexer — T20.
Orchestrates the final step: persist chunks to PostgreSQL and vectors to the vector store.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories import ChunkRepository, DocumentRepository
from app.services.embedding.providers import EmbeddingProvider
from app.services.ingestion.chunkers import ChunkResult
from app.services.ingestion.metadata import build_chunk_metadata
from app.services.vector.base import VectorPoint, VectorStore

logger = get_logger(__name__)

COLLECTION_PREFIX = "pyrag_"


def collection_name(dataset_id: str) -> str:
    """Each dataset gets its own vector collection."""
    return f"{COLLECTION_PREFIX}{dataset_id.replace('-', '_')}"


async def ensure_collection(
    vector_store: VectorStore,
    dataset_id: str,
    dimensions: int,
) -> None:
    """Create vector collection for a dataset if it doesn't exist."""
    name = collection_name(dataset_id)
    if not await vector_store.collection_exists(name):
        await vector_store.create_collection(name, dimensions=dimensions)
        logger.info("Created vector collection", collection=name)


async def index_chunks(
    *,
    session: AsyncSession,
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    dataset_id: str,
    document_id: str,
    doc_metadata: dict,
    chunks: list[ChunkResult],
    batch_size: int = 50,
) -> int:
    """
    Embed chunks, write to vector store, write to PostgreSQL.
    Returns the number of chunks indexed.
    """
    if not chunks:
        return 0

    chunk_repo = ChunkRepository(session)
    doc_repo = DocumentRepository(session)
    col_name = collection_name(dataset_id)

    await ensure_collection(vector_store, dataset_id, embedder.dimensions)

    total_indexed = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]

        # Embed
        embeddings = await embedder.embed_texts(texts)

        # Build DB + vector records
        vector_points: list[VectorPoint] = []
        db_chunks: list[dict] = []

        for chunk, embedding in zip(batch, embeddings):
            vector_id = str(uuid.uuid4())
            chunk_meta = build_chunk_metadata(
                doc_metadata=doc_metadata,
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                page_hint=chunk.metadata.get("page"),
                section_hint=chunk.metadata.get("section"),
            )
            chunk_meta.update(chunk.metadata)  # merge strategy-specific metadata

            vector_points.append(
                VectorPoint(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        "dataset_id": dataset_id,
                        "document_id": document_id,
                        "chunk_text": chunk.text,
                        **chunk_meta,
                    },
                )
            )
            db_chunks.append({
                "dataset_id": dataset_id,
                "document_id": document_id,
                "chunk_text": chunk.text,
                "chunk_index": chunk.index,
                "token_count": len(chunk.text.split()) * 4 // 3,  # rough estimate
                "vector_reference": vector_id,
                "chunk_metadata": chunk_meta,
            })

        # Write to vector store
        await vector_store.upsert(col_name, vector_points)

        # Write to PostgreSQL
        await chunk_repo.bulk_create(db_chunks)

        total_indexed += len(batch)
        logger.debug(
            "Indexed batch",
            dataset_id=dataset_id,
            document_id=document_id,
            batch_size=len(batch),
            total=total_indexed,
        )

    # Update document chunk count
    await doc_repo.increment_chunk_count(document_id, total_indexed)

    logger.info(
        "Indexing complete",
        dataset_id=dataset_id,
        document_id=document_id,
        chunks=total_indexed,
    )
    return total_indexed


async def delete_document_vectors(
    vector_store: VectorStore,
    dataset_id: str,
    document_id: str,
) -> None:
    """Remove all vectors for a document (used during reindex or delete)."""
    col = collection_name(dataset_id)
    if await vector_store.collection_exists(col):
        await vector_store.delete_by_filter(col, {"document_id": document_id})
        logger.info("Deleted document vectors", document_id=document_id)
