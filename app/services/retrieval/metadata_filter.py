from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chunk import Chunk


def _matches(chunk: Chunk, filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        actual: object | None
        if key == "chunk_id":
            actual = chunk.id
        elif key == "document_id":
            actual = chunk.document_id
        elif key == "dataset_id":
            actual = chunk.dataset_id
        elif key in {"filename", "document_name", "original_name"}:
            actual = getattr(chunk.document, "original_name", None) if chunk.document else None
            if actual is None:
                actual = chunk.chunk_metadata.get("filename") or chunk.chunk_metadata.get("document_name")
        elif key == "source_url":
            actual = getattr(chunk.document, "source_url", None) if chunk.document else None
            if actual is None:
                actual = chunk.chunk_metadata.get("source_url")
        elif key == "document_title":
            actual = chunk.chunk_metadata.get("document_title") or getattr(chunk.document, "name", None)
        else:
            actual = chunk.chunk_metadata.get(key)

        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue

        if actual != expected:
            return False
    return True


async def resolve_metadata_chunk_ids(
    session: AsyncSession,
    dataset_id: str,
    filters: dict[str, object] | None,
) -> list[str]:
    """
    Apply metadata filters against PostgreSQL first, then use the matching
    chunk IDs to constrain vector search.
    """
    if not filters:
        return []

    stmt = (
        select(Chunk)
        .options(joinedload(Chunk.document))
        .where(Chunk.dataset_id == dataset_id)
        .order_by(Chunk.chunk_index.asc())
    )
    result = await session.execute(stmt)
    chunks: Sequence[Chunk] = result.unique().scalars().all()

    return [chunk.id for chunk in chunks if _matches(chunk, filters)]
