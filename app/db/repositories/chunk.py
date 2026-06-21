from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chunk import Chunk
from app.db.repositories.base import BaseRepository


class ChunkRepository(BaseRepository[Chunk]):
    model = Chunk

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_document(self, document_id: str) -> Sequence[Chunk]:
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
        )
        return result.scalars().all()

    async def list_by_dataset(
        self, dataset_id: str, offset: int = 0, limit: int = 100
    ) -> Sequence[Chunk]:
        result = await self.session.execute(
            select(Chunk)
            .where(Chunk.dataset_id == dataset_id)
            .order_by(Chunk.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def delete_by_document(self, document_id: str) -> int:
        result = await self.session.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        return result.rowcount  # type: ignore[return-value]

    async def bulk_create(self, chunks: list[dict]) -> list[Chunk]:
        objs = [Chunk(**c) for c in chunks]
        self.session.add_all(objs)
        await self.session.flush()
        return objs

    async def get_by_vector_reference(self, vector_reference: str) -> Chunk | None:
        result = await self.session.execute(
            select(Chunk).where(Chunk.vector_reference == vector_reference)
        )
        return result.scalar_one_or_none()
