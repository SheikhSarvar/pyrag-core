from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.db.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_dataset(
        self, dataset_id: str, offset: int = 0, limit: int = 50
    ) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.dataset_id == dataset_id)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()

    async def set_status(
        self, id: str, status: str, error_message: str | None = None
    ) -> None:
        await self.session.execute(
            update(Document)
            .where(Document.id == id)
            .values(status=status, error_message=error_message)
        )

    async def increment_chunk_count(self, id: str, count: int) -> None:
        doc = await self.get_or_raise(id)
        doc.chunk_count += count
        await self.session.flush()

    async def list_by_status(self, status: str) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document).where(Document.status == status)
        )
        return result.scalars().all()
