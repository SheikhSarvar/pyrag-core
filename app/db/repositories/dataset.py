from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.dataset import Dataset
from app.db.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset]):
    model = Dataset

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_name(self, name: str) -> Dataset | None:
        result = await self.session.execute(
            select(Dataset).where(Dataset.name == name)
        )
        return result.scalar_one_or_none()

    async def list_active(self, offset: int = 0, limit: int = 50) -> list[Dataset]:
        result = await self.session.execute(
            select(Dataset)
            .where(Dataset.is_active == True)  # noqa: E712
            .order_by(Dataset.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())
