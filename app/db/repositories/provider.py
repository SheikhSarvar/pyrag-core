from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.provider import Provider
from app.db.repositories.base import BaseRepository


class ProviderRepository(BaseRepository[Provider]):
    model = Provider

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_default(self, provider_type: str = "llm") -> Provider | None:
        result = await self.session.execute(
            select(Provider)
            .where(Provider.provider_type == provider_type)
            .where(Provider.is_default == True)  # noqa: E712
            .where(Provider.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()

    async def list_active(self, provider_type: str | None = None) -> Sequence[Provider]:
        stmt = select(Provider).where(Provider.is_active == True)  # noqa: E712
        if provider_type:
            stmt = stmt.where(Provider.provider_type == provider_type)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def set_default(self, id: str, provider_type: str) -> Provider:
        # Clear existing default for this type
        await self.session.execute(
            update(Provider)
            .where(Provider.provider_type == provider_type)
            .values(is_default=False)
        )
        return await self.update(id, is_default=True)
