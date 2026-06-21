from typing import Any, Generic, TypeVar
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic async CRUD repository. Subclass and set `model`."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, id: str) -> ModelT | None:
        return await self.session.get(self.model, id)

    async def get_or_raise(self, id: str) -> ModelT:
        from app.core.exceptions import NotFoundError
        obj = await self.get(id)
        if obj is None:
            raise NotFoundError(f"{self.model.__name__} '{id}' not found")
        return obj

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> Sequence[ModelT]:
        stmt = select(self.model)
        if filters:
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
        col = getattr(self.model, order_by, None)
        if col is not None:
            stmt = stmt.order_by(col.desc() if descending else col.asc())
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        stmt = select(func.count()).select_from(self.model)
        if filters:
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def create(self, **kwargs: Any) -> ModelT:
        obj = self.model(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def update(self, id: str, **kwargs: Any) -> ModelT:
        obj = await self.get_or_raise(id)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def delete(self, id: str) -> bool:
        obj = await self.get(id)
        if obj is None:
            return False
        await self.session.delete(obj)
        await self.session.flush()
        return True

    async def exists(self, id: str) -> bool:
        return await self.get(id) is not None
