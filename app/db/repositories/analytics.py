from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.analytics import Analytics
from app.db.repositories.base import BaseRepository


class AnalyticsRepository(BaseRepository[Analytics]):
    model = Analytics

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def total_cost(
        self,
        since: datetime | None = None,
        provider: str | None = None,
    ) -> float:
        stmt = select(func.coalesce(func.sum(Analytics.cost_usd), 0.0))
        if since:
            stmt = stmt.where(Analytics.created_at >= since)
        if provider:
            stmt = stmt.where(Analytics.provider == provider)
        result = await self.session.execute(stmt)
        return float(result.scalar_one())

    async def total_tokens(self, since: datetime | None = None) -> int:
        stmt = select(func.coalesce(func.sum(Analytics.total_tokens), 0))
        if since:
            stmt = stmt.where(Analytics.created_at >= since)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def cost_by_provider(self, since: datetime | None = None) -> list[dict]:
        stmt = (
            select(Analytics.provider, func.sum(Analytics.cost_usd).label("total_cost"))
            .group_by(Analytics.provider)
        )
        if since:
            stmt = stmt.where(Analytics.created_at >= since)
        result = await self.session.execute(stmt)
        return [{"provider": row.provider, "total_cost": float(row.total_cost)} for row in result]

    async def requests_by_type(self, since: datetime | None = None) -> list[dict]:
        stmt = (
            select(Analytics.request_type, func.count().label("count"))
            .group_by(Analytics.request_type)
        )
        if since:
            stmt = stmt.where(Analytics.created_at >= since)
        result = await self.session.execute(stmt)
        return [{"request_type": row.request_type, "count": row.count} for row in result]

    async def avg_latency_ms(self, request_type: str | None = None) -> float:
        stmt = select(func.avg(Analytics.latency_ms))
        if request_type:
            stmt = stmt.where(Analytics.request_type == request_type)
        result = await self.session.execute(stmt)
        return float(result.scalar_one() or 0.0)
