from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.job import Job
from app.db.repositories.base import BaseRepository


class JobRepository(BaseRepository[Job]):
    model = Job

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_dataset(self, dataset_id: str) -> Sequence[Job]:
        result = await self.session.execute(
            select(Job)
            .where(Job.dataset_id == dataset_id)
            .order_by(Job.created_at.desc())
        )
        return result.scalars().all()

    async def list_pending(self) -> Sequence[Job]:
        result = await self.session.execute(
            select(Job)
            .where(Job.status.in_(["queued", "processing"]))
            .order_by(Job.created_at.asc())
        )
        return result.scalars().all()

    async def mark_started(self, id: str, celery_task_id: str) -> None:
        await self.session.execute(
            update(Job)
            .where(Job.id == id)
            .values(
                status="processing",
                celery_task_id=celery_task_id,
                started_at=datetime.now(timezone.utc),
            )
        )

    async def mark_completed(self, id: str, result: dict | None = None) -> None:
        await self.session.execute(
            update(Job)
            .where(Job.id == id)
            .values(
                status="completed",
                progress=100,
                result=result or {},
                completed_at=datetime.now(timezone.utc),
            )
        )

    async def mark_failed(self, id: str, error_message: str) -> None:
        await self.session.execute(
            update(Job)
            .where(Job.id == id)
            .values(
                status="failed",
                error_message=error_message,
                completed_at=datetime.now(timezone.utc),
            )
        )

    async def set_progress(self, id: str, progress: int) -> None:
        await self.session.execute(
            update(Job).where(Job.id == id).values(progress=min(max(progress, 0), 100))
        )
