from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, PortableJSON, TimestampMixin, UUIDMixin


class Job(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "jobs"

    job_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # ingest | reindex | embed
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="queued", index=True
    )  # queued | processing | completed | failed | cancelled
    dataset_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(PortableJSON, nullable=False, default=dict)
    result: Mapped[dict] = mapped_column(PortableJSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(nullable=False, default=0)  # 0–100
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_jobs_status_type", "status", "job_type"),
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} type={self.job_type!r} status={self.status!r}>"
