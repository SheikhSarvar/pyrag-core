from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDMixin


class Analytics(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "analytics"

    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    request_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # search | chat | agent | embed
    dataset_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="success"
    )  # success | error

    __table_args__ = (
        Index("ix_analytics_provider_created", "provider", "created_at"),
        Index("ix_analytics_dataset_created", "dataset_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Analytics id={self.id!r} type={self.request_type!r} cost=${self.cost_usd:.6f}>"
