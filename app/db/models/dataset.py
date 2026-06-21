from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.document import Document


class Dataset(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_strategy: Mapped[str] = mapped_column(
        String(50), nullable=False, default="recursive"
    )  # fixed | recursive | semantic | hierarchical
    embedding_model: Mapped[str] = mapped_column(
        String(100), nullable=False, default="text-embedding-3-small"
    )
    embedding_dimensions: Mapped[int] = mapped_column(nullable=False, default=1536)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id!r} name={self.name!r}>"
