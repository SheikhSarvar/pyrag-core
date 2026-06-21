from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, PortableJSON, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.document import Document


class Chunk(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "chunks"

    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    vector_reference: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # ID in the vector store
    chunk_metadata: Mapped[dict] = mapped_column(
        PortableJSON, nullable=False, default=dict
    )  # page, section, headings, etc.

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_dataset_document", "dataset_id", "document_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id!r} doc={self.document_id!r} idx={self.chunk_index}>"
