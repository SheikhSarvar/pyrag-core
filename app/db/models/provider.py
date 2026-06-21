from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, PortableJSON, TimestampMixin, UUIDMixin


class Provider(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "providers"

    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # openai | anthropic | gemini | openrouter | ollama | vllm
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="llm"
    )  # llm | embedding
    configuration: Mapped[dict] = mapped_column(
        PortableJSON, nullable=False, default=dict
    )  # base_url, api_key ref, max_tokens, temperature, etc.
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<Provider id={self.id!r} provider={self.provider!r} model={self.model!r}>"
