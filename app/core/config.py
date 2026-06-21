from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    secret_key: str = Field(min_length=32)
    api_v1_prefix: str = "/api/v1"
    project_name: str = "PyRAG Core"
    version: str = "1.0.0"

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_pre_ping: bool = True

    @computed_field  # type: ignore[misc]
    @property
    def sync_database_url(self) -> str:
        """psycopg2 URL for Alembic (sync)."""
        return self.database_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── MinIO ──────────────────────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw: str = "pyrag-raw"
    minio_bucket_processed: str = "pyrag-processed"
    minio_use_ssl: bool = False

    # ── Vector Store ───────────────────────────────────────────────────────────
    vector_provider: Literal["qdrant", "weaviate", "milvus", "pgvector", "elasticsearch"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # ── LLM Providers ──────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-sonnet-4-6"

    gemini_api_key: str = ""
    gemini_default_model: str = "gemini-1.5-flash"

    openrouter_api_key: str = ""
    openrouter_default_model: str = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"

    vllm_base_url: str = "http://localhost:8000"
    vllm_default_model: str = ""

    # ── Embedding ──────────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "sentence-transformers", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Observability ──────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    @computed_field  # type: ignore[misc]
    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    # ── Security ───────────────────────────────────────────────────────────────
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 60

    # ── Performance ────────────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    search_top_k: int = 10
    rerank_top_k: int = 5

    @computed_field  # type: ignore[misc]
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # ── CORS ───────────────────────────────────────────────────────────────────
    cors_origins: list[AnyHttpUrl | Literal["*"]] = ["*"]

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.is_production:
            if self.secret_key == "dev-secret-change-in-prod-32chars":
                raise ValueError("SECRET_KEY must be changed in production")
            if "*" in [str(o) for o in self.cors_origins]:
                raise ValueError("CORS_ORIGINS cannot be wildcard in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
