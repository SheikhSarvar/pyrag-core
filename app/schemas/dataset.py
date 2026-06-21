from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    chunk_strategy: Literal["fixed", "recursive", "semantic", "hierarchical"] = "recursive"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=64, le=4096)


class DatasetUpdate(BaseModel):
    description: str | None = None
    chunk_strategy: Literal["fixed", "recursive", "semantic", "hierarchical"] | None = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    description: str | None
    chunk_strategy: str
    embedding_model: str
    embedding_dimensions: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse]
    total: int
