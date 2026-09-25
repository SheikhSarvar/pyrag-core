from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChunkDetailResponse(BaseModel):
    id: str
    dataset_id: str
    document_id: str
    document_name: str = ""
    chunk_index: int
    chunk_text: str
    token_count: int
    vector_reference: str | None
    chunk_metadata: dict
    word_count: int = 0
    char_count: int = 0
    approx_token_count: int = 0
    quality_flags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChunkListResponse(BaseModel):
    items: list[ChunkDetailResponse]
    total: int
    offset: int
    limit: int

