from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str
    dataset_id: str
    filename: str
    file_size: int
    status: str
    message: str


class DocumentResponse(BaseModel):
    id: str
    dataset_id: str
    name: str
    original_name: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    offset: int
    limit: int


class ReindexRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1)


class JobStatusResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: int
    dataset_id: str | None
    document_id: str | None
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}
