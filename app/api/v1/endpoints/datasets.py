"""
Dataset API endpoints — T38.
POST   /datasets
GET    /datasets
GET    /datasets/{id}
DELETE /datasets/{id}
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.core.logging import get_logger
from app.db.session import get_db
from app.db.repositories import DatasetRepository
from app.schemas.dataset import DatasetCreate, DatasetListResponse, DatasetResponse, DatasetUpdate
from app.services.vector.factory import get_vector_store
from app.services.ingestion.indexer import collection_name

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    body: DatasetCreate,
    session: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    repo = DatasetRepository(session)

    existing = await repo.get_by_name(body.name)
    if existing:
        raise ConflictError(f"Dataset '{body.name}' already exists")

    dataset = await repo.create(**body.model_dump())

    # Pre-create the vector collection
    try:
        vs = get_vector_store()
        await vs.create_collection(
            collection_name=collection_name(dataset.id),
            dimensions=body.embedding_dimensions,
        )
    except Exception as exc:
        logger.warning("Vector collection pre-creation failed", error=str(exc))

    logger.info("Dataset created", dataset_id=dataset.id, name=dataset.name)
    return DatasetResponse.model_validate(dataset)


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    repo = DatasetRepository(session)
    items = await repo.list_active(offset=offset, limit=limit)
    total = await repo.count(filters={"is_active": True})
    return DatasetListResponse(
        items=[DatasetResponse.model_validate(d) for d in items],
        total=total,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    repo = DatasetRepository(session)
    dataset = await repo.get_or_raise(dataset_id)
    return DatasetResponse.model_validate(dataset)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    body: DatasetUpdate,
    session: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    repo = DatasetRepository(session)
    updated = await repo.update(dataset_id, **body.model_dump(exclude_none=True))
    return DatasetResponse.model_validate(updated)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_db),
) -> None:
    repo = DatasetRepository(session)
    dataset = await repo.get_or_raise(dataset_id)

    # Drop vector collection
    try:
        vs = get_vector_store()
        col = collection_name(dataset_id)
        if await vs.collection_exists(col):
            await vs.delete_collection(col)
    except Exception as exc:
        logger.warning("Vector collection deletion failed", error=str(exc))

    await repo.delete(dataset_id)
    logger.info("Dataset deleted", dataset_id=dataset_id, name=dataset.name)
