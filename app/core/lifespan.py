from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.session import engine

logger = get_logger(__name__)
settings = get_settings()


async def _startup(app: FastAPI) -> None:
    setup_logging()
    logger.info("Starting PyRAG Core", version=settings.version, env=settings.environment)

    # Verify DB connectivity
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
    except Exception as exc:
        logger.error("Database connection failed", error=str(exc))
        raise

    # Verify Redis connectivity
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=5)
        await r.ping()
        await r.aclose()
        logger.info("Redis connection verified")
    except Exception as exc:
        logger.warning("Redis connection failed — Celery tasks will not work", error=str(exc))

    # Init MinIO buckets
    try:
        from app.services.storage.minio_client import get_minio_client
        client = get_minio_client()
        for bucket in (settings.minio_bucket_raw, settings.minio_bucket_processed):
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("Created MinIO bucket", bucket=bucket)
        logger.info("MinIO buckets ready")
    except Exception as exc:
        logger.warning("MinIO init failed", error=str(exc))

    logger.info("PyRAG Core startup complete")


async def _shutdown(app: FastAPI) -> None:
    logger.info("Shutting down PyRAG Core")
    await engine.dispose()
    logger.info("Database connections closed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await _startup(app)
    yield
    await _shutdown(app)
