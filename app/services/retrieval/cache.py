from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis_client: Any | None = None
_redis_unavailable = False


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    return value


def build_search_cache_key(
    *,
    dataset_id: str,
    query: str,
    mode: str,
    top_k: int,
    rerank: bool,
    expand_query: bool,
    score_threshold: float | None,
    filters: dict | None,
    cache_version: int,
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "query": query.strip().lower(),
        "mode": mode,
        "top_k": top_k,
        "rerank": rerank,
        "expand_query": expand_query,
        "score_threshold": score_threshold,
        "filters": _normalize(filters or {}),
        "version": cache_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"search:{dataset_id}:{digest}"


async def get_redis_client():
    global _redis_client, _redis_unavailable
    if _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        return _redis_client
    except Exception as exc:
        _redis_unavailable = True
        logger.warning("Redis cache unavailable", error=str(exc))
        return None


async def get_dataset_cache_version(dataset_id: str) -> int:
    redis = await get_redis_client()
    if redis is None:
        return 1

    key = f"search:dataset:{dataset_id}:version"
    try:
        version = await redis.get(key)
        if version is None:
            await redis.set(key, "1")
            return 1
        return int(version)
    except Exception as exc:
        logger.warning("Failed to read cache version", dataset_id=dataset_id, error=str(exc))
        return 1


async def bump_dataset_cache_version(dataset_id: str) -> None:
    redis = await get_redis_client()
    if redis is None:
        return

    key = f"search:dataset:{dataset_id}:version"
    try:
        await redis.incr(key)
    except Exception as exc:
        logger.warning("Failed to bump cache version", dataset_id=dataset_id, error=str(exc))


async def get_cached_search(cache_key: str) -> dict[str, Any] | None:
    redis = await get_redis_client()
    if redis is None:
        return None

    try:
        raw = await redis.get(cache_key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to read search cache", cache_key=cache_key, error=str(exc))
        return None


async def set_cached_search(cache_key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
    redis = await get_redis_client()
    if redis is None:
        return

    try:
        await redis.set(cache_key, json.dumps(payload), ex=ttl_seconds)
    except Exception as exc:
        logger.warning("Failed to write search cache", cache_key=cache_key, error=str(exc))
