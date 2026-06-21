"""
Rate limiting — T55.
Sliding-window counter backed by Redis. Falls back to allowing all
requests if Redis is unavailable (fail-open, logged as a warning).
"""
from __future__ import annotations

import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)

_SKIP_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Fixed-window rate limiter: N requests per 60-second window per client key.
    Client key = API key header if present, else client IP.
    """

    def __init__(self, app: ASGIApp, *, requests_per_minute: int | None = None) -> None:
        super().__init__(app)
        settings = get_settings()
        self._limit = requests_per_minute or settings.rate_limit_per_minute
        self._redis: object | None = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                settings = get_settings()
                self._redis = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
            except Exception as exc:
                logger.warning("Rate limiter could not connect to Redis", error=str(exc))
        return self._redis

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get(get_settings().api_key_header)
        if api_key:
            return f"ratelimit:key:{api_key[:16]}"
        ip = request.client.host if request.client else "unknown"
        return f"ratelimit:ip:{ip}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        redis = await self._get_redis()
        if redis is None:
            # Fail-open: Redis unavailable, allow request but log it
            return await call_next(request)

        key = self._client_key(request)
        window = int(time.time() // 60)
        bucket_key = f"{key}:{window}"

        try:
            count = await redis.incr(bucket_key)
            if count == 1:
                await redis.expire(bucket_key, 65)  # slightly over 60s to avoid edge race

            if count > self._limit:
                remaining_ttl = await redis.ttl(bucket_key)
                logger.warning("Rate limit exceeded", key=key, count=count, limit=self._limit)
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "limit": self._limit,
                        "retry_after_seconds": max(remaining_ttl, 1),
                    },
                    headers={"Retry-After": str(max(remaining_ttl, 1))},
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self._limit)
            response.headers["X-RateLimit-Remaining"] = str(max(self._limit - count, 0))
            return response

        except Exception as exc:
            logger.warning("Rate limiter error, failing open", error=str(exc))
            return await call_next(request)
