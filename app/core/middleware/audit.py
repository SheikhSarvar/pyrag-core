"""
Audit logging middleware — T53.
Logs every request with method, path, status, latency, request_id.
Structured JSON in production; human-readable in dev.
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.core.logging import get_logger

logger = get_logger(__name__)

# Paths excluded from audit logs (health checks, metrics)
_SKIP_PATHS = frozenset({"/health", "/metrics", "/favicon.ico"})


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, log_body: bool = False) -> None:
        super().__init__(app)
        self._log_body = log_body

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.monotonic()

        # Attach request_id to the request state for downstream use
        request.state.request_id = request_id

        response: Response | None = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            logger.error(
                "Unhandled exception",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
                exc_info=True,
            )
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )
            status_code = 500
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            log_fn = logger.warning if status_code >= 400 else logger.info
            log_fn(
                "Request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                query=str(request.url.query) or None,
                status=status_code,
                latency_ms=latency_ms,
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent", "")[:100],
            )

        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Latency-Ms"] = str(latency_ms)
        return response  # type: ignore[return-value]
