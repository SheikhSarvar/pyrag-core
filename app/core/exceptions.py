from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class PyRAGError(Exception):
    """Base exception for all PyRAG errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None, **context: Any) -> None:
        self.detail = detail or self.__class__.detail
        self.context = context
        super().__init__(self.detail)


class NotFoundError(PyRAGError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"


class ConflictError(PyRAGError):
    status_code = status.HTTP_409_CONFLICT
    detail = "Resource already exists"


class ValidationError(PyRAGError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Validation error"


class AuthenticationError(PyRAGError):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Authentication required"


class AuthorizationError(PyRAGError):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Permission denied"


class RateLimitError(PyRAGError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    detail = "Rate limit exceeded"


class StorageError(PyRAGError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Storage operation failed"


class IngestionError(PyRAGError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Document ingestion failed"


class RetrievalError(PyRAGError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Retrieval operation failed"


class LLMError(PyRAGError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "LLM provider error"


class EmbeddingError(PyRAGError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Embedding generation failed"


class VectorStoreError(PyRAGError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Vector store operation failed"


class FileTooLargeError(PyRAGError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    detail = "File exceeds maximum allowed size"


class UnsupportedFileTypeError(PyRAGError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    detail = "File type not supported"


# ── FastAPI Exception Handlers ────────────────────────────────────────────────

def _error_response(status_code: int, detail: str, **extra: Any) -> JSONResponse:
    content: dict[str, Any] = {"error": detail}
    if extra:
        content["context"] = extra
    return JSONResponse(status_code=status_code, content=content)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PyRAGError)
    async def pyrag_error_handler(request: Request, exc: PyRAGError) -> JSONResponse:
        return _error_response(exc.status_code, exc.detail, **exc.context)

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "Endpoint not found")

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(status.HTTP_405_METHOD_NOT_ALLOWED, "Method not allowed")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Catches anything not already handled above — Starlette does NOT route
        # arbitrary exceptions through exception_handler(500); only this
        # Exception-class handler intercepts them before they crash the worker.
        import structlog
        structlog.get_logger(__name__).error(
            "Unhandled exception", path=str(request.url.path), error=str(exc), exc_info=True
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error"
        )
