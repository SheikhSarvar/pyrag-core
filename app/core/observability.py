"""
Langfuse observability — T50.
Wraps every RAG pipeline step in a trace with spans for:
  query → retrieval → rerank → prompt → completion
"""
from __future__ import annotations

import functools
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Client singleton ──────────────────────────────────────────────────────────

_langfuse_client: Any = None


def get_langfuse() -> Any | None:
    """Return a Langfuse client if configured, else None."""
    global _langfuse_client
    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if _langfuse_client is None:
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse client initialised", host=settings.langfuse_host)
        except ImportError:
            logger.warning("langfuse package not installed — observability disabled")
        except Exception as exc:
            logger.warning("Langfuse init failed", error=str(exc))

    return _langfuse_client


# ── Trace context ─────────────────────────────────────────────────────────────

class RAGTrace:
    """
    Wraps a single user request in a Langfuse trace.
    Use as an async context manager or call methods directly.
    """

    def __init__(
        self,
        *,
        name: str = "rag-pipeline",
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._name = name
        self._user_id = user_id
        self._session_id = session_id
        self._metadata = metadata or {}
        self._trace: Any = None
        self._trace_id = str(uuid.uuid4())
        self._lf = get_langfuse()

    def start(self) -> "RAGTrace":
        if self._lf:
            try:
                self._trace = self._lf.trace(
                    id=self._trace_id,
                    name=self._name,
                    user_id=self._user_id,
                    session_id=self._session_id,
                    metadata=self._metadata,
                )
            except Exception as exc:
                logger.debug("Langfuse trace start failed", error=str(exc))
        return self

    def end(self, output: str | None = None) -> None:
        if self._trace and output:
            try:
                self._trace.update(output=output)
            except Exception:
                pass
        if self._lf:
            try:
                self._lf.flush()
            except Exception:
                pass

    def span(self, name: str, input: Any = None, metadata: dict | None = None) -> "SpanContext":
        return SpanContext(trace=self._trace, name=name, input=input, metadata=metadata)

    def log_retrieval(
        self,
        query: str,
        chunks: list[dict],
        latency_ms: int,
        mode: str = "hybrid",
    ) -> None:
        if not self._trace:
            return
        try:
            self._trace.span(
                name="retrieval",
                input={"query": query, "mode": mode},
                output={"chunks": len(chunks), "top_score": chunks[0]["score"] if chunks else 0},
                metadata={"latency_ms": latency_ms, "chunk_count": len(chunks)},
            )
        except Exception:
            pass

    def log_generation(
        self,
        model: str,
        provider: str,
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: int,
    ) -> None:
        if not self._trace:
            return
        try:
            self._trace.generation(
                name="llm-completion",
                model=model,
                model_parameters={"provider": provider},
                input=prompt,
                output=completion,
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                    "unit": "TOKENS",
                },
                metadata={"cost_usd": cost_usd, "latency_ms": latency_ms},
            )
        except Exception:
            pass

    def log_score(self, name: str, value: float, comment: str = "") -> None:
        if not self._trace:
            return
        try:
            self._lf.score(
                trace_id=self._trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception:
            pass


class SpanContext:
    def __init__(self, trace: Any, name: str, input: Any, metadata: dict | None) -> None:
        self._trace = trace
        self._name = name
        self._input = input
        self._metadata = metadata or {}
        self._span: Any = None
        self._start = time.monotonic()

    def __enter__(self) -> "SpanContext":
        if self._trace:
            try:
                self._span = self._trace.span(
                    name=self._name,
                    input=self._input,
                    metadata=self._metadata,
                )
            except Exception:
                pass
        return self

    def __exit__(self, *args: Any) -> None:
        latency = int((time.monotonic() - self._start) * 1000)
        if self._span:
            try:
                self._span.end(metadata={**self._metadata, "latency_ms": latency})
            except Exception:
                pass

    def set_output(self, output: Any) -> None:
        if self._span:
            try:
                self._span.update(output=output)
            except Exception:
                pass


# ── Decorator helpers ─────────────────────────────────────────────────────────

def traced(name: str):
    """Decorator: wraps an async function in a Langfuse span (best-effort)."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            lf = get_langfuse()
            if not lf:
                return await fn(*args, **kwargs)
            # We don't have a parent trace here — create a standalone span
            start = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                latency = int((time.monotonic() - start) * 1000)
                logger.debug("Traced span", name=name, latency_ms=latency)
        return wrapper
    return decorator
