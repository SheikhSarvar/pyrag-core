from __future__ import annotations

import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(LLMProvider):
    """
    OpenRouter exposes an OpenAI-compatible API.
    Supports any model available on openrouter.ai via a single key.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.openrouter_default_model
        if not self._model:
            raise LLMError("OPENROUTER_DEFAULT_MODEL must be set")
        _key = api_key or settings.openrouter_api_key
        if not _key:
            raise LLMError("OPENROUTER_API_KEY not configured")
        self._client = AsyncOpenAI(
            api_key=_key,
            base_url=_OPENROUTER_BASE,
            default_headers={
                "HTTP-Referer": "https://github.com/pyrag-core",
                "X-Title": "PyRAG Core",
            },
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_name(self) -> str:
        return self._model

    def _to_oai(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        start = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=self._to_oai(messages),  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        except Exception as exc:
            raise LLMError(f"OpenRouter completion failed: {exc}") from exc

        latency = int((time.monotonic() - start) * 1000)
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=self._model,
            provider="openrouter",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,   # OpenRouter bills at model-level — cost lookup needs model-specific rates
            latency_ms=latency,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with await self._client.chat.completions.create(
                model=self._model,
                messages=self._to_oai(messages),  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    finish = chunk.choices[0].finish_reason
                    yield StreamChunk(delta=delta, finish_reason=finish)
        except Exception as exc:
            raise LLMError(f"OpenRouter stream failed: {exc}") from exc
