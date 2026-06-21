from __future__ import annotations

import time
from typing import AsyncIterator

import anthropic

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    calculate_cost,
)


class AnthropicAdapter(LLMProvider):

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.anthropic_default_model
        _key = api_key or settings.anthropic_api_key
        if not _key:
            raise LLMError("ANTHROPIC_API_KEY not configured")
        self._client = anthropic.AsyncAnthropic(api_key=_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, list[dict]]:
        """Anthropic separates system prompt from the message list."""
        system = ""
        chat: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat.append({"role": m.role, "content": m.content})
        return system, chat

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        system, chat = self._split_messages(messages)
        start = time.monotonic()
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=chat,
                stop_sequences=stop or [],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic completion failed: {exc}") from exc

        latency = int((time.monotonic() - start) * 1000)
        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens

        return LLMResponse(
            content=resp.content[0].text if resp.content else "",
            model=self._model,
            provider="anthropic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(self._model, prompt_tokens, completion_tokens),
            latency_ms=latency,
            finish_reason=resp.stop_reason or "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[StreamChunk]:
        system, chat = self._split_messages(messages)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=chat,
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)
                final = await stream.get_final_message()
                yield StreamChunk(delta="", finish_reason=final.stop_reason or "stop")
        except Exception as exc:
            raise LLMError(f"Anthropic stream failed: {exc}") from exc
