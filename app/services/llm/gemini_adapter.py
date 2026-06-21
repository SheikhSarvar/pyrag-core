from __future__ import annotations

import time
from typing import AsyncIterator

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.services.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    calculate_cost,
)


class GeminiAdapter(LLMProvider):

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.gemini_default_model
        _key = api_key or settings.gemini_api_key
        if not _key:
            raise LLMError("GEMINI_API_KEY not configured")
        try:
            import google.generativeai as genai
            genai.configure(api_key=_key)
            self._genai = genai
            self._client = genai.GenerativeModel(self._model)
        except ImportError as exc:
            raise LLMError("google-generativeai not installed. Run: pip install google-generativeai") from exc

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _to_gemini(messages: list[Message]) -> tuple[str, list[dict]]:
        """Convert to Gemini format: separate system_instruction + history."""
        system = ""
        history: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            elif m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})
        return system, history

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        system, history = self._to_gemini(messages)
        start = time.monotonic()
        try:
            model = self._genai.GenerativeModel(
                self._model,
                system_instruction=system or None,
            )
            generation_config = self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop or [],
            )
            # Extract last user message as the prompt
            user_content = history[-1]["parts"][0] if history else ""
            chat_history = history[:-1]

            chat = model.start_chat(history=chat_history)
            resp = await chat.send_message_async(
                user_content,
                generation_config=generation_config,
            )
        except Exception as exc:
            raise LLMError(f"Gemini completion failed: {exc}") from exc

        latency = int((time.monotonic() - start) * 1000)
        usage = resp.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return LLMResponse(
            content=resp.text,
            model=self._model,
            provider="gemini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=calculate_cost(self._model, prompt_tokens, completion_tokens),
            latency_ms=latency,
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[StreamChunk]:
        system, history = self._to_gemini(messages)
        try:
            model = self._genai.GenerativeModel(
                self._model,
                system_instruction=system or None,
            )
            generation_config = self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            user_content = history[-1]["parts"][0] if history else ""
            chat_history = history[:-1]
            chat = model.start_chat(history=chat_history)
            async for chunk in await chat.send_message_async(
                user_content,
                generation_config=generation_config,
                stream=True,
            ):
                yield StreamChunk(delta=chunk.text or "")
            yield StreamChunk(delta="", finish_reason="stop")
        except Exception as exc:
            raise LLMError(f"Gemini stream failed: {exc}") from exc
