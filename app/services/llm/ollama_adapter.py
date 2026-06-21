from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.services.llm.base import LLMProvider, LLMResponse, Message, StreamChunk


class OllamaAdapter(LLMProvider):
    """
    Ollama local LLM adapter using the /api/chat endpoint.
    Supports any model pulled via `ollama pull <model>`.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.ollama_default_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _to_ollama(messages: list[Message]) -> list[dict]:
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
        payload = {
            "model": self._model,
            "messages": self._to_ollama(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": stop or [],
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise LLMError(f"Ollama completion failed: {exc}") from exc

        latency = int((time.monotonic() - start) * 1000)
        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=self._model,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=0.0,   # local — no cost
            latency_ms=latency,
            finish_reason=data.get("done_reason", "stop"),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": self._model,
            "messages": self._to_ollama(messages),
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        delta = data.get("message", {}).get("content", "")
                        done = data.get("done", False)
                        yield StreamChunk(
                            delta=delta,
                            finish_reason="stop" if done else None,
                        )
        except Exception as exc:
            raise LLMError(f"Ollama stream failed: {exc}") from exc

    async def list_models(self) -> list[str]:
        """List models available in the local Ollama instance."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
