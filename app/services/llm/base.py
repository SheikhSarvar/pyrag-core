"""
LLM provider abstraction — T30.
All providers implement LLMProvider. Callers only see this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    role: str        # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    finish_reason: str = "stop"
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return self.cost_usd


@dataclass
class StreamChunk:
    delta: str
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base all LLM adapters must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Non-streaming completion."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming completion — yields StreamChunk deltas."""

    # ── Convenience helpers ───────────────────────────────────────────────────

    async def chat(self, system: str, user: str, **kwargs) -> LLMResponse:  # type: ignore[type-arg]
        """Shorthand for a simple system+user call."""
        return await self.complete(
            [Message(role="system", content=system), Message(role="user", content=user)],
            **kwargs,
        )


# ── Cost table (USD per 1k tokens) ───────────────────────────────────────────

COST_TABLE: dict[str, dict[str, float]] = {
    # model: {input: $/1k, output: $/1k}
    "gpt-4o":              {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":         {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":         {"input": 0.01,    "output": 0.03},
    "claude-opus-4-6":     {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-6":   {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5":    {"input": 0.00025, "output": 0.00125},
    "gemini-1.5-pro":      {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash":    {"input": 0.000075,"output": 0.0003},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost from token counts. Returns 0.0 for unknown models."""
    rates = COST_TABLE.get(model)
    if not rates:
        return 0.0
    return (prompt_tokens / 1000 * rates["input"]) + (completion_tokens / 1000 * rates["output"])
