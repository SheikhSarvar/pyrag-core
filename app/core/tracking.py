"""
Token + cost tracking — T51.
Single write path for all LLM usage.
Called after every chat/agent/search request.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.services.llm.base import LLMResponse, calculate_cost

logger = get_logger(__name__)


@dataclass
class UsageRecord:
    request_type: str                    # search | chat | agent | embed
    dataset_id: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "success"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_llm_response(
        cls,
        response: LLMResponse,
        request_type: str,
        dataset_id: str | None = None,
        latency_ms: int = 0,
    ) -> "UsageRecord":
        return cls(
            request_type=request_type,
            dataset_id=dataset_id,
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_usd=response.cost_usd or calculate_cost(
                response.model, response.prompt_tokens, response.completion_tokens
            ),
            latency_ms=latency_ms,
            status="success",
        )


async def track_usage(record: UsageRecord, session) -> None:
    """Persist a UsageRecord to the analytics table. Best-effort — never raises."""
    try:
        from app.db.repositories import AnalyticsRepository
        repo = AnalyticsRepository(session)
        await repo.create(
            request_id=record.request_id,
            request_type=record.request_type,
            dataset_id=record.dataset_id,
            provider=record.provider,
            model=record.model,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
            status=record.status,
        )
        logger.debug(
            "Usage tracked",
            type=record.request_type,
            tokens=record.total_tokens,
            cost=record.cost_usd,
        )
    except Exception as exc:
        logger.warning("Usage tracking failed", error=str(exc))


async def track_embedding_usage(
    model: str,
    token_count: int,
    dataset_id: str | None = None,
    latency_ms: int = 0,
    session=None,
) -> None:
    """Track embedding-specific usage (no completion tokens)."""
    # Rough cost: text-embedding-3-small = $0.00002/1k tokens
    _EMBED_COSTS = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001,
    }
    cost = token_count / 1000 * _EMBED_COSTS.get(model, 0.00002)
    record = UsageRecord(
        request_type="embed",
        dataset_id=dataset_id,
        provider="openai",
        model=model,
        prompt_tokens=token_count,
        total_tokens=token_count,
        cost_usd=cost,
        latency_ms=latency_ms,
    )
    if session:
        await track_usage(record, session)
