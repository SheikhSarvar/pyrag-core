"""
Agent API — T42.
POST /api/v1/agents/chat
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.db.repositories import AnalyticsRepository
from app.schemas.search import AgentChatRequest, AgentChatResponse, AgentStep, ChunkResult

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    body: AgentChatRequest,
    session: AsyncSession = Depends(get_db),
) -> AgentChatResponse:
    """
    Agentic RAG: a LangGraph ReAct agent iteratively retrieves and reasons
    over the dataset to produce a multi-step grounded answer.
    """
    start = time.monotonic()
    request_id = str(uuid.uuid4())

    from app.agents.graph import run_agent

    result = await run_agent(
        dataset_id=body.dataset_id,
        query=body.message,
        conversation_history=[h.model_dump() for h in body.conversation_history],
        provider=body.provider,
        model=body.model,
        max_iterations=body.max_iterations,
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    steps = [
        AgentStep(
            step=s["step"],
            tool=s["tool"],
            input=s["input"],
            output=s["output"],
        )
        for s in result.get("steps", [])
    ]

    # Sources extracted from `search_*` tool calls (structured JSON output).
    # Plain-text retriever/knowledge tool calls aren't parsed into discrete
    # sources — see steps[].output for that content instead.
    sources = [
        ChunkResult(
            chunk_id=s.get("chunk_id", ""),
            score=s.get("score", 0.0),
            text=s.get("text", ""),
            metadata={},
            filename=s.get("filename", ""),
        )
        for s in result.get("sources", [])
    ]

    resolved_model = result.get("model", body.model or "auto")
    total_tokens = result.get("total_tokens", 0)
    cost_usd = result.get("cost_usd", 0.0)

    # Track analytics
    try:
        await AnalyticsRepository(session).create(
            request_id=request_id,
            request_type="agent",
            dataset_id=body.dataset_id,
            provider=body.provider,
            model=resolved_model,
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            status="success",
        )
    except Exception:
        pass

    return AgentChatResponse(
        answer=result.get("answer", ""),
        dataset_id=body.dataset_id,
        steps=steps,
        sources=sources,
        provider=body.provider or "auto",
        model=resolved_model,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
