"""
Chat API — T41.
POST /api/v1/chat           — standard (buffered) response
POST /api/v1/chat/stream    — Server-Sent Events streaming response
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.observability import RAGTrace
from app.db.session import get_db
from app.db.repositories import AnalyticsRepository
from app.schemas.search import ChatRequest, ChatResponse, ChunkResult
from app.services.llm.base import Message
from app.services.llm.factory import get_llm_provider_from_db
from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    session: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    RAG chat: retrieve context, assemble prompt, generate answer.
    Returns the full response buffered.
    """
    total_start = time.monotonic()
    request_id = str(uuid.uuid4())
    trace = RAGTrace(name="chat", metadata={"dataset_id": body.dataset_id}).start()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_start = time.monotonic()
    config = RetrievalConfig(
        mode=body.mode,
        top_k=body.top_k,
        candidate_k=body.top_k * 3,
        rerank=body.rerank,
        rerank_top_k=body.top_k,
    )
    retrieval_result = await run_retrieval_pipeline(
        dataset_id=body.dataset_id,
        query=body.message,
        config=config,
        system_prompt=body.system_prompt,
        session=session,
    )
    retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)
    trace.log_retrieval(
        query=body.message,
        chunks=retrieval_result.context.chunks,
        latency_ms=retrieval_ms,
        mode=body.mode,
    )

    # ── Build message history ─────────────────────────────────────────────────
    messages: list[Message] = [
        Message(role="system", content=retrieval_result.prompt.system)
    ]
    for h in body.conversation_history:
        messages.append(Message(role=h.role, content=h.content))
    messages.append(Message(role="user", content=retrieval_result.prompt.user))

    # ── LLM generation ────────────────────────────────────────────────────────
    llm_start = time.monotonic()
    llm = await get_llm_provider_from_db(
        session=session,
        provider=body.provider,
        model=body.model,
    )
    llm_response = await llm.complete(
        messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )
    llm_ms = int((time.monotonic() - llm_start) * 1000)

    total_ms = int((time.monotonic() - total_start) * 1000)

    trace.log_generation(
        model=llm_response.model,
        provider=llm_response.provider,
        prompt=retrieval_result.prompt.user,
        completion=llm_response.content,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        cost_usd=llm_response.cost_usd,
        latency_ms=llm_ms,
    )
    trace.end(output=llm_response.content)

    # ── Analytics ─────────────────────────────────────────────────────────────
    try:
        await AnalyticsRepository(session).create(
            request_id=request_id,
            request_type="chat",
            dataset_id=body.dataset_id,
            provider=llm_response.provider,
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            cost_usd=llm_response.cost_usd,
            latency_ms=total_ms,
            status="success",
        )
    except Exception:
        pass

    sources = [
        ChunkResult(
            chunk_id=c["id"],
            score=c["score"],
            text=c["text"],
            metadata=c["metadata"],
            document_title=c["metadata"].get("document_title", ""),
            filename=c["metadata"].get("filename", ""),
            source_url=c["metadata"].get("source_url", ""),
        )
        for c in retrieval_result.context.chunks
    ]

    return ChatResponse(
        answer=llm_response.content,
        dataset_id=body.dataset_id,
        sources=sources,
        provider=llm_response.provider,
        model=llm_response.model,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        total_tokens=llm_response.total_tokens,
        cost_usd=llm_response.cost_usd,
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        llm_latency_ms=llm_ms,
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    RAG chat with Server-Sent Events streaming.
    Streams the LLM response token-by-token after retrieval completes.
    """
    config = RetrievalConfig(
        mode=body.mode,
        top_k=body.top_k,
        candidate_k=body.top_k * 3,
        rerank=body.rerank,
        rerank_top_k=body.top_k,
    )
    retrieval_result = await run_retrieval_pipeline(
        dataset_id=body.dataset_id,
        query=body.message,
        config=config,
        system_prompt=body.system_prompt,
        session=session,
    )

    messages: list[Message] = [
        Message(role="system", content=retrieval_result.prompt.system)
    ]
    for h in body.conversation_history:
        messages.append(Message(role=h.role, content=h.content))
    messages.append(Message(role="user", content=retrieval_result.prompt.user))

    llm = await get_llm_provider_from_db(
        session=session,
        provider=body.provider,
        model=body.model,
    )

    sources = [
        {
            "chunk_id": c["id"],
            "score": c["score"],
            "filename": c["metadata"].get("filename", ""),
            "document_title": c["metadata"].get("document_title", ""),
        }
        for c in retrieval_result.context.chunks
    ]

    async def event_generator():
        # First event: sources metadata
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream LLM tokens
        async for chunk in llm.stream(messages, max_tokens=body.max_tokens, temperature=body.temperature):
            if chunk.delta:
                yield f"data: {json.dumps({'type': 'token', 'delta': chunk.delta})}\n\n"
            if chunk.finish_reason:
                yield f"data: {json.dumps({'type': 'done', 'finish_reason': chunk.finish_reason})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
