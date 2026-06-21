from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    dataset_id: str
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["standard", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True
    expand_query: bool = False
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict | None = None


class ChunkResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict
    document_title: str = ""
    filename: str = ""
    source_url: str = ""


class SearchResponse(BaseModel):
    query: str
    dataset_id: str
    mode: str
    results: list[ChunkResult]
    total_results: int
    latency_ms: int
    reranked: bool


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    dataset_id: str
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    mode: Literal["standard", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    provider: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    system_prompt: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    dataset_id: str
    sources: list[ChunkResult]
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    retrieval_latency_ms: int
    llm_latency_ms: int


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentChatRequest(BaseModel):
    dataset_id: str
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=2048, ge=64, le=8192)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_iterations: int = Field(default=5, ge=1, le=20)


class AgentStep(BaseModel):
    step: int
    tool: str
    input: str
    output: str


class AgentChatResponse(BaseModel):
    answer: str
    dataset_id: str
    steps: list[AgentStep]
    sources: list[ChunkResult]
    provider: str
    model: str
    total_tokens: int
    cost_usd: float
    latency_ms: int


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    requests_by_type: list[dict]
    cost_by_provider: list[dict]


class CostBreakdown(BaseModel):
    total_cost_usd: float
    by_provider: list[dict]
    by_model: list[dict]


class TokenUsage(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    by_request_type: list[dict]
