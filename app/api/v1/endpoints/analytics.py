"""
Analytics API - T43.
GET /api/v1/analytics
GET /api/v1/analytics/summary
GET /api/v1/analytics/cost
GET /api/v1/analytics/tokens
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.analytics import Analytics
from app.db.repositories import AnalyticsRepository
from app.db.session import get_db
from app.schemas.search import AnalyticsSummary, CostBreakdown, TokenUsage

router = APIRouter()


def _parse_since(period: str | None) -> datetime | None:
    """Convert a period shorthand to a UTC datetime."""
    if not period:
        return None
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    periods = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
    delta = periods.get(period)
    return (now - delta) if delta else None


async def _build_summary(
    period: Literal["24h", "7d", "30d"] | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> AnalyticsSummary:
    """Overall usage summary - requests, tokens, cost, latency."""
    repo = AnalyticsRepository(session)
    since = _parse_since(period)

    total_requests = await repo.count()
    total_tokens = await repo.total_tokens(since=since)
    total_cost = await repo.total_cost(since=since)
    avg_latency = await repo.avg_latency_ms()
    requests_by_type = await repo.requests_by_type(since=since)
    cost_by_provider = await repo.cost_by_provider(since=since)

    return AnalyticsSummary(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 6),
        avg_latency_ms=round(avg_latency, 1),
        requests_by_type=requests_by_type,
        cost_by_provider=cost_by_provider,
    )


@router.get("", response_model=AnalyticsSummary)
async def get_analytics(
    period: Literal["24h", "7d", "30d"] | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> AnalyticsSummary:
    return await _build_summary(period=period, session=session)


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    period: Literal["24h", "7d", "30d"] | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> AnalyticsSummary:
    return await _build_summary(period=period, session=session)


@router.get("/cost", response_model=CostBreakdown)
async def get_cost(
    period: Literal["24h", "7d", "30d"] | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> CostBreakdown:
    """Cost breakdown by provider and model."""
    repo = AnalyticsRepository(session)
    since = _parse_since(period)

    total_cost = await repo.total_cost(since=since)
    by_provider = await repo.cost_by_provider(since=since)

    stmt = select(
        Analytics.model,
        func.sum(Analytics.cost_usd).label("total_cost"),
    ).group_by(Analytics.model)
    if since:
        stmt = stmt.where(Analytics.created_at >= since)
    result = await session.execute(stmt)
    by_model = [
        {"model": row.model, "total_cost": round(float(row.total_cost), 6)}
        for row in result
        if row.model
    ]
    by_model.sort(key=lambda x: x["total_cost"], reverse=True)

    return CostBreakdown(
        total_cost_usd=round(total_cost, 6),
        by_provider=by_provider,
        by_model=by_model,
    )


@router.get("/tokens", response_model=TokenUsage)
async def get_tokens(
    period: Literal["24h", "7d", "30d"] | None = Query(default=None),
    session: AsyncSession = Depends(get_db),
) -> TokenUsage:
    """Token usage breakdown by request type."""
    repo = AnalyticsRepository(session)
    since = _parse_since(period)

    total_tokens = await repo.total_tokens(since=since)
    by_type = await repo.requests_by_type(since=since)

    stmt = select(
        func.coalesce(func.sum(Analytics.prompt_tokens), 0).label("prompt"),
        func.coalesce(func.sum(Analytics.completion_tokens), 0).label("completion"),
    )
    if since:
        stmt = stmt.where(Analytics.created_at >= since)
    row = (await session.execute(stmt)).one()

    return TokenUsage(
        total_tokens=total_tokens,
        prompt_tokens=int(row.prompt),
        completion_tokens=int(row.completion),
        by_request_type=by_type,
    )
