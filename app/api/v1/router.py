from fastapi import APIRouter

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import agents, analytics, chat, datasets, documents, search
from app.core.middleware.auth import verify_api_key

api_router = APIRouter(dependencies=[Depends(verify_api_key)])

api_router.include_router(datasets.router,  prefix="/datasets",  tags=["Datasets"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(search.router,    prefix="/search",    tags=["Search"])
api_router.include_router(chat.router,      prefix="/chat",      tags=["Chat"])
api_router.include_router(agents.router,    prefix="/agents",    tags=["Agents"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
