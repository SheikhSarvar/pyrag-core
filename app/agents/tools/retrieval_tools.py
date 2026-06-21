"""
LangGraph-compatible retrieval tools — T45, T46, T47, T48.
Each factory function returns a LangChain/LangGraph Tool that
agents can invoke to search a dataset.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import Tool


def create_retriever_tool(
    dataset_id: str,
    name: str | None = None,
    description: str | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> Tool:
    """
    T45 — Retriever tool.
    Returns the top-k most relevant chunks for an agent's query.
    Used when the agent needs raw chunk text (e.g. for synthesis).
    """

    async def _retrieve(query: str) -> str:
        from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            config = RetrievalConfig(mode=mode, top_k=top_k, rerank=True, rerank_top_k=top_k)
            result = await run_retrieval_pipeline(
                dataset_id=dataset_id,
                query=query,
                config=config,
                session=session,
            )

        if not result.context.chunks:
            return "No relevant information found."

        parts = [
            f"[{i+1}] (source: {c['metadata'].get('filename', 'unknown')})\n{c['text']}"
            for i, c in enumerate(result.context.chunks)
        ]
        return "\n\n".join(parts)

    return Tool(
        name=name or f"retrieve_from_{dataset_id}",
        description=description or (
            f"Retrieve relevant information from dataset '{dataset_id}'. "
            "Input: a search query string. "
            "Output: relevant text chunks with source attribution."
        ),
        coroutine=_retrieve,
        func=lambda q: q,  # sync stub — agents should use async
    )


def create_search_tool(
    dataset_id: str,
    name: str | None = None,
    description: str | None = None,
    top_k: int = 10,
) -> Tool:
    """
    T46 — Search tool.
    Returns structured JSON results (id, score, text, metadata).
    Used when the agent needs scores or metadata, not just text.
    """
    import json

    async def _search(query: str) -> str:
        from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            config = RetrievalConfig(
                mode="hybrid", top_k=top_k, rerank=True, rerank_top_k=top_k
            )
            result = await run_retrieval_pipeline(
                dataset_id=dataset_id,
                query=query,
                config=config,
                session=session,
            )

        results = [
            {
                "id": c["id"],
                "score": round(c["score"], 4),
                "text": c["text"][:500],
                "filename": c["metadata"].get("filename", ""),
                "page": c["metadata"].get("page"),
            }
            for c in result.context.chunks
        ]
        return json.dumps(results, ensure_ascii=False)

    return Tool(
        name=name or f"search_{dataset_id}",
        description=description or (
            f"Search dataset '{dataset_id}' and return structured results with scores. "
            "Input: a search query string. "
            "Output: JSON array of {id, score, text, filename, page}."
        ),
        coroutine=_search,
        func=lambda q: q,
    )


def create_dataset_tool(
    dataset_id: str,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """
    T47 — Dataset tool.
    Returns metadata about a dataset: document count, chunk count, status.
    Useful for agents that need to reason about data availability.
    """
    import json

    async def _dataset_info(query: str) -> str:  # noqa: ARG001 — query unused; tool called for side-effect
        from app.db.session import AsyncSessionLocal
        from app.db.repositories import DatasetRepository, DocumentRepository, ChunkRepository

        async with AsyncSessionLocal() as session:
            ds_repo = DatasetRepository(session)
            doc_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)

            dataset = await ds_repo.get(dataset_id)
            if not dataset:
                return json.dumps({"error": f"Dataset '{dataset_id}' not found"})

            doc_count = await doc_repo.count(filters={"dataset_id": dataset_id})
            indexed_count = await doc_repo.count(
                filters={"dataset_id": dataset_id, "status": "indexed"}
            )
            chunk_count = await chunk_repo.count(filters={"dataset_id": dataset_id})

        return json.dumps({
            "id": dataset_id,
            "name": dataset.name,
            "chunk_strategy": dataset.chunk_strategy,
            "embedding_model": dataset.embedding_model,
            "total_documents": doc_count,
            "indexed_documents": indexed_count,
            "total_chunks": chunk_count,
        }, ensure_ascii=False)

    return Tool(
        name=name or f"dataset_info_{dataset_id}",
        description=description or (
            f"Get metadata about dataset '{dataset_id}': "
            "document count, chunk count, indexing status, and configuration. "
            "Useful for checking if data is available before searching. "
            "Input: any string (ignored). Output: JSON object with dataset info."
        ),
        coroutine=_dataset_info,
        func=lambda q: q,
    )


def create_knowledge_tool(
    dataset_id: str,
    name: str | None = None,
    description: str | None = None,
    top_k: int = 5,
    provider: str | None = None,
    model: str | None = None,
) -> Tool:
    """
    T48 — Knowledge tool.
    Full RAG pipeline in a single tool call: retrieve + generate a grounded answer.
    The agent gets a direct natural language answer, not raw chunks.
    Best used when the agent needs a synthesised answer to pass on.
    """

    async def _answer(query: str) -> str:
        from app.services.retrieval.pipeline import RetrievalConfig, run_retrieval_pipeline
        from app.services.llm.factory import get_llm_provider_from_db
        from app.services.llm.base import Message
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            config = RetrievalConfig(mode="hybrid", top_k=top_k, rerank=True, rerank_top_k=top_k)
            retrieval = await run_retrieval_pipeline(
                dataset_id=dataset_id,
                query=query,
                config=config,
                session=session,
            )

            if not retrieval.context.chunks:
                return "I could not find relevant information in the knowledge base to answer this question."

            llm = await get_llm_provider_from_db(
                session=session,
                provider=provider,
                model=model,
            )
            messages = [
                Message(role="system", content=retrieval.prompt.system),
                Message(role="user", content=retrieval.prompt.user),
            ]
            response = await llm.complete(messages, max_tokens=512, temperature=0.1)

        return response.content

    return Tool(
        name=name or f"knowledge_{dataset_id}",
        description=description or (
            f"Answer a question using knowledge from dataset '{dataset_id}'. "
            "Performs retrieval and generates a grounded natural language answer. "
            "Input: a natural language question. "
            "Output: a direct answer based only on the dataset's content."
        ),
        coroutine=_answer,
        func=lambda q: q,
    )
