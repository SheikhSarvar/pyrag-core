"""
pgvector adapter — stores vectors in PostgreSQL using the pgvector extension.
Requires: CREATE EXTENSION IF NOT EXISTS vector; in Postgres.
"""
from __future__ import annotations

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.services.vector.base import (
    SearchQuery,
    SearchResult,
    VectorPoint,
    VectorStore,
)

logger = get_logger(__name__)


class PgVectorAdapter(VectorStore):
    """
    pgvector adapter using raw asyncpg queries.
    Each collection becomes a dedicated table: vectors_{collection_name}
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _table(self, collection_name: str) -> str:
        safe = collection_name.replace("-", "_").replace(".", "_")
        return f"vectors_{safe}"

    async def _conn(self):  # type: ignore[return]
        import asyncpg
        return await asyncpg.connect(
            self._database_url.replace("postgresql+asyncpg://", "postgresql://")
        )

    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        table = self._table(collection_name)
        try:
            conn = await self._conn()
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dimensions}),
                    payload JSONB DEFAULT '{{}}'
                )
            """)
            index_op = {"cosine": "vector_cosine_ops", "dot": "vector_ip_ops"}.get(
                distance, "vector_cosine_ops"
            )
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                ON {table} USING hnsw (embedding {index_op})
            """)
            await conn.close()
            logger.info("Created pgvector table", table=table)
        except Exception as exc:
            raise VectorStoreError(f"pgvector create_collection failed: {exc}") from exc

    async def delete_collection(self, collection_name: str) -> None:
        table = self._table(collection_name)
        conn = await self._conn()
        await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await conn.close()

    async def collection_exists(self, collection_name: str) -> bool:
        table = self._table(collection_name)
        conn = await self._conn()
        row = await conn.fetchrow(
            "SELECT 1 FROM information_schema.tables WHERE table_name = $1", table
        )
        await conn.close()
        return row is not None

    async def collection_info(self, collection_name: str) -> dict:
        table = self._table(collection_name)
        conn = await self._conn()
        row = await conn.fetchrow(f"SELECT COUNT(*) AS cnt FROM {table}")
        await conn.close()
        return {"name": collection_name, "count": row["cnt"] if row else 0}

    async def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        import json
        table = self._table(collection_name)
        conn = await self._conn()
        try:
            await conn.executemany(
                f"INSERT INTO {table}(id, embedding, payload) VALUES($1, $2::vector, $3::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload",
                [(p.id, str(p.vector), json.dumps(p.payload)) for p in points],
            )
        finally:
            await conn.close()

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        table = self._table(collection_name)
        conn = await self._conn()
        await conn.execute(f"DELETE FROM {table} WHERE id = ANY($1)", ids)
        await conn.close()

    async def delete_by_filter(self, collection_name: str, filters: dict) -> None:
        import json
        table = self._table(collection_name)
        conn = await self._conn()
        for k, v in filters.items():
            await conn.execute(
                f"DELETE FROM {table} WHERE payload @> $1::jsonb",
                json.dumps({k: v}),
            )
        await conn.close()

    async def search(self, collection_name: str, query: SearchQuery) -> list[SearchResult]:
        table = self._table(collection_name)
        op = "<->"  # L2; use "<#>" for dot, "<=>" for cosine
        conn = await self._conn()
        rows = await conn.fetch(
            f"SELECT id, payload, 1 - (embedding <=> $1::vector) AS score "
            f"FROM {table} ORDER BY embedding <=> $1::vector LIMIT $2",
            str(query.vector),
            query.top_k,
        )
        await conn.close()
        return [SearchResult(id=r["id"], score=float(r["score"]), payload=r["payload"]) for r in rows]

    async def get(self, collection_name: str, ids: list[str]) -> list[VectorPoint]:
        table = self._table(collection_name)
        conn = await self._conn()
        rows = await conn.fetch(f"SELECT id, embedding::text, payload FROM {table} WHERE id = ANY($1)", ids)
        await conn.close()
        return [VectorPoint(id=r["id"], vector=[], payload=r["payload"]) for r in rows]

    async def count(self, collection_name: str, filters: dict | None = None) -> int:
        table = self._table(collection_name)
        conn = await self._conn()
        row = await conn.fetchrow(f"SELECT COUNT(*) AS cnt FROM {table}")
        await conn.close()
        return row["cnt"] if row else 0
