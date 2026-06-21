from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.services.vector.base import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    """
    Factory — returns the correct VectorStore implementation
    based on VECTOR_PROVIDER env var. Cached per process.
    """
    settings = get_settings()
    provider = settings.vector_provider

    if provider == "qdrant":
        from app.services.vector.qdrant_adapter import QdrantAdapter, get_qdrant_client
        return QdrantAdapter(get_qdrant_client())

    if provider == "weaviate":
        try:
            import weaviate
        except ImportError as exc:
            raise VectorStoreError(
                "weaviate-client not installed. Run: pip install pyrag-core[weaviate]"
            ) from exc
        client = weaviate.connect_to_local()
        from app.services.vector.weaviate_adapter import WeaviateAdapter
        return WeaviateAdapter(client)

    if provider == "milvus":
        try:
            import pymilvus  # noqa: F401
        except ImportError as exc:
            raise VectorStoreError(
                "pymilvus not installed. Run: pip install pyrag-core[milvus]"
            ) from exc
        from app.services.vector.milvus_adapter import MilvusAdapter
        return MilvusAdapter()

    if provider == "pgvector":
        from app.services.vector.pgvector_adapter import PgVectorAdapter
        return PgVectorAdapter(settings.database_url)

    if provider == "elasticsearch":
        try:
            import elasticsearch  # noqa: F401
        except ImportError as exc:
            raise VectorStoreError(
                "elasticsearch not installed. Run: pip install pyrag-core[elasticsearch]"
            ) from exc
        from app.services.vector.elasticsearch_adapter import ElasticsearchAdapter
        return ElasticsearchAdapter()

    raise VectorStoreError(f"Unknown vector provider: {provider!r}")
