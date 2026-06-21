from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VectorPoint:
    """A single vector to upsert."""
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result from a vector search."""
    id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Parameters for a vector search."""
    vector: list[float]
    top_k: int = 10
    filters: dict | None = None
    score_threshold: float | None = None


class VectorStore(ABC):
    """Abstract interface all vector store adapters must implement."""

    # ── Collection / Index management ─────────────────────────────────────────

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimensions: int,
        distance: str = "cosine",
    ) -> None:
        """Create a vector collection / index if it doesn't exist."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Drop a collection and all its vectors."""

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Return True if the collection exists."""

    @abstractmethod
    async def collection_info(self, collection_name: str) -> dict:
        """Return metadata about the collection (count, dimensions, etc.)."""

    # ── Write ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def upsert(
        self, collection_name: str, points: list[VectorPoint]
    ) -> None:
        """Insert or update vectors."""

    @abstractmethod
    async def delete(
        self, collection_name: str, ids: list[str]
    ) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    async def delete_by_filter(
        self, collection_name: str, filters: dict
    ) -> None:
        """Delete all vectors matching a metadata filter."""

    # ── Read ──────────────────────────────────────────────────────────────────

    @abstractmethod
    async def search(
        self, collection_name: str, query: SearchQuery
    ) -> list[SearchResult]:
        """Nearest-neighbour search."""

    @abstractmethod
    async def get(
        self, collection_name: str, ids: list[str]
    ) -> list[VectorPoint]:
        """Fetch vectors by ID."""

    @abstractmethod
    async def count(
        self, collection_name: str, filters: dict | None = None
    ) -> int:
        """Count vectors in a collection, optionally filtered."""
