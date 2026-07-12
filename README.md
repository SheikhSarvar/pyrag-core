# PyRAG Core

> Open-source, production-ready Retrieval-Augmented Generation platform. Python-first. API-driven. LangGraph-native.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Agent | LangGraph + LangChain |
| Queue | Celery + Redis |
| DB | PostgreSQL + SQLAlchemy 2.0 |
| Vector | Qdrant (pluggable) |
| Storage | MinIO |
| Observability | Langfuse |
| Runtime | Python 3.12 + Docker |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/SheikhSarvar/pyrag-core.git
cd pyrag-core
uv sync --extra dev

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Start infrastructure
docker compose up -d

# 4. Run migrations
make migrate

# 5. Start dev server
make dev
```

API available at `http://localhost:8000`
Docs at `http://localhost:8000/docs`
Qdrant browser docs at `http://localhost:6333/dashboard`

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how every pipeline stage works internally, the data model, security model, and a list of known limitations / unverified integration points.

## Developer UI (Streamlit)

Quick playground for parsing → cleaning → chunking (no DB/MinIO required).

```bash
uv sync --extra dev --extra ui
make ui
```

UI: `http://localhost:8501`

Run it via Docker (optional):

```bash
docker compose --profile ui up -d --build
```

## Qdrant UI

When you run the local Docker stack, Qdrant is exposed on:

- Browser UI: `http://localhost:6333/dashboard`
- REST API: `http://localhost:6333`
- gRPC: `localhost:6334`

Use the browser UI to inspect collections, payloads, and request/response shapes before you index data. If the UI does not load, confirm the container is running with `docker compose ps` and try the REST health check:


## Project Structure

```
pyrag-core/
├── app/
│   ├── api/v1/endpoints/     # Route handlers
│   ├── core/                 # Config, security, lifespan
│   ├── db/
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── repositories/     # CRUD abstractions
│   │   └── migrations/       # Alembic migrations
│   ├── services/
│   │   ├── ingestion/        # Parse → chunk → embed → index
│   │   ├── retrieval/        # Query → search → rerank → assemble
│   │   ├── llm/              # Multi-provider LLM abstraction
│   │   └── embedding/        # Embedding providers
│   ├── agents/               # LangGraph agents + tools
│   ├── schemas/              # Pydantic request/response models
│   └── utils/                # Shared utilities
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/                  # One-off ops scripts
├── docs/
├── docker-compose.yml          # Single compose file for local and deployment
├── Dockerfile
└── pyproject.toml
```

`docker-compose.yml` is the single compose file for the project. Use it for local development and deployment, and control runtime differences with environment variables such as `ENVIRONMENT`, `SECRET_KEY`, `API_KEYS`, `LANGFUSE_HOST`, and provider keys.

## Commands

```bash
make install       # Install with dev deps
make dev           # Start dev server
make lint          # Run ruff linter
make format        # Auto-format code
make typecheck     # Run mypy
make test          # Run full test suite
make migrate       # Apply DB migrations
make docker-up     # Start all services
```

## Supported LLM Providers

OpenAI · Anthropic · Gemini · OpenRouter · Ollama · vLLM

## Supported Vector Stores

Qdrant · Weaviate · Milvus · pgvector · Elasticsearch

## License

MIT
