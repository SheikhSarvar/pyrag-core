# PyRAG Core — Architecture & Internals

This document explains how PyRAG Core actually works end-to-end: request flow, every pipeline stage, the data model, and the operational concerns (security, observability). It assumes you've read the README quickstart and have the service running, or are about to.

---

## 1. System overview

PyRAG Core is a FastAPI application backed by four stateful services:

| Service | Role |
|---|---|
| PostgreSQL | System of record — datasets, documents, chunks, jobs, providers, analytics |
| Qdrant (or pluggable alt) | Vector index — one collection per dataset |
| Redis | Celery broker/backend + rate-limit counters |
| MinIO | Raw + processed file storage (S3-compatible) |

Everything else — parsing, chunking, embedding, retrieval, generation — is stateless application code that reads/writes those four stores. Langfuse is optional and purely additive (tracing); the app works identically with it absent.

```
Client -> FastAPI (app/main.py)
           |- Middleware: GZip -> CORS -> RateLimit(Redis) -> AuditLog
           |- Dependency: verify_api_key (per-route, all of /api/v1/*)
           `- Routers: /datasets /documents /search /chat /agents /analytics
                          |
              +-----------+--------------------+
              v           v                     v
        Ingestion    Retrieval Pipeline    LLM Provider Layer
        Pipeline     (search/chat/agent)   (OpenAI/Anthropic/...)
              |           |
              v           v
        Celery Worker  Vector Store (Qdrant/...)
        (async jobs)   + PostgreSQL (chunk text/metadata)
```

---

## 2. Repository layout

```
app/
|-- main.py                 FastAPI app factory, middleware wiring, /health
|-- core/
|   |-- config.py            Settings (pydantic-settings, env-driven)
|   |-- exceptions.py        Domain exceptions + handlers
|   |-- lifespan.py          Startup/shutdown: DB ping, Redis ping, MinIO buckets
|   |-- logging.py           structlog setup (JSON in prod, console in dev)
|   |-- observability.py     Langfuse trace/span wrapper (no-op if unconfigured)
|   |-- tracking.py          Single write-path for token/cost -> analytics table
|   |-- validation.py        Filename/URL/query sanitization (incl. SSRF guard)
|   `-- middleware/
|       |-- auth.py           API key verification
|       |-- rate_limit.py     Redis sliding-window limiter
|       `-- audit.py          Request logging + catches unhandled exceptions
|-- db/
|   |-- base.py               Declarative base, UUID/Timestamp mixins, PortableJSON
|   |-- session.py            Async engine + get_db() dependency
|   |-- models/                6 SQLAlchemy models (see Section 4)
|   |-- repositories/          One repo class per model, generic CRUD base
|   `-- migrations/            Alembic, async-native env.py
|-- services/
|   |-- ingestion/             parsers, cleaner, chunkers, metadata, indexer, pipeline
|   |-- retrieval/              query_understanding, query_expansion, dense, sparse,
|   |                            hybrid, reranker, context, pipeline
|   |-- embedding/              OpenAI / sentence-transformers / Ollama providers
|   |-- llm/                    6 provider adapters + factory (DB-backed hot-swap)
|   |-- vector/                 VectorStore interface + 5 adapters + factory
|   `-- storage/                MinIO client wrapper
|-- agents/
|   |-- graph.py                LangGraph ReAct agent
|   `-- tools/                  4 retrieval tools exposed to the agent
|-- api/v1/
|   |-- router.py               Mounts all endpoint routers behind verify_api_key
|   `-- endpoints/               datasets, documents, search, chat, agents, analytics
|-- schemas/                    Pydantic request/response models
`-- worker/
    |-- celery_app.py           Celery config (queues, retry policy)
    `-- tasks/ingestion.py       ingest_document / ingest_url Celery tasks
```

---

## 3. Request lifecycles

### 3.1 Document upload to indexed (async)

```
POST /documents/upload (multipart file + dataset_id)
  |
  |- validate dataset exists, file size <= MAX_UPLOAD_SIZE_MB, extension supported
  |- upload raw bytes to MinIO:  {dataset_id}/{document_id}/raw/{filename}
  |- create Document row (status=pending), Job row (status=queued)
  |- celery: ingest_document.apply_async(...)
  `- return 202 {document_id, job_id} immediately

Celery worker (queue="ingestion"):
  1. download raw bytes from MinIO
  2. parse_document()      - dispatches by extension (PDF/DOCX/PPTX/XLSX/CSV/TXT/MD/HTML)
  3. clean_text()          - unicode normalize, strip control chars/noise/page numbers
  4. extract_metadata()    - title/author/pages + word-count heuristics
  5. chunker.chunk()       - strategy from dataset.chunk_strategy (fixed/recursive/
                              semantic/hierarchical)
  6. index_chunks():
       a. ensure_collection()        - create vector collection if missing
       b. embedder.embed_texts()     - batched (OpenAI: 100/batch)
       c. vector_store.upsert()      - write vectors + payload
       d. chunk_repo.bulk_create()   - write chunk rows to Postgres
  7. document.status = "indexed" ; job.status = "completed" ; job.progress = 100

  On any exception at any step: document.status="failed", error_message set,
  job marked failed, Celery retries up to 3x (30s backoff) before giving up.
```

Poll `GET /documents/jobs/{job_id}` for progress (0 to 100, updated after every pipeline stage).

### 3.2 Search (synchronous)

```
POST /search {dataset_id, query, mode, top_k, rerank, ...}
  |
  `- run_retrieval_pipeline():
       1. understand_query()   - normalize, detect intent/language/keywords
       2. [optional] expand_query()  - synonym + LLM reformulation
       3. retrieve:
            mode=standard -> dense_search()  (embed query -> vector kNN)
            mode=hybrid   -> hybrid_search() (dense + BM25, fused via weighted RRF)
       4. [optional] rerank_results()  - cross-encoder re-scores top candidates
       5. compress_context()   - Jaccard dedup + token-budget trim
       6. assemble_prompt()    - builds system+user prompt (not used by /search,
                                  but context.chunks is what's returned)
  `- returns ChunkResult[] with score, text, filename, source_url
```

### 3.3 Chat (synchronous, with optional SSE)

Same retrieval pipeline as 3.2, then:

```
  -> assembled prompt (system + user with <context> block)
  -> get_llm_provider_from_db()   - resolves provider: explicit arg -> DB default
                                     Provider row -> env config -> Ollama fallback
  -> llm.complete(messages)        - non-streaming: full LLMResponse
       or
  -> llm.stream(messages)          - SSE: {type:sources} -> {type:token}* -> {type:done}
  -> RAGTrace logs retrieval + generation spans to Langfuse (if configured)
  -> AnalyticsRepository.create()  - tokens, cost, latency persisted
```

### 3.4 Agent chat (multi-step)

```
POST /agents/chat {dataset_id, message, max_iterations}
  |
  `- run_agent():
       1. build_rag_agent() - LangGraph StateGraph with 4 tools bound to an
          OpenAI/Anthropic chat model (whichever key is configured)
       2. ReAct loop: agent_node <-> tool_node until the LLM responds without
          a tool call, or max_iterations is hit (forces a final answer)
       3. aggregate usage_metadata across every AIMessage turn -> total_tokens, cost
       4. extract sources from search_* tool JSON outputs (retriever/knowledge
          tool outputs are prose and aren't parsed into discrete sources)
  `- returns {answer, steps[] (tool/input/output per iteration), sources, tokens, cost}
```

The four tools available to the agent (`app/agents/tools/retrieval_tools.py`):

| Tool | Returns | When the agent uses it |
|---|---|---|
| `retrieve_from_{dataset}` | Raw chunk text, source-attributed | Needs context to synthesize an answer |
| `search_{dataset}` | JSON: `[{id, score, text, filename, page}]` | Needs scores/metadata to reason about relevance |
| `dataset_info_{dataset}` | JSON: doc/chunk counts, status | Checking data availability before searching |
| `knowledge_{dataset}` | Direct grounded answer (mini RAG call) | Wants a synthesized sub-answer, not raw chunks |

---

## 4. Data model

```
datasets (1) --< documents (1) --< chunks
                                      |
                                      `- vector_reference -> vector store point ID

providers          analytics          jobs
(LLM config,        (every search/      (async task tracking,
 hot-swappable)      chat/agent call)    1:1 with Celery task)
```

Every table has `id` (UUID string), `created_at`, `updated_at` via `UUIDMixin`/`TimestampMixin`. JSON columns (`chunk_metadata`, `configuration`, `payload`, `result`) use `PortableJSON` — a `JSON().with_variant(JSONB, "postgresql")` type that renders as `JSONB` in production Postgres but degrades to plain `JSON` on SQLite, which is what lets the test suite run without a live Postgres instance.

`vector_reference` on `Chunk` is the join key between Postgres (source of truth for text/metadata) and the vector store (source of truth for the embedding). Deleting a document cascades chunk deletion in Postgres and triggers `delete_document_vectors()` against the vector store separately — they are not in the same transaction, so a crash between the two leaves an orphaned vector. This is a known eventual-consistency gap, acceptable for V1.

---

## 5. Chunking strategies

Set per-dataset via `Dataset.chunk_strategy`. All four live in `services/ingestion/chunkers.py`:

- **fixed** — token-count windows with overlap. Fastest, best for structured/tabular text (CSV exports).
- **recursive** (default) — splits on `\n\n -> \n -> ". " -> ...` in priority order, falling back to fixed-size only when a single sentence exceeds the chunk budget. Preserves paragraph/sentence boundaries.
- **semantic** — embeds consecutive sentences, splits where cosine similarity between neighbors drops below a threshold (0.75 default). Requires an embedding call per chunking pass — slower and costs tokens, but produces topically coherent chunks. Falls back silently to zero-vector splitting if the embedding call fails.
- **hierarchical** — two-level: large parent windows (2000 tokens) split further into child windows (500 tokens). Each child's metadata carries `parent_index` and a `parent_text` preview, so a future "expand context" feature could re-fetch the parent. Currently only the children are indexed/retrieved.

---

## 6. Retrieval internals

### Hybrid fusion (RRF)

`hybrid_search()` runs dense (vector kNN) and sparse (BM25) retrieval **concurrently** via `asyncio.gather`, then fuses with weighted Reciprocal Rank Fusion:

```
score(doc) = sum over sources of weight_source * 1/(k + rank_in_source)     k=60
```

Default weights: dense 0.7, sparse 0.3. This is rank-based, not score-based — it sidesteps the problem of cosine-similarity and BM25 scores living on incomparable scales.

### BM25 implementation

`services/retrieval/sparse.py` builds an **in-process** BM25 index on every search call by pulling up to 50,000 chunks from Postgres for the target dataset (`k1=1.5, b=0.75`, standard defaults). This is fine up to tens of thousands of chunks per dataset; it does not scale to the PRD's 10M-document target without swapping in the `ElasticsearchAdapter`'s native BM25 query instead — that adapter exists but isn't wired into `sparse_search()` yet. **This is a known scaling gap**, not an oversight: building it generically against "whatever the active vector provider is" would have meant either always requiring Elasticsearch or duplicating index-management logic, neither of which fit V1 scope.

### Reranking

`CrossEncoderReranker` (local, `sentence-transformers`, `ms-marco-MiniLM-L-6-v2`) is the default. `CohereReranker` is available via `backend="cohere"` but requires `pip install pyrag-core[cohere]` and `COHERE_API_KEY` — without it, `CohereReranker.rerank()` catches the `ImportError` internally and silently returns results in original order rather than crashing.

### Context compression

`compress_context()` does two independent things, in order: drops chunks below `min_score`, then deduplicates via 4+ character-word Jaccard similarity (default threshold 0.85) against already-kept chunks, then trims to `max_context_tokens` (~4-chars-per-token estimate, not a real tokenizer call — cheap but approximate).

---

## 7. LLM provider abstraction

Every provider implements the same `LLMProvider` ABC (`complete()`, `stream()`) and returns a normalized `LLMResponse` (content, tokens, cost, latency, finish_reason) regardless of vendor. Cost is computed from a static `COST_TABLE` in `services/llm/base.py` — **this table is a point-in-time snapshot and will drift from real vendor pricing**; update it when provider pricing changes, there is no live pricing API integration.

**Provider resolution order** (`get_llm_provider_from_db`):
1. Explicit `provider`/`model` argument on the request
2. Active default row in the `providers` table (hot-swappable without restart — change the DB row, next request picks it up)
3. Environment variable defaults (`OPENAI_API_KEY` -> `ANTHROPIC_API_KEY` -> `GEMINI_API_KEY` -> Ollama fallback)

This lets an operator switch the system-wide default model via a DB write with zero downtime, while still working out-of-the-box from `.env` alone with no DB rows configured.

---

## 8. Vector store abstraction

`VectorStore` ABC (`services/vector/base.py`) is intentionally narrow — 9 methods covering collection lifecycle, upsert, delete (by ID or filter), search, get, count. Five adapters implement it: Qdrant (primary, fully async via `AsyncQdrantClient`), Weaviate, Milvus, pgvector (raw `asyncpg`, stores vectors in Postgres itself — useful if you don't want to run a separate vector DB), and Elasticsearch (kNN via `dense_vector` fields).

`get_vector_store()` factory reads `VECTOR_PROVIDER` from settings and lazy-imports only the SDK that provider needs — installing `pyrag-core` doesn't pull in `pymilvus` or `weaviate-client` unless you opt into those extras.

Every dataset gets its own collection, named `pyrag_{dataset_id with hyphens replaced by underscores}`. This isolation means cross-dataset search isn't possible without iterating collections — a deliberate simplicity tradeoff for V1 (no cross-tenant leakage risk by construction, at the cost of no "search everything" mode).

---

## 9. Security

- **Auth**: `verify_api_key` dependency on the entire `/api/v1` router. Keys come from a comma-separated `API_KEYS` env var, compared with `hmac.compare_digest` (timing-attack resistant). **If `API_KEYS` is unset, auth is fully disabled** — this is the intentional dev-mode default; set `API_KEYS` before exposing the service externally.
- **Rate limiting**: Redis fixed-window counter, keyed by API key (preferred) or client IP. Fails open if Redis is unreachable — availability is prioritized over strict limiting during a Redis outage, logged as a warning.
- **Input validation**: `core/validation.py` — filename path-traversal stripping, dataset name character whitelisting, query length bounds, and an SSRF guard on URL ingestion (`validate_ingestion_url`) that blocks `localhost`, `169.254.169.254` (cloud metadata endpoints), and RFC1918 private ranges. **Note:** this validator exists but isn't yet called from the URL-ingestion Celery task (`ingest_url` in `worker/tasks/ingestion.py`) — wire it in before exposing URL ingestion to untrusted input.
- **Exception handling**: a generic `Exception` handler in `core/exceptions.py` converts unhandled errors to structured JSON, but because two custom `BaseHTTPMiddleware` instances are stacked (`RateLimitMiddleware`, `AuditLoggingMiddleware`), Starlette's anyio TaskGroup wrapping can let exceptions escape that handler under certain conditions. `AuditLoggingMiddleware` (the innermost middleware) independently catches and converts exceptions itself as a second line of defense — this is why both layers exist, not redundancy for its own sake.

---

## 10. Observability

`RAGTrace` (`core/observability.py`) wraps each chat/agent request in a Langfuse trace with two span types: `log_retrieval()` (query, chunk count, top score, latency) and `log_generation()` (model, prompt, completion, token usage, cost). If Langfuse isn't configured (`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` unset), every method on `RAGTrace` is a silent no-op — the application behaves identically with or without it, just without traces showing up anywhere.

Independently of Langfuse, every search/chat/agent request writes a row to the `analytics` Postgres table via `AnalyticsRepository` — this is what powers `/analytics`, `/analytics/cost`, `/analytics/tokens`. These two systems (Langfuse traces, Postgres analytics rows) are redundant by design: Langfuse for rich debugging/trace inspection, Postgres analytics for fast aggregate queries without needing a Langfuse instance running.

---

## 11. Configuration reference

All settings load via `Settings` (`core/config.py`), backed by `.env`. Full list in `.env.example`. The ones worth understanding beyond their name:

| Variable | Effect |
|---|---|
| `ENVIRONMENT=test` | Switches `db/session.py` to `NullPool` (no connection pooling) — used by the test suite to avoid connection leaks between tests, not meant for production |
| `VECTOR_PROVIDER` | One of `qdrant\|weaviate\|milvus\|pgvector\|elasticsearch` — determines which adapter `get_vector_store()` instantiates |
| `EMBEDDING_PROVIDER` / `EMBEDDING_DIMENSIONS` | Must stay consistent per-dataset; changing dimensions after documents are indexed requires a full reindex (the vector collection is dimension-locked at creation) |
| `API_KEYS` | Comma-separated. Empty = auth disabled (dev mode) |
| `RATE_LIMIT_PER_MINUTE` | Per API-key (or per-IP if no key) |

---

## 12. Known limitations (carried over honestly, not hidden)

These are real gaps, not hypothetical edge cases — listed so you know what to verify before depending on them in production:

1. **LangGraph agent path (`agents/graph.py`) has zero automated test coverage.** It's the highest-risk module — `langgraph`/`langchain-core` tool-calling APIs (`bind_tools`, `ToolNode`, `usage_metadata`) have shifted across versions before. Smoke-test this manually against your actual installed `langgraph`/`langchain` versions before relying on it.
2. **Celery tasks (`worker/tasks/ingestion.py`) were never executed**, not even against mocks. The pipeline functions they call (`run_ingestion_pipeline`) are tested directly; the Celery wrapping (retry policy, `asyncio.run` inside a worker process) is not.
3. **Weaviate, Milvus, and Elasticsearch adapters have zero test coverage.** Only Qdrant has been exercised, and only via an in-memory contract test (no real Qdrant server was hit either).
4. **`docker-compose.yml` has never been run.** Image builds, inter-service healthchecks, and the `minio_init` bucket-creation step are unverified.
5. **The Alembic migration (`0001_initial_schema.py`) has never been applied to a real Postgres database.** It's been validated only by reading it carefully and by the model definitions it mirrors passing SQLite-backed tests (which don't exercise Postgres-specific DDL like the `JSONB` variant or FK `ondelete` behavior).
6. **`scripts/load_test.py` has never been run against a live deployment.** The PRD's latency targets (search p95 <1s, chat p95 <3s) are unvalidated.
7. **BM25 sparse retrieval doesn't scale past ~50k chunks per dataset** (see Section 6) — it rebuilds an in-memory index from Postgres on every call.
8. **SSRF URL validation exists but isn't wired into the URL-ingestion path yet** (see Section 9).
9. **Cost table (`COST_TABLE` in `services/llm/base.py`) is a static snapshot** and will go stale as vendors change pricing.

None of these block local development or testing — the 143 unit tests cover ingestion parsing/chunking/metadata, retrieval (dense/sparse/hybrid/rerank/compress), all 6 LLM adapters (mocked), the vector store contract, repositories, config, and security/validation logic. They're listed because shipping code with unverified integration points and calling it "production-ready" without flagging them would be dishonest.

---

## 13. Running it

```bash
# Unit tests only - no Docker required
pip install -r requirements-test.txt
SECRET_KEY="dev-secret-key-32-characters-long" \
DATABASE_URL="postgresql+asyncpg://u:p@localhost/db" \
ENVIRONMENT="test" \
pytest tests/unit -v

# Full stack
cp .env.example .env          # fill in at least one LLM provider key
docker compose up -d
pip install -e ".[dev]"
make migrate
make dev                      # http://localhost:8000/docs
```
