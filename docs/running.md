# Running PyRAG Core Locally — Step-by-Step Guide

This guide covers three setup paths depending on what you're trying to do:

- **Path A** — run the unit test suite only (fastest, no Docker, no API keys)
- **Path B** — run the full stack locally (Docker services + live API you can call)
- **Path C** — manual end-to-end smoke test (upload a doc, search it, chat about it)

Pick the path that matches your goal. Path A takes about 2 minutes. Path B takes about 10–15 minutes the first time.

---

## Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Python 3.12+ | Project targets 3.12 explicitly (uses modern type syntax like `str \| None`) | `python3 --version` |
| Docker + Docker Compose | Postgres, Redis, Qdrant, and MinIO run as containers; Langfuse uses Cloud if configured | `docker --version` |
| pip | Package install | `pip --version` |
| ~3GB free disk | `uv sync --extra dev` pulls in `sentence-transformers` + `torch` for local reranking | `df -h` |

You do **not** need Docker for Path A. You do **not** need an LLM API key for Path A or for ingestion-only testing in Path B.

---

## Path A — Unit tests only (no Docker, no API keys)

This runs the unit test suite using SQLite in-memory and mocked external services. Nothing real is called.

```bash
cd pyrag-core

# 1. Install the project dependencies.
uv sync --extra dev

# 2. Run the suite
SECRET_KEY="dev-secret-key-32-characters-long" \
DATABASE_URL="postgresql+asyncpg://u:p@localhost/db" \
ENVIRONMENT="test" \
pytest tests/unit -v
```

**Why those three env vars are required even though no real Postgres is touched:** `Settings` (`app/core/config.py`) validates `secret_key` has a minimum length and `database_url` is present at import time — these are dummy values satisfying that validation; `ENVIRONMENT=test` is what switches the DB engine to `NullPool`/SQLite-compatible mode for the test fixtures.

**Expected output:** all unit tests pass in a few seconds. If you see import errors, re-run `uv sync --extra dev` and confirm it completed without errors.

To run a single test file instead of the whole suite:
```bash
pytest tests/unit/test_retrieval.py -v
```

To see what's NOT covered by this path: read `docs/ARCHITECTURE.md` section 12 — the LangGraph agent, Celery tasks, and three of the five vector store adapters have zero coverage here because they require live infrastructure or heavier dependencies.

---

## Path B — Full stack (Docker + live API)

### Step 1 — Configure environment

```bash
cd pyrag-core
cp .env.example .env
```

Open `.env` and set **at minimum one** LLM provider key, or chat/agent endpoints will fail at request time (ingestion and search work without any LLM key — only generation needs one):

```bash
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

Leave everything else at its default for local development. `SECRET_KEY` is pre-filled with a dev placeholder — fine for local use, but `Settings` will reject it if you ever set `ENVIRONMENT=production` (see `core/config.py`'s production validator).

### Step 2 — Start infrastructure

```bash
docker compose up -d
```

This starts 8 containers: `api`, `celery_worker`, `celery_beat`, `postgres`, `redis`, `qdrant`, `minio`, `minio_init` (one-shot bucket creator). First run pulls images and builds the app image — expect a few minutes.

If you want Langfuse traces, set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env` to point at your Langfuse Cloud workspace.

Check everything is healthy:
```bash
docker compose ps
```

All services should show `healthy` or `running` (note: `minio_init` is expected to exit with code 0 after creating buckets — that's success, not a crash).

If `api` keeps restarting, check its logs:
```bash
docker compose logs -f api
```

### Step 3 — Run database migrations

The `api` container does NOT auto-migrate on startup — you run this explicitly:

```bash
docker compose exec api alembic upgrade head
```

or, if you have the project installed locally too:
```bash
uv sync --extra dev
make migrate
```

This creates all 6 tables (`datasets`, `documents`, `chunks`, `providers`, `analytics`, `jobs`) via `app/db/migrations/versions/0001_initial_schema.py`.

### Step 4 — Verify the API is up

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0"}
```

Interactive API docs: open `http://localhost:8000/docs` in a browser.

### Step 5 — Check whether auth is required

```bash
grep API_KEYS .env
```

If `API_KEYS` is unset or empty in your `.env`, authentication is **disabled** (dev-mode default — see `core/middleware/auth.py`) and you can call the API directly. If you've set `API_KEYS=somekey123`, include it on every request:

```bash
curl -H "X-API-Key: somekey123" http://localhost:8000/api/v1/datasets
```

The examples below assume auth is disabled. Add the header if you've enabled it.

---

## Path C — Manual end-to-end smoke test

This walks through the full flow once, by hand, so you can confirm the whole pipeline actually works on your machine — not just that the server starts.

### 1. Create a dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-dataset",
    "description": "smoke test",
    "chunk_strategy": "recursive",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536
  }'
```

Save the `id` from the response — you'll need it for every following call. Export it for convenience:
```bash
export DATASET_ID="<id from response>"
```

### 2. Upload a document

```bash
echo "PyRAG Core is a Python-first RAG platform. It supports hybrid retrieval combining dense vector search with BM25 sparse retrieval, fused via reciprocal rank fusion." > /tmp/test.txt

curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "dataset_id=$DATASET_ID" \
  -F "file=@/tmp/test.txt"
```

This returns `202 Accepted` immediately with a `document_id` and `job_id` — ingestion happens asynchronously in the Celery worker.

### 3. Poll job status until indexed

```bash
export JOB_ID="<job_id from previous response>"

curl http://localhost:8000/api/v1/documents/jobs/$JOB_ID
```

Repeat until `"status": "completed"` and `"progress": 100`. For a one-line `.txt` file this should take a few seconds. If it gets stuck at `queued`, the Celery worker isn't picking up tasks — check `docker compose logs -f celery_worker`.

If `"status": "failed"`, the response includes `error_message` — this is your fastest path to diagnosing what broke (missing embedding API key is the most common cause here, since embedding happens during ingestion, not just at chat time).

### 4. Search it

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"query\": \"What retrieval methods does it support?\",
    \"mode\": \"hybrid\",
    \"top_k\": 5
  }"
```

You should get back the chunk you uploaded, with a relevance score.

### 5. Chat about it

Requires an LLM API key set in `.env` (Step 1 of Path B):

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"$DATASET_ID\",
    \"message\": \"What retrieval methods does this support?\"
  }"
```

Response includes the generated `answer`, the `sources` it was grounded in, token counts, and cost.

### 6. Confirm analytics captured it

```bash
curl http://localhost:8000/api/v1/analytics
```

`total_requests` should now be ≥2 (one for the search call, one for the chat call), with cost/token data from the chat call.

If all six steps above work, the full pipeline — upload, parse, chunk, embed, index, search, generate, track — is functioning on your machine end-to-end.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv sync --extra dev` fails to find package | Using an old copy without the `[tool.hatch.build.targets.wheel]` fix | Confirm `pyproject.toml` has `packages = ["app"]` under that section |
| `pytest` fails with `asyncpg` import error | Environment is missing project dependencies | Re-run `uv sync --extra dev` |
| `api` container restarts in a loop | Migrations haven't been run, or a required env var is missing | `docker compose logs api`; run Step 3 above |
| Upload succeeds but job stays `queued` forever | Celery worker isn't running or can't reach Redis | `docker compose ps` — confirm `celery_worker` is `running`; check `docker compose logs celery_worker` |
| Job fails with an embedding error | No `OPENAI_API_KEY` (or your configured `EMBEDDING_PROVIDER`'s key) set | Add the key to `.env`, `docker compose restart celery_worker api` |
| Chat endpoint returns 502 | LLM provider key missing/invalid | Check `.env` has a valid key for whichever provider is being resolved (see `docs/ARCHITECTURE.md` section 7 for resolution order) |
| `429 Too Many Requests` immediately | Rate limit hit, or testing in a tight loop | Check `RATE_LIMIT_PER_MINUTE` in `.env`; default is 60/min per key or IP |
| Search returns empty results for a document you know was indexed | Vector collection dimension mismatch after changing `EMBEDDING_DIMENSIONS` mid-stream | Collections are dimension-locked at creation — delete and recreate the dataset, or keep dimensions consistent |

---

## Switching components

**Different vector store** (default: Qdrant):
```bash
# .env
VECTOR_PROVIDER=pgvector   # or weaviate, milvus, elasticsearch
```
For `weaviate`/`milvus`/`elasticsearch`, also install the matching extra with `uv sync --extra weaviate` etc. `pgvector` needs no extra — it uses `asyncpg` directly, already a core dependency.

**Different embedding provider** (default: OpenAI):
```bash
# .env
EMBEDDING_PROVIDER=sentence-transformers   # local, no API key needed
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384                    # must match the model's actual output dim
```
Requires `sentence-transformers` installed (`uv sync --extra dev` pulls it in already).

**Different LLM for chat** — either set the env var default, or override per-request:
```bash
curl -X POST http://localhost:8000/api/v1/chat -d '{
  "dataset_id": "...", "message": "...",
  "provider": "anthropic", "model": "claude-sonnet-4-6"
}'
```

---

## Stopping everything

```bash
docker compose down              # stop containers, keep data volumes
docker compose down -v           # stop containers AND delete all data (fresh start)
```
