.PHONY: install dev lint format typecheck test test-unit test-integration migrate docker-up docker-down clean

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	uv sync --extra dev

# ── Dev server ────────────────────────────────────────────────────────────────
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check app tests

format:
	ruff format app tests
	ruff check --fix app tests

typecheck:
	mypy app

check: lint typecheck

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

test-fast:
	pytest tests/unit -v --no-cov

# ── Database ──────────────────────────────────────────────────────────────────
migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

migrate-new:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-rebuild:
	docker compose up -d --build

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache .ruff_cache
