# ---------------------------------------------------------------------------
# RAGOps developer commands.
#
# On Windows (the primary environment) run these through `make` from Git Bash
# or WSL. Every target also has a plain PowerShell equivalent in
# scripts/dev.ps1, so nothing here is required to use the project.
# ---------------------------------------------------------------------------

SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/Scripts/python.exe
PIP   := $(VENV)/Scripts/pip.exe
ALEMBIC := cd backend && ../$(VENV)/Scripts/alembic.exe -c alembic.ini

.DEFAULT_GOAL := help
.PHONY: help install install-ml infra infra-down migrate migrate-down seed \
        backend frontend test test-backend test-frontend lint fmt typecheck \
        demo-data demo-rag demo-agent verify api-contract clean reset-db

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
install: ## Install backend core + ML dependencies into .venv
	$(PIP) install -r backend/requirements.txt -r backend/requirements-ml.txt

install-ml: ## Install only the ML/retrieval extras
	$(PIP) install -r backend/requirements-ml.txt

infra: ## Start Postgres + Redis (+ Qdrant) in Docker
	docker compose up -d postgres redis qdrant

infra-down: ## Stop the infrastructure containers (data volumes are kept)
	docker compose down

# ------------------------------------------------------------ migrations
migrate: ## Apply all migrations (alembic upgrade head)
	$(ALEMBIC) upgrade head

migrate-down: ## Roll back one migration
	$(ALEMBIC) downgrade -1

reset-db: ## Drop every table and re-apply migrations from scratch
	$(ALEMBIC) downgrade base && $(ALEMBIC) upgrade head

seed: ## Seed the reference Model rows used by pricing and analytics
	cd backend && ../$(VENV)/Scripts/python.exe ../scripts/seed_models.py

# ------------------------------------------------------------------- run
backend: ## Run the FastAPI backend with reload on :8000
	cd backend && ../$(VENV)/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

frontend: ## Run the Vite dev server on :5173
	cd frontend && npm run dev

# ----------------------------------------------------------------- tests
test: test-backend test-frontend ## Run every test suite

test-backend: ## Run the backend unit + integration suite
	cd backend && ../$(VENV)/Scripts/python.exe -m pytest -q

test-frontend: ## Run the frontend vitest suite
	cd frontend && npm run test

# ------------------------------------------------------------- quality
lint: ## Ruff check + mypy
	cd backend && ../$(VENV)/Scripts/python.exe -m ruff check app tests
	cd backend && ../$(VENV)/Scripts/python.exe -m mypy app

fmt: ## Auto-fix ruff findings
	cd backend && ../$(VENV)/Scripts/python.exe -m ruff check --fix app tests

typecheck: ## TypeScript check for the frontend
	cd frontend && npm run typecheck

# ------------------------------------------------------------------ demo
demo-data: ## Generate 10,000+ synthetic traces (never hardcoded dashboard numbers)
	$(PY) scripts/generate_demo_data.py

demo-rag: ## Run the demonstration RAG support bot end to end
	cd evaluation && ../$(VENV)/Scripts/python.exe -m rag_demo.app --query "How do I reset my password?"

demo-agent: ## Run the multi-step agent example (LangGraph if installed, else the built-in runner)
	cd evaluation && ../$(VENV)/Scripts/python.exe -m agent_demo.app

# -------------------------------------------------------------- verify
verify: migrate test ## Migrate then run the whole test suite

api-contract: ## Regenerate docs/api-contract.json from the app's own OpenAPI
	$(PY) scripts/export_api_contract.py

clean: ## Remove caches and build output
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -rf frontend/dist
