# RAGOps

**Observability, evaluation, and cost analytics for RAG applications.** Runs entirely on your machine — no paid APIs, no external services, no telemetry leaving your network.

Point your RAG app at RAGOps and it tells you where your chatbot is going wrong: slow, expensive, or confidently making things up.

```
Your RAG app  ──POST /api/traces──▶  Postgres  ──▶  Dashboard
                                              "12% of answers contain
                                               claims not in the documents"
```

---

## Table of contents

- [What it does](#what-it-does)
- [Why the numbers are trustworthy](#why-the-numbers-are-trustworthy)
- [Quick start](#quick-start)
- [Sending your first trace](#sending-your-first-trace)
- [The dashboard](#the-dashboard)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Multi-tenancy](#multi-tenancy)
- [Configuration](#configuration)
- [Honest limitations](#honest-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## What it does

A user's question hits your RAG application and three things happen: it **retrieves documents**, it **calls an LLM**, and it **returns an answer**. RAGOps observes those three things and records them.

That gives it six capabilities:

| Capability | What it answers |
|---|---|
| **Trace observability** | What exactly happened for this one user request? |
| **Cost & token analytics** | Where are the tokens and the money actually going? |
| **Retrieval quality** | Is the retriever returning the right documents at k=5? |
| **Answer evaluation** | Is the answer faithful to the documents it was given? |
| **Anomaly detection** | What is statistically unusual right now? |
| **Optimization** | What should I change, and what evidence says so? |

It is **not** a chatbot. It does not generate answers for your users; it measures the ones they already produce.

### The data model, in one paragraph

A **`Trace`** is one user request. It has many **`RetrievalCall`**s (what was searched, what came back, at what rank), many **`LLMCall`**s (model, tokens, cost, latency), and many **`Span`**s (timings). From those three children you can ask "which documents did it read?" and "which model did it call?" without guessing.

---

## Why the numbers are trustworthy

This is the part most observability tools get wrong, so it is worth stating plainly.

**No metric is computed by an LLM.** Precision, Recall, MRR, nDCG, token counts, cost, and latency are all plain arithmetic over real recorded numbers. An LLM is consulted for exactly one optional thing — judging whether an answer is *faithful* to its context.

**That judge is off by default.** When `use_llm_judge` is false (the default), the judge fields return `null` — never `0.0`. A disabled judge and a judge that scored your batch at zero are completely different findings, and only one of them is a score. The UI shows "not run."

**Nothing is hardcoded.** Dashboard numbers come from queries over the database. There is no fixture that says "p95 latency: 240ms." Delete all the traces and the dashboard correctly shows zeros.

**Results are reproducible.** The deterministic metrics use sentence-transformers embeddings and a fixed model, so the same input produces the same score on every run. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how faithfulness is computed without a judge.

---

## Quick start

**Prerequisites:** Python 3.11+ (3.12 recommended), Node 18+, and Docker for Postgres and Redis.

```bash
git clone <your-repo-url> ragops
cd ragops

# 1. Create the virtualenv, then install backend + ML dependencies
python -m venv .venv
make install

# 2. Start Postgres + Redis (and Qdrant if you want it)
make infra

# 3. Configure. Copy the example and edit if you need to.
cp .env.example .env

# 4. Create the database schema
make migrate

# 5. Load the reference model/pricing table
make seed

# 6. Generate 10,000+ synthetic traces so the dashboard has something to show
make demo-data

# 7. Start the backend (:8000) and frontend (:5173)
make backend      # terminal 1
make frontend     # terminal 2
```

Open **http://localhost:5173**.

### Windows note

The Makefile targets assume a Git Bash / POSIX shell. On Windows PowerShell, run the underlying commands directly:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r backend/requirements.txt -r backend/requirements-ml.txt
docker compose up -d postgres redis qdrant
cd backend; ..\.venv\Scripts\python.exe -m alembic upgrade head
cd backend; ..\.venv\Scripts\python.exe ..\scripts\seed_models.py
cd backend; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
cd frontend; npm install; npm run dev
```

### All Makefile targets

| Target | What it does |
|---|---|
| `make install` | Install backend core + ML dependencies |
| `make infra` / `make infra-down` | Start/stop Postgres, Redis, Qdrant |
| `make migrate` | Apply Alembic migrations |
| `make reset-db` | Drop everything and re-migrate |
| `make seed` | Load the reference `Model` rows used for pricing |
| `make backend` / `make frontend` | Run the dev servers |
| `make test` | Every test suite |
| `make lint` | `ruff check` + `mypy` — **`ruff` is clean; `mypy` reports 58 known errors** |
| `make typecheck` | TypeScript check |
| `make verify` | Migrate, then run both test suites (not lint) |
| `make api-contract` | Regenerate `docs/api-contract.json` from the app's own OpenAPI |
| `make demo-data` | Generate 10,000+ synthetic traces |
| `make demo-rag` | Run the RAG support bot end to end |
| `make demo-agent` | Run the multi-step agent example |

---

## Sending your first trace

### With the Python SDK

```python
from ragops import RAGOps

client = RAGOps(
    endpoint="http://localhost:8000",
    api_key="dev-key",
    application="customer-support-bot",
)

def answer(question: str) -> str:
    with client.trace(name="answer_question") as trace:
        docs = retriever.search(question, k=5)
        trace.log_retrieval(
            query=question,
            documents=[{"id": d.id, "text": d.text, "score": d.score} for d in docs],
            top_k=5,
        )

        prompt = build_prompt(question, docs)          # your prompt, your call
        answer = llm.generate(prompt)

        trace.log_generation(
            model="llama3.1",
            input_tokens=estimate_tokens(prompt),      # plain arithmetic, never an LLM
            output_tokens=estimate_tokens(answer),
            answer=answer,
        )
        trace.set_output(answer)

    return answer
```

`estimate_tokens` is the SDK's own counter (`from ragops.tokens import estimate_tokens`) — a string operation, not a model call. Count however your system already counts; the only rule is that the number is measured, not guessed.

`client.trace(...)` is a context manager: on a clean exit the trace completes with status `success` and flushes; if the block raises, the error is recorded, the status becomes `error`, and the exception propagates unchanged.

The `Trace` object also exposes `log_span(...)`, `log_agent_step(...)`, `set_error(...)`, `set_tags(...)`, and `set_metadata(...)`. Telemetry is buffered and flushed in the background, so instrumentation adds no latency to the traced call.

### With plain HTTP

```bash
curl -X POST http://localhost:8000/api/traces \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "application": "customer-support-bot",
    "trace_id": "example-001",
    "input_text": "How do I reset my password?"
  }'
```

`docs/api-contract.json` is generated from the running app's own OpenAPI schema, so it always describes what the server actually does rather than what someone hoped it would do. Regenerate it with `make api-contract`.

---

## The dashboard

| Page | What it shows |
|---|---|
| **Dashboard** | Total calls, tokens, cost, error rate, latency percentiles over a window |
| **Traces** | Every recorded request; drill into one to see its retrievals, generations, and spans |
| **RAG Evaluation** | Retrieval and answer evaluation runs, with per-query results |
| **Token Analytics** | Cost and token trends broken down by model, application, and user |
| **Cost & Quality** | Cost per correct answer, and quality at each operating point |
| **Experiments** | Multi-arm comparisons against a baseline run |
| **Anomalies** | Statistically unusual traces, grouped by kind |
| **Optimization** | Evidence-backed recommendations, with the trace that motivated each |
| **Applications** | Registered applications and their per-app statistics |
| **Settings** | Companies, API keys, runtime configuration |

The Settings page doubles as the multi-tenancy console: create a company, mint its API key, and send telemetry that lands only in that company's data.

---

## Architecture

Full detail is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The shape:

```
┌──────────────┐   HTTP    ┌────────────────────────────────────────┐
│  RAG app     │──────────▶│  FastAPI  /api                          │
│  or SDK      │           │  ├─ routers      routes + validation    │
└──────────────┘           │  ├─ services     orchestration          │
                           │  ├─ repositories  SQL (ORM only)        │
                           │  ├─ evaluation   metrics + metrics      │
                           │  └─ ml           embed/rerank/retrieve  │
                           └────────┬──────────────────┬─────────────┘
                                    │                  │
                              ┌─────▼─────┐      ┌─────▼──────┐
                              │ PostgreSQL│      │  Redis     │
                              │  traces   │      │  cache     │
                              └───────────┘      └────────────┘
```

**Three rules that shape the code:**

1. **No business logic in routes.** A route resolves, validates, delegates, and returns. Anything reusable lives in a service, and all SQL lives in repositories.
2. **All SQL goes through the ORM.** No string-built queries anywhere. Tenant filtering happens in one helper so "unscoped" has exactly one meaning.
3. **An unmeasured thing is never zero.** Absent metrics are `null` or explicitly counted, never defaulted to `0.0`. This is enforced by tests.

---

## Project layout

```
ragops/
├── backend/
│   ├── app/
│   │   ├── api/v1/          16 route modules
│   │   ├── core/            config, database, security, cache, logging
│   │   ├── models/          SQLAlchemy models (incl. tenancy)
│   │   ├── schemas/         Pydantic request/response shapes
│   │   ├── services/        business logic
│   │   ├── repositories/    database access
│   │   ├── ml/              embeddings, chunking, retrieval, reranking
│   │   ├── evaluation/      metrics, evaluators, regression detection
│   │   ├── providers/       Ollama and other model backends
│   │   ├── telemetry/       logging and tracing plumbing
│   │   └── utils/
│   ├── alembic/             migrations
│   ├── tests/               14 test files
│   └── requirements.txt
├── frontend/                React + Vite + TypeScript
│   └── src/{pages,components,lib,test}
├── sdk/python/ragops/       Python SDK (client, tracer, batching, tokens)
├── evaluation/              demo RAG app, agent demo, datasets, benchmarks
├── docs/                    architecture, API contract, verification notes
├── scripts/                 seed, demo data, contract export
├── docker/                  Dockerfiles and compose fragments
├── docker-compose.yml       Postgres + Redis + Qdrant
└── Makefile
```

---

## Testing

```bash
make test          # everything
make test-backend  # 422 tests
make test-frontend # 152 tests
```

**Backend:** pytest with real Postgres and Redis — integration tests drive the actual HTTP app through `ASGITransport` rather than mocking it, so tenant isolation is verified end to end.

**Frontend:** Vitest and Testing Library against the real components with the API client spied, not replaced.

Two guards are worth knowing about, because they encode decisions rather than testing behaviour:

- **`test_scoping.py`** walks the source with `ast` and fails if a bare `application_id is not None` comparison appears anywhere outside the single allowed helper. It is how "did I convert all 25 filter sites?" is answered without trusting a grep.
- **`test_route_contracts.py`** does the same for route signatures — it fails if any route types its principal as the raw `Principal` class instead of the `TenantPrincipal` alias, because that silently turns an injected parameter into a required request-body field.

---

## Multi-tenancy

Each company gets its own API key, and its telemetry stays its own.

```bash
# Platform admin creates a company
curl -X POST http://localhost:8000/api/organizations \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme"}'

# ...and mints a key for it (the plaintext is returned exactly once)
curl -X POST http://localhost:8000/api/organizations/<id>/api-keys \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"name": "production", "scopes": "ingest,read"}'
```

**How isolation works:** `applications.organization_id` is the single enforcement seam. Traces, retrievals, LLM calls, evaluations, and recommendations all reach their company *through the application*, so one predicate closes every read at once.

**Keys are stored hashed.** HMAC-SHA256, keyed by `API_KEY_PEPPER`. The plaintext is returned once at creation and never written anywhere — not to the database, not to the logs, not to the frontend after you copy it.

**A bad key fails loudly.** A presented key that matches nothing returns 401 rather than silently widening to platform access. Omitting the header entirely is still the documented local-dev escape hatch.

---

## Configuration

All configuration is environment variables. `.env` is git-ignored and `.env.example` documents every option, commented out, with no real values. **RAGOps boots with no `.env` at all**, using the defaults in `backend/app/config.py`, which are tuned for the Docker Compose stack — you only need the file when running outside Docker or when changing a default. Precedence is: real environment variables > `.env` > code defaults.

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://ragops:ragops@localhost:55432/ragops` | Database connection string. Host port is `PG_PORT`, **not** the container's internal 5432 |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache backend |
| `RAGOPS_API_KEY` | `dev-key` | The shared platform key |
| `AUTH_ENABLED` | `true` | Set `false` to run with no key at all while developing |
| `CORS_ORIGINS` | `["http://localhost:5173", ...]` | Allowed frontend origins |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Default local model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding backend |
| `ENVIRONMENT` | `development` | `development` \| `test` \| `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

There is also an `API_KEY_PEPPER` setting in `backend/app/config.py`, documented in `.env.example`.

> **Set `API_KEY_PEPPER` before deploying.** The digest is always HMAC-SHA256, but with an empty pepper the key is an all-zero block — so anyone holding a database dump can recompute it without holding the pepper. It is still a perfectly good *identifier*; it stops being a secret-dependent value. `generate_api_key` logs a warning when the pepper is empty. Changing the pepper later invalidates every per-company key already minted.

> **The browser does not read `RAGOPS_API_KEY`.** The dashboard holds a key in `localStorage` under `ragops.apiKey`, set from the Settings page, and sends it as `X-API-Key` on each request. Only `VITE_API_URL` is read from the frontend environment — mint the key in Settings and paste it there.

---

## Honest limitations

Things this project does not do, stated rather than buried:

- **The LLM judge is off by default and optional.** Faithfulness scoring without a judge is a similarity proxy, not a judgement. When the judge is off, the platform says "not run" instead of guessing.
- **Retrieval evaluation needs a labelled dataset.** Real deployments rarely have relevance labels on day one. `evaluation/datasets/` ships a sample; you bootstrap your own.
- **Application names are globally unique.** Two companies cannot both name their bot `support-bot`. This is deliberate — a name collision should be a loud error, not a silent misplacement — but it is a real product constraint.
- **`AUTH_ENABLED=false` re-opens everything** and ignores every per-company key. It is for local development only.
- **The knowledge base starts unindexed,** so `/api/health` reports `degraded` for `vector_store` until you index it.
- **No multi-node deployment story.** The cache is Redis and the database is Postgres, but there is no documented cluster deployment.

---

## Contributing

1. Fork and create a branch.
2. Run `make verify` before opening a PR — it migrates and runs the entire suite.
3. Keep the three architecture rules: no business logic in routes, all SQL through the ORM, and never represent an unmeasured value as zero.
4. Tests are expected for new behaviour, especially around tenancy and metrics.

---

## License

MIT. See [`LICENSE`](LICENSE).

Built as a portfolio project demonstrating production-style RAG observability: real metrics, real evaluation, no fabricated results.