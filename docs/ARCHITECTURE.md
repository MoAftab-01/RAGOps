# Architecture

How RAGOps is put together, and why. This document assumes you have read the [README](../README.md) and know what the product does.

---

## Table of contents

- [The core idea](#the-coreidea)
- [Request lifecycle](#request-lifecycle)
- [Layered structure](#layered-structure)
- [Data model](#data-model)
- [The four invariants](#the-four-invariants)
- [Metrics: how they are computed](#metrics-how-they-are-computed)
- [Multi-tenancy](#multi-tenancy)
- [Caching](#caching)
- [Retrieval stack](#retrieval-stack)
- [Frontend](#frontend)
- [Testing strategy](#testing-strategy)
- [Extension points](#extension-points)
- [Known trade-offs](#known-trade-offs)

---

## The core idea

RAGOps is an **observability system**, not a chatbot. The distinction is architectural, not cosmetic: it never generates an answer for a user, so almost everything in it is deterministic.

That single constraint explains most of the design:

- Because it doesn't generate answers, it doesn't need an LLM to measure anything.
- Because it measures, it must store raw inputs and outputs — `traces.input_text` and `traces.output_text` exist because you cannot debug a RAG system that discarded what it saw.
- Because it stores raw text, tenancy is not optional. Any company's traces are readable by anyone who can guess an id, so isolation had to be structural.

---

## Request lifecycle

A trace arrives at `POST /api/traces`:

```
1. Principal resolution        X-API-Key → Principal (org, key_id, scopes)
2. WriteGuard                  must have "ingest" scope → else 401
3. Resolve application         name → Application.id, scoped to the org
4. Persist                     Trace + nested LLMCalls/RetrievalCalls/Spans/Documents
5. Invalidate cache            drop the "analytics" namespace
6. Return 201                   the stored trace_id
```

Three things are worth noting:

**The principal is resolved before anything else.** Every request knows who is asking before it touches the database, so no query can accidentally run unscoped.

**Application resolution is org-scoped.** A company posting `application="customer-support-bot"` gets a loud error if that name belongs to someone else, rather than silently landing in their row.

**Cache invalidation is global, not per-org.** One company's write drops another's cached dashboard. That is over-invalidation, which is safe. The dangerous direction — a stale hit served *across* orgs — is closed by including `organization_id` in the cache key.

---

## Layered structure

```
app/api/v1/          16 route modules. Resolve, validate, delegate, return.
app/schemas/         Pydantic request/response shapes. The wire contract.
app/services/        Business logic and orchestration. No HTTP concepts.
app/repositories/    All SQL. The only layer that touches the ORM.
app/ml/              Embeddings, chunking, retrieval, reranking, hybrid search.
app/evaluation/      Metric definitions, evaluators, regression comparison.
app/models/          SQLAlchemy models.
app/core/            config, database, security, cache, logging, middleware.
app/providers/       Model backends (Ollama, and where others would go).
```

The dependency direction is strictly downward. `app/repositories/common.py` never imports from `app/api`. `app/api` never imports an ORM model except to build a `select()`. Nothing in `app/ml` or `app/evaluation` knows a request exists.

### What lives where, concretely

| Layer | Example | Rule |
|---|---|---|
| Route | `answer_eval.py` | 20 lines max of logic, then delegate |
| Service | `recommendation_service.py` | Owns the rule set and the evidence |
| Repository | `analytics_repo.py` | Owns every `select()` and aggregation |
| Helper | `repositories/common.py` | `apply_tenant_filter` — the one place scoping happens |

`apply_tenant_filter` has **25 call sites** today. That number is a warning, not a
score: it is exactly the kind of count that goes stale, which is why
`tests/test_scoping.py` asserts it structurally instead of counting.

---

## Data model

**Core telemetry** (`app/models/core.py`):

```
Application ─┬─ Trace ─┬─ LLMCall
             │         ├─ RetrievalCall ── RetrievedDocument
             │         └─ Span
             └─ User
```

One `Trace` is one user request. Its children are the three things a RAG pipeline does: retrieve, generate, and time. `Span` is the generic timing primitive that can nest.

**Analytics** (`app/models/analytics.py`):

```
Document, EvaluationRun ── EvaluationResult
                    └───── AnswerEvaluation
Anomaly, OptimizationRecommendation
Experiment ── ExperimentVariant
```

**Tenancy** (`app/models/tenancy.py`):

```
Organization ── ApiKey
Application.organization_id  (nullable, ON DELETE SET NULL)
```

`ON DELETE SET NULL` on the application→organization link is a deliberate deviation from the `CASCADE` used elsewhere. **Deleting a company must not delete its telemetry** — that data is often the company's own compliance record.

### Conventions

- **UUID primary keys** everywhere, assigned in Python, never `gen_random_uuid()` — the existing code style, and it keeps migration scripts free of Postgres extensions.
- **No Postgres enum types.** Enums are Python `StrEnum` mapped to `String(32)`. Adding a value is then a code change, not a migration.
- **`created_at`/`updated_at`** from a shared `TimestampMixin`.
- **JSONB** for genuinely open-ended columns (`metrics`, `config`, `claims`), typed columns for anything the code actually filters or sorts on.

---

## The four invariants

These are the rules the codebase is organised around. Each has a test that fails if it is broken.

### 1. An unmeasured thing is never zero

This is the invariant the whole project is about. Zero is a score; `None` means "not run".

```python
# Wrong — a judge that never ran reads as the worst possible score
judge_faithfulness = 0.0

# Right
judge_faithfulness = None
```

The same rule governs unevaluated experiment arms, unlabelled dataset rows, and metrics a run did not record. `tests/test_route_contracts.py` asserts an unjudged answer run writes no `judge_*` keys at all.

### 2. No metric is computed by an LLM

Precision, Recall, MRR, nDCG, token counts, cost, latency — all arithmetic over recorded values. The one exception is the optional LLM judge, whose output is quarantined in `judge_*` fields and can never fall back into the deterministic columns.

### 3. All SQL goes through the ORM, and scoping happens in one place

There is no string-built SQL anywhere. Tenant filtering is `apply_tenant_filter(stmt, window, column)` and a sibling `tenant_predicates(...)` for lookups that have no time window.

`tests/test_scoping.py` walks the source with `ast` and fails if a bare `application_id is not None` comparison appears outside the one helper allowed to write it. It is the answer to "did I convert all 25 sites?" that does not depend on remembering to grep.

### 4. No business logic in routes

A route resolves the application, validates the request, delegates, and returns. This is what makes the evaluators reusable from scripts and the CLI, which is why `evaluation/` drives the same code paths the API does.

---

## Metrics: how they are computed

All in `app/evaluation/metrics.py`.

**Retrieval** — for each query with labelled relevant documents, at each k:

| Metric | Formula | Notes |
|---|---|---|
| Precision@k | relevant retrieved ÷ k | Denominator is k, not the number retrieved |
| Recall@k | relevant retrieved ÷ total relevant | |
| F1 | harmonic mean of the two | |
| MRR | 1 ÷ rank of first relevant | 0 if none found |
| nDCG@k | DCG ÷ ideal DCG | Uses graded relevance when supplied |
| Hit rate | ≥1 relevant in top k | The loosest useful signal |

**Answer quality** — `answer_evaluator.py`:

| Metric | How | Judge? |
|---|---|---|
| Faithfulness | Split answer into claims; for each, find the best-matching context passage by embedding similarity; score the fraction above threshold | No |
| Context relevance | How much of the retrieved context the answer actually used | No |
| Answer relevance | Similarity between question and answer | No |
| Citation coverage | Fraction of claims that cite a source | No |
| `judge_faithfulness` | LLM verdict on whether claims are supported | **Yes** |
| `judge_answer_relevance` | LLM verdict on whether the answer addresses the question | **Yes** |

The deterministic faithfulness score is a **similarity proxy, not a judgement**. It cannot tell you an answer is well-written or actually correct — only that its claims resemble the supplied context. The `judge_*` columns exist because that gap is real, and they are quarantined because a judge that hallucinated a number would be worse than no judge.

**Regression comparison** — `regression.py` deltas two *recorded* runs and lists the configuration keys that differ. It reports "`rerank` changed from `true` to `false`", never "reranking was disabled, which is why recall dropped." The second sentence is a hypothesis.

---

## Multi-tenancy

`applications.organization_id` is the single enforcement seam. Traces, LLM calls, retrievals, evaluations, and recommendations all reach their company *through the application*, so one predicate closes every read at once.

**Principal resolution** (`app/core/security.py`) has five outcomes:

| Presented | Resolves to |
|---|---|
| No header | Platform — reads stay open, the local-dev hatch |
| Any key + empty `RAGOPS_API_KEY` | Platform — the same hatch, unchanged |
| Matches `RAGOPS_API_KEY` | Platform, `key_id=None` |
| Matches an `ApiKey` row | That key's organization |
| Matches nothing, revoked, or expired | **401** |

The last row is load-bearing: **a presented key is authoritative.** A wrong key fails loudly rather than silently widening the request to platform access. That is what lets org scoping bind on reads without adding a required parameter to 45 routes.

**Key storage.** `secrets.token_urlsafe(32)` gives ~256 bits from the OS CSPRNG, so there is nothing to brute-force and slow hashing would defend nothing. The plaintext is hashed with HMAC-SHA256 keyed by `API_KEY_PEPPER` and looked up by that hash on a UNIQUE index — the hash *is* the identity, so no salt column is needed either.

**Honest limitation:** `WHERE key_hash = :presented` is an index probe, not a constant-time comparison. Wrapping the hash in argon2 would not help and would cost a dependency. `hmac.compare_digest` is retained and load-bearing on the legacy `ragops_api_key` path, which is the single-row comparison it exists for.

---

## Caching

Redis, namespace `ragops:{namespace}:{sha256(json parts)[:24]}`.

Three rules:

1. **`organization_id` is part of the cache key.** Without it, two companies reading the same window share a cached entry. This is the one line between tenants and each other's dashboards, and it is why `cache_parts` in `app/services/window.py` is worth reading carefully.
2. **Invalidation is global per namespace.** Over-invalidation, which is safe.
3. **The cache is an optimisation, not a dependency.** `app/core/cache.py` opens with the design rule: *every helper degrades to calling the underlying function when Redis is unavailable*. `get_redis()` returns `None` on construction failure, `ping()` returns `False` rather than raising, and each of `cached` / `set_cached` / `cached_call` treats a `RedisError` as a miss. So the backend boots and answers correctly with Redis down — it just recomputes every aggregate on every request.

The consequence worth stating: **there is no in-process fallback cache, and that is deliberate.** A cache that silently degraded to a per-process dict would serve one tenant another's data the moment the app ran with more than one worker. Losing the cache is recoverable; serving a stale cross-tenant dashboard is not.

---

## Retrieval stack

`app/ml/` implements several retrievers behind one interface:

| Module | What it does |
|---|---|
| `chunking.py` | Splits documents on separators, respecting size limits |
| `embeddings.py` | sentence-transformers, cached |
| `vector_store.py` | FAISS or Qdrant, persisted to disk |
| `bm25.py` | Sparse lexical retrieval |
| `hybrid.py` | Weighted fusion of the two (RRF or score-based) |
| `reranker.py` | Cross-encoder reranking of top-k |
| `retriever.py` | The facade the rest of the system uses |
| `anomaly.py` | Statistical detection over recorded values |

Hybrid retrieval is the default in the demo because it reflects a real finding: pure vector search misses exact-match terms (product IDs, error codes), and pure BM25 misses paraphrase. Neither alone is good enough.

---

## Frontend

React 18 + Vite + TypeScript, TanStack Query v5 for server state.

**The rule:** the frontend holds no derived analytics. Every number on screen comes from a query response. There is no client-side aggregation of metrics, because a dashboard that computes its own averages from partial data will disagree with the API, and the API is the one that is right.

**API client** (`src/lib/api.ts`) is a thin typed wrapper over `fetch` with one shared request path, so auth headers, error handling, and JSON parsing happen in exactly one place.

**Local storage holds two things:** the active API key (`ragops.apiKey`) and a per-organization key map (`ragops.apiKeys`). The map is additive — writing an entry does **not** switch the active key, so storing a second company's credential cannot silently repoint the console at another tenant's data. Switching is an explicit call.

**Tests** use Vitest and Testing Library against the real components, with the API client spied via `vi.spyOn` rather than replaced by a mock module — so a change to the real client's shape breaks the tests.

---

## Testing strategy

**422 backend tests.** pytest with real Postgres and Redis. Integration tests drive the actual ASGI app through `httpx.ASGITransport`, so `dependency_overrides` injects a managed session and teardown deletes exactly what the test wrote.

**152 frontend tests.**

**Two structural guards.** These are unusual and worth understanding, because they encode a decision rather than testing behaviour:

| Test | What it prevents |
|---|---|
| `test_scoping.py` | A new query added without a tenant filter, which fails silently as an empty dashboard rather than an error |
| `test_route_contracts.py` | A route typing `principal: Principal` instead of `TenantPrincipal`, which turns an injected parameter into a required body field and 422s every caller |

**A note on what tests did not catch.** All 416 original tests passed while four routes were completely broken — a handler shadowing an import, two fabricated `WindowFilter`s, and four `Principal`-as-body-field signatures. None were covered because no test exercised those happy paths. They were found by driving the live API with curl. The lesson shaped the structural guards: shape-level bugs need shape-level tests.

---

## Extension points

| To add | Do this |
|---|---|
| A new provider (OpenAI, Bedrock) | Implement the interface in `app/providers/`, add a `settings` block |
| A new metric | Add it to `app/evaluation/metrics.py` and its `metric_direction` |
| A new anomaly type | Extend `AnomalyType` — it is a `StrEnum`, no migration needed |
| A new route | Add to `app/api/v1/`, use `TenantPrincipal`, and put scoping in `apply_tenant_filter` |
| A new vector store | Implement the `vector_store` interface; `retriever.py` picks by config |

---

## Known trade-offs

**`application_id is None` means "all applications."** A `WindowFilter` with no application returns every application in the tenant. This is the right default for a dashboard but means a route that should require a specific application must validate it explicitly.

**Aggregations are computed in SQL, not in Python.** Percentiles use Postgres window functions, which keeps large windows fast but means some metrics are awkward to express. Where that happened, the code chose the readable formulation and accepted the cost.

**Redis is optional at runtime, and its absence is invisible in the numbers.** With Redis down every aggregate is recomputed per request, so latency rises while every metric stays correct. Worth knowing when the dashboard is slow: check `/api/health`'s `redis` component before suspecting the queries.

**No background job runner.** Evaluation runs execute synchronously in the request. A large labelled dataset will hold a connection for its duration. A production deployment would queue these.

**The `evaluation/` demos are not part of the API.** They are standalone applications that happen to use the same library code. That is intentional — it proves the evaluators are reusable — but it means a bug in a demo is not caught by the API tests.