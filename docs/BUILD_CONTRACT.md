# RAGOps — shared engineering contract (read this first)

This document is the single source of truth for every agent working on RAGOps.
It describes the spine that **already exists, is verified, and must not be
rewritten**. Read it completely before writing code.

## 0. Ground rules

**Repo root:** `E:/placement_personal/Projects/RAGOps`
**Python:** 3.12, venv already provisioned at `<root>/.venv`

```bash
# from repo root
.venv/Scripts/python.exe -c "import x"
# from backend/
../.venv/Scripts/python.exe -c "import x"
```

Installed and importable: `fastapi uvicorn pydantic pydantic-settings sqlalchemy
alembic asyncpg psycopg2-binary redis httpx orjson structlog tenacity numpy pandas
scipy scikit-learn xgboost torch sentence-transformers transformers rank-bm25 faiss-cpu
pytest pytest-asyncio pytest-cov ruff mypy`.

**Node:** v24, npm available.

### Verify your work

PostgreSQL is **not** running while you work. You therefore cannot test against
a live database. Verify by running Python and exercising pure logic:

```bash
cd E:/placement_personal/Projects/RAGOps/backend
../.venv/Scripts/python.exe -c "from app.ml.reranker import Reranker; print('ok')"
```

Every module you write must be **importable without a live DB or Redis**. Do not
put `engine.connect()` or `redis.ping()` at import time. Wrap the async
repository calls in plain async functions that are only awaited at request time.

### Non-negotiable engineering rules (from the project brief)

1. **Never hardcode a dashboard metric.** Every number shown in the UI comes from
   a SQL query over recorded telemetry.
2. **Never fabricate results, costs, savings, or root causes.** If a cause is not
   recorded in the data, do not display one.
3. **Never use an LLM for deterministic math.** Token counts come from
   `app.utils.tokens`; Precision/Recall/MRR/NDCG come from `app.evaluation.metrics`;
   anomalies come from scikit-learn. The local LLM is only ever an *additional*
   signal, and its output is stored in separate `judge_*` columns.
4. **No paid APIs.** Ollama on localhost is the only LLM provider required.
5. **No secrets in code.** All config via `app.config.settings`.
6. **No business logic in FastAPI routes.** Routes parse → call a service → return
   a schema. Services own the logic. Repositories own SQL.

## 1. The spine that already exists (DO NOT REWRITE)

```
backend/app/
  config.py              # pydantic-settings `Settings`, `settings` singleton
  core/
    database.py          # async engine, SessionLocal, `get_db()`, `session_scope()`
    cache.py             # redis client, `cached_call(ns, factory, ttl, **parts)`, `invalidate()`
    logging.py           # `get_logger()`, `Timer`, `log_db_query`, `log_evaluation`
    middleware.py        # RequestContextMiddleware, TraceContextMiddleware, APIErrorHandlerMiddleware
    security.py          # `require_api_key`, `verify_api_key`
  models/
    base.py              # `Base`, `UUIDPrimaryKeyMixin`, `TimestampMixin`, `utcnow`
    core.py              # Application, User, Model, Trace, Span, LLMCall, RetrievalCall, RetrievedDocument
    analytics.py         # Document, EvaluationRun, EvaluationResult, AnswerEvaluation, Anomaly,
                         # OptimizationRecommendation, Experiment, ExperimentVariant
    __init__.py          # re-exports everything (import from `app.models`)
  schemas/
    common.py            # Page[T], TimeSeriesPoint, BreakdownItem, ORMModel, HealthResponse
    ingest.py            # TraceCreate, SpanIn, RetrievalIn, RetrievedDocumentIn, GenerationIn, BatchIngest
    trace.py             # TraceSummary, TraceDetail, SpanOut, RetrievalOut, LLMCallOut
    analytics.py         # DashboardOverview, TokenAnalytics, CostAnalytics, LatencyAnalytics,
                         # TokenEfficiencyReport, CostQualityReport, ModelComparison
    evaluation.py        # EvalExample, RetrievalEvaluationRequest, RetrievalMetrics, PerQueryResult,
                         # AnswerEvaluation*, RegressionReport, MetricDelta, ConfigDifference
    insights.py          # AnomalyOut, AnomalyDetect*, RecommendationOut, Experiment*, ApplicationOut, ModelOut
  utils/
    tokens.py            # `estimate_tokens`, `count_tokens`, `count_message_tokens`,
                         # `truncate_to_tokens`, `token_efficiency`
    pricing.py           # `ModelPricing`, `DEFAULT_PRICING`, `calculate_cost`, `is_simulated`,
                         # `pricing_label`, `resolve_window`
```

16 tables are already registered on `Base.metadata`. All 53 schemas already exist
and import cleanly. Run the import check before you start and after you finish:

```bash
cd backend && ../.venv/Scripts/python.exe -c "from app import schemas; from app.models import Base; print(len(Base.metadata.tables), len(schemas.__all__))"
# must print: 16 53
```

### Key spine APIs

```python
from app.config import settings
# settings.database_url, settings.redis_url, settings.ollama_base_url,
# settings.ollama_model (default "qwen2.5:3b"), settings.embedding_model,
# settings.reranker_model, settings.chunk_size, settings.chunk_overlap,
# settings.default_top_k, settings.hybrid_rrf_k, settings.vector_store_backend,
# settings.pricing_enabled, settings.anomaly_contamination, settings.anomaly_min_samples,
# settings.llm_judge_enabled, settings.analytics_cache_ttl_seconds,
# settings.collector_buffer_size, settings.auth_enabled, settings.ragops_api_key

from app.core.database import get_db, session_scope, SessionLocal, engine
from app.core.cache import cached_call, invalidate, get_redis, ping
from app.core.logging import get_logger, Timer, log_db_query, log_evaluation
from app.core.security import require_api_key, verify_api_key
from app.utils.tokens import estimate_tokens, count_tokens, count_message_tokens, token_efficiency
from app.utils.pricing import calculate_cost, pricing_label, resolve_window, DEFAULT_PRICING
```

`resolve_window(window, start, end) -> (start, end, label)` where `window` is one of
`1h|24h|7d|30d|90d`. All datetimes are timezone-aware UTC.

### `token_efficiency` signature (already implemented — use it, don't reimplement)

```python
token_efficiency(
    *, input_tokens: int, context_tokens: int, output_tokens: int,
    retrieved_documents: list[str] | None = None,
    duplicate_document_ids: set[str] | None = None,
) -> dict  # {score, duplicate_document_count, duplicate_ratio, duplicate_tokens,
            #  context_share, output_yield, potential_waste_pct, wasted_tokens}
```

## 2. The file tree you are building INTO

```
backend/app/
  api/v1/          # routers: health, traces, analytics, dashboard, evaluations,
                   #         anomalies, recommendations, experiments, applications, models, rag
  services/        # business logic layer
  repositories/    # SQLAlchemy queries only
  ml/              # embeddings, reranker, chunking, vector store, bm25, hybrid, anomaly
  evaluation/      # metrics, retrieval_eval, answer_eval, judge, pipeline, regression
  providers/       # ollama (+ base adapter so cloud providers can be added later)
  telemetry/       # ingest service + buffered collector
  utils/
backend/tests/     # unit + integration
sdk/python/ragops/ # the client library
evaluation/        # datasets/ (knowledge_base/, retrieval_eval.jsonl), benchmarks/, scripts/
frontend/src/      # React app
docker/ scripts/ docs/ ml/ .github/workflows/
```

## 3. Module responsibilities (one concern per file, no god files)

### `app/repositories/` — SQL only
`application_repo.py`, `trace_repo.py`, `analytics_repo.py`, `evaluation_repo.py`,
`anomaly_repo.py`, `recommendation_repo.py`, `experiment_repo.py`, `model_repo.py`,
`document_repo.py`. Every function takes an `AsyncSession` and returns ORM objects
or plain dicts. **Use SQL aggregations (`func.avg`, `func.percentile_cont`, `date_trunc`)
for anything the dashboard charts — never load rows into Python to average them.**

### `app/services/` — business logic
`trace_service.py`, `analytics_service.py`, `dashboard_service.py`, `token_service.py`,
`cost_service.py`, `latency_service.py`, `efficiency_service.py`, `anomaly_service.py`,
`recommendation_service.py`, `experiment_service.py`, `application_service.py`,
`model_service.py`, `document_service.py`. Services orchestrate repositories + ML +
evaluation and are unit-testable with plain functions.

### `app/ml/` — the retrieval stack
- `embeddings.py` — `Embedder` singleton wrapping `sentence-transformers`. Lazy-loads
  on first use (never at import). `embed(texts: list[str]) -> np.ndarray`, `embed_one()`.
  Batch internally (`settings.embedding_batch_size`). Expose a `dimension` property.
- `chunking.py` — `chunk_text(text, chunk_size, overlap) -> list[str]`, recursive
  character splitting on sentence/paragraph boundaries, no mid-word cuts. Also
  `load_documents(path) -> list[RawDocument]` and `chunk_documents()`.
- `bm25.py` — `BM25Index` wrapping `rank_bm25.BM25Okapi`. `build(documents)`,
  `search(query, top_k) -> list[(doc_id, score)]`. Must rebuild lazily and expose
  `is_built`.
- `vector_store.py` — `VectorStore` interface + `FaissVectorStore` (uses
  `faiss.IndexFlatIP` over L2-normalised embeddings, so inner product == cosine).
  Persist to disk under `ml/models/`. Also provide `QdrantVectorStore` selected by
  `settings.vector_store_backend`; it must degrade with a clear log if Qdrant is
  unreachable rather than crashing the app.
- `reranker.py` — `CrossEncoderReranker` wrapping
  `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers `CrossEncoder`.
  `rerank(query, docs: list[(id, text)]) -> list[(id, score)]` sorted desc.
  `enabled` flag; no-op passthrough when disabled.
- `hybrid.py` — `HybridRetriever`. Runs BM25 and vector search, fuses with
  **Reciprocal Rank Fusion** (`score = Σ 1/(k + rank)`, k = `settings.hybrid_rrf_k`),
  then optionally cross-encoder reranks. Per-stage scores are preserved on each
  result. This is the component the `HybridScorer` in `app/evaluation/metrics.py`
  unit-tests.
- `retriever.py` — `RetrieverService` used by the demo app and by
  `POST /api/rag/query`: orchestrates chunking→embeddings→BM25+vector→rerank→top-k
  and records the configuration it used.
- `anomaly.py` — `AnomalyDetector` using `sklearn.ensemble.IsolationForest`.
  `fit_predict(features: np.ndarray) -> np.ndarray` (scores), plus percentile
  bands. Pure and unit-testable; no DB, no network.

### `app/evaluation/` — the metrics
- `metrics.py` — **pure deterministic functions, no DB, no I/O.** This file is the
  single most important one in the project and is unit-tested against
  hand-verified examples.
  ```python
  precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float
  recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float
  f1_at_k(...) -> float
  reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float
  mrr(retrieved: list[str], relevant: set[str]) -> float
  ndcg_at_k(retrieved, relevant, k, grades: dict[str,int] | None = None) -> float
  hit_rate_at_k(retrieved, relevant, k) -> float
  HybridScorer  # deterministic RRF implementation used in tests
  ```
  `retrieved` is an ordered list of ids, most relevant first. `ndcg_at_k` uses binary
  gains when `grades` is None. **Clamp every result to [0.0, 1.0]** and return 0.0
  (never raise) for an empty `relevant` set or an empty `retrieved` list.
- `retrieval_eval.py` — `RetrievalEvaluator`. Loads `EvalExample` datasets from
  `evaluation/datasets/*.jsonl` (one JSON object per line, with a top-level
  `{"name":..., "examples":[...]}` object form also supported), runs the retriever,
  computes metrics per query, aggregates per K, persists `EvaluationRun` +
  `EvaluationResult` rows.
- `answer_eval.py` — `AnswerEvaluator`. Deterministic, no LLM required.
  - Sentence-split the answer into **claims**.
  - **Faithfulness**: fraction of claims supported by the context. Support is
    measured by content-word alignment *and* embedding cosine similarity against
    the best-matching context sentence; a claim counts as supported when it clears
    `settings.grounding_similarity_threshold` or is covered by a context sentence
    with high word overlap. Report `citation_coverage` (supported claims / total
    claims) and `unsupported_claim_ratio`.
  - **Context relevance**: mean embedding similarity between the question and each
    context passage (cosine of `Embedder` output).
  - **Answer relevance**: embedding similarity between question and answer, combined
    with content-word coverage of the question in the answer.
  - Every result must carry `method="deterministic_embedding"`.
- `judge.py` — `LLMJudge`, **opt-in only**, guarded by `settings.llm_judge_enabled`
  or an explicit request flag. Calls the local Ollama model with a strict JSON
  prompt, parses defensively, and returns `None` on any failure (never raises into
  the deterministic path). Its scores are stored in `judge_*` fields only.
- `pipeline.py` — `EvaluationPipeline` tying evaluator + repository persistence +
  `log_evaluation()` together. Owns `EvaluationRun.status` transitions.
- `regression.py` — `RegressionDetector.compare(baseline_run, candidate_run) -> RegressionReport`.
  Deltas every metric present in both runs' `metrics` dicts; flags regressions using
  per-metric direction (higher-is-better for recall/mrr/ndcg/faithfulness,
  lower-is-better for cost/latency). `config_differences` is a **flat diff of the two
  runs' recorded `config` dicts** — only genuinely different keys. `contributing_config_changes`
  is phrased as "recorded setting X changed from A to B", never as "the cause was X".

### `app/providers/` — LLM access
- `base.py` — `LLMProvider` Protocol / ABC with `generate(messages, **opts) -> LLMResponse`
  and `available() -> bool`. `LLMResponse` has `text`, `input_tokens`, `output_tokens`,
  `latency_ms`, `model`, `provider`, `time_to_first_token_ms`.
- `ollama.py` — `OllamaProvider` using `httpx` against `settings.ollama_base_url`.
  Uses the `/api/chat` endpoint so Ollama returns **exact** `prompt_eval_count` and
  `eval_count` token counts (never estimate those). `available()` is a cheap GET
  `/api/tags` with a short timeout. Stream and non-stream supported.
- `registry.py` — `get_provider(name=None) -> LLMProvider`, resolving to Ollama by
  default. This is the seam a cloud adapter would slot into; do not add one.

### `app/telemetry/` — the write path
- `ingest_service.py` — `IngestService.ingest_trace(session, TraceCreate) -> str`.
  Upserts the `Application` and `User` by name/external id, creates the `Trace`,
  persists spans/retrievals/documents/generations, and **derives missing token counts
  with `count_tokens`**. Computes `estimated_cost` with `calculate_cost`,
  `context_tokens` from retrieved document token counts, and duplicate document ids
  for the efficiency analysis. Recomputes the trace's denormalised token totals from
  its `LLMCall` rows. Idempotent on `(application_id, trace_id)`.
- `collector.py` — `BufferedCollector`: an asyncio background task that batches
  incoming `TraceCreate` payloads (Redis list when available, in-memory fallback)
  and flushes every `settings.collector_flush_interval_seconds` or once
  `settings.collector_buffer_size` items accumulate. Started/stopped from the app
  lifespan. Flush failures log and requeue, never crash the app.

### `app/api/v1/` — thin routers
`health.py dashboard.py traces.py analytics.py evaluations.py anomalies.py
recommendations.py experiments.py applications.py models.py rag.py`
Plus `app/api/v1/__init__.py` exporting an `api_router` that mounts them all under
`settings.api_v1_prefix`. Use the shared `get_window` dependency (parse `window`,
`start`, `end`, optional `application` name → resolves to an application id).

## 4. Where the numbers come from (the anti-fabrication rule)

- Token totals, latency, cost, request counts, error counts → `llm_calls` and
  `traces` tables, aggregated in SQL.
- `retrieval_score` on the dashboard = mean `RetrievalCall.top_score` in the window.
  It is **null** when there is no retrieval data — never substitute a placeholder.
- `faithfulness` / `answer_relevance` on the dashboard = mean of persisted
  `AnswerEvaluation` rows in the window. Null when no evaluation has been run.
  Do not compute them on the fly for the dashboard; they are too expensive and the
  evaluation run is the unit of record.
- `token_efficiency` = mean of `token_efficiency(...)` results computed at ingest
  time and stored in the trace's `metadata["token_efficiency"]` dict.
- Percentiles: `percentile_cont` in PostgreSQL. Never numpy over a full table.

## 5. Writing style

- Python 3.12, `from __future__ import annotations`, full type hints on every
  public function, docstrings that explain *why* not *what*.
- `ruff` clean. Line length 100. No wildcard imports. No `print` outside scripts.
- Structured logging via `get_logger(__name__)`, never bare `logging.info`.
- Errors: raise specific exceptions in services; let the API layer's
  `APIErrorHandlerMiddleware` render them. Never swallow silently — log with context.
- Every file that is meant to be importable without side effects must be.
