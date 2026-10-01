# RAGOps — API contract

Base URL: `http://localhost:8000` · Prefix: `/api` · OpenAPI: `/docs`

🔑 = requires a valid `X-API-Key`. Everything else answers without one.

> This prose and `docs/api-contract.json` are two views of one surface. The JSON
> is **generated** — `make api-contract` rebuilds it from the running app's own
> OpenAPI document, so it cannot describe a shape the code does not have. Every
> route listed here appears there, and any route added to the backend shows up
> there within one command. Two fields in that file carry the security posture:
> `"write": true` marks a route behind the telemetry write guard (20 of 45), and
> a route **without** it is readable without a credential.
>
> One gap worth knowing about in `/docs` itself: FastAPI renders this guard as
> an ordinary optional `X-API-Key` header, not as a security scheme, so the
> interactive docs have no **Authorize** button and no lock icons. That is a
> limitation of OpenAPI's model for a dependency that is a header rather than a
> `Security()`. `docs/api-contract.json` and the 🔑 marks above are the
> authoritative statement of what is guarded.

## Authentication and tenancy

RAGOps resolves the presented key to a **principal**, and the principal decides
what the request can see. There are two kinds:

| Principal | How you get one | What it can do |
|---|---|---|
| **Platform** | send `X-API-Key: <RAGOPS_API_KEY>` (default `dev-key`), or send no key at all | every route, every company |
| **Tenant** | send `X-API-Key: <a key minted for one company>` | that company's routes and no other company's data |

### The rule that catches people out

> **A presented key is authoritative.** A key that matches nothing is rejected
> with **401** — it does *not* quietly fall back to platform-wide.
>
> **Omitting the header is different.** A request with no `X-API-Key` resolves to
> the platform principal, which is why the dashboard works before anything is
> instrumented.

So `X-API-Key: wrong` is a 401, while no header at all is a 200. That asymmetry is
deliberate: it makes a misconfigured SDK fail loudly instead of quietly uploading
a company's telemetry somewhere visible to everyone.

### Platform, tenant, and the empty-key escape hatch

`RAGOPS_API_KEY=` set to an empty string makes the shared key match nothing, and
**any** presented key then resolves to the platform principal. That is the
one-key local-demo configuration. Set `AUTH_ENABLED=false` for the same effect
and note its cost: it also ignores every tenant key, so a multi-company
deployment running under it has no isolation at all.

### Scopes

A tenant key carries scopes, stored as a comma-separated list, default
`ingest,read`:

| Scope | Grants |
|---|---|
| `ingest` | the 🔑 write routes |
| `read` | the analytics and trace routes |

A key with neither cannot be created — the API rejects an empty scope list with
a **422** rather than issuing a credential that authenticates and then does
nothing. A `read`-only key that tries to write gets a **403**, not a silent drop.

### What is deliberately *not* scoped

These stay platform-wide regardless of the key, each for a stated reason:

| Route | Why |
|---|---|
| `GET /api/health` | must answer when nothing else does; a health check that 401s tells you nothing |
| `GET /api/models`, `PUT /api/models` | a **price registry**. Per-company pricing would make cost incomparable across companies |
| `GET /api/rag/config`, `POST /api/rag/index` | pipeline configuration and the demo knowledge base |
| `/api/collector/*` counters | process-local instrumentation state |
| **All of** `/api/organizations/*` | platform-**only**. A tenant key gets **403** — otherwise a customer could mint themselves a new company and a fresh credential for it |

### Known limitation: application names are global

`applications.name` is unique **across the whole deployment**, not per company.
Two companies cannot both own `customer-support-bot`.

This is intentional. Because the name resolves to exactly one row, a company
whose key sends an application name another already owns gets a loud
**409** naming the collision, rather than its telemetry silently landing in
someone else's application. The cost is real — two customers cannot use the same
conventional name — so name applications for the company, e.g.
`acme-support-bot`. If a genuine need for per-company names appears, relax the
`ix_applications_name` unique index to `(organization_id, name)`; that also
requires threading `organization_id` through the six name→id resolution paths
that end in `scalar_one_or_none()`.

Deleting a company sets its applications' `organization_id` to `NULL` rather
than cascading. Deleting a company must not delete that company's telemetry.

Every analytics endpoint accepts the same query parameters:

| Param | Type | Default | Notes |
|---|---|---|---|
| `window` | `1h\|24h\|7d\|30d\|90d` | `7d` | ignored if `start`/`end` given |
| `start` | ISO-8601 datetime | — | enables custom range |
| `end` | ISO-8601 datetime | — | |
| `application` | application name | `null` | `null` = all applications |

Paginated list endpoints add `page` (1-based, default 1) and `page_size`
(default 25, max 100). All list responses are
`{ items: T[], total: int, page: int, page_size: int, has_next: bool }`.

Error body: `{ "detail": "human readable", "code": "snake_case_code" }`

---

## Health

### `GET /api/health`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "checks": {
    "database": {"ok": true, "latency_ms": 2.1},
    "redis":    {"ok": true, "latency_ms": 0.4},
    "ollama":   {"ok": true, "models": ["qwen2.5:3b"], "latency_ms": 30.2},
    "vector_store": {"ok": true, "backend": "faiss"}
  },
  "timestamp": "2026-09-30T10:00:00Z"
}
```
`status` is `healthy` | `degraded` (a non-essential check failed) | `unhealthy`.
Never fails the request over an optional dependency.

---

## Dashboard & analytics

### `GET /api/dashboard/overview`
`DashboardOverview`:
```json
{
  "window": "7d", "start": "...", "end": "...",
  "application_id": "uuid|null", "application_name": "customer-support|null",
  "total_calls": 0, "total_traces": 0, "error_count": 0, "error_rate": 0.0,
  "total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0,
  "estimated_cost": 0.0, "cost_label": "Local inference — no API cost",
  "avg_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0,
  "avg_tokens_per_request": 0.0,
  "retrieval_score": null, "faithfulness": null, "answer_relevance": null,
  "token_efficiency": null,
  "unique_users": 0, "anomaly_count": 0, "open_recommendation_count": 0,
  "time_series": [TimeSeriesPoint],
  "model_usage": [{"model_name":"qwen2.5:3b","provider":"ollama","call_count":0,"total_tokens":0,"estimated_cost":0.0}],
  "application_usage": [{"application":"customer-support","trace_count":0,"total_tokens":0,"estimated_cost":0.0}]
}
```

### `TimeSeriesPoint` (also used by every analytics endpoint)
```json
{
  "bucket": "2026-09-29T00:00:00Z",
  "request_count": 0, "error_count": 0,
  "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
  "estimated_cost": 0.0,
  "avg_latency_ms": null, "p50_latency_ms": null, "p95_latency_ms": null, "p99_latency_ms": null,
  "avg_retrieval_score": null, "avg_faithfulness": null, "avg_answer_relevance": null
}
```
Empty buckets are `null`, **never `0`**, so charts break the line instead of
diving to zero. One bucket per hour for `1h`/`24h`, per day otherwise.

### `GET /api/analytics/tokens` → `TokenAnalytics`
`total_tokens, input_tokens, output_tokens, avg_tokens_per_request, p50_tokens,
p95_tokens, p99_tokens, max_tokens, cost_per_request, total_cost, cost_label`,
plus `tokens_per_user[]`, `tokens_per_application[]`, `tokens_per_model[]`, `time_series[]`.
Breakdown item: `{ key, label, count, total_tokens, input_tokens, output_tokens, estimated_cost, avg_latency_ms, extra }`

### `GET /api/analytics/cost` → `CostAnalytics`
`total_cost, cost_label, cost_per_request, cost_per_user, cost_per_application, breakdown[], time_series[]`

### `GET /api/analytics/latency` → `LatencyAnalytics`
`avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms, max_latency_ms`,
plus `breakdown_by_stage[]` (per span kind), `breakdown_by_model[]`, `time_series[]`

### `GET /api/analytics/token-efficiency` → `TokenEfficiencyReport`
```json
{
  "window": "7d", "start": "...", "end": "...",
  "score": 0.0,               // 0-100
  "potential_waste_pct": 0.0, // measured, not projected savings
  "wasted_tokens": 0, "total_input_tokens": 0,
  "duplicate_document_count": 0,
  "avg_context_share": 0.0, "avg_output_yield": 0.0,
  "waste_by_application": [...],
  "top_offenders": [{"trace_id":"...","waste_pct":0.0,"score":0.0,"duplicate_document_count":0,"input_tokens":0}],
  "findings": ["Measured: 12% of retrieved documents were duplicates."]
}
```

### `GET /api/analytics/cost-quality` → `CostQualityReport`
```json
{
  "window": "7d", "start": "...", "end": "...",
  "cost_label": "Simulated list price",
  "quality_metric_definition": "Mean of faithfulness and answer relevance from answer_evaluations in this window; null when no evaluation has been run.",
  "models": [{
    "model_name":"qwen2.5:3b","provider":"ollama","call_count":0,"total_tokens":0,
    "avg_input_tokens":0.0,"avg_output_tokens":0.0,"avg_latency_ms":0.0,
    "estimated_cost":0.0,"cost_per_1k_tokens":0.0,"cost_label":"...","is_local":true,
    "retrieval_score":null,"faithfulness":null,"answer_relevance":null,"quality_score":null
  }],
  "best_value": "qwen2.5:3b|null", "best_quality": "qwen2.5:3b|null"
}
```

---

## Traces

### `GET /api/traces`
Filters: `status`, `kind`, `model`, `user_id`, `has_error`, `min_duration_ms`,
`max_duration_ms`, `min_tokens`, `search` (matches `input_text`/`output_text`/`trace_id`),
plus the standard window params and pagination.
→ `Page[TraceSummary]`

`TraceSummary`: `{ id, trace_id, application_id, application_name, user_external_id,
session_id, kind, status, name, start_time, end_time, duration_ms, total_tokens,
estimated_cost, context_tokens, agent_name, agent_iterations, input_preview, has_error }`
(`input_preview` is the first 200 chars of the input — do not ship full prompts in lists.)

### `GET /api/traces/{trace_id}`
Accepts either the internal UUID or the caller-supplied `trace_id` string.
→ `TraceDetail`: summary fields **plus** `input_text`, `output_text`, `error`,
`input_tokens`, `output_tokens`, `tags`, `metadata`, and:
```json
{
  "spans":     [{"id":"...","name":"...","kind":"llm","start_time":"...","end_time":"...","duration_ms":0.0,"status":"success","error":null,"attributes":{}}],
  "retrievals":[
    {"id":"...","query":"...","retriever":"hybrid","top_k":5,"num_results":5,
     "latency_ms":31.2,"context_tokens":1200,"top_score":0.83,
     "configuration":{"...":"..."},"created_at":"...",
     "documents":[
       {"id":"...","document_id":"doc_abc","title":"...","rank":1,"final_score":0.91,
        "bm25_score":12.4,"vector_score":0.78,"rerank_score":4.2,
        "token_count":143,"content_preview":"..."}
     ]}
  ],
  "llm_calls": [{"id":"...","model_name":"qwen2.5:3b","provider":"ollama",
    "input_tokens":4821,"output_tokens":238,"total_tokens":5059,"latency_ms":1420.0,
    "time_to_first_token_ms":null,"estimated_cost":0.0,"status":"success",
    "prompt":"...","completion":"...","temperature":0.1,"max_tokens":null,"created_at":"..."}],
  "retrieval_analysis": {
    "num_documents": 5, "duplicate_document_ids": [], "duplicate_ratio": 0.0,
    "context_tokens": 1200, "token_efficiency": { "...": 0 }
  }
}
```
404 when not found.

### `POST /api/traces` 🔑 → `{ "trace_id": "...", "status": "created" }`
Body `TraceCreate`. Creates or updates a trace in `running` state.

### `POST /api/traces/{trace_id}/retrieval` 🔑 → `{ "retrieval_id": "uuid", "num_documents": 5 }`
Body `RetrievalIn`. Adds a retrieval stage and its ranked documents.

### `POST /api/traces/{trace_id}/generation` 🔑 → `{ "llm_call_id": "uuid", "total_tokens": 5059, "estimated_cost": 0.0 }`
Body `GenerationIn`. Token counts are taken from the request when supplied,
otherwise derived server-side with the deterministic counter.

### `POST /api/traces/{trace_id}/complete` 🔑 → `{ "trace_id": "...", "status": "success", "total_tokens": 5059 }`
Body: partial `TraceCreate` fields. Closes the trace, recomputes denormalised
totals from its `LLMCall` rows, and invalidates the analytics cache.

### `POST /api/traces/batch` 🔑 → `IngestResponse`
Body `{ "traces": [TraceCreate, ...] }` (max 500). → `{ "accepted": 0, "trace_ids": [], "errors": [] }`

---

## Evaluations

### `POST /api/evaluations/retrieval` → `EvaluationRunDetail`
Body `RetrievalEvaluationRequest`:
```json
{
  "name": "hybrid-baseline",
  "application": "customer-support",
  "dataset_name": "support_eval_v1",
  "k": 5,
  "ks": [1, 3, 10],
  "load_from_path": true,
  "examples": null,
  "config_override": {"retriever": "hybrid", "top_k": 5, "rerank": true},
  "persist": true
}
```
Runs the labelled dataset through the retriever, computes metrics, persists the
run. Response:
```json
{
  "id": "uuid", "name": "...", "dataset_name": "...", "evaluation_type": "retrieval",
  "status": "completed", "k": 5, "num_queries": 0,
  "metrics": {
    "k": 5,
    "precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0,
    "mrr": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0,
    "zero_result_rate": 0.0, "num_queries": 0, "avg_documents_retrieved": 0.0,
    "by_k": { "1": {"recall_at_k": 0.0, "...": 0.0}, "3": {}, "5": {}, "10": {} }
  },
  "config": { "...": "recorded config actually used" },
  "duration_ms": 0.0,
  "results": [{
    "query": "...", "k": 5, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    "reciprocal_rank": 0.0, "ndcg": 0.0, "hit": true,
    "retrieved_document_ids": [], "relevant_document_ids": [], "missed_document_ids": []
  }]
}
```

### `POST /api/evaluations/answer` → `AnswerEvaluationSummary`
Body `AnswerEvaluationRequest`:
```json
{
  "name": "faithfulness-baseline",
  "application": "customer-support",
  "items": [{"question":"...","answer":"...","context":["..."],"reference_answer":null,"trace_id":null}],
  "use_llm_judge": false,
  "judge_model": null,
  "persist": true
}
```
→ `{ num_items, method: "deterministic_embedding", faithfulness, context_relevance,
answer_relevance, citation_coverage, unsupported_claim_ratio, judge_faithfulness: null,
judge_answer_relevance: null, judge_model: null, per_item: [...] }`
Judge fields stay `null` unless `use_llm_judge` is true **and** the judge succeeds.

### `GET /api/evaluations/runs` → `Page[EvaluationRunSummary]`
Filters: `evaluation_type`, `application`, `status`, `dataset_name`.

### `GET /api/evaluations/runs/{id}` → `EvaluationRunDetail`

### `POST /api/evaluations/compare` → `RegressionReport`
```json
{ "baseline_run_id": "uuid", "candidate_run_id": "uuid", "max_regression_pct": 5.0 }
```
→ `{ baseline_run_id, candidate_run_id, baseline_name, candidate_name,
deltas: [{metric, baseline, candidate, absolute_change, relative_change_pct, is_regression, direction}],
config_differences: [{key, baseline_value, candidate_value}],
has_regression, regression_summary,
contributing_config_changes: ["Recorded setting `rerank.enabled` changed from `true` to `false`"] }`

### `GET /api/evaluations/runs/{id}/results` → `Page[PerQueryResult]`

---

## Anomalies

### `GET /api/anomalies` → `Page[AnomalyOut]`
Filters: `application`, `anomaly_type`, `severity`, `window`, `is_resolved`.
`AnomalyOut`: `{ id, application_id, application_name, trace_id, anomaly_type, severity,
metric, observed_value, expected_low, expected_high, anomaly_score, peer_median, peer_p95,
evidence: {...}, detected_at, detection_method, is_resolved }`
`evidence` holds **measured facts only** (expected band, observed value, peer stats,
and trace signals such as `retrieved_documents`, `agent_iterations`, `context_tokens`).
It must never contain a claimed cause.

### `POST /api/anomalies/detect` → `AnomalyDetectResponse`
Body: `{ application, window, lookback_hours, contamination, features[], persist }`
Fits `IsolationForest` per application on the recorded feature matrix.
→ `{ application_id, application_name, num_samples, num_features, contamination,
detection_method: "isolation_forest", num_anomalies, anomaly_rate,
feature_importance_proxy: {"total_tokens": 0.4, "...": 0.0}, anomalies: [AnomalyOut], duration_ms }`

### `PATCH /api/anomalies/{id}` → `AnomalyOut`  · body `{"is_resolved": true}`

---

## Recommendations

### `GET /api/recommendations` → `Page[RecommendationOut]`
Filters: `application`, `category`, `priority`, `status`.
`RecommendationOut`: `{ id, application_id, application_name, category, title,
rationale, recommendation, priority, severity_score, evidence: {...}, metrics: {...},
status, confidence, created_at }`
`evidence` must contain the measured metric that triggered it.

### `POST /api/recommendations/generate` → `{ generated: 0, recommendations: [RecommendationOut] }`
Body: `{ application, window, persist, min_severity }`

---

## Experiments

### `GET /api/experiments` → `Page[ExperimentOut]`
### `GET /api/experiments/{id}` → `ExperimentOut`
`ExperimentOut.variants[]`: `{ id, name, config, run_id, metrics, deltas_vs_baseline: [{metric, baseline, variant, change, direction}] }`

### `POST /api/experiments` → `ExperimentOut`
```json
{
  "name": "Chunking optimization",
  "application": "customer-support",
  "hypothesis": "Larger chunks raise recall but cost latency",
  "dataset_name": "support_eval_v1",
  "variants": [{"name":"A","config":{"chunk_size":500,"top_k":5}}, {"name":"B","config":{"chunk_size":800,"top_k":5}}],
  "baseline_variant": "A"
}
```

---

## Applications & models

### `GET /api/applications` → `Page[ApplicationOut]`
`ApplicationOut`: `{ id, name, description, environment, is_active, created_at, trace_count, last_seen_at, total_tokens }`

### `GET /api/applications/{id}/stats` → `{ application, window, start, end, trace_count, total_tokens, estimated_cost, avg_latency_ms, error_rate, model_usage[], time_series[] }`

### `GET /api/models` → `ModelOut[]`
`{ id, name, provider, context_window, is_local, input_cost_per_1k, output_cost_per_1k, description }`

### `PUT /api/models` → `ModelOut`  · body `ModelUpsert` (upsert by provider+name)

---

## Organizations & API keys  · 🔑 platform only

Onboarding surface. **Every route here requires the platform principal** — a
tenant key is refused with **403**, because a company that could mint itself a
sibling company would not be isolated. This is also the only place in the API
where a usable credential is ever returned.

### `GET /api/organizations` → `OrganizationOut[]`
Active companies first, then by name. A plain list, not a `Page` — the set is
bounded by how many companies an operator has onboarded. `is_active=false` rows
are included rather than hidden: the useful question is which companies can
receive telemetry, not which exist.

`OrganizationOut`: `{ id, name, slug, is_active, num_applications, num_api_keys, created_at, updated_at }`

### `POST /api/organizations` → `201 OrganizationOut`  🔑
body `{ name, slug?, is_active? }`. `slug` is derived from `name` when omitted.

A repeated `name` is **409**, not a second indistinguishable row. **No key is
minted here** — creating a company and creating a credential are separate acts.

### `GET /api/organizations/{id}/api-keys` → `ApiKeyOut[]`  🔑
Newest first, revoked keys last but **still listed**. `last_used_at` on a revoked
key is the evidence that answers "was the leaked key used?"; hiding the row
would remove it.

`ApiKeyOut`: `{ id, organization_id, name, key_prefix, scopes, is_active, last_used_at, revoked_at, expires_at, created_at, updated_at }`

🔑 There is **no `key` field on this response, and there is no way to recover the
plaintext afterwards** — only its HMAC-SHA256 digest is stored. `key_prefix` is
the first 12 characters, kept so a list is recognisable; it is display material,
not a credential.

### `POST /api/organizations/{id}/api-keys` → `201 CreatedApiKey`  🔑
body `{ name, scopes?, expires_at? }`. `scopes` defaults to `["ingest", "read"]`;
omitting the field means "use the default", sending `[]` is a **422**.

Response is `ApiKeyOut` **plus** `key`: a 47-character `rag_<43>` string matching
`^rag_[A-Za-z0-9_-]{43}$`. **This is the only response in the entire API that
contains a usable credential, and the only one that ever will.** Copy it out of
the response body — there is no second chance.

`last_used_at` is stamped on first use rather than at creation, so a key that was
minted and never sent is identifiable as unused.

### `DELETE /api/organizations/{id}/api-keys/{key_id}` → `200 MessageResponse`  🔑

Revoking **sets `revoked_at`; it does not delete the row.** The row, its
`created_at` and its `last_used_at` stay as the audit trail, and the key stays
visible in the list. A real delete would return 204 and destroy the evidence an
operator needs most after a leak. The mismatch with the HTTP verb is deliberate.

Idempotent: revoking an already-revoked key succeeds and **keeps the original
`revoked_at`**, so the timestamp stays a fact about the first revocation.

A key id belonging to another organization is **404, not 403** — a 403 confirms
the id exists, which is itself the enumeration channel. Every cross-tenant lookup
in this API behaves this way.

---

## RAG demo endpoint

### `POST /api/rag/query` → `{ trace_id, answer, documents[], usage, timing, configuration }`
Runs the full pipeline (preprocess → BM25 + vector → RRF hybrid → cross-encoder
rerank → top-k → Ollama) and records a real trace.
```json
{
  "trace_id": "uuid-str",
  "answer": "...",
  "documents": [{"document_id":"...","title":"...","rank":1,"final_score":0.9,
                 "bm25_score":1.0,"vector_score":0.8,"rerank_score":3.1,
                 "token_count":120,"content_preview":"..."}],
  "usage": {"input_tokens":0,"output_tokens":0,"total_tokens":0,"context_tokens":0},
  "timing": {"total_ms":0.0,"embedding_ms":0.0,"bm25_ms":0.0,"vector_ms":0.0,
             "rerank_ms":0.0,"llm_ms":0.0},
  "configuration": {"retriever":"hybrid","top_k":5,"rerank":true,"chunk_size":500,
                    "embedding_model":"...","reranker_model":"...","llm_model":"qwen2.5:3b"}
}
```

### `GET /api/rag/config` → current pipeline configuration
### `POST /api/rag/index` → `{ "num_documents": 0, "num_chunks": 0, "backend": "faiss", "dimension": 384, "duration_ms": 0.0 }`
(Re)builds the retrieval index from `evaluation/datasets/knowledge_base/`.
