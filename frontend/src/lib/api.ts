/**
 * The one place that knows how to talk to the RAGOps backend.
 *
 * Everything HTTP lives here: the base URL, the `X-API-Key` header, the typed
 * wrapper that turns a non-2xx into an `ApiError` carrying the real status and
 * message, and one function per endpoint. No other file in `src/` calls
 * `fetch`.
 *
 * The interfaces below mirror the Pydantic schemas under
 * `backend/app/schemas/` and the two local models the routers declare
 * (`RecommendationGenerateResponse` in `api/v1/optimization.py` and
 * `ApplicationStats` in `api/v1/applications.py`). Field names are transcribed
 * from those files; nothing is renamed and nothing is invented. Two backend
 * conventions are load-bearing and are reproduced rather than smoothed over:
 *
 * 1. **Null is not zero.** Empty time-series buckets come back as `null`, not
 *    `0`, so a chart breaks its line instead of diving to the axis. Latency
 *    and quality percentiles are `null` when they were never measured. Every
 *    nullable field below is `| null` for that reason — rendering one as `0`
 *    would be a fabricated measurement.
 * 2. **`error_rate` is a fraction in `[0, 1]`**, not a percentage. The
 *    formatter that turns it into a percentage lives in `src/lib/format.ts`,
 *    not here.
 */

const DEFAULT_BASE_URL = "http://localhost:8000";

/**
 * `VITE_API_URL` when it is set *and non-empty*, otherwise the local default.
 *
 * An empty string is treated as unset on purpose. `??` only catches `undefined`
 * and `null`, so `VITE_API_URL=` on its own line — the shape an unset shell
 * variable takes when it is written into `.env` — would survive as `""`, and
 * `"".replace(...)` is `""`. Every request would then resolve against the page's
 * own origin and the console would report the backend as unreachable.
 */
const CONFIGURED_BASE_URL =
  typeof import.meta.env.VITE_API_URL === "string" && import.meta.env.VITE_API_URL.trim() !== ""
    ? import.meta.env.VITE_API_URL.trim()
    : DEFAULT_BASE_URL;

/** Base URL, trailing slashes stripped, never empty. */
export const API_BASE_URL: string = CONFIGURED_BASE_URL.replace(/\/+$/, "");

const API_KEY_STORAGE_KEY = "ragops.apiKey";

/** Path prefix every route sits under (`settings.api_v1_prefix`). */
export const API_PREFIX = "/api";

/* -------------------------------------------------------------------------- */
/* Errors                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * A non-2xx response, with the status and the backend's own message.
 *
 * `detail` is the FastAPI `ErrorResponse.detail` string when there is one. The
 * UI renders this verbatim, so it must never be replaced with a generic
 * "something went wrong".
 */
export class ApiError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly code: string | null;
  readonly url: string;

  constructor(params: {
    status: number;
    statusText: string;
    detail: string;
    code: string | null;
    url: string;
  }) {
    super(params.detail);
    this.name = "ApiError";
    this.status = params.status;
    this.statusText = params.statusText;
    this.code = params.code;
    this.url = params.url;
  }

  /** True when the backend rejected the request for want of an API key. */
  get isUnauthorized(): boolean {
    return this.status === 401 || this.status === 403;
  }

  /** True when the backend could not be reached at all. */
  get isNetworkError(): boolean {
    return this.status === 0;
  }
}

/* -------------------------------------------------------------------------- */
/* API key                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * The key is read from `localStorage` on every request and sent **only** as
 * the `X-API-Key` header. It is never placed in a query string, never logged,
 * and never sent anywhere but `API_BASE_URL`.
 */
export function getApiKey(): string | null {
  try {
    const stored = window.localStorage.getItem(API_KEY_STORAGE_KEY);
    return stored === null || stored === "" ? null : stored;
  } catch {
    // Private-mode Safari and a locked-down profile both throw here. A console
    // with no place to keep the key still works for every read-only endpoint.
    return null;
  }
}

export function setApiKey(value: string): void {
  try {
    const trimmed = value.trim();
    if (trimmed === "") {
      window.localStorage.removeItem(API_KEY_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
  } catch {
    // Same rationale as above: no storage is not a reason to crash the app.
  }
}

export function clearApiKey(): void {
  try {
    window.localStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    /* nothing to do */
  }
}

/* -------------------------------------------------------------------------- */
/* Per-organization keys                                                       */
/* -------------------------------------------------------------------------- */

const API_KEYS_STORAGE_KEY = "ragops.apiKeys";

/**
 * The key for each organization whose key this browser holds, keyed by org id.
 *
 * `ragops.apiKey` above stays *the active key* and is what `request()` sends —
 * untouched, deliberately, because 118 tests and every existing page depend on
 * it. This map is additive: writing an entry here does not switch the active
 * key, so storing a second company's credential cannot silently repoint the
 * console at a different tenant's data. Switching is an explicit act
 * (`switchApiKey`) because "which company's data am I looking at" should never
 * be a side effect of pasting a key somewhere.
 *
 * A JSON object rather than one localStorage entry per organization, so a
 * corrupt value can only ever cost the whole map instead of poisoning one
 * organization's key on read.
 *
 * `Partial` is the honest type: this map is built key by key from parsed JSON
 * (see `getApiKeys`), so looking up an organization nobody stored a key for
 * really does yield `undefined`. Written as `Record<string, string>` the
 * compiler would call that check unreachable and delete it, leaving a caller
 * that renders `undefined` as though it were a credential.
 */
type ApiKeyMap = Partial<Record<string, string>>;

/**
 * Every stored organization key, or `{}` when there are none.
 *
 * Never throws. Storage that has been cleared, denied, or written by an older
 * build holding a different shape all degrade to "no keys stored" rather than
 * taking the Settings page down — losing the list is recoverable, a crash on
 * render is not.
 */
export function getApiKeys(): ApiKeyMap {
  try {
    const raw = window.localStorage.getItem(API_KEYS_STORAGE_KEY);
    if (raw === null || raw.trim() === "") return {};
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    // Rebuilt key by key: a value that is not a string is dropped rather than
    // cast, so a hand-edited entry cannot end up in an `X-API-Key` header.
    const map: ApiKeyMap = {};
    for (const [orgId, key] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof key === "string" && key.trim() !== "") map[orgId] = key.trim();
    }
    return map;
  } catch {
    return {};
  }
}

/** Store (or replace) one organization's key. A blank value removes the entry. */
export function setApiKeyForOrganization(organizationId: string, value: string): void {
  try {
    const trimmed = value.trim();
    const map = getApiKeys();
    if (trimmed === "") {
      delete map[organizationId];
    } else {
      map[organizationId] = trimmed;
    }
    window.localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(map));
  } catch {
    /* Same rationale as getApiKey: no storage is not a reason to crash. */
  }
}

/** Forget one organization's key, leaving the others in place. */
export function clearApiKeyForOrganization(organizationId: string): void {
  setApiKeyForOrganization(organizationId, "");
}

/**
 * Make an organization's key the active one, so subsequent requests carry it.
 *
 * Writing a key here stores it under its own organization id rather than in the
 * legacy slot: switching to a company is what you do when you are working for
 * that company, and the map is the record of that. Passing the id of the
 * organization already active would otherwise lose the association.
 */
export function switchApiKey(organizationId: string): boolean {
  // `=== undefined` rather than `in`, because `ApiKeyMap` is a `Partial`: the
  // lookup already types as `string | undefined`, so the check is what the value
  // can be rather than a workaround for a type that claimed it could not be.
  // `setApiKey` below also drops a blank value, so a hand-edited empty entry
  // cannot become an `X-API-Key` header that clears the stored one.
  const stored = getApiKeys()[organizationId];
  if (stored === undefined) return false;
  setApiKey(stored);
  return true;
}

/* -------------------------------------------------------------------------- */
/* Query parameters                                                           */
/* -------------------------------------------------------------------------- */

/** The windows `backend/app/api/deps.py` accepts. Anything else is a 422. */
export const TIME_WINDOWS = ["1h", "24h", "7d", "30d", "90d"] as const;
export type TimeWindow = (typeof TIME_WINDOWS)[number];

export const DEFAULT_TIME_WINDOW: TimeWindow = "7d";

/** `true` on every analytics route; `undefined` omits the parameter entirely. */
export function isTimeWindow(value: string): value is TimeWindow {
  return (TIME_WINDOWS as readonly string[]).includes(value);
}

/**
 * Query parameters shared by every analytics endpoint. `application` is
 * omitted rather than sent empty when it is `null` — `application=` would be a
 * name that does not exist, not "all applications".
 */
export interface ScopeParams {
  window?: TimeWindow;
  start?: string | null;
  end?: string | null;
  application?: string | null;
}

/** 1-based page and its size, matching `deps.Pagination`. */
export interface PageParams {
  page?: number;
  page_size?: number;
}

/** A value the request sends only when it is neither `undefined` nor `null`. */
type QueryValue = string | number | boolean | null | undefined;

/**
 * Appends params to `path`, dropping `null` and `undefined` so an unset filter
 * is absent from the URL rather than sent as an empty string.
 */
export function buildUrl(path: string, params?: Record<string, QueryValue>): string {
  const suffix = path.startsWith(API_PREFIX) ? path : `${API_PREFIX}${path}`;
  const url = `${API_BASE_URL}${suffix}`;
  if (!params) return url;

  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const query = search.toString();
  return query === "" ? url : `${url}?${query}`;
}

/* -------------------------------------------------------------------------- */
/* The wrapper                                                                */
/* -------------------------------------------------------------------------- */

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  params?: Record<string, QueryValue>;
  /** Set false for the few endpoints that take no auth header. */
  authenticated?: boolean;
}

/** Pulls `detail` (string) or `code` out of the backend's `ErrorResponse`. */
function readErrorBody(payload: unknown): { detail: string; code: string | null } {
  if (typeof payload === "string" && payload.trim() !== "") {
    return { detail: payload, code: null };
  }
  if (payload !== null && typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    const detail = record.detail;
    if (typeof detail === "string" && detail !== "") {
      const code = record.code;
      return { detail, code: typeof code === "string" ? code : null };
    }
    // FastAPI's 422 puts a list of per-field problems under `detail`.
    if (Array.isArray(detail)) {
      const parts = detail
        .map((entry) => {
          if (entry === null || typeof entry !== "object") return null;
          const field = entry as Record<string, unknown>;
          const loc = Array.isArray(field.loc) ? field.loc.slice(-1).join(".") : "";
          const msg = typeof field.msg === "string" ? field.msg : "";
          return loc === "" ? msg : `${loc}: ${msg}`;
        })
        .filter((part): part is string => part !== null && part !== "");
      if (parts.length > 0) return { detail: parts.join("; "), code: null };
    }
  }
  return { detail: "", code: null };
}

/**
 * The single fetch call site. Adds `X-API-Key` when a key is stored, parses
 * JSON, and raises `ApiError` for anything that is not a 2xx — including a
 * transport failure, which is reported as status `0` rather than surfacing as
 * an unhandled `TypeError`.
 */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, params, authenticated = true } = options;
  const url = buildUrl(path, params);

  const headers: Record<string, string> = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (authenticated) {
    const key = getApiKey();
    if (key !== null) headers["X-API-Key"] = key;
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers,
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch (cause) {
    throw new ApiError({
      status: 0,
      statusText: "Network request failed",
      detail:
        cause instanceof Error && cause.message !== ""
          ? `Could not reach the RAGOps backend at ${API_BASE_URL}: ${cause.message}`
          : `Could not reach the RAGOps backend at ${API_BASE_URL}.`,
      code: null,
      url,
    });
  }

  const raw = await response.text();
  let payload: unknown = null;
  if (raw !== "") {
    try {
      payload = JSON.parse(raw);
    } catch {
      // A proxy or a crashed server can answer 200 with HTML. Treating that as
      // a successful empty body would render a blank page.
      if (response.ok) {
        throw new ApiError({
          status: response.status,
          statusText: response.statusText,
          detail: "The backend returned a response that was not JSON.",
          code: null,
          url,
        });
      }
    }
  }

  if (!response.ok) {
    const { detail, code } = readErrorBody(payload);
    throw new ApiError({
      status: response.status,
      statusText: response.statusText,
      detail:
        detail !== ""
          ? detail
          : `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""} from ${url}`,
      code,
      url,
    });
  }

  return payload as T;
}

const get = <T,>(path: string, params?: Record<string, QueryValue>): Promise<T> =>
  request<T>(path, params ? { params } : {});
const post = <T,>(path: string, body?: unknown, params?: Record<string, QueryValue>): Promise<T> =>
  request<T>(path, { method: "POST", body: body ?? {}, ...(params ? { params } : {}) });
const patch = <T,>(path: string, body?: unknown): Promise<T> =>
  request<T>(path, { method: "PATCH", body: body ?? {} });
/**
 * DELETE takes no body. It is a separate helper rather than a flag on `request`
 * because `post` and `patch` both default their body to `{}` -- sending `{}` to
 * a DELETE would be a body on a method that has no use for one, and the
 * backend's `revoke_api_key` takes its state entirely from the path.
 */
const del = <T,>(path: string): Promise<T> => request<T>(path, { method: "DELETE" });

/* -------------------------------------------------------------------------- */
/* Schemas — common.py                                                        */
/* -------------------------------------------------------------------------- */

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface TimeSeriesPoint {
  bucket: string;
  request_count?: number | null;
  error_count?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  estimated_cost?: number | null;
  avg_latency_ms?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  avg_retrieval_score?: number | null;
  avg_faithfulness?: number | null;
  avg_answer_relevance?: number | null;
}

export interface BreakdownItem {
  key: string;
  label?: string | null;
  count: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost: number;
  avg_latency_ms?: number | null;
  /** Per-grouping bag. Grouped-by-model carries `{provider, error_count, p95_latency_ms}`. */
  extra: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  /**
   * One entry per dependency, keyed by name (`database`, `redis`, `ollama`,
   * `vector_store`).
   *
   * Typed as a record of `unknown` rather than a fixed shape because the set of
   * names depends on what is configured: the server adds `ollama` and
   * `vector_store` only when those settings exist, and callers must narrow a
   * check's own shape before reading its `status`. A failed check is a value
   * with `status: "down"` and an `error`, never `null`.
   */
  checks: Record<string, unknown>;
  timestamp: string;
}

export interface MessageResponse {
  message: string;
  detail?: string | null;
}

/* -------------------------------------------------------------------------- */
/* Schemas — analytics.py                                                     */
/* -------------------------------------------------------------------------- */

export interface ModelUsageRow {
  model_name: string;
  provider: string;
  call_count: number;
  total_tokens: number;
  estimated_cost: number;
  cost_label: string;
}

export interface ApplicationUsageRow {
  application: string;
  trace_count: number;
  total_tokens: number;
  estimated_cost: number;
  cost_label: string;
}

export interface DashboardOverview {
  window: string;
  start: string;
  end: string;
  application_id?: string | null;
  application_name?: string | null;
  total_calls: number;
  total_traces: number;
  error_count: number;
  /** Fraction in [0, 1]. Not a percentage. */
  error_rate: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  estimated_cost: number;
  cost_label: string;
  avg_latency_ms?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  avg_tokens_per_request: number;
  retrieval_score?: number | null;
  faithfulness?: number | null;
  answer_relevance?: number | null;
  token_efficiency?: number | null;
  unique_users: number;
  anomaly_count: number;
  open_recommendation_count: number;
  time_series: TimeSeriesPoint[];
  model_usage: ModelUsageRow[];
  application_usage: ApplicationUsageRow[];
}

export interface TokenAnalytics {
  window: string;
  start: string;
  end: string;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  avg_tokens_per_request: number;
  p50_tokens?: number | null;
  p95_tokens?: number | null;
  p99_tokens?: number | null;
  max_tokens: number;
  cost_per_request: number;
  total_cost: number;
  cost_label: string;
  tokens_per_user: BreakdownItem[];
  tokens_per_application: BreakdownItem[];
  tokens_per_model: BreakdownItem[];
  time_series: TimeSeriesPoint[];
}

export interface CostAnalytics {
  window: string;
  start: string;
  end: string;
  total_cost: number;
  cost_label: string;
  cost_per_request: number;
  cost_per_user: number;
  cost_per_application: number;
  breakdown: BreakdownItem[];
  time_series: TimeSeriesPoint[];
}

export interface LatencyAnalytics {
  window: string;
  start: string;
  end: string;
  avg_latency_ms?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  max_latency_ms?: number | null;
  breakdown_by_stage: BreakdownItem[];
  breakdown_by_model: BreakdownItem[];
  time_series: TimeSeriesPoint[];
}

export interface WasteByApplication {
  application: string;
  avg_waste_pct: number;
  trace_count: number;
  total_tokens: number;
}

export interface TopOffender {
  trace_id: string;
  waste_pct: number;
  score: number;
  duplicate_document_count: number;
  input_tokens: number;
}

export interface TokenEfficiencyReport {
  window: string;
  start: string;
  end: string;
  score: number;
  /** Already 0–100: `100 * wasted / total_input`. */
  potential_waste_pct: number;
  wasted_tokens: number;
  total_input_tokens: number;
  duplicate_document_count: number;
  avg_context_share: number;
  avg_output_yield: number;
  waste_by_application: WasteByApplication[];
  top_offenders: TopOffender[];
  findings: string[];
}

export interface ModelComparison {
  model_name: string;
  provider: string;
  call_count: number;
  total_tokens: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  avg_latency_ms: number;
  estimated_cost: number;
  cost_per_1k_tokens: number;
  cost_label: string;
  is_local: boolean;
  retrieval_score?: number | null;
  faithfulness?: number | null;
  answer_relevance?: number | null;
  quality_score?: number | null;
}

export interface CostQualityReport {
  window: string;
  start: string;
  end: string;
  models: ModelComparison[];
  cost_label: string;
  quality_metric_definition: string;
  best_value?: number | null;
  best_quality?: number | null;
}

/* -------------------------------------------------------------------------- */
/* Schemas — trace.py                                                         */
/* -------------------------------------------------------------------------- */

export interface RetrievedDocument {
  id: string;
  document_id: string;
  title?: string | null;
  rank: number;
  final_score: number;
  bm25_score?: number | null;
  vector_score?: number | null;
  rerank_score?: number | null;
  token_count: number;
  content_preview?: string | null;
}

export interface Retrieval {
  id: string;
  query: string;
  retriever: string;
  top_k: number;
  num_results: number;
  latency_ms?: number | null;
  context_tokens: number;
  top_score?: number | null;
  configuration?: Record<string, unknown> | null;
  created_at: string;
  documents: RetrievedDocument[];
}

export interface Span {
  id: string;
  name: string;
  kind: string;
  start_time: string;
  end_time?: string | null;
  duration_ms?: number | null;
  status: string;
  error?: string | null;
  attributes?: Record<string, unknown> | null;
}

export interface LLMCall {
  id: string;
  model_name: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  latency_ms?: number | null;
  time_to_first_token_ms?: number | null;
  estimated_cost: number;
  status: string;
  prompt?: string | null;
  completion?: string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  created_at: string;
}

/** Derived at ingest and stored under `Trace.metadata.retrieval_analysis`. */
export interface RetrievalAnalysis {
  num_documents?: number;
  duplicate_document_ids?: string[];
  duplicate_ratio?: number;
  context_tokens?: number;
  token_efficiency?: number;
  [key: string]: unknown;
}

export interface TraceSummary {
  id: string;
  trace_id: string;
  application_id: string;
  application_name?: string | null;
  user_external_id?: string | null;
  session_id?: string | null;
  kind: string;
  status: string;
  name?: string | null;
  start_time: string;
  end_time?: string | null;
  duration_ms?: number | null;
  total_tokens: number;
  estimated_cost: number;
  context_tokens: number;
  agent_name?: string | null;
  agent_iterations?: number | null;
  input_preview?: string | null;
  has_error: boolean;
}

export interface TraceDetail {
  id: string;
  trace_id: string;
  application_id: string;
  application_name?: string | null;
  user_id?: string | null;
  session_id?: string | null;
  kind: string;
  status: string;
  name?: string | null;
  start_time: string;
  end_time?: string | null;
  duration_ms?: number | null;
  total_tokens: number;
  estimated_cost: number;
  context_tokens: number;
  agent_name?: string | null;
  agent_iterations?: number | null;
  input_text?: string | null;
  output_text?: string | null;
  error?: string | null;
  input_tokens: number;
  output_tokens: number;
  tags?: string[];
  metadata?: Record<string, unknown> | null;
  created_at: string;
  spans: Span[];
  retrievals: Retrieval[];
  llm_calls: LLMCall[];
  retrieval_analysis: RetrievalAnalysis;
}

export interface TraceFilters extends ScopeParams, PageParams {
  status?: string | null;
  kind?: string | null;
  model?: string | null;
  user_id?: string | null;
  has_error?: boolean | null;
  min_duration_ms?: number | null;
  max_duration_ms?: number | null;
  min_tokens?: number | null;
  search?: string | null;
}

/* -------------------------------------------------------------------------- */
/* Schemas — evaluation.py                                                    */
/* -------------------------------------------------------------------------- */

export interface RetrievalMetrics {
  k: number;
  precision_at_k: number;
  recall_at_k: number;
  f1_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  hit_rate_at_k: number;
  zero_result_rate: number;
  num_queries: number;
  avg_documents_retrieved: number;
}

export interface PerQueryResult {
  query: string;
  k: number;
  precision: number;
  recall: number;
  f1: number;
  reciprocal_rank: number;
  ndcg: number;
  hit: boolean;
  retrieved_document_ids: string[];
  relevant_document_ids: string[];
  missed_document_ids: string[];
  latency_ms?: number | null;
}

export interface EvalExample {
  query: string;
  relevant_documents: string[];
  relevance_grades?: Record<string, number> | null;
  reference_answer?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface RetrievalEvaluationRequest {
  name?: string | null;
  application?: string | null;
  dataset_name: string;
  k: number;
  ks?: number[];
  examples?: EvalExample[] | null;
  load_from_path: boolean;
  config_override: Record<string, unknown>;
  persist: boolean;
}

export interface EvaluationRunSummary {
  id: string | null;
  name: string;
  application_id: string | null;
  dataset_name: string;
  evaluation_type: string;
  status: string;
  k: number;
  num_queries: number;
  /**
   * For a retrieval run this is keyed by `k`, e.g. `{"k=5": RetrievalMetrics}`;
   * for an answer run it is the answer-metric blob. `readRetrievalMetrics`
   * handles the first shape without the caller guessing at it.
   */
  metrics?: Record<string, unknown> | null;
  config?: Record<string, unknown> | null;
  started_at?: string | null;
  completed_at?: string | null;
  duration_ms?: number | null;
  notes?: string | null;
  error?: string | null;
  created_at: string;
}

export interface EvaluationRunDetail extends EvaluationRunSummary {
  results: PerQueryResult[];
}

export interface AnswerEvaluationItem {
  question: string;
  answer: string;
  context: string[];
  reference_answer?: string | null;
  trace_id?: string | null;
  relevant_documents?: string[];
}

export interface AnswerEvaluationRequest {
  name?: string | null;
  application?: string | null;
  items: AnswerEvaluationItem[];
  use_llm_judge: boolean;
  judge_model?: string | null;
  persist: boolean;
}

export interface ClaimEvaluation {
  claim: string;
  supported: boolean;
  best_matching_context_index: number | null;
  similarity: number;
  method: string;
}

export interface AnswerEvaluationResult {
  question: string;
  faithfulness: number;
  context_relevance: number;
  answer_relevance: number;
  citation_coverage: number;
  unsupported_claim_ratio: number;
  judge_faithfulness?: number | null;
  judge_answer_relevance?: number | null;
  judge_model?: string | null;
  method: string;
  claims: ClaimEvaluation[];
  duration_ms?: number | null;
}

export interface AnswerEvaluationSummary {
  num_items: number;
  method: string;
  faithfulness: number;
  context_relevance: number;
  answer_relevance: number;
  citation_coverage: number;
  unsupported_claim_ratio: number;
  judge_faithfulness?: number | null;
  judge_answer_relevance?: number | null;
  judge_model?: string | null;
  per_item: AnswerEvaluationResult[];
}

export interface MetricDelta {
  metric: string;
  baseline: number;
  candidate: number;
  absolute_change: number;
  relative_change_pct: number;
  is_regression: boolean;
  direction: string;
}

export interface ConfigDifference {
  key: string;
  baseline_value: unknown;
  candidate_value: unknown;
}

export interface RegressionReport {
  baseline_run_id: string;
  candidate_run_id: string;
  baseline_name: string;
  candidate_name: string;
  deltas: MetricDelta[];
  config_differences: ConfigDifference[];
  has_regression: boolean;
  regression_summary?: string | null;
  contributing_config_changes: string[];
}

export interface CompareRequest {
  baseline_run_id: string;
  candidate_run_id: string;
  max_regression_pct: number;
}

/* -------------------------------------------------------------------------- */
/* Schemas — insights.py                                                      */
/* -------------------------------------------------------------------------- */

export interface Anomaly {
  id: string;
  application_id: string | null;
  application_name?: string | null;
  trace_id: string | null;
  anomaly_type: string;
  severity: string;
  metric: string;
  observed_value?: number | null;
  expected_low?: number | null;
  expected_high?: number | null;
  anomaly_score?: number | null;
  peer_median?: number | null;
  peer_p95?: number | null;
  evidence?: Record<string, unknown> | null;
  detected_at: string;
  detection_method: string;
  is_resolved: boolean;
}

export interface AnomalyDetectRequest {
  application?: string | null;
  window: string;
  lookback_hours: number;
  contamination?: number | null;
  features: string[];
  persist: boolean;
}

export interface AnomalyDetectResponse {
  application_id: string | null;
  application_name?: string | null;
  num_samples: number;
  num_features: number;
  contamination: number;
  detection_method: string;
  num_anomalies: number;
  anomaly_rate: number;
  /** Per-feature deviation magnitude — not a fitted importance score. */
  feature_importance_proxy: Record<string, number>;
  anomalies: Anomaly[];
  duration_ms: number;
}

export interface Recommendation {
  id: string | null;
  application_id: string | null;
  application_name?: string | null;
  category: string;
  title: string;
  rationale: string;
  recommendation: string;
  priority: string;
  severity_score: number;
  evidence?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  status: string;
  confidence: number;
  created_at?: string | null;
}

export interface RecommendationFilters extends ScopeParams, PageParams {
  category?: string | null;
  priority?: string | null;
  status?: string | null;
}

export interface RecommendationGenerateRequest {
  application?: string | null;
  window: string;
  persist: boolean;
  min_severity: number;
}

export interface RecommendationGenerateResponse {
  generated: number;
  recommendations: Recommendation[];
  notes?: string | null;
  /** Every rule that ran and what it measured, including the ones that stayed quiet. */
  rules_evaluated: Record<string, unknown>[];
}

export interface ExperimentVariant {
  id: string;
  name: string;
  /**
   * Nullable but always present — `ExperimentVariantOut.run_id` is
   * `uuid.UUID | None` with no default, so Pydantic serialises the key on every
   * variant. `null` means the variant was never evaluated, which is a different
   * statement from "no run", and the page renders the two differently.
   */
  run_id: string | null;
  config?: Record<string, unknown> | null;
  metrics?: Record<string, unknown> | null;
  deltas_vs_baseline: Array<Record<string, unknown>>;
}

export interface Experiment {
  id: string;
  name: string;
  application_id?: string | null;
  description?: string | null;
  hypothesis?: string | null;
  status: string;
  dataset_name?: string | null;
  baseline_run_id?: string | null;
  winner_variant?: string | null;
  completed_at?: string | null;
  created_at: string;
  variants: ExperimentVariant[];
}

export interface ExperimentVariantInput {
  name: string;
  config: Record<string, unknown>;
  run_id?: string | null;
}

export interface ExperimentCreate {
  name: string;
  application?: string | null;
  description?: string | null;
  hypothesis?: string | null;
  dataset_name?: string | null;
  variants: ExperimentVariantInput[];
  baseline_variant?: string | null;
}

export interface Application {
  id: string;
  name: string;
  description?: string | null;
  environment: string;
  is_active: boolean;
  created_at: string;
  /** Lifetime figures, not window figures. `null` when no traces exist at all. */
  trace_count?: number | null;
  last_seen_at?: string | null;
  total_tokens?: number | null;
}

export interface ModelRecord {
  id: string;
  name: string;
  provider: string;
  context_window?: number | null;
  is_local: boolean;
  input_cost_per_1k: number;
  output_cost_per_1k: number;
  description?: string | null;
}

/** Declared locally in `api/v1/applications.py`, not in `schemas/`. */
export interface ApplicationStats {
  application: string;
  application_id: string;
  window: string;
  start: string;
  end: string;
  trace_count: number;
  total_tokens: number;
  estimated_cost: number;
  cost_label: string;
  avg_latency_ms?: number | null;
  /** Fraction in [0, 1]. */
  error_rate: number;
  model_usage: ModelUsageRow[];
  time_series: TimeSeriesPoint[];
}

/* -------------------------------------------------------------------------- */
/* Schemas — tenancy.py                                                       */
/* -------------------------------------------------------------------------- */

/**
 * One company, as `OrganizationOut` returns it.
 *
 * `num_applications` and `num_api_keys` are counts of rows that exist, not
 * figures derived from telemetry, so they are exact and cheap. `slug` is
 * nullable because a name that slugifies to nothing — a company named entirely
 * in non-Latin script, say — yields no handle rather than an error.
 */
export interface Organization {
  id: string;
  name: string;
  slug?: string | null;
  is_active: boolean;
  num_applications: number;
  num_api_keys: number;
  created_at: string;
  updated_at: string;
}

/**
 * One API key's metadata. Deliberately has no plaintext field, and this type
 * says so by omission rather than by convention: `ApiKeyOut` on the backend has
 * no column to populate one from, so a key read from the list is a *shape* that
 * cannot carry a credential at all.
 */
export interface ApiKey {
  id: string;
  organization_id: string;
  name: string;
  /**
   * The first 12 characters of the key. Enough to recognise which key a row is,
 * never enough to authenticate — the remaining 35 characters come from 256 bits
   * of CSPRNG.
   */
  key_prefix: string;
  /** Canonicalised and sorted, so `read,ingest` and `ingest,read` are one grant. */
  scopes: string[];
  /** Derived server-side: not revoked, and not past `expires_at`. */
  is_active: boolean;
  last_used_at?: string | null;
  revoked_at?: string | null;
  expires_at?: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * A newly minted key, carrying its plaintext.
 *
 * This is the only shape in the file with a usable credential on it, and it
 * exists on exactly one response. Everything that reads keys afterwards works
 * with {@link ApiKey}, which has no field to leak. A caller that keeps the whole
 * object around past the moment the operator copied the key is holding a secret
 * in component state; clearing it is the panel's job, not the type's.
 */
export interface CreatedApiKey extends ApiKey {
  key: string;
}

/** The body `POST /organizations/{id}/api-keys` accepts. */
export interface ApiKeyCreateRequest {
  name: string;
  scopes?: string[] | null;
  expires_at?: string | null;
}

/* -------------------------------------------------------------------------- */
/* Endpoints                                                                  */
/* -------------------------------------------------------------------------- */

export const api = {
  /* Health ---------------------------------------------------------------- */
  health: (): Promise<HealthResponse> => get<HealthResponse>("/health"),

  /* Dashboard ------------------------------------------------------------- */
  dashboardOverview: (scope: ScopeParams = {}): Promise<DashboardOverview> =>
    get<DashboardOverview>("/dashboard/overview", { ...scope }),

  /* Analytics ------------------------------------------------------------- */
  tokenAnalytics: (scope: ScopeParams = {}): Promise<TokenAnalytics> =>
    get<TokenAnalytics>("/analytics/tokens", { ...scope }),
  tokensByUser: (scope: ScopeParams = {}): Promise<BreakdownItem[]> =>
    get<BreakdownItem[]>("/analytics/tokens/by-user", { ...scope }),
  tokensByModel: (scope: ScopeParams = {}): Promise<BreakdownItem[]> =>
    get<BreakdownItem[]>("/analytics/tokens/by-model", { ...scope }),
  costAnalytics: (scope: ScopeParams = {}): Promise<CostAnalytics> =>
    get<CostAnalytics>("/analytics/cost", { ...scope }),
  latencyAnalytics: (scope: ScopeParams = {}): Promise<LatencyAnalytics> =>
    get<LatencyAnalytics>("/analytics/latency", { ...scope }),
  tokenEfficiency: (scope: ScopeParams = {}): Promise<TokenEfficiencyReport> =>
    get<TokenEfficiencyReport>("/analytics/token-efficiency", { ...scope }),
  costQuality: (scope: ScopeParams = {}): Promise<CostQualityReport> =>
    get<CostQualityReport>("/analytics/cost-quality", { ...scope }),

  /* Traces ---------------------------------------------------------------- */
  listTraces: (filters: TraceFilters = {}): Promise<Page<TraceSummary>> =>
    get<Page<TraceSummary>>("/traces", { ...filters }),
  getTrace: (traceId: string): Promise<TraceDetail> =>
    get<TraceDetail>(`/traces/${encodeURIComponent(traceId)}`),

  /* Evaluations ----------------------------------------------------------- */
  runRetrievalEvaluation: (
    body: RetrievalEvaluationRequest,
  ): Promise<EvaluationRunDetail> => post<EvaluationRunDetail>("/evaluations/retrieval", body),
  evaluateAnswers: (body: AnswerEvaluationRequest): Promise<AnswerEvaluationSummary> =>
    post<AnswerEvaluationSummary>("/evaluations/answer", body),
  listEvaluationRuns: (
    filters: ScopeParams & PageParams & {
      evaluation_type?: string | null;
      status?: string | null;
      dataset_name?: string | null;
    } = {},
  ): Promise<Page<EvaluationRunSummary>> =>
    get<Page<EvaluationRunSummary>>("/evaluations/runs", { ...filters }),
  getEvaluationRun: (runId: string): Promise<EvaluationRunDetail> =>
    get<EvaluationRunDetail>(`/evaluations/runs/${encodeURIComponent(runId)}`),
  compareRuns: (body: CompareRequest): Promise<RegressionReport> =>
    post<RegressionReport>("/evaluations/compare", body),

  /* Anomalies ------------------------------------------------------------- */
  listAnomalies: (
    filters: ScopeParams &
      PageParams & {
        anomaly_type?: string | null;
        severity?: string | null;
        is_resolved?: boolean | null;
      } = {},
  ): Promise<Page<Anomaly>> => get<Page<Anomaly>>("/anomalies", { ...filters }),
  detectAnomalies: (body: AnomalyDetectRequest): Promise<AnomalyDetectResponse> =>
    post<AnomalyDetectResponse>("/anomalies/detect", body),
  resolveAnomaly: (anomalyId: string, isResolved: boolean): Promise<Anomaly> =>
    patch<Anomaly>(`/anomalies/${encodeURIComponent(anomalyId)}`, { is_resolved: isResolved }),

  /* Recommendations ------------------------------------------------------- */
  listRecommendations: (filters: RecommendationFilters = {}): Promise<Page<Recommendation>> =>
    get<Page<Recommendation>>("/recommendations", { ...filters }),
  generateRecommendations: (
    body: RecommendationGenerateRequest,
  ): Promise<RecommendationGenerateResponse> =>
    post<RecommendationGenerateResponse>("/recommendations/generate", body),

  /* Experiments ----------------------------------------------------------- */
  listExperiments: (filters: ScopeParams & PageParams = {}): Promise<Page<Experiment>> =>
    get<Page<Experiment>>("/experiments", { ...filters }),
  getExperiment: (id: string): Promise<Experiment> =>
    get<Experiment>(`/experiments/${encodeURIComponent(id)}`),
  createExperiment: (body: ExperimentCreate): Promise<Experiment> =>
    post<Experiment>("/experiments", body),

  /* Applications ---------------------------------------------------------- */
  listApplications: (
    filters: PageParams & { include_inactive?: boolean } = {},
  ): Promise<Page<Application>> =>
    get<Page<Application>>("/applications", { ...filters }),
  applicationStats: (id: string, scope: ScopeParams = {}): Promise<ApplicationStats> =>
    get<ApplicationStats>(`/applications/${encodeURIComponent(id)}/stats`, { ...scope }),

  /* Models ---------------------------------------------------------------- */
  listModels: (): Promise<ModelRecord[]> => get<ModelRecord[]>("/models"),

  /* Organizations --------------------------------------------------------- */
  /**
   * Every organization, active first then by name.
   *
   * A plain array rather than a `Page`: the set of companies is bounded by how
   * many an operator has onboarded, and the Settings panel needs all of them to
   * populate a picker. Platform-privileged — a tenant key gets a 403, which the
   * panel surfaces as such rather than as an empty list.
   */
  listOrganizations: (): Promise<Organization[]> => get<Organization[]>("/organizations"),

  createOrganization: (body: { name: string; slug?: string | null }): Promise<Organization> =>
    post<Organization>("/organizations", body),

  listApiKeys: (organizationId: string): Promise<ApiKey[]> =>
    get<ApiKey[]>(`/organizations/${encodeURIComponent(organizationId)}/api-keys`),

  /**
   * Mint a key. The response is the only place the plaintext exists.
   *
   * There is deliberately no second call that retrieves a key by id: the backend
   * has no route for one, and adding a client method for a route that does not
   * exist is how a "fetch my key" button gets built later and wired to a 404.
   */
  createApiKey: (organizationId: string, body: ApiKeyCreateRequest): Promise<CreatedApiKey> =>
    post<CreatedApiKey>(`/organizations/${encodeURIComponent(organizationId)}/api-keys`, body),

  /** Revoke. Stamps `revoked_at` server-side; the row stays for the audit trail. */
  revokeApiKey: (organizationId: string, keyId: string): Promise<MessageResponse> =>
    del<MessageResponse>(
      `/organizations/${encodeURIComponent(organizationId)}/api-keys/${encodeURIComponent(keyId)}`,
    ),

  /**
   * Post one trace, and nothing else.
   *
   * The only write this console makes, and it exists so "Send a test trace"
   * proves a key rather than merely reporting that the backend answered. Every
   * read in the API is reachable without a key, so a read succeeding says
   * nothing about the credential; `POST /traces` sits behind `WriteGuard`, so a
   * 201 here is the backend having accepted the key itself. Nothing is stored
   * locally and nothing is passed through `localStorage` — the key is already
   * the active one, or the request is the failure the operator needs to see.
   *
   * The response shape is `create_trace`'s own: `{trace_id, status}`, with no
   * `id` and no nullability. `trace_id` is generated by `ingest_trace`, so a 201
   * always carries one — declaring it nullable would invite a caller to branch
   * on an "absent" id that the backend never sends.
   */
  sendTestTrace: (application: string): Promise<{ trace_id: string; status: string }> =>
    post<{ trace_id: string; status: string }>("/traces", {
      application,
      kind: "chat",
      status: "running",
      name: "RAGOps onboarding check",
    }),
} as const;

/* -------------------------------------------------------------------------- */
/* Reads over the response shapes                                             */
/* -------------------------------------------------------------------------- */

/**
 * The retrieval metrics for one K out of `EvaluationRun.metrics`.
 *
 * A retrieval run stores `{"k=5": {...RetrievalMetrics...}, "k=10": {...}}`
 * (see `as_metric_rows` in the evaluation service), so a run whose headline `k`
 * is 5 has nothing at the top level. Returns `null` when that K was never
 * measured — which is a different answer from a metric of zero.
 */
export function readRetrievalMetrics(
  metrics: Record<string, unknown> | null | undefined,
  k: number,
): RetrievalMetrics | null {
  if (metrics === null || metrics === undefined) return null;
  const candidate = metrics[`k=${k}`];
  if (candidate === null || typeof candidate !== "object") return null;
  return candidate as RetrievalMetrics;
}

/** Every K a retrieval run reported, ascending. */
export function retrievalKValues(metrics: Record<string, unknown> | null | undefined): number[] {
  if (metrics === null || metrics === undefined) return [];
  const keys = Object.keys(metrics);
  const ks: number[] = [];
  for (const key of keys) {
    if (!key.startsWith("k=")) continue;
    const parsed = Number(key.slice(2));
    if (Number.isFinite(parsed)) ks.push(parsed);
  }
  return ks.sort((a, b) => a - b);
}

/** The K values a run recorded, falling back to the run's own `k`. */
export function runKValues(run: EvaluationRunSummary): number[] {
  const ks = retrievalKValues(run.metrics);
  if (ks.length > 0) return ks;
  return run.k > 0 ? [run.k] : [];
}
