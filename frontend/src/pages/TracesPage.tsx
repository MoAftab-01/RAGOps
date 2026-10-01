import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, ExternalLink, FileText, Search, Zap } from "lucide-react";
import { api, type Span, type TraceDetail, type TraceSummary, type Retrieval, type LLMCall } from "@/lib/api";
import { usePageParam, useScope } from "@/hooks/useScope";
import { PageHeader, InfoNote, PanelTitle, TextField, SelectField } from "@/components/PageHeader";
import { ScopeBar, QueryBoundary } from "@/components/ScopeBar";
import { DataTable, Pagination, DetailSection, KeyValueList, type Column } from "@/components/Table";
import { Badge, statusTone } from "@/components/Badge";
import { EmptyState } from "@/components/EmptyState";
import {
  formatCost,
  formatDateTime,
  formatLatencyMs,
  formatMs,
  formatNumber,
  formatScore,
  formatSigned,
  humanize,
  truncate,
} from "@/lib/format";

/**
 * §10: the trace list and the expanded trace view.
 *
 * A trace is a request's full lifecycle — spans, retrievals, and LLM calls —
 * and the reason to show all three separately is that they answer different
 * questions. Spans say where the time went. Retrievals say whether the right
 * passages were found. LLM calls say what was actually sent and charged for. A
 * single blended "duration" number would hide the case that matters most, which
 * is a slow trace that was slow in retrieval rather than in generation.
 */

// These option lists transcribe the values the backend documents on
// `GET /api/traces` (status: success | error | running | cancelled; kind:
// chat | rag | agent | evaluation). A value the server does not recognise
// matches nothing rather than being ignored, so an option list that drifts
// from those enumerations silently produces empty pages.
const STATUS_OPTIONS = [
  { value: "", label: "Any status" },
  { value: "success", label: "Success" },
  { value: "error", label: "Error" },
  { value: "running", label: "Running" },
  { value: "cancelled", label: "Cancelled" },
];

const KIND_OPTIONS = [
  { value: "", label: "Any kind" },
  { value: "chat", label: "Chat" },
  { value: "rag", label: "RAG" },
  { value: "agent", label: "Agent" },
  { value: "evaluation", label: "Evaluation" },
];

export default function TracesPage() {
  const { params } = useScope();
  const { page, pageSize, setPage } = usePageParam(25);
  const navigate = useNavigate();
  const { traceId: routeTraceId } = useParams();

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const [hasError, setHasError] = useState(false);
  const [selected, setSelected] = useState<string | null>(routeTraceId ?? null);

  const traces = useQuery({
    queryKey: ["traces", params, page, pageSize, search, status, kind, hasError],
    queryFn: () =>
      api.listTraces({
        ...params,
        page,
        page_size: pageSize,
        search: search === "" ? null : search,
        status: status === "" ? null : status,
        kind: kind === "" ? null : kind,
        has_error: hasError ? true : null,
      }),
  });

  // A deep link to one trace should open it, and the URL should stay the source
  // of truth — otherwise the back button cannot close a detail view.
  const activeId = routeTraceId ?? selected;
  const open = (id: string): void => {
    setSelected(id);
    navigate(`/traces/${encodeURIComponent(id)}`);
  };
  const close = (): void => {
    setSelected(null);
    navigate("/traces");
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Traces"
        description="Every recorded request, with the stages it went through. Select a row to expand its spans, retrievals and model calls."
        scope={
          <ScopeBar
            extra={
              <>
                <TextField
                  id="trace-search"
                  label="Search"
                  value={search}
                  onChange={(value) => {
                    setSearch(value);
                    setPage(1);
                  }}
                  placeholder="trace id, user, session"
                  className="min-w-40"
                />
                <SelectField
                  id="trace-status"
                  label="Status"
                  value={status}
                  options={STATUS_OPTIONS}
                  onChange={setStatus}
                />
                <SelectField id="trace-kind" label="Kind" value={kind} options={KIND_OPTIONS} onChange={setKind} />
                <label className="flex items-center gap-1.5 self-end pb-1.5 text-xs text-ink-secondary">
                  <input
                    type="checkbox"
                    checked={hasError}
                    onChange={(event) => {
                      setHasError(event.target.checked);
                      setPage(1);
                    }}
                    className="rounded border-line"
                  />
                  Errors only
                </label>
              </>
            }
          />
        }
      />

      <QueryBoundary query={traces} label="traces">
        {(data) => (
          <div className="card">
            <DataTable<TraceSummary>
              caption="Recorded traces"
              columns={traceColumns(open, activeId)}
              rows={data.items}
              rowKey={(row) => row.trace_id}
              onRowClick={(row) => open(row.trace_id)}
              isRowActive={(row) => row.trace_id === activeId}
              minWidth="64rem"
              empty={
                <EmptyState
                  title="No traces match these filters"
                  description="Try widening the time window, or clearing the search and status filters."
                  icon={<Search className="h-6 w-6" aria-hidden="true" />}
                />
              }
            />
            <div className="border-t border-line">
              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={setPage}
                itemLabel="traces"
              />
            </div>
          </div>
        )}
      </QueryBoundary>

      {activeId === null ? null : (
        <TraceDetailPanel traceId={activeId} onClose={close} />
      )}
    </div>
  );
}

function traceColumns(
  onOpen: (id: string) => void,
  activeId: string | null,
): Column<TraceSummary>[] {
  return [
    {
      key: "trace",
      header: "Trace",
      render: (row) => (
        <span className="flex items-center gap-1.5 font-mono text-xs text-ink">
          <ChevronRight
            className={row.trace_id === activeId ? "h-3 w-3 text-brand" : "h-3 w-3 text-ink-muted"}
            aria-hidden="true"
          />
          {truncate(row.trace_id, 16)}
        </span>
      ),
    },
    {
      key: "application",
      header: "Application",
      render: (row) => (
        <span className="flex flex-col">
          <span className="text-ink">{row.application_name ?? row.application_id}</span>
          <span className="text-2xs text-ink-muted">
            {humanize(row.kind)}
            {row.agent_name === null || row.agent_name === undefined ? "" : ` · ${row.agent_name}`}
          </span>
        </span>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (row) => (
        <span className="flex flex-wrap items-center gap-1">
          <Badge tone={statusTone(row.status)}>{row.status}</Badge>
          {row.has_error ? <Badge tone="critical">error</Badge> : null}
        </span>
      ),
    },
    { key: "started", header: "Started", render: (row) => formatDateTime(row.start_time) },
    { key: "duration", header: "Duration", align: "right", render: (row) => formatLatencyMs(row.duration_ms) },
    { key: "tokens", header: "Tokens", align: "right", render: (row) => formatNumber(row.total_tokens) },
    { key: "cost", header: "Cost", align: "right", render: (row) => formatCost(row.estimated_cost) },
    {
      key: "open",
      header: <span className="sr-only">Open</span>,
      align: "right",
      render: (row) => (
        <button
          type="button"
          className="btn-ghost"
          onClick={() => onOpen(row.trace_id)}
          aria-label={`Open trace ${row.trace_id}`}
        >
          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      ),
    },
  ];
}

function TraceDetailPanel({ traceId, onClose }: { traceId: string; onClose: () => void }) {
  const detail = useQuery({
    queryKey: ["trace", traceId],
    queryFn: () => api.getTrace(traceId),
  });

  return (
    <section className="card p-4" aria-label={`Trace ${traceId}`}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <PanelTitle className="mb-0">
          <span className="font-mono">{truncate(traceId, 24)}</span>
        </PanelTitle>
        <button type="button" className="btn-ghost" onClick={onClose}>
          Close
        </button>
      </div>

      <QueryBoundary query={detail} label="trace detail">
        {(trace) => <TraceBody trace={trace} />}
      </QueryBoundary>
    </section>
  );
}

function TraceBody({ trace }: { trace: TraceDetail }) {
  // `retrieval_analysis` is `default_factory=dict` on the backend and the trace
// router already coalesces a missing blob with `or {}`, so it is never null
// here — a `?? {}` would suggest the field can be absent, and a future reader
// would come to rely on that.
const analysis = trace.retrieval_analysis;
  const duplicates = analysis.duplicate_document_ids ?? [];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={statusTone(trace.status)}>{trace.status}</Badge>
        <Badge>{humanize(trace.kind)}</Badge>
        {trace.application_name === null || trace.application_name === undefined ? null : (
          <Badge>{trace.application_name}</Badge>
        )}
        {trace.agent_name === null || trace.agent_name === undefined ? null : (
          <Badge tone="brand">agent: {trace.agent_name}</Badge>
        )}
        {trace.agent_iterations === null || trace.agent_iterations === undefined ? null : (
          <Badge tone="brand">{trace.agent_iterations} iterations</Badge>
        )}
      </div>

      {trace.error === null || trace.error === undefined ? null : (
        <InfoNote tone="warning">
          <span className="font-medium text-ink">Error recorded on this trace: </span>
          {trace.error}
        </InfoNote>
      )}

      <KeyValueList
        columns={3}
        items={[
          { label: "Started", value: formatDateTime(trace.start_time) },
          { label: "Ended", value: formatDateTime(trace.end_time) },
          { label: "Duration", value: formatLatencyMs(trace.duration_ms) },
          { label: "Input tokens", value: formatNumber(trace.input_tokens) },
          { label: "Output tokens", value: formatNumber(trace.output_tokens) },
          { label: "Total tokens", value: formatNumber(trace.total_tokens) },
          { label: "Context tokens", value: formatNumber(trace.context_tokens) },
          { label: "Estimated cost", value: formatCost(trace.estimated_cost) },
          { label: "User", value: trace.user_id ?? "—" },
          { label: "Session", value: trace.session_id ?? "—" },
          { label: "Tags", value: (trace.tags ?? []).join(", ") || "—" },
          { label: "Recorded", value: formatDateTime(trace.created_at) },
        ]}
      />

      {duplicates.length > 0 ? (
        <InfoNote tone="warning">
          <span className="font-medium text-ink">Retrieved context contained repeats: </span>
          {duplicates.length} document{duplicates.length === 1 ? "" : "s"} appeared more than once in the
          top-k selection, costing context tokens without adding information.
        </InfoNote>
      ) : null}

      <div className="grid gap-5 xl:grid-cols-2">
        <DetailSection title="Spans" count={trace.spans.length}>
          {trace.spans.length === 0 ? (
            <p className="text-sm text-ink-muted">No spans recorded.</p>
          ) : (
            <ol className="space-y-1.5">
              {trace.spans.map((span) => (
                <SpanRow key={span.id} span={span} />
              ))}
            </ol>
          )}
        </DetailSection>

        <DetailSection title="Retrievals" count={trace.retrievals.length}>
          {trace.retrievals.length === 0 ? (
            <p className="text-sm text-ink-muted">This trace made no retrieval calls.</p>
          ) : (
            <ul className="space-y-2">
              {trace.retrievals.map((retrieval) => (
                <RetrievalRow key={retrieval.id} retrieval={retrieval} />
              ))}
            </ul>
          )}
        </DetailSection>
      </div>

      <DetailSection title="Model calls" count={trace.llm_calls.length}>
        {trace.llm_calls.length === 0 ? (
          <p className="text-sm text-ink-muted">This trace made no model calls.</p>
        ) : (
          <ul className="space-y-2">
            {trace.llm_calls.map((call) => (
              <LLMRow key={call.id} call={call} />
            ))}
          </ul>
        )}
      </DetailSection>

      <div className="grid gap-5 xl:grid-cols-2">
        <DetailSection title="Input">
          <pre className="max-h-64 overflow-auto scrollbar-thin whitespace-pre-wrap break-words rounded-md border border-line bg-surface-sunken p-2.5 text-xs text-ink-secondary">
            {trace.input_text ?? "—"}
          </pre>
        </DetailSection>
        <DetailSection title="Output">
          <pre className="max-h-64 overflow-auto scrollbar-thin whitespace-pre-wrap break-words rounded-md border border-line bg-surface-sunken p-2.5 text-xs text-ink-secondary">
            {trace.output_text ?? "—"}
          </pre>
        </DetailSection>
      </div>
    </div>
  );
}

function SpanRow({ span }: { span: Span }) {
  return (
    <li className="rounded-md border border-line px-2.5 py-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-sm text-ink">
          <Zap className="h-3 w-3 shrink-0 text-ink-muted" aria-hidden="true" />
          {span.name}
          <span className="text-2xs text-ink-muted">{span.kind}</span>
        </span>
        <span className="flex items-center gap-2">
          <Badge tone={statusTone(span.status)}>{span.status}</Badge>
          <span className="tabular-nums text-2xs text-ink-secondary">{formatMs(span.duration_ms)}</span>
        </span>
      </div>
      {span.error === null || span.error === undefined ? null : (
        <p className="mt-1 text-2xs text-status-critical">{span.error}</p>
      )}
      {span.attributes === null || span.attributes === undefined || Object.keys(span.attributes).length === 0 ? null : (
        <dl className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-2xs sm:grid-cols-3">
          {Object.entries(span.attributes).map(([key, value]) => (
            <div key={key} className="min-w-0">
              <dt className="text-ink-muted">{humanize(key)}</dt>
              <dd className="truncate text-ink-secondary" title={String(value)}>
                {String(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </li>
  );
}

function RetrievalRow({ retrieval }: { retrieval: Retrieval }) {
  return (
    <li className="rounded-md border border-line px-2.5 py-2">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-1.5 text-sm text-ink">
          <FileText className="h-3 w-3 shrink-0 text-ink-muted" aria-hidden="true" />
          <span className="truncate">{retrieval.query}</span>
        </span>
        <span className="shrink-0 text-2xs text-ink-secondary">
          top-k {retrieval.top_k} · {retrieval.num_results} returned
        </span>
      </div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-2xs text-ink-muted">
        <span>{retrieval.retriever}</span>
        <span>{formatMs(retrieval.latency_ms)}</span>
        <span>{formatNumber(retrieval.context_tokens)} context tokens</span>
        <span>top score {formatScore(retrieval.top_score, 4)}</span>
      </div>
      {retrieval.documents.length === 0 ? (
        <p className="text-2xs text-ink-muted">No documents returned.</p>
      ) : (
        <ol className="space-y-1">
          {retrieval.documents.map((document) => (
            <li key={document.id} className="text-2xs">
              <span className="tabular-nums text-ink-muted">#{document.rank}</span>{" "}
              <span className="text-ink">{document.title ?? truncate(document.document_id, 14)}</span>{" "}
              <span className="tabular-nums text-ink-secondary">
                {formatSigned(document.final_score, 4)}
                {document.rerank_score === null || document.rerank_score === undefined
                  ? ""
                  : ` (rerank ${formatSigned(document.rerank_score, 3)})`}
              </span>
              {document.content_preview === null || document.content_preview === undefined ? null : (
                <p className="mt-0.5 line-clamp-2 text-ink-muted">{document.content_preview}</p>
              )}
            </li>
          ))}
        </ol>
      )}
    </li>
  );
}

function LLMRow({ call }: { call: LLMCall }) {
  return (
    <li className="rounded-md border border-line px-2.5 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium text-ink">{call.model_name}</span>
        <Badge tone={statusTone(call.status)}>{call.status}</Badge>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-2xs text-ink-muted">
        <span>{call.provider}</span>
        <span>
          {formatNumber(call.input_tokens)} in / {formatNumber(call.output_tokens)} out
        </span>
        <span>{formatMs(call.latency_ms)}</span>
        {call.time_to_first_token_ms === null || call.time_to_first_token_ms === undefined ? null : (
          <span>TTFT {formatMs(call.time_to_first_token_ms)}</span>
        )}
        <span>{formatCost(call.estimated_cost)}</span>
      </div>
      {call.prompt === null || call.prompt === undefined ? null : (
        <details className="mt-1.5">
          <summary className="cursor-pointer text-2xs text-ink-secondary">Prompt and completion</summary>
          <pre className="mt-1 max-h-48 overflow-auto scrollbar-thin whitespace-pre-wrap break-words rounded border border-line bg-surface-sunken p-2 text-2xs text-ink-secondary">
            {call.prompt}
            {"\n\n--- completion ---\n\n"}
            {call.completion ?? ""}
          </pre>
        </details>
      )}
    </li>
  );
}
