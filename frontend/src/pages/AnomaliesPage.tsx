import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, CheckCircle2, Radar, Search, Wand2 } from "lucide-react";
import {
  api,
  type Anomaly,
  type AnomalyDetectRequest,
  type AnomalyDetectResponse,
} from "@/lib/api";
import { useScope, usePageParam, useFilterParam } from "@/hooks/useScope";
import { InfoNote, PageHeader, PanelTitle, SelectField, ToggleField } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { DataTable, Pagination, type Column } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import { Badge, SeverityBadge } from "@/components/Badge";
import { KeyValueList } from "@/components/Table";
import { StatCard, StatRow } from "@/components/StatCard";
import { CategoryBarChart } from "@/lib/charts";
import { ChartCard } from "@/components/ChartCard";
import { ErrorPanel } from "@/components/ErrorPanel";
import { formatDateTime, formatNumber, formatRelative, humanize, truncate } from "@/lib/format";

/**
 * §16: anomaly detection over recorded traces.
 *
 * The page is split into two halves that must not be confused:
 *
 * - **Recorded anomalies** are rows the backend already stored. This list is
 *   history.
 * - **A detection run** fits an IsolationForest on the traces in a lookback
 *   window and returns what it flags *now*. The result panel always reports the
 *   samples it was fit on, the contamination it was given, and the fact that
 *   below 10 samples it refuses to fit at all — a run that silently produced
 *   nothing would be indistinguishable from a clean system.
 *
 * An outlier is not a diagnosis. Nothing here says *why* a trace was unusual;
 * the peer medians the backend records are the evidence, and the page shows
 * them rather than a guess.
 */

/** `ml/anomaly.py` `DEFAULT_FEATURES`, in the order the backend lists them. */
const FEATURE_OPTIONS = [
  { value: "total_tokens", label: "Total tokens" },
  { value: "duration_ms", label: "Duration" },
  { value: "context_tokens", label: "Context tokens" },
  { value: "num_documents", label: "Documents retrieved" },
  { value: "agent_iterations", label: "Agent iterations" },
  { value: "estimated_cost", label: "Estimated cost" },
] as const;

const SEVERITY_OPTIONS = [
  { value: "", label: "Any severity" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

const RESOLVED_OPTIONS = [
  { value: "", label: "Open and resolved" },
  { value: "false", label: "Open only" },
  { value: "true", label: "Resolved only" },
];

export default function AnomaliesPage() {
  const { params } = useScope();
  const { page, pageSize, setPage } = usePageParam();
  const severity = useFilterParam("severity");
  const resolved = useFilterParam("resolved");
  const { reset } = useScope();

  const [expanded, setExpanded] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["anomalies", params, page, pageSize, severity.value, resolved.value],
    queryFn: () =>
      api.listAnomalies({
        ...params,
        page,
        page_size: pageSize,
        ...(severity.value === "" ? {} : { severity: severity.value }),
        ...(resolved.value === "" ? {} : { is_resolved: resolved.value === "true" }),
      }),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Anomalies"
        description="Traces flagged as outliers by IsolationForest over the recorded numeric features. A flag means the numbers were unusual for their peers; it is not a diagnosis, and the peer statistics it was scored against are shown with every row."
        scope={<ScopeBar />}
      />

      <DetectPanel application={params.application ?? null} timeWindow={params.window ?? "7d"} />

      <QueryBoundary query={list} label="anomalies">
        {(data) => (
          <>
            <StatRow>
              <StatCard label="Total flagged" value={formatNumber(data.total)} icon={<AlertTriangle className="h-4 w-4" aria-hidden="true" />} />
              <StatCard label="Open" value={formatNumber(data.items.filter((row) => !row.is_resolved).length)} hint="On this page" />
              <StatCard label="High severity" value={formatNumber(data.items.filter((row) => row.severity === "high").length)} hint="On this page" />
              <StatCard label="Distinct metrics" value={formatNumber(new Set(data.items.map((row) => row.metric)).size)} hint="On this page" />
              <StatCard label="Page" value={`${formatNumber(data.page)} of ${formatNumber(Math.max(1, Math.ceil(data.total / data.page_size)))}`} />
              <StatCard label="Page size" value={formatNumber(data.page_size)} hint="Rows per request" />
            </StatRow>

            <div className="card p-4">
              <PanelTitle
                actions={
                  <div className="flex flex-wrap items-end gap-3">
                    <SelectField
                      id="anomaly-severity"
                      label="Severity"
                      className="w-36"
                      value={severity.value}
                      options={SEVERITY_OPTIONS}
                      onChange={severity.setValue}
                    />
                    <SelectField
                      id="anomaly-resolved"
                      label="State"
                      className="w-40"
                      value={resolved.value}
                      options={RESOLVED_OPTIONS}
                      onChange={resolved.setValue}
                    />
                    {severity.isSet || resolved.isSet ? (
                      <button type="button" className="btn-ghost" onClick={reset}>
                        Clear
                      </button>
                    ) : null}
                  </div>
                }
              >
                Recorded anomalies
              </PanelTitle>

              <DataTable
                caption="Anomalies detected by IsolationForest"
                columns={anomalyColumns(setExpanded, expanded)}
                rows={data.items}
                rowKey={(row) => row.id}
                onRowClick={(row) => setExpanded((current) => (current === row.id ? null : row.id))}
                isRowActive={(row) => row.id === expanded}
                minWidth="56rem"
                empty={
                  <EmptyState
                    icon={<Search className="h-6 w-6" aria-hidden="true" />}
                    title="No anomalies recorded"
                    description="Nothing has been flagged in this window. That either means the traces were within their peer range, or that no detection run has been persisted yet — run one above to tell the two apart."
                  />
                }
              />
              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={setPage}
                itemLabel="anomalies"
              />
            </div>
          </>
        )}
      </QueryBoundary>
    </div>
  );
}

const anomalyColumns = (
  setExpanded: (next: (current: string | null) => string | null) => void,
  expanded: string | null,
): Column<Anomaly>[] => [
  {
    key: "metric",
    header: "Metric",
    render: (row) => (
      <span className="flex flex-col">
        <span className="font-medium text-ink">{humanize(row.metric)}</span>
        <span className="text-2xs text-ink-muted">
          {row.detection_method} · {row.application_name ?? "all applications"}
        </span>
      </span>
    ),
  },
  { key: "severity", header: "Severity", render: (row) => <SeverityBadge severity={row.severity} /> },
  {
    key: "observed",
    header: "Observed",
    align: "right",
    render: (row) => formatNumber(row.observed_value, 2),
  },
  {
    key: "peer",
    header: "Peer range",
    align: "right",
    render: (row) => (
      <span className="whitespace-nowrap text-2xs text-ink-secondary">
        {formatNumber(row.expected_low, 2)} – {formatNumber(row.expected_high, 2)}
        <br />
        <span className="text-ink-muted">
          median {formatNumber(row.peer_median, 2)}
        </span>
      </span>
    ),
  },
  {
    key: "score",
    header: "Score",
    align: "right",
    render: (row) => formatNumber(row.anomaly_score, 4),
  },
  {
    key: "detected",
    header: "Detected",
    align: "right",
    render: (row) => (
      <span title={formatDateTime(row.detected_at)}>{formatRelative(row.detected_at)}</span>
    ),
  },
  {
    key: "state",
    header: "State",
    render: (row) => <ResolveCell row={row} />,
  },
  {
    key: "trace",
    header: "Trace",
    render: (row) =>
      row.trace_id === null ? (
        <span className="no-value">—</span>
      ) : (
        <TraceLink traceId={row.trace_id} />
      ),
  },
  {
    key: "details",
    header: "Evidence",
    render: (row) => (
      <button
        type="button"
        className="btn-ghost px-2 py-1"
        aria-expanded={row.id === expanded}
        onClick={(event) => {
          event.stopPropagation();
          setExpanded((current) => (current === row.id ? null : row.id));
        }}
      >
        {row.id === expanded ? "Hide" : "Show"}
      </button>
    ),
  },
];

function TraceLink({ traceId }: { traceId: string }) {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      className="font-mono text-2xs text-brand hover:underline"
      onClick={(event) => {
        event.stopPropagation();
        navigate(`/traces/${encodeURIComponent(traceId)}`);
      }}
    >
      {truncate(traceId, 16)}
    </button>
  );
}

/** Resolve toggle, wired to `PATCH /anomalies/{id}`. */
function ResolveCell({ row }: { row: Anomaly }) {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => api.resolveAnomaly(row.id, !row.is_resolved),
    onSuccess: () => {
      // Invalidate rather than patch the cache in place: the list is filtered by
      // resolution state, so a row flipping may need to leave the list entirely.
      void queryClient.invalidateQueries({ queryKey: ["anomalies"] });
    },
  });

  return (
    <button
      type="button"
      className="btn-ghost px-2 py-1"
      disabled={mutation.isPending}
      onClick={(event) => {
        event.stopPropagation();
        mutation.mutate();
      }}
    >
      <Badge tone={row.is_resolved ? "good" : "warning"}>
        {mutation.isPending ? "saving…" : row.is_resolved ? "resolved" : "open"}
      </Badge>
    </button>
  );
}

/* -------------------------------------------------------------------------- */
/* Detection run                                                              */
/* -------------------------------------------------------------------------- */

function DetectPanel({
  application,
  timeWindow,
}: {
  application: string | null;
  timeWindow: string;
}) {
  const queryClient = useQueryClient();
  const [lookback, setLookback] = useState(24);
  const [contamination, setContamination] = useState(0.05);
  const [features, setFeatures] = useState<string[]>(FEATURE_OPTIONS.map((option) => option.value));
  const [persist, setPersist] = useState(false);
  const [result, setResult] = useState<AnomalyDetectResponse | null>(null);

  const run = useMutation({
    mutationFn: (body: AnomalyDetectRequest) => api.detectAnomalies(body),
    onSuccess: (data) => {
      setResult(data);
      // A persisting run writes rows the list page reads, so the list has to
      // be re-fetched; the scope may have changed underneath it too.
      void queryClient.invalidateQueries({ queryKey: ["anomalies"] });
    },
  });

  const toggleFeature = (value: string): void => {
    setFeatures((current) =>
      current.includes(value) ? current.filter((entry) => entry !== value) : [...current, value],
    );
  };

  return (
    <section className="card p-4" aria-label="Run anomaly detection">
      <PanelTitle
        actions={
          <button
            type="button"
            className="btn-primary"
            disabled={run.isPending || features.length === 0}
            onClick={() =>
              run.mutate({
                application,
                window: timeWindow,
                lookback_hours: lookback,
                contamination,
                features,
                persist,
              })
            }
          >
            <Radar className="h-3.5 w-3.5" aria-hidden="true" />
            {run.isPending ? "Detecting…" : "Run detection"}
          </button>
        }
      >
        Run detection
      </PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        IsolationForest is fit on the numeric features of every trace in the lookback window, then
        scores the same rows; anything the forest isolates is reported. Below ten samples the backend
        refuses to fit — it says so rather than returning an empty result that would read as a clean
        system.
      </p>

      <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SelectField
          id="anomaly-lookback"
          label="Lookback"
          className="lg:w-40"
          value={String(lookback)}
          options={[
            { value: "1", label: "Last hour" },
            { value: "6", label: "Last 6 hours" },
            { value: "24", label: "Last 24 hours" },
            { value: "72", label: "Last 3 days" },
            { value: "168", label: "Last 7 days" },
          ]}
          onChange={(value) => setLookback(Number(value))}
        />
        <SelectField
          id="anomaly-contamination"
          label="Expected outlier rate"
          className="lg:w-40"
          value={String(contamination)}
          options={[
            { value: "0.01", label: "1%" },
            { value: "0.05", label: "5%" },
            { value: "0.1", label: "10%" },
            { value: "0.2", label: "20%" },
          ]}
          onChange={(value) => setContamination(Number(value))}
          hint="Assumed fraction of outliers. The forest is tilted to flag roughly this many."
        />
        <div className="flex items-end">
          <ToggleField
            id="anomaly-persist"
            label="Save the findings"
            checked={persist}
            onChange={setPersist}
            hint="Off: the run is returned here and discarded."
          />
        </div>
      </div>

      <fieldset className="mb-4">
        <legend className="field-label">Features</legend>
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          {FEATURE_OPTIONS.map((option) => (
            <label key={option.value} className="flex items-center gap-1.5 text-sm text-ink-secondary">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 rounded border-line text-brand focus-visible:ring-2 focus-visible:ring-brand"
                checked={features.includes(option.value)}
                onChange={() => toggleFeature(option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
      </fieldset>

      {run.isError ? (
        <ErrorPanel
          error={run.error}
          subject="anomaly detection"
          onRetry={() => run.reset()}
        />
      ) : null}

      {result === null ? null : <DetectResult result={result} />}
    </section>
  );
}

function DetectResult({ result }: { result: AnomalyDetectResponse }) {
  const topFeatures = Object.entries(result.feature_importance_proxy)
    .map(([feature, magnitude]) => ({ feature, magnitude }))
    .sort((left, right) => right.magnitude - left.magnitude);

  return (
    <div className="space-y-3 border-t border-line pt-3">
      <StatRow>
        <StatCard label="Samples fit" value={formatNumber(result.num_samples)} hint="Traces with the features present" />
        <StatCard label="Features used" value={formatNumber(result.num_features)} />
        <StatCard
          label="Flagged"
          value={formatNumber(result.num_anomalies)}
          tone={result.num_anomalies === 0 ? "good" : "flat"}
          icon={<Wand2 className="h-4 w-4" aria-hidden="true" />}
        />
        <StatCard
          label="Flagged rate"
          value={`${formatNumber(result.anomaly_rate * 100, 2)}%`}
          hint={`Contamination assumed at ${formatNumber(result.contamination * 100, 1)}%`}
        />
        <StatCard label="Detection duration" value={`${formatNumber(result.duration_ms, 1)} ms`} />
        <StatCard
          label="Method"
          value={<span className="text-base">{result.detection_method}</span>}
          hint={result.application_name ?? "All applications"}
        />
      </StatRow>

      {result.num_samples < 10 ? (
        <InfoNote tone="warning">
          Only {formatNumber(result.num_samples)} samples were available, below the ten the backend
          requires. IsolationForest was not fit, so there are no findings — this is missing data,
          not a clean window.
        </InfoNote>
      ) : null}

      {result.anomalies.length === 0 ? (
        <EmptyState
          icon={<CheckCircle2 className="h-6 w-6" aria-hidden="true" />}
          title="No outliers in this window"
          description={`The forest isolated no traces among ${formatNumber(result.num_samples)} samples at ${formatNumber(result.contamination * 100, 1)}% contamination.`}
        />
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          <ChartCard
            title="Which features deviated most"
            description="Mean absolute z-score against the peer distribution, per feature. This is a deviation magnitude, not a fitted importance score — it says where the outliers were far from the pack, not which feature the model weighted."
            valueLabel="Mean absolute deviation"
            table={{
              columns: ["Feature", "Mean |z|"],
              rows: topFeatures.map((row) => [humanize(row.feature), formatNumber(row.magnitude, 3)]),
            }}
          >
            <CategoryBarChart
              data={topFeatures.map((row) => ({ label: humanize(row.feature), value: row.magnitude }))}
              valueLabel="Mean |z|"
              format={(value) => formatNumber(value, 2)}
              tone="warning"
              maxBars={8}
            />
          </ChartCard>

          <div className="card p-4">
            <h3 className="mb-1 text-sm font-semibold text-ink">Flagged traces</h3>
            <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
              Not persisted — this run was returned for inspection. Tick “Save the findings” and run
              again to record them in the list below.
            </p>
            <ul className="max-h-80 space-y-1.5 overflow-auto scrollbar-thin pr-1">
              {result.anomalies.map((anomaly) => (
                <li
                  key={anomaly.id === "" ? `${anomaly.trace_id}-${anomaly.metric}` : anomaly.id}
                  className="rounded-md border border-line bg-surface-sunken px-3 py-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-medium text-ink">{humanize(anomaly.metric)}</span>
                    <SeverityBadge severity={anomaly.severity} />
                  </div>
                  <KeyValueList
                    columns={3}
                    className="mt-2"
                    items={[
                      { label: "Observed", value: formatNumber(anomaly.observed_value, 2) },
                      { label: "Peer median", value: formatNumber(anomaly.peer_median, 2) },
                      { label: "Peer p95", value: formatNumber(anomaly.peer_p95, 2) },
                    ]}
                  />
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
