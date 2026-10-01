import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Scale } from "lucide-react";
import {
  api,
  readRetrievalMetrics,
  runKValues,
  type EvaluationRunDetail,
  type EvaluationRunSummary,
  type MetricDelta,
  type PerQueryResult,
  type RegressionReport,
  type RetrievalMetrics,
} from "@/lib/api";
import { usePageParam, useScope } from "@/hooks/useScope";
import { InfoNote, PageHeader, PanelTitle, SelectField, TextField } from "@/components/PageHeader";
import { ScopeBar, QueryBoundary } from "@/components/ScopeBar";
import { DataTable, Pagination, DetailSection, KeyValueList, type Column } from "@/components/Table";
import { StatCard, StatRow } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { CategoryBarChart, type SeriesSpec } from "@/lib/charts";
import { Badge, statusTone } from "@/components/Badge";
import { EmptyState } from "@/components/EmptyState";
import { ErrorPanel } from "@/components/ErrorPanel";
import {
  changeTone,
  formatDateTime,
  formatMs,
  formatNumber,
  formatScore,
  formatSignedPercent,
  humanize,
  metricDirection,
  TONE_CLASSES,
  type MetricTone,
} from "@/lib/format";

/**
 * §7 and §11: retrieval-quality evaluation and regression comparison.
 *
 * The five retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K, Hit Rate@K)
 * are all computed on the server by deterministic set and rank arithmetic over a
 * labelled dataset — never asked of a model. The page therefore shows per-query
 * rows alongside the aggregate, because an aggregate over a 20-query set hides
 * exactly the thing worth investigating: which queries retrieved nothing, and
 * which ones put the right document at rank 7.
 *
 * Regression comparison reports the *delta* and the config that changed with it.
 * It does not claim the config change caused the delta. §35 forbids fabricating
 * a root cause, and a config diff is a correlation, not a mechanism — with one
 * baseline and one candidate there is no way to separate the two.
 */

const K_OPTIONS = [1, 3, 5, 10, 20];

const METRIC_SERIES: SeriesSpec[] = [
  { key: "precision_at_k", label: "Precision@K", slot: 1, format: formatScore },
  { key: "recall_at_k", label: "Recall@K", slot: 2, format: formatScore },
  { key: "ndcg_at_k", label: "NDCG@K", slot: 3, format: formatScore },
  { key: "mrr", label: "MRR", slot: 4, format: formatScore },
  { key: "hit_rate_at_k", label: "Hit rate@K", slot: 5, format: formatScore },
];

export default function EvaluationPage() {
  const [tab, setTab] = useState<"runs" | "compare">("runs");

  return (
    <div className="space-y-5">
      <PageHeader
        title="RAG Evaluation"
        description="Precision@K, Recall@K, MRR, NDCG@K and hit rate over a labelled query set. Every figure is computed by set and rank arithmetic — no model is asked to score its own retrieval."
        scope={<ScopeBar />}
      />
      <div className="flex gap-1 border-b border-line" role="tablist" aria-label="Evaluation views">
        {(
          [
            ["runs", "Evaluation runs"],
            ["compare", "Compare two runs"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={tab === value}
            onClick={() => setTab(value)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-sm ${
              tab === value
                ? "border-brand font-medium text-brand"
                : "border-transparent text-ink-secondary hover:text-ink"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "runs" ? <RunsView /> : <CompareView />}
    </div>
  );
}

function RunsView() {
  const { params } = useScope();
  const { page, pageSize, setPage } = usePageParam(20);
  const [openRun, setOpenRun] = useState<string | null>(null);

  const runs = useQuery({
    queryKey: ["eval-runs", params, page, pageSize],
    queryFn: () => api.listEvaluationRuns({ ...params, page, page_size: pageSize }),
  });

  return (
    <>
      <div className="card p-4">
        <PanelTitle>Run a retrieval evaluation</PanelTitle>
        <p className="mb-3 max-w-prose text-2xs leading-relaxed text-ink-secondary">
          Scores a labelled dataset against the live retriever. Leave the dataset at its default to use
          the checked-in fixture, or name a path under <code className="text-ink">evaluation/</code> to
          score your own. Nothing is asked of a language model: each query's retrieved document ids are
          compared against the labels with set intersection and rank arithmetic.
        </p>
        <EvaluationRunner />
      </div>

      <QueryBoundary query={runs} label="evaluation runs">
        {(data) => (
          <div className="card">
            <DataTable<EvaluationRunSummary>
              caption="Evaluation runs"
              columns={runColumns(setOpenRun, openRun)}
              rows={data.items}
              rowKey={(row) => row.id ?? row.name}
              onRowClick={(row) => {
            if (row.id === null) return;
            setOpenRun(row.id === openRun ? null : row.id);
          }}
              isRowActive={(row) => row.id === openRun}
              minWidth="60rem"
              empty={
                <EmptyState
                  title="No evaluation runs yet"
                  description="Run one above, or compare two runs from the other tab."
                />
              }
            />
            <div className="border-t border-line">
              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={setPage}
                itemLabel="runs"
              />
            </div>
          </div>
        )}
      </QueryBoundary>

      {openRun === null ? null : <RunDetail runId={openRun} />}
    </>
  );
}

function runColumns(
  onOpen: (id: string) => void,
  openId: string | null,
): Column<EvaluationRunSummary>[] {
  return [
    {
      key: "name",
      header: "Run",
      render: (row) => (
        <span className="flex flex-col">
          <span className="font-medium text-ink">{row.name}</span>
          <span className="text-2xs text-ink-muted">
            {row.dataset_name} · k={row.k}
            {row.num_queries === 0 ? "" : ` · ${row.num_queries} queries`}
          </span>
        </span>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (row) => <Badge tone={statusTone(row.status)}>{humanize(row.status)}</Badge>,
    },
    {
      key: "metrics",
      header: "Headline metrics",
      render: (row) => {
        const ks = runKValues(row);
        const metrics = ks.length === 0 ? null : readRetrievalMetrics(row.metrics, ks[ks.length - 1]);
        if (metrics === null) return <span className="no-value">not measured</span>;
        return (
          <span className="flex flex-wrap gap-x-3 gap-y-0.5 text-2xs tabular-nums text-ink-secondary">
            <span>P@K {formatScore(metrics.precision_at_k)}</span>
            <span>R@K {formatScore(metrics.recall_at_k)}</span>
            <span>MRR {formatScore(metrics.mrr)}</span>
            <span>NDCG {formatScore(metrics.ndcg_at_k)}</span>
          </span>
        );
      },
    },
    { key: "started", header: "Started", render: (row) => formatDateTime(row.started_at ?? row.created_at) },
    { key: "duration", header: "Duration", align: "right", render: (row) => formatMs(row.duration_ms) },
    {
      key: "open",
      header: <span className="sr-only">Open</span>,
      align: "right",
      render: (row) => (
        <button
          type="button"
          className="btn-ghost"
          disabled={row.id === null}
          onClick={() => {
            if (row.id === null) return;
            onOpen(row.id === openId ? "" : row.id);
          }}
        >
          {row.id === openId ? "Close" : "Open"}
        </button>
      ),
    },
  ];
}

function EvaluationRunner() {
  const queryClient = useQueryClient();
  const { params } = useScope();
  const [dataset, setDataset] = useState("default");
  const [k, setK] = useState(5);
  const [persist, setPersist] = useState(true);
  const [result, setResult] = useState<{ metrics: Record<string, unknown> | null | undefined; k: number; ks: number[] } | null>(null);

  const run = useMutation({
    mutationFn: () =>
      api.runRetrievalEvaluation({
        dataset_name: dataset,
        k,
        ks: K_OPTIONS.filter((value) => value <= k),
        load_from_path: true,
        persist,
        application: params.application,
        name: null,
        config_override: {},
        examples: null,
      }),
    onSuccess: (summary) => {
      setResult({ metrics: summary.metrics, k: summary.k, ks: runKValues(summary) });
      void queryClient.invalidateQueries({ queryKey: ["eval-runs"] });
    },
  });

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <TextField
          id="eval-dataset"
          label="Dataset"
          value={dataset}
          onChange={setDataset}
          placeholder="default"
          className="min-w-44"
        />
        <SelectField
          id="eval-k"
          label="K (top-k)"
          value={String(k)}
          options={K_OPTIONS.map((value) => ({ value: String(value), label: `top-${value}` }))}
          onChange={(value) => setK(Number(value))}
        />
        <label className="flex items-center gap-1.5 pb-1.5 text-xs text-ink-secondary">
          <input
            type="checkbox"
            checked={persist}
            onChange={(event) => setPersist(event.target.checked)}
            className="rounded border-line"
          />
          Save the run
        </label>
        <button
          type="button"
          className="btn-primary"
          onClick={() => run.mutate()}
          disabled={run.isPending}
        >
          <Play className="h-3.5 w-3.5" aria-hidden="true" />
          {run.isPending ? "Running…" : "Run evaluation"}
        </button>
      </div>

      {run.isError ? <ErrorPanel error={run.error} subject="the evaluation run" onRetry={() => run.reset()} /> : null}

      {result !== null ? (
        <RunMetricsPanel metrics={result.metrics} k={result.k} ks={result.ks} />
      ) : (
        <InfoNote>
          The five metrics are computed over each query's retrieved document ids against its labels.
          Higher is better for all five; a K that retrieves nothing scores zero, and a dataset with no
          relevant documents for a query is reported as a miss rather than excluded.
        </InfoNote>
      )}
    </div>
  );
}

function RunMetricsPanel({
  metrics,
  k,
  ks,
}: {
  metrics: Record<string, unknown> | null | undefined;
  k: number;
  ks: number[];
}) {
  if (ks.length === 0) {
    return <InfoNote tone="warning">The run completed but recorded no retrieval metrics.</InfoNote>;
  }
  const primary = readRetrievalMetrics(metrics, ks[ks.length - 1]) ?? readRetrievalMetrics(metrics, k);
  if (primary === null) {
    return <InfoNote tone="warning">No metrics found for the K values this run reported.</InfoNote>;
  }

  return (
    <div className="space-y-4">
      <StatRow>
        <StatCard label={`Precision@${primary.k}`} value={formatScore(primary.precision_at_k, 3)} hint="Retrieved documents that were relevant, of those retrieved" />
        <StatCard label={`Recall@${primary.k}`} value={formatScore(primary.recall_at_k, 3)} hint="Relevant documents found, of those labelled" />
        <StatCard label="MRR" value={formatScore(primary.mrr, 3)} hint="Mean reciprocal rank of the first relevant hit" />
        <StatCard label={`NDCG@${primary.k}`} value={formatScore(primary.ndcg_at_k, 3)} hint="Rank-discounted gain, honouring graded relevance" />
        <StatCard label={`Hit rate@${primary.k}`} value={formatScore(primary.hit_rate_at_k, 3)} hint="Queries with at least one relevant document in top-K" />
        <StatCard
          label="Zero-result queries"
          value={formatScore(primary.zero_result_rate, 3)}
          hint="Fraction of queries that retrieved nothing at all"
          tone={primary.zero_result_rate > 0 ? "bad" : "good"}
        />
      </StatRow>

      <ChartCard
        title="Metrics by K"
        description="Each K is scored independently against the same labels, so a rising Recall with a falling Precision is a real trade-off and not a rounding artefact."
        valueLabel="Score, 0–1"
        legend={METRIC_SERIES}
        table={{
          columns: ["K", "Precision", "Recall", "MRR", "NDCG", "Hit rate"],
          rows: ks.map((value) => {
            const m = readRetrievalMetrics(metrics, value);
            return [
              `k=${value}`,
              formatScore(m?.precision_at_k, 3),
              formatScore(m?.recall_at_k, 3),
              formatScore(m?.mrr, 3),
              formatScore(m?.ndcg_at_k, 3),
              formatScore(m?.hit_rate_at_k, 3),
            ];
          }),
        }}
      >
        <CategoryBarChart
          data={ks.flatMap((value) => {
            const m = readRetrievalMetrics(metrics, value);
            if (m === null) return [];
            return METRIC_SERIES.map((series) => ({
              label: `${series.label} (k=${value})`,
              value: (m as unknown as Record<string, number>)[series.key] ?? null,
            }));
          })}
          valueLabel="Score"
          maxBars={20}
          format={(value) => formatScore(value, 3)}
        />
      </ChartCard>

      <RunDetailTable metrics={metrics} ks={ks} />
    </div>
  );
}

function RunDetail({ runId }: { runId: string }) {
  const detail = useQuery({
    queryKey: ["eval-run", runId],
    queryFn: () => api.getEvaluationRun(runId),
  });

  return (
    <div className="card p-4">
      <QueryBoundary query={detail} label="run detail">
        {(run) => <RunBody run={run} />}
      </QueryBoundary>
    </div>
  );
}

function RunBody({ run }: { run: EvaluationRunDetail }) {
  const ks = runKValues(run);
  return (
    <div className="space-y-4">
      <PanelTitle>
        {run.name} <span className="font-normal text-ink-secondary">· {humanize(run.evaluation_type)}</span>
      </PanelTitle>
      <KeyValueList
        columns={3}
        items={[
          { label: "Status", value: <Badge tone={statusTone(run.status)}>{humanize(run.status)}</Badge> },
          { label: "Dataset", value: run.dataset_name },
          { label: "K", value: String(run.k) },
          { label: "Queries", value: run.num_queries === 0 ? "—" : formatNumber(run.num_queries) },
          { label: "Started", value: formatDateTime(run.started_at) },
          { label: "Completed", value: formatDateTime(run.completed_at) },
          { label: "Duration", value: formatMs(run.duration_ms) },
          { label: "Method", value: ks.length > 0 ? "deterministic set/rank metrics" : humanize(run.evaluation_type) },
        ]}
      />
      {run.error === null || run.error === undefined ? null : (
        <InfoNote tone="warning">{run.error}</InfoNote>
      )}
      {ks.length > 0 ? <RunMetricsPanel metrics={run.metrics} k={run.k} ks={ks} /> : null}
      {run.results.length > 0 ? <PerQueryTable results={run.results} /> : null}
    </div>
  );
}

function RunDetailTable({ metrics, ks }: { metrics: Record<string, unknown> | null | undefined; ks: number[] }) {
  return (
    <ChartCard
      title="Per-K metric table"
      description="The raw numbers behind the chart above, in full."
      valueLabel="Score, 0–1"
      table={{
        columns: ["K", "Precision", "Recall", "MRR", "NDCG", "Hit rate", "Zero-result", "Queries"],
        rows: ks.map((value) => {
          const m: RetrievalMetrics | null = readRetrievalMetrics(metrics, value);
          return [
            `k=${value}`,
            formatScore(m?.precision_at_k, 3),
            formatScore(m?.recall_at_k, 3),
            formatScore(m?.mrr, 3),
            formatScore(m?.ndcg_at_k, 3),
            formatScore(m?.hit_rate_at_k, 3),
            formatScore(m?.zero_result_rate, 3),
            m === null ? "—" : formatNumber(m.num_queries),
          ];
        }),
      }}
    >
      <div className="text-sm text-ink-secondary">Switch to the table view for the exact values.</div>
    </ChartCard>
  );
}

function PerQueryTable({ results }: { results: PerQueryResult[] }) {
  const columns: Column<PerQueryResult>[] = [
    {
      key: "query",
      header: "Query",
      render: (row) => <span className="text-ink">{row.query}</span>,
    },
    { key: "k", header: "K", align: "right", render: (row) => String(row.k) },
    { key: "precision", header: "Precision", align: "right", render: (row) => formatScore(row.precision, 3) },
    { key: "recall", header: "Recall", align: "right", render: (row) => formatScore(row.recall, 3) },
    { key: "rr", header: "Reciprocal rank", align: "right", render: (row) => formatScore(row.reciprocal_rank, 3) },
    { key: "ndcg", header: "NDCG", align: "right", render: (row) => formatScore(row.ndcg, 3) },
    {
      key: "hit",
      header: "Hit",
      render: (row) => (
        <Badge tone={row.hit ? "good" : "critical"}>{row.hit ? "hit" : "miss"}</Badge>
      ),
    },
    { key: "latency", header: "Latency", align: "right", render: (row) => formatMs(row.latency_ms) },
    {
      key: "missed",
      header: "Missed",
      render: (row) =>
        row.missed_document_ids.length === 0 ? (
          <span className="no-value">none</span>
        ) : (
          <span className="text-2xs text-ink-muted">{row.missed_document_ids.join(", ")}</span>
        ),
    },
  ];

  return (
    <DetailSection title="Per-query results" count={results.length}>
      <DataTable
        caption="Per-query retrieval metrics"
        columns={columns}
        rows={results}
        rowKey={(row) => `${row.k}:${row.query}`}
        minWidth="72rem"
      />
    </DetailSection>
  );
}

/* -------------------------------------------------------------------------- */
/* Comparison                                                                 */
/* -------------------------------------------------------------------------- */

function CompareView() {
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [tolerance, setTolerance] = useState(5);
  const [report, setReport] = useState<RegressionReport | null>(null);

  const runs = useQuery({
    queryKey: ["eval-runs", "all", 1, 100],
    queryFn: () => api.listEvaluationRuns({ page: 1, page_size: 100 }),
  });

  const compare = useMutation({
    mutationFn: () =>
      api.compareRuns({
        baseline_run_id: baselineId,
        candidate_run_id: candidateId,
        max_regression_pct: tolerance,
      }),
    onSuccess: (value) => setReport(value),
  });

  const options = (runs.data?.items ?? [])
    .filter((run) => run.status === "completed" && run.id !== null)
    .map((run) => ({ value: run.id as string, label: `${run.name} (k=${run.k}, ${run.dataset_name})` }));

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <PanelTitle>Compare two completed runs</PanelTitle>
        <p className="mb-3 max-w-prose text-2xs leading-relaxed text-ink-secondary">
          Reports the change in each metric and the configuration that differed between the two runs.
          It does not claim one caused the other: a config difference alongside a metric change is a
          correlation, and a single baseline–candidate pair cannot separate the two.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <SelectField
            id="compare-baseline"
            label="Baseline"
            value={baselineId}
            options={[{ value: "", label: "Select a run" }, ...options]}
            onChange={setBaselineId}
            className="min-w-52"
          />
          <SelectField
            id="compare-candidate"
            label="Candidate"
            value={candidateId}
            options={[{ value: "", label: "Select a run" }, ...options]}
            onChange={setCandidateId}
            className="min-w-52"
          />
          <TextField
            id="compare-tolerance"
            label="Regression threshold (%)"
            value={String(tolerance)}
            onChange={(value) => {
              // `Number(value) || 0` would also swallow a negative number typed
              // as "-0" into a 0, and would turn a stray "abc" into a silent 0
              // that the next regression run would then compare against.
              const parsed = Number(value);
              setTolerance(Number.isFinite(parsed) ? parsed : 0);
            }}
            type="number"
            min={0}
            className="w-32"
          />
          <button
            type="button"
            className="btn-primary"
            disabled={baselineId === "" || candidateId === "" || baselineId === candidateId || compare.isPending}
            onClick={() => compare.mutate()}
          >
            <Scale className="h-3.5 w-3.5" aria-hidden="true" />
            {compare.isPending ? "Comparing…" : "Compare"}
          </button>
        </div>
        {options.length === 0 ? (
          <InfoNote tone="warning">
            No completed runs to compare. Run an evaluation first.
          </InfoNote>
        ) : null}
        {compare.isError ? (
          <div className="mt-3">
            <ErrorPanel error={compare.error} subject="the comparison" />
          </div>
        ) : null}
      </div>

      {report !== null ? <ReportBody report={report} /> : null}
    </div>
  );
}

function ReportBody({ report }: { report: RegressionReport }) {
  return (
    <div className="card p-4">
      <PanelTitle
        actions={
          <Badge tone={report.has_regression ? "critical" : "good"}>
            {report.has_regression ? "regression detected" : "no regression"}
          </Badge>
        }
      >
        {report.baseline_name} → {report.candidate_name}
      </PanelTitle>

      <InfoNote tone={report.has_regression ? "warning" : "neutral"} className="mb-4">
        {report.regression_summary ??
          "No metric moved by more than the configured threshold."}
      </InfoNote>

      <DetailSection title="Metric changes" count={report.deltas.length} className="mb-5">
        {report.deltas.length === 0 ? (
          <p className="text-sm text-ink-muted">The two runs reported no comparable metrics.</p>
        ) : (
          <DataTable<MetricDelta>
            caption="Metric deltas between baseline and candidate"
            columns={[
              { key: "metric", header: "Metric", render: (row) => humanize(row.metric) },
              { key: "direction", header: "Better when", render: (row) => humanize(row.direction) },
              { key: "baseline", header: "Baseline", align: "right", render: (row) => formatScore(row.baseline, 4) },
              { key: "candidate", header: "Candidate", align: "right", render: (row) => formatScore(row.candidate, 4) },
              {
                key: "relative",
                header: "Relative change",
                align: "right",
                render: (row) => {
                  // The backend already decided `is_regression` using the
                  // metric's own direction, so the verdict is not re-derived
                  // here from a guess about which way is good.
                  const tone: MetricTone = row.is_regression ? "bad" : changeTone(row.absolute_change, metricDirection(row.metric));
                  return <span className={TONE_CLASSES[tone]}>{formatSignedPercent(row.relative_change_pct)}</span>;
                },
              },
              {
                key: "verdict",
                header: "Verdict",
                render: (row) => (
                  <Badge tone={row.is_regression ? "critical" : "good"}>
                    {row.is_regression ? "regressed" : "within threshold"}
                  </Badge>
                ),
              },
            ]}
            rows={report.deltas}
            rowKey={(row) => row.metric}
            minWidth="56rem"
          />
        )}
      </DetailSection>

      <DetailSection
        title="Configuration differences"
        count={report.config_differences.length}
      >
        {report.config_differences.length === 0 ? (
          <p className="text-sm text-ink-muted">
            The two runs recorded identical configuration, so nothing changed between them but the
            measured behaviour.
          </p>
        ) : (
          <>
            <DataTable
              caption="Configuration differences between the two runs"
              columns={[
                { key: "key", header: "Setting", render: (row) => humanize(row.key) },
                { key: "baseline", header: "Baseline", render: (row) => String(row.baseline_value) },
                { key: "candidate", header: "Candidate", render: (row) => String(row.candidate_value) },
              ]}
              rows={report.config_differences}
              rowKey={(row) => row.key}
              minWidth="36rem"
            />
            <InfoNote className="mt-3">
              These settings differed between the runs. That is an observed fact, not an established
              cause: with one baseline and one candidate there is no way to rule out that the change in
              metric came from the data rather than the setting. Change one setting at a time to
              attribute an effect.
            </InfoNote>
          </>
        )}
      </DetailSection>
    </div>
  );
}
