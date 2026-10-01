import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Banknote,
  Boxes,
  Clock,
  Coins,
  Gauge,
  type LucideIcon,
} from "lucide-react";
import { api, type DashboardOverview } from "@/lib/api";
import { useScope } from "@/hooks/useScope";
import { InfoNote, PageHeader } from "@/components/PageHeader";
import { ScopeBar, QueryBoundary } from "@/components/ScopeBar";
import { StatCard, StatRow } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { CategoryBarChart, TimeSeriesChart, type SeriesSpec } from "@/lib/charts";
import { EmptyState } from "@/components/EmptyState";
import { DataTable, type Column } from "@/components/Table";
import {
  formatCompact,
  formatCost,
  formatNumber,
  formatPercent,
  formatLatencyMs,
  formatScore,
} from "@/lib/format";

/**
 * §6: the landing page. Six headline figures and seven chart/table cards over a
 * single `GET /api/dashboard/overview` call.
 *
 * The "tokens per request" figure is worth defending, because the obvious
 * implementations are both wrong in different directions. It is not the mean of
 * a per-trace total — that lets a handful of very large requests hide behind
 * many small ones. And it is not the mean over time buckets, which weights a
 * quiet hour the same as a busy one. It is `total_tokens / total_requests` over
 * the same rows, so every request that used tokens contributes its own usage
 * and the ratio cannot be moved by how the data happens to be bucketed.
 */

const REQUESTS_SERIES: SeriesSpec[] = [
  { key: "request_count", label: "Requests", slot: 1 },
  { key: "error_count", label: "Errors", slot: 2 },
];
const TOKEN_SERIES: SeriesSpec[] = [
  { key: "input_tokens", label: "Input tokens", slot: 1 },
  { key: "output_tokens", label: "Output tokens", slot: 2 },
];
const COST_SERIES: SeriesSpec[] = [
  { key: "estimated_cost", label: "Estimated cost", slot: 1, format: formatCost },
];
const LATENCY_SERIES: SeriesSpec[] = [
  { key: "avg_latency_ms", label: "Average latency", slot: 1, format: (value) => formatLatencyMs(value) },
];
const QUALITY_SERIES: SeriesSpec[] = [
  { key: "avg_retrieval_score", label: "Retrieval score", slot: 1, format: formatScore },
  { key: "avg_faithfulness", label: "Faithfulness", slot: 3, format: formatScore },
  { key: "avg_answer_relevance", label: "Answer relevance", slot: 5, format: formatScore },
];

const ms = (value: number): string => `${formatNumber(value)} ms`;

export default function DashboardPage() {
  const { params } = useScope();
  const overview = useQuery({
    queryKey: ["dashboard", params],
    queryFn: () => api.dashboardOverview(params),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dashboard"
        description="Every figure here is counted from stored request records. Nothing is estimated, sampled, or hardcoded."
        scope={<ScopeBar />}
      />

      <QueryBoundary query={overview} label="dashboard">
        {(data) => (
          <>
            <HeadlineCards data={data} />

            {data.total_traces === 0 ? (
              <EmptyState
                title="No requests recorded in this window"
                description="Widen the time window, or seed the database with the demo data generator. RAGOps shows nothing rather than a placeholder when there is nothing to show."
              />
            ) : null}

            <div className="grid gap-4 xl:grid-cols-2">
              <ChartCard
                title="Requests and errors over time"
                description="Error count is plotted against the same axis as request volume, so a spike is read against the traffic it happened in."
                valueLabel="Count per bucket"
                timeRange={range(data)}
                legend={REQUESTS_SERIES}
                table={{
                  columns: ["Bucket", "Requests", "Errors"],
                  rows: data.time_series.map((point) => [
                    point.bucket,
                    formatNumber(point.request_count ?? 0),
                    formatNumber(point.error_count ?? 0),
                  ]),
                }}
                footnote={`${formatNumber(data.total_calls)} requests, ${formatNumber(data.error_count)} errors (${formatPercent(data.error_rate)}).`}
              >
                <TimeSeriesChart data={data.time_series} series={REQUESTS_SERIES} area />
              </ChartCard>

              <ChartCard
                title="Token usage over time"
                description="Input and output are tracked separately because they are priced differently and wasted differently."
                valueLabel="Tokens per bucket"
                timeRange={range(data)}
                legend={TOKEN_SERIES}
                table={{
                  columns: ["Bucket", "Input", "Output", "Total"],
                  rows: data.time_series.map((point) => [
                    point.bucket,
                    formatNumber(point.input_tokens ?? 0),
                    formatNumber(point.output_tokens ?? 0),
                    formatNumber(point.total_tokens ?? 0),
                  ]),
                }}
                footnote={`${formatNumber(data.total_input_tokens)} in · ${formatNumber(data.total_output_tokens)} out.`}
              >
                <TimeSeriesChart data={data.time_series} series={TOKEN_SERIES} area />
              </ChartCard>

              <ChartCard
                title="Estimated cost over time"
                description="Simulated pricing. Local models are priced at zero, which is why a heavy local run can read as no cost at all."
                valueLabel="Simulated cost per bucket"
                timeRange={range(data)}
                table={{
                  columns: ["Bucket", "Estimated cost"],
                  rows: data.time_series.map((point) => [point.bucket, formatCost(point.estimated_cost)]),
                }}
                footnote={data.cost_label}
              >
                <TimeSeriesChart
                  data={data.time_series}
                  series={COST_SERIES}
                  area
                  yFormat={(value) => formatCost(value)}
                />
              </ChartCard>

              <ChartCard
                title="Average latency over time"
                description="Mean end-to-end duration per bucket. This is an average, not a percentile — the distribution is on Token Analytics."
                valueLabel="Mean latency per bucket"
                timeRange={range(data)}
                table={{
                  columns: ["Bucket", "Average latency"],
                  rows: data.time_series.map((point) => [point.bucket, formatLatencyMs(point.avg_latency_ms)]),
                }}
              >
                <TimeSeriesChart
                  data={data.time_series}
                  series={LATENCY_SERIES}
                  yFormat={ms}
                />
              </ChartCard>

              <ChartCard
                title="Quality metrics over time"
                description="Averaged across evaluations that actually ran in the window. A bucket with no evaluation plots as a gap, never as a zero."
                valueLabel="Score, 0–1"
                timeRange={range(data)}
                legend={QUALITY_SERIES}
                table={{
                  columns: ["Bucket", "Retrieval", "Faithfulness", "Answer relevance"],
                  rows: data.time_series.map((point) => [
                    point.bucket,
                    formatScore(point.avg_retrieval_score),
                    formatScore(point.avg_faithfulness),
                    formatScore(point.avg_answer_relevance),
                  ]),
                }}
                footnote="Gaps are missing measurements, not zeroes."
              >
                <TimeSeriesChart
                  data={data.time_series}
                  series={QUALITY_SERIES}
                  yFormat={(value) => value.toFixed(1)}
                />
              </ChartCard>

              <ChartCard
                title="Tokens by model"
                description="Where the token volume actually goes, so a fix can be aimed at the model rather than at the application."
                valueLabel="Tokens per model"
                table={{
                  columns: ["Model", "Calls", "Tokens", "Cost"],
                  rows: data.model_usage.map((row) => [
                    row.model_name,
                    formatNumber(row.call_count),
                    formatNumber(row.total_tokens),
                    formatCost(row.estimated_cost),
                  ]),
                }}
              >
                <CategoryBarChart
                  data={data.model_usage.map((row) => ({ label: row.model_name, value: row.total_tokens }))}
                  valueLabel="Tokens"
                />
              </ChartCard>

              <ChartCard
                title="Cost by application"
                description="Estimated spend attributed to each application over the window."
                valueLabel={data.cost_label}
                table={{
                  columns: ["Application", "Traces", "Tokens", "Cost"],
                  rows: data.application_usage.map((row) => [
                    row.application,
                    formatNumber(row.trace_count),
                    formatNumber(row.total_tokens),
                    formatCost(row.estimated_cost),
                  ]),
                }}
              >
                <CategoryBarChart
                  data={data.application_usage.map((row) => ({ label: row.application, value: row.estimated_cost }))}
                  valueLabel="Estimated cost"
                  format={formatCost}
                />
              </ChartCard>

              <ModelUsageTable data={data} />
            </div>
          </>
        )}
      </QueryBoundary>
    </div>
  );
}

interface HeadlineCard {
  label: string;
  value: string;
  hint: string;
  icon: LucideIcon;
  tone: "good" | "bad" | "flat" | "none";
}

function HeadlineCards({ data }: { data: DashboardOverview }) {
  const cards: HeadlineCard[] = [
    {
      label: "Total requests",
      value: formatCompact(data.total_calls),
      hint: `${formatNumber(data.total_traces)} traces · ${formatNumber(data.unique_users)} users`,
      icon: Activity,
      tone: "none",
    },
    {
      label: "Error rate",
      value: formatPercent(data.error_rate),
      hint: `${formatNumber(data.error_count)} failed of ${formatNumber(data.total_calls)}`,
      icon: AlertTriangle,
      // Zero is a good, measured result; above 5% is worth flagging. Between
      // the two it is reported flat rather than dressed up as either.
      tone: data.error_rate === 0 ? "good" : data.error_rate > 0.05 ? "bad" : "flat",
    },
    {
      label: "Total tokens",
      value: formatCompact(data.total_tokens),
      hint: `${formatCompact(data.total_input_tokens)} in · ${formatCompact(data.total_output_tokens)} out`,
      icon: Coins,
      tone: "none",
    },
    {
      label: "Tokens per request",
      value: formatNumber(data.avg_tokens_per_request),
      hint: "Total tokens ÷ requests",
      icon: Boxes,
      tone: "none",
    },
    {
      label: "Avg latency",
      value: formatLatencyMs(data.avg_latency_ms),
      hint: "Mean end-to-end duration",
      icon: Clock,
      tone: "none",
    },
    {
      label: "Estimated cost",
      value: formatCost(data.estimated_cost),
      hint: data.cost_label,
      icon: Banknote,
      tone: "none",
    },
  ];

  return (
    <StatRow>
      {cards.map((card) => (
        <StatCard
          key={card.label}
          label={card.label}
          value={card.value}
          hint={card.hint}
          icon={<card.icon className="h-4 w-4" aria-hidden="true" />}
          tone={card.tone}
        />
      ))}
    </StatRow>
  );
}

/** The per-model detail the two charts above summarise. */
function ModelUsageTable({ data }: { data: DashboardOverview }) {
  type Row = DashboardOverview["model_usage"][number];

  const columns: Column<Row>[] = [
    {
      key: "model",
      header: "Model",
      render: (row) => (
        <span className="font-medium text-ink">
          {row.model_name} <span className="font-normal text-ink-muted">· {row.provider}</span>
        </span>
      ),
    },
    { key: "calls", header: "Calls", align: "right", render: (row) => formatNumber(row.call_count) },
    { key: "tokens", header: "Tokens", align: "right", render: (row) => formatNumber(row.total_tokens) },
    { key: "cost", header: "Cost", align: "right", render: (row) => formatCost(row.estimated_cost) },
  ];

  return (
    <div className="card p-4">
      <h3 className="mb-1 text-sm font-semibold text-ink">Model usage detail</h3>
      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Per-model call counts and token totals for the selected window.
      </p>
      <DataTable
        caption="Model usage over the selected window"
        columns={columns}
        rows={data.model_usage}
        rowKey={(row) => row.model_name}
        minWidth="32rem"
        empty={<EmptyState title="No model calls" description="No LLM calls were recorded in this window." />}
      />
      <div className="mt-3 space-y-2">
        <InfoNote>{data.cost_label}</InfoNote>
        <p className="flex items-center gap-1.5 text-2xs text-ink-muted">
          <Gauge className="h-3 w-3 shrink-0" aria-hidden="true" />
          Quality figures come from evaluation runs, which are far fewer than the request count — the two
          are not directly comparable.
        </p>
      </div>
    </div>
  );
}

function range(data: DashboardOverview): { start: string; end: string } {
  return { start: data.start, end: data.end };
}
