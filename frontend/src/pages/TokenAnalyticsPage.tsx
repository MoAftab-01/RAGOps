import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, Coins, Gauge, Repeat, Search, TrendingUp } from "lucide-react";
import {
  api,
  type BreakdownItem,
  type TokenAnalytics,
  type TokenEfficiencyReport,
} from "@/lib/api";
import { useScope } from "@/hooks/useScope";
import { InfoNote, PageHeader } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { StatCard, StatRow } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { CategoryBarChart, TimeSeriesChart, type SeriesSpec } from "@/lib/charts";
import { DataTable, type Column } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import {
  formatCost,
  formatLatencyMs,
  formatNumber,
  formatPercent,
  formatScore,
  formatSigned,
  truncate,
} from "@/lib/format";

/**
 * §7 and §8: token analytics, and token waste detection.
 *
 * The two live on one page because waste is only interpretable next to volume —
 * 40% waste of 200 tokens is a rounding error next to 40% of four million.
 *
 * The waste section reports only waste the backend *measured*. Every figure here
 * is derived from recorded fields: context tokens that carried no document
 * beyond the first, duplicate documents in the top-k, and the share of input
 * tokens that were context rather than the user's own text. Nothing is inferred
 * from how an answer *read*, because that would be a guess wearing a number.
 */

const TOKEN_SERIES: SeriesSpec[] = [
  { key: "input_tokens", label: "Input tokens", slot: 1 },
  { key: "output_tokens", label: "Output tokens", slot: 2 },
];
const LATENCY_SERIES: SeriesSpec[] = [
  { key: "p50_latency_ms", label: "p50", slot: 1, format: (value) => formatLatencyMs(value) },
  { key: "p95_latency_ms", label: "p95", slot: 3, format: (value) => formatLatencyMs(value) },
  { key: "p99_latency_ms", label: "p99", slot: 5, format: (value) => formatLatencyMs(value) },
];

export default function TokenAnalyticsPage() {
  const { params } = useScope();
  const tokens = useQuery({
    queryKey: ["analytics-tokens", params],
    queryFn: () => api.tokenAnalytics(params),
  });
  const efficiency = useQuery({
    queryKey: ["analytics-token-efficiency", params],
    queryFn: () => api.tokenEfficiency(params),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Token Analytics"
        description="Volume, percentiles, and measured waste. Token counts come from the recorded usage fields, and the waste figures from the same rows — nothing here is sampled or modelled."
        scope={<ScopeBar />}
      />

      <QueryBoundary query={tokens} label="token analytics">
        {(data) => <TokenVolume data={data} />}
      </QueryBoundary>

      <QueryBoundary query={efficiency} label="token efficiency">
        {(report) => <TokenWaste report={report} />}
      </QueryBoundary>
    </div>
  );
}

function TokenVolume({ data }: { data: TokenAnalytics }) {
  return (
    <>
      <StatRow>
        <StatCard
          label="Total tokens"
          value={formatNumber(data.total_tokens)}
          hint={`${formatNumber(data.input_tokens)} in · ${formatNumber(data.output_tokens)} out`}
          icon={<Coins className="h-4 w-4" aria-hidden="true" />}
        />
        <StatCard
          label="Tokens per request"
          value={formatNumber(data.avg_tokens_per_request)}
          hint="Total tokens ÷ requests"
          icon={<Gauge className="h-4 w-4" aria-hidden="true" />}
        />
        <StatCard
          label="p50 tokens"
          value={formatNumber(data.p50_tokens)}
          hint="Median request"
        />
        <StatCard label="p95 tokens" value={formatNumber(data.p95_tokens)} hint="Tail begins" />
        <StatCard label="p99 tokens" value={formatNumber(data.p99_tokens)} hint="Worst 1%" />
        <StatCard
          label="Max tokens"
          value={formatNumber(data.max_tokens)}
          hint="Largest single request"
        />
      </StatRow>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard
          title="Token usage over time"
          description="Input and output tracked separately: they are priced differently and wasted differently."
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
        >
          <TimeSeriesChart data={data.time_series} series={TOKEN_SERIES} area />
        </ChartCard>

        <ChartCard
          title="Token percentiles over time"
          description="p99 is the one that matters for a latency budget — the mean hides it completely."
          valueLabel="Milliseconds per bucket"
          timeRange={range(data)}
          legend={LATENCY_SERIES}
          table={{
            columns: ["Bucket", "p50", "p95", "p99"],
            rows: data.time_series.map((point) => [
              point.bucket,
              formatLatencyMs(point.p50_latency_ms),
              formatLatencyMs(point.p95_latency_ms),
              formatLatencyMs(point.p99_latency_ms),
            ]),
          }}
        >
          <TimeSeriesChart data={data.time_series} series={LATENCY_SERIES} />
        </ChartCard>

        <BreakdownCard
          title="Tokens by model"
          description="Where the volume goes, so a fix can be aimed at the model rather than the application."
          valueLabel="Tokens per model"
          items={data.tokens_per_model}
        />

        <BreakdownCard
          title="Tokens by application"
          description="The same volume attributed to each application in the selected scope."
          valueLabel="Tokens per application"
          items={data.tokens_per_application}
        />

        <BreakdownCard
          title="Tokens by user"
          description="Top users by volume. The tail is folded into a visible “Other” bar rather than dropped."
          valueLabel="Tokens per user"
          items={data.tokens_per_user}
          maxBars={8}
        />
      </div>
    </>
  );
}

function TokenWaste({ report }: { report: TokenEfficiencyReport }) {
  const navigate = useNavigate();
  const offenders = report.top_offenders;

  return (
    <section className="space-y-4" aria-label="Token waste">
      <StatRow>
        <StatCard
          label="Efficiency score"
          value={formatScore(report.score)}
          hint="100 = no measured waste"
          icon={<Gauge className="h-4 w-4" aria-hidden="true" />}
          // A perfect score is a measured result worth showing as one; a low
          // one is flagged. In between, the number is reported flat rather
          // than dressed up as either good or bad.
          tone={report.score >= 90 ? "good" : report.score >= 70 ? "flat" : "bad"}
        />
        <StatCard
          label="Potential waste"
          value={formatPercent(report.potential_waste_pct)}
          hint="Wasted ÷ total input tokens"
          icon={<TrendingUp className="h-4 w-4" aria-hidden="true" />}
          tone={report.potential_waste_pct > 30 ? "bad" : report.potential_waste_pct > 10 ? "flat" : "good"}
        />
        <StatCard
          label="Wasted tokens / request"
          value={formatNumber(report.wasted_tokens, 1)}
          hint={`Mean over scored requests, of ${formatNumber(report.total_input_tokens)} input tokens in window`}
        />
        <StatCard
          label="Duplicates / request"
          value={formatNumber(report.duplicate_document_count, 1)}
          hint="Mean documents returned more than once in a top-k"
          icon={<Repeat className="h-4 w-4" aria-hidden="true" />}
        />
        <StatCard
          label="Context share"
          value={formatPercent(report.avg_context_share * 100)}
          hint="Context ÷ input tokens, averaged"
        />
        <StatCard
          label="Output yield"
          value={formatPercent(report.avg_output_yield * 100)}
          hint="Output ÷ input tokens, averaged. 25% is a healthy RAG summary"
        />
      </StatRow>

      <InfoNote>
        These are measured quantities, not estimates of what a better prompt might achieve. Waste is
        counted from recorded fields — context tokens carried without a distinct document, and
        documents returned more than once in the same top-k. A score of 100 means no waste of those
        two kinds was found; it does not mean the context was well chosen.
      </InfoNote>

      {report.findings.length > 0 ? (
        <ul className="space-y-1.5">
          {report.findings.map((finding) => (
            <li
              key={finding}
              className="flex items-start gap-2 rounded-md border border-line bg-surface px-3 py-2 text-2xs leading-relaxed text-ink-secondary"
            >
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-status-warning" aria-hidden="true" />
              {finding}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard
          title="Waste by application"
          description="Average measured waste share per application. Bars are coloured by severity, not by identity — the point is which application is worst, not which is which."
          valueLabel="Waste share of input tokens"
          table={{
            columns: ["Application", "Waste", "Traces", "Tokens"],
            rows: report.waste_by_application.map((row) => [
              row.application,
              formatPercent(row.avg_waste_pct),
              formatNumber(row.trace_count),
              formatNumber(row.total_tokens),
            ]),
          }}
        >
          <CategoryBarChart
            data={report.waste_by_application.map((row) => ({
              label: row.application,
              value: row.avg_waste_pct,
            }))}
            valueLabel="Average waste"
            format={(value) => formatPercent(value)}
            tone="warning"
          />
        </ChartCard>

        <div className="card p-4">
          <h3 className="mb-1 text-sm font-semibold text-ink">Worst individual requests</h3>
          <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
            Traces with the highest measured waste share. Select a row to open the trace it came
            from, so the number can be checked against the request that produced it.
          </p>
          <DataTable
            caption="Traces with the highest measured token waste"
            columns={offenderColumns}
            rows={offenders}
            rowKey={(row) => row.trace_id}
            onRowClick={(row) => {
              navigate(`/traces/${encodeURIComponent(row.trace_id)}`);
            }}
            minWidth="30rem"
            empty={
              <EmptyState
                title="No wasteful requests found"
                description="Nothing in this window exceeded the measured waste threshold."
                icon={<Search className="h-6 w-6" aria-hidden="true" />}
              />
            }
          />
        </div>
      </div>
    </section>
  );
}

const offenderColumns: Column<TokenEfficiencyReport["top_offenders"][number]>[] = [
  {
    key: "trace",
    header: "Trace",
    render: (row) => <span className="font-mono text-xs">{truncate(row.trace_id, 14)}</span>,
  },
  {
    key: "waste",
    header: "Waste",
    align: "right",
    render: (row) => formatPercent(row.waste_pct),
  },
  {
    key: "duplicates",
    header: "Duplicate docs",
    align: "right",
    render: (row) => formatNumber(row.duplicate_document_count),
  },
  {
    key: "input",
    header: "Input tokens",
    align: "right",
    render: (row) => formatNumber(row.input_tokens),
  },
  {
    key: "score",
    header: "Score",
    align: "right",
    render: (row) => <span title="Backend waste score for this trace">{formatSigned(row.score, 3)}</span>,
  },
];

/** A breakdown chart plus its full table, since the chart folds a visible tail. */
function BreakdownCard({
  title,
  description,
  valueLabel,
  items,
  maxBars,
}: {
  title: string;
  description: string;
  valueLabel: string;
  items: BreakdownItem[];
  maxBars?: number;
}) {
  return (
    <ChartCard
      title={title}
      description={description}
      valueLabel={valueLabel}
      table={{
        columns: ["Name", "Requests", "Input", "Output", "Total", "Cost"],
        rows: items.map((item) => [
          item.label ?? item.key,
          formatNumber(item.count),
          formatNumber(item.input_tokens),
          formatNumber(item.output_tokens),
          formatNumber(item.total_tokens),
          formatCost(item.estimated_cost),
        ]),
      }}
    >
      <CategoryBarChart
        data={items.map((item) => ({ label: item.label ?? item.key, value: item.total_tokens }))}
        valueLabel="Tokens"
        {...(maxBars === undefined ? {} : { maxBars })}
      />
    </ChartCard>
  );
}

function range(data: TokenAnalytics): { start: string; end: string } {
  return { start: data.start, end: data.end };
}
