import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Boxes, ChevronDown, ChevronRight } from "lucide-react";
import {
  api,
  type Application,
  type ApplicationStats,
} from "@/lib/api";
import { useScope, usePageParam } from "@/hooks/useScope";
import { useApplications } from "@/hooks/useApplications";
import { InfoNote, PageHeader, PanelTitle, ToggleField } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { Pagination } from "@/components/Table";
import { KeyValueList } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import { Badge } from "@/components/Badge";
import { StatCard, StatRow } from "@/components/StatCard";
import { ChartCard } from "@/components/ChartCard";
import { CategoryBarChart, TimeSeriesChart, type SeriesSpec } from "@/lib/charts";
import {
  formatCost,
  formatDateTime,
  formatLatencyMs,
  formatNumber,
  formatPercent,
  formatRelative,
} from "@/lib/format";

/**
 * §20: applications.
 *
 * The catalogue, and per-application figures for the currently scoped window.
 *
 * The distinction this page is careful about: the list rows carry
 * **lifetime** counts (`trace_count`, `total_tokens` — whatever has ever been
 * recorded), while the detail panel carries **window** figures from
 * `/applications/{id}/stats`. They are different questions and mixing them
 * would make a seven-day panel look like a lifetime total, or the reverse. The
 * two are labelled as window and lifetime wherever they appear, and the window
 * selector is shown *above* the detail so the reader knows which one they have.
 */

const ACTIVITY_SERIES: SeriesSpec[] = [
  { key: "request_count", label: "Requests", slot: 1 },
  { key: "error_count", label: "Errors", slot: 3 },
];

export default function ApplicationsPage() {
  const { setPage } = usePageParam();
  const [includeInactive, setIncludeInactive] = useState(false);

  // The same query key `useApplications` uses, so the scope selector and this
  // page share one cached catalogue rather than fetching it twice.
  const catalogue = useApplications({ includeInactive });
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Applications"
        description="Every application that has sent telemetry, and what each one did in the selected window. An application with no traces in the window is listed with an empty window panel rather than dropped, because 'never used' and 'not used recently' are different answers."
        scope={<ScopeBar />}
      />

      <div className="card p-4">
        <PanelTitle
          actions={
            <ToggleField
              id="applications-include-inactive"
              label="Include inactive"
              checked={includeInactive}
              onChange={setIncludeInactive}
              hint="Off: active applications only."
            />
          }
        >
          Catalogue
        </PanelTitle>

        <QueryBoundary query={catalogue} label="applications">
          {(data) => (
            <>
              {data.items.length === 0 ? (
                <EmptyState
                  icon={<Boxes className="h-6 w-6" aria-hidden="true" />}
                  title="No applications registered"
                  description="An application appears here once the SDK or the API has recorded a trace for it. Send one request, or run the demo data generator, and it will show up."
                />
              ) : (
                <ul className="space-y-2">
                  {data.items.map((application) => (
                    <li key={application.id}>
                      <ApplicationRow
                        application={application}
                        open={selected === application.id}
                        onSelect={() =>
                          setSelected((current) => (current === application.id ? null : application.id))
                        }
                      />
                    </li>
                  ))}
                </ul>
              )}

              <Pagination
                page={data.page}
                pageSize={data.page_size}
                total={data.total}
                onPageChange={setPage}
                itemLabel="applications"
              />
            </>
          )}
        </QueryBoundary>
      </div>
    </div>
  );
}

function ApplicationRow({
  application,
  open,
  onSelect,
}: {
  application: Application;
  open: boolean;
  onSelect: () => void;
}) {
  const panelId = `application-${application.id}`;

  return (
    <article className="rounded-md border border-line bg-surface">
      <h3>
        <button
          type="button"
          onClick={onSelect}
          aria-expanded={open}
          aria-controls={panelId}
          className="flex w-full items-start gap-3 px-3 py-2.5 text-left"
        >
          {open ? (
            <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
          ) : (
            <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
          )}
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-ink">{application.name}</span>
              <Badge tone={application.is_active ? "brand" : "neutral"}>
                {application.is_active ? "active" : "inactive"}
              </Badge>
              <Badge tone="neutral">{application.environment}</Badge>
            </span>
            {application.description !== null && application.description !== "" ? (
              <span className="mt-0.5 block text-2xs leading-relaxed text-ink-secondary">
                {application.description}
              </span>
            ) : null}
            <span className="mt-1 block text-2xs text-ink-muted">
              Lifetime: {formatNumber(application.trace_count)} traces ·{" "}
              {formatNumber(application.total_tokens)} tokens · last seen{" "}
              {application.last_seen_at === null || application.last_seen_at === undefined
                ? "never"
                : formatRelative(application.last_seen_at)}
            </span>
          </span>
        </button>
      </h3>

      {open ? <ApplicationPanel application={application} panelId={panelId} /> : null}
    </article>
  );
}

function ApplicationPanel({
  application,
  panelId,
}: {
  application: Application;
  panelId: string;
}) {
  const { params } = useScope();
  const stats = useQuery({
    queryKey: ["application-stats", application.id, params],
    queryFn: () => api.applicationStats(application.id, params),
  });

  return (
    <div id={panelId} className="space-y-3 border-t border-line bg-surface-sunken/40 px-3 py-3">
      <KeyValueList
        items={[
          { label: "Application id", value: <span className="font-mono text-2xs">{application.id}</span> },
          { label: "Environment", value: application.environment },
          { label: "Created", value: formatDateTime(application.created_at) },
        ]}
      />

      <QueryBoundary query={stats} label="application statistics">
        {(data) => <StatsBody data={data} />}
      </QueryBoundary>
    </div>
  );
}

function StatsBody({ data }: { data: ApplicationStats }) {
  const hasTraffic = data.trace_count > 0;

  return (
    <>
      <StatRow>
        <StatCard label="Traces in window" value={formatNumber(data.trace_count)} hint={data.window} />
        <StatCard
          label="Total tokens"
          value={formatNumber(data.total_tokens)}
          hint="Across every call in the window"
        />
        <StatCard
          label="Estimated cost"
          value={formatCost(data.estimated_cost, data.cost_label)}
          hint={data.cost_label}
        />
        <StatCard
          label="Avg latency"
          value={data.avg_latency_ms === null || data.avg_latency_ms === undefined
            ? null
            : formatLatencyMs(data.avg_latency_ms)}
          hint="Mean over the window"
        />
        <StatCard
          label="Error rate"
          value={formatPercent(data.error_rate * 100)}
          hint={hasTraffic ? "Failed calls ÷ total calls" : "No calls in this window"}
        />
        <StatCard label="Models used" value={formatNumber(data.model_usage.length)} hint="Distinct" />
      </StatRow>

      {hasTraffic ? null : (
        <InfoNote>
          No traces in this window, so there is nothing to plot below. The lifetime figures on the
          row above are still real — widen the time window in the filter above, or read{" "}
          <span className="font-medium text-ink-secondary">Token Analytics</span> scoped to this
          application, to see where its traffic went.
        </InfoNote>
      )}

      {data.time_series.length === 0 || !hasTraffic ? null : (
        <div className="grid gap-4 xl:grid-cols-2">
          <ChartCard
            title="Requests and errors over time"
            description="Errors are counted against requests in the same bucket, so the two lines are directly comparable."
            valueLabel="Requests per bucket"
            timeRange={{ start: data.start, end: data.end }}
            legend={ACTIVITY_SERIES}
            table={{
              columns: ["Bucket", "Requests", "Errors"],
              rows: data.time_series.map((point) => [
                point.bucket,
                formatNumber(point.request_count ?? 0),
                formatNumber(point.error_count ?? 0),
              ]),
            }}
          >
            <TimeSeriesChart data={data.time_series} series={ACTIVITY_SERIES} />
          </ChartCard>

          <ChartCard
            title="Tokens by model"
            description="Which models this application routed through, by total tokens."
            valueLabel="Tokens per model"
            table={{
              columns: ["Model", "Provider", "Calls", "Tokens", "Cost"],
              rows: data.model_usage.map((row) => [
                row.model_name,
                row.provider,
                formatNumber(row.call_count),
                formatNumber(row.total_tokens),
                formatCost(row.estimated_cost, row.cost_label),
              ]),
            }}
          >
            <CategoryBarChart
              data={data.model_usage.map((row) => ({
                label: row.model_name,
                value: row.total_tokens,
              }))}
              valueLabel="Tokens"
              format={(value) => formatNumber(value)}
            />
          </ChartCard>
        </div>
      )}

      {data.model_usage.length === 0 ? null : (
        <details className="rounded-md border border-line bg-surface px-3 py-2">
          <summary className="cursor-pointer text-2xs font-semibold uppercase tracking-wide text-ink-secondary">
            Per-model detail ({data.model_usage.length})
          </summary>
          <ul className="mt-2 space-y-1.5">
            {data.model_usage.map((row) => (
              <li key={row.model_name} className="flex flex-wrap items-baseline justify-between gap-2 text-2xs">
                <span className="text-ink">
                  {row.model_name} <span className="text-ink-muted">· {row.provider}</span>
                </span>
                <span className="tabular-nums text-ink-secondary">
                  {formatNumber(row.call_count)} calls · {formatNumber(row.total_tokens)} tokens ·{" "}
                  {formatCost(row.estimated_cost, row.cost_label)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}
