import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Lightbulb, Play, Search } from "lucide-react";
import {
  api,
  type Recommendation,
  type RecommendationGenerateRequest,
  type RecommendationGenerateResponse,
} from "@/lib/api";
import { useScope, usePageParam, useFilterParam } from "@/hooks/useScope";
import { InfoNote, PageHeader, PanelTitle, SelectField } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { DataTable, Pagination, type Column } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import { Badge, SeverityBadge } from "@/components/Badge";
import { ErrorPanel } from "@/components/ErrorPanel";
import { cn } from "@/lib/cn";
import {
  formatJsonValue,
  formatNumber,
  formatPercent,
  formatRelative,
  humanize,
} from "@/lib/format";

/**
 * §19: the optimization engine.
 *
 * Every rule that produces a recommendation measures something first and states
 * the measurement in its evidence. The page's job is to keep that visible, so:
 *
 * - The **rationale** and the **evidence** are always shown, never behind a
 *   hover or a toggle. A recommendation that only says "reduce latency" is a
 *   guess; one that says "p95 is 4.2 s against a 1.0 s budget" is a finding.
 * - **No saving is ever quoted.** The engine does not estimate what a change
 *   would have saved, so neither does this page — there is no projected-savings
 *   column to fill in with a made-up number.
 * - The generate response reports **every rule that ran**, including the ones
 *   that stayed quiet. A run that produced two findings out of nine rules
 *   evaluated is a much stronger statement than "two findings", and the quiet
 *   rules are what make it one.
 */

const CATEGORY_OPTIONS = [
  { value: "", label: "All categories" },
  { value: "latency", label: "Latency" },
  { value: "model_selection", label: "Model selection" },
  { value: "reliability", label: "Reliability" },
  { value: "retrieval", label: "Retrieval" },
  { value: "token_cost", label: "Token cost" },
  { value: "token_waste", label: "Token waste" },
];

const PRIORITY_OPTIONS = [
  { value: "", label: "Any priority" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

export default function OptimizationPage() {
  const { params } = useScope();
  const { page, pageSize, setPage } = usePageParam();
  const category = useFilterParam("category");
  const priority = useFilterParam("priority");
  const [expanded, setExpanded] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["recommendations", params, page, pageSize, category.value, priority.value],
    queryFn: () =>
      api.listRecommendations({
        ...params,
        page,
        page_size: pageSize,
        ...(category.value === "" ? {} : { category: category.value }),
        ...(priority.value === "" ? {} : { priority: priority.value }),
      }),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Optimization"
        description="Rule-based findings, each one attached to the measurement that produced it. The engine does not forecast what a change would have saved, and neither does this page — a projected saving nobody measured is a number with no evidence behind it."
        scope={<ScopeBar />}
      />

      <GeneratePanel application={params.application ?? null} timeWindow={params.window ?? "7d"} />

      <QueryBoundary query={list} label="recommendations">
        {(data) => (
          <div className="card p-4">
            <PanelTitle
              actions={
                <div className="flex flex-wrap items-end gap-3">
                  <SelectField
                    id="recommendation-category"
                    label="Category"
                    className="w-44"
                    value={category.value}
                    options={CATEGORY_OPTIONS}
                    onChange={category.setValue}
                  />
                  <SelectField
                    id="recommendation-priority"
                    label="Priority"
                    className="w-36"
                    value={priority.value}
                    options={PRIORITY_OPTIONS}
                    onChange={priority.setValue}
                  />
                </div>
              }
            >
              Findings
            </PanelTitle>

            {data.items.length === 0 ? (
              <EmptyState
                icon={<Search className="h-6 w-6" aria-hidden="true" />}
                title="No findings recorded"
                description="No rule crossed its threshold in this window. Run detection above to see which rules were evaluated, including the ones that found nothing."
              />
            ) : (
              <ul className="space-y-2">
                {data.items.map((item) => (
                  <li key={item.id ?? `${item.category}-${item.title}`}>
                    <RecommendationRow
                      item={item}
                      open={expanded === (item.id ?? item.title)}
                      onToggle={() =>
                        setExpanded((current) =>
                          current === (item.id ?? item.title) ? null : (item.id ?? item.title),
                        )
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
              itemLabel="findings"
            />
          </div>
        )}
      </QueryBoundary>
    </div>
  );
}

function RecommendationRow({
  item,
  open,
  onToggle,
}: {
  item: Recommendation;
  open: boolean;
  onToggle: () => void;
}) {
  const headingId = `rec-${item.id ?? item.title}`.replace(/[^a-zA-Z0-9-]/g, "");
  const evidence = useMemo(
    () => Object.entries(item.evidence ?? {}),
    [item.evidence],
  );

  return (
    <article
      className={cn(
        "rounded-md border bg-surface transition-colors",
        open ? "border-brand/40" : "border-line",
      )}
    >
      <h3>
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={headingId}
          className="flex w-full items-start gap-3 px-3 py-2.5 text-left"
        >
          {open ? (
            <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
          ) : (
            <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
          )}
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-ink">{item.title}</span>
              <SeverityBadge severity={item.priority} />
              <Badge tone="neutral">{humanize(item.category)}</Badge>
              <Badge tone="neutral">{item.status}</Badge>
              {item.application_name === null || item.application_name === undefined ? null : (
                <Badge tone="brand">{item.application_name}</Badge>
              )}
            </span>
            <span className="mt-0.5 block text-2xs leading-relaxed text-ink-secondary">
              {item.rationale}
            </span>
          </span>
          <span className="shrink-0 text-right">
            <span className="block text-2xs text-ink-muted">Severity</span>
            <span className="tabular-nums text-xs text-ink">{formatNumber(item.severity_score, 2)}</span>
          </span>
        </button>
      </h3>

      {open ? (
        <div id={headingId} className="border-t border-line px-3 py-3">
          <p className="text-2xs font-semibold uppercase tracking-wide text-ink-muted">
            Recommended action
          </p>
          <p className="mt-1 text-sm leading-relaxed text-ink">{item.recommendation}</p>

          {evidence.length > 0 ? (
            <>
              <p className="mt-3 text-2xs font-semibold uppercase tracking-wide text-ink-muted">
                Evidence
              </p>
              <dl className="mt-1 grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
                {evidence.map(([key, value]) => (
                  <div key={key} className="min-w-0">
                    <dt className="text-2xs text-ink-muted">{humanize(key)}</dt>
                    <dd className="break-words tabular-nums text-xs text-ink">
                      {typeof value === "number" ? formatEvidenceNumber(value) : humanize(String(value))}
                    </dd>
                  </div>
                ))}
              </dl>
            </>
          ) : null}

          <p className="mt-3 text-2xs text-ink-muted">
            Confidence {formatPercent(item.confidence * 100)} ·{" "}
            {item.created_at === null || item.created_at === undefined
              ? "not persisted"
              : `recorded ${formatRelative(item.created_at)}`}
          </p>
        </div>
      ) : null}
    </article>
  );
}

/**
 * Evidence values are printed verbatim apart from a trailing `%` on keys that
 * are already percentages. The backend reports `0.184` for a 18.4% field and
 * `184.0` for one already scaled, so guessing the unit from the key alone would
 * be wrong either way — the raw number plus its key is the honest rendering.
 */
function formatEvidenceNumber(value: number): string {
  return formatNumber(value, Math.abs(value) >= 100 ? 0 : 4);
}

/* -------------------------------------------------------------------------- */
/* Generation                                                                 */
/* -------------------------------------------------------------------------- */

function GeneratePanel({
  application,
  timeWindow,
}: {
  application: string | null;
  timeWindow: string;
}) {
  const queryClient = useQueryClient();
  const [minSeverity, setMinSeverity] = useState(0);
  const [persist, setPersist] = useState(false);
  const [result, setResult] = useState<RecommendationGenerateResponse | null>(null);

  const run = useMutation({
    mutationFn: (body: RecommendationGenerateRequest) => api.generateRecommendations(body),
    onSuccess: (data) => {
      setResult(data);
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });

  const rules = result?.rules_evaluated ?? [];

  return (
    <section className="card p-4" aria-label="Run the optimization engine">
      <PanelTitle
        actions={
          <button
            type="button"
            className="btn-primary"
            disabled={run.isPending}
            onClick={() => run.mutate({ application, window: timeWindow, persist, min_severity: minSeverity })}
          >
            <Play className="h-3.5 w-3.5" aria-hidden="true" />
            {run.isPending ? "Evaluating…" : "Evaluate rules"}
          </button>
        }
      >
        Evaluate rules
      </PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Each rule reads the recorded aggregates for this window and emits a finding only when its
        measurement crosses a threshold. A rule that finds nothing emits nothing, so the run report
        below is the honest picture of what was checked.
      </p>

      <div className="mb-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <SelectField
          id="recommendation-min-severity"
          label="Minimum severity"
          className="lg:w-48"
          value={String(minSeverity)}
          options={[
            { value: "0", label: "Any (0.00)" },
            { value: "0.2", label: "Medium and above (0.20)" },
            { value: "0.5", label: "High only (0.50)" },
            { value: "0.7", label: "Severe only (0.70)" },
          ]}
          onChange={(value) => setMinSeverity(Number(value))}
        />
        <div className="flex items-end">
          <label className="flex items-start gap-2" htmlFor="recommendation-persist">
            <input
              id="recommendation-persist"
              type="checkbox"
              className="mt-0.5 h-4 w-4 rounded border-line text-brand focus-visible:ring-2 focus-visible:ring-brand"
              checked={persist}
              onChange={(event) => setPersist(event.target.checked)}
            />
            <span className="text-sm text-ink">
              Save the findings
              <span className="mt-0.5 block text-2xs text-ink-muted">
                Off: returned here and discarded.
              </span>
            </span>
          </label>
        </div>
      </div>

      {run.isError ? (
        <ErrorPanel error={run.error} subject="the optimization engine" onRetry={() => run.reset()} />
      ) : null}

      {result === null ? null : (
        <div className="space-y-3 border-t border-line pt-3">
          <div className="flex flex-wrap items-center gap-2 text-2xs">
            <Badge tone={result.generated === 0 ? "good" : "warning"}>
              {result.generated} of {rules.length} rules produced a finding
            </Badge>
            <Badge tone="neutral">{persist ? "persisted" : "not persisted"}</Badge>
          </div>

          {result.notes === null || result.notes === undefined ? null : (
            <InfoNote>{result.notes}</InfoNote>
          )}

          {result.recommendations.length > 0 ? (
            <ul className="space-y-2">
              {result.recommendations.map((item, index) => (
                <li
                  key={item.id ?? `${item.category}-${index}`}
                  className="rounded-md border border-line bg-surface-sunken px-3 py-2"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-ink">{item.title}</span>
                    <SeverityBadge severity={item.priority} />
                    <Badge tone="neutral">{humanize(item.category)}</Badge>
                  </div>
                  <p className="mt-1 text-2xs leading-relaxed text-ink-secondary">{item.rationale}</p>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={<Lightbulb className="h-6 w-6" aria-hidden="true" />}
              title="No rule crossed its threshold"
              description={`${rules.length} rules ran against this window and none found a measurement outside its bounds. That is a real result, not an empty response.`}
            />
          )}

          <div>
            <h3 className="mb-1.5 text-2xs font-semibold uppercase tracking-wide text-ink-muted">
              Every rule that ran
            </h3>
            <DataTable
              caption="Rules evaluated by the optimization engine"
              columns={ruleColumns}
              rows={rules}
              rowKey={(row) => ruleLabel(row)}
              minWidth="40rem"
            />
          </div>
        </div>
      )}
    </section>
  );
}

/**
 * A stable, human-readable name for a rule row.
 *
 * Used both as the React key and as the visible label, so it must be a string
 * for every shape the engine can emit — including the case where the row is
 * missing both `rule` and `name`. Casting through `String()` instead would put
 * "[object Object]" in both places if the key ever became a nested object.
 */
function ruleLabel(row: Record<string, unknown>): string {
  for (const key of ["rule", "name"] as const) {
    const value = row[key];
    if (typeof value === "string" && value !== "") return value;
  }
  return "unnamed";
}

const ruleColumns: Column<Record<string, unknown>>[] = [
  {
    key: "rule",
    header: "Rule",
    render: (row) => (
      <span className="font-medium text-ink">{humanize(ruleLabel(row))}</span>
    ),
  },
  {
    key: "outcome",
    header: "Outcome",
    render: (row) => {
      const emitted = row.emitted ?? row.fired ?? row.triggered;
      if (typeof emitted === "boolean") {
        return <Badge tone={emitted ? "warning" : "good"}>{emitted ? "fired" : "quiet"}</Badge>;
      }
      if (typeof row.skipped === "boolean" && row.skipped) {
        return <Badge tone="neutral">skipped</Badge>;
      }
      return (
        <span className="text-2xs text-ink-muted">
          {typeof row.reason === "string" ? humanize(row.reason) : formatJsonValue(row.reason)}
        </span>
      );
    },
  },
  {
    key: "measurement",
    header: "What it measured",
    render: (row) => (
      <ul className="space-y-0.5 text-2xs text-ink-secondary">
        {Object.entries(row)
          .filter(([key]) => !META_KEYS.has(key))
          .map(([key, value]) => (
            <li key={key}>
              <span className="text-ink-muted">{humanize(key)}:</span>{" "}
              {typeof value === "number"
                ? formatEvidenceNumber(value)
                : typeof value === "boolean"
                  ? String(value)
                  : // A measurement the engine recorded can be a nested object —
                    // an evidence dict, a config it read. `humanize` is only for
                    // identifiers, and `String(value)` on an object would render
                    // "[object Object]", which reads like a value while telling
                    // the reader nothing.
                    typeof value === "string"
                    ? humanize(value)
                    : formatJsonValue(value)}
            </li>
          ))}
      </ul>
    ),
  },
];

/**
 * Keys that describe the rule rather than a measurement. Anything else in the
 * object is something the rule read, and is listed.
 */
const META_KEYS = new Set([
  "rule",
  "name",
  "emitted",
  "fired",
  "triggered",
  "skipped",
  "reason",
]);
