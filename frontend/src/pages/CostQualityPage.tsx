import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { api, type CostQualityReport, type ModelComparison } from "@/lib/api";
import { useScope } from "@/hooks/useScope";
import { InfoNote, PageHeader } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { ScatterPlot, type ScatterDatum } from "@/lib/charts";
import { ChartCard } from "@/components/ChartCard";
import { DataTable, type Column } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import { Badge } from "@/components/Badge";
import {
  formatCost,
  formatLatencyMs,
  formatNumber,
  formatScore,
  humanize,
} from "@/lib/format";

/**
 * §13: cost against quality, per model.
 *
 * Two things this page is careful about, because both are easy to fake:
 *
 * 1. **Pricing is simulated and labelled as such.** A local Ollama model has no
 *    metered price, so it is priced at zero — which means a local run reads as
 *    "no cost" and a comparison against a metered model is not a like-for-like
 *    saving. The label comes from the backend and is never rewritten here.
 * 2. **Quality is only plotted for models that were actually evaluated.** A
 *    model with no evaluation has no quality score, so it is excluded from the
 *    scatter rather than plotted at zero. A point at (cost, 0) would assert the
 *    model was terrible, when the truth is that nobody measured it.
 */

export default function CostQualityPage() {
  const { params } = useScope();
  const report = useQuery({
    queryKey: ["cost-quality", params],
    queryFn: () => api.costQuality(params),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Cost &amp; Quality"
        description="Where cost and answer quality can be traded against each other, per model. Quality comes from recorded evaluations, so a model with no evaluation run is absent rather than scored zero."
        scope={<ScopeBar />}
      />

      <QueryBoundary query={report} label="cost and quality">
        {(data) => <ReportBody data={data} />}
      </QueryBoundary>
    </div>
  );
}

function ReportBody({ data }: { data: CostQualityReport }) {
  // Only models with a measured quality score can be placed on a scatter.
  // Anything else would be a point at zero, which claims a result nobody made.
  const measured = data.models.filter((model) => isMeasured(model.quality_score));
  const unmeasured = data.models.filter((model) => !isMeasured(model.quality_score));

  const points: ScatterDatum[] = measured.map((model) => ({
    x: model.cost_per_1k_tokens,
    y: model.quality_score ?? 0,
    label: model.model_name,
    weight: model.call_count,
    // The best value the backend identified is the one point worth seeing
    // first, so it is the only one that gets a different ink.
    highlight: model.model_name === data.best_value?.toString(),
  }));

  return (
    <>
      {data.models.length === 0 ? (
        <EmptyState
          title="No model usage in this window"
          description="Nothing was routed through a model in the selected scope and window. Widen the window, or clear the application filter."
        />
      ) : (
        <>
          <ChartCard
            title="Cost per 1k tokens against quality"
            description={`Each point is one model, sized by call count. ${data.quality_metric_definition}`}
            valueLabel={`${data.cost_label} per 1k tokens (vertical) against 0–100 quality (horizontal)`}
            timeRange={{ start: data.start, end: data.end }}
            table={{
              columns: ["Model", "Calls", "Cost per 1k", "Quality", "Local"],
              rows: measured.map((model) => [
                model.model_name,
                formatNumber(model.call_count),
                formatCost(model.cost_per_1k_tokens),
                formatScore(model.quality_score),
                model.is_local ? "yes" : "no",
              ]),
            }}
            footnote={
              measured.length === 0
                ? "No model in this window has a recorded quality score, so there is nothing to plot."
                : `${measured.length} of ${data.models.length} models have a measured quality score.`
            }
          >
            <ScatterPlot
              data={points}
              xLabel="Cost per 1k tokens"
              yLabel="Quality score"
              xFormat={(value) => formatCost(value)}
              yFormat={(value) => formatScore(value)}
            />
          </ChartCard>

          <InfoNote tone="warning">
            {data.cost_label} — these prices are simulated for local comparison, not billed. Local
            models are priced at zero, so a local model appearing cheaper than a metered one is a
            fact about the pricing table, not a measured saving.
          </InfoNote>

          {unmeasured.length > 0 ? (
            <InfoNote>
              Not plotted: {unmeasured.map((model) => model.model_name).join(", ")}. These models
              were called in this window but have no evaluation run, so no quality score exists for
              them. They are listed in the table below with the quality cell left empty.
            </InfoNote>
          ) : null}

          <ModelTable models={data.models} costLabel={data.cost_label} />
        </>
      )}
    </>
  );
}

function isMeasured(value: number | null | undefined): value is number {
  return value !== null && value !== undefined && !Number.isNaN(value);
}

function ModelTable({ models, costLabel }: { models: ModelComparison[]; costLabel: string }) {
  const columns: Column<ModelComparison>[] = [
    {
      key: "model",
      header: "Model",
      render: (row) => (
        <span className="flex flex-col">
          <span className="font-medium text-ink">
            {row.model_name}
            {row.is_local ? <Badge className="ml-1.5" tone="brand">local</Badge> : null}
          </span>
          <span className="text-2xs text-ink-muted">{row.provider}</span>
        </span>
      ),
    },
    { key: "calls", header: "Calls", align: "right", render: (row) => formatNumber(row.call_count) },
    {
      key: "tokens",
      header: "Total tokens",
      align: "right",
      render: (row) => formatNumber(row.total_tokens),
    },
    {
      key: "input",
      header: "Avg input",
      align: "right",
      render: (row) => formatNumber(row.avg_input_tokens),
    },
    {
      key: "output",
      header: "Avg output",
      align: "right",
      render: (row) => formatNumber(row.avg_output_tokens),
    },
    {
      key: "latency",
      header: "Avg latency",
      align: "right",
      render: (row) => formatLatencyMs(row.avg_latency_ms),
    },
    {
      key: "cost",
      header: "Cost",
      align: "right",
      render: (row) => formatCost(row.estimated_cost, costLabel),
    },
    {
      key: "per1k",
      header: "Cost / 1k",
      align: "right",
      render: (row) => formatCost(row.cost_per_1k_tokens),
    },
    // Each quality cell is rendered on its own so a missing measurement stays
    // an em-dash. `formatScore` already does that, which is why there is no
    // `?? 0` anywhere near these values.
    {
      key: "retrieval",
      header: "Retrieval",
      align: "right",
      render: (row) => formatScore(row.retrieval_score),
    },
    {
      key: "faithfulness",
      header: "Faithfulness",
      align: "right",
      render: (row) => formatScore(row.faithfulness),
    },
    {
      key: "relevance",
      header: "Answer relevance",
      align: "right",
      render: (row) => formatScore(row.answer_relevance),
    },
    {
      key: "quality",
      header: "Quality",
      align: "right",
      render: (row) => (
        <span className={isMeasured(row.quality_score) ? "font-medium text-ink" : "no-value"}>
          {formatScore(row.quality_score)}
        </span>
      ),
    },
  ];

  return (
    <div className="card p-4">
      <h3 className="mb-1 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Info className="h-3.5 w-3.5 text-ink-muted" aria-hidden="true" />
        Per-model detail
      </h3>
      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Every model called in this window, with the quality measurements that exist for it. An
        em-dash in a quality column means no evaluation covered that model.
      </p>
      <DataTable
        caption="Per-model cost and quality detail"
        columns={columns}
        rows={models}
        rowKey={(row) => row.model_name}
        minWidth="64rem"
        empty={
          <EmptyState
            title="No models called"
            description="No LLM calls were recorded in the selected scope and window."
          />
        }
      />
      <p className="mt-3 text-2xs text-ink-muted">
        {humanize("quality_score")} is a weighted blend of the evaluation dimensions that were
        actually measured for each model; models with fewer measured dimensions are not comparable to
        models with all of them.
      </p>
    </div>
  );
}
