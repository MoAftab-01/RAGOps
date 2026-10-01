import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ChevronDown, ChevronRight, FlaskConical, Plus, Trophy } from "lucide-react";
import { api, type Experiment, type ExperimentCreate } from "@/lib/api";
import { useScope, usePageParam } from "@/hooks/useScope";
import { InfoNote, PageHeader, PanelTitle, TextAreaField, TextField } from "@/components/PageHeader";
import { QueryBoundary, ScopeBar } from "@/components/ScopeBar";
import { Pagination } from "@/components/Table";
import { EmptyState } from "@/components/EmptyState";
import { Badge, statusTone } from "@/components/Badge";
import { KeyValueList } from "@/components/Table";
import { ErrorPanel } from "@/components/ErrorPanel";
import {
  TONE_CLASSES,
  changeTone,
  formatDateTime,
  formatNumber,
  formatSigned,
  humanize,
  metricDirection,
} from "@/lib/format";

/**
 * §18: experiments — a hypothesis, a dataset, and configurations evaluated
 * against it.
 *
 * Two things this page refuses to do:
 *
 * - **It does not declare a winner from a missing run.** A variant with no
 *   `run_id` was never evaluated, so it has no metrics and is labelled
 *   "not evaluated" rather than shown as a zero it would lose on.
 *   `winner_variant` is echoed from the backend, which sets it only from runs
 *   that completed.
 * - **It does not colour a delta green because it went up.** Direction comes
 *   from `metricDirection`, transcribed from the backend's own
 *   `METRIC_DIRECTIONS`; a metric with no recorded direction is drawn flat
 *   rather than optimistically tinted.
 *
 * A variant's `metrics` is untyped JSON, so nothing is read out of it by
 * guessing a key. Deltas arrive already computed on `deltas_vs_baseline` and
 * those are the numbers shown.
 */

export default function ExperimentsPage() {
  const { params } = useScope();
  const { page, pageSize, setPage } = usePageParam();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const list = useQuery({
    queryKey: ["experiments", params, page, pageSize],
    queryFn: () => api.listExperiments({ ...params, page, page_size: pageSize }),
  });

  return (
    <div className="space-y-5">
      <PageHeader
        title="Experiments"
        description="A hypothesis, a dataset, and a set of configurations evaluated against it. A variant with no completed run has no score — that absence is the finding, and it is why this page will not pick a winner for you."
        scope={<ScopeBar />}
        actions={
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setCreating((open) => !open)}
            aria-expanded={creating}
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            {creating ? "Cancel" : "New experiment"}
          </button>
        }
      />

      {creating ? <CreatePanel application={params.application ?? null} onClose={() => setCreating(false)} /> : null}

      <QueryBoundary query={list} label="experiments">
        {(data) => (
          <div className="card p-4">
            <PanelTitle>Experiments</PanelTitle>

            {data.items.length === 0 ? (
              <EmptyState
                icon={<FlaskConical className="h-6 w-6" aria-hidden="true" />}
                title="No experiments yet"
                description="Create one to fix a hypothesis, a dataset, and the configurations to compare. Score a variant by running a retrieval evaluation against the same dataset and attaching that run."
                action={
                  <button type="button" className="btn-primary" onClick={() => setCreating(true)}>
                    <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                    New experiment
                  </button>
                }
              />
            ) : (
              <ul className="space-y-2">
                {data.items.map((experiment) => (
                  <li key={experiment.id}>
                    <ExperimentCard
                      experiment={experiment}
                      open={expanded === experiment.id}
                      onToggle={() => setExpanded((current) => (current === experiment.id ? null : experiment.id))}
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
              itemLabel="experiments"
            />
          </div>
        )}
      </QueryBoundary>
    </div>
  );
}

function ExperimentCard({
  experiment,
  open,
  onToggle,
}: {
  experiment: Experiment;
  open: boolean;
  onToggle: () => void;
}) {
  const panelId = `experiment-${experiment.id}`;

  return (
    <article className="rounded-md border border-line bg-surface">
      <h3>
        <button
          type="button"
          onClick={onToggle}
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
              <span className="text-sm font-medium text-ink">{experiment.name}</span>
              <Badge tone={statusTone(experiment.status)}>{humanize(experiment.status)}</Badge>
              <Badge tone="neutral">{experiment.dataset_name ?? "no dataset"}</Badge>
              <Badge tone="neutral">
                {experiment.variants.length} variant{experiment.variants.length === 1 ? "" : "s"}
              </Badge>
              {experiment.winner_variant !== null ? (
                <Badge tone="good">
                  <Trophy className="h-3 w-3" aria-hidden="true" />
                  {experiment.winner_variant}
                </Badge>
              ) : null}
            </span>
            {experiment.hypothesis !== null || experiment.description !== null ? (
              <span className="mt-0.5 block text-2xs leading-relaxed text-ink-secondary">
                {experiment.hypothesis ?? experiment.description}
              </span>
            ) : null}
          </span>
          <span className="shrink-0 text-right text-2xs text-ink-muted">
            {formatDateTime(experiment.created_at)}
          </span>
        </button>
      </h3>

      {open ? (
        <div id={panelId} className="space-y-3 border-t border-line bg-surface-sunken/40 px-3 py-3">
          <KeyValueList
            items={[
              { label: "Dataset", value: experiment.dataset_name ?? "—" },
              { label: "Application", value: experiment.application_id ?? "all applications" },
              { label: "Completed", value: formatDateTime(experiment.completed_at) },
            ]}
          />

          {experiment.variants.length === 0 ? (
            <EmptyState title="No variants" description="This experiment has nothing to compare." />
          ) : (
            <ul className="space-y-2">
              {experiment.variants.map((variant) => (
                <li key={variant.id} className="rounded-md border border-line bg-surface px-3 py-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-sm font-medium text-ink">
                      {variant.name}
                      {variant.run_id !== null ? null : <Badge tone="neutral">not evaluated</Badge>}
                    </span>
                    {variant.run_id !== null ? (
                      <Link
                        to="/rag-evaluation"
                        className="font-mono text-2xs text-brand hover:underline"
                        title="Open RAG Evaluation to inspect this run"
                      >
                        run {variant.run_id.slice(0, 8)}
                      </Link>
                    ) : null}
                  </div>

                  {variant.config && Object.keys(variant.config).length > 0 ? (
                    <p className="mt-1 break-words font-mono text-2xs text-ink-muted">
                      {JSON.stringify(variant.config)}
                    </p>
                  ) : null}

                  {variant.deltas_vs_baseline.length === 0 ? (
                    <p className="mt-2 text-2xs text-ink-muted">
                      {variant.run_id !== null
                        ? "No deltas were recorded for this variant."
                        : "No run, so there is nothing to compare against the baseline."}
                    </p>
                  ) : (
                    <table className="mt-2 w-full border-collapse text-left text-2xs">
                      <caption className="sr-only">Metric deltas for {variant.name}</caption>
                      <thead>
                        <tr className="border-b border-line">
                          <th scope="col" className="py-1 font-semibold text-ink-secondary">Metric</th>
                          <th scope="col" className="py-1 text-right font-semibold text-ink-secondary">Baseline</th>
                          <th scope="col" className="py-1 text-right font-semibold text-ink-secondary">This variant</th>
                          <th scope="col" className="py-1 text-right font-semibold text-ink-secondary">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {variant.deltas_vs_baseline.map((delta, index) => (
                          // The metric name arrives as an untyped key. The index
                          // fallback keeps React's keys unique when a delta
                          // arrives without one; it is a key, never rendered.
                          <DeltaRow key={typeof delta.metric === "string" ? delta.metric : `delta-${index}`} delta={delta} />
                        ))}
                      </tbody>
                    </table>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : null}
    </article>
  );
}

/**
 * One delta row, coloured by the backend's own metric direction.
 *
 * The metric name arrives as an untyped key, so it is narrowed to a string
 * rather than cast: `String(delta.metric)` on an object value would put
 * "[object Object]" in the first column, which is both unreadable and a claim
 * that a metric by that name exists. An absent name reads as "Metric" — the
 * label is still honest, and `metricDirection` is asked about it rather than
 * assumed, so an unrecognised metric draws flat, which is the same answer the
 * backend gives.
 */
function DeltaRow({ delta }: { delta: Record<string, unknown> }) {
  const metric = typeof delta.metric === "string" ? delta.metric : "metric";
  const change = typeof delta.absolute_change === "number" ? delta.absolute_change : null;
  const direction = metricDirection(metric);
  const tone = changeTone(change, direction);

  return (
    <tr className="border-b border-line/50 last:border-0">
      <td className="py-1 text-ink">
        {humanize(metric)}
        <span className="ml-1 text-ink-muted" title={direction}>
          {direction === "higher_is_better" ? "↑" : direction === "lower_is_better" ? "↓" : "·"}
        </span>
      </td>
      <td className="py-1 text-right tabular-nums text-ink-secondary">
        {typeof delta.baseline === "number" ? formatNumber(delta.baseline, 4) : "—"}
      </td>
      <td className="py-1 text-right tabular-nums text-ink-secondary">
        {typeof delta.candidate === "number" ? formatNumber(delta.candidate, 4) : "—"}
      </td>
      <td className={`py-1 text-right tabular-nums ${TONE_CLASSES[tone]}`}>
        {formatSigned(change, 4)}
        <span className="sr-only">
          {" "}
          — {tone === "good" ? "improved" : tone === "bad" ? "regressed" : tone === "flat" ? "unchanged" : "not measured"}
        </span>
      </td>
    </tr>
  );
}

/* -------------------------------------------------------------------------- */
/* Creation                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * A variant being edited: the config is held as raw text so a half-typed object
 * is not destroyed on every keystroke, and parsed on submit.
 */
interface DraftVariant {
  name: string;
  runId: string;
  configText: string;
}

function CreatePanel({ application, onClose }: { application: string | null; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [dataset, setDataset] = useState("");
  const [variants, setVariants] = useState<DraftVariant[]>([
    { name: "baseline", runId: "", configText: "{}" },
    { name: "candidate", runId: "", configText: "{}" },
  ]);

  // A parse failure is reported on its own field rather than sent as a string
  // and rejected server-side with a message the user has to decode.
  const parsed = variants.map((variant) => parseConfig(variant.configText));
  const configErrorIndex = parsed.findIndex((result) => !result.ok);

  const create = useMutation({
    mutationFn: (body: ExperimentCreate) => api.createExperiment(body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
      onClose();
    },
  });

  const update = (index: number, change: Partial<DraftVariant>): void => {
    setVariants((current) =>
      current.map((variant, position) => (position === index ? { ...variant, ...change } : variant)),
    );
  };

  return (
    <section className="card p-4" aria-label="Create an experiment">
      <PanelTitle
        actions={
          <button type="button" className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
        }
      >
        New experiment
      </PanelTitle>

      <div className="grid gap-3 sm:grid-cols-2">
        <TextField
          id="experiment-name"
          label="Name"
          value={name}
          onChange={setName}
          placeholder="hybrid-vs-bm25"
        />
        <TextField
          id="experiment-dataset"
          label="Dataset"
          value={dataset}
          onChange={setDataset}
          placeholder="demo-qa"
          hint="The evaluation dataset every variant is scored against."
        />
        <TextAreaField
          id="experiment-hypothesis"
          label="Hypothesis"
          className="sm:col-span-2"
          rows={2}
          value={hypothesis}
          onChange={setHypothesis}
          placeholder="Raising top_k to 10 lifts recall@10 without costing more than 400 extra context tokens."
        />
      </div>

      <div className="mt-4">
        <h3 className="field-label">Variants</h3>
        <ul className="space-y-2">
          {variants.map((variant, index) => (
            <li key={index} className="rounded-md border border-line bg-surface-sunken p-2.5">
              <div className="flex flex-wrap items-end gap-2">
                <TextField
                  id={`variant-name-${index}`}
                  className="w-44"
                  label="Name"
                  value={variant.name}
                  onChange={(value) => update(index, { name: value })}
                />
                <TextField
                  id={`variant-run-${index}`}
                  className="flex-1 min-w-52"
                  label="Evaluation run id"
                  value={variant.runId}
                  onChange={(value) => update(index, { runId: value })}
                  placeholder="optional"
                  hint="Attach a completed run to score this variant."
                />
                <button
                  type="button"
                  className="btn-ghost"
                  disabled={variants.length <= 2}
                  onClick={() =>
                    setVariants((current) => current.filter((_unused, position) => position !== index))
                  }
                >
                  Remove
                </button>
                {index === 0 ? <Badge tone="brand">baseline</Badge> : null}
              </div>
              <TextAreaField
                id={`variant-config-${index}`}
                className="mt-2"
                label="Config override (JSON)"
                rows={3}
                value={variant.configText}
                onChange={(value) => update(index, { configText: value })}
                hint='Applied to the retriever for this variant, e.g. {"top_k": 10, "rerank": true}.'
              />
              {configErrorIndex === index ? (
                <p className="mt-1 text-2xs text-status-critical">
                  {parsed[index].ok ? "" : parsed[index].error} — use a JSON object such as{" "}
                  <code>{"{"}"top_k": 10{"}"}</code>.
                </p>
              ) : null}
            </li>
          ))}
        </ul>
        <button
          type="button"
          className="btn-ghost mt-2"
          onClick={() =>
            setVariants((current) => [
              ...current,
              { name: `variant-${current.length + 1}`, runId: "", configText: "{}" },
            ])
          }
        >
          <Plus className="h-3.5 w-3.5" aria-hidden="true" />
          Add variant
        </button>
      </div>

      <InfoNote className="mt-3">
        Creating an experiment records the hypothesis and the variants; it does not run anything.
        Attach an evaluation run id to a variant once{" "}
        <span className="font-medium text-ink-secondary">RAG Evaluation</span> has scored that
        configuration against the dataset, and the deltas appear here.
      </InfoNote>

      {create.isError ? (
        <ErrorPanel
          error={create.error}
          subject="creating the experiment"
          onRetry={() => create.reset()}
          className="mt-3"
        />
      ) : null}

      <div className="mt-3 flex justify-end gap-2">
        <button type="button" className="btn-ghost" onClick={onClose}>
          Cancel
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={create.isPending || name.trim() === "" || variants.length < 2 || configErrorIndex !== -1}
          onClick={() =>
            create.mutate({
              name: name.trim(),
              application,
              hypothesis: hypothesis.trim() === "" ? null : hypothesis.trim(),
              dataset_name: dataset.trim() === "" ? null : dataset.trim(),
              baseline_variant: variants[0].name,
              variants: variants.map((variant, index) => ({
                name: variant.name.trim() === "" ? `variant-${index + 1}` : variant.name.trim(),
                // `parsed[index]` is known-good here: the button is disabled
                // whenever any config failed to parse, so this cannot be the
                // failure branch.
                config: parsed[index].ok ? parsed[index].value : {},
                run_id: variant.runId.trim() === "" ? null : variant.runId.trim(),
              })),
            })
          }
        >
          {create.isPending ? "Creating…" : "Create experiment"}
        </button>
      </div>
    </section>
  );
}

/**
 * Parses one config override.
 *
 * Held as text so a half-typed object is not destroyed on every keystroke; a
 * failure is reported against its own field rather than sent to the backend and
 * returned as a 422 the user has to decode.
 */
function parseConfig(
  text: string,
): { ok: true; value: Record<string, unknown> } | { ok: false; error: string } {
  if (text.trim() === "") return { ok: true, value: {} };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "not valid JSON";
    return { ok: false, error: detail };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: "config must be an object, not a list or a scalar" };
  }
  return { ok: true, value: parsed as Record<string, unknown> };
}
