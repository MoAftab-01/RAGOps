/**
 * §27: component tests for the pages whose whole job is not to mislead.
 *
 * Each test below pins one claim the page makes in its own copy. They are not
 * snapshot tests: a snapshot would happily record "recall@10: 0.0%" for an
 * unmeasured K and call it correct. What is asserted here is the *distinction*
 * each page exists to keep — unmeasured vs zero, lifetime vs window, missing
 * dependency vs clean system, not-evaluated vs lost.
 */

import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ExperimentsPage from "@/pages/ExperimentsPage";
import { renderPage, stubApi } from "@/test/render";

const WINDOW = "2026-09-30T12:00:00Z";

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("ExperimentsPage", () => {
  const experiment = {
    id: "exp-1",
    name: "hybrid-vs-bm25",
    status: "running",
    dataset_name: "demo-qa",
    application_id: "app-1",
    hypothesis: "Raising top_k lifts recall without costing more than 400 tokens.",
    created_at: WINDOW,
    completed_at: null,
    winner_variant: null,
    variants: [
      {
        id: "v-baseline",
        name: "baseline",
        config: { top_k: 5 },
        run_id: "run-baseline",
        metrics: null,
        deltas_vs_baseline: [],
      },
      {
        // Never evaluated: no run_id, so there is nothing to compare.
        id: "v-candidate",
        name: "candidate",
        config: { top_k: 10 },
        run_id: null,
        metrics: null,
        deltas_vs_baseline: [],
      },
    ],
  };

  function listExperiment() {
    return {
      items: [experiment],
      total: 1,
      page: 1,
      page_size: 25,
      has_next: false,
    };
  }

  it("labels an unevaluated variant instead of scoring it zero", async () => {
    stubApi({ listExperiments: vi.fn().mockResolvedValue(listExperiment()) });
    renderPage(<ExperimentsPage />);

    fireEvent.click(await screen.findByRole("button", { name: /hybrid-vs-bm25/i }));

    const candidate = screen.getByText("candidate").closest("li") as HTMLElement;
    // The claim: this variant was never run. It is not a zero it lost on.
    expect(within(candidate).getByText(/not evaluated/i)).toBeInTheDocument();
    expect(candidate.textContent).toMatch(/no run, so there is nothing to compare/i);
  });

  it("declares no winner while a variant has no completed run", async () => {
    stubApi({ listExperiments: vi.fn().mockResolvedValue(listExperiment()) });
    renderPage(<ExperimentsPage />);

    await screen.findByRole("button", { name: /hybrid-vs-bm25/i });

    // `winner_variant` is echoed from the backend, which sets it only from runs
    // that completed. This one is null, so the page must not invent a trophy.
    expect(screen.queryByText("candidate")).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /trophy/i })).not.toBeInTheDocument();
  });

  it("colours a delta by the backend's own metric direction", async () => {
    // recall rising is good; latency rising is bad. Same sign, opposite colour.
    const withDeltas = {
      ...experiment,
      status: "completed",
      winner_variant: "candidate",
      variants: [
        { ...experiment.variants[0], run_id: "run-baseline", deltas_vs_baseline: [] },
        {
          ...experiment.variants[1],
          run_id: "run-candidate",
          deltas_vs_baseline: [
            { metric: "recall_at_k", baseline: 0.6, candidate: 0.81, absolute_change: 0.21 },
            { metric: "p95_latency_ms", baseline: 800, candidate: 1400, absolute_change: 600 },
            // No recorded direction: drawn flat rather than optimistically tinted.
            { metric: "vendor_blend", baseline: 1, candidate: 2, absolute_change: 1 },
          ],
        },
      ],
    };
    stubApi({
      listExperiments: vi
        .fn()
        .mockResolvedValue({ ...listExperiment(), items: [withDeltas] }),
    });
    renderPage(<ExperimentsPage />);

    fireEvent.click(await screen.findByRole("button", { name: /hybrid-vs-bm25/i }));

    const row = (metric: RegExp) => screen.getByText(metric).closest("tr") as HTMLElement;

    const recall = row(/^Recall at k/);
    expect(within(recall).getByText("+0.2100")).toHaveClass("text-status-good");
    expect(recall.textContent).toContain("improved");

    const latency = row(/^P95 latency ms/);
    expect(within(latency).getByText("+600.0000")).toHaveClass("text-status-critical");
    expect(latency.textContent).toContain("regressed");

    const unknown = row(/Vendor blend/);
    expect(within(unknown).getByText("+1.0000")).toHaveClass("text-ink-secondary");
    expect(unknown.textContent).toContain("unchanged");
  });

  it("renders an unmeasured delta as no value rather than +0.0000", async () => {
    const withNullDelta = {
      ...experiment,
      variants: [
        experiment.variants[0],
        {
          ...experiment.variants[1],
          run_id: "run-candidate",
          deltas_vs_baseline: [{ metric: "mrr", baseline: 0.7, candidate: 0.7, absolute_change: null }],
        },
      ],
    };
    stubApi({
      listExperiments: vi.fn().mockResolvedValue({ ...listExperiment(), items: [withNullDelta] }),
    });
    renderPage(<ExperimentsPage />);

    fireEvent.click(await screen.findByRole("button", { name: /hybrid-vs-bm25/i }));

    const cell = screen.getByText(/^Mrr/).closest("tr") as HTMLElement;
    expect(cell.textContent).toContain("not measured");
    expect(cell.textContent).not.toContain("+0.0000");
  });

  it("reports a JSON config parse failure on its own field and blocks submit", async () => {
    stubApi({
      listExperiments: vi.fn().mockResolvedValue(listExperiment()),
      createExperiment: vi.fn(),
    });
    renderPage(<ExperimentsPage />);

    fireEvent.click(await screen.findByRole("button", { name: /New experiment/i }));

    fireEvent.change(screen.getAllByLabelText(/Config override/i)[0], {
      target: { value: '{{"top_k": 10}' },
    });

    // The message names the field and shows the shape that would work.
    expect(await screen.findByText(/use a JSON object such as/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Create experiment/i })).toBeDisabled();
  });

  it("rejects a JSON list, which is valid JSON but not a config", async () => {
    stubApi({
      listExperiments: vi.fn().mockResolvedValue(listExperiment()),
      createExperiment: vi.fn(),
    });
    renderPage(<ExperimentsPage />);

    fireEvent.click(await screen.findByRole("button", { name: /New experiment/i }));
    fireEvent.change(screen.getAllByLabelText(/Config override/i)[0], {
      target: { value: "[[1,2]]" },
    });
    expect(await screen.findByText(/must be an object, not a list or a scalar/i)).toBeInTheDocument();
  });
});
