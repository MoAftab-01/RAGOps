import type { ReactNode } from "react";
import { type UseQueryResult } from "@tanstack/react-query";
import { RotateCcw } from "lucide-react";
import { TIME_WINDOWS, type TimeWindow } from "@/lib/api";
import { useScope } from "@/hooks/useScope";
import { useApplicationNames } from "@/hooks/useApplications";
import { FilterBar, SelectField } from "@/components/PageHeader";
import { ErrorPanel } from "@/components/ErrorPanel";
import { LoadingPanel } from "@/components/Spinner";

const WINDOW_LABELS: Record<TimeWindow, string> = {
  "1h": "Last hour",
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
};

/**
 * The time-window and application selector every analytics page shows.
 *
 * State lives in the URL, so a filtered view is a link. The two controls are
 * deliberately *not* the same control: the window narrows the query, while the
 * application narrows the subject, and conflating them is what makes a filtered
 * dashboard impossible to explain after the fact.
 */
export function ScopeBar({ extra }: { extra?: ReactNode }) {
  const { timeWindow, application, setTimeWindow, setApplication, reset } = useScope();
  const names = useApplicationNames();
  const dirty = application !== null || timeWindow !== "7d";

  return (
    <FilterBar
      onReset={dirty ? reset : undefined}
    >
      <SelectField
        id="scope-window"
        label="Time window"
        value={timeWindow}
        options={TIME_WINDOWS.map((value) => ({ value, label: WINDOW_LABELS[value] }))}
        onChange={(value) => setTimeWindow(value as TimeWindow)}
      />
      <SelectField
        id="scope-application"
        label="Application"
        value={application ?? ""}
        options={[
          { value: "", label: "All applications" },
          ...names.map((name) => ({ value: name, label: name })),
        ]}
        onChange={(value) => setApplication(value === "" ? null : value)}
        hint={names.length === 0 ? "No applications recorded yet." : undefined}
      />
      {extra}
    </FilterBar>
  );
}

/**
 * The state machine every page's root follows.
 *
 * Collapsing the five states into one place is what keeps "loading" from
 * looking like "no data" and "failed" from looking like "empty" — three
 * conditions that are easy to conflate when each page writes its own version,
 * and a conflation that would report a dead API as an empty database.
 */
export function QueryBoundary<T>({
  query,
  label,
  children,
}: {
  query: UseQueryResult<T>;
  /** What is loading, for the spinner and the screen-reader announcement. */
  label: string;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <LoadingPanel label={label} />;
  if (query.isError) return <ErrorPanel error={query.error} subject={label} onRetry={() => void query.refetch()} />;
  return <>{children(query.data)}</>;
}

/** `isFetching` on its own, for the "refetching, showing previous data" flag. */
export function useIsFetching(query: Pick<UseQueryResult<unknown>, "isFetching" | "isPending">): boolean {
  return query.isFetching && !query.isPending;
}

/** Retry button for a mutation that failed, phrased for the action that failed. */
export function MutationRetry({ onRetry, label = "Retry" }: { onRetry: () => void; label?: string }) {
  return (
    <button type="button" className="btn-secondary" onClick={onRetry}>
      <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </button>
  );
}
