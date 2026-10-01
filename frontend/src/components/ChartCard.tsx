import { useId, useState, type ReactNode } from "react";
import { BarChart3, Table2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { seriesSwatchClass } from "@/lib/palette";

export interface ChartLegendEntry {
  /** Series key — also the 1-based palette slot, so colour follows the entity. */
  key: string;
  label: string;
  /** Overrides the slot colour, for a series whose colour comes from elsewhere. */
  color?: string;
}

export interface ChartTableSpec {
  columns: string[];
  rows: Array<Array<string | number>>;
}

export interface ChartCardProps {
  title: string;
  /** What is plotted and what the axis means. */
  description?: ReactNode;
  /** The measure and its unit, so the y-axis is never a bare number. */
  valueLabel: string;
  timeRange?: { start: string; end: string };
  legend?: ChartLegendEntry[];
  children: ReactNode;
  /**
   * The table twin. Mandatory: three of the light-mode palette steps sit below
   * 3:1 against the light surface, and the validator's relief rule is a visible
   * label or a table view. Every chart here ships one, so the relief is never
   * optional and never quietly dropped.
   */
  table: ChartTableSpec;
  footnote?: ReactNode;
  className?: string;
}

/**
 * The frame around every chart: title, unit label, legend, and the table twin.
 *
 * Charts are switched, not duplicated — a card renders either the plot or the
 * table, so there is no second chart on the page competing with the first.
 * Legends are always shown for two or more series, and never for one: the
 * title names a single series already.
 */
export function ChartCard({
  title,
  description,
  valueLabel,
  timeRange,
  legend,
  children,
  table,
  footnote,
  className,
}: ChartCardProps) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const titleId = useId();
  const showLegend = (legend?.length ?? 0) >= 2;

  return (
    <figure className={cn("card flex flex-col p-4", className)} aria-labelledby={titleId}>
      <div className="mb-1 flex flex-wrap items-start justify-between gap-2">
        <figcaption id={titleId} className="min-w-0">
          <h3 className="text-sm font-semibold text-ink">{title}</h3>
          {description === undefined ? null : (
            <p className="mt-0.5 text-2xs leading-relaxed text-ink-secondary">{description}</p>
          )}
        </figcaption>
        <div
          className="flex shrink-0 items-center gap-1"
          role="group"
          aria-label={`${title} view`}
        >
          <button
            type="button"
            onClick={() => setView("chart")}
            aria-pressed={view === "chart"}
            className={cn("btn-ghost px-2 py-1", view === "chart" && "bg-surface-raised text-ink")}
          >
            <BarChart3 className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="sr-only sm:not-sr-only">Chart</span>
          </button>
          <button
            type="button"
            onClick={() => setView("table")}
            aria-pressed={view === "table"}
            className={cn("btn-ghost px-2 py-1", view === "table" && "bg-surface-raised text-ink")}
          >
            <Table2 className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="sr-only sm:not-sr-only">Table</span>
          </button>
        </div>
      </div>

      {showLegend ? (
        <ul className="mb-2 mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          {legend?.map((entry, index) => (
            <li key={entry.key} className="flex items-center gap-1.5 text-2xs text-ink-secondary">
              <span
                className={cn(
                  "h-2 w-2 shrink-0 rounded-[2px]",
                  entry.color === undefined && seriesSwatchClass(index + 1),
                )}
                style={entry.color === undefined ? undefined : { backgroundColor: entry.color }}
                aria-hidden="true"
              />
              {entry.label}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="min-w-0 flex-1">
        {view === "chart" ? (
          children
        ) : (
          <div className="max-h-72 overflow-auto scrollbar-thin">
            <table className="w-full border-collapse text-left text-xs">
              <caption className="sr-only">{title} — table view</caption>
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-line">
                  {table.columns.map((column, index) => (
                    <th
                      key={column}
                      scope="col"
                      className={cn(
                        "whitespace-nowrap px-2 py-1.5 font-semibold text-ink-secondary",
                        index === 0 ? "text-left" : "text-right",
                      )}
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.length === 0 ? (
                  <tr>
                    <td colSpan={table.columns.length} className="px-2 py-3 text-ink-muted">
                      No rows in this window.
                    </td>
                  </tr>
                ) : (
                  table.rows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="border-b border-line/50 last:border-0">
                      {row.map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className={cn(
                            "whitespace-nowrap px-2 py-1.5",
                            cellIndex === 0 ? "text-left text-ink" : "text-right tabular-nums text-ink-secondary",
                          )}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-t border-line/60 pt-2">
        <p className="text-2xs text-ink-muted">
          <span className="font-medium text-ink-secondary">{valueLabel}</span>
          {timeRange === undefined ? null : (
            <>
              {" · "}
              <span className="tabular-nums">
                {timeRange.start} → {timeRange.end}
              </span>
            </>
          )}
        </p>
        {footnote === undefined ? null : (
          <p className="text-2xs text-ink-muted">{footnote}</p>
        )}
      </div>
    </figure>
  );
}

/** Marks a card as stale during a refetch: previous render held, dimmed. */
export function RefetchingOverlay({ active, children }: { active: boolean; children: ReactNode }) {
  return (
    <div className={cn("transition-opacity", active && "pointer-events-none opacity-60")}>
      {children}
    </div>
  );
}
