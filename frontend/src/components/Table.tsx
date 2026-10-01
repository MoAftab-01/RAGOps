import type { ReactNode } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";
import { formatNumber } from "@/lib/format";
import { EmptyState } from "@/components/EmptyState";

export interface Column<T> {
  key: string;
  header: ReactNode;
  /** Cell renderer. Receives the whole row so a cell can reach its siblings. */
  render: (row: T) => ReactNode;
  /** Right-align numeric columns; the header follows. */
  align?: "left" | "right";
  className?: string;
  headerClassName?: string;
}

export interface DataTableProps<T> {
  caption: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** Rendered under the body when `rows` is empty. */
  empty?: ReactNode;
  onRowClick?: (row: T) => void;
  /** Marks the row a detail panel is currently showing. */
  isRowActive?: (row: T) => boolean;
  className?: string;
  /** Scrolls the table sideways inside its own box rather than the page. */
  minWidth?: string;
}

/**
 * The one table in the app.
 *
 * Every page that shows rows uses this, so the "no results" case, the focus
 * ring, and the horizontal overflow behaviour are identical everywhere. A table
 * that is wider than the viewport scrolls inside its own container — the page
 * itself never scrolls sideways.
 */
export function DataTable<T>({
  caption,
  columns,
  rows,
  rowKey,
  empty,
  onRowClick,
  isRowActive,
  className,
  minWidth = "44rem",
}: DataTableProps<T>) {
  if (rows.length === 0) {
    return (
      <>
        {empty ?? <EmptyState title="No rows" description="The request returned an empty list." />}
      </>
    );
  }

  return (
    <div className={cn("overflow-x-auto scrollbar-thin", className)}>
      <table className="w-full border-collapse text-left text-sm" style={{ minWidth }}>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-line">
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cn(
                  "whitespace-nowrap px-3 py-2 text-2xs font-semibold uppercase tracking-wide text-ink-secondary",
                  column.align === "right" ? "text-right" : "text-left",
                  column.headerClassName,
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = rowKey(row);
            const active = isRowActive?.(row) ?? false;
            return (
              <tr
                key={key}
                onClick={onRowClick === undefined ? undefined : () => onRowClick(row)}
                className={cn(
                  "border-b border-line/60 last:border-0",
                  onRowClick === undefined ? undefined : "cursor-pointer hover:bg-surface-raised",
                  active ? "bg-surface-raised" : undefined,
                )}
              >
                {columns.map((column, index) => (
                  <td
                    key={column.key}
                    className={cn(
                      "px-3 py-2 align-top text-ink",
                      column.align === "right" ? "text-right tabular-nums" : "text-left",
                      column.className,
                    )}
                  >
                    {onRowClick === undefined ? (
                      column.render(row)
                    ) : (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          onRowClick(row);
                        }}
                        className="block w-full text-left align-top focus-visible:rounded-sm"
                        aria-expanded={active}
                      >
                        {index === 0 ? column.render(row) : <span className="contents">{column.render(row)}</span>}
                      </button>
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export interface PaginationProps {
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  /** What the rows are, so the count can be labelled. */
  itemLabel?: string;
}

/** Page controls driven by the backend's own `total`, not by the row count. */
export function Pagination({
  page,
  pageSize,
  total,
  onPageChange,
  itemLabel = "rows",
}: PaginationProps) {
  const pageCount = Math.max(1, Math.ceil(total / Math.max(pageSize, 1)));
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);

  return (
    <nav
      className="flex flex-wrap items-center justify-between gap-3 px-3 py-2 text-xs text-ink-secondary"
      aria-label="Pagination"
    >
      <p>
        {total === 0 ? (
          <>No {itemLabel} in this window</>
        ) : (
          <>
            Showing <span className="tabular-nums text-ink">{formatNumber(first)}</span>–
            <span className="tabular-nums text-ink">{formatNumber(last)}</span> of{" "}
            <span className="tabular-nums text-ink">{formatNumber(total)}</span> {itemLabel}
          </>
        )}
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn-ghost"
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 1}
        >
          <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
          Previous
        </button>
        <span className="tabular-nums">
          Page {formatNumber(page)} of {formatNumber(pageCount)}
        </span>
        <button
          type="button"
          className="btn-ghost"
          onClick={() => onPageChange(page + 1)}
          disabled={page >= pageCount}
        >
          Next
          <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      </div>
    </nav>
  );
}

export interface KeyValueListProps {
  items: Array<{ label: string; value: ReactNode }>;
  className?: string;
  columns?: 1 | 2 | 3;
}

/** The detail grid used inside expanded rows. */
export function KeyValueList({ items, className, columns = 3 }: KeyValueListProps) {
  return (
    <dl
      className={cn(
        "grid gap-x-4 gap-y-3",
        columns === 1 ? "grid-cols-1" : columns === 2 ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
        className,
      )}
    >
      {items.map((item) => (
        <div key={item.label} className="min-w-0">
          <dt className="text-2xs font-medium uppercase tracking-wide text-ink-muted">
            {item.label}
          </dt>
          <dd className="mt-0.5 break-words text-sm text-ink">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A titled block inside an expanded row. */
export function DetailSection({
  title,
  children,
  count,
  className,
}: {
  title: string;
  children: ReactNode;
  count?: number;
  className?: string;
}) {
  return (
    <section className={cn("min-w-0", className)}>
      <h4 className="mb-2 flex items-center gap-2 text-2xs font-semibold uppercase tracking-wide text-ink-secondary">
        {title}
        {count === undefined ? null : (
          <span className="chip border-line text-ink-muted">{count}</span>
        )}
      </h4>
      {children}
    </section>
  );
}
