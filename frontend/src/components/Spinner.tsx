import { cn } from "@/lib/cn";

export interface SpinnerProps {
  className?: string;
  /** Announced to screen readers when the spinner stands alone. */
  label?: string;
}

/** A 2px ring. Used inline inside buttons and beside text. */
export function Spinner({ className, label }: SpinnerProps) {
  return (
    <span
      className={cn(
        "inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent align-[-0.125em]",
        className,
      )}
      role={label === undefined ? "status" : undefined}
      aria-hidden={label === undefined ? true : undefined}
      aria-label={label}
    />
  );
}

/** Centred spinner for a panel that has no content yet. */
export function LoadingPanel({ label = "Loading…" }: { label?: string }) {
  return (
    <div
      className="flex min-h-[10rem] flex-col items-center justify-center gap-3 py-12 text-ink-muted"
      role="status"
      aria-live="polite"
    >
      <span className="h-6 w-6 animate-spin rounded-full border-2 border-current border-r-transparent" />
      <span className="text-sm">{label}</span>
    </div>
  );
}

/** Skeleton block. Never used where a refetch is in flight over stale data. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-surface-sunken", className)}
      aria-hidden="true"
    />
  );
}

/** Card-shaped skeletons for a dashboard's first paint. */
export function SkeletonCards({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: count }, (_unused, index) => (
        <div key={index} className="card p-4">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="mt-3 h-7 w-32" />
          <Skeleton className="mt-3 h-3 w-40" />
        </div>
      ))}
    </div>
  );
}
