import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { NO_VALUE } from "@/lib/format";

export interface StatCardProps {
  label: string;
  /** Pre-formatted, or `null` for a measurement the API did not record. */
  value: ReactNode;
  /** Where the number came from and what it is — shown under the value. */
  hint?: ReactNode;
  icon?: ReactNode;
  /** Colours the value as an improvement or a regression. `null` stays neutral. */
  tone?: "good" | "bad" | "flat" | "none";
  className?: string;
}

const TONE_TEXT = {
  good: "text-status-good",
  bad: "text-status-critical",
  flat: "text-ink",
  none: "no-value",
} as const;

/**
 * One number, its label, and where it came from.
 *
 * `value` is passed in already formatted by `src/lib/format.ts`, so the "no
 * value" case is decided in one place. A card never shows a metric it was not
 * given — there is no placeholder here, and `NO_VALUE` renders as an em-dash
 * rather than a zero.
 */
export function StatCard({ label, value, hint, icon, tone = "none", className }: StatCardProps) {
  const isMissing = value === null || value === undefined || value === NO_VALUE;

  return (
    <div className={cn("card flex flex-col justify-between gap-3 p-4", className)}>
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-ink-secondary">{label}</p>
        {icon === undefined ? null : (
          <span className="shrink-0 text-ink-muted" aria-hidden="true">
            {icon}
          </span>
        )}
      </div>
      <p
        className={cn(
          "text-2xl font-semibold tabular-nums leading-tight",
          isMissing ? "no-value" : TONE_TEXT[tone],
        )}
      >
        {isMissing ? NO_VALUE : value}
      </p>
      {hint === undefined ? null : (
        <p className="text-2xs leading-relaxed text-ink-muted">{hint}</p>
      )}
    </div>
  );
}

export interface StatRowProps {
  children: ReactNode;
  className?: string;
}

/** Six cards in the responsive grid every page's stat block uses. */
export function StatRow({ children, className }: StatRowProps) {
  return (
    <div
      className={cn("grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6", className)}
    >
      {children}
    </div>
  );
}
