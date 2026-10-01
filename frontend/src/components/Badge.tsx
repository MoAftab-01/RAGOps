import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { severityColor } from "@/lib/palette";

export type BadgeTone = "neutral" | "good" | "warning" | "serious" | "critical" | "brand";

const TONE_CLASS: Record<BadgeTone, string> = {
  neutral: "border-line text-ink-secondary",
  good: "border-status-good/40 text-status-good",
  warning: "border-status-warning/40 text-status-warning",
  serious: "border-status-serious/40 text-status-serious",
  critical: "border-status-critical/40 text-status-critical",
  brand: "border-brand/40 text-brand",
};

export interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
  /** A colour swatch instead of a text tone — for series identity in legends. */
  swatch?: string;
  title?: string;
}

/**
 * A small labelled chip.
 *
 * The label is always present, so the state is never carried by colour alone —
 * which is what makes the status tones legal here.
 */
export function Badge({ children, tone = "neutral", className, swatch, title }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-2xs font-medium",
        TONE_CLASS[tone],
        className,
      )}
      title={title}
    >
      {swatch === undefined ? null : (
        <span
          className="h-2 w-2 shrink-0 rounded-[2px]"
          style={{ backgroundColor: swatch }}
          aria-hidden="true"
        />
      )}
      {children}
    </span>
  );
}

/**
 * Maps a backend status word onto a tone. An unrecognised status gets
 * `neutral` and keeps its text, rather than being forced into a colour.
 */
export function statusTone(status: string | null | undefined): BadgeTone {
  switch ((status ?? "").toLowerCase()) {
    case "success":
    case "completed":
    case "ok":
    case "healthy":
    case "low":
    case "resolved":
      return "good";
    case "running":
    case "pending":
    case "not_persisted":
      return "warning";
    case "error":
    case "failed":
    case "cancelled":
    case "medium":
      return "serious";
    case "high":
    case "critical":
      return "critical";
    case "active":
      return "brand";
    default:
      return "neutral";
  }
}

/** A badge carrying the severity swatch the anomaly and recommendation lists use. */
export function SeverityBadge({ severity }: { severity: string | null | undefined }) {
  return (
    <Badge tone={statusTone(severity)} swatch={severityColor(severity)}>
      {(severity ?? "unknown").toLowerCase()}
    </Badge>
  );
}
