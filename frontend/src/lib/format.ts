/**
 * Formatting and metric-direction rules.
 *
 * Two rules run through all of it:
 *
 * 1. **A missing measurement formats as "no value", never as `0`.** Every
 *    numeric formatter takes `number | null | undefined` and returns the
 *    em-dash placeholder for the missing case. A percentile the backend never
 *    computed is a gap in the record, and printing `0 ms` for it would be a
 *    fabricated measurement.
 * 2. **`error_rate` is a fraction, not a percentage.** `backend`'s
 *    `analytics_repo.error_rate()` returns `error_count / total`. Only
 *    `formatFractionAsPercent` turns it into a percentage, and the conversion
 *    happens in exactly one place so it cannot be applied twice.
 */

/** Rendered wherever the API returned no value. */
export const NO_VALUE = "—";

export const isMissing = (value: number | null | undefined): value is null | undefined =>
  value === null || value === undefined || Number.isNaN(value);

const groupDigits = (digits: string): string => digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** Thousands-separated integer, or "no value". */
export function formatNumber(value: number | null | undefined, fractionDigits = 0): string {
  if (isMissing(value)) return NO_VALUE;
  if (!Number.isFinite(value)) return NO_VALUE;
  const fixed = value.toFixed(fractionDigits);
  // `toFixed(0)` emits no fractional part and `toFixed(n > 0)` always does, so
  // the position of the "." is a real distinction rather than a defensive
  // check. It has to be found with `lastIndexOf`: `String.split` is typed as
  // returning `string[]`, not `[string] | [string, string]`, so indexing the
  // result and testing it against `undefined` is a comparison TypeScript
  // considers always false — the separator silently disappears at every
  // decimal precision. Only the integer part is grouped; `groupDigits` works on
  // `\B` lookarounds that match nothing once a "." is present.
  const dot = fixed.lastIndexOf(".");
  if (dot === -1) return groupDigits(fixed);
  if (fractionDigits > 0) return `${groupDigits(fixed.slice(0, dot))}${fixed.slice(dot)}`;
  // `fractionDigits` is 0, so `toFixed` produced no "." after all and the value
  // is exponential — one of the extreme magnitudes `formatCompact` exists to
  // avoid. Returned verbatim rather than mangled by `groupDigits`, which would
  // group digits inside the exponent.
  return fixed;
}

/** Compact count for axis ticks and dense table cells: 1.2k, 3.4M. */
export function formatCompact(value: number | null | undefined): string {
  if (isMissing(value)) return NO_VALUE;
  const magnitude = Math.abs(value);
  if (magnitude >= 1_000_000_000) return `${trimZero(value / 1_000_000_000)}B`;
  if (magnitude >= 1_000_000) return `${trimZero(value / 1_000_000)}M`;
  if (magnitude >= 10_000) return `${trimZero(value / 1_000)}k`;
  if (magnitude >= 1000) return groupDigits(String(Math.round(value)));
  return trimZero(value);
}

const trimZero = (value: number): string => {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
};

/** Integer percent from a value already on a 0–100 scale. */
export function formatPercent(value: number | null | undefined, fractionDigits = 1): string {
  if (isMissing(value)) return NO_VALUE;
  return `${formatNumber(value, fractionDigits)}%`;
}

/**
 * Percent from a fraction in [0, 1], the shape `error_rate` and every
 * retrieval metric arrive in. `0.0234` becomes `2.34%`.
 */
export function formatFractionAsPercent(
  value: number | null | undefined,
  fractionDigits = 1,
): string {
  if (isMissing(value)) return NO_VALUE;
  return `${formatNumber(value * 100, fractionDigits)}%`;
}

/** Milliseconds, switching to seconds past 1000 so p99 does not read `14500ms`. */
export function formatMs(value: number | null | undefined, fractionDigits = 0): string {
  if (isMissing(value)) return NO_VALUE;
  if (Math.abs(value) >= 1000) return `${formatNumber(value / 1000, 2)} s`;
  return `${formatNumber(value, fractionDigits)} ms`;
}

/**
 * A duration in milliseconds, always with one decimal.
 *
 * The precision is deliberate: p50 and p95 are often a few milliseconds apart
 * on a fast local model, and integer rounding would render that gap as two
 * identical numbers. Which percentile a value is comes from the column it sits
 * in, not from the formatter — so there is no percentile argument to pass.
 */
export function formatLatencyMs(value: number | null | undefined): string {
  if (isMissing(value)) return NO_VALUE;
  return `${formatNumber(value, 1)} ms`;
}

/**
 * Cost, labelled with whatever pricing basis the backend used. The label is
 * the backend's, not ours: it distinguishes a metered model from a local one,
 * and the simulated/local distinction on this page is load-bearing.
 */
export function formatCost(
  value: number | null | undefined,
  costLabel?: string | null,
): string {
  if (isMissing(value)) return NO_VALUE;
  const digits = value !== 0 && Math.abs(value) < 0.01 ? 6 : 4;
  const rendered = `$${formatNumber(value, digits)}`;
  return costLabel === null || costLabel === undefined || costLabel === ""
    ? rendered
    : `${rendered} ${costLabel}`;
}

/** A signed relative change, e.g. `-12.4%`. */
export function formatSignedPercent(value: number | null | undefined, fractionDigits = 1): string {
  if (isMissing(value)) return NO_VALUE;
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, fractionDigits)}%`;
}

/** A signed absolute change with an explicit plus sign, for delta tables. */
export function formatSigned(value: number | null | undefined, fractionDigits = 4): string {
  if (isMissing(value)) return NO_VALUE;
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, fractionDigits)}`;
}

/**
 * A score on a 0–1 scale. Retrieval metrics are stored as fractions and are
 * shown as percentages, which is why this is separate from `formatPercent`.
 */
export function formatScore(value: number | null | undefined, fractionDigits = 1): string {
  if (isMissing(value)) return NO_VALUE;
  return formatNumber(value * 100, fractionDigits);
}

/** ISO timestamp → `2026-09-30 14:03 UTC`. Unparseable input is shown as-is. */
export function formatDateTime(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return NO_VALUE;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const pad = (part: number): string => String(part).padStart(2, "0");
  return (
    `${parsed.getUTCFullYear()}-${pad(parsed.getUTCMonth() + 1)}-${pad(parsed.getUTCDate())} ` +
    `${pad(parsed.getUTCHours())}:${pad(parsed.getUTCMinutes())} UTC`
  );
}

/** ISO timestamp → `Sep 30, 14:03`. For axis ticks, where the year is noise. */
export function formatBucket(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return NO_VALUE;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const hours = String(parsed.getUTCHours()).padStart(2, "0");
  const minutes = String(parsed.getUTCMinutes()).padStart(2, "0");
  return `${months[parsed.getUTCMonth()]} ${parsed.getUTCDate()}, ${hours}:${minutes}`;
}

/**
 * Relative time against `now`, for "last seen" columns.
 *
 * `now` is injected rather than read from the clock so the value is testable.
 */
export function formatRelative(
  value: string | null | undefined,
  now: number = Date.now(),
): string {
  if (value === null || value === undefined || value === "") return NO_VALUE;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const seconds = Math.round((now - parsed.getTime()) / 1000);
  if (seconds < 0) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.round(months / 12)}y ago`;
}

/** Any JSON blob rendered as a readable key/value list. Never "[object Object]". */
export function formatJsonValue(value: unknown): string {
  if (value === null || value === undefined) return NO_VALUE;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return "(empty list)";
    return value.map((entry) => formatJsonValue(entry)).join(", ");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return "(empty object)";
    return entries
      .map(([key, item]) => `${key}: ${formatJsonValue(item)}`)
      .join(" · ");
  }
  // Unreachable by construction: the guards above consume `null`, strings,
  // numbers, booleans, arrays and objects. `formatJsonValue` exists so that no
  // caller can ever render "[object Object]", and this line would have been the
  // one place that could.
  return NO_VALUE;
}

/** Human label for a snake_case or kebab-case identifier. */
export function humanize(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return NO_VALUE;
  const spaced = value.replace(/[_-]+/g, " ").replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1).toLowerCase();
}

/** Shortens a long identifier for a dense table cell. */
export function truncate(value: string | null | undefined, max = 12): string {
  if (value === null || value === undefined || value === "") return NO_VALUE;
  return value.length <= max ? value : `${value.slice(0, max - 1)}…`;
}

/* -------------------------------------------------------------------------- */
/* Metric direction                                                           */
/* -------------------------------------------------------------------------- */

export type MetricDirection = "higher_is_better" | "lower_is_better" | "neutral";

/**
 * Whether a rise in a metric is good news, transcribed from
 * `backend/app/evaluation/metrics.py` `METRIC_DIRECTIONS`. Anything absent is
 * `neutral`, which is what the backend itself does — an unrecognised metric has
 * no direction, and colouring it either way would assert something unknown.
 */
const HIGHER_IS_BETTER = new Set([
  "precision_at_k",
  "recall_at_k",
  "f1_at_k",
  "mrr",
  "ndcg_at_k",
  "hit_rate_at_k",
  "precision",
  "recall",
  "f1",
  "rr",
  "reciprocal_rank",
  "ndcg",
  "hit_rate",
  "faithfulness",
  "answer_relevance",
  "context_relevance",
  "citation_coverage",
  "quality_score",
  "token_efficiency",
  "retrieval_score",
]);

const LOWER_IS_BETTER = new Set([
  "estimated_cost",
  "avg_cost_per_1k_tokens",
  "avg_latency_ms",
  "duration_ms",
  "avg_tokens_per_request",
  "cost_per_1k_tokens",
  "cost_per_request",
  "cost_per_user",
  "cost_per_application",
  "total_cost",
  "p50_latency_ms",
  "p95_latency_ms",
  "p99_latency_ms",
  "max_latency_ms",
  "p95_duration_ms",
  "unsupported_claim_ratio",
  "zero_result_rate",
  "error_rate",
  "duplicate_ratio",
  "potential_waste_pct",
  "wasted_tokens",
]);

export function metricDirection(metric: string | null | undefined): MetricDirection {
  if (metric === null || metric === undefined || metric === "") return "neutral";
  const key = metric.trim().toLowerCase();
  if (HIGHER_IS_BETTER.has(key)) return "higher_is_better";
  if (LOWER_IS_BETTER.has(key)) return "lower_is_better";
  return "neutral";
}

export type MetricTone = "good" | "bad" | "flat" | "none";

/**
 * Whether a movement is an improvement, an improvement's opposite, or neither.
 *
 * This is the function the colouring tests pin. The two null cases are
 * deliberately distinct from zero:
 *
 * - `null` change → `none`. Nothing moved, so nothing is coloured. A change of
 *   exactly `0` also returns `none`: a real measured zero *is* flat, and
 *   painting it red would read as a regression that did not happen.
 * - `neutral` metric → `flat` for any non-zero change. The backend records no
 *   direction for it, so a rise is not asserted to be good.
 */
export function changeTone(
  change: number | null | undefined,
  direction: MetricDirection,
): MetricTone {
  if (isMissing(change)) return "none";
  if (direction === "neutral" || change === 0) return "flat";
  const rising = change > 0;
  if (direction === "higher_is_better") return rising ? "good" : "bad";
  return rising ? "bad" : "good";
}

/** Tailwind classes for a tone. Tokens, not raw colours, so theming holds. */
export const TONE_CLASSES: Record<MetricTone, string> = {
  good: "text-status-good",
  bad: "text-status-critical",
  flat: "text-ink-secondary",
  none: "no-value",
};

/** Screen-reader text for a tone, so colour is never the only signal. */
export const TONE_LABEL: Record<MetricTone, string> = {
  good: "improved",
  bad: "regressed",
  flat: "unchanged",
  none: "not measured",
};

/**
 * Tone for a metric value at a given level, where "good" means the value is
 * good in absolute terms (a 95th-percentile latency is bad above any
 * threshold, a zero error rate is good).
 */
export function thresholdTone(
  value: number | null | undefined,
  direction: MetricDirection,
  threshold: number,
): MetricTone {
  if (isMissing(value)) return "none";
  if (value === threshold) return "flat";
  if (direction === "higher_is_better") return value > threshold ? "good" : "bad";
  if (direction === "lower_is_better") return value < threshold ? "good" : "bad";
  return "flat";
}

/** Seconds → `1.4s`, for durations where the millisecond view is noise. */
export function formatSeconds(value: number | null | undefined): string {
  if (isMissing(value)) return NO_VALUE;
  return `${formatNumber(value, 2)}s`;
}
