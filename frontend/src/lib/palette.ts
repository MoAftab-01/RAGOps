/**
 * Chart colour, read from the CSS variables `--viz-1..8`.
 *
 * Reading through `getComputedStyle` is what lets one palette serve both
 * themes: the values live in `index.css` per mode and the components never hold
 * a colour of their own.
 *
 * The steps behind `--viz-*` were put through the data-viz validator, and two of
 * its results are load-bearing here:
 *
 * - Slots are assigned in **fixed order and never cycled**. A ninth series is
 *   not a generated hue — callers fold it into "Other" (`MAX_CATEGORICAL`).
 * - Only the **first three slots** validate all-pairs, so scatter and heatmap,
 *   whose marks sit next to each other in arbitrary combination rather than in
 *   a fixed adjacent order, cap at three (`MAX_ALL_PAIRS`).
 *
 * Three of the light steps sit below 3:1 against the light surface. That is a
 * WARN rather than a FAIL and it is not dismissable: it obliges relief, which
 * here is the always-available table view on every `ChartCard` plus the
 * visible legend labels. Neither is optional decoration.
 */

export const MAX_CATEGORICAL = 8;
export const MAX_ALL_PAIRS = 3;

const SLOT_COUNT = 8;

/** The `--viz-N` variable for a 1-based slot, as `R G B` or `r, g, b`. */
function readSlot(slot: number): string {
  const index = ((slot - 1) % SLOT_COUNT) + 1;
  if (typeof window === "undefined" || typeof getComputedStyle !== "function") {
    return FALLBACK[index - 1];
  }
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(`--viz-${index}`)
    .trim();
  if (raw === "") return FALLBACK[index - 1];
  return raw.includes(",") ? raw : raw.replace(/\s+/g, ", ");
}

/** Used when there is no DOM to measure (SSR, or a bare unit test). */
const FALLBACK = [
  "57, 135, 229",
  "217, 89, 38",
  "25, 158, 112",
  "201, 133, 0",
  "213, 81, 129",
  "0, 131, 0",
  "144, 133, 233",
  "230, 103, 103",
];

/** The categorical colour for a 1-based slot. */
export function seriesColor(slot: number): string {
  return `rgb(${readSlot(slot)})`;
}

/**
 * The whole palette, in slot order. Callers that need more than eight series
 * must fold the tail into "Other" rather than asking for slot nine.
 */
export function seriesPalette(count: number = MAX_CATEGORICAL): string[] {
  const total = Math.min(Math.max(count, 0), MAX_CATEGORICAL);
  return Array.from({ length: total }, (_unused, index) => seriesColor(index + 1));
}

/** Tailwind text token for a slot, so a legend label can wear its own colour. */
export function seriesSwatchClass(slot: number): string {
  const classes = [
    "text-[rgb(var(--viz-1))]",
    "text-[rgb(var(--viz-2))]",
    "text-[rgb(var(--viz-3))]",
    "text-[rgb(var(--viz-4))]",
    "text-[rgb(var(--viz-5))]",
    "text-[rgb(var(--viz-6))]",
    "text-[rgb(var(--viz-7))]",
    "text-[rgb(var(--viz-8))]",
  ];
  const index = ((slot - 1) % SLOT_COUNT) + 1;
  return classes[index - 1];
}

/** Ink tokens, so axes and gridlines follow the theme rather than a constant. */
export function chartInk() {
  return {
    primary: "rgb(var(--text-primary))",
    secondary: "rgb(var(--text-secondary))",
    muted: "rgb(var(--text-muted))",
    grid: "rgb(var(--chart-grid))",
    surface: "rgb(var(--surface-1))",
  };
}

/** Status tokens, reserved for state and never reused as a series colour. */
export function statusColor(state: "good" | "warning" | "serious" | "critical"): string {
  return `rgb(var(--status-${state}))`;
}

/** Resolves a severity word from the backend onto a status token. */
export function severityColor(severity: string | null | undefined): string {
  switch ((severity ?? "").toLowerCase()) {
    case "critical":
    case "high":
      return statusColor("critical");
    case "serious":
    case "medium":
      return statusColor("serious");
    case "warning":
      return statusColor("warning");
    case "good":
    case "low":
      return statusColor("good");
    default:
      return chartInk().muted;
  }
}
