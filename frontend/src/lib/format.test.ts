/**
 * §27: unit tests for the formatters.
 *
 * The rule under test throughout is the one in `format.ts`'s own header: **a
 * missing measurement formats as "no value", never as `0`.** Every numeric
 * formatter therefore gets a `null`, an `undefined` and a `NaN` case, because
 * those three are the ways the backend says "never measured" and a regression in
 * any one of them is a fabricated measurement on screen.
 *
 * The numbers below are hand-computed rather than captured from the
 * implementation — a test that asserts whatever the code printed pins nothing.
 */

import { describe, expect, it } from "vitest";
import {
  NO_VALUE,
  TONE_CLASSES,
  TONE_LABEL,
  changeTone,
  formatBucket,
  formatCompact,
  formatCost,
  formatDateTime,
  formatFractionAsPercent,
  formatJsonValue,
  formatLatencyMs,
  formatMs,
  formatNumber,
  formatPercent,
  formatRelative,
  formatScore,
  formatSeconds,
  formatSigned,
  formatSignedPercent,
  humanize,
  isMissing,
  metricDirection,
  thresholdTone,
  truncate,
  type MetricTone,
} from "@/lib/format";

describe("isMissing", () => {
  it("treats null, undefined and NaN as missing but not zero", () => {
    expect(isMissing(null)).toBe(true);
    expect(isMissing(undefined)).toBe(true);
    expect(isMissing(Number.NaN)).toBe(true);
    // 0 is a measurement. This is the distinction the whole module rests on.
    expect(isMissing(0)).toBe(false);
  });
});

describe("formatNumber", () => {
  it("groups thousands", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(7)).toBe("7");
    expect(formatNumber(999)).toBe("999");
    expect(formatNumber(1000)).toBe("1,000");
    expect(formatNumber(1047296)).toBe("1,047,296");
  });

  it("does not group the fractional part", () => {
    expect(formatNumber(1234.5678, 2)).toBe("1,234.57");
  });

  /**
   * The separator has to survive at *every* decimal precision.
   *
   * `formatNumber` once split the fixed string on "." and tested the second
   * element against `undefined`. TypeScript types `String.split` as returning
   * `string[]` rather than a two-tuple, so that comparison was always false and
   * the branch it guarded was dead — which meant the thousands separator was
   * dropped at every precision except zero. Nothing caught it: the zero-decimal
   * case, the only one the old code handled, was already covered above.
   *
   * Every value here is computed by hand from what `toFixed` produces, not read
   * back from the function's own output.
   */
  it("groups the integer part at every decimal precision", () => {
    expect(formatNumber(1234567, 2)).toBe("1,234,567.00");
    expect(formatNumber(1234.5, 3)).toBe("1,234.500");
    expect(formatNumber(1000, 1)).toBe("1,000.0");
    expect(formatNumber(1.23456789, 8)).toBe("1.23456789");
    expect(formatNumber(0.0234, 2)).toBe("0.02");
  });

  it("groups a negative magnitude without moving the sign", () => {
    expect(formatNumber(-1234567, 0)).toBe("-1,234,567");
    expect(formatNumber(-1234567, 2)).toBe("-1,234,567.00");
  });

  it("returns an exponential value verbatim rather than grouping the exponent", () => {
    // At this magnitude `toFixed` switches to exponential form. Running the
    // separator over "1e+21" would produce nonsense, so this is passed through.
    expect(formatNumber(1e21, 0)).toBe("1e+21");
  });

  it("rounds rather than truncating", () => {
    expect(formatNumber(0.005, 2)).toBe("0.01");
    expect(formatNumber(2.675, 2)).toBe("2.67"); // toFixed's half-to-even-ish behaviour
  });

  it("renders a missing value as the placeholder", () => {
    expect(formatNumber(null)).toBe(NO_VALUE);
    expect(formatNumber(undefined)).toBe(NO_VALUE);
    expect(formatNumber(Number.NaN)).toBe(NO_VALUE);
  });

  it("renders a non-finite value as the placeholder rather than Infinity", () => {
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe(NO_VALUE);
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe(NO_VALUE);
  });
});

describe("formatCompact", () => {
  it("abbreviates only above the documented thresholds", () => {
    expect(formatCompact(999)).toBe("999");
    expect(formatCompact(1000)).toBe("1,000");
    expect(formatCompact(12_345)).toBe("12.3k");
    expect(formatCompact(3_400_000)).toBe("3.4M");
    expect(formatCompact(1_500_000_000)).toBe("1.5B");
  });

  it("drops the decimal when the tenth is zero", () => {
    expect(formatCompact(10_000)).toBe("10k");
    expect(formatCompact(10_500)).toBe("10.5k");
  });

  it("renders a missing value as the placeholder", () => {
    expect(formatCompact(null)).toBe(NO_VALUE);
  });
});

describe("formatPercent and formatFractionAsPercent", () => {
  it("treats formatPercent's input as already scaled to 0–100", () => {
    expect(formatPercent(18.42)).toBe("18.4%");
    expect(formatPercent(18.42, 2)).toBe("18.42%");
    expect(formatPercent(0)).toBe("0.0%");
  });

  it("multiplies a fraction by 100", () => {
    // `error_rate` arrives as error_count / total, so 0.0234 is 2.34%.
    expect(formatFractionAsPercent(0.0234, 2)).toBe("2.34%");
    expect(formatFractionAsPercent(1)).toBe("100.0%");
    expect(formatFractionAsPercent(0)).toBe("0.0%");
  });

  it("renders a missing value as the placeholder in both", () => {
    expect(formatPercent(null)).toBe(NO_VALUE);
    expect(formatFractionAsPercent(null)).toBe(NO_VALUE);
    expect(formatFractionAsPercent(undefined)).toBe(NO_VALUE);
  });
});

describe("formatMs and formatLatencyMs", () => {
  it("switches to seconds past 1000 ms so p99 does not read 14500ms", () => {
    expect(formatMs(450)).toBe("450 ms");
    expect(formatMs(999)).toBe("999 ms");
    expect(formatMs(1000)).toBe("1.00 s");
    expect(formatMs(14_500)).toBe("14.50 s");
  });

  it("keeps one decimal in formatLatencyMs so p50 and p95 stay distinguishable", () => {
    expect(formatLatencyMs(12)).toBe("12.0 ms");
    expect(formatLatencyMs(12.4)).toBe("12.4 ms");
  });

  it("renders a missing value as the placeholder in both", () => {
    expect(formatMs(null)).toBe(NO_VALUE);
    expect(formatLatencyMs(undefined)).toBe(NO_VALUE);
  });
});

describe("formatCost", () => {
  it("appends the backend's cost label when there is one", () => {
    expect(formatCost(1.2345, "simulated")).toBe("$1.2345 simulated");
    expect(formatCost(1.2345, null)).toBe("$1.2345");
    expect(formatCost(1.2345, "")).toBe("$1.2345");
  });

  it("widens precision for a fractional cent rather than rounding it to zero", () => {
    // A local Ollama call costs a fraction of a cent; $0.0000 would read as free.
    expect(formatCost(0.0000123)).toBe("$0.000012");
    expect(formatCost(0)).toBe("$0.0000");
    expect(formatCost(2)).toBe("$2.0000");
  });

  it("renders a missing value as the placeholder", () => {
    expect(formatCost(null, "simulated")).toBe(NO_VALUE);
  });
});

describe("formatSignedPercent and formatSigned", () => {
  it("marks a positive change explicitly and leaves a negative one alone", () => {
    expect(formatSignedPercent(12.4)).toBe("+12.4%");
    expect(formatSignedPercent(-12.4)).toBe("-12.4%");
    expect(formatSignedPercent(0)).toBe("0.0%");
  });

  it("signs absolute changes for delta tables", () => {
    expect(formatSigned(0.0231)).toBe("+0.0231");
    expect(formatSigned(-0.0231)).toBe("-0.0231");
    expect(formatSigned(null)).toBe(NO_VALUE);
  });
});

describe("formatScore", () => {
  it("multiplies a 0–1 metric by 100", () => {
    expect(formatScore(0.8234)).toBe("82.3");
    expect(formatScore(0.8234, 3)).toBe("82.340");
    expect(formatScore(0)).toBe("0.0");
    expect(formatScore(1)).toBe("100.0");
  });

  it("does not append a percent sign — the column supplies the unit", () => {
    expect(formatScore(0.5).includes("%")).toBe(false);
  });

  it("renders a missing score as the placeholder rather than 0.0", () => {
    // The failure this guards: an unmeasured NDCG drawn as 0.0% would read as a
    // measured zero, which is a much worse claim than "no value".
    expect(formatScore(null)).toBe(NO_VALUE);
    expect(formatScore(undefined)).toBe(NO_VALUE);
  });
});

describe("formatSeconds", () => {
  it("renders seconds with two decimals", () => {
    expect(formatSeconds(1.44)).toBe("1.44s");
    expect(formatSeconds(null)).toBe(NO_VALUE);
  });
});

describe("formatDateTime", () => {
  it("renders ISO input in UTC, not local time", () => {
    // 14:03 UTC must not become 19:03 on a UTC+05:00 host — the console is
    // read next to a server log, which is UTC.
    expect(formatDateTime("2026-09-30T14:03:22Z")).toBe("2026-09-30 14:03 UTC");
    expect(formatDateTime("2026-09-30T14:03:22+00:00")).toBe("2026-09-30 14:03 UTC");
  });

  it("zero-pads single-digit months and days", () => {
    expect(formatDateTime("2026-01-02T03:04:00Z")).toBe("2026-01-02 03:04 UTC");
  });

  it("renders a missing value as the placeholder", () => {
    expect(formatDateTime(null)).toBe(NO_VALUE);
    expect(formatDateTime("")).toBe(NO_VALUE);
  });

  it("shows unparseable input verbatim rather than an invented date", () => {
    expect(formatDateTime("not a date")).toBe("not a date");
  });
});

describe("formatBucket", () => {
  it("drops the year for axis ticks", () => {
    expect(formatBucket("2026-09-30T14:03:00Z")).toBe("Sep 30, 14:03");
    expect(formatBucket("2026-01-02T03:04:00Z")).toBe("Jan 2, 03:04");
  });
});

describe("formatRelative", () => {
  // `now` is injected rather than read from the clock, so these are exact.
  const now = Date.parse("2026-09-30T12:00:00Z");

  it("walks the unit ladder", () => {
    expect(formatRelative("2026-09-30T11:59:30Z", now)).toBe("30s ago");
    expect(formatRelative("2026-09-30T11:45:00Z", now)).toBe("15m ago");
    expect(formatRelative("2026-09-30T09:00:00Z", now)).toBe("3h ago");
    expect(formatRelative("2026-09-27T12:00:00Z", now)).toBe("3d ago");
    expect(formatRelative("2026-08-01T12:00:00Z", now)).toBe("2mo ago");
    expect(formatRelative("2024-09-30T12:00:00Z", now)).toBe("2y ago");
  });

  it("clamps a future timestamp to 'just now' rather than a negative age", () => {
    expect(formatRelative("2026-09-30T12:05:00Z", now)).toBe("just now");
  });

  it("renders a never-seen application as the placeholder", () => {
    expect(formatRelative(null, now)).toBe(NO_VALUE);
  });
});

describe("formatJsonValue", () => {
  it("never produces [object Object]", () => {
    expect(formatJsonValue({ top_k: 10, rerank: true })).toBe("top_k: 10 · rerank: true");
  });

  it("labels empty containers instead of printing nothing", () => {
    expect(formatJsonValue({})).toBe("(empty object)");
    expect(formatJsonValue([])).toBe("(empty list)");
  });

  it("passes scalars and nullish values through", () => {
    expect(formatJsonValue("text")).toBe("text");
    expect(formatJsonValue(42)).toBe("42");
    expect(formatJsonValue(false)).toBe("false");
    expect(formatJsonValue(null)).toBe(NO_VALUE);
  });
});

describe("humanize", () => {
  it("turns snake_case and kebab-case into a sentence", () => {
    expect(humanize("p95_latency_ms")).toBe("P95 latency ms");
    expect(humanize("higher-is-better")).toBe("Higher is better");
    expect(humanize("camelCase")).toBe("Camel case");
  });

  it("renders an empty value as the placeholder", () => {
    expect(humanize(null)).toBe(NO_VALUE);
    expect(humanize("")).toBe(NO_VALUE);
  });
});

describe("truncate", () => {
  it("leaves a short value alone and ellipsises a long one", () => {
    expect(truncate("abc", 12)).toBe("abc");
    expect(truncate("0123456789abcdef", 8)).toBe("0123456…");
  });
});

describe("metricDirection", () => {
  it("knows which retrieval metrics reward a rise", () => {
    expect(metricDirection("recall_at_k")).toBe("higher_is_better");
    expect(metricDirection("mrr")).toBe("higher_is_better");
    expect(metricDirection("ndcg_at_k")).toBe("higher_is_better");
    expect(metricDirection("faithfulness")).toBe("higher_is_better");
  });

  it("knows which cost and latency metrics reward a fall", () => {
    expect(metricDirection("p95_latency_ms")).toBe("lower_is_better");
    expect(metricDirection("estimated_cost")).toBe("lower_is_better");
    expect(metricDirection("error_rate")).toBe("lower_is_better");
    expect(metricDirection("wasted_tokens")).toBe("lower_is_better");
  });

  it("is case-insensitive", () => {
    expect(metricDirection("RECALL_AT_K")).toBe("higher_is_better");
    expect(metricDirection("  mrr  ")).toBe("higher_is_better");
  });

  it("returns neutral for a metric it has never heard of", () => {
    // Asserting a direction for an unknown metric would be inventing a fact.
    expect(metricDirection("some_vendor_score")).toBe("neutral");
    expect(metricDirection(null)).toBe("neutral");
    expect(metricDirection("")).toBe("neutral");
  });
});

describe("changeTone", () => {
  it("separates 'nothing moved' from 'not measured'", () => {
    expect(changeTone(null, "higher_is_better")).toBe("none");
    expect(changeTone(undefined, "lower_is_better")).toBe("none");
    expect(changeTone(Number.NaN, "neutral")).toBe("none");
    // A measured zero is flat, not a regression.
    expect(changeTone(0, "higher_is_better")).toBe("flat");
    expect(changeTone(0, "lower_is_better")).toBe("flat");
  });

  it("colours a rise in a higher-is-better metric as good", () => {
    expect(changeTone(0.05, "higher_is_better")).toBe("good");
    expect(changeTone(-0.05, "higher_is_better")).toBe("bad");
  });

  it("colours a fall in a lower-is-better metric as good", () => {
    expect(changeTone(-0.05, "lower_is_better")).toBe("good");
    expect(changeTone(0.05, "lower_is_better")).toBe("bad");
  });

  it("draws any movement in an unrecorded direction flat, not optimistic", () => {
    // The point of `neutral`: the backend records no direction, so a rise is not
    // asserted to be good news.
    expect(changeTone(0.05, "neutral")).toBe("flat");
    expect(changeTone(-0.05, "neutral")).toBe("flat");
  });
});

describe("thresholdTone", () => {
  it("judges a value against a threshold in the metric's own direction", () => {
    expect(thresholdTone(0.9, "higher_is_better", 0.8)).toBe("good");
    expect(thresholdTone(0.7, "higher_is_better", 0.8)).toBe("bad");
    expect(thresholdTone(100, "lower_is_better", 200)).toBe("good");
    expect(thresholdTone(300, "lower_is_better", 200)).toBe("bad");
  });

  it("treats a value exactly on the threshold as flat", () => {
    expect(thresholdTone(0.8, "higher_is_better", 0.8)).toBe("flat");
  });

  it("renders an unmeasured value as not measured", () => {
    expect(thresholdTone(null, "higher_is_better", 0.8)).toBe("none");
  });
});

describe("TONE_CLASSES and TONE_LABEL", () => {
  const tones: MetricTone[] = ["good", "bad", "flat", "none"];

  it("covers every tone with a class", () => {
    for (const tone of tones) expect(TONE_CLASSES[tone]).toBeTruthy();
  });

  it("covers every tone with a screen-reader label, so colour is not the only signal", () => {
    for (const tone of tones) expect(TONE_LABEL[tone]).toBeTruthy();
    // The labels must be distinct words, not one placeholder reused.
    expect(new Set(tones.map((tone) => TONE_LABEL[tone])).size).toBe(tones.length);
  });
});
