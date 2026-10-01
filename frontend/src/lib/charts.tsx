import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { chartInk, seriesColor, seriesSwatchClass, statusColor } from "@/lib/palette";
import { formatBucket, formatNumber } from "@/lib/format";

/**
 * Chart primitives.
 *
 * Every chart in the app is one of these five, and each one takes a
 * `table` prop of its own so `ChartCard` can offer the table twin. That
 * redundancy is not decoration: three of the light-theme steps sit below the
 * 3:1 contrast bar, and a chart nobody can read is not a chart.
 *
 * Two rules run through all of them:
 *
 * - **Colour comes from `palette.ts`, never from a literal.** `chartInk()`
 *   reads the CSS variables so axes and gridlines follow the theme.
 * - **A series is identified by a legend label, not only by its colour.** Every
 *   two-or-more-series chart renders a legend; `ChartCard` hides it for a single
 *   series, where the title already names it.
 */

const AXIS_TICK = { fontSize: 11, fill: "rgb(var(--text-muted))" } as const;

/** A chart slot: its label, the value key to read, and the colour slot it wears. */
export interface SeriesSpec {
  key: string;
  label: string;
  /** 1-based palette slot. Fixed order, never cycled past eight. */
  slot: number;
  /** Renders a value for tooltips and the table twin. */
  format?: (value: number | null | undefined) => string;
}

/**
 * One point of a time series: an x label plus whatever measures ride along.
 *
 * Generic over the point type rather than typed as `Record<string, unknown>`,
 * because an `interface` has no implicit index signature — a non-generic prop
 * would reject every API type the backend ships.
 */
export type TimeSeriesDatum = object;

interface TimeSeriesChartProps<T> {
  data: T[];
  series: SeriesSpec[];
  height?: number;
  /** Fill under the line. Only sensible for a single series. */
  area?: boolean;
  /** Render as discrete bars rather than a continuous line. */
  bars?: boolean;
  xKey?: string;
  yFormat?: (value: number) => string;
}

export function TimeSeriesChart<T>({
  data,
  series,
  height = 220,
  area = false,
  bars = false,
  xKey = "bucket",
  yFormat = (value) => formatNumber(value),
}: TimeSeriesChartProps<T>) {
  const ink = chartInk();
  const axisLabel = series[0]?.label ?? "value";

  if (bars) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={ink.grid} vertical={false} />
          <XAxis dataKey={xKey} tick={AXIS_TICK} tickFormatter={formatBucket} tickLine={false} axisLine={{ stroke: ink.grid }} />
          <YAxis tick={AXIS_TICK} tickFormatter={yFormat} tickLine={false} axisLine={false} width={48} />
          <Tooltip content={<ChartTooltip formatter={series[0]?.format} />} cursor={{ fill: ink.grid, fillOpacity: 0.4 }} />
          {series.map((entry) => (
            <Bar
              key={entry.key}
              dataKey={entry.key}
              name={entry.label}
              fill={seriesColor(entry.slot)}
              radius={[2, 2, 0, 0]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  const Chart = area ? AreaChart : LineChart;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <Chart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={ink.grid} vertical={false} />
        <XAxis dataKey={xKey} tick={AXIS_TICK} tickFormatter={formatBucket} tickLine={false} axisLine={{ stroke: ink.grid }} />
        <YAxis
          tick={AXIS_TICK}
          tickFormatter={yFormat}
          tickLine={false}
          axisLine={false}
          width={48}
          label={series.length > 1 ? undefined : { value: axisLabel, angle: -90, position: "insideLeft", fontSize: 10, fill: ink.muted }}
        />
        <Tooltip content={<ChartTooltip formatter={series[0]?.format} />} />
        {series.map((entry) =>
          area ? (
            <Area
              key={entry.key}
              type="monotone"
              dataKey={entry.key}
              name={entry.label}
              stroke={seriesColor(entry.slot)}
              fill={seriesColor(entry.slot)}
              fillOpacity={0.15}
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls
            />
          ) : (
            <Line
              key={entry.key}
              type="monotone"
              dataKey={entry.key}
              name={entry.label}
              stroke={seriesColor(entry.slot)}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
              connectNulls
            />
          ),
        )}
      </Chart>
    </ResponsiveContainer>
  );
}

interface CategoryBarChartProps {
  data: Array<{ label: string; value: number | null | undefined }>;
  height?: number;
  valueLabel: string;
  format?: (value: number | null | undefined) => string;
  /** Colour every bar one status token instead of the categorical slot 1. */
  tone?: "good" | "warning" | "serious" | "critical" | null;
  /** Cap the number of bars drawn; the rest are folded into "Other". */
  maxBars?: number;
}

export function CategoryBarChart({
  data,
  height = 220,
  valueLabel,
  format = (value) => formatNumber(value),
  tone = null,
  maxBars = 10,
}: CategoryBarChartProps) {
  const ink = chartInk();
  // Truncation has to be visible. A silently capped bar chart reads as a
  // complete one, which is the exact failure the "no silent caps" rule is about.
  const prepared = useMemo(() => {
    const sorted = [...data].sort((left, right) => (right.value ?? 0) - (left.value ?? 0));
    if (sorted.length <= maxBars) return { rows: sorted, hidden: 0 };
    const head = sorted.slice(0, maxBars - 1);
    const tail = sorted.slice(maxBars - 1);
    return {
      rows: [...head, { label: `Other (${tail.length})`, value: tail.reduce((sum, row) => sum + (row.value ?? 0), 0) }],
      hidden: tail.length,
    };
  }, [data, maxBars]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={prepared.rows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={ink.grid} horizontal={false} />
        <XAxis type="number" tick={AXIS_TICK} tickFormatter={format} tickLine={false} axisLine={false} />
        <YAxis
          type="category"
          dataKey="label"
          tick={AXIS_TICK}
          tickLine={false}
          axisLine={false}
          width={140}
          interval={0}
        />
        <Tooltip content={<ChartTooltip formatter={format} />} cursor={{ fill: ink.grid, fillOpacity: 0.4 }} />
        <Bar dataKey="value" name={valueLabel} radius={[0, 2, 2, 0]} isAnimationActive={false}>
          {prepared.rows.map((row, index) => (
            <Cell
              key={row.label}
              fill={tone === null ? seriesColor((index % 8) + 1) : statusColor(tone)}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export interface ScatterDatum {
  x: number;
  y: number | null | undefined;
  label: string;
  /** Optional third dimension, drawn as mark size. */
  weight?: number;
  highlight?: boolean;
}

interface ScatterPlotProps {
  data: ScatterDatum[];
  xLabel: string;
  yLabel: string;
  xFormat: (value: number) => string;
  yFormat: (value: number) => string;
  height?: number;
}

/**
 * Cost against quality.
 *
 * Capped at three colours by the palette contract (`MAX_ALL_PAIRS`): a scatter
 * places marks next to each other in arbitrary combination rather than in a
 * fixed adjacent order, so only the first three slots are contrast-validated for
 * every pair. The default colour is a single ink tone and `highlight` puts a
 * status token on the point worth looking at, which is the one distinction this
 * chart actually needs.
 */
export function ScatterPlot({
  data,
  xLabel,
  yLabel,
  xFormat,
  yFormat,
  height = 280,
}: ScatterPlotProps) {
  const ink = chartInk();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 12, left: 0 }}>
        <CartesianGrid stroke={ink.grid} />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tick={AXIS_TICK}
          tickFormatter={xFormat}
          tickLine={false}
          axisLine={{ stroke: ink.grid }}
          label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 10, fill: ink.muted }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={AXIS_TICK}
          tickFormatter={yFormat}
          tickLine={false}
          axisLine={false}
          width={48}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 10, fill: ink.muted }}
        />
        <ZAxis type="number" dataKey="weight" range={[80, 400]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3", stroke: ink.muted }}
          content={({ active, payload }) => {
            const point = payload?.[0]?.payload as ScatterDatum | undefined;
            if (active !== true || point === undefined) return null;
            return (
              <div className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-sm">
                <p className="font-medium text-ink">{point.label}</p>
                <p className="tabular-nums text-ink-secondary">
                  {xLabel}: {xFormat(point.x)} · {yLabel}: {yFormat(point.y ?? 0)}
                </p>
              </div>
            );
          }}
        />
        <Scatter
          data={data}
          isAnimationActive={false}
          shape={(props: { cx?: number; cy?: number; payload?: ScatterDatum }) => {
            const { cx = 0, cy = 0, payload } = props;
            const base = payload?.highlight === true ? statusColor("good") : seriesColor(1);
            return <circle cx={cx} cy={cy} r={5} fill={base} fillOpacity={0.85} stroke={ink.surface} strokeWidth={1.5} />;
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

interface LegendRowProps {
  series: SeriesSpec[];
}

/** Standalone legend, for charts mounted outside a `ChartCard`. */
export function LegendRow({ series }: LegendRowProps) {
  if (series.length < 2) return null;
  return (
    <ul className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
      {series.map((entry) => (
        <li key={entry.key} className="flex items-center gap-1.5 text-2xs text-ink-secondary">
          <span className={cnSwatch(entry.slot)} style={{ backgroundColor: seriesColor(entry.slot) }} aria-hidden="true" />
          {entry.label}
        </li>
      ))}
    </ul>
  );
}

function cnSwatch(slot: number): string {
  return `h-2 w-2 shrink-0 rounded-[2px] ${seriesSwatchClass(slot)}`;
}

/* -------------------------------------------------------------------------- */
/* Tooltip                                                                    */
/* -------------------------------------------------------------------------- */

interface TooltipEntry {
  name?: string | number;
  value?: number | string | null;
  color?: string;
  dataKey?: string | number;
}

function ChartTooltip({
  active,
  payload,
  label,
  formatter,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
  formatter?: (value: number | null | undefined) => string;
}) {
  if (active !== true || payload === undefined || payload.length === 0) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-sm">
      {label === undefined ? null : (
        <p className="mb-1 font-medium text-ink">{formatBucket(String(label))}</p>
      )}
      <ul className="space-y-0.5">
        {payload.map((entry) => (
          <li key={String(entry.dataKey ?? entry.name)} className="flex items-center gap-2 text-ink-secondary">
            <span
              className="h-2 w-2 shrink-0 rounded-[2px]"
              style={{ backgroundColor: entry.color ?? seriesColor(1) }}
              aria-hidden="true"
            />
            <span>{String(entry.name ?? entry.dataKey ?? "")}</span>
            <span className="tabular-nums text-ink">
              {typeof entry.value === "number"
                ? formatter === undefined
                  ? formatNumber(entry.value)
                  : formatter(entry.value)
                : String(entry.value ?? "—")}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Re-exported so pages do not import `recharts` directly. */
export { Legend };
