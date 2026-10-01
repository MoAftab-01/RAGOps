/**
 * Tailwind theme for the RAGOps console.
 *
 * The colour system lives in `src/index.css` as CSS custom properties (one set
 * per mode) and is re-exported here as semantic Tailwind colours. Recharts
 * cannot read Tailwind classes, so the chart components read the *same*
 * variables directly via `getComputedStyle` — which is what keeps dark mode
 * working without duplicating a palette in JavaScript.
 *
 * The eight series slots and their light/dark steps are the validated
 * categorical palette (see docs/ design notes): adjacent-pair CVD ΔE ≥ 8 and
 * normal-vision ΔE ≥ 15 in both modes against this app's own surfaces.
 *
 * @type {import('tailwindcss').Config}
 */
const withAlpha = (variable) => `rgb(var(${variable}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      colors: {
        surface: {
          DEFAULT: withAlpha("--surface-1"),
          page: withAlpha("--surface-page"),
          raised: withAlpha("--surface-2"),
          sunken: withAlpha("--surface-3"),
        },
        ink: {
          DEFAULT: withAlpha("--text-primary"),
          secondary: withAlpha("--text-secondary"),
          muted: withAlpha("--text-muted"),
          inverse: withAlpha("--text-inverse"),
        },
        line: {
          DEFAULT: withAlpha("--border-subtle"),
          strong: withAlpha("--border-strong"),
          grid: withAlpha("--chart-grid"),
        },
        brand: {
          DEFAULT: withAlpha("--brand"),
          hover: withAlpha("--brand-hover"),
          soft: withAlpha("--brand-soft"),
          fg: withAlpha("--brand-fg"),
        },
        state: {
          good: withAlpha("--status-good"),
          warning: withAlpha("--status-warning"),
          serious: withAlpha("--status-serious"),
          critical: withAlpha("--status-critical"),
        },
        // Fixed categorical slots. Assigned in order, never cycled — a 9th
        // series folds into "Other" rather than inventing a hue.
        series: {
          1: withAlpha("--chart-series-1"),
          2: withAlpha("--chart-series-2"),
          3: withAlpha("--chart-series-3"),
          4: withAlpha("--chart-series-4"),
          5: withAlpha("--chart-series-5"),
          6: withAlpha("--chart-series-6"),
          7: withAlpha("--chart-series-7"),
          8: withAlpha("--chart-series-8"),
        },
      },
      borderRadius: {
        card: "0.75rem",
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "translateY(4px) scale(0.98)" },
          to: { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 150ms ease-out",
        "scale-in": "scale-in 150ms ease-out",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
