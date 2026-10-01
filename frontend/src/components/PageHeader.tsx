import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface PageHeaderProps {
  title: string;
  /** One line on what this page reports and from where. */
  description?: ReactNode;
  /**
   * The filter row, echoed above every panel.
   *
   * A `<div>`, not a `<p>`: every page passes `<ScopeBar />` here, and a filter
   * bar is a group of form controls — putting a `<div>` inside a `<p>` is
   * invalid HTML that React logs on every page load.
   */
  scope?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ title, description, scope, actions, className }: PageHeaderProps) {
  return (
    <header className={cn("flex flex-wrap items-start justify-between gap-3", className)}>
      <div className="min-w-0">
        <h1 className="text-lg font-semibold tracking-tight text-ink">{title}</h1>
        {description === undefined ? null : (
          <p className="mt-1 max-w-prose text-sm text-ink-secondary">{description}</p>
        )}
        {scope === undefined ? null : <div className="mt-1">{scope}</div>}
      </div>
      {actions === undefined ? null : (
        <div className="flex flex-wrap items-center gap-2">{actions}</div>
      )}
    </header>
  );
}

export interface FilterBarProps {
  children: ReactNode;
  /** "Reset" appears whenever anything is set. */
  onReset?: () => void;
  className?: string;
}

/**
 * The single filter row above a page's panels.
 *
 * One row, not a filter per card: charts on the same page must always be
 * describing the same slice of data, and separate filters per card is how a
 * dashboard ends up showing two windows side by side without saying so.
 */
export function FilterBar({ children, onReset, className }: FilterBarProps) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-end gap-3 rounded-card border border-line bg-surface p-3",
        className,
      )}
      role="group"
      aria-label="Filters"
    >
      {children}
      {onReset === undefined ? null : (
        <button type="button" onClick={onReset} className="btn-ghost ml-auto">
          Reset filters
        </button>
      )}
    </div>
  );
}

export interface SelectFieldProps {
  id: string;
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  hint?: ReactNode;
  className?: string;
}

export function SelectField({
  id,
  label,
  value,
  options,
  onChange,
  hint,
  className,
}: SelectFieldProps) {
  return (
    <div className={cn("min-w-0", className)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        className="field-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {hint === undefined ? null : <p className="mt-1 text-2xs text-ink-muted">{hint}</p>}
    </div>
  );
}

export interface TextFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: "text" | "password" | "number";
  hint?: ReactNode;
  className?: string;
  min?: number;
  step?: number;
  autoComplete?: string;
  /**
   * A value the user may select and copy but not edit.
   *
   * Used for the one-time display of a freshly minted credential: the browser
   * has to be able to read the text out to copy it, and it must not look like a
   * field someone is expected to type into.
   */
  readOnly?: boolean;
  /** Overrides the `<label>` text for screen readers without changing what is shown. */
  ariaLabel?: string;
}

export function TextField({
  id,
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  hint,
  className,
  min,
  step,
  autoComplete,
  readOnly = false,
  ariaLabel,
}: TextFieldProps) {
  return (
    <div className={cn("min-w-0", className)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="field-input"
        type={type}
        value={value}
        placeholder={placeholder}
        min={min}
        step={step}
        autoComplete={autoComplete}
        readOnly={readOnly}
        aria-label={ariaLabel}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint === undefined ? null : <p className="mt-1 text-2xs text-ink-muted">{hint}</p>}
    </div>
  );
}

export interface TextAreaFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
  placeholder?: string;
  hint?: ReactNode;
  className?: string;
}

export function TextAreaField({
  id,
  label,
  value,
  onChange,
  rows = 4,
  placeholder,
  hint,
  className,
}: TextAreaFieldProps) {
  return (
    <div className={cn("min-w-0", className)}>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <textarea
        id={id}
        className="field-input scrollbar-thin font-mono text-2xs"
        rows={rows}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint === undefined ? null : <p className="mt-1 text-2xs text-ink-muted">{hint}</p>}
    </div>
  );
}

export interface ToggleFieldProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: ReactNode;
  className?: string;
}

/**
 * A real checkbox, not a styled div: it is keyboard-operable and announced
 * correctly for free, and a switch built from buttons is neither.
 */
export function ToggleField({ id, label, checked, onChange, hint, className }: ToggleFieldProps) {
  return (
    <div className={cn("flex items-start gap-2", className)}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-4 w-4 shrink-0 rounded border-line text-brand focus-visible:ring-2 focus-visible:ring-brand"
      />
      <div className="min-w-0">
        <label htmlFor={id} className="text-sm text-ink">
          {label}
        </label>
        {hint === undefined ? null : <p className="text-2xs text-ink-muted">{hint}</p>}
      </div>
    </div>
  );
}

/** A panel section title with an optional trailing control. */
export function PanelTitle({
  children,
  actions,
  className,
}: {
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-3 flex flex-wrap items-center justify-between gap-2", className)}>
      <h2 className="text-sm font-semibold text-ink">{children}</h2>
      {actions}
    </div>
  );
}

/**
 * A visible, non-dismissible note that qualifies what a number means.
 *
 * Used for the simulated-pricing disclosure, which must never be dismissable
 * and must never be styled like an error — it is a statement about the data,
 * not a failure.
 */
export function InfoNote({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: "neutral" | "warning";
  className?: string;
}) {
  return (
    <p
      className={cn(
        "rounded-md border px-3 py-2 text-2xs leading-relaxed",
        tone === "warning"
          ? "border-status-warning/40 bg-status-warning/5 text-ink-secondary"
          : "border-line bg-surface-sunken text-ink-secondary",
        className,
      )}
    >
      {children}
    </p>
  );
}
