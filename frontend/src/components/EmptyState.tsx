import type { ReactNode } from "react";
import { Inbox } from "lucide-react";
import { cn } from "@/lib/cn";

export interface EmptyStateProps {
  title: string;
  /** What the absence means, and what would change it. */
  description?: ReactNode;
  /** The way forward: a filter to clear, a window to widen, an action to run. */
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
}

/**
 * The state for an empty array.
 *
 * The description says *why* it is empty and the action offers the way forward,
 * because "no data" and "nothing happened" are different answers and the user
 * cannot tell them apart from an empty table.
 */
export function EmptyState({ title, description, action, icon, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-card border border-dashed border-line px-6 py-10 text-center",
        className,
      )}
    >
      <span className="text-ink-muted" aria-hidden="true">
        {icon ?? <Inbox className="h-6 w-6" />}
      </span>
      <p className="text-sm font-medium text-ink">{title}</p>
      {description === undefined ? null : (
        <p className="max-w-prose text-sm text-ink-secondary">{description}</p>
      )}
      {action === undefined ? null : <div className="mt-2">{action}</div>}
    </div>
  );
}
