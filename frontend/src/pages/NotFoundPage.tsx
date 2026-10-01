import { Link, useLocation } from "react-router-dom";
import { Compass } from "lucide-react";
import { EmptyState } from "@/components/EmptyState";
import { NAV_ITEMS } from "@/components/navItems";

/**
 * The catch-all route.
 *
 * It offers the ten real destinations rather than a dead end, and it echoes the
 * path that did not match — on a client-side router the most common cause is a
 * stale bookmark, and seeing which URL failed is what makes that diagnosable.
 */
export default function NotFoundPage() {
  const location = useLocation();

  return (
    <div className="py-8">
      <EmptyState
        icon={<Compass className="h-6 w-6" aria-hidden="true" />}
        title="No page at this address"
        description={
          <>
            Nothing is routed at{" "}
            <code className="rounded bg-surface-sunken px-1 py-0.5 font-mono text-2xs text-ink">
              {location.pathname}
            </code>
            . It may have been renamed, or the link may predate a route change.
          </>
        }
      />

      <nav aria-label="Available pages" className="mx-auto mt-6 max-w-prose">
        <ul className="grid gap-2 sm:grid-cols-2">
          {NAV_ITEMS.map((item) => (
            <li key={item.to}>
              <Link
                to={item.to}
                className="flex items-center gap-2 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink-secondary transition-colors hover:bg-surface-raised hover:text-ink"
              >
                <item.icon className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden="true" />
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
