import { Suspense, lazy, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { NavLink, Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import { Activity, Menu, Moon, Sun, X } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useTheme } from "@/hooks/useTheme";
import { LoadingPanel } from "@/components/Spinner";
import { NAV_ITEMS, type NavItem } from "@/components/navItems";

/**
 * Routes are lazy so the first paint is not blocked by eleven pages of charts.
 * Each page is a separate chunk, and the `Suspense` boundary below means a
 * chunk that fails to load shows a recoverable panel instead of a blank
 * document.
 */
const DashboardPage = lazy(() => import("@/pages/DashboardPage"));
const TracesPage = lazy(() => import("@/pages/TracesPage"));
const EvaluationPage = lazy(() => import("@/pages/EvaluationPage"));
const TokenAnalyticsPage = lazy(() => import("@/pages/TokenAnalyticsPage"));
const CostQualityPage = lazy(() => import("@/pages/CostQualityPage"));
const AnomaliesPage = lazy(() => import("@/pages/AnomaliesPage"));
const ExperimentsPage = lazy(() => import("@/pages/ExperimentsPage"));
const OptimizationPage = lazy(() => import("@/pages/OptimizationPage"));
const ApplicationsPage = lazy(() => import("@/pages/ApplicationsPage"));
const SettingsPage = lazy(() => import("@/pages/SettingsPage"));
const NotFoundPage = lazy(() => import("@/pages/NotFoundPage"));

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route
          path="dashboard"
          element={
            <Suspense fallback={<LoadingPanel label="Loading dashboard…" />}>
              <DashboardPage />
            </Suspense>
          }
        />
        <Route
          path="traces"
          element={
            <Suspense fallback={<LoadingPanel label="Loading traces…" />}>
              <TracesPage />
            </Suspense>
          }
        />
        <Route
          path="traces/:traceId"
          element={
            <Suspense fallback={<LoadingPanel label="Loading trace…" />}>
              <TracesPage />
            </Suspense>
          }
        />
        <Route
          path="rag-evaluation"
          element={
            <Suspense fallback={<LoadingPanel label="Loading evaluation…" />}>
              <EvaluationPage />
            </Suspense>
          }
        />
        <Route
          path="token-analytics"
          element={
            <Suspense fallback={<LoadingPanel label="Loading token analytics…" />}>
              <TokenAnalyticsPage />
            </Suspense>
          }
        />
        <Route
          path="cost-quality"
          element={
            <Suspense fallback={<LoadingPanel label="Loading cost & quality…" />}>
              <CostQualityPage />
            </Suspense>
          }
        />
        <Route
          path="anomalies"
          element={
            <Suspense fallback={<LoadingPanel label="Loading anomalies…" />}>
              <AnomaliesPage />
            </Suspense>
          }
        />
        <Route
          path="experiments"
          element={
            <Suspense fallback={<LoadingPanel label="Loading experiments…" />}>
              <ExperimentsPage />
            </Suspense>
          }
        />
        <Route
          path="optimization"
          element={
            <Suspense fallback={<LoadingPanel label="Loading recommendations…" />}>
              <OptimizationPage />
            </Suspense>
          }
        />
        <Route
          path="applications"
          element={
            <Suspense fallback={<LoadingPanel label="Loading applications…" />}>
              <ApplicationsPage />
            </Suspense>
          }
        />
        <Route
          path="settings"
          element={
            <Suspense fallback={<LoadingPanel label="Loading settings…" />}>
              <SettingsPage />
            </Suspense>
          }
        />
        <Route
          path="*"
          element={
            <Suspense fallback={<LoadingPanel />}>
              <NotFoundPage />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}

function AppShell() {
  const { theme, toggleTheme } = useTheme();
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // A route change on a phone should close the drawer, or the new page lands
  // behind it.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-surface-page">
      {/* Desktop sidebar. Hidden below `lg`, where the drawer takes over. */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-line bg-surface lg:flex">
        <SidebarContents onNavigate={() => undefined} />
      </aside>

      {/* Mobile drawer. Rendered only while open so it cannot trap focus. */}
      {navOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-black/40"
            onClick={() => setNavOpen(false)}
            aria-label="Close navigation"
          />
          <aside className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-line bg-surface">
            <SidebarContents onNavigate={() => setNavOpen(false)} />
          </aside>
        </div>
      ) : null}

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-line bg-surface/90 px-4 py-2 backdrop-blur">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              className="btn-ghost lg:hidden"
              onClick={() => setNavOpen((open) => !open)}
              aria-label={navOpen ? "Close navigation" : "Open navigation"}
              aria-expanded={navOpen}
            >
              {navOpen ? <X className="h-4 w-4" aria-hidden="true" /> : <Menu className="h-4 w-4" aria-hidden="true" />}
            </button>
            <Brand />
          </div>
          <div className="flex items-center gap-2">
            <BackendBadge />
            <button
              type="button"
              className="btn-ghost"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" aria-hidden="true" /> : <Moon className="h-4 w-4" aria-hidden="true" />}
              <span className="sr-only sm:not-sr-only">{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
          </div>
        </header>

        <main className="mx-auto w-full max-w-[110rem] px-4 py-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function Brand() {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-brand text-brand-fg">
        <Activity className="h-4 w-4" aria-hidden="true" />
      </span>
      <span className="truncate text-sm font-semibold text-ink">RAGOps</span>
    </span>
  );
}

/**
 * Sidebar body, shared by the fixed rail and the drawer.
 *
 * The nav is grouped rather than flat: ten items in one list stops being
 * scannable, and the groups map to the questions the tool is asked (what is
 * happening, what does it cost, is it healthy, how is it configured).
 */
function SidebarContents({ onNavigate }: { onNavigate: () => void }) {
  const groups = NAV_ITEMS.reduce<Record<string, NavItem[]>>((accumulator, item) => {
    (accumulator[item.group] ??= []).push(item);
    return accumulator;
  }, {});

  return (
    <>
      <div className="flex h-14 shrink-0 items-center border-b border-line px-4">
        <Brand />
      </div>
      <nav className="min-h-0 flex-1 overflow-y-auto scrollbar-thin px-2 py-3" aria-label="Sections">
        {Object.entries(groups).map(([group, items]) => (
          <div key={group} className="mb-4 last:mb-0">
            <p className="px-2 pb-1 text-2xs font-semibold uppercase tracking-wide text-ink-muted">{group}</p>
            <ul className="space-y-0.5">
              {items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                        isActive
                          ? "bg-brand/10 font-medium text-brand"
                          : "text-ink-secondary hover:bg-surface-raised hover:text-ink",
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <item.icon
                          className={cn("h-4 w-4 shrink-0", isActive ? "text-brand" : "text-ink-muted")}
                          aria-hidden="true"
                        />
                        <span className="truncate">{item.label}</span>
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="shrink-0 border-t border-line px-4 py-2.5">
        <p className="text-2xs leading-relaxed text-ink-muted">
          Local build. Token counts and retrieval metrics are computed
          deterministically, not asked of a model.
        </p>
      </div>
    </>
  );
}

/**
 * Health probe in the header.
 *
 * Deliberately thin: it reports only whether the API answered and what
 * environment it claims. Anything richer belongs on Settings, where a failure
 * can be read in full rather than inferred from a coloured dot. A request that
 * fails here is *not* surfaced as a page-level error — a dead backend should
 * not replace the whole UI with an error panel when the last known data is
 * still on screen.
 */
function BackendBadge() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    staleTime: 15_000,
    retry: 0,
  });

  if (health.isPending) {
    return <span className="chip border-line text-ink-muted">checking…</span>;
  }
  if (health.isError) {
    return (
      <span className="chip border-status-critical/40 text-status-critical" title="The API did not respond.">
        API unreachable
      </span>
    );
  }

  const data = health.data;
  // `checks` is always a record (`HealthResponse`), and every value in it is
  // either a dependency result or absent, so there is no null case to branch on.
  const degraded = Object.values(data.checks).some(
    (value) => typeof value === "object" && value !== null && (value as { status?: string }).status === "degraded",
  );

  return (
    <span
      className={cn(
        "chip",
        data.status === "ok" && !degraded
          ? "border-status-good/40 text-status-good"
          : "border-status-warning/40 text-status-warning",
      )}
      title={`${data.status} · ${data.environment} · v${data.version}`}
    >
      {degraded ? "degraded" : data.status}
    </span>
  );
}
