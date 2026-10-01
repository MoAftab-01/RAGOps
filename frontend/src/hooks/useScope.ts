import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  DEFAULT_TIME_WINDOW,
  isTimeWindow,
  type ScopeParams,
  type TimeWindow,
} from "@/lib/api";

/**
 * The one filter row's state, kept in the URL.
 *
 * `window` and `application` travel as query params so a filtered view is a
 * link someone can paste, and the browser's back button steps through filter
 * changes the way a user expects. Anything unrecognised in the URL falls back
 * to the backend's own default rather than being forwarded — `window=99d` is a
 * 422, and sending it would blank the page for a typo.
 */
export interface Scope {
  timeWindow: TimeWindow;
  application: string | null;
  params: ScopeParams;
  setTimeWindow: (next: TimeWindow) => void;
  setApplication: (next: string | null) => void;
  reset: () => void;
}

/** Reads a page number out of the URL, clamped to something the backend accepts. */
export function usePageParam(defaultPageSize = 25): {
  page: number;
  pageSize: number;
  setPage: (next: number) => void;
} {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawPage = Number(searchParams.get("page") ?? "1");
  const rawSize = Number(searchParams.get("page_size") ?? String(defaultPageSize));
  const page = Number.isFinite(rawPage) && rawPage >= 1 ? Math.floor(rawPage) : 1;
  const pageSize =
    Number.isFinite(rawSize) && rawSize >= 1 ? Math.min(Math.floor(rawSize), 100) : defaultPageSize;

  const setPage = (next: number): void => {
    const params = new URLSearchParams(searchParams);
    const clamped = Math.max(1, Math.floor(next));
    if (clamped === 1) params.delete("page");
    else params.set("page", String(clamped));
    setSearchParams(params, { replace: true });
  };

  return { page, pageSize, setPage };
}

export function useScope(): Scope {
  const [searchParams, setSearchParams] = useSearchParams();

  const rawWindow = searchParams.get("window");
  const rawApplication = searchParams.get("application");
  const timeWindow: TimeWindow = rawWindow !== null && isTimeWindow(rawWindow) ? rawWindow : DEFAULT_TIME_WINDOW;
  const application = rawApplication !== null && rawApplication !== "" ? rawApplication : null;

  const params = useMemo<ScopeParams>(
    () => ({ window: timeWindow, application }),
    [timeWindow, application],
  );

  const update = (mutate: (next: URLSearchParams) => void): void => {
    const next = new URLSearchParams(searchParams);
    mutate(next);
    // Any filter change invalidates the current page number: staying on page 7
    // of a result set that just changed size lands on an empty page.
    next.delete("page");
    setSearchParams(next, { replace: true });
  };

  return {
    timeWindow,
    application,
    params,
    setTimeWindow: (next) => update((params) => params.set("window", next)),
    setApplication: (next) =>
      update((params) => {
        if (next === null || next === "") params.delete("application");
        else params.set("application", next);
      }),
    reset: () =>
      setSearchParams(
        new URLSearchParams(),
        { replace: true },
      ),
  };
}

/** A single page-scoped filter value, in the URL, with a typed setter. */
export function useFilterParam(key: string, initial = ""): {
  value: string;
  setValue: (next: string) => void;
  clear: () => void;
  isSet: boolean;
} {
  const [searchParams, setSearchParams] = useSearchParams();
  const value = searchParams.get(key) ?? initial;

  const mutate = (change: (next: URLSearchParams) => void): void => {
    const next = new URLSearchParams(searchParams);
    change(next);
    // As in `useScope`: a filter change invalidates the page number.
    next.delete("page");
    setSearchParams(next, { replace: true });
  };

  return {
    value,
    setValue: (next) =>
      mutate((params) => {
        if (next === "" || next === initial) params.delete(key);
        else params.set(key, next);
      }),
    clear: () => mutate((params) => params.delete(key)),
    isSet: searchParams.has(key),
  };
}
