/**
 * A render harness for pages that read the URL.
 *
 * Every page in this console takes its filters from the query string
 * (`useScope`), so a test that renders one has to provide a router for the hooks
 * to suspend against. Two things live here:
 *
 * - `renderPage` wraps a page in a `MemoryRouter` at a chosen URL and a fresh
 *   `QueryClient` with retries off, so a test fails fast instead of waiting out
 *   the three retries the real app uses against a live backend.
 * - `stubApi` replaces the `api` object's methods per test. Mocking the module
 *   would work too, but these tests are about *what a page says*, and pointing
 *   `api.listTraces` at fixture data keeps the page's own code path — including
 *   its formatters — exactly as shipped.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { api } from "@/lib/api";
import type { ReactElement } from "react";

/** A client that never retries and never caches, so one test cannot see another's data. */
export function makeClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  });
}

export function renderPage(
  element: ReactElement,
  { route = "/", client = makeClient() }: { route?: string; client?: QueryClient } = {},
): RenderResult & { client: QueryClient } {
  const result = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>
        {element}
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return Object.assign(result, { client });
}

/** Stubs named `api` methods with `vi.fn()`s, restoring them after the test. */
export function stubApi(overrides: Partial<Record<keyof typeof api, unknown>>): void {
  for (const [name, implementation] of Object.entries(overrides)) {
    vi.spyOn(api, name as keyof typeof api).mockImplementation(
      implementation as never,
    );
  }
}
