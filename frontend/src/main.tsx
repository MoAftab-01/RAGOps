import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "@/App";
import "@/index.css";

/**
 * Query client defaults.
 *
 * `refetchOnWindowFocus: false` is the non-obvious one. These panels are read
 * over a rolling time window, and a window refetch after every tab switch
 * re-requests the whole dashboard and makes every card flash its skeleton. The
 * data is seconds old either way; the refetch buys nothing and costs a round
 * trip per panel.
 *
 * One retry, and only for GETs. A POST that timed out may well have been
 * applied server-side, so replaying an evaluation run or an anomaly sweep
 * because the response was lost is how you end up with three copies of the
 * same job.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 30_000,
      retry: (failureCount, error) => {
        const status = (error as { status?: number } | null)?.status;
        if (status !== undefined && status >= 400 && status < 500) return false;
        return failureCount < 1;
      },
    },
    mutations: {
      retry: false,
    },
  },
});

const container = document.getElementById("root");
if (!container) {
  throw new Error("#root is missing from index.html");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
