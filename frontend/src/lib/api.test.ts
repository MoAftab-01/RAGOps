/**
 * §27: unit tests for the API layer.
 *
 * Four things are pinned here, in order of how much damage a regression would do:
 *
 * 1. **`buildUrl` omits unset filters.** A dropped `null` is a filter the backend
 *    never sees; a leaked `undefined` would serialise as the string
 *    "undefined" and produce a filter nobody meant.
 * 2. **The API key travels as a header and nowhere else.** These tests assert
 *    the negative: that it is absent from the query string, absent from the
 *    body, and absent when nothing was stored. A key in a URL ends up in server
 *    logs, browser history and shared links.
 * 3. **Non-2xx becomes an `ApiError` carrying the backend's own message.** A
 *    swallowed status turns a 401 into "something went wrong".
 * 4. **An unmeasured K is `null`, not zero.** `readRetrievalMetrics` returning
 *    `null` is what stops an unevaluated recall@10 being charted as 0%.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  API_BASE_URL,
  API_PREFIX,
  ApiError,
  DEFAULT_TIME_WINDOW,
  TIME_WINDOWS,
  api,
  buildUrl,
  clearApiKey,
  clearApiKeyForOrganization,
  getApiKey,
  getApiKeys,
  isTimeWindow,
  readRetrievalMetrics,
  request,
  retrievalKValues,
  runKValues,
  setApiKey,
  setApiKeyForOrganization,
  switchApiKey,
  type EvaluationRunSummary,
} from "@/lib/api";

/** A `fetch` stand-in that records its call and answers with what it is given. */
function stubFetch(
  body: unknown,
  init: { status?: number; statusText?: string } = {},
): { calls: Array<{ url: string; init: RequestInit }> } {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const fetchMock = vi.fn(async (url: string, requestInit: RequestInit) => {
    calls.push({ url: String(url), init: requestInit });
    const status = init.status ?? 200;
    const text = typeof body === "string" ? body : JSON.stringify(body);
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: init.statusText ?? "OK",
      text: async () => text,
    } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls };
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("API_BASE_URL", () => {
  it("never ends in a slash, so paths join with exactly one separator", () => {
    expect(API_BASE_URL.endsWith("/")).toBe(false);
  });
});

describe("buildUrl", () => {
  it("prefixes a bare path with /api", () => {
    expect(buildUrl("/health")).toBe(`${API_BASE_URL}${API_PREFIX}/health`);
  });

  it("does not double-prefix a path that already carries the prefix", () => {
    expect(buildUrl(`${API_PREFIX}/health`)).toBe(`${API_BASE_URL}${API_PREFIX}/health`);
  });

  it("omits null and undefined rather than sending them as strings", () => {
    const url = buildUrl("/traces", {
      window: "7d",
      application: null,
      start: undefined,
    });
    expect(url).toBe(`${API_BASE_URL}${API_PREFIX}/traces?window=7d`);
    expect(url).not.toContain("application");
    expect(url).not.toContain("undefined");
    expect(url).not.toContain("null");
  });

  it("omits an empty string — application= is a name that does not exist", () => {
    const url = buildUrl("/traces", { application: "" });
    expect(url).toBe(`${API_BASE_URL}${API_PREFIX}/traces`);
  });

  it("sends zero and false, which are measurements rather than absences", () => {
    const url = buildUrl("/experiments", { page: 0, persist: false });
    expect(url).toContain("page=0");
    expect(url).toContain("persist=false");
  });

  it("returns no query string when every parameter was dropped", () => {
    expect(buildUrl("/traces", { application: null })).not.toContain("?");
  });

  it("percent-encodes an application name with a space in it", () => {
    const url = buildUrl("/traces", { application: "demo support bot" });
    expect(url).toContain("application=demo+support+bot");
  });
});

describe("isTimeWindow", () => {
  it("accepts exactly the windows the backend accepts", () => {
    for (const window of TIME_WINDOWS) expect(isTimeWindow(window)).toBe(true);
  });

  it("rejects anything else, so a typo falls back to the default instead of 422-ing", () => {
    expect(isTimeWindow("99d")).toBe(false);
    expect(isTimeWindow("")).toBe(false);
    expect(isTimeWindow("7D")).toBe(false);
  });

  it("defaults to seven days", () => {
    expect(DEFAULT_TIME_WINDOW).toBe("7d");
    expect(isTimeWindow(DEFAULT_TIME_WINDOW)).toBe(true);
  });
});

describe("API key storage", () => {
  it("round-trips a trimmed value", () => {
    setApiKey("  sk-local  ");
    expect(getApiKey()).toBe("sk-local");
  });

  it("reports no key before one is set", () => {
    expect(getApiKey()).toBe(null);
  });

  it("treats a blank value as clearing the key", () => {
    setApiKey("sk-local");
    setApiKey("   ");
    expect(getApiKey()).toBe(null);
  });

  it("clears on request", () => {
    setApiKey("sk-local");
    clearApiKey();
    expect(getApiKey()).toBe(null);
  });
});

describe("request: the X-API-Key header", () => {
  it("sends the stored key as a header", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ ok: true });
    await request("/health");
    const headers = calls[0].init.headers as Record<string, string>;
    expect(headers["X-API-Key"]).toBe("sk-local");
  });

  it("never puts the key in the URL", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ ok: true });
    await request("/traces", { params: { window: "7d" } });
    // The assertion that matters: a key in a query string is a key in server
    // logs, browser history and every shared link built from this view.
    expect(calls[0].url).not.toContain("sk-local");
  });

  it("never puts the key in the request body", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ ok: true });
    await request("/experiments", { method: "POST", body: { name: "exp" } });
    expect(String(calls[0].init.body)).not.toContain("sk-local");
  });

  it("omits the header entirely when no key is stored", async () => {
    const { calls } = stubFetch({ ok: true });
    await request("/health");
    expect(calls[0].init.headers as Record<string, string>).not.toHaveProperty("X-API-Key");
  });

  it("omits the header when a request opts out of authentication", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ ok: true });
    await request("/health", { authenticated: false });
    expect(calls[0].init.headers as Record<string, string>).not.toHaveProperty("X-API-Key");
  });

  it("still sends the header on a write", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ ok: true });
    await request("/anomalies/abc", { method: "PATCH", body: { is_resolved: true } });
    expect((calls[0].init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-local");
    expect(calls[0].init.method).toBe("PATCH");
  });
});

describe("request: errors", () => {
  it("raises ApiError with the backend's own detail", async () => {
    stubFetch({ detail: "Not authenticated", code: "missing_api_key" }, { status: 401 });
    const error = await request("/traces", { method: "POST" }).catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(401);
    expect(apiError.message).toBe("Not authenticated");
    expect(apiError.code).toBe("missing_api_key");
    expect(apiError.isUnauthorized).toBe(true);
    expect(apiError.isNetworkError).toBe(false);
  });

  it("treats 403 as unauthorized too", async () => {
    stubFetch({ detail: "Forbidden" }, { status: 403 });
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error.isUnauthorized).toBe(true);
  });

  it("flattens FastAPI's 422 list into one readable message", async () => {
    stubFetch(
      {
        detail: [
          { loc: ["body", "window"], msg: "value is not a valid enumeration member" },
          { loc: ["body", "k"], msg: "must be greater than 0" },
        ],
      },
      { status: 422 },
    );
    const error = (await request("/evaluations/retrieval", { method: "POST" }).catch(
      (cause: unknown) => cause,
    )) as ApiError;
    expect(error.message).toContain("window: value is not a valid enumeration member");
    expect(error.message).toContain("k: must be greater than 0");
  });

  it("falls back to a named HTTP message when the error body is not JSON", async () => {
    // A proxy or gunicorn answers 500 with HTML or plain text. That text is not
    // JSON, so it is discarded in favour of a message that names the status and
    // the URL — which is the part that identifies which hop failed.
    stubFetch("<html><body>502 Bad Gateway</body></html>", {
      status: 500,
      statusText: "Internal Server Error",
    });
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error.message).toContain("500");
    expect(error.message).toContain("Internal Server Error");
    expect(error.message).toContain(API_BASE_URL);
    // Not dumped into the message: an HTML error page would be noise.
    expect(error.message).not.toContain("<html>");
  });

  it("uses a quoted JSON string detail verbatim", async () => {
    stubFetch('"Not authenticated"', { status: 401 });
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error.message).toBe("Not authenticated");
  });

  it("falls back to a named HTTP message when the body carries nothing", async () => {
    stubFetch({}, { status: 503, statusText: "Service Unavailable" });
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error.status).toBe(503);
    expect(error.message).toContain("503");
    expect(error.message).toContain("Service Unavailable");
  });

  it("reports an unreachable backend as status 0, not an unhandled TypeError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.isNetworkError).toBe(true);
    // Naming the base URL is what lets the reader tell "backend is down" from
    // "this console is pointed at the wrong host".
    expect(error.message).toContain(API_BASE_URL);
  });

  it("rejects a 200 that is not JSON instead of rendering a blank page", async () => {
    // A proxy or a crashed dev server answers 200 with HTML.
    stubFetch("<!doctype html><html></html>");
    const error = (await request("/traces").catch((cause: unknown) => cause)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toContain("not JSON");
  });

  it("returns the parsed body on success", async () => {
    stubFetch({ items: [], total: 0, page: 1, page_size: 25, has_next: false });
    const result = await request<{ total: number }>("/traces");
    expect(result.total).toBe(0);
  });

  it("accepts an empty 200 body", async () => {
    stubFetch("");
    await expect(request("/health")).resolves.toBe(null);
  });
});

describe("readRetrievalMetrics", () => {
  // Shaped as `as_metric_rows` in the evaluation service produces it: the
  // headline `k` is a key, not a top-level field.
  const metrics = {
    "k=5": { precision_at_k: 0.8, recall_at_k: 0.6, mrr: 0.7, num_queries: 10 },
    "k=10": { precision_at_k: 0.5, recall_at_k: 0.9, mrr: 0.65, num_queries: 10 },
  };

  it("reads the metrics for one K", () => {
    expect(readRetrievalMetrics(metrics, 5)?.recall_at_k).toBe(0.6);
    expect(readRetrievalMetrics(metrics, 10)?.recall_at_k).toBe(0.9);
  });

  it("returns null for a K that was never measured", () => {
    // Not zero. The distinction between "recall@20 is 0" and "recall@20 was
    // never computed" is the whole reason this function can return null.
    expect(readRetrievalMetrics(metrics, 20)).toBe(null);
  });

  it("returns null for a missing or empty metrics blob", () => {
    expect(readRetrievalMetrics(null, 5)).toBe(null);
    expect(readRetrievalMetrics(undefined, 5)).toBe(null);
    expect(readRetrievalMetrics({}, 5)).toBe(null);
  });

  it("returns null when the key exists but is not an object", () => {
    expect(readRetrievalMetrics({ "k=5": null }, 5)).toBe(null);
    expect(readRetrievalMetrics({ "k=5": "nope" }, 5)).toBe(null);
  });

  it("does not read a top-level field as though it were per-K", () => {
    // A retrieval run has nothing at the top level; guessing here would let an
    // answer run's metrics masquerade as retrieval metrics.
    expect(readRetrievalMetrics({ recall_at_k: 0.9 }, 5)).toBe(null);
  });
});

describe("retrievalKValues", () => {
  it("lists the Ks ascending, whatever order they arrived in", () => {
    expect(retrievalKValues({ "k=10": {}, "k=3": {}, "k=5": {} })).toEqual([3, 5, 10]);
  });

  it("ignores keys that are not K buckets", () => {
    expect(retrievalKValues({ "k=5": {}, num_queries: 10, summary: {} })).toEqual([5]);
  });

  it("returns an empty list rather than guessing", () => {
    expect(retrievalKValues(null)).toEqual([]);
    expect(retrievalKValues({})).toEqual([]);
  });
});

describe("runKValues", () => {
  const base: EvaluationRunSummary = {
    id: "run-1",
    k: 5,
    status: "completed",
    metrics: null,
    created_at: "2026-09-30T12:00:00Z",
  } as unknown as EvaluationRunSummary;

  it("prefers the K buckets the run actually recorded", () => {
    expect(runKValues({ ...base, metrics: { "k=3": {}, "k=5": {} } })).toEqual([3, 5]);
  });

  it("falls back to the run's own k when there are no buckets", () => {
    expect(runKValues(base)).toEqual([5]);
  });

  it("returns empty when neither source names a positive K", () => {
    // An unmeasured run plots nothing. Defaulting to a K here would chart a
    // number nobody measured.
    expect(runKValues({ ...base, k: 0 })).toEqual([]);
  });
});

/* -------------------------------------------------------------------------- */
/* Multi-tenancy: per-organization keys                                       */
/* -------------------------------------------------------------------------- */

describe("per-organization key storage", () => {
  it("starts empty", () => {
    expect(getApiKeys()).toEqual({});
  });

  it("round-trips a trimmed value per organization", () => {
    setApiKeyForOrganization("org-a", "  rag_a  ");
    setApiKeyForOrganization("org-b", "rag_b");
    expect(getApiKeys()).toEqual({ "org-a": "rag_a", "org-b": "rag_b" });
  });

  it("keeps two companies' keys apart — the whole point of the map", () => {
    setApiKeyForOrganization("org-a", "rag_a");
    setApiKeyForOrganization("org-b", "rag_b");
    clearApiKeyForOrganization("org-a");
    expect(getApiKeys()).toEqual({ "org-b": "rag_b" });
  });

  it("stores into the map without switching the active key", () => {
    // The load-bearing property. If pasting a second company's key silently
    // repointed the console, every following read would come from a different
    // tenant than the one on screen — and nothing in the UI would say so.
    setApiKey("original");
    setApiKeyForOrganization("org-b", "rag_b");
    expect(getApiKey()).toBe("original");
  });

  it("treats a blank value as forgetting that organization only", () => {
    setApiKeyForOrganization("org-a", "rag_a");
    setApiKeyForOrganization("org-b", "rag_b");
    setApiKeyForOrganization("org-b", "   ");
    expect(getApiKeys()).toEqual({ "org-a": "rag_a" });
  });

  it("degrades to empty rather than throwing on a corrupt value", () => {
    // Storage that was hand-edited, written by an older build, or truncated.
    // Losing the list is recoverable; a thrown error during render is not.
    window.localStorage.setItem("ragops.apiKeys", "{not json");
    expect(getApiKeys()).toEqual({});
  });

  it("ignores a stored value that is not an object", () => {
    for (const raw of ["[]", '"a string"', "42", "null"]) {
      window.localStorage.setItem("ragops.apiKeys", raw);
      expect(getApiKeys()).toEqual({});
    }
  });

  it("drops a non-string entry instead of putting it in an auth header", () => {
    window.localStorage.setItem(
      "ragops.apiKeys",
      JSON.stringify({ "org-a": "rag_a", "org-b": 42, "org-c": null }),
    );
    expect(getApiKeys()).toEqual({ "org-a": "rag_a" });
  });
});

describe("switchApiKey", () => {
  it("promotes a stored key to the active one", () => {
    setApiKeyForOrganization("org-a", "rag_a");
    expect(switchApiKey("org-a")).toBe(true);
    expect(getApiKey()).toBe("rag_a");
  });

  it("reports false and changes nothing for an organization with no key", () => {
    setApiKey("original");
    expect(switchApiKey("org-unknown")).toBe(false);
    // Leaving the active key alone matters: a bad switch must not log the
    // operator out of the company they were already working in.
    expect(getApiKey()).toBe("original");
  });
});

describe("tenancy endpoints", () => {
  const org = { id: "org-1", name: "Acme", num_api_keys: 0, num_applications: 0 };

  it("puts organization ids through encodeURIComponent", async () => {
    // An id reaching the path raw is either a malformed URL or — worse — a path
    // segment that changes which organization is being addressed.
    const { calls } = stubFetch([]);
    await api.listApiKeys("org/../../admin");
    expect(calls[0].url).toContain("org%2F..%2F..%2Fadmin");
  });

  it("sends DELETE with no body", async () => {
    const { calls } = stubFetch({ message: "API key 'CI' revoked." });
    await api.revokeApiKey("org-1", "key-1");
    expect(calls[0].init.method).toBe("DELETE");
    // A body on a DELETE whose state comes entirely from the path is at best
    // noise and at worst a proxy that rejects the request outright.
    expect(calls[0].init.body).toBeUndefined();
  });

  it("never puts a key in the URL when minting one", async () => {
    setApiKey("sk-local");
    const { calls } = stubFetch({ key: "rag_minted" });
    await api.createApiKey("org-1", { name: "CI" });
    expect(calls[0].url).not.toContain("rag_minted");
    expect(calls[0].url).not.toContain("sk-local");
  });

  it("omits scopes rather than sending null when none were chosen", async () => {
    const { calls } = stubFetch({ key: "rag_minted" });
    await api.createApiKey("org-1", { name: "CI" });
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ name: "CI" });
  });

  it("posts a test trace as a write, which is the only thing that proves a key", async () => {
    setApiKey("sk-local");
    // The real body of a 201: `create_trace` returns `{trace_id, status}` and
    // no `id`. Asserting on the response here keeps the client's declared type
    // honest — this call's value is the proof, so the id it comes back with is
    // the evidence that the backend minted a trace rather than a 204.
    const { calls } = stubFetch({ trace_id: "onboarding-check", status: "created" });
    const created = await api.sendTestTrace("onboarding-check");
    expect(calls[0].init.method).toBe("POST");
    // The claim the Settings panel rests on: reads are open to anyone, so a
    // 200 here is the backend having accepted the credential.
    expect((calls[0].init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-local");
    expect(JSON.parse(String(calls[0].init.body)).application).toBe("onboarding-check");
    expect(created.trace_id).toBe("onboarding-check");
  });

  it("surfaces a tenant key's 403 rather than an empty list", async () => {
    // `listOrganizations` returning `[]` on failure would read as "you have no
    // companies", which is a lie the operator cannot detect.
    stubFetch({ detail: "Only a platform key can manage organizations." }, { status: 403 });
    const error = (await api.listOrganizations().catch((cause: unknown) => cause)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(403);
  });

  it("names the organization when creating one, and no slug when not given", async () => {
    const { calls } = stubFetch(org);
    await api.createOrganization({ name: "Acme" });
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ name: "Acme" });
  });
});
