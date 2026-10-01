/**
 * §27: component tests for SettingsPage and ApplicationsPage.
 *
 * Settings is where the console could most easily leak or overclaim, so the
 * tests here are about two refusals:
 *
 * - **The stored key is never read back into the DOM.** The field starts empty
 *   and the page reports only *that* a key is stored. A test asserting the value
 *   is absent is the only thing that keeps that true through future edits.
 * - **The page does not claim to have verified the key.** A key that is not
 *   presented resolves to the platform principal (`security.py`: "a bad key must
 *   fail loudly rather than silently widen the request's visibility" — the
 *   inverse, and equally load-bearing), so every analytics *read* answers 200
 *   whether a key was sent or not. A "test" that hit one would prove nothing. The
 *   copy has to say so.
 *
 * Applications pins the other distinction the console keeps: list rows carry
 * **lifetime** figures, the detail panel carries **window** figures, and each is
 * labelled so neither is mistaken for the other.
 *
 * The third refusal is about *which company* the console is pointed at: holding a
 * credential for a company is not the same as being that company, and a mint must
 * not quietly repoint every panel at another tenant's data.
 */

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ApplicationsPage from "@/pages/ApplicationsPage";
import SettingsPage from "@/pages/SettingsPage";
import {
  ApiError,
  getApiKey,
  getApiKeys,
  setApiKeyForOrganization,
  switchApiKey,
} from "@/lib/api";
import { renderPage, stubApi } from "@/test/render";

const AT = "2026-09-30T12:00:00Z";

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

const health = {
  status: "degraded",
  version: "0.1.0",
  environment: "local",
  timestamp: AT,
  checks: {
    database: { status: "ok", latency_ms: 3 },
    vector_store: { status: "degraded", detail: "knowledge base not indexed" },
  },
};

describe("SettingsPage: the API key", () => {
  it("never renders a stored key back into the DOM", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Key"), {
      target: { value: "sk-super-secret-value" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Save key/i }));

    await waitFor(() => expect(getApiKey()).toBe("sk-super-secret-value"));

    // The header is sent on every request, so React unmounts the input on save
    // and the DOM keeps no copy of the secret — not in the field, not in a
    // toggle-to-reveal, not in a "current key" line.
    expect(document.body.textContent).not.toContain("sk-super-secret-value");
    expect(screen.queryByDisplayValue("sk-super-secret-value")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save key/i })).toBeInTheDocument();
    expect(document.body.textContent).toContain("key stored");
  });

  it("reports only that a key is stored, never a masked version of it", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Key"), { target: { value: "sk-abc-123456" } });
    fireEvent.click(screen.getByRole("button", { name: /Save key/i }));

    await waitFor(() => expect(getApiKey()).toBe("sk-abc-123456"));
    expect(document.body.textContent).not.toContain("sk-abc");
    expect(document.body.textContent).not.toContain("••••");
  });

  it("says plainly that saving is not verification", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    // The claim under test. `api.health()` is a GET, and reads are open, so a
    // "verify" button would pass with any key or none.
    expect(await screen.findByText(/Saved, not verified/i)).toBeInTheDocument();
    expect(document.body.textContent).toMatch(/reads are open by design/i);
    expect(screen.queryByRole("button", { name: /verify|test key/i })).not.toBeInTheDocument();
  });

  it("offers no way to read the stored key back", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Key"), { target: { value: "sk-abc-123456" } });
    fireEvent.click(screen.getByRole("button", { name: /Save key/i }));

    await waitFor(() => expect(getApiKey()).toBe("sk-abc-123456"));
    expect(screen.getByRole("button", { name: /Forget key/i })).toBeInTheDocument();
    expect(document.body.textContent).toMatch(/not recoverable from this page by design/i);
  });

  it("forgets the key on request", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Key"), { target: { value: "sk-abc-123456" } });
    fireEvent.click(screen.getByRole("button", { name: /Save key/i }));
    await waitFor(() => expect(getApiKey()).not.toBeNull());

    fireEvent.click(screen.getByRole("button", { name: /Forget key/i }));
    await waitFor(() => expect(getApiKey()).toBe(null));
    expect(screen.getByText("no key stored")).toBeInTheDocument();
  });

  it("holds a minted company's key without making it the active identity", async () => {
    // Assembled, never pasted: this is a test file, and test files get committed.
    const minted = `rag_${"qz7F".repeat(11).slice(0, 43)}`;
    const org = {
      id: "org-acme",
      name: "Acme Support",
      slug: "acme-support",
      is_active: true,
      num_applications: 0,
      num_api_keys: 1,
      created_at: AT,
      updated_at: AT,
    };
    stubApi({
      health: vi.fn().mockResolvedValue(health),
      listOrganizations: vi.fn().mockResolvedValue([org]),
      listApiKeys: vi
        .fn()
        .mockResolvedValue([
          {
            id: "key-1",
            organization_id: org.id,
            name: "laptop",
            key_prefix: minted.slice(0, 12),
            scopes: ["ingest", "read"],
            is_active: true,
            last_used_at: null,
            revoked_at: null,
            expires_at: null,
            created_at: AT,
            updated_at: AT,
          },
        ]),
      createApiKey: vi.fn().mockResolvedValue({
        id: "key-1",
        organization_id: org.id,
        name: "laptop",
        key_prefix: minted.slice(0, 12),
        scopes: ["ingest", "read"],
        is_active: true,
        last_used_at: null,
        revoked_at: null,
        expires_at: null,
        created_at: AT,
        updated_at: AT,
        key: minted,
      }),
    });

    // Signed in as the platform key before the mint.
    setApiKeyForOrganization("org-other", "sk-existing-active-key");
    switchApiKey("org-other");

    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New key"), { target: { value: "laptop" } });
    fireEvent.click(screen.getByRole("button", { name: /Mint key/i }));
    await screen.findByLabelText("New API key — shown once");

    // Kept, under the company it belongs to...
    await waitFor(() => expect(getApiKeys()[org.id]).toBe(minted));
    // ...but *not* promoted. The 201 above proves the platform key was accepted,
    // not this one; activating it now would put every panel on another tenant's
    // data while claiming to have validated it.
    expect(getApiKey()).toBe("sk-existing-active-key");

    // The row says which of the two states this is, so the operator is not left
    // guessing whether minting switched them.
    expect(await screen.findByText(/key kept in this browser/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Use this company.s key/i })).toBeInTheDocument();
  });

  it("keeps a company credential out of the DOM after the mint banner is dismissed", async () => {
    // Same production path as the case above, one step further. Dismissing the
    // banner takes the plaintext out of the page; storing it must not put it
    // back, because the stored row is rendered from localStorage on every
    // subsequent visit.
    const minted = `rag_${"qz7F".repeat(11).slice(0, 43)}`;
    setApiKeyForOrganization("org-acme", minted);
    stubApi({
      health: vi.fn().mockResolvedValue(health),
      listOrganizations: vi.fn().mockResolvedValue([
        {
          id: "org-acme",
          name: "Acme Support",
          slug: "acme-support",
          is_active: true,
          num_applications: 0,
          num_api_keys: 1,
          created_at: AT,
          updated_at: AT,
        },
      ]),
      // The list cannot carry a plaintext — that is the backend's whole
      // guarantee — so this row is the only place a stored key could resurface.
      listApiKeys: vi.fn().mockResolvedValue([]),
    });
    renderPage(<SettingsPage />);

    // The row rendered purely from the stored map, with no mint in this session.
    expect(await screen.findByText(/key kept in this browser/i)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain(minted);
    expect(screen.queryByDisplayValue(minted)).not.toBeInTheDocument();
    // Not even a prefix of it. The backend renders `key_prefix` in its list; the
    // browser has the whole secret and derives nothing from it, so a stored key
    // can't be narrowed down by what this page chooses to show.
    expect(document.body.textContent).not.toContain(minted.slice(0, 12));
  });

  it("warns when a dependency is degraded instead of reporting a clean system", async () => {
    stubApi({ health: vi.fn().mockResolvedValue(health) });
    renderPage(<SettingsPage />);

    expect(await screen.findByText(/vector_store/i)).toBeInTheDocument();
    expect(screen.getAllByText("degraded").length).toBeGreaterThan(0);
    expect(screen.getByText(/not fully initialised/i)).toBeInTheDocument();
  });

  it("names the backend as down when a check fails, rather than showing empty data", async () => {
    stubApi({
      health: vi.fn().mockResolvedValue({
        ...health,
        status: "unhealthy",
        checks: { database: { status: "down", error: "connection refused" } },
      }),
    });
    renderPage(<SettingsPage />);

    expect(await screen.findByText(/connection refused/i)).toBeInTheDocument();
    expect(screen.getByText(/A dependency is down/i)).toBeInTheDocument();
  });

  it("shows a real error panel when the backend is unreachable", async () => {
    stubApi({
      health: vi.fn().mockRejectedValue(
        new ApiError({
          status: 0,
          statusText: "Network request failed",
          detail: "Could not reach the RAGOps backend at http://localhost:8000: failed",
          code: null,
          url: "http://localhost:8000/api/health",
        }),
      ),
    });
    renderPage(<SettingsPage />);

    // An empty check table would read as "no dependencies configured", which is
    // a very different claim from "cannot reach the backend".
    expect(await screen.findByText(/Could not reach the RAGOps backend/i)).toBeInTheDocument();
  });
});

describe("ApplicationsPage", () => {
  const application = {
    id: "app-1",
    name: "demo-support-bot",
    description: "The §16 support bot.",
    environment: "local",
    is_active: true,
    created_at: AT,
    // Lifetime figures — these are on the list row.
    trace_count: 12480,
    total_tokens: 9_812_004,
    last_seen_at: AT,
  };

  function listPage(items = [application]) {
    return { items, total: items.length, page: 1, page_size: 25, has_next: false };
  }

  /**
   * The application name also appears as an `<option>` in the scope bar's
   * selector, so every lookup here is scoped to the catalogue `<ul>` — otherwise
   * `getByText` finds two nodes and the test fails for a reason that has
   * nothing to do with the claim being checked.
   */
  async function openFirst(): Promise<HTMLElement> {
    const list = await screen.findByRole("list");
    fireEvent.click(within(list).getByText("demo-support-bot"));
    return list;
  }

  it("labels list figures as lifetime, not as the selected window", async () => {
    stubApi({ listApplications: vi.fn().mockResolvedValue(listPage()) });
    renderPage(<ApplicationsPage />, { route: "/applications?window=24h" });

    const list = await screen.findByRole("list");
    const row = within(list).getByText("demo-support-bot").closest("article") as HTMLElement;
    expect(within(row).getByText(/Lifetime:/i)).toBeInTheDocument();
    expect(row.textContent).toContain("12,480");
  });

  it("says a quiet window is missing data rather than an unused application", async () => {
    stubApi({
      listApplications: vi.fn().mockResolvedValue(listPage()),
      applicationStats: vi.fn().mockResolvedValue({
        application_id: "app-1",
        application_name: "demo-support-bot",
        window: "24h",
        start: AT,
        end: AT,
        trace_count: 0,
        total_tokens: 0,
        estimated_cost: 0,
        cost_label: "local (Ollama)",
        avg_latency_ms: null,
        error_rate: 0,
        model_usage: [],
        time_series: [],
      }),
    });
    renderPage(<ApplicationsPage />, { route: "/applications?window=24h" });
    await openFirst();

    expect(await screen.findByText(/No traces in this window/i)).toBeInTheDocument();
    // The application is still listed, with its real lifetime figures.
    expect(screen.getByText(/Lifetime:/i)).toBeInTheDocument();
  });

  it("renders an unmeasured average latency as no value, not 0 ms", async () => {
    stubApi({
      listApplications: vi.fn().mockResolvedValue(listPage()),
      applicationStats: vi.fn().mockResolvedValue({
        application_id: "app-1",
        application_name: "demo-support-bot",
        window: "24h",
        start: AT,
        end: AT,
        trace_count: 4,
        total_tokens: 900,
        estimated_cost: 0,
        cost_label: "local (Ollama)",
        avg_latency_ms: null,
        error_rate: 0,
        model_usage: [],
        time_series: [],
      }),
    });
    renderPage(<ApplicationsPage />, { route: "/applications?window=24h" });
    await openFirst();

    await screen.findByText(/Traces in window/i);
    // `StatCard` is a div holding the label and the value, so the card is the
    // label's grandparent rather than its parent.
    const card = screen.getByText("Avg latency").closest(".card") as HTMLElement;
    expect(card.textContent).toContain("—");
    expect(card.textContent).not.toContain("0.0 ms");
    // StatCard marks a missing value with the placeholder class, which is what
    // keeps it visually distinct from a measured zero.
    expect(within(card).getByText("—")).toHaveClass("no-value");
  });

  it("invites a new application rather than showing an empty catalogue", async () => {
    stubApi({
      listApplications: vi.fn().mockResolvedValue(listPage([])),
    });
    renderPage(<ApplicationsPage />);

    expect(await screen.findByText(/No applications registered/i)).toBeInTheDocument();
    expect(screen.getByText(/Send one request, or run the demo data generator/i)).toBeInTheDocument();
  });
});
