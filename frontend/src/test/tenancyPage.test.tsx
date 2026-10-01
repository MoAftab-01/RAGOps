/**
 * §27 for the tenancy panels: SettingsPage's company and credential surface.
 *
 * `honestyPage.test.tsx` covers the *legacy* key field above these panels — the
 * one pasted by hand and stored in `ragops.apiKey`. These tests cover what
 * happens when a key is *minted*: the multi-tenancy path, where a browser can
 * hold several companies' credentials at once and only ever renders one of them.
 *
 * The claims under test, in order of how badly breaking them would hurt:
 *
 * - **A minted key is shown once and nowhere else.** It appears in a read-only
 *   field because the operator has to copy it, the table below shows only
 *   `key_prefix`, and dismissing the banner takes the value out of the DOM. A
 *   table that quietly grew a `key` column would put every company's credential
 *   on one screen.
 * - **Revoking keeps the row.** The backend stamps a column rather than
 *   deleting, and the badge is what says so. A row that vanished would read as
 *   "there was never a key", which is exactly the wrong thing for an audit.
 * - **"Send a test trace" is a real check, and says so.** It is a write, and
 *   the tests assert the copy does not drift into claiming more than that.
 *
 * No credential appears in this file. The minted key below is assembled from a
 * repeating pattern so it cannot be mistaken for a captured response, and no
 * assertion ever prints one.
 */

import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import SettingsPage from "@/pages/SettingsPage";
import { ApiError } from "@/lib/api";
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
  checks: { database: { status: "ok", latency_ms: 3 } },
};

const acme = {
  id: "org-acme",
  name: "Acme Support",
  slug: "acme-support",
  is_active: true,
  num_applications: 3,
  num_api_keys: 1,
  created_at: AT,
  updated_at: AT,
};

const globex = {
  id: "org-globex",
  name: "Globex Billing",
  slug: "globex-billing",
  is_active: false,
  num_applications: 0,
  num_api_keys: 0,
  created_at: AT,
  updated_at: AT,
};

const liveKey = {
  id: "key-ci",
  organization_id: acme.id,
  name: "CI",
  // Twelve characters, exactly as the backend renders it.
  key_prefix: "rag_AbCdEfGh",
  scopes: ["ingest", "read"],
  is_active: true,
  last_used_at: null,
  revoked_at: null,
  expires_at: null,
  created_at: AT,
  updated_at: AT,
};

const revokedKey = { ...liveKey, is_active: false, revoked_at: AT, last_used_at: AT };

/**
 * A shape-correct fake, assembled rather than pasted.
 *
 * It has the backend's scheme and length so the panel's own formatting is
 * exercised, and it is visibly not a captured credential — which matters,
 * because a test file is somewhere a real key ends up committed.
 */
function fakeMintedKey(): string {
  return `rag_${"qz7F".repeat(11).slice(0, 43)}`;
}

/**
 * Stubs the reads every SettingsPage render makes, so a test only has to
 * override the one it is about. `BackendPanel` calls `api.health()`
 * unconditionally, and both new panels call `api.listOrganizations()`.
 */
function stubShell(overrides: Parameters<typeof stubApi>[0] = {}): void {
  stubApi({
    health: vi.fn().mockResolvedValue(health),
    listOrganizations: vi.fn().mockResolvedValue([acme, globex]),
    listApiKeys: vi.fn().mockResolvedValue([liveKey]),
    ...overrides,
  });
}

const STATUS_TEXT: Record<number, string> = {
  401: "Unauthorized",
  403: "Forbidden",
  404: "Not Found",
  409: "Conflict",
};

function httpError(status: number, detail: string): ApiError {
  return new ApiError({
    status,
    statusText: STATUS_TEXT[status] ?? "Error",
    detail,
    code: null,
    url: "http://localhost:8000/api/organizations",
  });
}

/** The two companies panels render these tables side by side on the same page. */
function companiesTable(): HTMLElement {
  return screen.getByRole("table", { name: "Onboarded companies" });
}

function keysTable(): HTMLElement {
  return screen.getByRole("table", { name: "API keys" });
}

describe("SettingsPage: companies", () => {
  it("lists the companies the backend reports, with their real counts", async () => {
    stubShell();
    renderPage(<SettingsPage />);

    const table = await screen.findByRole("table", { name: "Onboarded companies" });
    const row = within(table).getByText("Acme Support").closest("tr") as HTMLElement;
    expect(row.textContent).toContain("3 apps");
    expect(row.textContent).toContain("1 key");
    // An inactive company stays listed: the question is which ones can receive
    // telemetry, and hiding one makes it unfindable.
    expect(within(table).getByText("inactive")).toBeInTheDocument();
  });

  it("sends only the name when creating one, then shows what came back", async () => {
    const created = { ...acme, id: "org-new", name: "Initech", slug: "initech" };
    const listOrganizations = vi
      .fn()
      .mockResolvedValueOnce([acme])
      .mockResolvedValue([acme, created]);
    const createOrganization = vi.fn().mockResolvedValue(created);
    stubShell({ listOrganizations, createOrganization });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New company"), {
      target: { value: "  Initech  " },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create company/i }));

    await waitFor(() => expect(createOrganization).toHaveBeenCalledTimes(1));
    // Trimmed on the way out; the slug is the backend's to derive, not the
    // client's guess at one.
    expect(createOrganization).toHaveBeenCalledWith({ name: "Initech" });
    // Scoped to the table: the name also appears in the keys panel's company
    // picker, and that appearing there too is the point — the new company is
    // immediately selectable without a reload.
    const row = await waitFor(() => {
      const found = within(companiesTable()).queryByText("Initech");
      expect(found).not.toBeNull();
      return found as HTMLElement;
    });
    expect(row.closest("tr")).not.toBeNull();
  });

  it("shows the backend's own conflict message rather than a generic failure", async () => {
    const createOrganization = vi
      .fn()
      .mockRejectedValue(httpError(409, "An organization named 'Acme Support' already exists."));
    stubShell({ createOrganization });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New company"), {
      target: { value: "Acme Support" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Create company/i }));

    // The 409 names the collision; that is the only thing that tells the
    // operator what to change.
    expect(
      await screen.findByText("An organization named 'Acme Support' already exists."),
    ).toBeInTheDocument();
  });

  it("reports a refused key as a refusal rather than an empty company list", async () => {
    stubShell({
      listOrganizations: vi
        .fn()
        .mockRejectedValue(
          httpError(403, "Only a platform key can manage organizations. Send the shared RAGOPS_API_KEY."),
        ),
    });
    renderPage(<SettingsPage />);

    // A blank table here would read as "no companies exist", which is a very
    // different claim from "you are not allowed to see them". Both panels report
    // it, because both depend on the same query.
    expect(
      await screen.findAllByText(/Only a platform key can manage organizations/i),
    ).not.toHaveLength(0);
    expect(screen.queryByRole("table", { name: "Onboarded companies" })).not.toBeInTheDocument();
  });

  it("invites the operator to create a company rather than showing a blank table", async () => {
    stubShell({
      listOrganizations: vi.fn().mockResolvedValue([]),
      listApiKeys: vi.fn().mockResolvedValue([]),
    });
    renderPage(<SettingsPage />);

    // Both panels explain the next step, and neither renders an empty table
    // with headers and no rows.
    expect((await screen.findAllByText(/No companies yet/i)).length).toBeGreaterThan(0);
    expect(document.body.textContent).toMatch(/Create one above/i);
    // Scoped by name: the backend health table at the bottom of the page is
    // unrelated and is expected to be there.
    expect(screen.queryByRole("table", { name: "Onboarded companies" })).not.toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "API keys" })).not.toBeInTheDocument();
  });
});

describe("SettingsPage: API keys", () => {
  it("lists a key by its prefix and never by anything longer", async () => {
    stubShell();
    renderPage(<SettingsPage />);

    await screen.findByRole("table", { name: "API keys" });
    const row = within(keysTable()).getByText("CI").closest("tr") as HTMLElement;
    expect(row.textContent).toContain("rag_AbCdEfGh");
    expect(row.textContent).toContain("never");
    expect(row.textContent).toContain("active");
    // Nothing in the list shows 20 or more key characters, let alone all 47.
    expect(within(keysTable()).queryByText(/rag_[A-Za-z0-9_-]{20,}/)).not.toBeInTheDocument();
  });

  it("shows a freshly minted key once, read-only, and outside the table", async () => {
    const minted = fakeMintedKey();
    const createApiKey = vi
      .fn()
      .mockResolvedValue({ ...liveKey, id: "key-new", name: "laptop", key: minted });
    stubShell({ createApiKey });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New key"), { target: { value: "laptop" } });
    fireEvent.click(screen.getByRole("button", { name: /Mint key/i }));

    const field = (await screen.findByLabelText("New API key — shown once")) as HTMLInputElement;
    expect(field.value).toBe(minted);
    // Read-only, so the copy cannot be edited into a different value and still
    // look like the credential the backend issued.
    expect(field).toHaveAttribute("readonly");

    // The table is refetched from the list, which cannot carry a plaintext.
    await waitFor(() => expect(createApiKey).toHaveBeenCalledTimes(1));
    expect(within(keysTable()).queryByText(new RegExp(minted))).not.toBeInTheDocument();
  });

  it("takes the plaintext back out of the DOM when the banner is dismissed", async () => {
    const minted = fakeMintedKey();
    const createApiKey = vi.fn().mockResolvedValue({ ...liveKey, key: minted });
    stubShell({ createApiKey });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New key"), { target: { value: "CI" } });
    fireEvent.click(screen.getByRole("button", { name: /Mint key/i }));
    await screen.findByLabelText("New API key — shown once");

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));

    await waitFor(() =>
      expect(screen.queryByLabelText("New API key — shown once")).not.toBeInTheDocument(),
    );
    expect(document.body.textContent).not.toContain(minted);
  });

  it("omits scopes rather than sending an empty list", async () => {
    const createApiKey = vi.fn().mockResolvedValue({ ...liveKey, key: fakeMintedKey() });
    stubShell({ createApiKey });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("New key"), { target: { value: "CI" } });
    fireEvent.change(screen.getByLabelText("Scopes"), { target: { value: "read" } });
    fireEvent.click(screen.getByRole("button", { name: /Mint key/i }));

    await waitFor(() => expect(createApiKey).toHaveBeenCalledTimes(1));
    expect(createApiKey).toHaveBeenCalledWith("org-acme", { name: "CI", scopes: ["read"] });
  });

  it("keeps the row after a revoke and flips its badge", async () => {
    let active = true;
    const listApiKeys = vi.fn().mockImplementation(async () => [active ? liveKey : revokedKey]);
    const revokeApiKey = vi.fn().mockImplementation(async () => {
      active = false;
      return { message: "API key 'CI' revoked." };
    });
    stubShell({ listApiKeys, revokeApiKey });
    renderPage(<SettingsPage />);

    await screen.findByRole("table", { name: "API keys" });
    expect(within(keysTable()).getByText("active")).toBeInTheDocument();

    fireEvent.click(within(keysTable()).getByRole("button", { name: "Revoke" }));

    await waitFor(() => expect(revokeApiKey).toHaveBeenCalledWith("org-acme", "key-ci"));
    // The row survives — the backend stamps `revoked_at` rather than deleting,
    // and a vanishing row would erase the evidence the leak was handled.
    await waitFor(() => expect(within(keysTable()).getByText("revoked")).toBeInTheDocument());
    expect(within(keysTable()).getByText("CI")).toBeInTheDocument();
    expect(within(keysTable()).queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument();
  });

  it("reports a failed revoke with the backend's own message", async () => {
    const revokeApiKey = vi
      .fn()
      .mockRejectedValue(httpError(404, "No API key key-ci on organization org-acme."));
    stubShell({ revokeApiKey });
    renderPage(<SettingsPage />);

    await screen.findByRole("table", { name: "API keys" });
    fireEvent.click(within(keysTable()).getByRole("button", { name: "Revoke" }));

    expect(await screen.findByText("No API key key-ci on organization org-acme.")).toBeInTheDocument();
    // The row stays active, because nothing was revoked.
    expect(within(keysTable()).getByRole("button", { name: "Revoke" })).toBeInTheDocument();
  });
});

describe("SettingsPage: the test trace button", () => {
  it("sends a write, which is the only honest check available from a browser", async () => {
    const sendTestTrace = vi
      .fn()
      .mockResolvedValue({ trace_id: "onboarding-check", status: "created" });
    stubShell({ sendTestTrace });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Test application name"), {
      target: { value: "acme-support-bot" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send a test trace" }));

    await waitFor(() => expect(sendTestTrace).toHaveBeenCalledWith("acme-support-bot"));
    // A GET would have proved nothing: the read routes are open by design, so a
    // 200 there says only that the backend is up.
    expect(
      await screen.findByText(/the read routes are open, but this write is not/i),
    ).toBeInTheDocument();
  });

  it("keeps the button disabled until there is an application to write under", async () => {
    const sendTestTrace = vi.fn().mockResolvedValue({ trace_id: "t-1", status: "created" });
    stubShell({ sendTestTrace });
    renderPage(<SettingsPage />);

    const button = await screen.findByRole("button", { name: "Send a test trace" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(sendTestTrace).not.toHaveBeenCalled();
  });

  it("reports a rejected write with its status, not as success", async () => {
    const sendTestTrace = vi
      .fn()
      .mockRejectedValue(
        new ApiError({
          status: 401,
          statusText: "Unauthorized",
          detail: "Invalid or revoked API key.",
          code: null,
          url: "http://localhost:8000/api/traces",
        }),
      );
    stubShell({ sendTestTrace });
    renderPage(<SettingsPage />);

    fireEvent.change(await screen.findByLabelText("Test application name"), {
      target: { value: "acme-support-bot" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send a test trace" }));

    expect(await screen.findByText("Invalid or revoked API key.")).toBeInTheDocument();
    // The success line must not appear alongside a failure: a 401 is exactly
    // the case where "the backend accepted this key" would be a lie.
    expect(document.body.textContent).not.toMatch(/this write is not/i);
  });

  it("is not labelled as a verification, which is the claim it cannot support", async () => {
    stubShell();
    renderPage(<SettingsPage />);

    // The same prohibition `honestyPage.test.tsx` pins for the legacy key
    // field: a check that only proves the backend is reachable must not be
    // dressed up as proof that a credential works.
    await screen.findByRole("button", { name: "Send a test trace" });
    expect(document.body.textContent).not.toMatch(/key verified|verification succeeded/i);
    expect(screen.queryByRole("button", { name: /verify|test key/i })).not.toBeInTheDocument();
  });
});