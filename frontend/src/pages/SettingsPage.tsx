import { useState } from "react";
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import {
  AlertTriangle,
  Building2,
  CheckCircle2,
  Copy,
  Eye,
  EyeOff,
  KeyRound,
  Monitor,
  Moon,
  Plus,
  RefreshCw,
  Server,
  Sun,
} from "lucide-react";
import {
  API_BASE_URL,
  API_PREFIX,
  api,
  clearApiKey,
  clearApiKeyForOrganization,
  getApiKey,
  getApiKeys,
  setApiKey,
  setApiKeyForOrganization,
  switchApiKey,
  type ApiKey,
  type HealthResponse,
  type Organization,
} from "@/lib/api";
import { THEME_STORAGE_KEY, useTheme, type Theme } from "@/hooks/useTheme";
import {
  InfoNote,
  PageHeader,
  PanelTitle,
  SelectField,
  TextField,
} from "@/components/PageHeader";
import { QueryBoundary } from "@/components/ScopeBar";
import { DataTable, type Column } from "@/components/Table";
import { Badge, statusTone } from "@/components/Badge";
import { ErrorPanel, describeError } from "@/components/ErrorPanel";
import { cn } from "@/lib/cn";
import { formatDateTime, humanize } from "@/lib/format";

/**
 * §20: settings.
 *
 * Three things only, because those are the only three things this build lets a
 * user change at runtime: **the API key**, **the companies and credentials on
 * this deployment**, and **the theme**. Everything else is configured in `.env`
 * on the server; the browser has no business editing it, and a settings page
 * full of fields that silently do nothing is worse than a short one.
 *
 * On the key, specifically:
 *
 * - The value is **write-only from here on**. Reading it back would mean
 *   rendering a secret into the DOM, and the browser only has to hold it to
 *   send it. The field starts empty and reports *that a key is stored*, not
 *   what it is.
 * - It is kept in `localStorage` and sent only as the `X-API-Key` header —
 *   never a query string, never a log line, never anywhere but `API_BASE_URL`.
 * - Nothing on this page can write a key to the repository. The `.env` that
 *   holds the real one is git-ignored, and the SDK reads it from the
 *   environment.
 *
 * **The two rules the tenancy panels below must not break**, both of which the
 * panels are structured around rather than merely respecting:
 *
 * 1. **No stored key is ever rendered.** Not the legacy active key above, not a
 *    per-company key, not a masked fragment of one. The key list shows
 *    `key_prefix` — 12 characters that identify a row without being a
 *    credential — and nothing else. A newly minted key is the one exception,
 *    and it is an exception to a *server* rule, not a UI one: the backend has
 *    no endpoint that could return it again.
 * 2. **"Saved" is never dressed up as "verified."** `api.health()` is a GET and
 *    succeeds with any key or none, so a button that checks health proves
 *    nothing about the credential. The one honest check available from here is a
 *    *write*: `POST /traces` sits behind the write guard, so a 201 means the
 *    backend accepted this key. That is what the button below sends, and it is
 *    labelled for what it does.
 */

type CheckStatus = "ok" | "degraded" | "down" | "unknown";

export default function SettingsPage() {
  return (
    <div className="space-y-5">
      <PageHeader
        title="Settings"
        description="Runtime configuration for this browser. Server configuration lives in .env on the backend and is deliberately not editable from here."
      />

      <ApiKeyPanel />
      <CompaniesPanel />
      <ApiKeysPanel />
      <ThemePanel />
      <BackendPanel />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* API key                                                                    */
/* -------------------------------------------------------------------------- */

function ApiKeyPanel() {
  const queryClient = useQueryClient();

  // Read once on mount rather than subscribing: a re-render must not reveal the
  // stored value into component state where it could be logged or snapshotted.
  const [hasStoredKey, setHasStoredKey] = useState(() => getApiKey() !== null);
  const [draft, setDraft] = useState("");
  const [revealed, setRevealed] = useState(false);
  const [saved, setSaved] = useState(false);

  const save = (): void => {
    setApiKey(draft);
    setHasStoredKey(getApiKey() !== null);
    setDraft("");
    setRevealed(false);
    setSaved(true);
    // Every cached response was fetched under the previous identity, so the
    // whole cache is stale now — not just the one panel.
    void queryClient.invalidateQueries();
  };

  const forget = (): void => {
    clearApiKey();
    setHasStoredKey(false);
    setSaved(false);
    void queryClient.invalidateQueries();
  };

  return (
    <section className="card p-4" aria-label="API key">
      <PanelTitle
        actions={
          <Badge tone={hasStoredKey ? "good" : "neutral"}>
            {hasStoredKey ? "key stored" : "no key stored"}
          </Badge>
        }
      >
        <span className="flex items-center gap-1.5">
          <KeyRound className="h-3.5 w-3.5 text-ink-muted" aria-hidden="true" />
          API key
        </span>
      </PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Set <code className="text-2xs">RAGOPS_API_KEY</code> in the backend{" "}
        <code className="text-2xs">.env</code> to require one, then paste the same value here. The key
        is kept in this browser&apos;s <code className="text-2xs">localStorage</code> and sent only as
        the <code className="text-2xs">X-API-Key</code> header to{" "}
        <code className="text-2xs">{API_BASE_URL}</code>. It is never placed in a URL, so it cannot
        end up in server logs, browser history, or a shared link.
      </p>

      <div className="flex flex-wrap items-end gap-2">
        <TextField
          id="settings-api-key"
          className="min-w-64 flex-1"
          label="Key"
          type={revealed ? "text" : "password"}
          value={draft}
          onChange={(value) => {
            setDraft(value);
            setSaved(false);
          }}
          placeholder={hasStoredKey ? "Stored — paste a new value to replace it" : "Paste the key"}
          autoComplete="off"
        />
        <button
          type="button"
          className="btn-secondary"
          aria-pressed={revealed}
          onClick={() => setRevealed((open) => !open)}
        >
          {revealed ? (
            <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <Eye className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          {revealed ? "Hide" : "Show"}
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={draft.trim() === ""}
          onClick={save}
        >
          Save key
        </button>
        {hasStoredKey ? (
          <button type="button" className="btn-ghost" onClick={forget}>
            Forget key
          </button>
        ) : null}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {saved ? (
          <span className="flex items-center gap-1.5 text-2xs text-status-good">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            Saved to this browser. Existing panels have been refetched under it.
          </span>
        ) : null}
        <span className="text-2xs text-ink-muted">
          Saved, not verified. The analytics reads are open by design, so they succeed with or
          without a key — the first thing that proves this one works is a write, and a 401 names
          the problem.
        </span>
      </div>

      <InfoNote className="mt-3">
        The key is not recoverable from this page by design. If it is lost, generate a new one on the
        server and set it in{" "}
        <code className="text-2xs">.env</code> there. Nothing in this browser can read the
        backend&apos;s environment, and this page never echoes a stored key back into the DOM.
      </InfoNote>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Companies and credentials                                                  */
/* -------------------------------------------------------------------------- */

/**
 * The onboarding surface: create a company, mint its keys, revoke them.
 *
 * Split into `CompaniesPanel` (which company) and `ApiKeysPanel` (whose
 * credentials) because that is the order they are used in, and because they fail
 * independently: an operator may have the platform key and no company yet, or a
 * company and no reason to mint a second key.
 *
 * **Everything here is platform-only.** The backend returns 403 for a tenant key
 * on all of it, so the panel says so up front rather than letting someone
 * discover it by clicking through a list they can never see.
 *
 * `useScope` is deliberately *not* used. These are management rows, not
 * analytics, and putting "last 90 days" next to a list of companies would
 * invite the reader to think the list was filtered by it.
 */

const PLATFORM_ONLY_NOTE =
  "Managing companies and keys needs the shared platform key from RAGOPS_API_KEY. A company's own key is refused here by design — otherwise a customer could mint themselves a new company and a fresh credential for it.";

/** Both panels below re-fetch on this key, so one invalidation reaches both. */
const COMPANIES_QUERY_KEY = ["organizations"] as const;

function CompaniesPanel() {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");

  const companies = useQuery({
    queryKey: COMPANIES_QUERY_KEY,
    queryFn: () => api.listOrganizations(),
    retry: 0,
  });

  const create = useMutation({
    mutationFn: () => api.createOrganization({ name: name.trim() }),
    onSuccess: async () => {
      setName("");
      await queryClient.invalidateQueries({ queryKey: COMPANIES_QUERY_KEY });
    },
  });

  // Normalised rather than rendered from `unknown`: React renders an object by
  // throwing, so a rejected non-Error would blank the panel instead of failing.
  const failure = create.error === null ? null : describeError(create.error).detail;

  return (
    <section className="card p-4" aria-label="Companies">
      <PanelTitle
        actions={
          <Badge tone={companies.data === undefined ? "neutral" : "brand"}>
            {companies.data === undefined ? "loading" : `${companies.data.length} onboarded`}
          </Badge>
        }
      >
        <span className="flex items-center gap-1.5">
          <Building2 className="h-3.5 w-3.5 text-ink-muted" aria-hidden="true" />
          Companies
        </span>
      </PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Each company gets its own API key, and telemetry sent with that key lands only in its own
        data. Names are unique across the deployment, so one company&apos;s key sending an
        application name another already owns fails loudly rather than landing in the wrong place.
      </p>

      <div className="mb-3 flex flex-wrap items-end gap-2">
        <TextField
          id="company-name"
          className="min-w-64 flex-1"
          label="New company"
          value={name}
          onChange={(value) => {
            setName(value);
            create.reset();
          }}
          placeholder="e.g. Acme Support"
          autoComplete="off"
        />
        <button
          type="button"
          className="btn-primary"
          disabled={name.trim() === "" || create.isPending}
          onClick={() => create.mutate()}
        >
          <Plus className="h-3.5 w-3.5" aria-hidden="true" />
          {create.isPending ? "Creating…" : "Create company"}
        </button>
      </div>

      {failure === null ? null : (
        <ErrorPanel error={failure} subject="creating a company" className="mb-3" />
      )}

      <QueryBoundary query={companies} label="companies">
        {(rows: Organization[]) => (
          <DataTable
            caption="Onboarded companies"
            columns={COMPANY_COLUMNS}
            rows={rows}
            rowKey={(row) => row.id}
            minWidth="34rem"
            empty={
              <p className="text-2xs text-ink-muted">
                No companies yet. Create one above, then mint its first key.
              </p>
            }
          />
        )}
      </QueryBoundary>

      <InfoNote className="mt-3">{PLATFORM_ONLY_NOTE}</InfoNote>
    </section>
  );
}

const COMPANY_COLUMNS: Column<Organization>[] = [
  {
    key: "name",
    header: "Company",
    render: (row) => (
      <div className="min-w-0">
        <span className="block font-medium text-ink">{row.name}</span>
        {row.slug === null || row.slug === undefined ? null : (
          <span className="text-2xs text-ink-muted">{row.slug}</span>
        )}
      </div>
    ),
  },
  {
    key: "counts",
    header: "Data",
    align: "right",
    render: (row) => (
      <span className="text-2xs text-ink-secondary">
        {row.num_applications} app{row.num_applications === 1 ? "" : "s"} · {row.num_api_keys}{" "}
        key{row.num_api_keys === 1 ? "" : "s"}
      </span>
    ),
  },
  {
    key: "status",
    header: "Status",
    render: (row) => (
      <Badge tone={row.is_active ? "good" : "warning"}>
        {row.is_active ? "active" : "inactive"}
      </Badge>
    ),
  },
  {
    key: "created_at",
    header: "Created",
    render: (row) => (
      <span className="whitespace-nowrap text-2xs text-ink-muted">
        {formatDateTime(row.created_at)}
      </span>
    ),
  },
];

/**
 * Per-company credentials: mint one, store it in this browser, revoke it.
 *
 * **What this panel does and does not do with a credential**, which is the whole
 * design and the reason the pieces are split the way they are:
 *
 * - The **list** renders `key_prefix` and metadata. There is no field, no
 *   reveal toggle and no fetch that could put a stored key into the DOM — the
 *   backend has no route that would return one.
 * - A **newly minted** key is shown once, in a read-only field, because the
 *   operator has to be able to copy it into the SDK's environment. That value is
 *   held in component state for as long as the panel stays mounted and is
 *   replaced by the next mint; it is never written to `localStorage` under the
 *   legacy slot unless the operator explicitly switches to that company.
 * - **Storing** is a separate, explicit act. Pasting a key here records which
 *   company it belongs to; it does not repoint the console, because "which
 *   company's data am I looking at" should never be a side effect of pasting a
 *   credential somewhere.
 */
function ApiKeysPanel() {
  const queryClient = useQueryClient();
  const companies = useQuery({
    queryKey: COMPANIES_QUERY_KEY,
    queryFn: () => api.listOrganizations(),
    retry: 0,
  });

  // Keyed by nothing: one selected company, one list. Hoisted to the panel so
  // switching company cannot leave the previous company's keys on screen.
  const [selected, setSelected] = useState("");
  const effective = selected === "" ? (companies.data?.[0]?.id ?? "") : selected;

  // One query, owned here rather than in `KeyList`, so the count in the panel
  // title and the rows in the table can never come from two different fetches
  // that disagree.
  const keys = useQuery({
    queryKey: ["api-keys", effective],
    queryFn: () => api.listApiKeys(effective),
    enabled: effective !== "",
    retry: 0,
  });

  return (
    <section className="card p-4" aria-label="API keys">
      <PanelTitle
        actions={
          <Badge tone={keys.data === undefined ? "neutral" : "brand"}>
            {effective === ""
              ? "no company selected"
              : keys.data === undefined
                ? "loading"
                : `${keys.data.length} key${keys.data.length === 1 ? "" : "s"}`}
          </Badge>
        }
      >
        <span className="flex items-center gap-1.5">
          <KeyRound className="h-3.5 w-3.5 text-ink-muted" aria-hidden="true" />
          API keys
        </span>
      </PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        A key is shown in full exactly once, on the response that creates it. The list below can
        only show its first 12 characters, which is enough to tell two keys apart and never enough
        to authenticate with. If a key is lost, revoke it and mint another — nothing here can
        recover it.
      </p>

      <QueryBoundary query={companies} label="companies">
        {(rows: Organization[]) => (
          <>
            {rows.length === 0 ? (
              <p className="text-2xs text-ink-muted">
                No companies yet. Create one above; keys belong to a company.
              </p>
            ) : (
              <>
                <div className="mb-3">
                  <SelectField
                    id="api-keys-company"
                    className="min-w-64"
                    label="Company"
                    value={effective}
                    onChange={setSelected}
                    options={rows.map((row) => ({ value: row.id, label: row.name }))}
                  />
                </div>
                <ApiKeyActions organizationId={effective} />
                <KeyTable
                  keys={keys}
                  onRevoked={() =>
                    void queryClient.invalidateQueries({ queryKey: ["api-keys", effective] })
                  }
                />
              </>
            )}
          </>
        )}
      </QueryBoundary>

      <InfoNote className="mt-3">{PLATFORM_ONLY_NOTE}</InfoNote>
    </section>
  );
}

/**
 * Mint a key, and the one honest thing this browser can do with one: post a
 * trace, which is a write and therefore proves the credential.
 *
 * The button is labelled for what it does rather than for what a user hopes it
 * means. `api.health()` is the obvious thing to check and it is useless here —
 * it is a GET, and the read routes are open by design, so it answers 200 with
 * any key or none. `POST /traces` is behind the write guard, so a 201 is the
 * backend having accepted *this* key for *this* company.
 */
function ApiKeyActions({ organizationId }: { organizationId: string }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState("ingest,read");
  const [minted, setMinted] = useState<{ name: string; key: string } | null>(null);
  // Editable rather than derived from the organization id: an operator may
  // already have an application they want the first trace under, and this
  // project names applications globally so the name is theirs to choose.
  const [testApplication, setTestApplication] = useState("");

  const create = useMutation({
    mutationFn: () =>
      api.createApiKey(organizationId, {
        name: name.trim(),
        // Omitted rather than sent empty: the backend defaults a missing list to
        // both scopes, and an empty list is a 422 there ("a key that can do
        // nothing is a credential that looks valid and is not").
        ...(scopes === "" ? {} : { scopes: scopes.split(",").map((s) => s.trim()) }),
      }),
    onSuccess: async (created) => {
      // Stored under the company it belongs to, *not* made active. Two reasons,
      // and the second is the important one: `POST .../api-keys` is already
      // behind the write guard, so the backend answered 201 having accepted the
      // platform key — the new key has not been exercised at all. Making it
      // active here would claim a working identity for a credential nothing has
      // used yet, and the banner would then be testing itself. It becomes
      // active when the operator says so, which is what `switchApiKey` is for.
      setApiKeyForOrganization(organizationId, created.key);
      setMinted({ name: created.name, key: created.key });
      setName("");
      await queryClient.invalidateQueries({ queryKey: ["api-keys", organizationId] });
    },
  });

  const sendTrace = useMutation({
    mutationFn: () => api.sendTestTrace(testApplication.trim()),
  });

  return (
    <div className="mb-4 space-y-3">
      <div className="flex flex-wrap items-end gap-2">
        <TextField
          id="api-key-name"
          className="min-w-56 flex-1"
          label="New key"
          value={name}
          onChange={(value) => {
            setName(value);
            create.reset();
          }}
          placeholder="e.g. CI, staging, laptop"
          autoComplete="off"
        />
        <SelectField
          id="api-key-scopes"
          className="min-w-48"
          label="Scopes"
          value={scopes}
          onChange={setScopes}
          options={[
            { value: "ingest,read", label: "Ingest + read" },
            { value: "ingest", label: "Ingest only" },
            { value: "read", label: "Read only" },
          ]}
          hint="A read-only key cannot send telemetry; it gets a 403, not a silent drop."
        />
        <button
          type="button"
          className="btn-primary"
          disabled={name.trim() === "" || create.isPending}
          onClick={() => create.mutate()}
        >
          <Plus className="h-3.5 w-3.5" aria-hidden="true" />
          {create.isPending ? "Minting…" : "Mint key"}
        </button>
      </div>

      {create.error === null ? null : (
        <ErrorPanel error={create.error} subject="minting a key" />
      )}

      {minted === null ? null : <MintedKey minted={minted} onDismiss={() => setMinted(null)} />}

      <StoredKeyForCompany organizationId={organizationId} />

      <div className="flex flex-wrap items-end gap-2 border-t border-line/60 pt-3">
        <TextField
          id="onboarding-application"
          className="min-w-64 flex-1"
          label="Test application name"
          value={testApplication}
          onChange={setTestApplication}
          placeholder="my-app"
          autoComplete="off"
          hint="The application this test trace will be recorded under."
        />
        <button
          type="button"
          className="btn-secondary"
          disabled={sendTrace.isPending || testApplication.trim() === ""}
          onClick={() => sendTrace.mutate()}
        >
          {sendTrace.isPending ? "Sending…" : "Send a test trace"}
        </button>
      </div>

      {sendTrace.error === null ? null : (
        <ErrorPanel error={sendTrace.error} subject="sending a test trace" />
      )}
      {sendTrace.isSuccess ? (
        <p className="flex items-center gap-1.5 text-2xs text-status-good">
          <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
          The backend accepted this key and recorded the trace. That is a real check — the read
          routes are open, but this write is not.
        </p>
      ) : null}
    </div>
  );
}

/**
 * The one place a plaintext credential is ever on screen.
 *
 * Read-only so it cannot be edited into a different value and mistaken for the
 * real one, with a copy button because a key nobody can copy is not much of an
 * onboarding flow. The dismissal is explicit: React unmounting this block is the
 * only thing that removes the value from the DOM, so it does not happen on a
 * timer or on a refetch.
 */
function MintedKey({
  minted,
  onDismiss,
}: {
  minted: { name: string; key: string };
  onDismiss: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  // jsdom has no clipboard, and a non-secure origin has none either, so a
  // refused write is reported rather than left looking like a silent success.
  // The rejection handler covers both, since a missing `navigator.clipboard` is
  // itself a thrown TypeError rather than a null to test for.
  const copy = (): void => {
    void navigator.clipboard.writeText(minted.key).then(
      () => {
        setCopied(true);
        setCopyFailed(false);
      },
      () => {
        setCopied(false);
        setCopyFailed(true);
      },
    );
  };

  return (
    <div className="rounded-md border border-status-warning/40 bg-status-warning/5 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Badge tone="warning">{minted.name}</Badge>
        <span className="text-2xs font-medium text-ink">
          Copy this now — it is not shown again.
        </span>
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <TextField
          id="minted-api-key"
          className="min-w-72 flex-1 font-mono"
          label="New API key — shown once"
          ariaLabel="New API key — shown once"
          value={minted.key}
          onChange={() => undefined}
          readOnly
          autoComplete="off"
        />
        <button type="button" className="btn-secondary" onClick={copy}>
          <Copy className="h-3.5 w-3.5" aria-hidden="true" />
          {copied ? "Copied" : "Copy"}
        </button>
        <button type="button" className="btn-ghost" onClick={onDismiss}>
          Dismiss
        </button>
      </div>
      {copyFailed ? (
        <p role="alert" className="mt-2 text-2xs text-status-critical">
          This browser refused the clipboard write. Select the field above and copy it by hand — do
          not close this panel until you have.
        </p>
      ) : null}
      <p className="mt-2 text-2xs text-ink-secondary">
        Put it in the SDK&apos;s environment as <code className="text-2xs">RAGOPS_API_KEY</code>. This
        browser has kept a copy under this company, and the row below lets you make it the active
        one. If it is lost, revoke it above and mint another.
      </p>
    </div>
  );
}

/**
 * What this browser holds for one company, and the two acts that are on it.
 *
 * The panel stores a credential per company (`ragops.apiKeys`) but sends only
 * the *active* one (`ragops.apiKey`). Those are deliberately different slots:
 * writing a second company's key must not silently repoint the console at
 * another tenant's dashboard, so switching is something a person does and
 * storing is something the page does.
 *
 * Nothing here reads or renders a stored value. Which company a stored key
 * belongs to is not secret — the operator picked the company from a list a
 * moment ago — but the key itself is, and the button names only the company.
 * That is the same rule `ApiKeyPanel` follows with the legacy field: the badge
 * says a key exists, and no code path puts its characters on the screen.
 *
 * `switchApiKey` returning false means the id is not in the map, which the
 * badge above has already established cannot happen; it is still checked
 * because a bare `true`-typed function would otherwise be called for its
 * side effect and a future change making it conditional would leave nothing
 * to catch it.
 */
function StoredKeyForCompany({ organizationId }: { organizationId: string }) {
  const queryClient = useQueryClient();
  // The map lives in localStorage, which React does not subscribe to. Bumping
  // this is how the two buttons below get the row to redraw -- and it has to be
  // explicit, because the alternative is hoping an invalidated query happens to
  // return a new object and re-render this subtree for the right reason.
  const [, redraw] = useState(0);
  const stored = getApiKeys()[organizationId];

  if (stored === undefined) return null;

  const isActive = getApiKey() === stored;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge tone={isActive ? "good" : "neutral"}>
        {isActive ? "this browser is signed in as this company" : "key kept in this browser"}
      </Badge>
      {isActive ? null : (
        <button
          type="button"
          className="btn-secondary"
          onClick={() => {
            if (!switchApiKey(organizationId)) return;
            redraw((n) => n + 1);
            // Every cached response was fetched under the previous company.
            // Invalidating the lot is the only way a stale panel cannot survive
            // the switch and be read as the new company's numbers.
            void queryClient.invalidateQueries();
          }}
        >
          Use this company&apos;s key
        </button>
      )}
      <button
        type="button"
        className="btn-ghost"
        onClick={() => {
          clearApiKeyForOrganization(organizationId);
          // Forgetting the *active* copy has to clear the legacy slot too, or
          // the console keeps sending a key the operator just discarded.
          if (isActive) clearApiKey();
          redraw((n) => n + 1);
          void queryClient.invalidateQueries();
        }}
      >
        Forget this key
      </button>
    </div>
  );
}

/** The key list. Metadata only — a stored plaintext has no route and no column. */
function KeyTable({
  keys,
  onRevoked,
}: {
  keys: UseQueryResult<ApiKey[]>;
  onRevoked: () => void;
}) {
  return (
    <QueryBoundary query={keys} label="API keys">
      {(rows: ApiKey[]) => (
        <DataTable
          caption="API keys"
          columns={keyColumns(onRevoked)}
          rows={rows}
          rowKey={(row) => row.id}
          minWidth="40rem"
          empty={
            <p className="text-2xs text-ink-muted">
              No keys for this company yet. Mint one above — a company with no key cannot send
              telemetry.
            </p>
          }
        />
      )}
    </QueryBoundary>
  );
}

function keyColumns(onRevoked: () => void): Column<ApiKey>[] {
  return [
    {
      key: "name",
      header: "Name",
      render: (row) => <span className="font-medium text-ink">{row.name}</span>,
    },
    {
      key: "key_prefix",
      header: "Prefix",
      render: (row) => (
        <code className="text-2xs text-ink-secondary">
          {/* 12 characters of a 47-character key. Display material, never a credential. */}
          {row.key_prefix}…
        </code>
      ),
    },
    {
      key: "scopes",
      header: "Scopes",
      render: (row) => (
        <div className="flex flex-wrap gap-1">
          {row.scopes.map((scope) => (
            <Badge key={scope} tone="neutral">
              {scope}
            </Badge>
          ))}
        </div>
      ),
    },
    {
      key: "is_active",
      header: "Status",
      render: (row) =>
        row.is_active ? (
          <Badge tone="good">active</Badge>
        ) : row.revoked_at === null || row.revoked_at === undefined ? (
          <Badge tone="warning">expired</Badge>
        ) : (
          <Badge tone="serious">revoked</Badge>
        ),
    },
    {
      key: "last_used_at",
      header: "Last used",
      render: (row) => (
        <span className="whitespace-nowrap text-2xs text-ink-muted">
          {row.last_used_at === null || row.last_used_at === undefined
            ? "never"
            : formatDateTime(row.last_used_at)}
        </span>
      ),
    },
    {
      key: "revoked_at",
      header: "Created",
      render: (row) => (
        <span className="whitespace-nowrap text-2xs text-ink-muted">
          {formatDateTime(row.created_at)}
        </span>
      ),
    },
    {
      key: "actions",
      header: "Actions",
      render: (row) =>
        row.is_active ? (
          <RevokeButton organizationId={row.organization_id} apiKey={row} onRevoked={onRevoked} />
        ) : (
          <span className="text-2xs text-ink-muted">
            {row.revoked_at === null || row.revoked_at === undefined
              ? "no longer valid"
              : `revoked ${formatDateTime(row.revoked_at)}`}
          </span>
        ),
    },
  ];
}

/**
 * Revoke.
 *
 * Deletes nothing: the backend stamps `revoked_at` and keeps the row, because
 * `created_at` and `last_used_at` on a revoked key are the evidence that answers
 * "was it used after it leaked?". The row therefore stays in the table with a
 * revoked badge rather than vanishing, which is what makes the action legible
 * after the fact.
 */
function RevokeButton({
  organizationId,
  apiKey,
  onRevoked,
}: {
  organizationId: string;
  apiKey: ApiKey;
  onRevoked: () => void;
}) {
  const [pending, setPending] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const revoke = useMutation({
    mutationFn: () => api.revokeApiKey(organizationId, apiKey.id),
    onSuccess: () => {
      setPending(false);
      onRevoked();
    },
    onError: (error) => {
      setPending(false);
      // `ApiError.message` is the backend's own detail string; anything else
      // falls back to the type name rather than `String(error)`, which would
      // render "[object Object]" for a non-Error rejection.
      setFailure(describeError(error).detail);
    },
  });

  return (
    <div className="flex flex-col items-start gap-1">
      <button
        type="button"
        className="btn-ghost px-2 py-1 text-status-critical"
        disabled={pending}
        onClick={() => {
          setPending(true);
          setFailure(null);
          revoke.mutate();
        }}
      >
        <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
        {pending ? "Revoking…" : "Revoke"}
      </button>
      {failure === null ? null : (
        <span role="alert" className="text-2xs text-status-critical">
          {failure}
        </span>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Theme                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * `useTheme` reports a resolved `Theme` plus a `followsSystem` flag; "follow the
 * OS" is a third *choice*, not a third theme, so it is modelled here rather than
 * widened into `Theme` where it would not describe what the DOM carries.
 */
type ThemeChoice = Theme | "system";

const THEME_OPTIONS: Array<{ value: ThemeChoice; label: string; icon: typeof Sun }> = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

function ThemePanel() {
  const { theme, followsSystem, setTheme, followSystemTheme } = useTheme();
  const current: ThemeChoice = followsSystem ? "system" : theme;

  return (
    <section className="card p-4" aria-label="Appearance">
      <PanelTitle>Appearance</PanelTitle>

      <p className="mb-3 text-2xs leading-relaxed text-ink-secondary">
        Stored in this browser under{" "}
        <code className="text-2xs">{THEME_STORAGE_KEY}</code>. The choice is applied before the first
        paint by a small inline script, so the page never flashes the wrong theme; this control
        mirrors that same decision.
      </p>

      <div role="radiogroup" aria-label="Theme" className="flex flex-wrap gap-2">
        {THEME_OPTIONS.map((option) => {
          const Icon = option.icon;
          const selected = current === option.value;
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              className={cn("btn-secondary", selected && "btn-primary")}
              onClick={() =>
                option.value === "system" ? followSystemTheme() : setTheme(option.value)
              }
            >
              <Icon className="h-3.5 w-3.5" aria-hidden="true" />
              {option.label}
            </button>
          );
        })}
      </div>

      <p className="mt-3 text-2xs text-ink-muted">
        {followsSystem
          ? "Following the operating system preference. Choosing an explicit theme above detaches it."
          : "Explicitly set in this browser; the operating system preference is ignored."}
      </p>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Backend                                                                    */
/* -------------------------------------------------------------------------- */

const CHECK_ROWS: Column<{ name: string; value: unknown }>[] = [
  { key: "name", header: "Check", render: (row) => <span className="font-medium text-ink">{row.name}</span> },
  {
    key: "status",
    header: "Status",
    render: (row) => {
      const status = readStatus(row.value);
      return <Badge tone={statusTone(status === "unknown" ? "unknown" : status)}>{status}</Badge>;
    },
  },
  {
    key: "detail",
    header: "Detail",
    render: (row) => (
      <ul className="space-y-0.5 text-2xs text-ink-secondary">
        {Object.entries(
          typeof row.value === "object" && row.value !== null
            ? (row.value as Record<string, unknown>)
            : { value: row.value },
        )
          .filter(([key]) => key !== "status")
          .map(([key, value]) => (
            <li key={key}>
              <span className="text-ink-muted">{humanize(key)}:</span>{" "}
              {typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}
            </li>
          ))}
      </ul>
    ),
  },
];

function readStatus(value: unknown): CheckStatus {
  if (typeof value === "string") {
    return value === "ok" || value === "degraded" || value === "down" ? value : "unknown";
  }
  if (typeof value === "object" && value !== null) {
    const status = (value as { status?: unknown }).status;
    if (status === "ok" || status === "degraded" || status === "down") return status;
  }
  return "unknown";
}

function BackendPanel() {
  const health = useQuery({
    queryKey: ["health", "settings"],
    queryFn: () => api.health(),
    retry: 0,
  });

  const checks = health.data === undefined
    ? []
    : Object.entries(health.data.checks).map(([name, value]) => ({ name, value }));

  const anyDown = checks.some((check) => readStatus(check.value) === "down");
  const anyDegraded = checks.some((check) => readStatus(check.value) === "degraded");

  return (
    <section className="card p-4" aria-label="Backend">
      <PanelTitle
        actions={
          <button
            type="button"
            className="btn-secondary"
            disabled={health.isFetching}
            onClick={() => void health.refetch()}
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            {health.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        <span className="flex items-center gap-1.5">
          <Server className="h-3.5 w-3.5 text-ink-muted" aria-hidden="true" />
          Backend
        </span>
      </PanelTitle>

      <QueryBoundary query={health} label="the backend">
        {(data: HealthResponse) => (
          <>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Badge tone={data.status === "ok" ? "good" : "warning"}>{data.status}</Badge>
              <Badge tone="neutral">v{data.version}</Badge>
              <Badge tone="neutral">{data.environment}</Badge>
              <span className="text-2xs text-ink-muted">
                checked {formatDateTime(data.timestamp)}
              </span>
            </div>

            <p className="mb-3 text-2xs text-ink-muted">
              Endpoint <code className="text-2xs">{API_BASE_URL}{API_PREFIX}</code>
            </p>

            {anyDown || anyDegraded ? (
              <InfoNote tone="warning" className="mb-3">
                {anyDown
                  ? "A dependency is down. Pages that depend on it will show an error panel rather than an empty result — an empty result would read as a clean system."
                  : "A dependency is degraded, which usually means it is reachable but not fully initialised. The vector store reports this until the knowledge base has been indexed."}
              </InfoNote>
            ) : null}

            <DataTable
              caption="Backend dependency health"
              columns={CHECK_ROWS}
              rows={checks}
              rowKey={(row) => row.name}
              minWidth="32rem"
              empty={
                <p className="text-2xs text-ink-muted">
                  The backend reported no dependency checks. It is answering, which is the part that
                  matters for a local build.
                </p>
              }
            />
          </>
        )}
      </QueryBoundary>
    </section>
  );
}
